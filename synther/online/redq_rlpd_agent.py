import numpy as np
import torch
from redq.algos.core import (ReplayBuffer,
                             soft_update_model1_with_model2,
                             ACTION_BOUND_EPSILON)
from redq.algos.redq_sac import REDQSACAgent
from synther.online.conditional_nets import Curiosity, Predictor
from synther.online.utils import RunningMeanStd, RMS
from torch import Tensor
from torch.distributions import Normal
from tqdm import trange

import pdb


def combine_two_tensors(tensor1, tensor2):
    return Tensor(np.concatenate([tensor1, tensor2], axis=0))


class REDQRLPDCondAgent(REDQSACAgent):

    def __init__(self, cond_hidden_size, diffusion_buffer_size=int(1e6), diffusion_sample_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=diffusion_buffer_size)
        self.diffusion_sample_ratio = diffusion_sample_ratio

        self.cond_net = Curiosity(input_size=self.obs_dim, 
                                    hidden_size=cond_hidden_size, 
                                    output_size=self.act_dim).to(self.device)
        self.cond_optimizer = torch.optim.Adam(self.cond_net.parameters(), lr=self.lr)
        
        # rnd introduction
        # if self.rnd:
        self.pred_net = Predictor(input_size=self.obs_dim, normalize=False).to(self.device)
        # self.pred_net_target = Predictor(input_size=self.obs_dim, normalize=False).to(self.device)
        self.fix_net = Predictor(input_size=self.obs_dim, normalize=False).to(self.device)
        
        # target rnd every
        # if self.target_rnd_every > 0:
        self.temp_net = Predictor(input_size=self.obs_dim, normalize=False).to(self.device)
        
        self.pred_optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=1e-4)
        
        self.topk_threshold = None
        
        # For RND intrinsic reward normalization (PGRrnd)
        # Track discounted intrinsic return variance for normalization
        self.discounted_return_rms = RunningMeanStd()
        # Accumulate intrinsic rewards during episode
        self.episode_intrinsic_rewards = []
        # Flag to enable normalization (set to True for PGRrnd)
        self.normalize_intrinsic_reward = False
        
        # For next_obs normalization (used in pred_net training and reward computation)
        # Track next_obs statistics from original buffer (not diffusion buffer)
        self.next_obs_rms = RMS(device=self.device, epsilon=1e-4, shape=(self.obs_dim,))
        
        self.max_onpolicy_reward = 0.0
        self.total_onpolicy_reward = []
    
    def get_current_num_data(self):
        # used to determine whether we should get action from policy or take random starting actions
        return self.replay_buffer.size

    def get_exploration_action(self, obs, env):
        # given an observation, output a sampled action in numpy form
        with torch.no_grad():
            if self.get_current_num_data() > self.start_steps:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
                action_tensor = self.policy_net.forward(obs_tensor, deterministic=False,
                                             return_log_prob=False)[0]
                action = action_tensor.cpu().numpy().reshape(-1)
            else:
                action = env.action_space.sample()
        return action
    
    # get novelty score based on random network distillation
    def compute_intrinsic_reward(self, next_obs, accumulate=False):
        # Normalize next_obs before computing intrinsic reward
        # Use statistics from original buffer (not diffusion buffer)
        if self.next_obs_rms.n > 1.0:
            next_obs_normalized = self.next_obs_rms.normalize(next_obs)
        else:
            next_obs_normalized = next_obs
        # Clip normalized observations to [-5, 5] range
        next_obs_normalized = torch.clamp(next_obs_normalized, -5.0, 5.0)
        
        # if not square: 
        #     # 어짜피 이때는 weight 계산용, evaluation 용으로만 쓸꺼니까, temperature 조절 용도
        #     with torch.no_grad():
        #     # assert
        #         pred_next_feature = self.pred_net(next_obs_normalized)
        #         with torch.no_grad():
        #             fix_next_feature = self.fix_net(next_obs_normalized)
        #         fix_next_feature = fix_next_feature.detach()
        #         # square root of the original difference
        #         intrinsic_reward = torch.sqrt((fix_next_feature - pred_next_feature).pow(2).sum(1) / 2.0).pow(pow_reward)
        #         # intrinsic_reward = (fix_next_feature - pred_next_feature).pow(2).sum(1) / 2.0
        # else:
        pred_next_feature = self.pred_net(next_obs_normalized)
        with torch.no_grad():
            fix_next_feature = self.fix_net(next_obs_normalized)
        fix_next_feature = fix_next_feature.detach()
        # intrinsic_reward = ((fix_next_feature - pred_next_feature).pow(2).sum(1) / 2.0).pow(pow_reward)
        intrinsic_reward = ((fix_next_feature - pred_next_feature).pow(2).sum(1) / 2.0)
        
        # Accumulate original (unnormalized) intrinsic rewards during episode (for discounted return calculation)
        # IMPORTANT: We must accumulate the original reward BEFORE normalization
        if accumulate and self.normalize_intrinsic_reward:
            # Store original (unnormalized) reward as numpy for accumulation
            intrinsic_reward_original_np = intrinsic_reward.detach().cpu().numpy()
            if len(intrinsic_reward_original_np.shape) == 0:
                intrinsic_reward_original_np = np.array([intrinsic_reward_original_np])
            self.episode_intrinsic_rewards.extend(intrinsic_reward_original_np.tolist())
        
        # Normalize by discounted return std if enabled (PGRrnd)
        # This normalization is only for the returned reward, not for accumulation
        if self.normalize_intrinsic_reward and self.discounted_return_rms.count > 1.0:
            # Normalize by std (not mean-subtracted, just scale normalization)
            std = np.sqrt(self.discounted_return_rms.var + 1e-8)
            std_tensor = torch.tensor(std, device=intrinsic_reward.device, dtype=intrinsic_reward.dtype)
            intrinsic_reward = intrinsic_reward / std_tensor
                
        return intrinsic_reward
    
    
    def compute_onpolicy_reward(self, obs, act, low=-12.0, high=7.0):
        """
        Compute on-policy reward for a given (obs, act) pair.
        Formula: exp(Clip(log π_θ(a|s), low, high) - p_max)
        where p_max = max_{d in D} (Clip(log π_θ(a|s), low, high))
        """
        # Convert inputs to tensors if they are numpy arrays
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        if isinstance(act, np.ndarray):
            act = torch.FloatTensor(act).to(self.device)
        
        # Ensure correct shape
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        if len(act.shape) == 1:
            act = act.unsqueeze(0)
        
        # Compute log likelihood of action under current policy
        with torch.no_grad():
            # Get policy distribution parameters
            h = obs
            for fc_layer in self.policy_net.hidden_layers:
                h = self.policy_net.hidden_activation(fc_layer(h))
            mean = self.policy_net.last_fc_layer(h)
            log_std = self.policy_net.last_fc_log_std(h)
            log_std = torch.clamp(log_std, -20, 2)  # LOG_SIG_MIN, LOG_SIG_MAX
            std = torch.exp(log_std)
            
            # Normalize action by action_limit to get tanh-normalized action
            action_normalized = act / self.policy_net.action_limit
            
            # Invert tanh to get pre_tanh_value: atanh(x) = 0.5 * ln((1+x)/(1-x))
            # Clamp to avoid numerical issues near boundaries
            action_normalized_clamped = torch.clamp(action_normalized, -0.999999, 0.999999)
            pre_tanh_value = 0.5 * torch.log((1 + action_normalized_clamped) / (1 - action_normalized_clamped))
            
            # Compute log probability from normal distribution
            normal = Normal(mean, std)
            log_prob = normal.log_prob(pre_tanh_value)
            
            # Apply tanh correction: log_prob -= log(1 - tanh^2(x))
            log_prob -= torch.log(1 - action_normalized_clamped.pow(2) + ACTION_BOUND_EPSILON)
            
            # Sum over action dimensions
            log_prob = log_prob.sum(1, keepdim=True)
            
            # Clip log probability
            clipped_log_prob = torch.clamp(log_prob, low, high)
            
            # Compute on-policy reward: exp(clipped_log_prob - max_onpolicy_reward)
            onpolicy_reward = torch.exp(clipped_log_prob - self.max_onpolicy_reward)
            
            # Remove keepdim dimension if present (from sum(1, keepdim=True))
            if len(onpolicy_reward.shape) > 1 and onpolicy_reward.shape[-1] == 1:
                onpolicy_reward = onpolicy_reward.squeeze(-1)
            
            # Convert to numpy/scalar if needed
            if onpolicy_reward.numel() == 1:
                onpolicy_reward = onpolicy_reward.item()
            else:
                onpolicy_reward = onpolicy_reward.cpu().numpy()
        
        return onpolicy_reward
    
    def update_onpolicy_reward(self, low=-12.0, high=7.0):
        """
        Compute on-policy reward for all transitions in the original replay buffer.
        Store clipped log probabilities in self.total_onpolicy_reward.
        Return the maximum clipped log probability (p_max).
        """
        self.total_onpolicy_reward = []
        
        # Get all transitions from original replay buffer
        ptr_location = self.replay_buffer.ptr
        if ptr_location == 0:
            self.max_onpolicy_reward = 0.0
            return self.max_onpolicy_reward
        
        obs_all = self.replay_buffer.obs1_buf[:ptr_location]
        acts_all = self.replay_buffer.acts_buf[:ptr_location]
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs_all).to(self.device)
        acts_tensor = torch.FloatTensor(acts_all).to(self.device)
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        num_transitions = len(obs_all)
        
        with torch.no_grad():
            for i in range(0, num_transitions, batch_size):
                end_idx = min(i + batch_size, num_transitions)
                obs_batch = obs_tensor[i:end_idx]
                acts_batch = acts_tensor[i:end_idx]
                
                # Get policy distribution parameters
                h = obs_batch
                for fc_layer in self.policy_net.hidden_layers:
                    h = self.policy_net.hidden_activation(fc_layer(h))
                mean = self.policy_net.last_fc_layer(h)
                log_std = self.policy_net.last_fc_log_std(h)
                log_std = torch.clamp(log_std, -20, 2)  # LOG_SIG_MIN, LOG_SIG_MAX
                std = torch.exp(log_std)
                
                # Normalize action by action_limit to get tanh-normalized action
                action_normalized = acts_batch / self.policy_net.action_limit
                
                # Invert tanh to get pre_tanh_value
                action_normalized_clamped = torch.clamp(action_normalized, -0.999999, 0.999999)
                pre_tanh_value = 0.5 * torch.log((1 + action_normalized_clamped) / (1 - action_normalized_clamped))
                
                # Compute log probability from normal distribution
                normal = Normal(mean, std)
                log_prob = normal.log_prob(pre_tanh_value)
                
                # Apply tanh correction
                log_prob -= torch.log(1 - action_normalized_clamped.pow(2) + ACTION_BOUND_EPSILON)
                
                # Sum over action dimensions
                log_prob = log_prob.sum(1, keepdim=False)
                
                # Clip log probability
                clipped_log_prob = torch.clamp(log_prob, low, high)
                
                # Store in list
                self.total_onpolicy_reward.extend(clipped_log_prob.cpu().numpy().tolist())
        
        # Compute maximum
        if len(self.total_onpolicy_reward) > 0:
            self.max_onpolicy_reward = max(self.total_onpolicy_reward)
        else:
            self.max_onpolicy_reward = 0.0

    def train(self, logger):
        # Put conditional net in training mode
        self.cond_net.train()
        # this function is called after each datapoint collected.
        # when we only have very limited data, we don't make updates
        num_update = 0 if self.get_current_num_data() <= self.delay_update_steps else self.utd_ratio
        for i_update in range(num_update):
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)
            
            """Q loss"""
            y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction_list = []
            for q_i in range(self.num_Q):
                q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                q_prediction_list.append(q_prediction)
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()

            """policy and alpha loss"""
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                # get policy loss
                a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = self.policy_net.forward(obs_tensor)
                q_a_tilda_list = []
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(False)
                    q_a_tilda = self.q_net_list[sample_idx](torch.cat([obs_tensor, a_tilda], 1))
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(True)

                # get alpha loss
                if self.auto_alpha:
                    alpha_loss = -(self.log_alpha * (log_prob_a_tilda + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.cpu().exp().item()
                else:
                    alpha_loss = Tensor([0])

            """update networks"""
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                self.policy_optimizer.step()

            # polyak averaged Q target networks
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)


            """Update Cond Net"""
            # Only on last iteration for now
            # And only with real data
            if i_update == num_update - 1:
                if self.cond_optimizer is not None:
                    cond_obs_tensor, cond_obs_next_tensor, cond_acts_tensor, _, _ = self.sample_real_data(self.batch_size)
                    self.cond_optimizer.zero_grad()
                    cond_loss = self.cond_net.forward_loss(cond_obs_tensor, cond_obs_next_tensor, cond_acts_tensor)
                    cond_loss.backward()
                    self.cond_optimizer.step()
                else:
                    cond_loss = Tensor([0])
            else:
                cond_loss = Tensor([0])

            # by default only log for the last update out of <num_update> updates
            if i_update == num_update - 1:
                logger.store(LossCond=cond_loss.cpu().item(), LossPi=policy_loss.cpu().item(), LossQ1=q_loss_all.cpu().item() / self.num_Q,
                             LossAlpha=alpha_loss.cpu().item(), Q1Vals=q_prediction.detach().cpu().numpy(),
                             Alpha=self.alpha, LogPi=log_prob_a_tilda.detach().cpu().numpy(),
                             PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1))

        # if there is no update, log 0 to prevent logging problems
        if num_update == 0:
            logger.store(LossCond=0, LossPi=0, LossQ1=0, LossAlpha=0, Q1Vals=0, Alpha=0, LogPi=0, PreTanh=0)

    def sample_data(self, batch_size):
        diffusion_batch_size = int(batch_size * self.diffusion_sample_ratio)
        online_batch_size = int(batch_size - diffusion_batch_size)
        # Sample from the diffusion buffer
        if self.diffusion_buffer.size < diffusion_batch_size:
            return super().sample_data(batch_size)
        diffusion_batch = self.diffusion_buffer.sample_batch(batch_size=diffusion_batch_size)
        online_batch = self.replay_buffer.sample_batch(batch_size=online_batch_size)
        obs_tensor = combine_two_tensors(online_batch['obs1'], diffusion_batch['obs1']).to(self.device)
        obs_next_tensor = combine_two_tensors(online_batch['obs2'], diffusion_batch['obs2']).to(self.device)
        acts_tensor = combine_two_tensors(online_batch['acts'], diffusion_batch['acts']).to(self.device)
        rews_tensor = combine_two_tensors(online_batch['rews'], diffusion_batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = combine_two_tensors(online_batch['done'], diffusion_batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def sample_real_data(self, batch_size):
        return super().sample_data(batch_size)
    
    def sample_real_data_recent(self, batch_size):
        return super().sample_data_recent(batch_size)
    
    def sample_real_data_cpu(self, batch_size):
        return super().sample_data_cpu(batch_size)
    
    def sample_diffusion_data_cpu(self, batch_size):
        return super().sample_diffusion_data_cpu(batch_size)
        

    def reset_diffusion_buffer(self):
        self.diffusion_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                             size=self.diffusion_buffer.max_size)
        
    def train_pred_net(self, batch_size, mask=True, update_proportion=0.25):
        if self.pred_optimizer is not None:
            
            _, obs_next_tensor, _, _, _ = self.sample_real_data_recent(batch_size=batch_size)
            self.pred_optimizer.zero_grad()
            # Don't accumulate during training
            pred_loss = self.compute_intrinsic_reward(obs_next_tensor, accumulate=False) # .mean()
            if mask:
                mask_tensor = torch.rand(len(pred_loss)).to(self.device)
                mask_tensor = (mask_tensor < update_proportion).type(torch.FloatTensor).to(self.device)
                pred_loss = (pred_loss * mask_tensor).mean() / torch.max(mask_tensor.sum(), torch.Tensor([1.0]).to(self.device))
            else:
                pred_loss = pred_loss.mean()
            # pdb.set_trace()
            pred_loss.backward()
            self.pred_optimizer.step()
        else:
            pred_loss = Tensor([0])
        return pred_loss.detach().cpu().numpy()
    
    def update_discounted_return_stats(self, gamma=0.99):
        """
        Update discounted intrinsic return statistics at end of episode.
        This computes the discounted return from accumulated intrinsic rewards and updates RMS.
        """
        if not self.normalize_intrinsic_reward or len(self.episode_intrinsic_rewards) == 0:
            self.episode_intrinsic_rewards = []
            return
        
        # Compute discounted return from accumulated intrinsic rewards
        intrinsic_rewards = np.array(self.episode_intrinsic_rewards)
        discounted_return = 0.0
        for t in range(len(intrinsic_rewards)):
            discounted_return += (gamma ** t) * intrinsic_rewards[t]
        
        # Update RMS with discounted return (single value per episode)
        self.discounted_return_rms.update(discounted_return)
        
        # Reset episode accumulation
        self.episode_intrinsic_rewards = []
    
    def set_normalize_intrinsic_reward(self, enable):
        """Enable/disable intrinsic reward normalization (for PGRrnd)"""
        self.normalize_intrinsic_reward = enable
    
    def update_next_obs_stats(self):
        """
        Update next_obs statistics from original buffer (not diffusion buffer).
        This should be called at the end of each epoch.
        """
        ptr_location = self.replay_buffer.ptr
        if ptr_location == 0:
            return
        
        # Get all next_obs from original buffer (not diffusion buffer)
        next_obs_all = self.replay_buffer.obs2_buf[:ptr_location]
        
        # Convert to torch tensor and update RMS
        next_obs_tensor = torch.FloatTensor(next_obs_all).to(self.device)
        self.next_obs_rms(next_obs_tensor)
              

