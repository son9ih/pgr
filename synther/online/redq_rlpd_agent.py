import numpy as np
import torch
from redq.algos.core import (ReplayBuffer,
                             soft_update_model1_with_model2)
from redq.algos.redq_sac import REDQSACAgent
from synther.online.conditional_nets import Curiosity
from torch import Tensor
from tqdm import trange


def combine_two_tensors(tensor1, tensor2):
    return Tensor(np.concatenate([tensor1, tensor2], axis=0))


class REDQRLPDCondAgent(REDQSACAgent):

    def __init__(self, cond_hidden_size, diffusion_buffer_size=int(1e6), diffusion_sample_ratio=0.5, hyper=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=diffusion_buffer_size)
        self.diffusion_sample_ratio = diffusion_sample_ratio

        self.cond_net = Curiosity(input_size=self.obs_dim, 
                                    hidden_size=cond_hidden_size, 
                                    output_size=self.act_dim).to(self.device)
        self.cond_optimizer = torch.optim.Adam(self.cond_net.parameters(), lr=self.lr)
        self.hyper = hyper
    
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

    def train(self, logger):
        # Put conditional net in training mode
        self.cond_net.train()
        # this function is called after each datapoint collected.
        # when we only have very limited data, we don't make updates
        num_update = 0 if self.get_current_num_data() <= self.delay_update_steps else self.utd_ratio
        for i_update in range(num_update):
            sample_result = self.sample_data(self.batch_size)
            if len(sample_result) == 6:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor, is_diffusion_mask = sample_result
            else:
                # Fallback for when sample_data returns old format
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = sample_result
                is_diffusion_mask = torch.zeros(self.batch_size, dtype=torch.bool).to(self.device)
            
            """Q loss"""
            y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction_list = []
            curio_list = []
            self.cond_net.eval()  # Set conditional net to eval mode
            for q_i in range(self.num_Q):
                q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                curio = self.cond_net.compute_reward_torch(obs_tensor, obs_next_tensor, acts_tensor)
                q_prediction_list.append(q_prediction)
                curio_list.append(curio)
            self.cond_net.train()  # Set conditional net back to train mode
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            
            # Get diffusion buffer curio sum for weighted loss calculation
            diffusion_curio_sum = self.get_diffusion_buffer_curio_sum()
            
            q_loss_all = 0
            for q_i in range(self.num_Q):
                q_prediction = q_prediction_cat[:, q_i].unsqueeze(1)
                curio = curio_list[q_i].unsqueeze(1)
                
                # Compute base loss
                base_loss = self.mse_criterion(q_prediction, y_q)
                
                # Compute weights for each sample
                weights = torch.ones_like(curio)
                
                # For online data: use curio as weight (original behavior)
                online_indices = ~is_diffusion_mask
                weights[online_indices] = curio[online_indices]
                
                # For diffusion data: use weighted curio (diffusion_buffer_size * curio / total_curio_sum)
                diffusion_indices = is_diffusion_mask
                if diffusion_indices.any():
                    diffusion_weight = (self.diffusion_buffer.size * curio[diffusion_indices]) / diffusion_curio_sum
                    weights[diffusion_indices] = diffusion_weight
                
                # Apply weighted loss
                weighted_loss = base_loss * weights
                q_loss_all += weighted_loss.mean()
            
            
            # ==============================================
            # q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

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
                    cond_sample_result = self.sample_real_data(self.batch_size)
                    if len(cond_sample_result) == 6:
                        cond_obs_tensor, cond_obs_next_tensor, cond_acts_tensor, _, _, _ = cond_sample_result
                    else:
                        cond_obs_tensor, cond_obs_next_tensor, cond_acts_tensor, _, _ = cond_sample_result
                    self.cond_optimizer.zero_grad()
                    cond_loss = self.cond_net.forward_loss(cond_obs_tensor, cond_obs_next_tensor, cond_acts_tensor)
                    cond_loss.backward()
                    self.cond_optimizer.step()
                else:
                    cond_loss = Tensor([0])
            else:
                cond_loss = Tensor([0])

            # by default only log for the last update out of <num_update> updates
            # logging every information in wandb
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
            data = super().sample_data(batch_size)
            # Return data with indicator that all samples are from online buffer
            is_diffusion_mask = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            return data + (is_diffusion_mask,)
        
        diffusion_batch = self.diffusion_buffer.sample_batch(batch_size=diffusion_batch_size)
        online_batch = self.replay_buffer.sample_batch(batch_size=online_batch_size)
        obs_tensor = combine_two_tensors(online_batch['obs1'], diffusion_batch['obs1']).to(self.device)
        obs_next_tensor = combine_two_tensors(online_batch['obs2'], diffusion_batch['obs2']).to(self.device)
        acts_tensor = combine_two_tensors(online_batch['acts'], diffusion_batch['acts']).to(self.device)
        rews_tensor = combine_two_tensors(online_batch['rews'], diffusion_batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = combine_two_tensors(online_batch['done'], diffusion_batch['done']).unsqueeze(1).to(self.device)
        
        # Create mask to indicate which samples are from diffusion buffer
        is_diffusion_mask = torch.cat([
            torch.zeros(online_batch_size, dtype=torch.bool),
            torch.ones(diffusion_batch_size, dtype=torch.bool)
        ]).to(self.device)
        
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor, is_diffusion_mask

    def get_diffusion_buffer_curio_sum(self):
        """Compute the sum of all curios in the diffusion buffer"""
        if self.diffusion_buffer.size == 0:
            return 1.0  # Avoid division by zero
            
        # Sample all data from diffusion buffer
        all_data = self.diffusion_buffer.sample_batch(batch_size=self.diffusion_buffer.size)
        obs_tensor = torch.Tensor(all_data['obs1']).to(self.device)
        obs_next_tensor = torch.Tensor(all_data['obs2']).to(self.device)
        acts_tensor = torch.Tensor(all_data['acts']).to(self.device)
        
        with torch.no_grad():
            self.cond_net.eval()
            curios = self.cond_net.compute_reward_torch(obs_tensor, obs_next_tensor, acts_tensor)
            self.cond_net.train()
            curio_sum = curios.sum().item()
        
        return curio_sum if curio_sum > 0 else 1.0  # Avoid division by zero

    def sample_real_data(self, batch_size):
        data = super().sample_data(batch_size)
        # Add mask indicating all samples are from online buffer (not diffusion)
        is_diffusion_mask = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        return data + (is_diffusion_mask,)

    def reset_diffusion_buffer(self):
        self.diffusion_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                             size=self.diffusion_buffer.max_size)

