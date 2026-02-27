"""
TD3 agent with generative replay (diffusion buffer) augmentation.

Based on rlkit/torch/td3 and adapted to match the REDQRLPDCondAgent interface
for use with the PGR diffusion-based data augmentation pipeline.
Uses deterministic policy, twin Q-networks, policy noise, and delayed policy updates.
"""
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from redq.algos.core import (
    ReplayBuffer,
    soft_update_model1_with_model2,
    Mlp,
)
from synther.online.conditional_nets import Curiosity, Predictor
from synther.online.utils import RunningMeanStd, RMS
from synther.online.eco import ECO
from tqdm import trange


def combine_two_tensors(tensor1, tensor2):
    return Tensor(np.concatenate([tensor1, tensor2], axis=0))


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class DeterministicPolicy(nn.Module):
    """
    Deterministic policy with tanh output for TD3.
    Maps obs -> action in [-act_limit, act_limit].
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, action_limit=1.0):
        super().__init__()
        self.action_limit = action_limit
        self.hidden_layers = nn.ModuleList()
        in_size = obs_dim
        for next_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_size, next_size))
            in_size = next_size
        self.last_fc_layer = nn.Linear(in_size, act_dim)
        self.apply(weights_init_)

    def forward(self, obs):
        h = obs
        for fc_layer in self.hidden_layers:
            h = torch.relu(fc_layer(h))
        return torch.tanh(self.last_fc_layer(h)) * self.action_limit


class TD3RLPDCondAgent:
    """
    TD3 agent with diffusion buffer augmentation.
    Compatible with the PGR pipeline (prior/posterior diffusion, QFlow, etc.).
    """

    def __init__(
        self,
        cond_hidden_size,
        diffusion_buffer_size=int(1e6),
        diffusion_sample_ratio=0.5,
        env_name=None,
        obs_dim=None,
        act_dim=None,
        act_limit=1.0,
        device=None,
        hidden_sizes=(256, 256),
        replay_size=int(1e6),
        batch_size=256,
        lr=3e-4,
        gamma=0.99,
        polyak=0.995,
        start_steps=5000,
        delay_update_steps='auto',
        utd_ratio=20,
        policy_update_delay=2,
        target_policy_noise=0.2,
        target_policy_noise_clip=0.5,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.device = device
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr
        self.start_steps = start_steps
        self.delay_update_steps = start_steps if delay_update_steps == 'auto' else delay_update_steps
        self.utd_ratio = utd_ratio
        self.policy_update_delay = policy_update_delay
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip

        # Policy (deterministic)
        self.policy_net = DeterministicPolicy(
            obs_dim, act_dim, hidden_sizes, action_limit=act_limit
        ).to(device)
        self.target_policy_net = DeterministicPolicy(
            obs_dim, act_dim, hidden_sizes, action_limit=act_limit
        ).to(device)
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())

        # Twin Q-networks
        self.qf1 = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(device)
        self.qf2 = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(device)
        self.target_qf1 = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(device)
        self.target_qf2 = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(device)
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=lr)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=lr)

        # Buffers
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.diffusion_buffer = ReplayBuffer(
            obs_dim=obs_dim, act_dim=act_dim, size=diffusion_buffer_size
        )
        self.diffusion_sample_ratio = diffusion_sample_ratio

        # Conditional / novelty nets (for PGR pipeline)
        self.cond_net = Curiosity(
            input_size=obs_dim,
            hidden_size=cond_hidden_size,
            output_size=act_dim,
        ).to(device)
        self.cond_optimizer = torch.optim.Adam(self.cond_net.parameters(), lr=lr)

        self.pred_net = Predictor(input_size=obs_dim, normalize=False).to(device)
        self.fix_net = Predictor(input_size=obs_dim, normalize=False).to(device)
        self.temp_net = Predictor(input_size=obs_dim, normalize=False).to(device)
        self.pred_optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=1e-4)

        self.topk_threshold = None
        self.discounted_return_rms = RunningMeanStd()
        self.episode_intrinsic_rewards = []
        self.normalize_intrinsic_reward = False
        self.next_obs_rms = RMS(device=device, epsilon=1e-4, shape=(obs_dim,))
        self.max_onpolicy_reward = 0.0
        self.total_onpolicy_reward = []
        self.eco = None

        self.mse_criterion = nn.MSELoss()
        self._n_train_steps_total = 0

    def get_current_num_data(self):
        return self.replay_buffer.size

    def get_exploration_action(self, obs, env):
        with torch.no_grad():
            if self.get_current_num_data() > self.start_steps:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
                action_tensor = self.policy_net(obs_tensor)
                # Add exploration noise (Gaussian, clipped)
                noise = torch.randn_like(action_tensor) * 0.1
                noise = torch.clamp(noise, -0.5, 0.5)
                action_tensor = action_tensor + noise
                action_tensor = torch.clamp(
                    action_tensor, -self.act_limit, self.act_limit
                )
                action = action_tensor.cpu().numpy().reshape(-1)
            else:
                action = env.action_space.sample()
        return action

    def get_test_action(self, obs):
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy_net(obs_tensor)
            action = action_tensor.cpu().numpy().reshape(-1)
        return action

    def get_action_and_logprob_for_bias_evaluation(self, obs):
        # TD3 is deterministic; return action and dummy log_prob for API compatibility
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy_net(obs_tensor)
            action = action_tensor.cpu().numpy().reshape(-1)
        log_prob = torch.tensor(0.0, device=self.device)
        return action, log_prob

    def get_ave_q_prediction_for_bias_evaluation(self, obs_tensor, acts_tensor):
        obs_tensor = obs_tensor.to(self.device)
        acts_tensor = acts_tensor.to(self.device)
        q1 = self.qf1(torch.cat([obs_tensor, acts_tensor], 1))
        q2 = self.qf2(torch.cat([obs_tensor, acts_tensor], 1))
        return (q1 + q2) / 2

    def store_data(self, o, a, r, o2, d):
        self.replay_buffer.store(o, a, r, o2, d)

    def sample_data(self, batch_size):
        diffusion_batch_size = int(batch_size * self.diffusion_sample_ratio)
        online_batch_size = int(batch_size - diffusion_batch_size)
        if self.diffusion_buffer.size < diffusion_batch_size:
            batch = self.replay_buffer.sample_batch(batch_size)
            obs_tensor = Tensor(batch['obs1']).to(self.device)
            obs_next_tensor = Tensor(batch['obs2']).to(self.device)
            acts_tensor = Tensor(batch['acts']).to(self.device)
            rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
            done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
            return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor
        diffusion_batch = self.diffusion_buffer.sample_batch(batch_size=diffusion_batch_size)
        online_batch = self.replay_buffer.sample_batch(batch_size=online_batch_size)
        obs_tensor = combine_two_tensors(online_batch['obs1'], diffusion_batch['obs1']).to(self.device)
        obs_next_tensor = combine_two_tensors(online_batch['obs2'], diffusion_batch['obs2']).to(self.device)
        acts_tensor = combine_two_tensors(online_batch['acts'], diffusion_batch['acts']).to(self.device)
        rews_tensor = combine_two_tensors(online_batch['rews'], diffusion_batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = combine_two_tensors(online_batch['done'], diffusion_batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def sample_real_data(self, batch_size):
        batch = self.replay_buffer.sample_batch(batch_size)
        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def sample_real_data_recent(self, batch_size):
        ptr = self.replay_buffer.ptr
        if ptr >= batch_size:
            idxs = np.arange(ptr - batch_size, ptr)
        else:
            idxs1 = np.arange(0, ptr)
            idxs2 = np.arange(
                self.replay_buffer.max_size - (batch_size - ptr),
                self.replay_buffer.max_size,
            )
            idxs = np.concatenate([idxs2, idxs1], axis=0)
        batch = self.replay_buffer.sample_batch(batch_size, idxs=idxs)
        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def sample_real_data_cpu(self, batch_size):
        batch = self.replay_buffer.sample_batch(batch_size)
        obs_tensor = Tensor(batch['obs1'])
        obs_next_tensor = Tensor(batch['obs2'])
        acts_tensor = Tensor(batch['acts'])
        rews_tensor = Tensor(batch['rews']).unsqueeze(1)
        done_tensor = Tensor(batch['done']).unsqueeze(1)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def sample_diffusion_data_cpu(self, batch_size):
        batch = self.diffusion_buffer.sample_batch(batch_size)
        obs_tensor = Tensor(batch['obs1'])
        obs_next_tensor = Tensor(batch['obs2'])
        acts_tensor = Tensor(batch['acts'])
        rews_tensor = Tensor(batch['rews']).unsqueeze(1)
        done_tensor = Tensor(batch['done']).unsqueeze(1)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def reset_diffusion_buffer(self):
        self.diffusion_buffer = ReplayBuffer(
            obs_dim=self.obs_dim, act_dim=self.act_dim,
            size=self.diffusion_buffer.max_size,
        )

    def compute_intrinsic_reward(self, next_obs, accumulate=False):
        if self.next_obs_rms.n > 1.0:
            next_obs_normalized = self.next_obs_rms.normalize(next_obs)
        else:
            next_obs_normalized = next_obs
        next_obs_normalized = torch.clamp(next_obs_normalized, -5.0, 5.0)
        pred_next_feature = self.pred_net(next_obs_normalized)
        with torch.no_grad():
            fix_next_feature = self.fix_net(next_obs_normalized)
        intrinsic_reward = ((fix_next_feature - pred_next_feature).pow(2).sum(1) / 2.0)
        if accumulate and self.normalize_intrinsic_reward:
            intrinsic_reward_original_np = intrinsic_reward.detach().cpu().numpy()
            if len(intrinsic_reward_original_np.shape) == 0:
                intrinsic_reward_original_np = np.array([intrinsic_reward_original_np])
            self.episode_intrinsic_rewards.extend(intrinsic_reward_original_np.tolist())
        if self.normalize_intrinsic_reward and self.discounted_return_rms.count > 1.0:
            std = np.sqrt(self.discounted_return_rms.var + 1e-8)
            std_tensor = torch.tensor(std, device=intrinsic_reward.device, dtype=intrinsic_reward.dtype)
            intrinsic_reward = intrinsic_reward / std_tensor
        return intrinsic_reward

    def compute_eco_reward(self, current_obs, done=None):
        if self.eco is None:
            self.eco = ECO(
                obs_dim=self.obs_dim,
                embedding_dim=512,
                hidden_dim=512,
                memory_capacity=200,
                replacement='random',
                alpha=0.03,
                beta=1.0,
                similarity_threshold=0.5,
                similarity_aggregation='percentile',
                device=self.device,
            )
        return self.eco.compute_reward(current_obs, done=None)

    def compute_onpolicy_reward(self, obs, act, low=-12.0, high=7.0):
        """
        For TD3 deterministic policy: use Gaussian approximation with fixed std
        centered at policy(obs) to get a proxy for "how on-policy" the action is.
        """
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        if isinstance(act, np.ndarray):
            act = torch.FloatTensor(act).to(self.device)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        if len(act.shape) == 1:
            act = act.unsqueeze(0)
        with torch.no_grad():
            policy_act = self.policy_net(obs)
            # Gaussian log prob: -0.5 * ||a - mu||^2 / sigma^2 - const
            # Use fixed std=0.3 as proxy for "policy spread"
            std = 0.3
            diff = (act - policy_act) / (std + 1e-8)
            log_prob = -0.5 * (diff ** 2).sum(1, keepdim=True) - act.shape[1] * np.log(std * np.sqrt(2 * np.pi))
            clipped_log_prob = torch.clamp(log_prob.squeeze(-1), low, high)
            onpolicy_reward = torch.exp(clipped_log_prob - self.max_onpolicy_reward)
            if onpolicy_reward.numel() == 1:
                return onpolicy_reward.item()
            return onpolicy_reward.cpu().numpy()

    def update_onpolicy_reward(self, low=-12.0, high=7.0):
        self.total_onpolicy_reward = []
        ptr = self.replay_buffer.ptr
        if ptr == 0:
            self.max_onpolicy_reward = 0.0
            return self.max_onpolicy_reward
        obs_all = self.replay_buffer.obs1_buf[:ptr]
        acts_all = self.replay_buffer.acts_buf[:ptr]
        obs_tensor = torch.FloatTensor(obs_all).to(self.device)
        acts_tensor = torch.FloatTensor(acts_all).to(self.device)
        batch_size = 1000
        with torch.no_grad():
            for i in range(0, len(obs_all), batch_size):
                end_idx = min(i + batch_size, len(obs_all))
                obs_batch = obs_tensor[i:end_idx]
                acts_batch = acts_tensor[i:end_idx]
                policy_act = self.policy_net(obs_batch)
                std = 0.3
                diff = (acts_batch - policy_act) / (std + 1e-8)
                log_prob = -0.5 * (diff ** 2).sum(1) - self.act_dim * np.log(std * np.sqrt(2 * np.pi))
                clipped_log_prob = torch.clamp(log_prob, low, high)
                self.total_onpolicy_reward.extend(clipped_log_prob.cpu().numpy().tolist())
        if len(self.total_onpolicy_reward) > 0:
            self.max_onpolicy_reward = max(self.total_onpolicy_reward)
        else:
            self.max_onpolicy_reward = 0.0
        return self.max_onpolicy_reward

    def train(self, logger):
        self.cond_net.train()
        num_update = (
            0 if self.get_current_num_data() <= self.delay_update_steps else self.utd_ratio
        )
        for i_update in range(num_update):
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(
                self.batch_size
            )

            # TD3: target with policy noise
            with torch.no_grad():
                next_actions = self.target_policy_net(obs_next_tensor)
                noise = torch.randn_like(next_actions) * self.target_policy_noise
                noise = torch.clamp(
                    noise,
                    -self.target_policy_noise_clip,
                    self.target_policy_noise_clip,
                )
                noisy_next_actions = next_actions + noise
                noisy_next_actions = torch.clamp(
                    noisy_next_actions, -self.act_limit, self.act_limit
                )
                target_q1 = self.target_qf1(
                    torch.cat([obs_next_tensor, noisy_next_actions], 1)
                )
                target_q2 = self.target_qf2(
                    torch.cat([obs_next_tensor, noisy_next_actions], 1)
                )
                target_q = torch.min(target_q1, target_q2)
                y_q = rews_tensor + (1.0 - done_tensor) * self.gamma * target_q

            # Q loss
            q1_pred = self.qf1(torch.cat([obs_tensor, acts_tensor], 1))
            q2_pred = self.qf2(torch.cat([obs_tensor, acts_tensor], 1))
            qf1_loss = self.mse_criterion(q1_pred, y_q)
            qf2_loss = self.mse_criterion(q2_pred, y_q)

            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer.step()

            policy_loss = None
            if (self._n_train_steps_total % self.policy_update_delay == 0):
                policy_actions = self.policy_net(obs_tensor)
                q1_pi = self.qf1(torch.cat([obs_tensor, policy_actions], 1))
                policy_loss = -q1_pi.mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                soft_update_model1_with_model2(
                    self.target_policy_net, self.policy_net, self.polyak
                )
                soft_update_model1_with_model2(
                    self.target_qf1, self.qf1, self.polyak
                )
                soft_update_model1_with_model2(
                    self.target_qf2, self.qf2, self.polyak
                )

            self._n_train_steps_total += 1

            # Update cond net (curiosity)
            if i_update == num_update - 1 and self.cond_optimizer is not None:
                cond_obs, cond_obs_next, cond_acts, _, _ = self.sample_real_data(
                    self.batch_size
                )
                self.cond_optimizer.zero_grad()
                cond_loss = self.cond_net.forward_loss(
                    cond_obs, cond_obs_next, cond_acts
                )
                cond_loss.backward()
                self.cond_optimizer.step()
            else:
                cond_loss = Tensor([0])

            if i_update == num_update - 1:
                if policy_loss is None:
                    policy_actions = self.policy_net(obs_tensor)
                    policy_loss = -self.qf1(torch.cat([obs_tensor, policy_actions], 1)).mean()
                # policy_actions is set either above or in the policy update block
                logger.store(
                    LossCond=cond_loss.cpu().item() if isinstance(cond_loss, Tensor) else 0,
                    LossPi=policy_loss.cpu().item(),
                    LossQ1=(qf1_loss.cpu().item() + qf2_loss.cpu().item()) / 2,
                    LossAlpha=0.0,
                    Q1Vals=q1_pred.detach().cpu().numpy(),
                    Alpha=0.0,
                    LogPi=np.zeros(1),
                    PreTanh=policy_actions.detach().abs().cpu().numpy().reshape(-1),
                )

        if num_update == 0:
            logger.store(
                LossCond=0, LossPi=0, LossQ1=0, LossAlpha=0,
                Q1Vals=0, Alpha=0, LogPi=0, PreTanh=0,
            )

    def train_pred_net(self, batch_size, mask=True, update_proportion=0.25):
        if self.pred_optimizer is not None:
            _, obs_next_tensor, _, _, _ = self.sample_real_data_recent(batch_size=batch_size)
            self.pred_optimizer.zero_grad()
            pred_loss = self.compute_intrinsic_reward(obs_next_tensor, accumulate=False)
            if mask:
                mask_tensor = torch.rand(len(pred_loss)).to(self.device)
                mask_tensor = (mask_tensor < update_proportion).float().to(self.device)
                pred_loss = (pred_loss * mask_tensor).mean() / torch.max(
                    mask_tensor.sum(), torch.tensor(1.0, device=self.device)
                )
            else:
                pred_loss = pred_loss.mean()
            pred_loss.backward()
            self.pred_optimizer.step()
        else:
            pred_loss = Tensor([0])
        return pred_loss.detach().cpu().numpy()

    def update_discounted_return_stats(self, gamma=0.99):
        if not self.normalize_intrinsic_reward or len(self.episode_intrinsic_rewards) == 0:
            self.episode_intrinsic_rewards = []
            return
        intrinsic_rewards = np.array(self.episode_intrinsic_rewards)
        discounted_return = sum((gamma ** t) * intrinsic_rewards[t] for t in range(len(intrinsic_rewards)))
        self.discounted_return_rms.update(discounted_return)
        self.episode_intrinsic_rewards = []

    def set_normalize_intrinsic_reward(self, enable):
        self.normalize_intrinsic_reward = enable

    def update_next_obs_stats(self):
        ptr = self.replay_buffer.ptr
        if ptr == 0:
            return
        next_obs_all = self.replay_buffer.obs2_buf[:ptr]
        next_obs_tensor = torch.FloatTensor(next_obs_all).to(self.device)
        self.next_obs_rms(next_obs_tensor)
