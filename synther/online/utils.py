import gym
import numpy as np
from gym.wrappers.flatten_observation import FlattenObservation
from redq.algos.core import ReplayBuffer

import torch


# Make transition dataset from REDQ replay buffer.
def make_inputs_from_replay_buffer(
        replay_buffer: ReplayBuffer,
        model_terminals: bool = False,
) -> np.ndarray:
    ptr_location = replay_buffer.ptr
    obs = replay_buffer.obs1_buf[:ptr_location]
    actions = replay_buffer.acts_buf[:ptr_location]
    next_obs = replay_buffer.obs2_buf[:ptr_location]
    rewards = replay_buffer.rews_buf[:ptr_location]
    inputs = [obs, actions, rewards[:, None], next_obs]
    if model_terminals:
        terminals = replay_buffer.done_buf[:ptr_location].astype(np.float32)
        inputs.append(terminals[:, None])
    return np.concatenate(inputs, axis=1)

def compute_intr_reward(pbe, obs):
    reward = pbe(obs)
    reward = reward.reshape(-1,1)
    reward = reward.cpu().numpy()
    return reward

class PBE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, rms, knn_k, device):
        self.rms = rms
        self.knn_rms = True
        self.knn_avg = True
        self.knn_k = knn_k
        self.knn_clip = 0.0
        self.device = device

    def __call__(self, rep):
        source = target = rep
        b1, b2 = source.size(0), target.size(0)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        reward, _ = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        reward = reward.reshape(-1, 1)  # (b1 * k, 1)
        reward /= self.rms(reward)[0]
        reward = torch.maximum(
            reward - self.knn_clip,
            torch.zeros_like(reward).to(
                self.device)) if self.knn_clip >= 0.0 else reward
                
        reward = reward.reshape((b1, self.knn_k))  # (b1, k)
        reward = reward.mean(dim=1, keepdim=True)  # (b1, 1)
        reward = torch.log(reward + 1.0)
        return reward
    
class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon
        self.device = device

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S
    
    def normalize(self, x):
        if self.n <= 1.0:
            # Not enough data, return as is
            return x
        std = torch.sqrt(self.S + 1e-8)
        return (x - self.M) / std


class RunningMeanStd(object):
    """Running mean and std for numpy arrays (used for discounted intrinsic return normalization)"""
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update running statistics from batch moments using parallel variance algorithm"""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def update(self, x):
        """Update statistics from a batch of values"""
        if isinstance(x, np.ndarray):
            batch_mean = np.mean(x)
            batch_var = np.var(x)
            batch_count = len(x)
        else:
            # Handle scalar
            batch_mean = float(x)
            batch_var = 0.0
            batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)
        return batch_mean
