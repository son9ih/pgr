import numpy as np
import torch

from torch import Tensor


# helpers
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


# tensor helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


class CondDistri(object):
    def __init__(self, cond_net, train_batch_size, buffer, top_frac):
        self.top_frac = top_frac
        self.buffer = buffer

        self.irews_buf = np.zeros_like(buffer.rews_buf)
        for i in range(0, buffer.size, train_batch_size):
            idxs = np.arange(i, min(i + train_batch_size, buffer.size))
            obs = self.buffer.obs1_buf[idxs]
            next_obs = self.buffer.obs2_buf[idxs]
            actions = self.buffer.acts_buf[idxs]
            rewards = self.buffer.rews_buf[idxs][:, None]
            done = self.buffer.done_buf[idxs][:, None]
            with torch.no_grad():
                self.irews_buf[idxs] = cond_net.compute_reward(obs, next_obs, actions, rewards, done).squeeze().cpu().numpy()
        
        self.top_frac_indices = np.argsort(self.irews_buf, axis=0)[-int(top_frac * buffer.size):]

    def sample_batch(self, batch_size=32, idxs=None):
        """
        :param batch_size: size of minibatch
        :param idxs: specify indexes if you want specific data points
        :return: mini-batch data as a dictionary
        """
        if idxs is None:
            idxs = np.random.randint(0, self.buffer.size, size=batch_size)
        return dict(obs1=self.buffer.obs1_buf[idxs],
                    obs2=self.buffer.obs2_buf[idxs],
                    acts=self.buffer.acts_buf[idxs],
                    rews=self.buffer.rews_buf[idxs],
                    done=self.buffer.done_buf[idxs],
                    irews=self.irews_buf[idxs],
                    idxs=idxs)

    def sample_uncond(self, batch_size):
        return self.irews_buf[np.random.choice(self.irews_buf.shape[0], batch_size, replace=True), None]
    
    def sample_cond(self, batch_size):
        best_indices = np.random.choice(self.top_frac_indices, batch_size, replace=True)
        return self.irews_buf[best_indices, None]
    
    
class CondDistri_RND(object):
    def __init__(self, agent, train_batch_size, buffer, top_frac):
        self.top_frac = top_frac
        self.buffer = buffer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.irews_buf = np.zeros_like(buffer.rews_buf)
        # Use some large batch size
        for i in range(0, buffer.size, train_batch_size):
            idxs = np.arange(i, min(i + train_batch_size, buffer.size))
            obs = self.buffer.obs1_buf[idxs]
            next_obs = self.buffer.obs2_buf[idxs]
            actions = self.buffer.acts_buf[idxs]
            rewards = self.buffer.rews_buf[idxs][:, None]
            done = self.buffer.done_buf[idxs][:, None]
            with torch.no_grad():
                next_obs = Tensor(next_obs).to(self.device)
                self.irews_buf[idxs] = agent.compute_intrinsic_reward(next_obs).squeeze().cpu().numpy()
        self.top_frac_indices = np.argsort(self.irews_buf, axis=0)[-int(top_frac * buffer.size):]

    def sample_batch(self, batch_size=32, idxs=None):
        """
        :param batch_size: size of minibatch
        :param idxs: specify indexes if you want specific data points
        :return: mini-batch data as a dictionary
        """
        if idxs is None:
            idxs = np.random.randint(0, self.buffer.size, size=batch_size)
        return dict(obs1=self.buffer.obs1_buf[idxs],
                    obs2=self.buffer.obs2_buf[idxs],
                    acts=self.buffer.acts_buf[idxs],
                    rews=self.buffer.rews_buf[idxs],
                    done=self.buffer.done_buf[idxs],
                    irews=self.irews_buf[idxs],
                    idxs=idxs)

    def sample_uncond(self, batch_size):
        return self.irews_buf[np.random.choice(self.irews_buf.shape[0], batch_size, replace=True), None]
    
    def sample_cond(self, batch_size):
        best_indices = np.random.choice(self.top_frac_indices, batch_size, replace=True)
        return self.irews_buf[best_indices, None]
