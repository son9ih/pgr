"""
Curiosity-based reward function for SMC sampling.
Uses agent's conditional network to compute curiosity rewards for transition data.
"""

import torch
import numpy as np
from synther.diffusion.utils import split_diffusion_samples

class CuriosityRewardFunction:
    """
    Reward function that uses the agent's curiosity network (cond_net) to evaluate transitions.
    """
    
    def __init__(self, agent, env, device="cpu"):
        """
        Initialize the curiosity reward function.
        
        Args:
            agent: The RL agent containing the cond_net (curiosity network)
            env: The environment for extracting transition components
            device: Device to run computations on
        """
        self.agent = agent
        self.env = env
        self.device = device
        self.cond_net = agent.cond_net
        
    def __call__(self, transition_data):
        """
        Compute curiosity rewards for transition data.
        
        Args:
            transition_data: Tensor of shape (batch_size, transition_dim)
                            Contains concatenated [state, action, reward, next_state] data
        
        Returns:
            rewards: Tensor of shape (batch_size,) containing curiosity values
        """
        if isinstance(transition_data, torch.Tensor):
            transition_data = transition_data.cpu().numpy()
        
        # Split transition data into components
        transitions = split_diffusion_samples(transition_data, self.env)
        
        if len(transitions) == 4:
            states, actions, rewards, next_states = transitions
        else:
            states, actions, rewards, next_states, terminals = transitions
        
        # Convert to tensors and move to device
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        
        # Compute curiosity using the agent's conditional network
        with torch.no_grad():
            curiosity_rewards = self.cond_net.compute_reward_torch(states, next_states, actions)
            # Squeeze to remove dimension if necessary
            if curiosity_rewards.dim() > 1:
                curiosity_rewards = curiosity_rewards.squeeze(-1)
        
        return curiosity_rewards

class CuriosityRewardFunctionWithTargetNet:
    """
    Enhanced curiosity reward function that can optionally use target network for stable training.
    """
    
    def __init__(self, agent, env, device="cpu", use_target_net=False, reward_scale=1.0):
        """
        Initialize the curiosity reward function with optional target network.
        
        Args:
            agent: The RL agent containing the cond_net and possibly target_cond_net
            env: The environment for extracting transition components
            device: Device to run computations on
            use_target_net: Whether to use target conditional network for stability
            reward_scale: Scale factor for curiosity rewards
        """
        self.agent = agent
        self.env = env
        self.device = device
        self.use_target_net = use_target_net
        self.reward_scale = reward_scale
        
        # Use target network if available and requested
        if use_target_net and hasattr(agent, 'target_cond_net') and agent.target_cond_net is not None:
            self.cond_net = agent.target_cond_net
        else:
            self.cond_net = agent.cond_net
            
    def __call__(self, transition_data):
        """
        Compute scaled curiosity rewards for transition data.
        
        Args:
            transition_data: Tensor of shape (batch_size, transition_dim)
                            Contains concatenated [state, action, reward, next_state] data
        
        Returns:
            rewards: Tensor of shape (batch_size,) containing scaled curiosity values
        """
        if isinstance(transition_data, torch.Tensor):
            transition_data = transition_data.cpu().numpy()
        
        # Split transition data into components
        transitions = split_diffusion_samples(transition_data, self.env)
        
        if len(transitions) == 4:
            states, actions, rewards, next_states = transitions
        else:
            states, actions, rewards, next_states, terminals = transitions
        
        # Convert to tensors and move to device
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        
        # Compute curiosity using the conditional network
        with torch.no_grad():
            curiosity_rewards = self.cond_net.compute_reward_torch(states, next_states, actions)
            # Squeeze to remove dimension if necessary
            if curiosity_rewards.dim() > 1:
                curiosity_rewards = curiosity_rewards.squeeze(-1)
            
            # Apply scaling
            curiosity_rewards = curiosity_rewards * self.reward_scale
        
        return curiosity_rewards

def create_curiosity_reward_fn(agent, env, device="cpu", use_target_net=False, reward_scale=1.0):
    """
    Factory function to create a curiosity reward function.
    
    Args:
        agent: The RL agent containing the conditional networks
        env: The environment for extracting transition components
        device: Device to run computations on
        use_target_net: Whether to use target conditional network
        reward_scale: Scale factor for curiosity rewards
    
    Returns:
        Callable reward function for SMC sampling
    """
    if use_target_net or reward_scale != 1.0:
        return CuriosityRewardFunctionWithTargetNet(agent, env, device, use_target_net, reward_scale)
    else:
        return CuriosityRewardFunction(agent, env, device) 