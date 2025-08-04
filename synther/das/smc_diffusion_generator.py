"""
SMC-based diffusion generator for transition data.
Replaces CondDiffusionGenerator with curiosity-aligned sampling using Sequential Monte Carlo.
"""

import gin
import gym
import numpy as np
import torch
from typing import Tuple, Optional

from synther.diffusion.norm import MinMaxNormalizer
from synther.diffusion.utils import split_diffusion_samples
from synther.diffusion.elucidated_diffusion import CondDistri
from synther.das.transition_smc_sampler import sample_with_smc
from synther.das.curiosity_reward import create_curiosity_reward_fn

@gin.configurable
class SMCDiffusionGenerator:
    """
    SMC-based diffusion generator that uses curiosity rewards for aligned sampling.
    """
    
    def __init__(
            self,
            env: gym.Env,
            ema_model,
            agent,
            cond_distri: CondDistri = None,
            num_sample_steps: int = 128,
            sample_batch_size: int = 10000,
            # sample_batch_size: int = 1000,  # Reduced default for SMC
            # SMC parameters
            num_particles: int = 8,
            batch_p: int = 1,
            resample_strategy: str = "ssp",
            ess_threshold: float = 0.5,
            tempering: str = "schedule",
            tempering_schedule: str = "exp",
            tempering_gamma: float = 0.1,
            tempering_start: float = 0.3,
            kl_coeff: float = 1.0,
            reward_scale: float = 1.0,
            use_target_net: bool = False,
            verbose: bool = False,
    ):
        """
        Initialize SMC-based diffusion generator.
        
        Args:
            env: The environment
            ema_model: The trained diffusion model (EMA)
            agent: The RL agent containing cond_net for curiosity
            cond_distri: Conditional distribution (optional, for compatibility)
            num_sample_steps: Number of diffusion sampling steps
            sample_batch_size: Batch size for sampling
            num_particles: Number of SMC particles
            batch_p: Number of particles to process in parallel
            resample_strategy: SMC resampling strategy
            ess_threshold: Effective sample size threshold
            tempering: Tempering strategy
            tempering_schedule: Tempering schedule
            tempering_gamma: Tempering parameter
            tempering_start: When to start tempering (fraction of steps)
            kl_coeff: KL coefficient for reward scaling
            reward_scale: Additional reward scaling factor
            use_target_net: Whether to use target conditional network
            verbose: Whether to print debug information
        """
        self.env = env
        self.diffusion = ema_model
        self.diffusion.eval()
        self.agent = agent
        self.cond_distri = cond_distri
        
        # Clamp samples if normalizer is MinMaxNormalizer
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_sample_steps = num_sample_steps
        self.sample_batch_size = sample_batch_size
        
        # SMC parameters
        self.num_particles = num_particles
        self.batch_p = batch_p
        self.resample_strategy = resample_strategy
        self.ess_threshold = ess_threshold
        self.tempering = tempering
        self.tempering_schedule = tempering_schedule
        self.tempering_gamma = tempering_gamma
        self.tempering_start = tempering_start
        self.kl_coeff = kl_coeff
        self.reward_scale = reward_scale
        self.use_target_net = use_target_net
        self.verbose = verbose
        
        # Get device from diffusion model
        self.device = next(self.diffusion.parameters()).device
        
        print(f'SMC Sampling using: {self.num_sample_steps} steps, {self.sample_batch_size} batch size, '
              f'{self.num_particles} particles, {self.batch_p} parallel particles.')

    def sample(
            self,
            num_samples: int,
            cfg_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate samples using SMC-based diffusion sampling with curiosity alignment.
        
        Args:
            num_samples: Total number of samples to generate
            cfg_scale: Classifier-free guidance scale
            
        Returns:
            Tuple of (observations, actions, rewards, next_observations, terminals)
        """
        assert num_samples % self.sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'
        num_batches = num_samples // self.sample_batch_size
        
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        
        # Create curiosity reward function
        reward_fn = create_curiosity_reward_fn(
            agent=self.agent,
            env=self.env,
            device=self.device,
            use_target_net=self.use_target_net,
            reward_scale=self.reward_scale
        )
        
        for i in range(num_batches):
            print(f'Generating SMC split {i + 1} of {num_batches}')
            
            # Generate condition (for compatibility with existing code)
            cond = None
            if self.cond_distri is not None:
                # Could sample conditions here if needed
                pass
            
            # SMC sampling
            sampled_outputs, log_weights, additional_info = sample_with_smc(
                diffusion_model=self.diffusion,
                reward_fn=reward_fn,
                batch_size=self.sample_batch_size,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
                cond=cond,
                cfg_scale=cfg_scale,
                num_particles=self.num_particles,
                batch_p=self.batch_p,
                resample_strategy=self.resample_strategy,
                ess_threshold=self.ess_threshold,
                tempering=self.tempering,
                tempering_schedule=self.tempering_schedule,
                tempering_gamma=self.tempering_gamma,
                tempering_start=self.tempering_start,
                kl_coeff=self.kl_coeff,
                verbose=self.verbose,
                device=self.device
            )
            
            sampled_outputs = sampled_outputs.cpu().numpy()
            
            if self.verbose:
                print(f"Batch {i+1} - ESS trace: {additional_info['ess_trace'].mean(dim=0) if len(additional_info['ess_trace']) > 0 else 'N/A'}")
                print(f"Batch {i+1} - Final rewards: {additional_info['rewards_trace'][-1] if len(additional_info['rewards_trace']) > 0 else 'N/A'}")
            
            # Split samples into (s, a, r, s') format
            transitions = split_diffusion_samples(sampled_outputs, self.env)
            if len(transitions) == 4:
                obs, act, rew, next_obs = transitions
                terminal = np.zeros_like(next_obs[:, 0])
            else:
                obs, act, rew, next_obs, terminal = transitions
                
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            next_observations.append(next_obs)
            terminals.append(terminal)
            
        # Concatenate all batches
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        next_observations = np.concatenate(next_observations, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        
        return observations, actions, rewards, next_observations, terminals

    def sample_single_batch(
            self,
            batch_size: int,
            cfg_scale: float = 1.0,
            return_additional_info: bool = False
    ):
        """
        Generate a single batch of samples with optional additional information.
        
        Args:
            batch_size: Number of samples to generate
            cfg_scale: Classifier-free guidance scale
            return_additional_info: Whether to return SMC diagnostics
            
        Returns:
            If return_additional_info is False:
                Tuple of (observations, actions, rewards, next_observations, terminals)
            If return_additional_info is True:
                Tuple of (observations, actions, rewards, next_observations, terminals, additional_info)
        """
        # Create curiosity reward function
        reward_fn = create_curiosity_reward_fn(
            agent=self.agent,
            env=self.env,
            device=self.device,
            use_target_net=self.use_target_net,
            reward_scale=self.reward_scale
        )
        
        # Generate condition
        cond = None
        if self.cond_distri is not None:
            # Could sample conditions here if needed
            pass
        
        # SMC sampling
        sampled_outputs, log_weights, additional_info = sample_with_smc(
            diffusion_model=self.diffusion,
            reward_fn=reward_fn,
            batch_size=batch_size,
            num_sample_steps=self.num_sample_steps,
            clamp=self.clamp_samples,
            cond=cond,
            cfg_scale=cfg_scale,
            num_particles=self.num_particles,
            batch_p=self.batch_p,
            resample_strategy=self.resample_strategy,
            ess_threshold=self.ess_threshold,
            tempering=self.tempering,
            tempering_schedule=self.tempering_schedule,
            tempering_gamma=self.tempering_gamma,
            tempering_start=self.tempering_start,
            kl_coeff=self.kl_coeff,
            verbose=self.verbose,
            device=self.device
        )
        
        sampled_outputs = sampled_outputs.cpu().numpy()
        
        # Split samples into (s, a, r, s') format
        transitions = split_diffusion_samples(sampled_outputs, self.env)
        if len(transitions) == 4:
            obs, act, rew, next_obs = transitions
            terminal = np.zeros_like(next_obs[:, 0])
        else:
            obs, act, rew, next_obs, terminal = transitions
        
        if return_additional_info:
            return obs, act, rew, next_obs, terminal, additional_info
        else:
            return obs, act, rew, next_obs, terminal

# Factory function for backwards compatibility
def create_smc_diffusion_generator(env, ema_model, agent, cond_distri=None, **kwargs):
    """
    Factory function to create SMC diffusion generator with default parameters.
    """
    return SMCDiffusionGenerator(
        env=env,
        ema_model=ema_model,
        agent=agent,
        cond_distri=cond_distri,
        **kwargs
    ) 