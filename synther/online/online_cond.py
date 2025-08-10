import warnings

warnings.filterwarnings("ignore")

import sys
sys.path.append('.')
import time

import dmcgym
import gin
import gym
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from gym.wrappers.flatten_observation import FlattenObservation
from redq.algos.core import mbpo_epoches, test_agent
from redq.utils.bias_utils import log_bias_evaluation
from redq.utils.logx import EpochLogger
from redq.utils.run_utils import setup_logger_kwargs
from synther.diffusion.elucidated_diffusion import REDQCondTrainer
from synther.diffusion.diffusion_generator import CondDiffusionGenerator
# from synther.diffusion.smc_diffusion_generator import SMCDiffusionGenerator
from synther.diffusion.utils import construct_diffusion_model
from synther.online.redq_rlpd_agent import REDQRLPDCondAgent
import copy

import wandb


def _compute_gaussian_log_prob(x, mean, std):
    """Compute log probability of x under Gaussian with given mean and std"""
    return -0.5 * ((x - mean) / std) ** 2 - torch.log(std) - 0.5 * torch.log(torch.tensor(2 * np.pi))


def elucidated_step_with_kl(
    diffusion_model,
    pre_trained_model,
    x_t,
    sigma_t,
    sigma_next,
    cond=None,
    eta=1.0
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Single denoising step for ElucidatedDiffusion with KL divergence computation.
    Only computes KL at the final step (when sigma_next is close to 0).
    """
    # Get denoised predictions from both models
    pred_x0_finetune = diffusion_model.preconditioned_network_forward(x_t, sigma_t, cond=cond)
    
    with torch.no_grad():
        pred_x0_pretrained = pre_trained_model.preconditioned_network_forward(x_t, sigma_t, cond=cond)
    
    # If this is the final step (sigma_next ≈ 0), return the denoised sample and compute KL
    if sigma_next < 1e-6:
        # Final step - return denoised sample directly
        kl_terms = torch.mean((pred_x0_finetune - pred_x0_pretrained) ** 2, dim=-1)
        return pred_x0_finetune, kl_terms
    
    # For intermediate steps, use standard Euler method from ElucidatedDiffusion
    # Compute score functions
    score_finetune = (x_t - pred_x0_finetune) / sigma_t
    score_pretrained = (x_t - pred_x0_pretrained) / sigma_t
    
    # Euler step
    x_next_finetune = x_t + (sigma_next - sigma_t) * score_finetune
    x_next_pretrained = x_t + (sigma_next - sigma_t) * score_pretrained
    
    # Compute KL divergence only if there's noise variance
    if eta > 0 and sigma_next > 1e-6:
        # Approximate KL between the two distributions
        kl_terms = torch.mean((x_next_finetune - x_next_pretrained) ** 2, dim=-1) / (2 * sigma_next ** 2)
    else:
        kl_terms = torch.zeros(x_t.size(0), device=x_t.device)
    
    return x_next_finetune, kl_terms


def fine_tune_diffusion_step(
    pre_trained_model, 
    fine_tune_model, 
    agent, 
    env,
    batch_size, 
    kl_weight, 
    num_sample_steps,
    max_grad_norm=1.0
):
    """
    Single fine-tuning step for diffusion model using reward maximization.
    Only collects loss from the final timestep as requested.
    """
    device = next(fine_tune_model.parameters()).device
    
    # Get diffusion sigmas (noise levels) 
    sigmas = fine_tune_model.sample_schedule(num_sample_steps)
    
    # Initialize from noise
    shape = (batch_size, *fine_tune_model.event_shape)
    x_t = sigmas[0] * torch.randn(shape, device=device)
    
    total_kl_loss = 0.0
    
    # Reverse diffusion process
    for i, sigma in enumerate(sigmas[:-1]):
        sigma_val = sigma.item()
        next_sigma = sigmas[i + 1].item() if i + 1 < len(sigmas) else 0.0
        
        x_t, kl_div = elucidated_step_with_kl(
            fine_tune_model,
            pre_trained_model, 
            x_t,
            sigma_val,
            next_sigma,
            cond=None
        )
        
        # Only accumulate KL loss from final step
        if i == len(sigmas) - 2:  # Final step
            total_kl_loss = kl_div.mean()
    
    # Final sample should be denoised transitions
    final_sample = x_t
    
    # Compute reward using agent's conditional network (keep gradients for backprop)
    # Split transitions into components for reward computation
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Manual splitting to maintain gradients
    start_idx = 0
    obs = final_sample[:, start_idx:start_idx + obs_dim]
    start_idx += obs_dim
    
    act = final_sample[:, start_idx:start_idx + act_dim]
    start_idx += act_dim
    
    # Skip reward (1 dimension)
    start_idx += 1
    
    next_obs = final_sample[:, start_idx:start_idx + obs_dim]
    
    # Compute curiosity reward using conditional network 
    # Keep gradients for final_sample but not for conditional network parameters
    if hasattr(agent, 'cond_net_target') and agent.cond_net_target is not None:
        # Use detached inputs to prevent gradient flow to conditional network
        with torch.no_grad():
            curiosity = agent.cond_net_target.compute_reward_abs_torch(
                obs.detach(), next_obs.detach(), act.detach()
            )
        # But maintain the gradient connection by multiplying with 1.0 * final_sample.mean() * 0 + curiosity
        # This tricks PyTorch into thinking curiosity depends on final_sample
        curiosity = curiosity + 0.0 * final_sample.sum(dim=1, keepdim=True)
    else:
        print('We are using the main conditional network')
        # with torch.no_grad():
        curiosity = agent.cond_net.compute_reward_abs_torch(
                obs, next_obs, act
            )
        # Same trick to maintain gradient connection
        curiosity = curiosity + 0.0 * final_sample.sum(dim=1, keepdim=True)
    
    reward = curiosity.squeeze(-1)
    
    # Compute loss: negative reward (for maximization) + KL regularization
    loss = -reward.mean() + kl_weight * total_kl_loss
    
    return loss, reward.mean().item(), total_kl_loss.item()


def fine_tune_diffusion_model(
    pre_trained_model,
    agent,
    env, 
    ft_epochs,
    ft_batch_size,
    kl_weight,
    lr=1e-4,
    max_grad_norm=1.0,
    num_sample_steps=32
):
    """
    Fine-tune diffusion model for reward maximization.
    """
    device = next(pre_trained_model.parameters()).device
    
    # Create a copy of the pre-trained model for fine-tuning
    fine_tune_model = copy.deepcopy(pre_trained_model)
    fine_tune_model.train()
    
    # Set up optimizer
    optimizer = torch.optim.Adam(fine_tune_model.parameters(), lr=lr)
    
    print(f"Starting diffusion fine-tuning for {ft_epochs} epochs...")
    
    for epoch in range(ft_epochs):
        optimizer.zero_grad()
        
        # avg_kl: Value should be zero in epoch 0
        loss, avg_reward, avg_kl = fine_tune_diffusion_step(
            pre_trained_model,
            fine_tune_model,
            agent,
            env,
            ft_batch_size,
            kl_weight,
            num_sample_steps,
            max_grad_norm
        )
        print(loss.item())
        loss.backward()
        
        # Clip gradients and perform optimization step
        nn.utils.clip_grad_norm_(fine_tune_model.parameters(), max_grad_norm)
        optimizer.step()
        
        # if epoch % 10 == 0:
        print(f"Fine-tuning Epoch {epoch}, Loss: {loss.item():.4f}, Avg Reward: {avg_reward:.4f}, KL Div: {avg_kl:.6f}")
    
    print("Diffusion fine-tuning completed.")
    fine_tune_model.eval()
    return fine_tune_model


@gin.configurable
def redq_sac(
        env_name,
        seed=3,
        epochs=-1,
        steps_per_epoch=1000,
        max_ep_len=1000,
        n_evals_per_epoch=1,
        logger_kwargs=dict(),
        # following are agent related hyperparameters
        hidden_sizes=(256, 256),
        replay_size=int(1e6),
        batch_size=256,
        lr=3e-4,
        gamma=0.99,
        polyak=0.995,
        alpha=0.2,
        auto_alpha=True,
        target_entropy='mbpo',
        start_steps=5000,
        delay_update_steps='auto',
        utd_ratio=20,
        num_Q=10,
        num_min=2,
        q_target_mode='min',
        policy_update_delay=20,
        # diffusion_buffer_size=int(1e6),
        diffusion_buffer_size=int(1e5),
        diffusion_sample_ratio=0.5,
        # diffusion hyperparameters
        retrain_diffusion_every=10_000,
        # num_samples=100_000,
        num_samples=10_000,
        diffusion_start=0,
        disable_diffusion=True,
        print_buffer_stats=True,
        skip_reward_norm=True,
        model_terminals=False,
        # conditional generation hyperparameters
        cfg_dropout=0.25,
        cond_top_frac=0.05,
        cfg_scale=1.0,
        cond_hidden_size=128,
        # fine-tuning hyperparameters
        enable_finetuning=False,
        ft_epochs=50,
        ft_batch_size=256,
        ft_kl_weight=0.1,
        ft_lr=1e-4,
        # # target network flag
        # target=False,
        # following are bias evaluation related
        evaluate_bias=True,
        n_mc_eval=1000,
        n_mc_cutoff=350,
        reseed_each_epoch=True,
        # wandb related
        use_wandb=False,
        wandb_project='PGR',
        wandb_group='PGR',
        wandb_name=None,
):
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    # set number of epoch
    if epochs == 'mbpo' or epochs < 0:
        epochs = mbpo_epoches.get(env_name, 300)
    total_steps = steps_per_epoch * epochs + 1
    
    # Initialize wandb if enabled
    if use_wandb:
        run_name = wandb_name or f"{env_name}_baseline_{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=wandb_project,
            group=wandb_group,
            name=run_name,
            config={
                "env_name": env_name,
                "seed": seed,
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "hidden_sizes": hidden_sizes,
                "replay_size": replay_size,
                "batch_size": batch_size,
                "lr": lr,
                "gamma": gamma,
                "polyak": polyak,
                "alpha": alpha,
                "auto_alpha": auto_alpha,
                "target_entropy": target_entropy,
                "start_steps": start_steps,
                "delay_update_steps": delay_update_steps,
                "utd_ratio": utd_ratio,
                "num_Q": num_Q,
                "num_min": num_min,
                "q_target_mode": q_target_mode,
                "policy_update_delay": policy_update_delay,
                "diffusion_buffer_size": diffusion_buffer_size,
                "diffusion_sample_ratio": diffusion_sample_ratio,
                "retrain_diffusion_every": retrain_diffusion_every,
                "num_samples": num_samples,
                "disable_diffusion": disable_diffusion,
                "cfg_dropout": cfg_dropout,
                "cond_top_frac": cond_top_frac,
                "cfg_scale": cfg_scale,
                "cond_hidden_size": cond_hidden_size,
            }
        )
        print(f"Initialized wandb run: {run_name}")

    """set up logger"""
    logger_kwargs['use_wandb'] = use_wandb
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    """set up environment and seeding"""
    env_fn = lambda: wrap_gym(gym.make(env_name))
    env, test_env, bias_eval_env = env_fn(), env_fn(), env_fn()
    print(f"Environment: {env_name} | Seed: {seed}")
    # seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    # seed environment along with env action space so that everything is properly seeded for reproducibility
    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        test_env.seed(test_env_seed)
        test_env.action_space.np_random.seed(test_env_seed)
        bias_eval_env.seed(bias_eval_env_seed)
        bias_eval_env.action_space.np_random.seed(bias_eval_env_seed)

    seed_all(epoch=0)

    """prepare to init agent"""
    # get obs and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # if environment has a smaller max episode length, then use the environment's max episode length
    env_time_limit = get_time_limit(env)
    max_ep_len = env_time_limit if max_ep_len > env_time_limit else max_ep_len
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # we need .item() to convert it from numpy float to python float
    act_limit = env.action_space.high[0].item()
    # keep track of run time
    start_time = time.time()
    # flush logger (optional)
    sys.stdout.flush()
    #################################################################################################

    """init agent + buffer and start training"""
    agent_config = {
        'env_name': env_name,
        'cond_hidden_size': cond_hidden_size,
        'hidden_sizes': hidden_sizes,
        'replay_size': replay_size,
        'batch_size': batch_size,
        'lr': lr,
        'gamma': gamma,
        'polyak': polyak,
        'alpha': alpha,
        'auto_alpha': auto_alpha,
        'target_entropy': target_entropy,
        'start_steps': start_steps,
        'delay_update_steps': delay_update_steps,
        'utd_ratio': utd_ratio,
        'num_Q': num_Q,
        'num_min': num_min,
        'q_target_mode': q_target_mode,
        'policy_update_delay': policy_update_delay,
    }
    agent = REDQRLPDCondAgent(cond_hidden_size, diffusion_buffer_size, diffusion_sample_ratio, env_name, obs_dim, act_dim, act_limit, device,
                              hidden_sizes, replay_size, batch_size,lr, gamma, polyak,
                              alpha, auto_alpha, target_entropy,
                              start_steps, delay_update_steps,
                              utd_ratio, num_Q, num_min, q_target_mode,
                              policy_update_delay)

    # set up diffusion model
    diff_dims = obs_dim + act_dim + 1 + obs_dim
    if model_terminals:
        diff_dims += 1
    inputs = torch.zeros((128, diff_dims)).float()
    if skip_reward_norm:
        skip_dims = [obs_dim + act_dim]
    else:
        skip_dims = []

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    for t in range(total_steps):
        # get action from agent
        a = agent.get_exploration_action(o, env)
        # Step the env, get next observation, reward and done signal
        o2, r, d, _ = env.step(a)

        # Very important: before we let agent store this transition,
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        ep_len += 1
        d = False if ep_len == max_ep_len else d
        

        # give new data to replay buffer
        agent.store_data(o, a, r, o2, d)
        # let agent update
        agent.train(logger)
        # set obs to next obs
        o = o2
        ep_ret += r

        if d or (ep_len == max_ep_len):
            # store episode return and length to logger
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            # reset environment
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if not disable_diffusion and (t + 1) % retrain_diffusion_every == 0 and (t + 1) >= diffusion_start:
            print(f'Retraining diffusion model at step {t + 1}')

            # import ipdb; ipdb.set_trace()

            # Train new diffusion model
            diffusion_trainer = REDQCondTrainer(
                construct_diffusion_model(
                    inputs=inputs,
                    skip_dims=skip_dims,
                    disable_terminal_norm=model_terminals,
                    cond_dim=1,
                    cfg_dropout=cfg_dropout,
                ),
                results_folder=args.results_folder,
                model_terminals=model_terminals,
            )
            diffusion_trainer.update_normalizer(agent.replay_buffer, device=device)
            cond_distri = diffusion_trainer.train_from_redq_buffer(agent.replay_buffer, agent.cond_net, top_frac=cond_top_frac,
                                                                   curr_epoch=(t // steps_per_epoch) + 1)
            agent.reset_diffusion_buffer()
            
            # Start diffusion finetuning
            if enable_finetuning:
                print("Starting diffusion fine-tuning...")
                fine_tuned_model = fine_tune_diffusion_model(
                    # 이 미친놈이 범인이었네 슈밤바, 찝찝한데 ema.model을 그냥 넘어가는 건.. 나중에 알아보자
                    # diffusion_trainer.ema.ema_model,
                    diffusion_trainer.model,
                    agent,
                    env,
                    ft_epochs=ft_epochs,
                    ft_batch_size=ft_batch_size,
                    kl_weight=ft_kl_weight,
                    lr=ft_lr,
                    # num_sample_steps=diffusion_trainer.ema.ema_model.num_sample_steps
                    num_sample_steps=128
                )
                # Use fine-tuned model for generation
                used_model = fine_tuned_model
            else:
                used_model = diffusion_trainer.ema.ema_model

            
            generator = CondDiffusionGenerator(env=env, ema_model=used_model, cond_distri=cond_distri)
            
            observations, actions, rewards, next_observations, terminals = generator.sample(num_samples=num_samples,
                                                                                            cfg_scale=cfg_scale)

            print(f'Adding {num_samples} samples to replay buffer.')
            for o, a, r, o2, term in zip(observations, actions, rewards, next_observations, terminals):
                agent.diffusion_buffer.store(o, a, r, o2, term)

            if print_buffer_stats:
                ptr_location = agent.replay_buffer.ptr
                real_observations = agent.replay_buffer.obs1_buf[:ptr_location]
                real_actions = agent.replay_buffer.acts_buf[:ptr_location]
                real_next_observations = agent.replay_buffer.obs2_buf[:ptr_location]
                real_rewards = agent.replay_buffer.rews_buf[:ptr_location]
                # Print min, max, mean, std of each dimension in the obs, rew and action
                print('Buffer stats:')
                for i in range(observations.shape[1]):
                    print(f'Diffusion Obs {i}: {np.mean(observations[:, i]):.2f} {np.std(observations[:, i]):.2f}')
                    print(
                        f'     Real Obs {i}: {np.mean(real_observations[:, i]):.2f} {np.std(real_observations[:, i]):.2f}')
                for i in range(actions.shape[1]):
                    print(f'Diffusion Action {i}: {np.mean(actions[:, i]):.2f} {np.std(actions[:, i]):.2f}')
                    print(f'     Real Action {i}: {np.mean(real_actions[:, i]):.2f} {np.std(real_actions[:, i]):.2f}')
                print(f'Diffusion Reward: {np.mean(rewards):.2f} {np.std(rewards):.2f}')
                print(f'     Real Reward: {np.mean(real_rewards):.2f} {np.std(real_rewards):.2f}')
                print(f'Replay buffer size: {ptr_location}')
                print(f'Diffusion buffer size: {agent.diffusion_buffer.ptr}')

        # End of epoch wrap-up
        if (t + 1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            returns = test_agent(agent, test_env, max_ep_len, logger, n_evals_per_epoch)  # add logging here
            if evaluate_bias:
                log_bias_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)

            # reseed should improve reproducibility (should make results the same whether bias evaluation is on or not)
            if reseed_each_epoch:
                seed_all(epoch)

            """logging"""
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time() - start_time)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('LossCond', with_min_and_max=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)

            if evaluate_bias:
                logger.log_tabular("MCDisRet", with_min_and_max=True)
                logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                logger.log_tabular("QPred", with_min_and_max=True)
                logger.log_tabular("QBias", with_min_and_max=True)
                logger.log_tabular("QBiasAbs", with_min_and_max=True)
                logger.log_tabular("NormQBias", with_min_and_max=True)
                logger.log_tabular("QBiasSqr", with_min_and_max=True)
                logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
            logger.dump_tabular()

            # flush logged information to disk
            sys.stdout.flush()
            
    # finish wandb run when training is complete
    if use_wandb:
        wandb.finish()

def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)

    return env


def get_time_limit(env: gym.Env):
    if hasattr(env, 'spec'):
        if hasattr(env.spec, 'max_episode_steps'):
            return env.spec.max_episode_steps
    if hasattr(env, 'env'):
        return get_time_limit(env.env)
    if hasattr(env, 'unwrapped'):
        return get_time_limit(env.unwrapped)
    else:
        raise ValueError("Cannot find time limit for env")


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=3,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='online_logs')
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--gin_config_files', nargs='*', type=str,
                        default=['config/online/sac_synther_dmc.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[])
    
    # wandb related arguments
    parser.add_argument('--wandb_project', type=str, default='PGR')
    parser.add_argument('--wandb_group', type=str, default='PGR')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', default=False, 
                        help='Enable wandb logging')
    
    # Fine-tuning related arguments
    parser.add_argument('--enable_finetuning', action='store_true', default=False,
                        help='Enable diffusion fine-tuning for reward maximization')
    parser.add_argument('--ft_epochs', type=int, default=50,
                        help='Number of epochs for diffusion fine-tuning')
    parser.add_argument('--ft_batch_size', type=int, default=256,
                        help='Batch size for diffusion fine-tuning')
    parser.add_argument('--ft_kl_weight', type=float, default=0.1,
                        help='KL divergence weight for fine-tuning regularization')
    parser.add_argument('--ft_lr', type=float, default=1e-4,
                        help='Learning rate for fine-tuning')
    
    # Target network flag
    # parser.add_argument('--target', action='store_true', default=False,
    #                     help='Use target conditional network and SMC sampling')
    
    args = parser.parse_args()
    
    args.results_folder = f'./{args.results_folder}/{args.results_folder}_{args.env}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    print(args.results_folder)
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    logger_kwargs = setup_logger_kwargs(args.env, args.log_dir)

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    redq_sac(args.env, seed=args.seed, target_entropy='auto', logger_kwargs=logger_kwargs,
             use_wandb=args.wandb, wandb_project=args.wandb_project,
             wandb_group=args.wandb_group, wandb_name=args.wandb_name,
             enable_finetuning=args.enable_finetuning, ft_epochs=args.ft_epochs,
             ft_batch_size=args.ft_batch_size, ft_kl_weight=args.ft_kl_weight,
             ft_lr=args.ft_lr)
