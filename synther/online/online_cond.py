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
from gym.wrappers.flatten_observation import FlattenObservation
from redq.algos.core import mbpo_epoches, test_agent
from redq.utils.bias_utils import log_bias_evaluation
from redq.utils.logx import EpochLogger
from redq.utils.run_utils import setup_logger_kwargs
from synther.diffusion.elucidated_diffusion import REDQCondTrainer
from synther.diffusion.diffusion_generator import CondDiffusionGenerator
from synther.das.smc_diffusion_generator import SMCDiffusionGenerator
from synther.diffusion.utils import construct_diffusion_model
from synther.online.redq_rlpd_agent import REDQRLPDCondAgent
from tqdm import tqdm

import wandb


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
        start_steps=5000, # number of steps to take random actions for initial exploration
        delay_update_steps='auto',
        utd_ratio=20,
        num_Q=10,
        num_min=2,
        q_target_mode='min',
        policy_update_delay=20,
        diffusion_buffer_size=int(1e6),
        # diffusion_buffer_size=int(1e5),
        diffusion_sample_ratio=0.5,
        # diffusion hyperparameters
        retrain_diffusion_every=10_000,
        num_samples=100_000, # same as diffusion buffer size
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
        # Loss weight hyperparameters
        hyper = 1.0,
        importance_weight = False,
        gclip = False,
        use_target = False,
        ampli = 1.0,
        # SMC-DAS hyperparameters
        use_smc_sampling = False,
        smc_num_particles = 8,
        smc_batch_p = 1,
        smc_resample_strategy = "ssp",
        smc_ess_threshold = 0.5,
        smc_tempering = "schedule",
        smc_tempering_schedule = "exp",
        smc_tempering_gamma = 0.1,
        smc_tempering_start = 0.3,
        smc_kl_coeff = 1.0,
        smc_reward_scale = 1.0,
        smc_use_target_net = False,
        smc_verbose = False
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
        run_name = wandb_name or f"{env_name}_{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
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
                "hyper": hyper,
                "importance_weight": importance_weight,
                "gclip": gclip,
                "use_target": use_target,
                "ampli": ampli,
                "use_smc_sampling": use_smc_sampling,
                "smc_num_particles": smc_num_particles,
                "smc_batch_p": smc_batch_p,
                "smc_resample_strategy": smc_resample_strategy,
                "smc_ess_threshold": smc_ess_threshold,
                "smc_tempering": smc_tempering,
                "smc_tempering_schedule": smc_tempering_schedule,
                "smc_tempering_gamma": smc_tempering_gamma,
                "smc_tempering_start": smc_tempering_start,
                "smc_kl_coeff": smc_kl_coeff,
                "smc_reward_scale": smc_reward_scale,
                "smc_use_target_net": smc_use_target_net,
                "smc_verbose": smc_verbose,
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
    agent = REDQRLPDCondAgent(cond_hidden_size, diffusion_buffer_size, diffusion_sample_ratio,hyper, importance_weight, gclip, use_target, ampli, env_name, obs_dim, act_dim, act_limit, device,
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

    # Initialize progress bar for the first epoch
    current_epoch = 0
    epoch_start_step = 0
    pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {current_epoch}", 
                unit="step", position=0, leave=True)

    for t in range(total_steps):
        # Update progress bar
        steps_in_current_epoch = t - epoch_start_step
        if steps_in_current_epoch < steps_per_epoch:
            pbar.n = steps_in_current_epoch
            pbar.refresh()
        
        # get action from agent
        # 이것 떄문에 시간이 더 걸리구나 5000전까지 되게빠른 이유는 policy network evaluation도 안하고, 
        # curiosity 게산도 안하기 때문
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
            
        # End of epoch wrap-up; make sure to update target_cond_net before evaluation
        if (t + 1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            # update the epoch for agent training
            agent.update_epoch(epoch, logger)
            

        if not disable_diffusion and (t + 1) % retrain_diffusion_every == 0 and (t + 1) >= diffusion_start:
            print(f'Retraining diffusion model at step {t + 1}')

            # import ipdb; ipdb.set_trace()

            # 이 classifier-free diffusion model을 unconditional diffusion으로 바꿔야해
            # Train new diffusion model
            diffusion_trainer = REDQCondTrainer(
                # 이게 ema.ema_model이랑 같음
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
            # 그대로 두어라
            diffusion_trainer.update_normalizer(agent.replay_buffer, device=device)
            cond_distri = diffusion_trainer.train_from_redq_buffer(agent.replay_buffer, agent.cond_net, top_frac=cond_top_frac,
                                                                   curr_epoch=(t // steps_per_epoch) + 1)
            agent.reset_diffusion_buffer()

            # Add samples to agent replay buffer
            if use_smc_sampling:
                print("Using SMC-based sampling with curiosity alignment")
                generator = SMCDiffusionGenerator(
                    env=env, 
                    ema_model=diffusion_trainer.ema.ema_model, 
                    agent=agent,
                    cond_distri=cond_distri,
                    num_particles=smc_num_particles,
                    batch_p=smc_batch_p,
                    resample_strategy=smc_resample_strategy,
                    ess_threshold=smc_ess_threshold,
                    tempering=smc_tempering,
                    tempering_schedule=smc_tempering_schedule,
                    tempering_gamma=smc_tempering_gamma,
                    tempering_start=smc_tempering_start,
                    kl_coeff=smc_kl_coeff,
                    reward_scale=smc_reward_scale,
                    use_target_net=smc_use_target_net,
                    verbose=smc_verbose
                )
            else:
                print("Using standard conditional diffusion sampling")
                generator = CondDiffusionGenerator(env=env, ema_model=diffusion_trainer.ema.ema_model, cond_distri=cond_distri)
            
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
            
            
            agent.diffusion_curiosity_sum = agent.get_diffusion_buffer_curio_sum()
            
                
            

        # End of epoch wrap-up
        if (t + 1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            
            # Complete current progress bar
            pbar.n = steps_per_epoch
            pbar.refresh()
            pbar.close()
            
            # # update the epoch for agent training
            # agent.update_epoch(epoch, logger)
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
            
            # Log curiosity statistics (with mean, std, min, max automatically calculated)
            logger.log_tabular('Curiosity', with_min_and_max=True)

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
            
            # Start new progress bar for next epoch (if not the last epoch)
            if epoch < epochs - 1:
                current_epoch = epoch + 1
                epoch_start_step = t + 1
                pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {current_epoch}", 
                           unit="step", position=0, leave=True)
            
    # Close the final progress bar if it exists
    if 'pbar' in locals():
        pbar.close()
        
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
    
    # Loss weight hyperparameters
    parser.add_argument('--hyper', type=float, default=1.0,
                        help='Loss weight hyperparameter')
    parser.add_argument('--importance_weight', action='store_true', default=False,
                        help='Use importance weight for loss calculation')
    parser.add_argument('--gclip', action='store_true', default=False,
                        help='Use gradient clipping with L1 loss for high curio values')
    parser.add_argument('--target', action='store_true', default=False,
                        help='Use target conditional network for stable curiosity computation')
    parser.add_argument('--ampli', type=float, default=1.0,
                        help='Amplification factor for curiosity-based weighting')
    
    # SMC-DAS arguments
    parser.add_argument('--use_smc', action='store_true', default=False,
                        help='Use SMC-based sampling with curiosity alignment')
    parser.add_argument('--smc_num_particles', type=int, default=8,
                        help='Number of SMC particles')
    parser.add_argument('--smc_batch_p', type=int, default=1,
                        help='Number of particles to process in parallel')
    parser.add_argument('--smc_resample_strategy', type=str, default='ssp',
                        help='SMC resampling strategy')
    parser.add_argument('--smc_ess_threshold', type=float, default=0.5,
                        help='Effective sample size threshold for resampling')
    parser.add_argument('--smc_tempering', type=str, default='schedule',
                        help='SMC tempering strategy')
    parser.add_argument('--smc_tempering_schedule', type=str, default='exp',
                        help='SMC tempering schedule')
    parser.add_argument('--smc_tempering_gamma', type=float, default=0.1,
                        help='SMC tempering gamma parameter')
    parser.add_argument('--smc_tempering_start', type=float, default=0.3,
                        help='When to start tempering (fraction of steps)')
    parser.add_argument('--smc_kl_coeff', type=float, default=1.0,
                        help='KL coefficient for reward scaling')
    parser.add_argument('--smc_reward_scale', type=float, default=1.0,
                        help='Additional reward scaling factor')
    parser.add_argument('--smc_use_target_net', action='store_true', default=False,
                        help='Use target conditional network for SMC rewards')
    parser.add_argument('--smc_verbose', action='store_true', default=False,
                        help='Print SMC debug information')
    
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
             hyper=args.hyper, importance_weight=args.importance_weight, 
             gclip=args.gclip, use_target=args.target, ampli=args.ampli,
             use_smc_sampling=args.use_smc,
             smc_num_particles=args.smc_num_particles,
             smc_batch_p=args.smc_batch_p,
             smc_resample_strategy=args.smc_resample_strategy,
             smc_ess_threshold=args.smc_ess_threshold,
             smc_tempering=args.smc_tempering,
             smc_tempering_schedule=args.smc_tempering_schedule,
             smc_tempering_gamma=args.smc_tempering_gamma,
             smc_tempering_start=args.smc_tempering_start,
             smc_kl_coeff=args.smc_kl_coeff,
             smc_reward_scale=args.smc_reward_scale,
             smc_use_target_net=args.smc_use_target_net,
             smc_verbose=args.smc_verbose)
