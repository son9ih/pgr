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
from synther.diffusion.utils import construct_diffusion_model, split_diffusion_samples
from synther.online.redq_rlpd_agent import REDQRLPDCondAgent

import wandb
from synther.online.utils import PBE, RMS, compute_intr_reward
import pdb

import copy
import torch.optim as optim
import torch.nn as nn
from typing import Optional, Tuple, Union
import math
import pdb

import matplotlib.pyplot as plt
import os

from sklearn.manifold import TSNE

from torch.utils.data import TensorDataset, DataLoader
# from utils import split_diffusion_samples


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
        diffusion_buffer_size=int(1e6),
        diffusion_sample_ratio=0.5,
        # diffusion hyperparameters
        retrain_diffusion_every=10_000,
        num_samples=100_000,
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
        args=None,
        run_name=None,
):
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    # set number of epoch
    if epochs == 'mbpo' or epochs < 0:
        # epochs = mbpo_epoches.get(env_name, 100)
        epochs = 100
    total_steps = steps_per_epoch * epochs + 1
    
    # set seed
    seed = args.seed
    
    # set domain
    if args.env in ['quadruped-walk-v0','cheetah-run-v0','reacher-hard-v0']:
        args.domain = 'dmc'
    else:
        args.domain = 'muj'
    
    disable_diffusion = args.disable_diffusion
    
    if args.wandb:
        
        if args.rtb:
            
            wandb.init(
            project = f'{env_name}',
            group = f'{run_name.split("_")[-1]}',
            name = f' {run_name}_ft_batch_size{args.ft_batch_size}_epochs{args.backprop_epochs}_alpha{args.alpha}_beta{args.beta}',
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
            })
        
        elif args.finetune:
            wandb.init(
            project = f'{env_name}',
            group = f'{run_name.split("_")[-1]}',
            name = f' {run_name}_epochs{args.backprop_epochs}_kl_weight{args.kl_weight}_reward_coef{args.reward_coef}',
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
            })
            
        else:
                    
        # args.results_folder = run_name

            wandb.init(
                project = f'{env_name}',
                group = f'{run_name.split("_")[-1]}',
                name = run_name,
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
        print(f'Initialized wandb with run name {run_name}')

    """set up logger"""
    logger_kwargs['use_wandb'] = args.wandb
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

    # user define seed
    seed_all(seed)

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
    agent = REDQRLPDCondAgent(cond_hidden_size, diffusion_buffer_size, diffusion_sample_ratio,
                              env_name, obs_dim, act_dim, act_limit, device,
                              hidden_sizes, replay_size, batch_size,lr, gamma, polyak,
                              alpha, auto_alpha, target_entropy,
                              start_steps, delay_update_steps,
                              utd_ratio, num_Q, num_min, q_target_mode,
                              policy_update_delay,
                              args.rnd)
    
    # pbe for state entropy evaluation
    # if args.state_ent:
    print('Logging state entropy with PBE')
    rms = RMS(device=torch.device('cpu'))
    pbe = PBE(rms, args.knn_clip, args.knn_k, args.knn_avg, args.knn_rms, device=torch.device('cpu'))

    # set up diffusion model
    diff_dims = obs_dim + act_dim + 1 + obs_dim
    if model_terminals:
        diff_dims += 1
    inputs = torch.zeros((128, diff_dims)).float()
    if skip_reward_norm:
        skip_dims = [obs_dim + act_dim]
    else:
        skip_dims = []

    # o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    # Because they truncate before 1000, never get to 1000
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0   

    # One-time header registration guard for StateEnt to avoid header errors
    # state_ent_header_initialized = False

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
        
        # train RND predictor network, once in a epoch
        # if (t + 1) % steps_per_epoch == 0 and args.rnd:
        # if (t + 1) % steps_per_epoch == 0:
        # if d or (ep_len == max_ep_len):
        if (ep_len == max_ep_len):
            # print(t)
            # print(d)
            # print(ep_len == max_ep_len)
            agent.pred_net.train()
            pred_loss = agent.train_pred_net(batch_size=steps_per_epoch, mask=True)
            agent.pred_net.eval()
            logger.store(PredLoss=pred_loss)
            logger.log_tabular('PredLoss', average_only=True)

        # if d or (ep_len == max_ep_len):
        if (ep_len == max_ep_len):
            # store episode return and length to logger
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            # reset environment
            # o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            # Because they truncate before 1000, never get to 1000
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Retrain diffusion model periodically, then finetune if specified
        if not disable_diffusion and (t + 1) % retrain_diffusion_every == 0 and (t + 1) >= diffusion_start:
            # if args.rnd:
                # Regularly load predictor network weights to target network for stability reasons
            # agent.pred_net_target.load_state_dict(agent.pred_net.state_dict())
            
            
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
                args=args,
            )
            diffusion_trainer.update_normalizer(agent.replay_buffer, device=device)
            if not args.rnd:
                cond_distri = diffusion_trainer.train_from_redq_buffer(agent.replay_buffer, agent.cond_net, top_frac=cond_top_frac,
                                                                   curr_epoch=(t // steps_per_epoch) + 1)
            else:
                cond_distri = diffusion_trainer.train_from_redq_buffer_rnd(agent.replay_buffer, agent, top_frac=cond_top_frac,
                                                                   curr_epoch=(t // steps_per_epoch) + 1)
            agent.reset_diffusion_buffer()
            
            # if args.finetune:
            #     print('Setting up for fine-tuning...')
            #     # backprop_model = diffusion_trainer.ema.ema_model.to(device)
            #     backprop_model = diffusion_trainer.model.to(device)
            #     pre_trained_model = copy.deepcopy(backprop_model).to(device)
                
            #     pre_trained_model.sigmas = pre_trained_model.sigmas.to(device)
                
            #     backprop_model.train()
            #     pre_trained_model.eval()
                
            #     # scheduler = DDIMScheduler(num_train_timesteps=diffusion_trainer.train_num_stpes, device=device)
            #     scheduler = DDIMScheduler(num_train_timesteps=128, device=device)
            #     scheduler.set_timesteps(num_inference_steps=128)
                
            #     # optimizer = optim.Adam(backprop_model.parameters(), lr=args.finetune_lr)
            #     optimizer = optim.Adam(backprop_model.parameters(), lr=args.finetune_lr)
            #     kl_weight = args.kl_weight
                
            #     for epoch in range(args.backprop_epochs):
            #         loss, reward, kl_div = fine_tune_step(pre_trained_model, backprop_model, scheduler, optimizer, kl_weight, max_grad_norm=1.0,
            #                                               device=device, compute_reward = agent.compute_intrinsic_reward,
            #                                               input_dim=diff_dims, batch_size=args.ft_batch_size,
            #                                               obs_dim=obs_dim, act_dim=act_dim)
            #         # pdb.set_trace()
            #         # if epoch % 10 == 0:
            #         #     print(f"Fine-tuning Epoch {epoch}, Loss: {loss:.4f}, Reward: {reward:.4f}, KL Div: {kl_div:.16f}")
            #         print(f"Fine-tuning Epoch {epoch}, Loss: {loss:.16f}, Reward: {reward:.16f}, KL Div: {kl_div:.16f}")  
                
            #     # After fine-tuning, sync EMA model so sampling uses updated weights
            #     diffusion_trainer.ema.ema_model.load_state_dict(backprop_model.state_dict())
            #     print('Synced EMA model with fine-tuned weights for sampling.')
            
            if args.rtb:
                print('Setting up for RTB fine-tuning...')
                # Initialize fine-tuning model
                # diffusion model이 epsilon을 뱉으면 됨
                backprop_model = diffusion_trainer.model.to(device)
                backprop_model.train()  # Set to training mode for fine-tuning
                
                pre_trained_model = copy.deepcopy(backprop_model).to(device)
                pre_trained_model.eval()  # Freeze pre-trained model
                
                log_Z = torch.nn.Parameter(torch.tensor(0.0, device=device))
                
                
                # sigmas = backprop_model.sample_schedule(backprop_model.diffusion_steps).to(device)
                # sigmas의 크기는 diffusion model init할 때의 sample_schedule의 크기
                sigmas = backprop_model.sigmas.to(device)
                # print(f'length of sigmas: {sigmas.shape}')
                gammas = torch.where(
                    (sigmas >= backprop_model.S_tmin) & (sigmas <= backprop_model.S_tmax),
                    min(backprop_model.S_churn / backprop_model.num_sample_steps, math.sqrt(2) - 1),
                    0.
                )
                # sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))
                
                sigma_hats = sigmas * (1 + gammas)
                
                # print(f'sigmas: {sigmas}')
                # print(f'gammas: {gammas}')
                # print(f'sigma_hats: {sigma_hats}')
                # pdb.set_trace()
                
                # backprop_model.beta_t = backprop_model.beta_t.to(device)
                # backprop_model.alpha_t = backprop_model.alpha_t.to(device)
                # backprop_model.oneover_sqrta = backprop_model.oneover_sqrta.to(device)
                # backprop_model.sqrtmab = backprop_model.sqrtmab.to(device)
                # backprop_model.mab_over_sqrtmab_inv = backprop_model.mab_over_sqrtmab_inv.to(device)
                
                # Setup optimizer and log partition function
                optimizer = optim.Adam(backprop_model.parameters(), lr=args.finetune_lr)
                optimizer_z = optim.Adam([log_Z], lr=args.finetune_lr)
                
                # load Data from replay buffer
                ptr_location = agent.replay_buffer.ptr
                all_next_obs = torch.FloatTensor(agent.replay_buffer.obs2_buf[:ptr_location]).to(device)
                
                # Compute average of total rewards in batch-level computation
                all_rewards = []
                batch_size_stat = args.ft_batch_size
                with torch.no_grad():
                    for i in range(0, ptr_location, batch_size_stat):
                        batch_next_obs = all_next_obs[i:i+batch_size_stat]
                        batch_rewards = agent.compute_intrinsic_reward(batch_next_obs)
                        all_rewards.append(batch_rewards)
                    # calculate reward of one batch
                    all_rewards = torch.cat(all_rewards, dim=0)
                
                # This is average novelty for original buffer
                reward_mean = all_rewards.mean().item()
                reward_std = all_rewards.std().item()
                print(f"Reward statistics - Mean: {reward_mean:.7f}, Std: {reward_std:.7f}")
                
                # del, caching
                # del all_rewards, all_next_obs
                # torch.cuda.empty_cache()
                
                
                # Setup dataloader
                # return unnormalized data
                obs_data = torch.FloatTensor(agent.replay_buffer.obs1_buf[:ptr_location])
                obs_next_data = torch.FloatTensor(agent.replay_buffer.obs2_buf[:ptr_location])
                acts_data = torch.FloatTensor(agent.replay_buffer.acts_buf[:ptr_location])
                rews_data = torch.FloatTensor(agent.replay_buffer.rews_buf[:ptr_location])
                done_data = torch.FloatTensor(agent.replay_buffer.done_buf[:ptr_location])
                
                dataset = TensorDataset(obs_data, obs_next_data, acts_data, rews_data, done_data)
                dataloader = DataLoader(dataset, batch_size=args.ft_batch_size, shuffle=True, drop_last=True)
                
                # Training loop
                global_step = 0
                accumulation_steps = args.accumulation_steps
                
                print('Running RTB fine-tuning...')
                print(f'Total batches: {len(dataloader)}')
                
                for epoch in range(args.backprop_epochs):
                    # print(f'Epoch')
                    # epoch_loss = 0.0
                    # epoch_log_z = 0.0
                    # epoch_reward = 0.0
                    # epoch_log_ratio = 0.0
                    # num_batches = 0
                    
                    epoch_loss_on = []
                    epoch_reward_on = []
                    epoch_loss_off = []
                    
                    
                    
                    
                    for batch_idx, (obs, next_obs, act, rew, done) in enumerate(dataloader):
                        # on_policy_reward_norm_list = []   
                        # unnormalized data
                        # print('Processing batch ', batch_idx)
                        # print the number of total batches
                        # print(f'Total batches: {len(dataloader)}')
                        obs = obs.to(device)
                        next_obs = next_obs.to(device)
                        act = act.to(device)
                        rew = rew.to(device)
                        current_batch_size = obs.size(0)
                        
                        # Construct x1 (clean samples): [obs, act, rew, next_obs]
                        x1 = torch.cat([obs, act, rew.unsqueeze(-1), next_obs], dim=-1).to(device)
                        
                        # Normalize x1
                        x1_normalized = backprop_model.normalizer.normalize(x1)
                        
                    
                        with torch.no_grad():
                            rewards = agent.compute_intrinsic_reward(next_obs)  # r(x1)
                            # Normalize rewards using average and std computed from entire buffer
                            rewards_norm = (rewards - reward_mean) / (reward_std + 1e-8)
                        
                            
                        # On-policy Training
                        print('Starting on-policy training step...')
                        if global_step % args.sample_freq == 0:
                            optimizer.zero_grad()
                            optimizer_z.zero_grad()
                            
                            # Denoising process
                            # x, logpf_pi, logpf_p = backprop_model.sample_rtb(batch_size = args.gfn_batch_size, num_sample_steps=backprop_model.diffusion_steps, cond=None)
                            # normal_dist = torch.distributions.Normal(torch.zeros((args.gfn_batch_size, *backprop_model.event_shape),device=device), torch.ones((args.gfn_batch_size, *backprop_model.event_shape), device=device))
                            # normal_dist = torch.distributions.Normal(torch.zeros((args.gfn_batch_size, *backprop_model.event_shape),device=device), backprop_model.sigma_max * torch.ones((args.gfn_batch_size, *backprop_model.event_shape), device=device))
                            normal_dist = torch.distributions.Normal(torch.zeros((args.ft_batch_size, *backprop_model.event_shape),device=device), backprop_model.sigma_max * torch.ones((args.ft_batch_size, *backprop_model.event_shape), device=device))
                            x = normal_dist.sample()
                            # t = torch.zeros((args.gfn_batch_size,), device=x.device)
                            # dt = 1/ backprop_model.diffusion_steps
                            
                            logpf_pi = normal_dist.log_prob(x).sum(1)
                            logpf_p = normal_dist.log_prob(x).sum(1)
                            
                            # extra_steps = 1
                            # if False:
                            #     extra_steps = 20
                                
                            # x, logpf_pi, logpf_p = backprop_model.sample_rtb(batch_size = args.gfn_batch_size, cond=None, logpf_pi=logpf_pi, logpf_p=logpf_p, pre_trained_model=pre_trained_model)
                                
                            x, logpf_pi, logpf_p = backprop_model.sample_rtb(batch_size = args.ft_batch_size, cond=None, logpf_pi=logpf_pi, logpf_p=logpf_p, pre_trained_model=pre_trained_model)
                            
                            
                           
                            # caution: may not be correct
                            # need to debug
                            # pdb.set_trace()
                            # Construct x1 (clean samples): [obs, act, rew, next_obs]
                            # why don't you unnmormalize x here?
                            # x = backprop_model.normalizer.unnormalize(x)
                            # 이렇게 수동으로 index하지 말고 utils.split_diffusion_samples() 사용하기
                            # obs_next_x = x[:, obs_dim + act_dim + 1:]
                            # obs_next_x = utils.split_diffusion_samples(x, obs_dim, act_dim)
                            _, _, _, obs_next_x = split_diffusion_samples(x, env)
                            
                            # compute reward for sampled x
                            with torch.no_grad():
                                rewards_sample = agent.compute_intrinsic_reward(obs_next_x)
                            # # please log 'rewards_sample_mean'
                            # rewards_sample_mean = rewards_sample.mean().item()
                            # rewards_sample_std = rewards_sample.std().item()
                            # rewards_sample_norm = (rewards_sample - rewards_sample_mean) / (rewards_sample_std + 1e-8)
                            rewards_sample_norm = (rewards_sample-reward_mean) / (reward_std + 1e-8)
                            
                            logr = rewards_sample_norm
                            # on_policy_reward_norm_list.append(rewards_sample_norm.mean().item())
                            
                            # 수정) we are using parameterized logZ
                            # logC = (logr + args.alpha*logpf_pi - args.alpha*logpf_p).view(-1, args.gfn_batch_size)
                            # logC = logC.mean(1).repeat_interleave(args.gfn_batch_size, 0).detach()
                            # loss = 0.5*((args.alpha*logpf_p + logC - args.alpha*logpf_pi - logr.detach())**2).mean()
                            # detach()?
                            loss = 0.5*((args.alpha*logpf_p + log_Z - args.alpha*logpf_pi - args.beta*logr.detach())**2).mean()
                            
                            # Check for NaN/Inf in loss before backward to prevent corrupting model parameters
                            if torch.isnan(loss) or torch.isinf(loss):
                                print(f'Warning: NaN/Inf loss detected in on-policy training at batch {batch_idx}, skipping step')
                                optimizer.zero_grad()
                                optimizer_z.zero_grad()
                                continue
                            
                            loss.backward()
                            
                            # Gradient clipping to prevent parameter explosion and NaN weights
                            torch.nn.utils.clip_grad_norm_(backprop_model.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_([log_Z], max_norm=1.0)
                            
                            optimizer.step()
                            optimizer_z.step()
                            
                            # Check if model parameters became NaN after optimizer step
                            has_nan = False
                            for param in backprop_model.parameters():
                                if torch.isnan(param).any():
                                    print(f'Warning: Model parameters contain NaN after on-policy training step at batch {batch_idx}')
                                    has_nan = True
                                    break
                            if has_nan:
                                print('Model parameters corrupted, skipping remaining training steps for this batch')
                                optimizer.zero_grad()
                                optimizer_z.zero_grad()
                                continue
                            
                            sample_loss = loss.item()
                            # print(f'[on-policy] sample_loss: {sample_loss}, batch_size: {args.gfn_batch_size}')
                            # 아직까지 optimizer의 parameter는 오염되지 않음
                            # pdb.set_trace()
                            
                            
                            # logging
                            # This is normalized reward, prior is of course normal(0,1)
                            epoch_reward_on.append(rewards_sample_norm.mean().item())
                            epoch_loss_on.append(sample_loss)
                            
                            # logZSample = logC.mean().item()
                            # loss = logZSample = backprop_model.compute_loss()
                           

                        # pdb.set_trace()
                    
                    
                    
                        # Off-policy Training
                        print('Starting off-policy training step...')
                        
                        # Check if model parameters are corrupted BEFORE starting off-policy training
                        model_has_nan = False
                        for param in backprop_model.parameters():
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                print(f'Warning: Model parameters contain NaN/Inf before off-policy training at batch {batch_idx}')
                                print('Model was corrupted during on-policy training, skipping off-policy step')
                                model_has_nan = True
                                break
                        
                        if model_has_nan:
                            optimizer.zero_grad()
                            optimizer_z.zero_grad()
                            continue
                        
                        # Batch size becomes (args.ft_batch_size * args.gfn_batch_size) -> gradient exploding
                        # e.g.) 128 * 16 = 2048
                        # Use gradient accumulation to reduce memory usage
                        # x1_repeat = x1_normalized.repeat_interleave(args.gfn_batch_size, dim=0)
                        x1_repeat = x1_normalized
                        # batch size
                        bs = x1_repeat.shape[0]
                        # t = torch.zeros((bs,), device=x1_repeat.device)
                        # dt = 1/backprop_model.diffusion_steps
                        
                        # compute the reward
                        logr = rewards_norm
                        # logr = rewards
                        # logr = logr.repeat_interleave(args.gfn_batch_size, dim=0)
                        
                        # Gradient accumulation: split batch into chunks
                        chunk_size = bs // accumulation_steps
                        if chunk_size == 0:
                            chunk_size = bs
                            num_chunks = 1
                        else:
                            num_chunks = accumulation_steps
                        
                        # Initialize accumulators
                        accumulated_loss = 0.0
                        nan_detected = False
                        
                        # Zero gradients at the start of accumulation
                        optimizer.zero_grad()
                        optimizer_z.zero_grad()
                        
                        # Process each chunk
                        for chunk_idx in range(num_chunks):
                            start_idx = chunk_idx * chunk_size
                            end_idx = start_idx + chunk_size if chunk_idx < num_chunks - 1 else bs
                            
                            # Extract chunk
                            x1_chunk = x1_repeat[start_idx:end_idx]
                            logr_chunk = logr[start_idx:end_idx]
                            chunk_bs = x1_chunk.shape[0]
                            
                            # Initialize logpf_pi and logpf_p for this chunk
                            logpf_pi_chunk = torch.zeros((chunk_bs,), device=x1_chunk.device)
                            logpf_p_chunk = torch.zeros((chunk_bs,), device=x1_chunk.device)
                            
                            # Check model parameters before forward pass
                            param_has_nan = False
                            for param in backprop_model.parameters():
                                if torch.isnan(param).any() or torch.isinf(param).any():
                                    param_has_nan = True
                                    break
                            
                            if param_has_nan:
                                print(f'Warning: Model parameters contain NaN/Inf at off-policy chunk {chunk_idx}, batch {batch_idx}')
                                nan_detected = True
                                break
                            
                            # Forward pass for this chunk
                            logpf_pi_chunk, logpf_p_chunk = backprop_model.sample_rtb_reverse(
                                x=x1_chunk, 
                                logpf_pi=logpf_pi_chunk, 
                                logpf_p=logpf_p_chunk, 
                                pre_trained_model=pre_trained_model
                            )
                        
                        
                        
                        
                        # for i in range(backprop_model.num_sample_steps-1, -1, -1):
                        #     # sigma, sigma_next, gamma = map(lambda t: t.item(),(sigmas_and_gammas[i]))
                        #     # sigma_hat = sigma * (1 + gamma)
                        #     # pdb.set_trace()
                        #     # make forward kernel
                        #     pb_dist = torch.distributions.Normal(torch.sqrt(backprop_model.alpha_t[i])*x1_repeat, torch.sqrt(backprop_model.beta_t[i])*torch.ones_like(x1_repeat))
                        #     # sample noisy one from the kernel
                        #     new_x = pb_dist.sample()
                            
                        #     # Check model parameters before forward pass (some weights may have become NaN)
                        #     param_has_nan = False
                        #     for param in backprop_model.parameters():
                        #         if torch.isnan(param).any() or torch.isinf(param).any():
                        #             param_has_nan = True
                        #             break
                            
                        #     if param_has_nan:
                        #         print(f'Warning: Model parameters contain NaN/Inf at off-policy step {i}, batch {batch_idx}')
                        #         print('Model was corrupted, skipping remaining steps')
                        #         nan_detected = True
                        #         break
                            
                        #     # right sigma?
                        #     q_epsilon = backprop_model.score_fn(new_x, sigma=sigma_hats[i].item())
                        #     with torch.no_grad():
                        #         bc_epsilon = pre_trained_model.score_fn(new_x, sigma=sigma_hats[i].item()).detach()
                        #     epsilon = q_epsilon + bc_epsilon
                            
                        #     # Check for NaN/Inf in epsilon before using it (prevents crash)
                        #     if torch.isnan(epsilon).any() or torch.isinf(epsilon).any():
                        #         print(f'Warning: NaN/Inf detected in epsilon at off-policy step {i}, batch {batch_idx}')
                        #         # Check which component is NaN
                        #         if torch.isnan(q_epsilon).any():
                        #             print('q_epsilon (trainable model) contains NaN - model parameters are corrupted')
                        #             # Double-check model parameters
                        #             for param in backprop_model.parameters():
                        #                 if torch.isnan(param).any():
                        #                     print(f'Confirmed: Model parameter with shape {param.shape} contains NaN')
                        #         elif torch.isnan(bc_epsilon).any():
                        #             print('bc_epsilon (frozen model) contains NaN - unexpected')
                        #         # Skip this batch since model is corrupted
                        #         nan_detected = True
                        #         break
                            
                            
                            
                        #     pf_pi_dist = torch.distributions.Normal(backprop_model.oneover_sqrta[i] * (new_x - backprop_model.mab_over_sqrtmab_inv[i] * bc_epsilon), torch.sqrt(backprop_model.beta_t[i]) * torch.ones_like(new_x))
                        #     logpf_pi += pf_pi_dist.log_prob(x1_repeat).sum(1)
                        #     # pdb.set_trace()
                        #     pf_p_dist = torch.distributions.Normal(backprop_model.oneover_sqrta[i] * (new_x - backprop_model.mab_over_sqrtmab_inv[i] * epsilon), torch.sqrt(backprop_model.beta_t[i]) * torch.ones_like(new_x))
                        #     logpf_p += pf_p_dist.log_prob(x1_repeat).sum(1)
                            
                        #     x1_repeat = new_x
                        
                            # Compute loss for this chunk (scaled by 1/accumulation_steps to maintain effective learning rate)
                            loss_chunk = 0.5*((args.alpha*logpf_p_chunk + log_Z - args.alpha*logpf_pi_chunk - args.beta*logr_chunk.detach())**2).mean() / accumulation_steps
                            
                            # Check for NaN/Inf in loss
                            if torch.isnan(loss_chunk) or torch.isinf(loss_chunk):
                                print(f'Warning: NaN/Inf loss detected in off-policy chunk {chunk_idx}, batch {batch_idx}, skipping chunk')
                                nan_detected = True
                                break
                            
                            # Backward pass (accumulate gradients)
                            loss_chunk.backward()
                            
                            # Accumulate loss for logging (multiply by accumulation_steps to get actual loss)
                            accumulated_loss += loss_chunk.item() * accumulation_steps
                        
                        # Skip optimizer step if NaN was detected
                        if nan_detected:
                            print(f'Skipping off-policy optimizer step due to NaN in model output at batch {batch_idx}')
                            optimizer.zero_grad()
                            optimizer_z.zero_grad()
                            continue
                        
                        # Gradient clipping to prevent parameter explosion
                        torch.nn.utils.clip_grad_norm_(backprop_model.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_([log_Z], max_norm=1.0)
                        
                        # Update optimizer after accumulating gradients from all chunks
                        optimizer.step()
                        optimizer_z.step()
                        
                        # Average loss for logging
                        batch_loss = accumulated_loss / num_chunks if num_chunks > 0 else accumulated_loss
                    
                        # logging
                        epoch_loss_off.append(batch_loss)
                        
                        global_step += 1                 
                    
                        
                    # epoch_loss_on = 0.0
                    # epoch_reward_on = 0.0
                    # epoch_loss_off = 0.0
                    if epoch % 1 == 0:
                        avg_epoch_loss_on = np.mean(epoch_loss_on)
                        avg_epoch_reward_on = np.mean(epoch_reward_on)
                        avg_epoch_loss_off = np.mean(epoch_loss_off)
                        print('======================================================================')
                        # print(f'RTB Fine-tuning Epoch {epoch + 1}/{args.backprop_epochs} | On-policy Loss: {avg_epoch_loss_on:.6f} | On-policy Reward: {avg_epoch_reward_on:.6f} | Off-policy Loss: {avg_epoch_loss_off:.6f}')
                        print(f'RTB Fine-tuning Epoch {epoch + 1}/{args.backprop_epochs} | On-policy Loss: {avg_epoch_loss_on:.6f} | On-policy Reward: {avg_epoch_reward_on:.6f} | Off-policy Loss: {avg_epoch_loss_off:.6f}')
                        print(f'log_Z item: {log_Z.item()}')
                        print('======================================================================')
                        
                # Sync EMA model with fine-tuned weights
                diffusion_trainer.ema.ema_model.load_state_dict(backprop_model.state_dict())
                print('RTB fine-tuning complete. Synced EMA model with fine-tuned weights.')
   

            # Add samples to agent replay buffer
            generator = CondDiffusionGenerator(args=args, env=env, ema_model=diffusion_trainer.ema.ema_model, cond_distri=cond_distri)
            # 샘플링 스텝 수는 128
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
                    print(f'     Real Obs {i}: {np.mean(real_observations[:, i]):.2f} {np.std(real_observations[:, i]):.2f}')
                for i in range(actions.shape[1]):
                    print(f'Diffusion Action {i}: {np.mean(actions[:, i]):.2f} {np.std(actions[:, i]):.2f}')
                    print(f'     Real Action {i}: {np.mean(real_actions[:, i]):.2f} {np.std(real_actions[:, i]):.2f}')
                print(f'Diffusion Reward: {np.mean(rewards):.2f} {np.std(rewards):.2f}')
                print(f'     Real Reward: {np.mean(real_rewards):.2f} {np.std(real_rewards):.2f}')
                print(f'Replay buffer size: {ptr_location}')
                print(f'Diffusion buffer size: {agent.diffusion_buffer.ptr}')
                
            # =============================================================================
            # Novelty computation for histogram and t-SNE visualization
            # =============================================================================
            
            # Sample real and diffusion observations
            with torch.no_grad():
                # Sample real data from replay buffer
                real_obs_tensor, _, _, _, _ = agent.sample_real_data(batch_size=5000)
                diffusion_obs_tensor, _, _, _, _ = agent.sample_diffusion_data(batch_size=5000)
                # Compute novelty (squeezed)
                real_novelty = agent.compute_intrinsic_reward(real_obs_tensor).cpu().numpy().squeeze()
                diffusion_novelty = agent.compute_intrinsic_reward(diffusion_obs_tensor).cpu().numpy().squeeze()
                # Combined 10k observations and novelty
                combined_obs_tensor = torch.cat([real_obs_tensor, diffusion_obs_tensor], dim=0)
                combined_novelty = agent.compute_intrinsic_reward(combined_obs_tensor).cpu().numpy().squeeze()     
            
            
            cur_epoch = t // steps_per_epoch
            
            # 1. Histogram plotting
            if (args.algorithm == 'PGRrnd' or args.algorithm == 'PGR' or args.algorithm == 'Ours'):

                # Prepare output directory
                out_dir = os.path.join(args.results_folder, 'histograms')
                os.makedirs(out_dir, exist_ok=True)
                # cur_epoch = t // steps_per_epoch

                # # Build shared bins/ranges using the widest x-range across the three arrays
                # x_min = float(min(real_novelty.min(), diffusion_novelty.min(), combined_novelty.min()))
                # x_max = float(max(real_novelty.max(), diffusion_novelty.max(), combined_novelty.max()))
                # if x_min == x_max:
                #     # avoid zero-width bins if all values are identical
                #     x_min -= 1e-8
                #     x_max += 1e-8
                # num_bins = 100
                # bins = np.linspace(x_min, x_max, num_bins + 1)
                # Build shared bins/ranges using the widest x-range across the three arrays
                x_min = float(min(real_novelty.min(), diffusion_novelty.min(), combined_novelty.min()))
                x_max = float(np.percentile(combined_novelty, 98))  # Top 2% threshold
                if x_min == x_max:
                    # avoid zero-width bins if all values are identical
                    x_min -= 1e-8
                    x_max += 1e-8
                num_bins = 100
                bins = np.linspace(x_min, x_max, num_bins + 1)

                # Pre-compute counts to unify y-axis range by the maximum count among the three
                counts_real, _ = np.histogram(real_novelty, bins=bins)
                counts_diff, _ = np.histogram(diffusion_novelty, bins=bins)
                counts_comb, _ = np.histogram(combined_novelty, bins=bins)
                y_max = int(max(counts_real.max(), counts_diff.max(), counts_comb.max()))
                # small headroom on y-axis
                y_max = max(1, int(np.ceil(y_max * 1.05)))
                
                # Compute mean values
                real_mean = float(real_novelty.mean())
                diffusion_mean = float(diffusion_novelty.mean())
                combined_mean = float(combined_novelty.mean())
                
                print(f'Real novelty mean: {real_mean:.7f}')
                print(f'Diffusion novelty mean: {diffusion_mean:.7f}')
                print(f'Combined novelty mean: {combined_mean:.7f}')


                # Plot and save combined histogram figure with shared axes
                fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
                axes[0].hist(real_novelty, bins=bins, color='tab:blue', alpha=0.8)
                axes[0].axvline(real_mean, color='red', linestyle='--', linewidth=2)
                axes[0].set_title('Real Obs Novelty')
                axes[0].set_xlabel('Novelty')
                axes[0].set_ylabel('Count')

                axes[1].hist(diffusion_novelty, bins=bins, color='tab:orange', alpha=0.8)
                axes[1].axvline(diffusion_mean, color='red', linestyle='--', linewidth=2)
                axes[1].set_title('Diffusion Obs Novelty')
                axes[1].set_xlabel('Novelty')
                axes[1].set_ylabel('Count')

                axes[2].hist(combined_novelty, bins=bins, color='tab:green', alpha=0.8)
                axes[2].axvline(combined_mean, color='red', linestyle='--', linewidth=2)
                axes[2].set_title('Combined (10k) Novelty')
                axes[2].set_xlabel('Novelty')
                axes[2].set_ylabel('Count')

                # Unify axis ranges across all three plots
                for ax in axes:
                    ax.set_xlim(bins[0], bins[-1])
                    ax.set_ylim(0, y_max)

                plt.tight_layout()

                # Optionally log to Weights & Biases
                if args.wandb:
                    wandb.log({
                        'images/novelty_hist': wandb.Image(fig, caption=f'Epoch {cur_epoch}')
                    # }, step=t+1)novelty_hist': wandb.Image(fig, caption=f'Epoch {cur_epoch}')
                    }, step=cur_epoch)
                out_path = os.path.join(out_dir, f'novelty_hist_epoch{cur_epoch:04d}.png')
                fig.savefig(out_path)
                plt.close(fig)
                print(f'Saved novelty histogram to {out_path}')
                    
                    
                    
            # 2. T-SNE
            if args.algorithm != 'REDQ' and args.algorithm != 'SAC':  # only for methods with diffusion model
                # cur_epoch = t // steps_per_epoch
                    
                # Prepare t-SNE visualization directory
                tsne_dir = os.path.join(args.results_folder, 't-sne')
                os.makedirs(tsne_dir, exist_ok=True)
                
                # Combine real and diffusion observations for t-SNE
                combined_obs = torch.cat([real_obs_tensor, diffusion_obs_tensor], dim=0).cpu().numpy()
                
                # Create labels: 0 for real, 1 for diffusion
                labels = np.concatenate([
                    np.zeros(real_obs_tensor.shape[0]),
                    np.ones(diffusion_obs_tensor.shape[0])
                ])
                
                # Apply t-SNE
                print('Computing t-SNE embedding...')
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                embedded = tsne.fit_transform(combined_obs)
                
                # Split embeddings back into real and diffusion
                real_embedded = embedded[labels == 0]
                diffusion_embedded = embedded[labels == 1]
                
                # Create t-SNE plot
                fig_tsne, ax_tsne = plt.subplots(figsize=(10, 8))
                ax_tsne.scatter(real_embedded[:, 0], real_embedded[:, 1], 
                                c='red', alpha=0.5, s=10, label='Real Data')
                ax_tsne.scatter(diffusion_embedded[:, 0], diffusion_embedded[:, 1], 
                                c='blue', alpha=0.5, s=10, label='Diffusion Data')
                ax_tsne.set_xlabel('t-SNE Dimension 1')
                ax_tsne.set_ylabel('t-SNE Dimension 2')
                ax_tsne.set_title(f't-SNE Visualization (Epoch {cur_epoch})')
                ax_tsne.legend()
                ax_tsne.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Log to wandb
                if args.wandb:
                    wandb.log({
                        'images/t-sne': wandb.Image(fig_tsne, caption=f'Epoch {cur_epoch}')
                    }, step=cur_epoch)
                
                # Save to disk
                tsne_path = os.path.join(tsne_dir, f'tsne_epoch{cur_epoch:04d}.png')
                fig_tsne.savefig(tsne_path)
                plt.close(fig_tsne)
                print(f'Saved t-SNE plot to {tsne_path}')

        # End of epoch wrap-up
        if (t + 1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            returns = test_agent(agent, test_env, max_ep_len, logger, n_evals_per_epoch)  # add logging here
            
            # Evaluate bias as in REDQ
            if evaluate_bias:
                log_bias_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)

            # reseed should improve reproducibility (should make results the same whether bias evaluation is on or not)
            if reseed_each_epoch:
                seed_all(epoch)
                
            # Evaluation of state entropy
            # 헤더 고정형 로거 대비 안전장치: 최초 dump 전에 한 번만 헤더를 미리 등록
            # if not state_ent_header_initialized and epoch == 0:
            #     logger.log_tabular('StateEnt', val=float('nan'), average_only=True)
            #     state_ent_header_initialized = True

            # 매 5의 배수 epoch에서만 계산 및 로깅 (그 외에는 저장/로깅하지 않음)
            # if (epoch % 5 == 0) and (epoch > 1):
                # obs_tensor, _, _, _, _ = agent.sample_real_data(batch_size=args.ent_eval_num)
            obs_tensor, _, _, _, _ = agent.sample_real_data_cpu(batch_size=4000)
            intr_rew = compute_intr_reward(pbe, obs_tensor)
            logger.store(StateEnt=intr_rew)
            logger.log_tabular('StateEnt', average_only=True)
            print(f'State Entropy: {intr_rew.mean():.4f}')
            

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
            
    if args.wandb:
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
    


# def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
#     return torch.linspace(start, end, timesteps)

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

class DDIMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_train_timesteps = num_train_timesteps
        
        self.betas = linear_beta_schedule(num_train_timesteps, beta_start, beta_end).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.final_alpha_cumprod = self.alphas_cumprod[0]
        self.num_inference_steps = None
        
    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        # from large number of t to smaller
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long)
        
    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = torch.where(
            prev_timestep >= 0,
            self.alphas_cumprod[prev_timestep],
            self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
    
def _compute_gaussian_log_prob(x, mean, std):
    """Compute log probability for Gaussian distribution."""
    log_scale = torch.log(std)
    return -((x - mean) ** 2) / (2 * std ** 2) - log_scale - math.log(math.sqrt(2 * math.pi))
    

    
def ddim_step_KL(
    scheduler: DDIMScheduler,
    model_output: torch.FloatTensor,
    old_model_output: torch.FloatTensor,
    timestep: torch.LongTensor,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    generator=None,
    variance_noise: Optional[torch.FloatTensor] = None,
) -> Union[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple]:
    
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - scheduler.num_train_timesteps // scheduler.num_inference_steps
    timestep = 127-timestep
    prev_timestep = timestep-1
    # prev_timestep = 127-prev_timestep
    print(f'prev_timestep: {prev_timestep}')
    print(f'timestep: {timestep}')

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0,
        scheduler.alphas_cumprod[prev_timestep],
        scheduler.final_alpha_cumprod
    )

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise
    pred_original_sample = (sample - beta_prod_t.sqrt().unsqueeze(-1) * model_output) / alpha_prod_t.sqrt().unsqueeze(-1)
    old_pred_original_sample = (sample - beta_prod_t.sqrt().unsqueeze(-1) * old_model_output) / alpha_prod_t.sqrt().unsqueeze(-1)

    # 4. compute variance
    # variance가 왜 커지지?
    variance = scheduler._get_variance(timestep, prev_timestep)
    print(f'variance: {variance}')
    std_dev_t = eta * variance.sqrt()

    # 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).sqrt().unsqueeze(-1) * model_output
    old_pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).sqrt().unsqueeze(-1) * old_model_output

    # 6. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    
    # pdb.set_trace()
    prev_sample_mean = alpha_prod_t_prev.sqrt().unsqueeze(-1) * pred_original_sample + pred_sample_direction
    old_prev_sample_mean = alpha_prod_t_prev.sqrt().unsqueeze(-1) * old_pred_original_sample + old_pred_sample_direction

    # if eta > 0 and timestep[0] > 0:
    if eta > 0 and timestep > 0:
        # print('in ddim-step-KL')
        device = model_output.device
        noise = torch.randn(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
        variance = std_dev_t.unsqueeze(-1) * noise

        prev_sample = prev_sample_mean + variance
        # print((prev_sample_mean - old_prev_sample_mean))
        # print(f'std_dev_t: {std_dev_t}')
        # print(f'prev_sample_mean: {prev_sample_mean}')
        # print(f'old_prev_sample_mean: {old_prev_sample_mean}')
        # This becomes zero
        kl_terms = (prev_sample_mean - old_prev_sample_mean)**2 / (2 * (std_dev_t**2).unsqueeze(-1))
        kl_terms = kl_terms.sum(dim=-1)  # Sum over the 2D dimensions
        print(f'kl_terms: {kl_terms}')
    else:
        pdb.set_trace()
        # non-callable
        # print('in ddim-step-KL else')
        prev_sample = prev_sample_mean
        kl_terms = torch.zeros(prev_sample_mean.size(0), device=prev_sample_mean.device)
    # print('in ddim-step-KL end')

    # 7. Compute log probability
    log_prob = _compute_gaussian_log_prob(prev_sample, prev_sample_mean, std_dev_t.unsqueeze(-1)).mean(-1)

    return prev_sample, log_prob, kl_terms

    
def fine_tune_step(pre_trained_model, fine_tune_model, scheduler, optimizer, kl_weight, max_grad_norm=1.0,
                   device='cpu', compute_reward=None,
                   input_dim=10, batch_size=256,
                   obs_dim: int = None, act_dim: int = None):
    optimizer.zero_grad()

    kl_loss = 0.0

    # x_prev = torch.randn((256, 2)).to(device)
    x_prev = torch.randn((batch_size, input_dim)).to(device)
    # batch_size = x_prev.shape[0]
    # pdb.set_trace()

    # t is getting lower (127, 126, 125, ...    )
    # then reversed
    for t in reversed(scheduler.timesteps):
        t = torch.full((batch_size,), t, device=x_prev.device, dtype=torch.long)
        cond = None
        
        # alpha_prod_t = scheduler.alphas_cumprod[t]
        # beta_prod_t = 1 - alpha_prod_t
        # sigma_t = torch.sqrt(beta_prod_t/alpha_prod_t)
        
        
        with torch.no_grad():
            # pre_trained_noise_pred = pre_trained_model(x_prev, t.float() / scheduler.num_train_timesteps)
            # pre_trained_noise_pred = pre_trained_model(x_prev, t.float() / scheduler.num_train_timesteps, cond)
            # pre_trained_noise_pred = pre_trained_model.preconditioned_network_forward(x_prev, t.float() / scheduler.num_train_timesteps, cond)
            # pre_trained_epsilon = pre_trained_model.preconditioned_network_forward(x_prev, sigma_t, cond)
            # pre_trained_epsilon = pre_trained_model.score_fn(x_prev, pre_trained_model.sigmas[t], cond)
            # OK reversed ``
            pre_trained_epsilon = pre_trained_model.score_fn(x_prev, pre_trained_model.sigmas[t][0].item(), cond)
    
        # fine_tune_noise_pred = fine_tune_model(x_prev, t.float() / scheduler.num_train_timesteps)
        # fine_tune_noise_pred = fine_tune_model(x_prev, t.float() / scheduler.num_train_timesteps, cond)
        # fine_tune_noise_pred = fine_tune_model.preconditioned_network_forward(x_prev, t.float() / scheduler.num_train_timesteps, cond)
        fine_tune_epsilon = fine_tune_model.score_fn(x_prev, pre_trained_model.sigmas[t][0].item(), cond)
        
        # sqrt_alpha = alpha_prod_t.sqrt().unsqueeze(-1)
        # sqrt_beta = beta_prod_t.sqrt().unsqueeze(-1)
        # pre_trained_epsilon = (x_prev - sqrt_alpha * denoised_pre) / sqrt_beta
        # fine_tune_epsilon = (x_prev - sqrt_alpha * denoised_ft) / sqrt_beta
    
        # x_prev, _, kl_div = ddim_step_KL(scheduler, fine_tune_noise_pred, pre_trained_noise_pred, t, x_prev, eta=1.0)
        # pdb.set_trace()
        # reversed_t = scheduler.timesteps[len(scheduler.timesteps) - t - 1]
        reversed_t = scheduler.timesteps[len(scheduler.timesteps) - t[0].item() - 1]
        # x_prev, _, kl_div = ddim_step_KL(scheduler, fine_tune_epsilon, pre_trained_epsilon, t, x_prev, eta=1.0)
        x_prev, _, kl_div = ddim_step_KL(scheduler, fine_tune_epsilon, pre_trained_epsilon, reversed_t, x_prev, eta=1.0)
        kl_loss += kl_div
        # pdb.set_trace()
        
    # Compute intrinsic reward using only next_state slice from transition vector
    if obs_dim is None or act_dim is None:
        raise ValueError('fine_tune_step requires obs_dim and act_dim to slice next_state from x_prev')
    next_obs_start = obs_dim + act_dim + 1
    next_obs_end = next_obs_start + obs_dim
    next_obs = x_prev[:, next_obs_start:next_obs_end]
    # pdb.set_trace()
    reward = compute_reward(next_obs)
    # pdb.set_trace()
    
    loss = -reward.mean() * args.reward_coef + kl_weight * kl_loss.mean()
    
    loss.backward()

    # Clip gradients and perform optimization step
    nn.utils.clip_grad_norm_(fine_tune_model.parameters(), max_grad_norm)
    optimizer.step()
    
    return loss.item(), reward.mean().item(), kl_loss.mean().item()




if __name__ == '__main__':
    import argparse
    from datetime import datetime
    import os
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--log_dir', type=str, default='online_logs')
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--gin_config_files', nargs='*', type=str,
                        default=['config/online/sac_synther_dmc.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[])
    
    # Additional arguments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--synther', action='store_true', default=False)
    
    # parser.add_argument('--state_ent', action='store_true', default=False)
    # parser.add_argument('--state_ent_every', type=int, default=5)
    parser.add_argument('--knn_clip', type=float, default=0.0)
    parser.add_argument('--knn_k', type=int, default=12)
    parser.add_argument('--knn_avg', action='store_true', default=False) # default: True
    parser.add_argument('--knn_rms', action='store_true', default=False)
    # parser.add_argument('--ent_eval_num', type=int, default=5000)
    
    parser.add_argument('--rnd', action='store_true', default=False)
    
    # finetune arguments
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--backprop_epochs', type=int, default=10)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--reward_coef', type=float, default=1.0)
    parser.add_argument('--ft_batch_size', type=int, default=1024)
    parser.add_argument('--rtb', action='store_true', default=False)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--kl_weight', type=float, default=10.0)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    
    # REDQ
    parser.add_argument('--disable_diffusion', action='store_true', default=False)
    parser.add_argument('--algorithm', type=str, default='REDQ')  # placeholder, not used directly
    
    parser.add_argument('--sample_freq', type=int, default=1)
    parser.add_argument('--gfn_batch_size', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=1.0)
    
    parser.add_argument('--domain', type=str, default=None)  # 'dmc' or 'muj'
    
    args = parser.parse_args()
    
    assert args.algorithm in ['REDQ', 'PGR', 'PGRrnd', 'SER', 'Ours', 'SAC', 'ft']
    run_name = f"{args.env}_{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}_{args.algorithm}"
    
    if args.algorithm == 'SAC':
        args.disable_diffusion = True
        args.synther = False
        args.rnd = False
        args.rtb = False
    if args.algorithm == 'REDQ':
        args.disable_diffusion = True
        args.synther = False
        args.rnd = False
        args.rtb = False
    if args.algorithm == 'SER':
        args.disable_diffusion = False
        args.synther = True
        args.rnd = False
        args.rtb = False
    if args.algorithm == 'PGR':
        args.disable_diffusion = False
        args.synther = False
        args.rnd = False
        args.rtb = False
    if args.algorithm == 'PGRrnd':
        args.disable_diffusion = False
        args.synther = False
        args.rnd = True
        args.rtb = False
    if args.algorithm == 'ft':
        args.disable_diffusion = False
        args.synther = True
        args.rnd = True
        args.rtb = False
        args.finetune = True
    if args.algorithm == 'Ours':
        args.disable_diffusion = False
        args.synther = True
        args.rnd = True
        args.rtb = True
        
        
    
    args.results_folder = f'./{args.results_folder}/{args.results_folder}_{run_name}'
    print(args.results_folder)
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    logger_kwargs = setup_logger_kwargs(args.env, args.log_dir)

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    redq_sac(args.env, target_entropy='auto', logger_kwargs=logger_kwargs, args=args, run_name=run_name)
