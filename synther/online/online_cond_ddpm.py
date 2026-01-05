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
from synther.diffusion.elucidated_diffusion import REDQCondTrainer, CondDistri_RND
from synther.diffusion.diffusion_generator import CondDiffusionGenerator
from synther.diffusion.utils import construct_diffusion_model, split_diffusion_samples
from synther.online.redq_rlpd_agent import REDQRLPDCondAgent

import wandb
from synther.online.utils import PBE, RMS, compute_intr_reward, make_inputs_from_replay_buffer
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
from torch.utils.data import WeightedRandomSampler

# 1/2: import diffusion model
from synther.diffusion.diffusion import DiffusionModel, QFlow
from tqdm import tqdm
import random
from collections import namedtuple

from torch.utils.data import Dataset
from synther.diffusion.norm import MinMaxNormalizer


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
            name = f' {run_name}_square{args.square}_pow_reward{args.pow_reward}_alpha_rtb{args.alpha_rtb}_num_prior_epochs{args.num_prior_epochs}_num_posterior_epochs{args.num_posterior_epochs}_uniform{args.uniform}',
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
                "beta": args.beta,
                "backprop_iters": args.backprop_iters,
                # "amplify": args.amplify,
                "finetune_lr": args.finetune_lr,
                "ft_batch_size": args.ft_batch_size,
                "accumulation_steps": args.accumulation_steps,
                # "uniform": args.uniform,
                # "target_rnd_every": args.target_rnd_every,
                "finetune_lr": args.finetune_lr,
                "top_reward_exclude_ratio": args.top_reward_exclude_ratio,
                "pow_reward": args.pow_reward,
                "alpha_rtb": args.alpha_rtb,
                "num_prior_epochs": args.num_prior_epochs,
                "num_posterior_epochs": args.num_posterior_epochs,
                "uniform": args.uniform,
                # "sample_freq": args.sample_freq,
                "gin_config_files": args.gin_config_files,
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


            

            # import ipdb; ipdb.set_trace()

            
            # +++++++++++ 1. Training ++++++++++
            print(f'Retraining diffusion model at step {t + 1}')
            
            # +++++++++++ 1.1. unconditional diffusion prior training +++++++++++
            # set up hyperparameters
            dtype = torch.float
            
            # define prior model and optimizer
            prior_model = DiffusionModel(x_dim=diff_dims, diffusion_steps=args.diffusion_steps, inputs=inputs, skip_dims=skip_dims, disable_terminal_norm=model_terminals).to(dtype=dtype, device=device)
            prior_model.dtype=dtype
            prior_model.train()
            # prior_model_optimizer = torch.optim.Adam(prior_model.parameters(), lr=args.prior_lr)
            
            # prior_model_optimizer = torch.optim.AdamW(prior_model.parameters(), lr=args.prior_lr)
            no_decay = ['bias', 'LayerNorm.weight', 'norm.weight', '.g']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in prior_model.named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.,
                },
                {
                    'params': [p for n, p in prior_model.named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                },
            ]
            
            prior_model_optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.prior_lr, betas=args.prior_adam_betas)
            # scheduler for prior model
            if args.prior_lr_scheduler == 'linear':
                print('using linear learning rate scheduler')
                prior_model_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    prior_model_optimizer,
                    lambda step: max(0, 1 - step / args.num_prior_epochs)
                )
            elif args.prior_lr_scheduler == 'cosine':
                print('using cosine learning rate scheduler')
                prior_model_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    prior_model_optimizer,
                    # 원래는 100,000 steps
                    args.num_prior_epochs
                )
            else:
                prior_model_lr_scheduler = None
            
            # load data from replay buffer
            test_function_x = None
            test_function_y = None
            all_novelty_list = []
            
            # sample every data in replay buffer
            print(f'Loading every data in replay buffer...')
            ptr_location = agent.replay_buffer.ptr
            
            # Use make_inputs_from_replay_buffer to ensure consistency with update_normalizer
            # This ensures the data format matches exactly what update_normalizer uses
            test_function_x_np = make_inputs_from_replay_buffer(agent.replay_buffer, model_terminals=model_terminals)
            test_function_x = torch.from_numpy(test_function_x_np).float()
            
            # Extract next_obs for novelty computation (needed for test_function_y)
            # Format: [obs, actions, rewards, next_obs] (or with terminals)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            next_obs_start = obs_dim + act_dim + 1
            next_obs_end = next_obs_start + obs_dim
            all_next_obs = test_function_x[:, next_obs_start:next_obs_end]
            
            # compute test function y for all data in replay buffer
            batch_size_novelty = 4096
            with torch.no_grad():
                for i in range(0, ptr_location, batch_size_novelty):
                    batch_next_obs = all_next_obs[i:i+batch_size_novelty].to(device)
                    batch_novelty_tensor = agent.compute_intrinsic_reward(batch_next_obs, square=args.square, pow_reward=args.pow_reward)
                    batch_novelty = batch_novelty_tensor.cpu().numpy().squeeze()
                    all_novelty_list.append(batch_novelty)
                test_function_y = np.concatenate(all_novelty_list)    
            
            # test_function_x is already a tensor, just convert y
            test_function_x_tensor = test_function_x
            test_function_y_tensor = torch.FloatTensor(test_function_y)
            
            # Create a simple dataset class
            class SimpleDataset(Dataset):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                
                def __len__(self):
                    return len(self.x)
                
                def __getitem__(self, idx):
                    return self.x[idx], self.y[idx]
            test_function_dataset = SimpleDataset(test_function_x_tensor, test_function_y_tensor)
            
            # define data normalizer (x의 통계량 계산을 대체)
            # Now test_function_x uses the same format as update_normalizer, so they are consistent
            prior_model.update_normalizer(agent.replay_buffer, device=device, model_terminals=model_terminals)
            # y의 통계량 계산
            y_mean = test_function_y_tensor.mean().item()
            y_std = test_function_y_tensor.std().item()
            print(f'Mean of test function y: {y_mean}')
            print(f'Std of test function y: {y_std}')
            
            # define weights for weighted or uniform sampling 
            if args.uniform:
                # For uniform sampling, use shuffle=True instead of WeightedRandomSampler
                data_loader = DataLoader(test_function_dataset, batch_size=args.train_batch_size, shuffle=True)
            else:
                # For weighted sampling, use WeightedRandomSampler
                # use torch.exp? or not?
                weights = torch.exp((test_function_y_tensor.squeeze() - y_mean) / (y_std + 1e-7))
                sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
                data_loader = DataLoader(test_function_dataset, batch_size=args.train_batch_size, sampler=sampler)
            
            
            # define hyperparameters for training
            num_prior_epochs = args.num_prior_epochs
            num_posterior_epochs = args.num_posterior_epochs
            
            
            # training loop
            print(f'Training conditional diffusion prior...')
            agent.pred_net.eval()
            agent.fix_net.eval()
            # unnecessary except for PGR
            cond_distri = CondDistri_RND(agent, args.train_batch_size, agent.replay_buffer, args.cond_top_frac, square=args.square, pow_reward=args.pow_reward)
            prior_model.update_cond_normalizer(cond_distri, device=device)
            # round_num is not used in our settings
            # round_num = (t // retrain_diffusion_every) + 1
            
            # Calculate current epoch for logging
            cur_epoch = t // steps_per_epoch
            
            # Initialize wandb table for prior training logs
            if args.wandb:
                prior_log_table = wandb.Table(columns=["Epoch", "Training_Epoch", "Loss"])
            
            for epoch in tqdm(range(num_prior_epochs), dynamic_ncols=True):
                total_loss = 0.0
                
                # y is not used in prior model training
                for x, y in data_loader:
                    # x is already concatenated [obs, act, rew, next_obs] from test_function_x
                    
                    # normalize data
                    x_normalized = prior_model.normalizer.normalize(x.to(device))
                    x_normalized += torch.randn_like(x_normalized) * 0.001
                    prior_model_optimizer.zero_grad()
                    
                    if args.algorithm == 'PGR':
                        # normalize condition
                        print('y before normalization: ', y)
                        y = prior_model.cond_normalizer.normalize(y.to(device))
                        print('y after normalization: ', y)
                        pdb.set_trace()
                        loss = prior_model.compute_loss(x_normalized, cond=y)
                    else:
                        loss = prior_model.compute_loss(x_normalized)
                    loss.backward()
                    prior_model_optimizer.step()
                    if prior_model_lr_scheduler is not None:
                        prior_model_lr_scheduler.step()
                    total_loss += loss.item()
                print(f'Epoch: {epoch+1}/{num_prior_epochs} \tLoss: {total_loss:.7f}')
                
                # Add data to wandb table
                if args.wandb:
                    prior_log_table.add_data(
                        cur_epoch,
                        epoch + 1,
                        f"{total_loss:.7f}"
                    )
            
            # Log table at the end of prior training (with epoch-specific key to avoid overwriting)
            if args.wandb:
                wandb.log({f"Prior_Training_Log_Epoch_{cur_epoch}": prior_log_table}, step=cur_epoch)
                
            
            # reset diffusion buffer
            agent.reset_diffusion_buffer()
            print(f'Diffusion buffer reset')
            
            
            
            
            if args.rtb:
                print(f'Training conditional diffusion posterior...')
                
                
                # +++++++++++ 1.2. conditional diffusion posterior training (RTB fine-tuning) +++++++++++
                # set up hyperparameters
                alpha_rtb = args.alpha_rtb
                beta = args.beta
                
                
                # define reward proxy 
                proxy_model_ens = agent.compute_intrinsic_reward
                
                # define posterior model and optimizer
                # proxy_model_ens is not used in our settings, so replace it with agent.compute_intrinsic_reward()
                # posterior_model = QFlow(x_dim=diff_dims, diffusion_steps=args.diffusion_steps, q_net=proxy_model_ens, bc_net=prior_model, alpha=alpha, beta=beta).to(dtype=dtype, device=device)
                posterior_model = QFlow(x_dim=diff_dims, diffusion_steps=args.diffusion_steps, q_net=proxy_model_ens, bc_net=prior_model, alpha=alpha_rtb, beta=beta,
                                        square=args.square, pow_reward=args.pow_reward, obs_dim=obs_dim, act_dim=act_dim, dtype=dtype).to(device=device)
                posterior_model_optimizer = torch.optim.Adam(posterior_model.parameters(), lr=args.finetune_lr)
                
                # define weights
                xs = test_function_x_tensor.clone().detach().to(device)
                xs = prior_model.normalizer.normalize(xs)
                # Extract next_obs from xs for reward computation
                # xs shape: [N, obs_dim + act_dim + 1 + obs_dim]
                # next_obs starts at obs_dim + act_dim + 1
                next_obs_start = obs_dim + act_dim + 1
                next_obs_end = next_obs_start + obs_dim
                # xs_next_obs = xs[:, next_obs_start:next_obs_end]
                # ys = proxy_model_ens(xs_next_obs, square=args.square, pow_reward=args.pow_reward)
                ys = test_function_y_tensor.clone().detach().to(device).log()
                y_weights = torch.softmax(ys, dim=0)
                
                # fine-tuning loop
                if num_posterior_epochs > 0:
                    # Initialize wandb table for posterior training logs
                    if args.wandb:
                        posterior_log_table = wandb.Table(columns=["Epoch", "Training_Epoch", "Loss", "logZ"])
                    
                    for epoch in tqdm(range(num_posterior_epochs), dynamic_ncols=True):
                        if args.training_posterior == "both":
                            s1 = random.randint(0, 1)
                        elif args.training_posterior == "on":
                            s1 = 0
                        else: # off
                            s1 = 1
                            
                        if s1 == 0:
                            # on-policy
                            loss, logZ, x, logr = posterior_model.compute_loss(device, gfn_batch_size=args.ft_batch_size)
                            # Extract next_obs from x for reward computation
                            x_next_obs = x[:, next_obs_start:next_obs_end]
                            y = proxy_model_ens(x_next_obs, square=args.square, pow_reward=args.pow_reward).squeeze()
                        else:
                            # off-policy (reward prioritization)
                            idx = torch.multinomial(y_weights.squeeze(), args.ft_batch_size, replacement=True)
                            x = xs[idx]
                            x += torch.randn_like(x) * 0.01
                            loss, logZ = posterior_model.compute_loss_with_sample(x, device)
                            # Extract next_obs from x for reward computation
                            x_next_obs = x[:, next_obs_start:next_obs_end]
                            y = proxy_model_ens(x_next_obs, square=args.square, pow_reward=args.pow_reward).squeeze()
                            
                            
                        xs = torch.cat([xs, x], dim=0)
                        ys = torch.cat([ys, y], dim=0)
                        y_weights = torch.softmax(ys, dim=0)
                        
                        posterior_model_optimizer.zero_grad()
                        loss.backward()
                        posterior_model_optimizer.step()                
                        print(f'Epoch: {epoch+1}/{num_posterior_epochs} \tLoss: {loss.item():.9f}')
                        
                        # Add data to wandb table
                        if args.wandb:
                            logZ_value = logZ.item() if isinstance(logZ, torch.Tensor) else logZ
                            posterior_log_table.add_data(
                                cur_epoch,
                                epoch + 1,
                                f"{loss.item():.9f}",
                                f"{logZ_value:.9f}"
                            )
                    
                    # Log table at the end of posterior training (with epoch-specific key to avoid overwriting)
                    if args.wandb:
                        wandb.log({f"Posterior_Training_Log_Epoch_{cur_epoch}": posterior_log_table}, step=cur_epoch)
                        
            else:
                print(f'No posterior training')
                print(f'Posterior model is the same as prior model')
            
            
            
            # +++++++++++ training over +++++++++++
            
            # +++++++++++ 2. Sampling +++++++++++
            print(f'Sampling...')
            X_sample_total = []
            # we only need X_sample, not logR_sample
            # logR_sample_total = []
            eval_epochs = int(args.num_samples // args.sample_batch_size)
            assert args.num_samples % args.sample_batch_size == 0
            # In our settings, filtering is not used
            # if args.filtering:
            #     M = args.num_proposals
            # else: # 
            #     M = 1
                
            for _ in tqdm(range(eval_epochs)): #NOTE B * M**2 samples proposal.
                # Split into batches due to memory constraints
                # X_sample, logpf_pi, logpf_p = posterior_model.sample(bs=args.sample_batch_size * M, device=device)
                if args.algorithm == 'Ours':
                    posterior_model.eval()
                    X_sample = posterior_model.sample(bs=args.sample_batch_size, device=device, eval=True)
                elif args.algorithm == 'PGRrnd':
                    prior_model.eval()
                    cond = torch.FloatTensor(cond_distri.sample_cond(args.sample_batch_size)).to(device)
                    # pdb.set_trace()
                    cond = prior_model.cond_normalizer.normalize(cond)
                    X_sample = prior_model.sample(bs=args.sample_batch_size, device=device, eval=True, cond=cond, cfg_scale=args.cfg_scale)
                elif args.algorithm == 'SER':
                    prior_model.eval()
                    X_sample = prior_model.sample(bs=args.sample_batch_size, device=device, eval=True, cond=None, cfg_scale=None)
                else:
                    raise ValueError(f'Invalid algorithm: {args.algorithm}')
                
                # local search is not used in our settings
                if args.local_search and args.local_search_epochs > 0:
                    break
                else:
                    X_sample_total.append(X_sample)
                    # logR_sample_total.append(logR)
                    
            X_sample = torch.cat(X_sample_total, dim=0)
            # logR_sample = torch.cat(logR_sample_total, dim=0)
            
            print(f'Sampling complete')
            
            # clip
            print(f'X_sample before clipping: {X_sample}')
            # originally in PGR source code
            if isinstance(prior_model.normalizer, MinMaxNormalizer):
                print('Clipping X_sample to [-1, 1]')
                print('it works anyway')
                X_sample = torch.clamp(X_sample, -1., 1.)
            else:
                print('Not clipping X_sample')
            # unnormalize samples after clipping
            X_sample_unnorm = prior_model.normalizer.unnormalize(X_sample)
            # if actions are not clipped, then clamp them to [-1, 1]
            # X_sample_unnorm = torch.clamp(X_sample_unnorm, -1, 1)
                
            
            # put it in diffusion replay buffer
            # Convert to numpy if it's a tensor
            if isinstance(X_sample_unnorm, torch.Tensor):
                X_sample_unnorm_np = X_sample_unnorm.cpu().numpy()
            else:
                X_sample_unnorm_np = X_sample_unnorm
            
            transitions = split_diffusion_samples(X_sample_unnorm_np, env, modelled_terminals=model_terminals)
            if len(transitions) == 4:
                obs, act, rew, next_obs = transitions
                # Convert to numpy if tensors
                if isinstance(next_obs, torch.Tensor):
                    next_obs = next_obs.cpu().numpy()
                terminal = np.zeros_like(next_obs[:, 0])
            else:
                # won't be chosen
                obs, act, rew, next_obs, terminal = transitions
                # Convert to numpy if tensors
                if isinstance(next_obs, torch.Tensor):
                    next_obs = next_obs.cpu().numpy()
                if isinstance(terminal, torch.Tensor):
                    terminal = terminal.cpu().numpy()
            
            # Convert all to numpy arrays if they're tensors
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
            if isinstance(act, torch.Tensor):
                act = act.cpu().numpy()
            if isinstance(rew, torch.Tensor):
                rew = rew.cpu().numpy()
            if isinstance(next_obs, torch.Tensor):
                next_obs = next_obs.cpu().numpy()
            if isinstance(terminal, torch.Tensor):
                terminal = terminal.cpu().numpy()
                
            observations = np.array(obs).squeeze()
            actions = np.array(act).squeeze()
            rewards = np.array(rew).squeeze()
            next_observations = np.array(next_obs).squeeze()
            terminals = np.array(terminal).squeeze()
    
            num_samples_actual = len(observations)
            print(f'Adding {num_samples_actual} samples to replay buffer.')
            for o, a, r, o2, term in zip(observations, actions, rewards, next_observations, terminals):
                agent.diffusion_buffer.store(o, a, r, o2, term)
                
            # +++++++++++ sampling over +++++++++++
            print(f'Sampling over')


            
            # =============================================================================
            # Novelty computation for histogram and t-SNE visualization
            # =============================================================================
            
            if print_buffer_stats:
                ptr_location = agent.replay_buffer.ptr
                real_observations = agent.replay_buffer.obs1_buf[:ptr_location]
                real_actions = agent.replay_buffer.acts_buf[:ptr_location]
                real_next_observations = agent.replay_buffer.obs2_buf[:ptr_location]
                real_rewards = agent.replay_buffer.rews_buf[:ptr_location]
                # Print min, max, mean, std of each dimension in the obs, rew and action
                print('Buffer stats:')
                if len(observations) > 0:
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


            # Sample real and diffusion observations
            with torch.no_grad():
                # Sample real data from replay buffer
                real_obs_tensor, real_next_obs_tensor, _, _, _ = agent.sample_real_data(batch_size=5000)
                diffusion_obs_tensor, diffusion_next_obs_tensor, _, _, _ = agent.sample_diffusion_data(batch_size=5000)
                # Compute novelty (squeezed)
                real_novelty = agent.compute_intrinsic_reward(real_next_obs_tensor, square=args.square, pow_reward=args.pow_reward).cpu().numpy().squeeze()
                diffusion_novelty = agent.compute_intrinsic_reward(diffusion_next_obs_tensor, square=args.square, pow_reward=args.pow_reward).cpu().numpy().squeeze()
                # Combined 10k observations and novelty
                combined_next_obs_tensor = torch.cat([real_next_obs_tensor, diffusion_next_obs_tensor], dim=0)
                combined_novelty = agent.compute_intrinsic_reward(combined_next_obs_tensor, square=args.square, pow_reward=args.pow_reward).cpu().numpy().squeeze()     


            cur_epoch = t // steps_per_epoch

            # 1. Histogram plotting
            if (args.algorithm == 'PGRrnd' or args.algorithm == 'PGR' or args.algorithm == 'Ours' or args.algorithm == 'SER'):

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
                x_max = float(np.percentile(combined_novelty, 100))  # Top 100% threshold
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
                # real_mean = float(real_novelty.mean())
                real_mean = float(real_novelty.mean())
                # 12/24: 상위 10% reward를 제외한 평균 reward 사용, 즉 Stdnormalizer의 mean 사용
                # real_mean = reward_mean
                # real_mean = reward_mean
                diffusion_mean = float(diffusion_novelty.mean())
                # combined_mean = float(combined_novelty.mean())
                combined_mean = float(combined_novelty.mean())
                
                real_median = float(np.median(real_novelty))
                diffusion_median = float(np.median(diffusion_novelty))
                combined_median = float(np.median(combined_novelty))

                print(f'Real novelty mean: {real_mean:.7f}')
                print(f'Diffusion novelty mean: {diffusion_mean:.7f}')
                print(f'Combined novelty mean: {combined_mean:.7f}')


                # Plot and save combined histogram figure with shared axes
                fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
                axes[0].hist(real_novelty, bins=bins, color='tab:blue', alpha=0.8)
                axes[0].axvline(real_mean, color='red', linestyle='--', linewidth=2)
                axes[0].axvline(real_median, color='blue', linestyle='--', linewidth=2)
                axes[0].set_title('Real Obs Novelty')
                axes[0].set_xlabel('Novelty')
                axes[0].set_ylabel('Count')

                axes[1].hist(diffusion_novelty, bins=bins, color='tab:orange', alpha=0.8)
                axes[1].axvline(diffusion_mean, color='red', linestyle='--', linewidth=2)
                axes[1].axvline(diffusion_median, color='blue', linestyle='--', linewidth=2)
                axes[1].set_title('Diffusion Obs Novelty')
                axes[1].set_xlabel('Novelty')
                axes[1].set_ylabel('Count')

                axes[2].hist(combined_novelty, bins=bins, color='tab:green', alpha=0.8)
                # axes[2].axvline(combined_mean, color='red', linestyle='--', linewidth=2)
                axes[2].axvline(combined_mean, color='red', linestyle='--', linewidth=2)
                axes[2].axvline(combined_median, color='blue', linestyle='--', linewidth=2)
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
                
                # =============================================================================
                # Full replay buffer novelty histogram
                # =============================================================================
                print('Computing novelty for full replay buffer...')
                ptr_location = agent.replay_buffer.ptr
                all_next_obs = agent.replay_buffer.obs2_buf[:ptr_location]
                
                # Compute novelty in batches to avoid memory issues
                batch_size_novelty = 5000
                all_novelty_list = []
                # topk_threshold = getattr(agent, 'topk_threshold', None)
                with torch.no_grad():
                    for i in range(0, ptr_location, batch_size_novelty):
                        batch_next_obs = torch.FloatTensor(all_next_obs[i:i+batch_size_novelty]).to(device)
                        # if args.target_rnd_every > 0:
                        #     batch_novelty_tensor = agent.compute_intrinsic_reward_temp(batch_next_obs)
                        # else:
                        
                        batch_novelty_tensor = agent.compute_intrinsic_reward(batch_next_obs, square=args.square, pow_reward=args.pow_reward)
                        
                        # # Clip novelty values to topk_threshold if available
                        # if topk_threshold is not None:
                        #     print(f'Clipping novelty values to topk_threshold, drawing full-batch histogram: {topk_threshold}')
                        #     batch_novelty_tensor = torch.clamp(batch_novelty_tensor, max=topk_threshold)
                        
                        batch_novelty = batch_novelty_tensor.cpu().numpy().squeeze()
                        all_novelty_list.append(batch_novelty)
                
                all_novelty = np.concatenate(all_novelty_list)
                
                # Build bins for full buffer histogram
                x_min_full = float(all_novelty.min())
                x_max_full = float(np.percentile(all_novelty, 100))  # Top 5% threshold
                if x_min_full == x_max_full:
                    x_min_full -= 1e-8
                    x_max_full += 1e-8
                num_bins_full = 100
                bins_full = np.linspace(x_min_full, x_max_full, num_bins_full + 1)
                
                # Compute mean
                all_mean = float(all_novelty.mean())
                print(f'Full replay buffer novelty mean: {all_mean:.7f}')
                print(f'Full replay buffer size: {ptr_location}')
                
                # Compute percentiles (90, 80, 70, 60, 50, 40, 30, 20, 10)
                # 5% percentile is also included
                percentiles = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]
                percentile_values = {p: float(np.percentile(all_novelty, p)) for p in percentiles}
                for p, val in percentile_values.items():
                    print(f'Full replay buffer novelty {p}th percentile: {val:.7f}')
                
                # Create histogram figure for full replay buffer
                fig_full, ax_full = plt.subplots(figsize=(8, 6))
                ax_full.hist(all_novelty, bins=bins_full, color='tab:purple', alpha=0.8)
                ax_full.axvline(all_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {all_mean:.7f}')
                
                # Add vertical lines for percentiles
                colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
                for i, p in enumerate(percentiles):
                    val = percentile_values[p]
                    ax_full.axvline(val, color=colors[i], linestyle=':', linewidth=1.5, 
                                   label=f'{p}th percentile: {val:.7f}', alpha=0.8)
                
                ax_full.set_title(f'Full Replay Buffer Novelty (Epoch {cur_epoch})')
                ax_full.set_xlabel('Novelty')
                ax_full.set_ylabel('Count')
                ax_full.legend(loc='best', fontsize=8)
                ax_full.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Log to wandb
                if args.wandb:
                    wandb.log({
                        'images/full_replay_buffer_novelty_hist': wandb.Image(fig_full, caption=f'Epoch {cur_epoch}')
                    }, step=cur_epoch)
                
                # Save to disk
                out_path_full = os.path.join(out_dir, f'full_replay_buffer_novelty_hist_epoch{cur_epoch:04d}.png')
                fig_full.savefig(out_path_full)
                plt.close(fig_full)
                print(f'Saved full replay buffer novelty histogram to {out_path_full}')



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
                
            # ================Visualization Over====================

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
    parser.add_argument('--backprop_iters', type=int, default=10)
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

    parser.add_argument('--amplify', type=float, default=1.0)
    
    parser.add_argument('--square', action='store_true', default=False)
    
    parser.add_argument('--top_reward_exclude_ratio', type=float, default=0.0, 
                        help='Ratio of top rewards to exclude when computing reward statistics and threshold (default: 0.3)')
    
    parser.add_argument('--pow_reward', type=float, default=1.0)
    
    # ddqm
    parser.add_argument('--uniform', action='store_true', default=False)
    parser.add_argument('--diffusion_steps', type=int, default=128)
    parser.add_argument('--num_prior_epochs', type=int, default=100)
    parser.add_argument('--num_posterior_epochs', type=int, default=100)
    parser.add_argument('--training_posterior', type=str, default='both') # 'both', 'on', 'off'
    parser.add_argument('--filtering', action='store_true', default=False)
    parser.add_argument('--num_proposals', type=int, default=10)
    parser.add_argument('--local_search', action='store_true', default=False)
    parser.add_argument('--local_search_epochs', type=int, default=10)
    
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--num_samples', type=int, default=1000000)
    parser.add_argument('--sample_batch_size', type=int, default=100000)
    parser.add_argument('--prior_lr_scheduler', type=str, default='cosine')
    parser.add_argument('--prior_adam_betas', type=tuple, default=(0.9, 0.99))
    
    parser.add_argument('--prior_lr', type=float, default=1e-4)
    parser.add_argument('--alpha_rtb', type=float, default=1e-5)
    parser.add_argument('--cond_top_frac', type=float, default=0.25)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    

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