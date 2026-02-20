import os

os.environ["MUJOCO_GL"] = "egl"        # 필수
os.environ["PYOPENGL_PLATFORM"] = "egl"
import warnings

warnings.filterwarnings("ignore")

import sys
sys.path.append('.')
import time

# import dmcgym
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
from synther.diffusion.utils import construct_diffusion_model
from synther.online.redq_rlpd_agent import REDQRLPDCondAgent, REDQRLPDCondAgent_visual

from synther.online.visual.vis_env import DMControlVisualGymEnv

import wandb
from synther.online.utils import PBE, RMS, compute_intr_reward
import pdb


from synther.diffusion.diffusion import DiffusionModel, QFlow
from synther.diffusion.elucidated_diffusion import REDQCondTrainer, CondDistri_RND, CondDistri, CondDistri_ECO
from synther.online.utils import PBE, RMS, compute_intr_reward, make_inputs_from_replay_buffer
from synther.diffusion.utils import construct_diffusion_model, split_diffusion_samples
from synther.diffusion.norm import MinMaxNormalizer
from ema_pytorch import EMA
from tqdm import tqdm
import random

# For Dynamic MSE logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _mujoco_set_state_from_obs(env: gym.Env, obs: np.ndarray) -> bool:
    """
    Best-effort helper to set MuJoCo simulator state from a Gym MuJoCo observation.

    Supports common observation layouts:
      - obs = concat(qpos[1:], qvel)  (e.g., Hopper/Walker2d/HalfCheetah in Gym)
      - obs = concat(qpos, qvel)

    Returns:
      True if state was set successfully; False otherwise.
    """
    unwrapped = getattr(env, "unwrapped", env)
    if not hasattr(unwrapped, "set_state"):
        return False
    if not hasattr(unwrapped, "model"):
        return False
    if not hasattr(unwrapped, "sim") or not hasattr(unwrapped.sim, "data"):
        return False

    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    nq = int(getattr(unwrapped.model, "nq", 0))
    nv = int(getattr(unwrapped.model, "nv", 0))
    if nq <= 0 or nv <= 0:
        return False

    # Current state for padding/fallback
    qpos_cur = np.array(unwrapped.sim.data.qpos, dtype=np.float64).copy()
    qvel_cur = np.array(unwrapped.sim.data.qvel, dtype=np.float64).copy()

    if obs.shape[0] == (nq - 1 + nv):
        qpos = qpos_cur.copy()
        qpos[1:] = obs[: nq - 1]
        qvel = obs[nq - 1 :]
    elif obs.shape[0] == (nq + nv):
        qpos = obs[:nq]
        qvel = obs[nq:]
    else:
        return False

    try:
        unwrapped.set_state(qpos, qvel)
        return True
    except Exception:
        return False


def compute_dynamic_mse_from_diffusion_buffer(
    diffusion_buffer,
    gt_env: gym.Env,
    n_samples: int = 5000,
) -> dict:
    """
    Computes per-transition Dynamic MSE on synthetic transitions stored in diffusion_buffer,
    using a "ground-truth" one-step simulator model:
      s', r = step(env | set_state(s), a)

    Dynamic MSE (per sample):
      0.5 * ( mean((s'_true - s'_gt)^2) + (r_true - r_gt)^2 )

    Returns a dict with arrays + summary stats.
    """
    n_samples = int(n_samples)
    if diffusion_buffer.size <= 0:
        return dict(ok=False, reason="empty_diffusion_buffer")

    batch = diffusion_buffer.sample_batch(batch_size=min(n_samples, diffusion_buffer.size))
    obs1 = batch["obs1"]
    acts = batch["acts"]
    obs2 = batch["obs2"]
    rews = batch["rews"]

    # Ensure gt_env is initialized once (some envs require reset before use)
    try:
        gt_env.reset()
    except Exception:
        pass

    next_obs_gt = np.zeros_like(obs2, dtype=np.float64)
    rew_gt = np.zeros_like(rews, dtype=np.float64)
    ok_mask = np.zeros((obs1.shape[0],), dtype=bool)

    unwrapped = getattr(gt_env, "unwrapped", gt_env)

    for i in range(obs1.shape[0]):
        state = obs1[i]
        action = acts[i]

        # 1) Try env.reset(state=state) style API (works for custom MuJoCo/DMC envs like in env_test.py)
        reset_ok = False
        try:
            unwrapped.reset(state=state)
            reset_ok = True
        except TypeError:
            reset_ok = False
        except Exception:
            reset_ok = False
            
        # print(f"reset_ok: {reset_ok}")

        # 2) Fallback: try MuJoCo-style set_state from observation
        if not reset_ok:
            if not _mujoco_set_state_from_obs(gt_env, state):
                continue

        try:
            o2_pred, r_pred, d_pred, _ = gt_env.step(action)
            # Some gym envs return scalar reward; ensure float
            next_obs_gt[i] = np.asarray(o2_pred, dtype=np.float64)
            rew_gt[i] = float(r_pred)
            ok_mask[i] = True
        except Exception:
            continue

    if not np.any(ok_mask):
        return dict(ok=False, reason="gt_env_set_state_failed_for_all")

    obs2_ok = obs2[ok_mask].astype(np.float64, copy=False)
    rews_ok = rews[ok_mask].astype(np.float64, copy=False)
    next_obs_gt_ok = next_obs_gt[ok_mask]
    rew_gt_ok = rew_gt[ok_mask]

    state_mse = np.mean((obs2_ok - next_obs_gt_ok) ** 2, axis=1)
    reward_mse = (rews_ok - rew_gt_ok) ** 2
    dyn_mse = 0.5 * (state_mse + reward_mse)

    return dict(
        ok=True,
        n_total=int(obs1.shape[0]),
        n_ok=int(ok_mask.sum()),
        dyn_mse=dyn_mse,
        state_mse=state_mse,
        reward_mse=reward_mse,
        dyn_mse_mean=float(np.mean(dyn_mse)),
        dyn_mse_std=float(np.std(dyn_mse)),
        dyn_mse_min=float(np.min(dyn_mse)),
        dyn_mse_max=float(np.max(dyn_mse)),
        dyn_mse_median=float(np.median(dyn_mse)),
    )



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
):
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    # set number of epoch
    # if epochs == 'mbpo' or epochs < 0:
    #     epochs = mbpo_epoches.get(env_name, 150)
    epochs = 100
    total_steps = steps_per_epoch * epochs + 1
    
    # set seed
    seed = args.seed
    
    if args.wandb:
        run_name = f"{env_name}_{seed}_{time.strftime('%Y%m%d-%H%M%S')}_Ours+{args.novelty_measure}_ftlr{args.finetune_lr}_clip{args.ft_clip_grad}_A{args.alpha_rtb}_On{args.inter_onpolicy}_anl{args.anneal}"
        wandb.init(
            project = env_name,
            group = f'Ours+{args.novelty_measure}',
            name = run_name,
            config={
                "env_name": env_name,
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
                # Arguments from parser
                "seed": args.seed,
                "wandb": args.wandb,
                "synther": args.synther,
                "knn_clip": args.knn_clip,
                "knn_k": args.knn_k,
                "knn_avg": args.knn_avg,
                "knn_rms": args.knn_rms,
                "ent_eval_num": args.ent_eval_num,
                "novelty_measure": args.novelty_measure,
                "diffusion_steps": args.diffusion_steps,
                "num_prior_epochs": args.num_prior_epochs,
                "num_posterior_epochs": args.num_posterior_epochs,
                "training_posterior": args.training_posterior,
                "train_batch_size": args.train_batch_size,
                "sample_batch_size": args.sample_batch_size,
                "prior_lr_scheduler": args.prior_lr_scheduler,
                "rtb_lr_scheduler": args.rtb_lr_scheduler,
                "prior_adam_betas": list(args.prior_adam_betas),
                "rtb_adam_betas": list(args.rtb_adam_betas),
                "prior_lr": args.prior_lr,
                "finetune_lr": args.finetune_lr,
                "alpha_rtb": args.alpha_rtb,
                "accumulation_steps": args.accumulation_steps,
                "ft_batch_size": args.ft_batch_size,
                "inter_onpolicy": args.inter_onpolicy,
                "ddim": args.ddim,
                "eta": args.eta,
                "clip_reward": args.clip_reward,
                "ft_clip_grad": args.ft_clip_grad,
                "anneal": args.anneal,
            }
        )
        print(f'Initialized wandb with run name {run_name}')

    """set up logger"""
    logger_kwargs['use_wandb'] = args.wandb
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    """set up environment and seeding"""
    pixel_envs = {'cheetah_run', 'walker_walk'}

    def env_fn():
        # Pixel-based DMControl envs with pretrained frozen visual encoder
        if env_name in pixel_envs:
            # env_name is expected to be e.g. 'cheetah-run' or 'walker-walk'
            return DMControlVisualGymEnv(
                task_name=env_name,
                env_key_for_encoder=env_name,
                device=device,
                frame_stack=1,
                action_repeat=1,
                seed=seed,
            )
        # Default: classic Gym-based state env
        return wrap_gym(gym.make(env_name))

    # breakpoint()
    # pdb.set_trace()
    env, test_env, bias_eval_env = env_fn(), env_fn(), env_fn()
    # Separate env instance used for "ground-truth" one-step dynamics/reward evaluation
    gt_dyn_env = env_fn()
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
        gt_dyn_env_seed = (seed + 30000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        test_env.seed(test_env_seed)
        test_env.action_space.np_random.seed(test_env_seed)
        bias_eval_env.seed(bias_eval_env_seed)
        bias_eval_env.action_space.np_random.seed(bias_eval_env_seed)
        gt_dyn_env.seed(gt_dyn_env_seed)
        gt_dyn_env.action_space.np_random.seed(gt_dyn_env_seed)

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

    # Use visual-aware agent when environment is DMControlVisualGymEnv
    if isinstance(env, DMControlVisualGymEnv):
        actor_dim = env.actor_dim
        critic_dim = env.critic_dim
        agent = REDQRLPDCondAgent_visual(
            cond_hidden_size,
            diffusion_buffer_size,
            diffusion_sample_ratio,
            env_name,
            obs_dim,
            act_dim,
            act_limit,
            device,
            hidden_sizes,
            replay_size,
            batch_size,
            lr,
            gamma,
            polyak,
            alpha,
            auto_alpha,
            target_entropy,
            start_steps,
            delay_update_steps,
            utd_ratio,
            num_Q,
            num_min,
            q_target_mode,
            policy_update_delay,
            actor_dim=actor_dim,
            critic_dim=critic_dim,
        )
    else:
        agent = REDQRLPDCondAgent(
            cond_hidden_size,
            diffusion_buffer_size,
            diffusion_sample_ratio,
            env_name,
            obs_dim,
            act_dim,
            act_limit,
            device,
            hidden_sizes,
            replay_size,
            batch_size,
            lr,
            gamma,
            polyak,
            alpha,
            auto_alpha,
            target_entropy,
            start_steps,
            delay_update_steps,
            utd_ratio,
            num_Q,
            num_min,
            q_target_mode,
            policy_update_delay,
        )
    
    if not args.synther and args.novelty_measure == 'rnd':
        agent.set_normalize_intrinsic_reward(True)
        print('Enabled intrinsic reward normalization for rnd')
    
    # pbe for state entropy evaluation
    print('Logging state entropy with PBE')
    rms = RMS(device=torch.device('cpu'))
    pbe = PBE(rms, args.knn_clip, args.knn_k, args.knn_avg, args.knn_rms, device=torch.device('cpu'))

    # set up diffusion model
    diff_dims = obs_dim + act_dim + 1 + obs_dim
    # true if env is mujoco
    if model_terminals:
        diff_dims += 1
    inputs = torch.zeros((128, diff_dims)).float()
    # false if env is mujoco
    if skip_reward_norm:
        skip_dims = [obs_dim + act_dim]
    else:
        skip_dims = []

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    # Guard to ensure Dynamic MSE is computed at most once per epoch

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
        
        # New novelty measure: RND
        if not args.synther and args.novelty_measure == 'rnd' and agent.normalize_intrinsic_reward:
            o2_tensor = torch.FloatTensor(o2).unsqueeze(0).to(device)
            agent.pred_net.eval()
            _ = agent.compute_intrinsic_reward(o2_tensor, accumulate=True)
            agent.pred_net.train()
            
        # TODO: diffusion sample ratio, linearly annealing from 0.5 (default) to 0.0
        # e.g. epoch 0: 0.5, epoch 100: 0.0
        if args.anneal:     
            diffusion_sample_ratio = 0.5 - (t // steps_per_epoch) * 0.5 / (epochs - 1)
            if diffusion_sample_ratio < 0.0:
                diffusion_sample_ratio = 0.0
        else:
            diffusion_sample_ratio = diffusion_sample_ratio
        
        agent.diffusion_sample_ratio = diffusion_sample_ratio
        
        # let agent update
        agent.train(logger)
        # set obs to next obs
        o = o2
        ep_ret += r
        
        # train RND predictor network, once in a epoch
        if not args.synther and d or (ep_len == max_ep_len) and args.novelty_measure == 'rnd':
            agent.pred_net.train()
            pred_loss = agent.train_pred_net(batch_size=steps_per_epoch, mask=True)
            agent.pred_net.eval()
            # logger.store(RNDPredLoss=pred_loss)
            # logger.log_tabular('RNDPredLoss', average_only=True)

        if d or (ep_len == max_ep_len):
            
            if not args.synther and args.novelty_measure == 'rnd' and agent.normalize_intrinsic_reward:
                agent.update_discounted_return_stats(gamma=gamma)
            
            # store episode return and length to logger
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            # reset environment
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if not disable_diffusion and (t + 1) % retrain_diffusion_every == 0 and (t + 1) >= diffusion_start:
            print(f'Retraining diffusion model at step {t + 1}')
            
            # ===========================================================================================================================
            # Original code
            # # import ipdb; ipdb.set_trace()

            # # Train new diffusion model
            # diffusion_trainer = REDQCondTrainer(
            #     construct_diffusion_model(
            #         inputs=inputs,
            #         skip_dims=skip_dims,
            #         disable_terminal_norm=model_terminals,
            #         cond_dim=1,
            #         cfg_dropout=cfg_dropout,
            #     ),
            #     results_folder=args.results_folder,
            #     model_terminals=model_terminals,
            #     args=args,
            # )
            # diffusion_trainer.update_normalizer(agent.replay_buffer, device=device)
            
            # if args.novelty_measure == 'curiosity':
            #     cond_distri = diffusion_trainer.train_from_redq_buffer(agent.replay_buffer, agent.cond_net, top_frac=cond_top_frac,
            #                                                        curr_epoch=(t // steps_per_epoch) + 1)
            # elif args.novelty_measure == 'rnd':
            #     cond_distri = diffusion_trainer.train_from_redq_buffer_rnd(agent.replay_buffer, agent, top_frac=cond_top_frac,
            #                                                        curr_epoch=(t // steps_per_epoch) + 1)
            # else: 
            #     raise ValueError(f'Invalid novelty measure: {args.novelty_measure}')
            
            # agent.reset_diffusion_buffer()

            # # Add samples to agent replay buffer
            # generator = CondDiffusionGenerator(args=args, env=env, ema_model=diffusion_trainer.ema.ema_model, cond_distri=cond_distri)
            # observations, actions, rewards, next_observations, terminals = generator.sample(num_samples=num_samples,
            #                                                                                 cfg_scale=cfg_scale)

            # print(f'Adding {num_samples} samples to replay buffer.')
            # for o, a, r, o2, term in zip(observations, actions, rewards, next_observations, terminals):
            #     agent.diffusion_buffer.store(o, a, r, o2, term)
            
            # ===========================================================================================================================
            
            
            dtype = torch.float32
            
            prior_model = DiffusionModel(x_dim=diff_dims, diffusion_steps=args.diffusion_steps, inputs=inputs, skip_dims=skip_dims, disable_terminal_norm=model_terminals, eta=args.eta).to(dtype=dtype, device=device)
            prior_model.train()
            
            prior_ema = EMA(prior_model, beta=0.995, update_every=10)
            prior_ema.ema_model.normalizer.to(device)
            prior_ema.ema_model.cond_normalizer.to(device)
            
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
            prior_model_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    prior_model_optimizer,
                    args.num_prior_epochs
                )
            
            test_function_x = None
            test_function_y = None
            all_novelty_list = []
            
            # sample every data in replay buffer
            print(f'Loading every data in replay buffer...')
            ptr_location = agent.replay_buffer.ptr
            
            test_function_x_np = make_inputs_from_replay_buffer(agent.replay_buffer, model_terminals=model_terminals)
            test_function_x = torch.from_numpy(test_function_x_np).float()
            
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            next_obs_start = obs_dim + act_dim + 1
            next_obs_end = next_obs_start + obs_dim
            all_next_obs = test_function_x[:, next_obs_start:next_obs_end]
            all_actions = test_function_x[:, obs_dim:obs_dim+act_dim]
            all_rewards = test_function_x[:, obs_dim+act_dim:obs_dim+act_dim+1]
            
            all_done = test_function_x[:, obs_dim+act_dim+1:obs_dim+act_dim+2]
            all_obs = test_function_x[:, :obs_dim]
            
            # compute test function y for all data in replay buffer
            batch_size_novelty = 4096
            with torch.no_grad():
                for i in range(0, ptr_location, batch_size_novelty):
                    batch_next_obs = all_next_obs[i:i+batch_size_novelty].to(device)
                    batch_obs = all_obs[i:i+batch_size_novelty].to(device)
                    # set curiosity as a measure of novelty
                    if args.novelty_measure == 'curiosity':
                        agent.cond_net.eval()
                        batch_next_obs_np = batch_next_obs.cpu().numpy()
                        batch_actions = all_actions[i:i+batch_size_novelty].cpu().numpy()
                        batch_rewards = all_rewards[i:i+batch_size_novelty].cpu().numpy()
                        batch_done = all_done[i:i+batch_size_novelty].cpu().numpy()
                        batch_obs_np = batch_obs.cpu().numpy()
                        batch_novelty_tensor = agent.cond_net.compute_reward(batch_obs_np, batch_next_obs_np, batch_actions, batch_rewards, batch_done).squeeze().to(device)
                        agent.cond_net.train()
                    elif args.novelty_measure == 'rnd':
                        batch_novelty_tensor = agent.compute_intrinsic_reward(batch_next_obs, accumulate=False)
                    elif args.novelty_measure == 'eco':
                        batch_done_tensor = all_done[i:i+batch_size_novelty].to(device) if len(all_done.shape) > 0 else None
                        batch_novelty_tensor = agent.compute_eco_reward(batch_obs)
                    else:
                        raise ValueError(f'Invalid novelty measure: {args.novelty_measure}')
                    batch_novelty = batch_novelty_tensor.cpu().numpy().squeeze()
                    all_novelty_list.append(batch_novelty)
                test_function_y = np.concatenate(all_novelty_list)    
            
            # test_function_x is already a tensor, just convert y
            test_function_x_tensor = test_function_x
            test_function_y_tensor = torch.FloatTensor(test_function_y)
            
            # compute 95-percentile of test_function_y
            if args.clip_reward > 0.0:
                test_function_y_percentile = torch.quantile(test_function_y_tensor, args.clip_reward)
                print(f'{args.clip_reward}-percentile of test function y: {test_function_y_percentile:.7f}')
            else:
                test_function_y_percentile = None
            
            # define data normalizer (x의 통계량 계산을 대체)
            # Now test_function_x uses the same format as update_normalizer, so they are consistent
            prior_model.update_normalizer(agent.replay_buffer, device=device, model_terminals=model_terminals)
            prior_ema.ema_model.update_normalizer(agent.replay_buffer, device=device, model_terminals=model_terminals)
            
            # define hyperparameters for training
            num_prior_epochs = args.num_prior_epochs
            num_posterior_epochs = args.num_posterior_epochs
            
            
            # training loop
            print(f'Training conditional diffusion prior...')
            agent.pred_net.eval()
            agent.fix_net.eval()
            # unnecessary except for PGR
            if args.novelty_measure == 'curiosity':
                cond_distri = CondDistri(agent.cond_net, args.train_batch_size, agent.replay_buffer, args.cond_top_frac)
            elif args.novelty_measure == 'rnd':
                cond_distri = CondDistri_RND(agent, args.train_batch_size, agent.replay_buffer, args.cond_top_frac)
            elif args.novelty_measure == 'eco':
                cond_distri = CondDistri_ECO(agent, args.train_batch_size, agent.replay_buffer, args.cond_top_frac)
            else:
                raise ValueError(f'Invalid novelty measure: {args.novelty_measure}')
            prior_model.update_cond_normalizer(cond_distri, device=device)
            prior_ema.ema_model.update_cond_normalizer(cond_distri, device=device)
            # round_num is not used in our settings
            # round_num = (t // retrain_diffusion_every) + 1
            
            # Calculate current epoch for logging
            cur_epoch = t // steps_per_epoch
            
            # Initialize wandb table for prior training logs
            if args.wandb:
                prior_log_table = wandb.Table(columns=["Epoch", "Training_Epoch", "Loss"])
            
            for epoch in tqdm(range(num_prior_epochs), dynamic_ncols=True):
                total_loss = 0.0
                
                # iteration based training
                b = cond_distri.sample_batch(args.train_batch_size)
                obs = b['obs1']
                next_obs = b['obs2']
                actions = b['acts']
                rewards = b['rews'][:, None]
                done = b['done'][:, None]
                cond_signal = b['irews'][:, None]

                data = [obs, actions, rewards, next_obs]
                if model_terminals:
                    data.append(done)
                data = np.concatenate(data, axis=1)
                
                # move to cuda (옮기고 normalize하는 순서가 맞음)
                data = torch.from_numpy(data).float().to(device)
                cond_signal = torch.from_numpy(cond_signal).float().to(device)
                
                # normalize data
                data = prior_model.normalizer.normalize(data)
                cond_signal = prior_model.cond_normalizer.normalize(cond_signal)
                
                if args.synther:
                    loss = prior_model.compute_loss(data, cond=None)
                else:
                    loss = prior_model.compute_loss(data, cond=cond_signal)
                    
                prior_model_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prior_model.parameters(), 1.0)
                prior_model_optimizer.step()
                
                # update learning rate scheduler
                if prior_model_lr_scheduler is not None:
                    prior_model_lr_scheduler.step()
                    
                # update EMA model
                prior_ema.to(device)
                prior_ema.update()
                
                total_loss += loss.item()
                
                
                if epoch % 1000 == 0:
                    print(f'[{epoch}/{num_prior_epochs}] loss: {total_loss:.4f}')
                
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
            
            
            
            # rtb fine-tuning
            
            alpha_rtb = args.alpha_rtb
            # beta = args.beta
            
            
            # define reward proxy 
            # update basic onpolicy reward
            agent.update_onpolicy_reward()
            print(f'max_onpolicy_reward: {agent.max_onpolicy_reward}')
            
            # Choose different reward function to optimize for different algorithms
            # For unity, each compute_intrinsic_reward and compute_reward should take the whole transition as input
            # TODO
            if args.novelty_measure == 'curiosity':
                proxy_model_ens = agent.cond_net.compute_reward_torch
            elif args.novelty_measure == 'rnd':
                proxy_model_ens = agent.compute_intrinsic_reward
            elif args.novelty_measure == 'eco':
                proxy_model_ens = agent.compute_eco_reward
            else:
                raise ValueError(f'Invalid novelty measure: {args.novelty_measure}')
            
            # define posterior model and optimizer
            # proxy_model_ens is not used in our settings, so replace it with agent.compute_intrinsic_reward()
            # posterior_model = QFlow(x_dim=diff_dims, diffusion_steps=args.diffusion_steps, q_net=proxy_model_ens, bc_net=prior_model, alpha=alpha, beta=beta).to(dtype=dtype, device=device)
            # TODO: requires an argument for onpolicy reward function in QFlow()
            # add 1) onpolicy reward function, 2) args.novelty measure (curiosity, rnd, eco)
            # TODO: We need to normalize the novelty, so that we can handle different scales of novelty measures with on-policy reward
            # EMA: Instead of using original prior, we use EMA model
            
            # TODO: deep copy prior_ema.ema_model to prior_model
            # TODO: This is the main cause of not decreasing the loss
            posterior_model = QFlow(x_dim=diff_dims, diffusion_steps=args.diffusion_steps, q_net=proxy_model_ens, bc_net=prior_ema.ema_model, alpha=alpha_rtb,
                                    obs_dim=obs_dim, act_dim=act_dim, dtype=dtype, novelty_measure=args.novelty_measure, 
                                    agent=agent, inter_onpolicy=args.inter_onpolicy, reward_percentile=test_function_y_percentile, eta=args.eta, ddim=args.ddim).to(device=device)
            
            # posterior_model = QFlow(x_dim=diff_dims, diffusion_steps=args.diffusion_steps, q_net=proxy_model_ens, bc_net=prior_model, alpha=alpha_rtb, beta=beta,
            #                         square=args.square, pow_reward=args.pow_reward, obs_dim=obs_dim, act_dim=act_dim, dtype=dtype, novelty_measure=args.novelty_measure, 
            #                         agent=agent, inter_onpolicy=args.inter_onpolicy).to(device=device)
            
            # posterior_model = QFlow(x_dim=diff_dims, diffusion_steps=args.diffusion_steps, q_net=proxy_model_ens, bc_net=prior_model, alpha=alpha_rtb, beta=beta,
            #                         square=args.square, pow_reward=args.pow_reward, obs_dim=obs_dim, act_dim=act_dim, dtype=dtype, novelty_measure=args.novelty_measure, 
            #                         agent=agent, inter_onpolicy=args.inter_onpolicy).to(device=device)
            
            # def n_trainable(m):
            #     return sum(p.numel() for p in m.parameters() if p.requires_grad)

            # print("qflow trainable params:", n_trainable(posterior_model.qflow))
            # print("bc_net policy trainable params:", n_trainable(posterior_model.bc_net.policy))
            # print("logZ requires_grad:", posterior_model.logZ.requires_grad)
            
            # posterior_model_optimizer = torch.optim.Adam(posterior_model.parameters(), lr=args.finetune_lr)
            # posterior_model_optimizer = torch.optim.AdamW(posterior_model.parameters(), lr=args.finetune_lr)
            
            # special optimizer
            no_decay = ['bias', 'LayerNorm.weight', 'norm.weight', '.g']
            optimizer_grouped_parameters = [
                    {
                        'params': [p for n, p in posterior_model.named_parameters() if not any(nd in n for nd in no_decay)],
                        'weight_decay': 0.,
                    },
                    {
                        'params': [p for n, p in posterior_model.named_parameters() if any(nd in n for nd in no_decay)],
                        'weight_decay': 0.0,
                    },
                ]
            posterior_model_optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.finetune_lr, betas=args.rtb_adam_betas)
            posterior_model_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    posterior_model_optimizer,
                    args.num_posterior_epochs
                )
            
            xs = test_function_x_tensor.clone().detach().to(device)
            xs = prior_model.normalizer.normalize(xs)
            next_obs_start = obs_dim + act_dim + 1
            next_obs_end = next_obs_start + obs_dim
            ys = test_function_y_tensor.clone().detach().to(device)
            y_weights = torch.softmax(ys, dim=0)
            
            posterior_model.train()
            
            # fine-tuning loop
            if num_posterior_epochs > 0:
                # Initialize wandb table for posterior training logs
                if args.wandb:
                    posterior_log_table = wandb.Table(
                        columns=["Epoch", "Training_Epoch", "Loss", "logZ", "OnPolicy_Reward"]
                    )
                s1 = 1
                
                for epoch in tqdm(range(num_posterior_epochs), dynamic_ncols=True):
                    if args.training_posterior == "both":
                        # toggle between on-policy and off-policy
                        s1 = (s1+1) % 2
                    elif args.training_posterior == "on":
                        s1 = 0
                    else: # off
                        s1 = 1
                    
                    # Gradient accumulation settings
                    accumulation_steps = args.accumulation_steps
                    micro_batch_size = args.ft_batch_size // accumulation_steps
                    posterior_model_optimizer.zero_grad()
                    
                    # Accumulate gradients over micro-batches
                    total_loss = 0.0
                    total_logZ = 0.0
                    # all_x_list = []
                    # all_y_list = []
                    
                    on_policy_rewards = []
                    for acc_step in range(accumulation_steps):
                        if s1 == 0:
                            # on-policy
                            if acc_step == 0:
                                print(f'On-policy training (gradient accumulation: {accumulation_steps} steps)')
                            # return normalized samples x
                            loss, logZ, x, logr = posterior_model.compute_loss(device, gfn_batch_size=micro_batch_size)
                            # Extract obs and next_obs from x for reward computation
                            x_unnormalized = prior_model.normalizer.unnormalize(x)
                            x_obs_unnormalized = x_unnormalized[:, :obs_dim]
                            x_next_obs_unnormalized = x_unnormalized[:, next_obs_start:next_obs_end]
                            # x_next_obs should be unnormalized as input of proxy_model_ens
                            if args.novelty_measure == 'rnd':
                                y = proxy_model_ens(x_next_obs_unnormalized).squeeze()
                            elif args.novelty_measure == 'curiosity':
                                x_act_unnormalized = x_unnormalized[:, obs_dim:obs_dim+act_dim]
                                y = proxy_model_ens(x_obs_unnormalized, x_next_obs_unnormalized, x_act_unnormalized).squeeze()
                            elif args.novelty_measure == 'eco':
                                # ECO reward computation
                                # According to paper: "takes the current observation o as input"
                                # Use current obs (obs at time t, before action was taken)
                                y = proxy_model_ens(x_obs_unnormalized).squeeze()
                            else:
                                raise ValueError(f'Invalid novelty measure: {args.novelty_measure}')
                            on_policy_rewards.append(y.detach().mean().item())
                        else:
                            # off-policy (reward prioritization)
                            if acc_step == 0:
                                print(f'Off-policy training (gradient accumulation: {accumulation_steps} steps)')
                            idx = torch.multinomial(y_weights.squeeze(), micro_batch_size, replacement=True)
                            # this is normalized samples x
                            x = xs[idx]
                            # [Optional] Add noise to x
                            # x += torch.randn_like(x) * 0.01
                            loss, logZ = posterior_model.compute_loss_with_sample(x, device)
                            # Extract obs and next_obs from x for reward computation
                            x_unnormalized = prior_model.normalizer.unnormalize(x)
                            x_obs_unnormalized = x_unnormalized[:, :obs_dim]
                            x_next_obs_unnormalized = x_unnormalized[:, next_obs_start:next_obs_end]
                            if args.novelty_measure == 'rnd':
                                y = proxy_model_ens(x_next_obs_unnormalized).squeeze()
                            elif args.novelty_measure == 'curiosity':
                                x_act_unnormalized = x_unnormalized[:, obs_dim:obs_dim+act_dim]
                                y = proxy_model_ens(x_obs_unnormalized, x_next_obs_unnormalized, x_act_unnormalized).squeeze()
                            elif args.novelty_measure == 'eco':
                                # ECO reward computation
                                # According to paper: "takes the current observation o as input"
                                # Use current obs (obs at time t, before action was taken)
                                y = proxy_model_ens(x_obs_unnormalized).squeeze()
                            else:
                                raise ValueError(f'Invalid novelty measure: {args.novelty_measure}')
                        
                        # Scale loss by accumulation_steps to maintain same effective learning rate
                        (loss / accumulation_steps).backward()
                        total_loss += loss.item()
                        if isinstance(logZ, torch.Tensor):
                            total_logZ += logZ.item()
                        else:
                            total_logZ += logZ
                        # all_x_list.append(x)
                        # all_y_list.append(y)
                    
                    # Update weights after accumulating all gradients
                    # posterior_model_optimizer.step()              
                    if args.ft_clip_grad > 0.0:
                        print(f'Clipping gradients during finetuning')
                        torch.nn.utils.clip_grad_norm_(posterior_model.parameters(), max_norm=args.ft_clip_grad)
                    posterior_model_optimizer.step()
                    if posterior_model_lr_scheduler is not None:
                        posterior_model_lr_scheduler.step()
                    
                    # Concatenate all samples
                    # x = torch.cat(all_x_list, dim=0)
                    # y = torch.cat(all_y_list, dim=0)
                    # 데이터 하나 당 평균 loss
                    loss = total_loss / accumulation_steps  # Average loss for logging
                    logZ = total_logZ / accumulation_steps  # Average logZ for logging
                    
                    # xs = torch.cat([xs, x], dim=0)
                    # ys = torch.cat([ys, y], dim=0)
                    
                    # y_weights = torch.softmax(ys, dim=0)
                    print(f'Epoch: {epoch+1}/{num_posterior_epochs} \tLoss: {loss:.9f}')
                    
                    # Add data to wandb table
                    if args.wandb:
                        loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
                        logZ_value = logZ.item() if isinstance(logZ, torch.Tensor) else logZ
                        if on_policy_rewards:
                            on_policy_reward_value = torch.tensor(on_policy_rewards).mean().item()
                            on_policy_reward_value = f"{on_policy_reward_value:.9f}"
                        else:
                            on_policy_reward_value = "NA"
                        posterior_log_table.add_data(
                            cur_epoch,
                            epoch + 1,
                            f"{loss_value:.9f}",
                            f"{logZ_value:.9f}",
                            on_policy_reward_value
                        )
                        print(f'On-policy reward (Mean): {on_policy_reward_value}')
                
                # Log table at the end of posterior training (with epoch-specific key to avoid overwriting)
                if args.wandb:
                    wandb.log({f"Posterior_Training_Log_Epoch_{cur_epoch}": posterior_log_table}, step=cur_epoch)
                    
            posterior_model.eval()
        
        
        
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
            posterior_model.eval()
                
            for _ in tqdm(range(eval_epochs)): #NOTE B * M**2 samples proposal.
                # Split into batches due to memory constraints
                # X_sample, logpf_pi, logpf_p = posterior_model.sample(bs=args.sample_batch_size * M, device=device)
                # if args.algorithm == 'Ours':
                # posterior_model.eval()
                X_sample = posterior_model.sample(bs=args.sample_batch_size, device=device, eval=True)
                # elif args.algorithm == 'PGRrnd' or args.algorithm == 'PGR':
                #     # prior_model.eval()
                #     # cond = torch.FloatTensor(cond_distri.sample_cond(args.sample_batch_size)).to(device)
                #     # # pdb.set_trace()
                #     # cond = prior_model.cond_normalizer.normalize(cond)
                #     # X_sample = prior_model.sample(bs=args.sample_batch_size, device=device, eval=True, cond=cond, cfg_scale=cfg_scale)
                #     prior_ema.ema_model.eval()
                #     cond = torch.FloatTensor(cond_distri.sample_cond(args.sample_batch_size)).to(device)
                #     cond = prior_ema.ema_model.cond_normalizer.normalize(cond)
                #     X_sample = prior_ema.ema_model.sample(bs=args.sample_batch_size, device=device, eval=True, cond=cond, cfg_scale=cfg_scale, ddim=args.ddim)
                # elif args.algorithm == 'SER':
                #     # prior_model.eval()
                #     # X_sample = prior_model.sample(bs=args.sample_batch_size, device=device, eval=True, cond=None, cfg_scale=None)
                #     prior_ema.ema_model.eval()
                #     X_sample = prior_ema.ema_model.sample(bs=args.sample_batch_size, device=device, eval=True, cond=None, cfg_scale=None, ddim=args.ddim)
                # else:
                #     raise ValueError(f'Invalid algorithm: {args.algorithm}')
                
                # local search is not used in our settings
                # if args.local_search and args.local_search_epochs > 0:
                #     break
                # else:
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

            # ---- Dynamic MSE logging (placed right after print_buffer_stats, as requested) ----
            print(f'Computing Dynamic MSE...')
            # Compute at most once per epoch to avoid double-counting if diffusion sampling happens multiple times.
            epoch_cur = t // steps_per_epoch
            # dyn_every = getattr(args, "dynamic_mse_every", 1)
            dyn_n = getattr(args, "dynamic_mse_samples", 5000)
            # do_dyn = (dyn_every is not None) and (dyn_every > 0) and (epoch_cur % dyn_every == 0)

            # if do_dyn and (epoch_cur != last_dynmse_epoch):
            dyn_res = compute_dynamic_mse_from_diffusion_buffer(
                agent.diffusion_buffer,
                gt_dyn_env,
                n_samples=dyn_n,
            )
            
            if dyn_res.get("ok", False):
                # logger.store(DynMSE=dyn_res["dyn_mse"])
                # logger.store(DynStateMSE=dyn_res["state_mse"])
                # logger.store(DynRewardMSE=dyn_res["reward_mse"])
                
                print('Starting to plot Dynamic MSE...')

                if args.wandb:
                    # Dynamic MSE boxplot
                    fig_dyn = plt.figure(figsize=(6, 4))
                    plt.boxplot(dyn_res["dyn_mse"], showfliers=False)
                    plt.title(f"Dynamic MSE (n_ok={dyn_res['n_ok']}/{dyn_res['n_total']})")
                    plt.ylabel("0.5*(state_mse + reward_mse)")
                    plt.tight_layout()
                    
                    # State MSE boxplot
                    fig_state = plt.figure(figsize=(6, 4))
                    plt.boxplot(dyn_res["state_mse"], showfliers=False)
                    plt.title(f"State MSE (n_ok={dyn_res['n_ok']}/{dyn_res['n_total']})")
                    plt.ylabel("mean((s'_true - s'_gt)^2)")
                    plt.tight_layout()
                    
                    # Reward MSE boxplot
                    fig_reward = plt.figure(figsize=(6, 4))
                    plt.boxplot(dyn_res["reward_mse"], showfliers=False)
                    plt.title(f"Reward MSE (n_ok={dyn_res['n_ok']}/{dyn_res['n_total']})")
                    plt.ylabel("(r_true - r_gt)^2")
                    plt.tight_layout()
                    
                    wandb.log(
                        {
                            "eval/DynMSE_boxplot": wandb.Image(fig_dyn),
                            "eval/DynStateMSE_boxplot": wandb.Image(fig_state),
                            "eval/DynRewardMSE_boxplot": wandb.Image(fig_reward),
                            "eval/DynMSE_mean": dyn_res["dyn_mse_mean"],
                            "eval/DynMSE_median": dyn_res["dyn_mse_median"],
                        },
                        step=epoch_cur,
                    )
                    plt.close(fig_dyn)
                    plt.close(fig_state)
                    plt.close(fig_reward)
            # else:
            #     logger.store(DynMSE=np.array([np.nan], dtype=np.float32))
            #     logger.store(DynStateMSE=np.array([np.nan], dtype=np.float32))
            #     logger.store(DynRewardMSE=np.array([np.nan], dtype=np.float32))

            # last_dynmse_epoch = epoch_cur

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
                
            # Evaluation of state entropy
            # # 첫 번째 에포크에서 StateEnt를 0으로 초기화하여 헤더에 포함
            # if args.state_ent:
            #     if epoch % 5 == 0 and epoch > 1:
            #         # obs_tensor, _, _, _, _ = agent.sample_real_data(batch_size=args.ent_eval_num)
            #         obs_tensor, _, _, _, _ = agent.sample_real_data_cpu(batch_size=args.ent_eval_num)
            #         intr_rew = compute_intr_reward(pbe, obs_tensor)
            #         logger.store(StateEnt=intr_rew)
            #         print(f'State Entropy: {intr_rew.mean():.4f}')
            #     else:
            #         # 헤더 등록을 위해 빈 값 저장 (실제 계산은 하지 않음)
            #         logger.store(StateEnt=0.0)
            #     logger.log_tabular('StateEnt', average_only=True)
            
            obs_tensor, _, _, _, _ = agent.sample_real_data_cpu(batch_size=4000)
            intr_rew = compute_intr_reward(pbe, obs_tensor)
            logger.store(StateEnt=intr_rew)
            logger.log_tabular('StateEnt', average_only=True)
            print(f'State Entropy: {intr_rew.mean():.4f}')
            
            # logger.log_tabular('DynMSE', with_min_and_max=True)
            # logger.log_tabular('DynStateMSE', with_min_and_max=True)
            # logger.log_tabular('DynRewardMSE', with_min_and_max=True)
            

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
    
    parser.add_argument('--knn_clip', type=float, default=0.0)
    parser.add_argument('--knn_k', type=int, default=12)
    parser.add_argument('--knn_avg', action='store_true', default=False) # default: True
    parser.add_argument('--knn_rms', action='store_true', default=False)
    parser.add_argument('--ent_eval_num', type=int, default=5000)
    
    parser.add_argument('--novelty_measure', type=str, default='curiosity')
    
    # new arguments
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--num_prior_epochs', type=int, default=100000)
    parser.add_argument('--num_posterior_epochs', type=int, default=50)
    parser.add_argument('--training_posterior', type=str, default='both') # 'both', 'on', 'off'
    
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=1000000)
    parser.add_argument('--sample_batch_size', type=int, default=100000)
    parser.add_argument('--prior_lr_scheduler', type=str, default='cosine')
    parser.add_argument('--rtb_lr_scheduler', type=str, default='cosine')
    parser.add_argument('--prior_adam_betas', type=tuple, default=(0.9, 0.99))
    parser.add_argument('--rtb_adam_betas', type=tuple, default=(0.9, 0.99))
    
    parser.add_argument('--prior_lr', type=float, default=3e-4)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--ft_clip_grad', type=float, default=1.0)
    parser.add_argument('--alpha_rtb', type=float, default=1.0)
    parser.add_argument('--cond_top_frac', type=float, default=0.25)
    
    
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--ft_batch_size', type=int, default=256)
    
    parser.add_argument('--inter_onpolicy', type=float, default=0.1)
    
    parser.add_argument('--ddim', action='store_true', default=False)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--clip_reward', type=float, default=0.95)
    
    parser.add_argument('--anneal', action='store_true', default=False)

    # Dynamic MSE logging (synthetic transition plausibility)
    parser.add_argument('--dynamic_mse_samples', type=int, default=5000)
    parser.add_argument('--dynamic_mse_every', type=int, default=1)
    
    args = parser.parse_args()
    
    args.results_folder = f'./{args.results_folder}/{args.results_folder}_{args.env}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    print(args.results_folder)
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    logger_kwargs = setup_logger_kwargs(args.env, args.log_dir)

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    # args를 한번에 넘기는게 좋음
    redq_sac(args.env, target_entropy='auto', logger_kwargs=logger_kwargs, args=args)
