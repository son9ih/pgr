import numpy as np
import gym



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

    gt_env.reset()

    next_obs_gt = np.zeros_like(obs2, dtype=np.float64)
    rew_gt = np.zeros_like(rews, dtype=np.float64)
    ok_mask = np.zeros((obs1.shape[0],), dtype=bool)

    unwrapped = getattr(gt_env, "unwrapped", gt_env)

    for i in range(obs1.shape[0]):
        state = obs1[i]
        action = acts[i]

        reset_ok = False
        try:
            unwrapped.reset(state=state)
            reset_ok = True
        except TypeError:
            reset_ok = False
        except Exception:
            reset_ok = False
            
        if not reset_ok:
            if not _mujoco_set_state_from_obs(gt_env, state):
                continue

        try:
            o2_pred, r_pred, d_pred, _ = gt_env.step(action)
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
