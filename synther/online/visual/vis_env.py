import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

import gym
from gym import spaces

from dm_env import specs as dm_specs
from dm_env import StepType
from dm_control import suite
from dm_control.suite.wrappers import pixels, action_scale


class Encoder(nn.Module):
    """DrQ-v2 style visual encoder (copied structurally from v-d4rl.drqbc.drqv2.Encoder)."""

    def __init__(self, obs_shape):
        super().__init__()
        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, C, H, W), uint8 or float
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Trunk(nn.Module):
    """MLP trunk used by both actor and critic (matches Actor.trunk / Critic.trunk)."""

    def __init__(self, repr_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisualEncoder(nn.Module):
    """
    Wrapper that loads pretrained encoder + actor/critic trunks exported by
    v-d4rl/drqbc/extract_visual_params.py.

    It exposes an encode(obs) -> (actor_state, critic_state) API.
    All parameters are frozen (no grad).
    """

    def __init__(self, obs_shape, env_key: str, device: torch.device):
        """
        Args:
            obs_shape: tuple like (C, H, W) from dm_control observation_spec.
            env_key: 'cheetah_run', 'walker_walk', ... (directory name under visual/encoder).
            device: torch device.
        """
        super().__init__()
        self.device = device

        base_dir = Path(__file__).resolve().parent / "encoder" / env_key
        enc_path = base_dir / "agent_encoder.pt"
        act_trunk_path = base_dir / "agent_actor_trunk.pt"
        cri_trunk_path = base_dir / "agent_critic_trunk.pt"

        if not enc_path.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {enc_path}")
        if not act_trunk_path.exists():
            raise FileNotFoundError(f"Actor trunk checkpoint not found: {act_trunk_path}")
        if not cri_trunk_path.exists():
            raise FileNotFoundError(f"Critic trunk checkpoint not found: {cri_trunk_path}")

        # Build modules
        self.encoder = Encoder(obs_shape).to(device)

        # Infer feature_dim from saved actor trunk weights
        actor_trunk_sd = torch.load(act_trunk_path, map_location=device)
        # Find first weight tensor (Linear weight)
        first_w_key = None
        for k, v in actor_trunk_sd.items():
            if k.endswith("weight") and v.ndim == 2:
                first_w_key = k
                break
        if first_w_key is None:
            raise RuntimeError("Could not infer feature_dim from actor trunk state_dict")
        feature_dim = actor_trunk_sd[first_w_key].shape[0]

        self.actor_trunk = Trunk(self.encoder.repr_dim, feature_dim).to(device)

        critic_trunk_sd = torch.load(cri_trunk_path, map_location=device)
        self.critic_trunk = Trunk(self.encoder.repr_dim, feature_dim).to(device)

        # Load weights
        encoder_sd = torch.load(enc_path, map_location=device)
        self.encoder.load_state_dict(encoder_sd, strict=True)
        self.actor_trunk.load_state_dict(actor_trunk_sd, strict=True)
        self.critic_trunk.load_state_dict(critic_trunk_sd, strict=True)

        # Freeze parameters
        for m in (self.encoder, self.actor_trunk, self.critic_trunk):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        self.actor_dim = feature_dim
        self.critic_dim = feature_dim

    @torch.no_grad()
    def encode(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            obs: numpy array (C, H, W) or (B, C, H, W) uint8.

        Returns:
            actor_state: (feature_dim,) or (B, feature_dim)
            critic_state: (feature_dim,) or (B, feature_dim)
        """
        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)

        h = self.encoder(obs_t)
        actor_state = self.actor_trunk(h)
        critic_state = self.critic_trunk(h)

        actor_np = actor_state.cpu().numpy()
        critic_np = critic_state.cpu().numpy()

        if actor_np.shape[0] == 1:
            actor_np = actor_np[0]
            critic_np = critic_np[0]

        return actor_np.astype(np.float32), critic_np.astype(np.float32)


class ExtendedTimeStep(tuple):
    """Lightweight container matching v-d4rl/drqbc.dmc.ExtendedTimeStep."""

    __slots__ = ()

    def __new__(cls, step_type, reward, discount, observation, action):
        return tuple.__new__(cls, (step_type, reward, discount, observation, action))

    @property
    def step_type(self):
        return self[0]

    @property
    def reward(self):
        return self[1]

    @property
    def discount(self):
        return self[2]

    @property
    def observation(self):
        return self[3]

    @property
    def action(self):
        return self[4]

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST


class ActionRepeatWrapper:
    def __init__(self, env, num_repeats: int):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        time_step = None
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break
        return time_step._replace(reward=reward, discount=discount)

    def reset(self):
        return self._env.reset()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper:
    def __init__(self, env, num_frames: int, pixels_key: str = "pixels"):
        from collections import deque

        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim if present
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]

        c, h, w = pixels_shape[2], pixels_shape[0], pixels_shape[1]
        self._obs_spec = dm_specs.BoundedArray(
            shape=(c * num_frames, h, w),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def _extract_pixels(self, time_step):
        pix = time_step.observation[self._pixels_key]
        if len(pix.shape) == 4:
            pix = pix[0]
        # (H, W, C) -> (C, H, W)
        return pix.transpose(2, 0, 1).copy()

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        pixels_np = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels_np)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels_np = self._extract_pixels(time_step)
        self._frames.append(pixels_np)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper:
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = dm_specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper:
    def __init__(self, env):
        self._env = env

    def reset(self):
        ts = self._env.reset()
        return self._augment(ts)

    def step(self, action):
        ts = self._env.step(action)
        return self._augment(ts, action)

    def _augment(self, ts, action=None):
        if action is None:
            a_spec = self.action_spec()
            action = np.zeros(a_spec.shape, dtype=a_spec.dtype)
        return ExtendedTimeStep(
            step_type=ts.step_type,
            reward=ts.reward or 0.0,
            discount=ts.discount or 1.0,
            observation=ts.observation,
            action=action,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_dmc_pixel_env(name: str, frame_stack: int, action_repeat: int, seed: int):
    """
    Minimal DMControl pixel env loader for standard control suite tasks.
    name: e.g. 'cheetah_run', 'walker_walk'
    """
    pixel_hw = 84
    if "offline" in name:
        # Not expected in this script, but keep parity with drqbc.
        name = "_".join(name.split("_")[1:3])
    domain, task = name.split("_", 1)

    env = suite.load(
        domain,
        task,
        task_kwargs={"random": seed},
        visualize_reward=False,
    )
    pixels_key = "pixels"

    # Action wrappers and pixel rendering
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    camera_id = {"quadruped": 2}.get(domain, 0)
    render_kwargs = dict(height=pixel_hw, width=pixel_hw, camera_id=camera_id)
    env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)

    # Frame stack and extended time step
    env = FrameStackWrapper(env, frame_stack, pixels_key=pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env


class DMControlVisualGymEnv(gym.Env):
    """
    Gym-style wrapper around DMControl pixel env with fixed, frozen DrQv2 visual encoder.

    Observations are concatenated (actor_state, critic_state) vectors.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        task_name: str,
        env_key_for_encoder: str,
        device: torch.device,
        frame_stack: int = 3,
        action_repeat: int = 2,
        seed: int = 0,
        max_episode_steps: int = 1000,
    ):
        super().__init__()
        self._dmc_env = make_dmc_pixel_env(
            task_name, frame_stack=frame_stack, action_repeat=action_repeat, seed=seed
        )
        self._device = device
        self._max_episode_steps = max_episode_steps

        obs_spec = self._dmc_env.observation_spec()
        obs_shape = obs_spec.shape  # (C, H, W)

        # Build visual encoder
        self._encoder = VisualEncoder(obs_shape, env_key=env_key_for_encoder, device=device)

        # Infer obs_dim by encoding a dummy observation
        dummy_obs = np.zeros(obs_shape, dtype=np.uint8)
        actor_state, critic_state = self._encoder.encode(dummy_obs)
        self._actor_dim = int(actor_state.shape[-1])
        self._critic_dim = int(critic_state.shape[-1])
        obs_dim = self._actor_dim + self._critic_dim

        # Expose for agents
        self.actor_dim = self._actor_dim
        self.critic_dim = self._critic_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        act_spec = self._dmc_env.action_spec()
        self.action_space = spaces.Box(
            low=act_spec.minimum,
            high=act_spec.maximum,
            shape=act_spec.shape,
            dtype=np.float32,
        )

        # Gym-style spec with max_episode_steps so get_time_limit() works
        class _Spec:
            def __init__(self, max_episode_steps):
                self.max_episode_steps = max_episode_steps

        self.spec = _Spec(max_episode_steps)
        self._episode_steps = 0

    def _encode_obs(self, obs_raw: np.ndarray) -> np.ndarray:
        actor_state, critic_state = self._encoder.encode(obs_raw)
        obs_vec = np.concatenate([actor_state, critic_state], axis=-1).astype(np.float32)
        return obs_vec

    def reset(self):
        self._episode_steps = 0
        ts = self._dmc_env.reset()
        obs_raw = ts.observation  # (C, H, W)
        obs_vec = self._encode_obs(obs_raw)
        return obs_vec

    def step(self, action):
        self._episode_steps += 1
        ts = self._dmc_env.step(action)
        obs_raw = ts.observation
        obs_vec = self._encode_obs(obs_raw)
        reward = float(ts.reward)
        done = bool(ts.last() or (self._episode_steps >= self._max_episode_steps))
        info = {}
        return obs_vec, reward, done, info

    def seed(self, seed=None):
        # DMControl env is already seeded via make_dmc_pixel_env.
        pass

    def render(self, mode="human"):
        # Rendering is handled via DMControl viewer if needed; not exposed here.
        return None


