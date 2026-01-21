import copy

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from torch.nn import functional as F

from synther.diffusion.norm import normalizer_factory
from redq.algos.core import ReplayBuffer
from synther.online.utils import make_inputs_from_replay_buffer

# new
import math
from einops import rearrange
import gin
from torch.distributions import Bernoulli

import pdb


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(
            self,
            dim: int,
            is_random: bool = False,
    ):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered
    
class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: str = "relu", layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        if layer_norm:
            # when we use layer norm,
            self.ln = nn.LayerNorm(dim_in)
        else:
            # current style
            self.ln = torch.nn.Identity()
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.activation(self.ln(x)))
    
    
# class ResidualMLP(nn.Module):
#     def __init__(
#             self,
#             input_dim: int,
#             width: int,
#             depth: int,
#             output_dim: int,
#             activation: str = "gelu",
#             layer_norm: bool = False,
#     ):
#         super().__init__()

#         self.network = nn.Sequential(
#             nn.Linear(input_dim, width),
#             *[ResidualBlock(width, width, activation, layer_norm) for _ in range(depth)],
#             nn.LayerNorm(width) if layer_norm else torch.nn.Identity(),
#         )

#         self.activation = getattr(F, activation)
#         self.final_linear = nn.Linear(width, output_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.final_linear(self.activation(self.network(x)))    

class ResidualMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            cond_dim: int,
            width: int,
            depth: int,
            output_dim: int,
            activation: str = "relu",
            layer_norm: bool = False,
    ):
        super().__init__()

        assert cond_dim is not None, "Residual MLP constructor requires cond_dim"
        self.x_proj = nn.Linear(input_dim, width)
        self.cond_proj = nn.Linear(cond_dim, width)

        self.network = nn.ModuleList(
            [ResidualBlock(width * 2, width * 2, activation, layer_norm) for _ in range(depth)]
        )

        self.activation = getattr(F, activation)
        self.final_linear = nn.Linear(2 * width, output_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.x_proj(x)
        cond = self.cond_proj(cond)
        x = torch.cat((x, cond), dim=-1)
        for layer in self.network:
            x = layer(x)
        return self.final_linear(self.activation(x))

# @gin.configurable
# class QFlowMLP(nn.Module):
#     def __init__(self, x_dim, hidden_dim=512, is_qflow=False, q_net=None, beta=None, dtype = torch.float32):
#         super(QFlowMLP, self).__init__()
#         self.is_qflow = is_qflow
#         self.q = q_net
#         self.beta = beta

#         # self.x_model = nn.Sequential(
#         #     nn.Linear(x_dim + 128, hidden_dim, dtype=dtype), nn.GELU(), nn.Linear(hidden_dim, hidden_dim, dtype=dtype), nn.GELU()
#         # )

#         # self.out_model = nn.Sequential(
#         #     nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
#         #     nn.LayerNorm(hidden_dim, dtype=dtype),
#         #     nn.GELU(),
#         #     nn.Linear(hidden_dim, x_dim, dtype=dtype),
#         # )

#         self.proj = nn.Linear(x_dim, hidden_dim)
#         self.residual_mlp = ResidualMLP(
#             input_dim=hidden_dim + 128,
#             width=hidden_dim,
#             depth=3,
#             output_dim=x_dim,
#             activation="gelu",
#             layer_norm=True,
#         )

#         self.means_scaling_model = nn.Sequential(
#             nn.Linear(128, hidden_dim // 2, dtype=dtype),
#             nn.LayerNorm(hidden_dim // 2, dtype=dtype),
#             nn.GELU(),
#             nn.Linear(hidden_dim // 2, hidden_dim // 2, dtype=dtype),
#             nn.LayerNorm(hidden_dim // 2, dtype=dtype),
#             nn.GELU(),
#             nn.Linear(hidden_dim // 2, x_dim, dtype=dtype),
#         )

#         self.harmonics = nn.Parameter(torch.arange(1, 64 + 1, dtype=dtype) * 2 * np.pi).requires_grad_(False)

#     def forward(self, x, t):
#         t_fourier1 = (t.unsqueeze(1) * self.harmonics).sin()
#         t_fourier2 = (t.unsqueeze(1) * self.harmonics).cos()
#         t_emb = torch.cat([t_fourier1, t_fourier2], 1)
#         # if not self.is_qflow:
#         #     x_emb = self.x_model(torch.cat([x, t_emb], 1))
#         # if self.is_qflow:
#         #     # with torch.no_grad():
#         #     x_emb = self.x_model(torch.cat([x, t_emb], 1))
#         #     with torch.enable_grad():
#         #         x.requires_grad_(True)
#         #         means_scaling = self.means_scaling_model(t_emb) * self.q.score(x, beta=self.beta)
#         #     return self.out_model(x_emb) + means_scaling
#         # return self.out_model(x_emb)
#         # x = self.proj(x) + t_emb
#         # problem: needs debugging
#         x = torch.cat([self.proj(x), t_emb], 1)
#         return self.residual_mlp(x)

@gin.configurable
class QFlowMLP(nn.Module):
    def __init__(
            self,
            d_in: int,
            dim_t: int = 256,
            mlp_width: int = 1024,
            num_layers: int = 6,
            learned_sinusoidal_cond: bool = False,
            random_fourier_features: bool = True,
            learned_sinusoidal_dim: int = 16,
            # gin referenced activation: mish
            activation: str = "mish",
            # activation: str = "gelu",
            layer_norm: bool = True,
            cond_dim: int = None,
            cfg_dropout: float = 0.25,
    ):
        super().__init__()
        self.residual_mlp = ResidualMLP(
            input_dim=d_in,
            # including both time and cond embeddings
            cond_dim=dim_t * 2,
            width=mlp_width,
            depth=num_layers,
            output_dim=d_in,
            # activation="gelu",
            activation=activation,
            # in pgr, originally false
            # layer_norm=True,
            layer_norm=layer_norm,
        )
        assert cond_dim is not None, "Conditional denoiser constructor requires cond_dim"

        # Conditional dropout
        self.cond_dropout = Bernoulli(probs=1 - cfg_dropout)

        # time embeddings
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            # activates
            sinu_pos_emb = SinusoidalPosEmb(dim_t)
            fourier_dim = dim_t

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, dim_t * 2),
            nn.Mish(),
            nn.Linear(dim_t * 2, dim_t),
        )

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, dim_t * 4),
            nn.Mish(),
            nn.Linear(dim_t * 4, dim_t)
        )
        

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            cond=None,
    ) -> torch.Tensor:
        
        t = self.time_mlp(timesteps)
        
        if cond is not None:
            
            # pdb.set_trace()
            c = self.cond_mlp(cond)
            
            # Do conditional dropout during training
            if self.training:
                mask = self.cond_dropout.sample(sample_shape=(c.shape[0], 1)).to(c.device)
                c = c * mask
        else:
            c = torch.zeros_like(t).to(t.device)

        t = torch.cat((c, t), dim=-1)

        return self.residual_mlp(x, t)
    
    

class DiffusionModel(nn.Module):
    def __init__(self, x_dim, diffusion_steps, inputs, skip_dims, disable_terminal_norm=False, schedule="linear", predict="epsilon", policy_net="mlp", hidden_dim=1024, dtype=torch.float32, cond_dim=1, cfg_dropout=0.25):
        super(DiffusionModel, self).__init__()
        # ================================
        # new things
        self.inputs = inputs
        self.skip_dims = skip_dims
        self.disable_terminal_norm = disable_terminal_norm
        self.event_dim = self.inputs.shape[1]
        if disable_terminal_norm:
            terminal_dim = self.event_dim - 1
            if terminal_dim not in skip_dims:
                self.skip_dims.append(terminal_dim)
        if skip_dims:
            print(f"Skipping normalization for dimensions {self.skip_dims}.")
        # normalizer of pgr 
        self.normalizer = normalizer_factory('minmax', inputs, skip_dims=self.skip_dims)
        # cond normalizer for conditional generation
        cond_inputs = torch.zeros((128, cond_dim)).float()
        self.cond_normalizer = normalizer_factory('minmax', cond_inputs, skip_dims=[])
        # ================================
        self.x_dim = x_dim
        self.diffusion_steps = diffusion_steps
        self.schedule = schedule
        self.dtype = dtype
        # QFlowMLP requires cond_dim, default to 1 if not provided
        if cond_dim is None:
            cond_dim = 1
        # this initialization is carefully designed w.r.t. PGR
        self.policy = QFlowMLP(d_in=x_dim, dim_t=256, mlp_width=hidden_dim, num_layers=6, cond_dim=cond_dim, cfg_dropout=cfg_dropout)
        self.predict = predict
        if self.schedule == "linear":
            beta1 = 0.02
            beta2 = 1e-4
            beta_t = (beta1 - beta2) * torch.arange(diffusion_steps + 1, 0, step=-1, dtype=dtype) / (
                diffusion_steps
            ) + beta2
        alpha_t = 1 - torch.flip(beta_t, dims=[0])
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)
        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
        self.register_buffer("beta_t", beta_t)
        self.register_buffer("alpha_t", torch.flip(alpha_t, dims=[0]))
        self.register_buffer("log_alpha_t", torch.flip(log_alpha_t, dims=[0]))
        self.register_buffer("alphabar_t", torch.flip(alphabar_t, dims=[0]))
        self.register_buffer("sqrtab", torch.flip(sqrtab, dims=[0]))
        self.register_buffer("oneover_sqrta", torch.flip(oneover_sqrta, dims=[0]))
        self.register_buffer("sqrtmab", torch.flip(sqrtmab, dims=[0]))
        self.register_buffer("mab_over_sqrtmab_inv", torch.flip(mab_over_sqrtmab_inv, dims=[0]))

    def forward(self, x, t, cond=None):
        # conditional generation
        epsilon = self.policy(x, t, cond=cond)
        return epsilon

    def score(self, x, t):
        t_idx = (t * self.diffusion_steps).long().unsqueeze(1)
        epsilon = self(x, t)
        if self.predict == "epsilon":
            score = -epsilon / self.sqrtmab[t_idx]
        elif self.predict == "x0":
            score = (self.sqrtab[t_idx] * epsilon - x) / (1 - self.alphabar_t[t_idx])
        return score

    def sample(self, bs, device, cond=None, cfg_scale=None, eval=False, ddim=True):
        """
        Sample from the diffusion model with optional classifier-free guidance.
        
        Args:
            bs: batch size
            device: device to run on
            cond: conditional input (optional)
            cfg_scale: classifier-free guidance scale. If 1.0, no guidance is applied.
                      If > 1.0, conditional generation is strengthened.
            eval: if True, disable dropout during sampling
        """
        with torch.no_grad():
            # if ddim:
            #     # TODO: Implement DDIM sampling
            #     pass
            if ddim:
                # --- DDIM hyperparams ---
                ddim_steps = 128        # 원하는 step 수 (1000 -> 128)
                eta = 0.0               # 0.0: deterministic DDIM, >0.0: stochastic DDIM
                # ------------------------

                x = torch.randn(bs, self.x_dim, dtype=self.dtype, device=device)

                T = self.diffusion_steps
                assert ddim_steps <= T

                # 1000-step index 중 128개를 고름 (i=0 noisy -> i=T-1 clean 방향 유지)
                idxs = torch.linspace(0, T - 1, steps=ddim_steps, device=device).long()

                for k in range(ddim_steps):
                    i = idxs[k].item()
                    j = idxs[k + 1].item() if k < ddim_steps - 1 else None

                    # training에서 썼던 time encoding과 동일하게: t = i / T
                    t = torch.full((bs,), float(i) / float(T), dtype=self.dtype, device=device)

                    # ----- epsilon prediction (CFG 지원) -----
                    if cond is not None and cfg_scale is not None:
                        eps_uncond = self(x, t, cond=None)
                        cond_input = cond.unsqueeze(-1) if cond.dim() == 1 else cond
                        eps_cond = self(x, t, cond=cond_input)
                        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
                    else:
                        eps = self(x, t, cond=cond)
                    # ----------------------------------------

                    # ----- DDIM uses alphabar directly -----
                    abar_i = self.alphabar_t[i]  # scalar tensor
                    sqrt_abar_i = torch.sqrt(abar_i)
                    sqrt_one_m_abar_i = torch.sqrt(1.0 - abar_i)

                    # x0 prediction: x0 = (x - sqrt(1-abar)*eps) / sqrt(abar)
                    x0 = (x - sqrt_one_m_abar_i * eps) / (sqrt_abar_i + 1e-12)

                    if j is None:
                        # 마지막은 x0를 반환하는 게 보통 가장 안정적
                        x = x0
                        break

                    abar_j = self.alphabar_t[j]
                    sqrt_abar_j = torch.sqrt(abar_j)

                    # eta로 sigma 조절 (eta=0이면 0)
                    if eta > 0.0:
                        sigma = eta * torch.sqrt(
                            torch.clamp(
                                (1.0 - abar_j) / (1.0 - abar_i + 1e-12) * (1.0 - abar_i / (abar_j + 1e-12)),
                                min=0.0,
                            )
                        )
                    else:
                        sigma = torch.zeros((), dtype=self.dtype, device=device)

                    # direction coefficient
                    c = torch.sqrt(torch.clamp(1.0 - abar_j - sigma * sigma, min=0.0))

                    noise = torch.randn_like(x, dtype=self.dtype, device=device) if eta > 0.0 else 0.0

                    # DDIM update: x_j
                    x = sqrt_abar_j * x0 + c * eps + sigma * noise

                # # DDIM 끝나면 x가 sample
                # return x

            else:
                x = torch.randn(bs, self.x_dim, dtype=self.dtype, device=device)
                t = torch.zeros((bs,), dtype=self.dtype, device=device)
                dt = 1 / self.diffusion_steps
                
                # assumes that time t, when denoising, start from 0 to 1
                # It means, when training, variance or betas are larger in t=1
                for i in range(self.diffusion_steps):
                    # Classifier-free guidance: combine unconditional and conditional scores
                    if cond is not None and cfg_scale is not None:
                        # print("Now conditioned generation")
                        # Compute unconditional score (cond=None)
                        epsilon_uncond = self(x, t, cond=None)
                            
                        cond_input = cond.unsqueeze(-1) if cond.dim() == 1 else cond
                        epsilon_cond = self(x, t, cond=cond_input)
                        # pdb.set_trace()
                        
                        # Combine scores using classifier-free guidance formula:
                        # epsilon = epsilon_uncond + cfg_scale * (epsilon_cond - epsilon_uncond)
                        # This can be rewritten as:
                        # epsilon = (1 + cfg_scale) * epsilon_cond - cfg_scale * epsilon_uncond
                        epsilon = epsilon_uncond + cfg_scale * (epsilon_cond - epsilon_uncond)
                    else:
                        # No guidance: use conditional or unconditional score directly
                        epsilon = self(x, t, cond=cond)
                    
                    # if it is the last step, no noise is added
                    if i < self.diffusion_steps - 1:
                        if self.predict == "epsilon":
                            x = self.oneover_sqrta[i] * (x - self.mab_over_sqrtmab_inv[i] * epsilon) + torch.sqrt(
                                self.beta_t[i]
                            ) * torch.randn_like(x, dtype=self.dtype, device=device)
                        elif self.predict == "x0":
                            x = (1 / torch.sqrt(self.alpha_t[i])) * (
                                (1 - (1 - self.alpha_t[i]) / (1 - self.alphabar_t[i])) * x
                                + ((1 - self.alpha_t[i]) / (1 - self.alphabar_t[i])) * self.sqrtab[i] * epsilon
                            ) + torch.sqrt(self.beta_t[i]) * torch.randn_like(x, dtype=self.dtype, device=device)
                    else:
                        if self.predict == "epsilon":
                            x = self.oneover_sqrta[i] * (x - self.mab_over_sqrtmab_inv[i] * epsilon)
                        elif self.predict == "x0":
                            raise ValueError("x0 prediction is not supported for the last step")
                        
                    t += dt
                
            
        return x

    # code for training prior or on-policy posterior training
    def compute_loss(self, x, cond=None):
        t_idx = torch.randint(0, self.diffusion_steps, (x.shape[0], 1)).to(x.device)
        # continuous time index 0 to 1
        t = t_idx.float().squeeze(1) / self.diffusion_steps
        epsilon = torch.randn_like(x, dtype=self.dtype).to(x.device)
        # t_idx가 0일때, sqrtmab[t_idx]가 제일 커야함. 이게 중요함.
        x_t = self.sqrtab[t_idx] * x + self.sqrtmab[t_idx] * epsilon
        # for debugging
        # t: training is aligned with sampling
        # t_idx_first = t_idx[0]
        # print(f'self.sqrtab[{t_idx_first}]: {self.sqrtmab[t_idx_first]}')
        # pdb.set_trace()
        # if cond is 1D tensor, we need to expand it to 2D tensor
        if cond is not None and cond.dim() == 1:
            cond = cond.unsqueeze(-1)
        epsilon_pred = self(x_t, t, cond=cond)
        if self.predict == "epsilon":
            w = torch.minimum(
                torch.tensor(5, dtype=self.dtype) / ((self.sqrtab[t_idx] / self.sqrtmab[t_idx]) ** 2), torch.tensor(1, dtype=self.dtype)
            )  # Min-SNR-gamma weights
            loss = (w * (epsilon - epsilon_pred) ** 2).mean()
        # we are not using x0
        elif self.predict == "x0":
            w = torch.minimum((self.sqrtab[t_idx] / self.sqrtmab[t_idx]) ** 2, torch.tensor(5, dtype=self.dtype))
            loss = (w * (x - epsilon_pred) ** 2).mean()
        return loss
    
    def update_normalizer(self, buffer: ReplayBuffer, device=None, model_terminals=False):
        data = make_inputs_from_replay_buffer(buffer, model_terminals=model_terminals)
        data = torch.from_numpy(data).float()
        # self.model.normalizer.reset(data)
        self.normalizer.reset(data)
        # self.ema.ema_model.normalizer.reset(data)
        if device:
            # self.model.normalizer.to(device)
            self.normalizer.to(device)
            # self.ema.ema_model.normalizer.to(device)
            
    def update_cond_normalizer(self, cond_distri, device=None):
        data = cond_distri.irews_buf[:, None]
        data = torch.from_numpy(data).float()
        self.cond_normalizer.reset(data)
        # self.ema.ema_model.cond_normalizer.reset(data)
        if device:
            self.cond_normalizer.to(device)
            # self.ema.ema_model.cond_normalizer.to(device)

class QFlow(nn.Module):
    def __init__(
        self,
        x_dim,
        diffusion_steps,
        schedule="linear",
        # predict="epsilon",
        q_net=None,
        bc_net=None,
        alpha=1.0,
        beta=1.0,
        dtype=torch.float64,
        square=True,
        pow_reward=1.0,
        obs_dim=None,
        act_dim=None,
        novelty_measure=None,
        agent=None,
        inter_onpolicy=0.1,
        reward_percentile=None,
    ):
        super(QFlow, self).__init__()
        self.x_dim = x_dim
        self.diffusion_steps = diffusion_steps
        self.schedule = schedule
        # self.predict = predict
        self.logZ = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.q_net = q_net
        self.bc_net = bc_net
        self.qflow = copy.deepcopy(bc_net.policy)
        # Convert qflow to the correct dtype to match QFlow's dtype
        self.qflow = self.qflow.to(dtype=dtype)
        for p in self.qflow.parameters():
            p.requires_grad_(True)
        # self.qflow.is_qflow = True  # This makes things more than 1.5x slower
        # This appears to be not used
        self.qflow.q = q_net
        self.qflow.beta = beta

        self.alpha = alpha
        self.beta = beta
        self.dtype = dtype
        
        # hyperparameters for reward function
        self.square = square
        self.pow_reward = pow_reward
        
        # Store dimensions for extracting next_obs from tensor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Store novelty measure
        self.novelty_measure = novelty_measure
        
        # on-policyness measure
        self.agent = agent
        self.inter_onpolicy = inter_onpolicy
        
        self.cond_normalizer = bc_net.cond_normalizer
        self.reward_percentile = reward_percentile
        print(f'Before replacing max: {self.cond_normalizer.max}')
        if self.reward_percentile is not None:
            new_max = self.reward_percentile
            self.cond_normalizer.max.copy_(new_max.reshape_as(self.cond_normalizer.max))
        print(f'After replacing max: {self.cond_normalizer.max}')

    def forward(self, x, t, cond=None):
        # problem: needs debugging
        q_epsilon = self.qflow(x, t, cond=cond)
        with torch.no_grad():
            bc_epsilon = self.bc_net(x, t, cond=cond).detach()
        return q_epsilon, bc_epsilon

    def sample(self, bs, device, extra=False, eval=False, cond=None, ddim=True):
        if eval:
            with torch.no_grad():
                # if ddim:
                #     # TODO: Implement DDIM sampling
                #     pass
                
                #     # return x
                if ddim:
                    # --- DDIM hyperparams ---
                    ddim_steps = 128
                    eta = 0.2
                    # ------------------------

                    normal_dist = torch.distributions.Normal(
                        torch.zeros((bs, self.x_dim), device=device, dtype=self.dtype),
                        torch.ones((bs, self.x_dim), device=device, dtype=self.dtype),
                    )
                    x = normal_dist.sample()

                    T = self.diffusion_steps
                    assert ddim_steps <= T

                    # 1000-step index 중 128개 선택 (0 noisy -> T-1 clean)
                    idxs = torch.linspace(0, T - 1, steps=ddim_steps, device=device).long()

                    extra_steps = 1
                    if extra:
                        extra_steps = 20

                    for k in range(ddim_steps):
                        i = idxs[k].item()
                        j = idxs[k + 1].item() if k < ddim_steps - 1 else None

                        # 학습과 동일한 t 정의
                        t = torch.full((bs,), float(i) / float(T), dtype=self.dtype, device=device)

                        # 원래 코드처럼 extra refinement를 유지하고 싶으면,
                        # 같은 t에서 여러 번 epsilon을 재평가할 수 있음.
                        # (DDIM은 원래 1회 업데이트가 일반적이지만, 너 코드는 extra 옵션이 있으니 보존)
                        for _ in range(extra_steps):
                            q_eps, bc_eps = self(x, t, cond=cond)
                            eps = (q_eps + bc_eps).detach()

                            abar_i = self.bc_net.alphabar_t[i]
                            sqrt_abar_i = torch.sqrt(abar_i)
                            sqrt_one_m_abar_i = torch.sqrt(1.0 - abar_i)

                            # x0 prediction
                            x0 = (x - sqrt_one_m_abar_i * eps) / (sqrt_abar_i + 1e-12)

                            if j is None:
                                x = x0
                                break

                            abar_j = self.bc_net.alphabar_t[j]
                            sqrt_abar_j = torch.sqrt(abar_j)

                            if eta > 0.0:
                                sigma = eta * torch.sqrt(
                                    torch.clamp(
                                        (1.0 - abar_j) / (1.0 - abar_i + 1e-12) * (1.0 - abar_i / (abar_j + 1e-12)),
                                        min=0.0,
                                    )
                                )
                            else:
                                sigma = torch.zeros((), dtype=self.dtype, device=device)

                            c = torch.sqrt(torch.clamp(1.0 - abar_j - sigma * sigma, min=0.0))
                            noise = torch.randn_like(x, dtype=self.dtype, device=device) if eta > 0.0 else 0.0

                            # DDIM update
                            x = sqrt_abar_j * x0 + c * eps + sigma * noise

                            # DDIM에서는 보통 extra loop를 돌 필요가 없어서,
                            # extra_steps>1을 쓸 거면 "같은 i->j 이동을 여러 번 반복"하게 되는데,
                            # 너 의도(약간 더 refine)라면 괜찮고, 아니라면 아래 break로 1회만 하자.
                            if not extra:
                                break

                        if j is None:
                            break

                    return x

                else:
                    normal_dist = torch.distributions.Normal(
                        torch.zeros((bs, self.x_dim), device=device, dtype=self.dtype), 
                        torch.ones((bs, self.x_dim), device=device, dtype=self.dtype)
                    )
                    x = normal_dist.sample()
                    t = torch.zeros((bs,), device=device, dtype=self.dtype)
                    dt = 1 / self.diffusion_steps

                    # logpf_pi = normal_dist.log_prob(x).sum(1)
                    # logpf_p = normal_dist.log_prob(x).sum(1)
                    # print(logpf_pi[:4])
                    extra_steps = 1
                    if extra:
                        extra_steps = 20
                    for i in range(self.diffusion_steps):
                        for j in range(extra_steps):
                            # problem: needs debugging
                            q_epsilon, bc_epsilon = self(x, t, cond=cond)

                            epsilon = q_epsilon + bc_epsilon
                            
                            # if it is the last step, no noise is added
                            if i < self.diffusion_steps - 1:
                                new_x = self.bc_net.oneover_sqrta[i] * (
                                    x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon.detach()
                                ) + torch.sqrt(self.bc_net.beta_t[i]) * torch.randn_like(x, dtype=self.dtype)
                            else:
                                new_x = self.bc_net.oneover_sqrta[i] * (
                                x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon.detach()
                                )
                                
                            # new_x = self.bc_net.oneover_sqrta[i] * (
                            #     x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon.detach()
                            # ) + torch.sqrt(self.bc_net.beta_t[i]) * torch.randn_like(x, dtype=self.dtype)
                            
                            
                            

                            # pf_pi_dist = torch.distributions.Normal(
                            #     self.bc_net.oneover_sqrta[i] * (x - self.bc_net.mab_over_sqrtmab_inv[i] * bc_epsilon),
                            #     torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(x, dtype=self.dtype),
                            # )
                            # logpf_pi += pf_pi_dist.log_prob(new_x).sum(1)

                            # pf_p_dist = torch.distributions.Normal(
                            #     self.bc_net.oneover_sqrta[i] * (x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon),
                            #     torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(new_x, dtype=self.dtype),
                            # )
                            # logpf_p += pf_p_dist.log_prob(new_x).sum(1)

                            x = new_x
                            if i < self.diffusion_steps - 1:
                                break
                        t = t + dt
                
                    return x
        
        # This is for training   
        else:
            normal_dist = torch.distributions.Normal(
                torch.zeros((bs, self.x_dim), device=device, dtype=self.dtype), 
                torch.ones((bs, self.x_dim), device=device, dtype=self.dtype)
                )
            x = normal_dist.sample()
            t = torch.zeros((bs,), device=device, dtype=self.dtype)
            dt = 1 / self.diffusion_steps

            logpf_pi = normal_dist.log_prob(x).sum(1)
            logpf_p = normal_dist.log_prob(x).sum(1)
            # print(logpf_pi[:4])
            extra_steps = 1
            if extra:
                extra_steps = 20
            for i in range(self.diffusion_steps):
                for j in range(extra_steps):
                    # problem: needs debugging
                    q_epsilon, bc_epsilon = self(x, t)

                    epsilon = q_epsilon + bc_epsilon
                    new_x = self.bc_net.oneover_sqrta[i] * (
                        x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon.detach()
                    ) + torch.sqrt(self.bc_net.beta_t[i]) * torch.randn_like(x, dtype=self.dtype)

                    pf_pi_dist = torch.distributions.Normal(
                        self.bc_net.oneover_sqrta[i] * (x - self.bc_net.mab_over_sqrtmab_inv[i] * bc_epsilon),
                        torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(x, dtype=self.dtype),
                    )
                    logpf_pi += pf_pi_dist.log_prob(new_x).sum(1)

                    pf_p_dist = torch.distributions.Normal(
                        self.bc_net.oneover_sqrta[i] * (x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon),
                        torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(new_x, dtype=self.dtype),
                    )
                    logpf_p += pf_p_dist.log_prob(new_x).sum(1)

                    x = new_x
                    if i < self.diffusion_steps - 1:
                        break
                t = t + dt
            
            # we don't need logpf_pi and logpf_p
            return x, logpf_pi, logpf_p
            # return x
    
    def back_and_forth(self, x, ratio, device):
        # Back
        t = torch.ones((x.shape[0],), device=device, dtype=self.dtype) * (1.0 - ratio)
        t_idx = (t * self.diffusion_steps).to(dtype=torch.long).unsqueeze(1)
        epsilon = torch.randn_like(x, dtype=self.dtype).to(x.device)
        x = self.bc_net.sqrtab[t_idx] * x + self.bc_net.sqrtmab[t_idx] * epsilon
        
        # Forth
        dt = 1 / self.diffusion_steps
        for i in range(int(self.diffusion_steps * (1.0 - ratio)), self.diffusion_steps):
            q_epsilon, bc_epsilon = self(x, t)
            epsilon = q_epsilon + bc_epsilon
            
            new_x = self.bc_net.oneover_sqrta[i] * (
                    x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon.detach()
                ) + torch.sqrt(self.bc_net.beta_t[i]) * torch.randn_like(x, dtype=self.dtype)

            x = new_x
            t = t + dt
        
        x_gen = x.clone().detach()
            
        # Compute the log probability
        t = torch.zeros((x.shape[0],), device=device, dtype=self.dtype)
        logr = self.posterior_log_reward(x)
        logpf_pi = torch.zeros((x.shape[0],), device=device, dtype=self.dtype)
        for i in range(self.diffusion_steps - 1, -1, -1):
            pb_dist = torch.distributions.Normal(
                torch.sqrt(self.bc_net.alpha_t[i]) * x,
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(x, dtype=self.dtype),
            )
            new_x = pb_dist.sample()
            
            q_epsilon, bc_epsilon = self(new_x, t + i * dt)
            epsilon = q_epsilon + bc_epsilon
            
            pf_pi_dist = torch.distributions.Normal(
                self.bc_net.oneover_sqrta[i] * (new_x - self.bc_net.mab_over_sqrtmab_inv[i] * bc_epsilon),
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(new_x, dtype=self.dtype),
            )
            logpf_pi += pf_pi_dist.log_prob(x).sum(1)
            
            x = new_x
        prior_dist = torch.distributions.Normal(torch.zeros_like(x, dtype=self.dtype), torch.ones_like(x, dtype=self.dtype))
        logpf_pi += prior_dist.log_prob(x).sum(1)
        return x_gen, logr, logpf_pi * self.aslpha
    
    def compute_likelihood(self, x, device):
        dt = 1 / self.diffusion_steps
        t = torch.zeros((x.shape[0],), device=device, dtype=self.dtype)
        logpf_pi = torch.zeros((x.shape[0],), device=device, dtype=self.dtype)
        for i in range(self.diffusion_steps - 1, -1, -1):
            pb_dist = torch.distributions.Normal(
                torch.sqrt(self.bc_net.alpha_t[i]) * x,
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(x, dtype=self.dtype),
            )
            new_x = pb_dist.sample()
            
            q_epsilon, bc_epsilon = self(new_x, t + i * dt)
            epsilon = q_epsilon + bc_epsilon
            
            pf_pi_dist = torch.distributions.Normal(
                self.bc_net.oneover_sqrta[i] * (new_x - self.bc_net.mab_over_sqrtmab_inv[i] * bc_epsilon),
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(new_x, dtype=self.dtype),
            )
            logpf_pi += pf_pi_dist.log_prob(x).sum(1)
            
            x = new_x
        prior_dist = torch.distributions.Normal(torch.zeros_like(x, dtype=self.dtype), torch.ones_like(x, dtype=self.dtype))
        logpf_pi += prior_dist.log_prob(x).sum(1)
        return logpf_pi
    
    def get_sigma(self):
        beta1 = 0.02
        beta2 = 1e-4
        beta_t = (beta1 - beta2) * torch.arange(self.diffusion_steps + 1, 0, step=-1, dtype=self.dtype) / (
            self.diffusion_steps
        ) + beta2
        alpha_t = 1 - torch.flip(beta_t, dims=[0]) #alphas
        log_alpha_t = torch.log(alpha_t) #log_alphas
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp() #alphas_cumprod
        sigmas = torch.sqrt((1 - alphabar_t) / alphabar_t) #sqrt((1-alphas_cumprod)/alphas_cumprod)
        log_sigmas = torch.log(sigmas)
        return sigmas[-1], sigmas[0], log_sigmas

    # @torch.no_grad()
    def compute_marginal_likelihood(self, x):
        v = torch.randint_like(x, 2) * 2 -1
        sigma_min, sigma_max, log_sigmas = self.get_sigma()
        log_sigmas = log_sigmas.to(x.device)
        model = self.bc_net
        dtype = self.dtype  # Capture dtype for use in nested class
        def sigma_to_t(sigma, log_sigmas, x):
            # get log sigma
            log_sigma = torch.log(sigma)

            # get distribution
            dists = log_sigma - log_sigmas[:, None]

            # get sigmas range
            low_idx = torch.cumsum((dists >= 0), dim=0).argmax(dim=0).clamp(max=log_sigmas.shape[0] - 2)
            high_idx = low_idx + 1

            low = log_sigmas[low_idx]
            high = log_sigmas[high_idx]

            # interpolate sigmas
            w = (low - log_sigma) / (low - high)
            w = torch.clamp(w, 0, 1)

            # transform interpolation to time range
            t = (1 - w) * low_idx + w * high_idx
            t = t / self.diffusion_steps
            t = torch.ones((x.shape[0],), device=x.device, dtype=dtype) * t
            return t
        
        class ODEfunc(torch.nn.Module):
            def __init__(self):
                super(ODEfunc, self).__init__()

                self.nfev = 0

            def forward(self, sigma, x):
                with torch.enable_grad():
                    x = x[0].requires_grad_()

                    x = x.to(dtype=dtype)
                    x = x / ((sigma**2 + 1) ** 0.5)

                    t = sigma_to_t(sigma, log_sigmas, x)

                    # predict the noise residual
                    noise_pred = model(x,t)

                    noise_pred = noise_pred.to(dtype=dtype)

                    d = noise_pred
                    
                    x_clone = x.clone().detach().requires_grad_()
                    d_clone = model(x_clone,t)
                    grad = torch.autograd.grad((d_clone * v).sum(), x_clone)[0].detach()
                    d_ll = (v * grad).flatten(1).sum(1)
                self.nfev += 1

                return d, d_ll
            
        x_min = x, x.new_zeros([x.shape[0]], dtype=dtype)
        t = x.new_tensor([sigma_min, sigma_max], dtype=dtype)
        ode_func = ODEfunc().cuda()

        method = "rk4"
        atol = 1e-5
        rtol = 1e-5
        step_size = abs(sigma_min - sigma_max) / 4
        sol = odeint(ode_func, x_min, t, atol=atol, rtol=rtol, method=method)
        
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = torch.distributions.Normal(torch.tensor(0.0, dtype=dtype, device=latent.device), sigma_max).log_prob(latent).flatten(1).sum(1)
        return ll_prior + delta_ll
        
    def posterior_log_reward(self, x):
        # Handle both dict and tensor inputs
        if isinstance(x, dict):
            # This will not be probably called
            # Dict input: extract next_obs_tensor
            next_obs = x['next_obs_tensor']
            next_act = x['next_act_tensor']
        elif isinstance(x, torch.Tensor):
            # This will probably be called
            # Tensor input: extract next_obs from concatenated tensor
            # x shape: [batch_size, obs_dim + act_dim + 1 + obs_dim]
            # next_obs starts at obs_dim + act_dim + 1
            if self.obs_dim is not None and self.act_dim is not None:
                next_obs_start = self.obs_dim + self.act_dim + 1
                next_obs_end = next_obs_start + self.obs_dim
                next_obs = x[:, next_obs_start:next_obs_end]
                obs = x[:, :self.obs_dim]
                act = x[:, self.obs_dim:self.obs_dim+self.act_dim]
            else:
                raise ValueError("obs_dim and act_dim must be provided to QFlow when using tensor inputs")
        else:
            raise TypeError(f"x must be dict or torch.Tensor, got {type(x)}")
        
        # According to the measure of novelty, we use different code to compute q_r
        # TODO
        if self.novelty_measure == 'curiosity':
            q_r = self.q_net(obs, next_obs, act).squeeze()
        elif self.novelty_measure == 'rnd':
            q_r = self.agent.compute_intrinsic_reward(next_obs).squeeze()
        elif self.novelty_measure == 'eco':
            # ECO uses current obs (according to paper: "takes the current observation o as input")
            q_r = self.agent.compute_eco_reward(obs).squeeze()
        else:
            raise ValueError(f'Invalid novelty measure: {self.novelty_measure}')
        
        if self.reward_percentile is not None:
            q_r = self.cond_normalizer.normalize(q_r)
            q_r = torch.clamp(q_r, -1.0, 1.0)
            q_r = (q_r + 1) / 2
        else:
            q_r = (self.cond_normalizer.normalize(q_r) + 1) / 2
        
        
        # Bound nice
        print(f'Check if q_r is bounded between 0 and 1: {q_r.min()}, {q_r.max()}')
        # print('Here is diffusion.py')
        
        # combine novelty reward with on-policyness reward
        if self.inter_onpolicy > 0:
            with torch.no_grad():
                # ranging from 0 to 1
                on_policy_reward = self.agent.compute_onpolicy_reward(obs, act)
                # on_policy_reward = on_policy_reward.pow(self.inter_onpolicy)
                on_policy_reward = np.power(on_policy_reward, self.inter_onpolicy)
                print(f'Check if on-policyness reward is bounded between 0 and 1: {on_policy_reward.min()}, {on_policy_reward.max()}')
            # convert numpy to tensor
            on_policy_reward = torch.tensor(on_policy_reward, device=q_r.device, dtype=q_r.dtype)
            # inter_onpolicy_tensor = torch.tensor(self.inter_onpolicy, device=q_r.device, dtype=q_r.dtype)
            # # 1.
            # q_r = q_r * (1 - inter_onpolicy_tensor) + inter_onpolicy_tensor * on_policy_reward
            # 2.
            q_r = q_r * on_policy_reward
        else:
            q_r = q_r
        # q_r = self.q_net(next_obs).squeeze()
        
        # print("q_r min/max:", q_r.min().item(), q_r.max().item())
        # print("q_r <= 0 count:", (q_r <= 0).sum().item())
        # print("q_r finite:", torch.isfinite(q_r).all().item())
        return q_r
    
    def combined_posterior_log_reward(self, x, alpha=0.1):
        # alpha is the weight of the on-policyness reward
        # TODO
        return 0.0

    def compute_loss_with_sample(self, x, device):
        bs = x.shape[0]
        # minlogvar, maxlogvar = -4, 4
        t = torch.zeros((bs,), device=device, dtype=self.dtype)
        dt = 1 / self.diffusion_steps

        logpf_pi = torch.zeros((bs,), device=device, dtype=self.dtype)
        logpf_p = torch.zeros((bs,), device=device, dtype=self.dtype)
        # I think we need to log here
        # Also, we need to unnormalize x here
        x_unnormalized = self.bc_net.normalizer.unnormalize(x)
        logr = self.posterior_log_reward(x_unnormalized).log()
        
        # going to noisy x
        for i in range(self.diffusion_steps - 1, -1, -1):
            pb_dist = torch.distributions.Normal(
                torch.sqrt(self.bc_net.alpha_t[i]) * x,
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(x, dtype=self.dtype),
            )
            new_x = pb_dist.sample()

            q_epsilon, bc_epsilon = self(new_x, t + i * dt)
            epsilon = q_epsilon + bc_epsilon

            pf_pi_dist = torch.distributions.Normal(
                self.bc_net.oneover_sqrta[i] * (new_x - self.bc_net.mab_over_sqrtmab_inv[i] * bc_epsilon),
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(new_x, dtype=self.dtype),
            )
            logpf_pi += pf_pi_dist.log_prob(x).sum(1)

            pf_p_dist = torch.distributions.Normal(
                self.bc_net.oneover_sqrta[i] * (new_x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon),
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(new_x, dtype=self.dtype),
            )
            logpf_p += pf_p_dist.log_prob(x).sum(1)

            x = new_x
        prior_dist = torch.distributions.Normal(torch.zeros_like(x, dtype=self.dtype), torch.ones_like(x, dtype=self.dtype))
        logpf_pi += prior_dist.log_prob(x).sum(1)
        logpf_p += prior_dist.log_prob(x).sum(1)
        loss = 0.5 * ((self.logZ + logpf_p * self.alpha - logr.detach() - logpf_pi * self.alpha) ** 2).mean()
        return loss, self.logZ

    def compute_loss(self, device, gfn_batch_size=512):
        # return normalized samples x
        x, logpf_pi, logpf_p = self.sample(bs=gfn_batch_size, device=device)
        # need to unnormalize x
        # self.bc_net.normalizer is the prior normalizer
        x_unnormalized = self.bc_net.normalizer.unnormalize(x)
        # I think we need to log here
        logr = self.posterior_log_reward(x_unnormalized)
        # logr = self.bc_net.cond_normalizer.unnormalize(logr*2-1)
        # print(f'logr: {logr}')
        logr = logr.log()
        
        # print("logr finite:", torch.isfinite(logr).all().item())
        # breakpoint()
        # pdb.set_trace()
        loss = 0.5 * ((self.logZ + logpf_p * self.alpha - logr.detach() - logpf_pi * self.alpha) ** 2).mean()
        return loss, self.logZ, x, logr