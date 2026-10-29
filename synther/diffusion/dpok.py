"""DPOK fine-tuning (Fan et al., 2023) as a drop-in replacement for RTB.

`DPOKFlow` keeps everything of `QFlow` (prior, posterior parameterization, novelty
reward, DDIM sampling) and only swaps the fine-tuning objective for the
KL-regularized policy gradient of DPOK, Eq. (8)/(9), Algorithm 1:

    loss = -reward_weight * (log r(x0) - b) * sum_t log p_theta(x_{t-1}|x_t)
           + kl_weight * sum_t KL( p_theta(x_{t-1}|x_t) || p_prior(x_{t-1}|x_t) )

Its minimizer is p_prior * exp((reward_weight/kl_weight) * log r), so with
reward_weight/kl_weight = 1/alpha_rtb it is exactly the RTB target p_prior * r^{1/alpha}.
Each optimizer step uses fresh on-policy trajectories, so the importance ratio of
DPOK Appendix A.6 is identically 1 and no PPO-style clipping is needed.
"""
import torch

from synther.diffusion.diffusion import QFlow


class DPOKFlow(QFlow):
    def __init__(self, *args, reward_weight=1.0, kl_weight=1.0, kl_form='exact',
                 baseline='mean', **kwargs):
        super().__init__(*args, **kwargs)
        # 'exact': closed-form Gaussian KL ||mu_post - mu_prior||^2 / (2 sigma_t^2).
        # 'eps_mse': the official code's surrogate mean((eps_post - eps_prior)^2).
        assert kl_form in ('exact', 'eps_mse'), kl_form
        assert baseline in ('mean', 'none'), baseline
        assert self.ddim and self.eta > 0.0, 'DPOKFlow needs the stochastic DDIM chain'
        self.reward_weight = reward_weight
        self.kl_weight = kl_weight
        self.kl_form = kl_form
        self.baseline = baseline

    def rollout(self, bs, device, cond=None):
        """On-policy DDIM rollout.

        Returns x0 (normalized), sum_t log p_post(x_{t-1}|x_t) and sum_t KL_t, both
        differentiable w.r.t. the fine-tuned net, plus sum_t log p_prior (no grad).
        """
        x = torch.randn((bs, self.x_dim), device=device, dtype=self.dtype)
        T = self.diffusion_steps
        idxs = torch.linspace(0, T - 1, steps=self.ddim_steps, device=device).long()
        logp = torch.zeros((bs,), device=device, dtype=self.dtype)
        logp_prior = torch.zeros((bs,), device=device, dtype=self.dtype)
        kl = torch.zeros((bs,), device=device, dtype=self.dtype)
        for k in range(self.ddim_steps):
            i = idxs[k].item()
            j = idxs[k + 1].item() if k < self.ddim_steps - 1 else None
            t = torch.full((bs,), float(i) / float(T), dtype=self.dtype, device=device)
            eps, bc_eps = self(x, t, cond=cond)

            abar_i = self.bc_net.alphabar_t[i]
            sqrt_abar_i = torch.sqrt(abar_i)
            sqrt_one_m_abar_i = torch.sqrt(1.0 - abar_i)
            x0 = (x - sqrt_one_m_abar_i * eps) / (sqrt_abar_i + 1e-12)
            if j is None:
                x = x0.detach()
                break
            x0_bc = (x - sqrt_one_m_abar_i * bc_eps) / (sqrt_abar_i + 1e-12)

            abar_j = self.bc_net.alphabar_t[j]
            sqrt_abar_j = torch.sqrt(abar_j)
            sigma = self.eta * torch.sqrt(torch.clamp(
                (1.0 - abar_j) / (1.0 - abar_i + 1e-12) * (1.0 - abar_i / (abar_j + 1e-12)), min=0.0))
            c = torch.sqrt(torch.clamp(1.0 - abar_j - sigma * sigma, min=0.0))
            mu_post = sqrt_abar_j * x0 + c * eps
            mu_prior = sqrt_abar_j * x0_bc + c * bc_eps
            scale = sigma * torch.ones_like(x)

            # The trajectory is a fixed sample; only the mean carries gradient.
            new_x = (mu_post + sigma * torch.randn_like(x)).detach()
            logp = logp + torch.distributions.Normal(mu_post, scale).log_prob(new_x).sum(1)
            logp_prior = logp_prior + torch.distributions.Normal(mu_prior, scale).log_prob(new_x).sum(1)
            if self.kl_form == 'exact':
                kl = kl + ((mu_post - mu_prior) ** 2).sum(1) / (2.0 * sigma ** 2)
            else:
                kl = kl + ((eps - bc_eps) ** 2).mean(1)
            x = new_x
        return x, logp, kl, logp_prior.detach()

    def compute_loss(self, device, gfn_batch_size=512):
        """Same signature as QFlow.compute_loss; the second return is a stats dict, not logZ."""
        x, logp, kl, logp_prior = self.rollout(gfn_batch_size, device)
        x_unnormalized = self.bc_net.normalizer.unnormalize(x)
        logr = self.posterior_log_reward(x_unnormalized).clamp_min(1e-6).log().detach()
        adv = logr - logr.mean() if self.baseline == 'mean' else logr
        pg = -(adv * logp).mean()
        loss = self.reward_weight * pg + self.kl_weight * kl.mean()
        stats = {
            'pg': pg.item(),
            'kl': kl.mean().item(),
            # Monte-Carlo trajectory KL, for checking the closed-form term.
            'kl_mc': (logp.detach() - logp_prior).mean().item(),
            'logr': logr.mean().item(),
        }
        return loss, stats, x, logr
