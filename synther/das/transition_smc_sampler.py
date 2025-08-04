"""
SMC-based sampling for transition data generation.
Adapted from DAS pipeline_using_smc for transition data instead of images.
"""

import math
import torch
import numpy as np
from synther.das.smc_utils import compute_ess_from_log_w, normalize_log_weights, resampling_function, normalize_weights, adaptive_tempering
from typing import Callable, Optional, Union
from tqdm import tqdm

def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

@torch.no_grad()
def sample_with_smc(
    diffusion_model,
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int = 16,
    num_sample_steps: Optional[int] = None,
    clamp: bool = True,
    cond=None,
    cfg_scale: float = 1.0,
    # SMC parameters
    num_particles: int = 4,
    batch_p: int = 1, # number of particles to run parallely
    resample_strategy: str = "ssp",
    ess_threshold: float = 0.5,
    tempering: str = "schedule",
    tempering_schedule: Union[float, int, str] = "exp",
    tempering_gamma: float = 1.,
    tempering_start: float = 0.,
    kl_coeff: float = 1.,
    verbose: bool = False,
    device: str = "cpu"
):
    """
    SMC-based sampling for transition data generation using a diffusion model.
    
    Args:
        diffusion_model: The trained diffusion model (ElucidatedDiffusion)
        reward_fn: Function that takes transition data and returns reward (curiosity)
        batch_size: Number of samples to generate per prompt
        num_sample_steps: Number of diffusion sampling steps
        clamp: Whether to clamp the outputs
        cond: Conditional input
        cfg_scale: Classifier-free guidance scale
        num_particles: Number of SMC particles
        batch_p: Number of particles to process in parallel
        resample_strategy: SMC resampling strategy
        ess_threshold: Effective sample size threshold for resampling
        tempering: Tempering strategy
        tempering_schedule: Tempering schedule
        tempering_gamma: Tempering gamma parameter
        tempering_start: When to start tempering (as fraction of total steps)
        kl_coeff: KL divergence coefficient
        verbose: Whether to print debug information
        device: Device to run on
    
    Returns:
        tuple: (samples, log_weights, additional_info)
    """
    
    # Check inputs
    assert num_particles >= batch_p, "num_particles should be greater than or equal to batch_p"
    
    # Get the sample schedule from diffusion model
    num_sample_steps = num_sample_steps or diffusion_model.num_sample_steps
    sigmas = diffusion_model.sample_schedule(num_sample_steps)
    gammas = torch.where(
        (sigmas >= diffusion_model.S_tmin) & (sigmas <= diffusion_model.S_tmax),
        min(diffusion_model.S_churn / num_sample_steps, math.sqrt(2) - 1),
        0.
    )
    
    sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))
    
    # Prepare initial samples (proposal distribution)
    shape = (batch_size * num_particles, *diffusion_model.event_shape)
    init_sigma = sigmas[0]
    prop_samples = init_sigma * torch.randn(shape, device=device)
    
    if cond is not None:
        cond = torch.from_numpy(cond).float().to(device)
        cond = diffusion_model.cond_normalizer.normalize(cond)
    
    # Initialize SMC variables
    rewards = torch.zeros(prop_samples.shape[0], device=device)
    log_twist_func = torch.zeros(prop_samples.shape[0], device=device)
    log_twist_func_prev = torch.zeros(prop_samples.shape[0], device=device)
    log_Z = torch.zeros(batch_size, device=device)
    log_w = torch.zeros(prop_samples.shape[0], device=device)
    log_prob_diffusion = torch.zeros(prop_samples.shape[0], device=device)
    log_prob_proposal = torch.zeros(prop_samples.shape[0], device=device)
    
    resample_fn = resampling_function(resample_strategy=resample_strategy, ess_threshold=ess_threshold)
    all_samples = []
    all_log_w = []
    all_resample_indices = []
    ess_trace = []
    scale_factor_trace = []
    rewards_trace = []
    
    kl_coeff = torch.tensor(kl_coeff, device=device).to(torch.float32)
    lookforward_fn = lambda r: r / kl_coeff
    
    start = int(len(sigmas_and_gammas) * tempering_start)
    scale_factor = torch.zeros(batch_size, device=device)
    min_scale_next = torch.zeros(batch_size, device=device)
    
    def _calc_guidance(samples, i):
        """Calculate reward-based guidance for SMC"""
        if i >= start:
            with torch.enable_grad():
                total_rewards = torch.zeros(samples.shape[0], device=device)
                total_log_twist_func = torch.zeros(samples.shape[0], device=device)
                total_approx_guidance = torch.zeros_like(samples, device=device)
                
                for idx in range(math.ceil(num_particles / batch_p)): 
                    tmp_samples = samples[batch_p*idx : batch_p*(idx+1)].detach().to(torch.float32).requires_grad_(True)
                    
                    # Denormalize and calculate rewards
                    denormalized_samples = diffusion_model.normalizer.unnormalize(tmp_samples)
                    tmp_rewards = reward_fn(denormalized_samples).to(torch.float32)
                    tmp_log_twist_func = lookforward_fn(tmp_rewards).to(torch.float32)
                    
                    # Calculate approximate guidance
                    if tmp_log_twist_func.requires_grad:
                        tmp_approx_guidance = torch.autograd.grad(
                            outputs=tmp_log_twist_func.sum(), 
                            inputs=tmp_samples, 
                            create_graph=False, 
                            retain_graph=False
                        )[0].detach()
                    else:
                        tmp_approx_guidance = torch.zeros_like(tmp_samples)
                    
                    total_rewards[batch_p*idx : batch_p*(idx+1)] = tmp_rewards.detach()
                    total_log_twist_func[batch_p*idx : batch_p*(idx+1)] = tmp_log_twist_func.detach()
                    total_approx_guidance[batch_p*idx : batch_p*(idx+1)] = tmp_approx_guidance
                
                # Handle NaN values
                if torch.isnan(total_log_twist_func).any():
                    if verbose:
                        print("NaN in log twist func, changing it to 0")
                    total_log_twist_func = torch.nan_to_num(total_log_twist_func)
                if torch.isnan(total_approx_guidance).any():
                    if verbose:
                        print("NaN in approx guidance, changing it to 0")
                    total_approx_guidance = torch.nan_to_num(total_approx_guidance)
                
                return total_rewards, total_log_twist_func, total_approx_guidance
        else:
            # Before tempering starts, no guidance
            return torch.zeros(samples.shape[0], device=device), torch.zeros(samples.shape[0], device=device), torch.zeros_like(samples, device=device)
    
    # SMC sampling loop
    with tqdm(total=len(sigmas_and_gammas), desc='SMC sampling time step', disable=not verbose) as pbar:
        for i, (sigma, sigma_next, gamma) in enumerate(sigmas_and_gammas):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))
            
            samples = prop_samples.clone()
            log_twist_func_prev = log_twist_func.clone()
            
            # Calculate guidance
            rewards, log_twist_func, approx_guidance = _calc_guidance(samples, i)
            rewards_trace.append(rewards.view(-1, num_particles).max(dim=1)[0].cpu())
            
            if i >= start:
                # Temperature selection
                if isinstance(tempering_schedule, float) or isinstance(tempering_schedule, int):
                    min_scale = torch.tensor([min((tempering_gamma * (i - start))**tempering_schedule, 1.)]*batch_size, device=device)
                    min_scale_next = torch.tensor([min((tempering_gamma * (i + 1 - start))**tempering_schedule, 1.)]*batch_size, device=device)
                elif tempering_schedule == "exp":
                    min_scale = torch.tensor([min((1 + tempering_gamma) ** (i - start) - 1, 1.)]*batch_size, device=device)
                    min_scale_next = torch.tensor([min((1 + tempering_gamma) ** (i + 1 - start) - 1, 1.)]*batch_size, device=device)
                elif tempering_schedule == "adaptive":
                    min_scale = scale_factor.clone()
                else:
                    min_scale = torch.tensor([1.]*batch_size, device=device)
                    min_scale_next = torch.tensor([1.]*batch_size, device=device)
                
                if tempering == "adaptive" and i > 0 and (min_scale < 1.).any():
                    scale_factor = adaptive_tempering(
                        log_w.view(-1, num_particles), 
                        log_prob_diffusion.view(-1, num_particles), 
                        log_twist_func.view(-1, num_particles), 
                        log_prob_proposal.view(-1, num_particles), 
                        log_twist_func_prev.view(-1, num_particles), 
                        min_scale=min_scale, 
                        ess_threshold=ess_threshold
                    )
                    min_scale_next = scale_factor.clone()
                elif tempering == "schedule":
                    scale_factor = min_scale
                else:
                    scale_factor = torch.ones(batch_size, device=device)
                
                scale_factor_trace.append(scale_factor.cpu())
                
                if verbose:
                    print(f"Step {i}, scale factor (lambda_t): {scale_factor}")
                
                # Apply tempering
                log_twist_func *= scale_factor.repeat_interleave(num_particles, dim=0)
                approx_guidance *= min_scale_next.repeat_interleave(num_particles, dim=0).view([-1] + [1]*(approx_guidance.dim()-1))
                
                # Weight calculation and resampling
                incremental_log_w = log_prob_diffusion + log_twist_func - log_prob_proposal - log_twist_func_prev
                log_w += incremental_log_w.detach()
                log_Z += torch.logsumexp(log_w, dim=-1)
                
                ess = [compute_ess_from_log_w(log_w_prompt).item() for log_w_prompt in log_w.view(-1, num_particles)]
                all_log_w.append(log_w.clone())
                ess_trace.append(torch.tensor(ess).cpu())
                
                # Resample
                resample_indices, is_resampled, log_w = resample_fn(log_w.view(-1, num_particles))
                log_w = log_w.view(-1)
                all_resample_indices.append(resample_indices)
                
                if verbose:
                    print(f"Step {i}, ESS: {ess}")
                    print(f"Step {i}, resampled: {is_resampled}")
                
                # Update variables based on resampling
                samples = samples.view(-1, num_particles, *samples.shape[1:])[torch.arange(samples.size(0)//num_particles).unsqueeze(1), resample_indices].view(-1, *samples.shape[1:])
                approx_guidance = approx_guidance.view(-1, num_particles, *approx_guidance.shape[1:])[torch.arange(approx_guidance.size(0)//num_particles).unsqueeze(1), resample_indices].view(-1, *approx_guidance.shape[1:])
            
            all_samples.append(samples.cpu())
            
            # Diffusion step with guidance
            eps = diffusion_model.S_noise * torch.randn(samples.shape, device=device)
            sigma_hat = sigma + gamma * sigma
            samples_hat = samples + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps
            
            # Score function evaluation
            cond_denoised_over_sigma = diffusion_model.score_fn(samples_hat, sigma_hat, clamp=clamp, cond=cond)
            uncond_denoised_over_sigma = diffusion_model.score_fn(samples_hat, sigma_hat, clamp=clamp, cond=None)
            denoised_over_sigma = uncond_denoised_over_sigma + cfg_scale * (cond_denoised_over_sigma - uncond_denoised_over_sigma)
            
            # Proposal step
            prev_sample_mean = samples_hat + (sigma_next - sigma_hat) * denoised_over_sigma
            
            # Calculate variance for this step
            if sigma_next > 0:
                variance = (sigma_next / sigma_hat) ** 2 * (sigma_hat ** 2 - sigma_next ** 2)
                variance = _left_broadcast(torch.tensor(variance, device=device), samples.shape)
                std_dev_t = variance.sqrt()
                
                # Add guidance to proposal
                if i >= start:
                    prop_samples = prev_sample_mean + variance * approx_guidance
                else:
                    prop_samples = prev_sample_mean
                
                # Calculate probabilities
                log_prob_diffusion = -0.5 * (prop_samples - prev_sample_mean).pow(2) / variance - torch.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                log_prob_diffusion = log_prob_diffusion.sum(dim=tuple(range(1, log_prob_diffusion.ndim)))
                
                if i >= start:
                    log_prob_proposal = -0.5 * (prop_samples - prev_sample_mean - variance * approx_guidance).pow(2) / variance - torch.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                else:
                    log_prob_proposal = log_prob_diffusion.clone()
                log_prob_proposal = log_prob_proposal.sum(dim=tuple(range(1, log_prob_proposal.ndim)))
                
                log_prob_diffusion = torch.nan_to_num(log_prob_diffusion, nan=-1e6)
                log_prob_proposal = torch.nan_to_num(log_prob_proposal, nan=1e6)
            else:
                prop_samples = prev_sample_mean
                
            pbar.update(1)
    
    # Final step
    samples = prop_samples.detach()
    log_twist_func_prev = log_twist_func.clone()
    
    # Final reward calculation
    final_rewards = torch.zeros(samples.shape[0], device=device)
    for idx in range(math.ceil(num_particles / batch_p)):
        tmp_samples = samples[batch_p*idx : batch_p*(idx+1)]
        denormalized_samples = diffusion_model.normalizer.unnormalize(tmp_samples)
        tmp_rewards = reward_fn(denormalized_samples).detach().to(torch.float32)
        final_rewards[batch_p*idx : batch_p*(idx+1)] = tmp_rewards
    
    log_twist_func = lookforward_fn(final_rewards)
    scale_factor_trace.append(min_scale_next.cpu())
    rewards_trace.append(final_rewards.view(-1, num_particles).max(dim=1)[0].cpu())
    
    # Final weight calculation
    if len(sigmas_and_gammas) > 0:
        log_w += log_prob_diffusion + log_twist_func - log_prob_proposal - log_twist_func_prev
        log_Z += torch.logsumexp(log_w, dim=-1)
    
    normalized_w = normalize_weights(log_w.view(-1, num_particles), dim=-1).view(-1)
    ess = [compute_ess_from_log_w(log_w_prompt) for log_w_prompt in log_w.view(-1, num_particles)]
    
    if verbose:
        print(f"Final ESS: {ess}")
        print(f"Final log Z: {log_Z}")
    
    all_log_w.append(log_w)
    ess_trace.append(torch.tensor(ess).cpu())
    
    # Return best sample (highest weight)
    best_sample = samples[torch.argmax(log_w)].unsqueeze(0)
    
    if clamp:
        best_sample = best_sample.clamp(-1., 1.)
    
    output = diffusion_model.normalizer.unnormalize(best_sample)
    
    additional_info = {
        'log_w': log_w,
        'normalized_w': normalized_w,
        'all_samples': all_samples,
        'all_log_w': all_log_w,
        'all_resample_indices': all_resample_indices,
        'ess_trace': torch.stack(ess_trace, dim=1) if ess_trace else torch.tensor([]),
        'scale_factor_trace': torch.stack(scale_factor_trace, dim=1) if scale_factor_trace else torch.tensor([]),
        'rewards_trace': torch.stack(rewards_trace, dim=1) if rewards_trace else torch.tensor([]),
        'log_Z': log_Z
    }
    
    return output, log_w, additional_info 