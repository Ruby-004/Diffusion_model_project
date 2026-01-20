"""
DDPM Diffusion Scheduler for Latent Diffusion Model.

This module implements the noise scheduling and sampling procedures for the
denoising diffusion probabilistic model (DDPM) operating in the VAE latent space.

Key concepts:
- Forward diffusion: q(x_t | x_0) gradually adds noise over T timesteps
- Reverse diffusion: p(x_{t-1} | x_t) learned by the U-Net to denoise
- The U-Net predicts the noise (epsilon), not x_0 directly

The scheduler uses a linear beta schedule from 0.0001 to 0.02 over 1000 timesteps.

Usage:
    scheduler = DiffusionScheduler(num_timesteps=1000, device='cuda')
    
    # Forward: add noise to clean latent
    x_t = scheduler.q_sample(x_start=clean_latent, t=timestep, noise=noise)
    
    # Reverse: remove noise using model prediction
    x_prev = scheduler.p_sample(model_output=predicted_noise, x_t=x_t, t=t)

References:
    Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
    https://arxiv.org/abs/2006.11239
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionScheduler(nn.Module):
    """
    DDPM Noise Scheduler with improved numerical stability.
    
    Based on "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Define beta schedule (linear schedule)
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Store as float32 buffers for efficiency
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())
        
        # Pre-compute values for q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod).float())
        
        # Pre-compute values for posterior q(x_{t-1} | x_t, x_0)
        # Posterior variance: sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # Clip to avoid numerical issues at t=0
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        self.register_buffer('posterior_variance', posterior_variance.float())
        self.register_buffer('posterior_log_variance', torch.log(posterior_variance).float())
        
        # Posterior mean coefficients
        # mu_theta = coeff1 * x_0 + coeff2 * x_t
        # coeff1 = sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)
        # coeff2 = sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1.float())
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2.float())
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Handle both scalar and tensor t
        if isinstance(t, int):
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            # Expand dims for broadcasting
            while sqrt_alphas_cumprod_t.dim() < x_start.dim():
                sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
                sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        else:
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            # Reshape for broadcasting: (batch,) -> (batch, 1, 1, 1)
            while sqrt_alphas_cumprod_t.dim() < x_start.dim():
                sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
                sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_x0_from_noise(self, x_t, t, noise):
        """
        Reconstruct x_0 from x_t and predicted noise.
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        """
        if isinstance(t, int):
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            while sqrt_alphas_cumprod_t.dim() < x_t.dim():
                sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
                sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        else:
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            while sqrt_alphas_cumprod_t.dim() < x_t.dim():
                sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
                sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        # Avoid division by zero
        sqrt_alphas_cumprod_t = torch.clamp(sqrt_alphas_cumprod_t, min=1e-8)
        
        x_0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
        return x_0_pred

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute posterior distribution q(x_{t-1} | x_t, x_0).
        Returns mean and variance.
        """
        if isinstance(t, int):
            posterior_mean_coef1 = self.posterior_mean_coef1[t]
            posterior_mean_coef2 = self.posterior_mean_coef2[t]
            posterior_variance = self.posterior_variance[t]
            while posterior_mean_coef1.dim() < x_0.dim():
                posterior_mean_coef1 = posterior_mean_coef1.unsqueeze(-1)
                posterior_mean_coef2 = posterior_mean_coef2.unsqueeze(-1)
                posterior_variance = posterior_variance.unsqueeze(-1)
        else:
            posterior_mean_coef1 = self.posterior_mean_coef1[t]
            posterior_mean_coef2 = self.posterior_mean_coef2[t]
            posterior_variance = self.posterior_variance[t]
            while posterior_mean_coef1.dim() < x_0.dim():
                posterior_mean_coef1 = posterior_mean_coef1.unsqueeze(-1)
                posterior_mean_coef2 = posterior_mean_coef2.unsqueeze(-1)
                posterior_variance = posterior_variance.unsqueeze(-1)
        
        posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        return posterior_mean, posterior_variance

    def p_sample(self, model_output, x_t, t, clip_denoised=True, clip_range=(-20.0, 20.0)):
        """
        Sample from model posterior: p(x_{t-1} | x_t)
        Assumes model predicts noise epsilon.
        
        Args:
            model_output: Predicted noise from the model
            x_t: Current noisy sample
            t: Current timestep (scalar int)
            clip_denoised: Whether to clip the predicted x_0
            clip_range: Range for clipping predicted x_0 (for latent space)
        """
        # 1. Predict x_0 from noise
        x_0_pred = self.predict_x0_from_noise(x_t, t, model_output)
        
        # 2. Optionally clip x_0 to prevent explosion
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, clip_range[0], clip_range[1])
        
        # 3. Compute posterior mean and variance
        posterior_mean, posterior_variance = self.q_posterior_mean_variance(x_0_pred, x_t, t)
        
        # 4. Sample
        noise = torch.randn_like(x_t)
        
        # No noise when t == 0
        if isinstance(t, int):
            if t == 0:
                return posterior_mean
            else:
                return posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            # For tensor t, create a mask
            nonzero_mask = (t != 0).float()
            while nonzero_mask.dim() < x_t.dim():
                nonzero_mask = nonzero_mask.unsqueeze(-1)
            return posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise

    def to(self, device):
        """Move all tensors to specified device."""
        self.device = device
        return super().to(device)
    
    def ddim_sample(self, model_output, x_t, t, t_prev, eta=0.0, clip_range=(-30.0, 30.0)):
        """
        DDIM sampling step for faster inference.
        
        Args:
            model_output: Predicted noise from the model
            x_t: Current noisy sample  
            t: Current timestep
            t_prev: Previous timestep (target)
            eta: DDIM stochasticity (0 = deterministic, 1 = DDPM)
            clip_range: Range for clipping predicted x_0
        """
        # Get alpha values
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        
        # Reshape for broadcasting
        if isinstance(t, int):
            while alpha_bar_t.dim() < x_t.dim():
                alpha_bar_t = alpha_bar_t.unsqueeze(-1)
                alpha_bar_t_prev = alpha_bar_t_prev.unsqueeze(-1)
        
        # Predict x_0
        x_0_pred = self.predict_x0_from_noise(x_t, t, model_output)
        x_0_pred = torch.clamp(x_0_pred, clip_range[0], clip_range[1])
        
        # Compute sigma for stochasticity
        sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
        
        # Compute direction pointing to x_t
        pred_dir = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * model_output
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_bar_t_prev) * x_0_pred + pred_dir
        
        if eta > 0 and t > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev

