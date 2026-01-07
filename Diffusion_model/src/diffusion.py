import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionScheduler(nn.Module):
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means no diffusion)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) # Match dimensions
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, model_output, x_t, t, clip_denoised=True):
        """
        Sample from the model: p(x_{t-1} | x_t)
        Assume model predicts noise epsilon
        """
        # Equation 11 in DDPM paper
        # mu_theta(x_t, t) = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta(x_t, t))
        
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )
        
        if clip_denoised:
            # Optional: clip x_0 prediction if we were predicting x_0 directly, but here we predict noise.
            # We can reconstruct x_0 and clip it, then recompute mean?
            # For latents, clipping might not make sense unless we know range. 
            # We'll skip complex clipping for now.
            pass

        # Compute variance: sigma_t^2 = posterior_variance
        posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
        
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        mask = (t > 0).float().view(-1, 1, 1, 1)
        
        return model_mean + torch.sqrt(posterior_variance_t) * noise * mask

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device) 
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.device = device
        return self

