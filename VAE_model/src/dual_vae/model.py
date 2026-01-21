"""
Dual-Branch VAE: Separate encoders/decoders for 2D and 3D flow with latent alignment.

Architecture:
    E2D: 2D flow → latent z
    D2D: latent z → 2D flow reconstruction
    E3D: 3D flow → latent z  
    D3D: latent z → 3D flow reconstruction

Key innovation: Cross-reconstruction loss forces E2D to learn information sufficient
for D3D to predict the w-component, instead of relying on weak conditioning.
"""

import torch
import torch.nn as nn
from typing import Tuple

# Import existing encoder/decoder from vae module
import sys
import os

# Add VAE_model directory to path for proper imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_vae_model_dir = os.path.dirname(os.path.dirname(_current_dir))  # VAE_model/
if _vae_model_dir not in sys.path:
    sys.path.insert(0, _vae_model_dir)

from src.vae.encoder import Encoder
from src.vae.decoder import Decoder


class DualBranchVAE(nn.Module):
    """
    Dual-branch VAE with separate paths for 2D and 3D flow.
    
    Training:
        - L_rec_2d: E2D → D2D reconstruction
        - L_rec_3d: E3D → D3D reconstruction
        - L_align: Latent alignment between E2D and E3D (paired data)
        - L_cross: Cross-reconstruction E2D → D3D (2D→3D prediction)
    
    Inference (2D → 3D prediction):
        x_2d → E2D → [diffusion in latent space] → D3D → x_3d
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 8,
        kernel_size: int = 3,
        share_encoders: bool = False,
        share_decoders: bool = False
    ):
        """
        Args:
            in_channels: Number of velocity channels (3 for vx, vy, vz)
            latent_channels: Latent space dimension
            kernel_size: Convolution kernel size
            share_encoders: If True, E2D and E3D share parameters (not recommended initially)
            share_decoders: If True, D2D and D3D share parameters (not recommended initially)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.share_encoders = share_encoders
        self.share_decoders = share_decoders
        
        # === 2D Branch ===
        self.encoder_2d = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            kernel_size=kernel_size,
            conditional=False  # No conditioning needed, separate architectures
        )
        
        self.decoder_2d = Decoder(
            in_channels=latent_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            conditional=False
        )
        
        # === 3D Branch ===
        if share_encoders:
            self.encoder_3d = self.encoder_2d  # Share parameters
        else:
            self.encoder_3d = Encoder(
                in_channels=in_channels,
                out_channels=latent_channels,
                kernel_size=kernel_size,
                conditional=False
            )
        
        if share_decoders:
            self.decoder_3d = self.decoder_2d  # Share parameters
        else:
            self.decoder_3d = Decoder(
                in_channels=latent_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                conditional=False
            )
    
    @classmethod
    def from_directory(cls, folder, device=None, in_channels=3, latent_channels=8):
        """
        Load DualBranchVAE from a directory containing saved weights.
        
        Args:
            folder: Directory containing vae.pt, best_model.pt, or model.pt
            device: Device to load model on
            in_channels: Number of input channels (default 3 for velocity)
            latent_channels: Number of latent channels (default 8)
        
        Returns:
            Loaded DualBranchVAE model
        """
        import json
        
        # Try to load log if it exists to get latent_channels
        log_files = ['vae_log.json', 'log.json']
        for log_file in log_files:
            log_path = os.path.join(folder, log_file)
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        log = json.load(f)
                    if 'latent_channels' in log:
                        latent_channels = log['latent_channels']
                    if 'in_channels' in log:
                        in_channels = log['in_channels']
                    break
                except (json.JSONDecodeError, KeyError):
                    pass
        
        # Create model
        model = cls(
            in_channels=in_channels,
            latent_channels=latent_channels,
            share_encoders=False,
            share_decoders=False
        )
        
        # Find model file
        possible_files = ['vae.pt', 'best_model.pt', 'model.pt']
        model_path = None
        for filename in possible_files:
            candidate = os.path.join(folder, filename)
            if os.path.exists(candidate):
                model_path = candidate
                break
        
        if model_path is None:
            raise FileNotFoundError(f"No model file found in {folder}. Looked for: {', '.join(possible_files)}")
        
        # Load weights
        import torch
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded DualBranchVAE from {model_path}")
        
        return model

    def encode_2d(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode 2D flow (w=0) to latent space.
        
        Args:
            x: 2D velocity field (B, 3, D, H, W) with x[:, 2, :, :, :] ≈ 0
        
        Returns:
            z: Latent representation (B, latent_channels, D, H/4, W/4)
            (mu, logvar): VAE distribution parameters
        """
        mu, logvar = self.encoder_2d(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.encoder_2d.sample(mu, logvar)
        return z, (mu, logvar)
    
    def decode_2d(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to 2D flow.
        
        Args:
            z: Latent representation (B, latent_channels, D, H/4, W/4)
        
        Returns:
            x_2d: Reconstructed 2D flow (B, 3, D, H, W) with w≈0
        """
        x = self.decoder_2d(z)
        # Explicitly zero w channel for 2D flow
        x[:, 2, :, :, :] = 0.0
        return x
    
    def encode_3d(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode 3D flow (w≠0) to latent space.
        
        Args:
            x: 3D velocity field (B, 3, D, H, W) with w≠0
        
        Returns:
            z: Latent representation (B, latent_channels, D, H/4, W/4)
            (mu, logvar): VAE distribution parameters
        """
        mu, logvar = self.encoder_3d(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.encoder_3d.sample(mu, logvar)
        return z, (mu, logvar)
    
    def decode_3d(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to 3D flow.
        
        Args:
            z: Latent representation (B, latent_channels, D, H/4, W/4)
                Can come from E2D (cross-reconstruction) or E3D (reconstruction)
        
        Returns:
            x_3d: Reconstructed 3D flow (B, 3, D, H, W) with w≠0
        """
        x = self.decoder_3d(z)
        return x
    
    def encode_2d_deterministic(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Deterministic 2D encoding: z = mu (no sampling).
        Useful for Stage-2 teacher/student alignment and diffusion conditioning.
        """
        mu, logvar = self.encoder_2d(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = mu
        return z, (mu, logvar)

    def encode_3d_deterministic(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Deterministic 3D encoding: z = mu (no sampling).
        Useful for stable latent targets during alignment.
        """
        mu, logvar = self.encoder_3d(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = mu
        return z, (mu, logvar)

    def forward_2d_deterministic(
        self,
        x_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic 2D reconstruction: x_2d → E2D(mu) → D2D → x̂_2d
        Returns:
            x_2d_recon, mu_2d
        """
        z, (mu, _) = self.encode_2d_deterministic(x_2d)
        x_2d_recon = self.decode_2d(z)
        return x_2d_recon, mu

    def forward_2d(
        self,
        x_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full 2D reconstruction: x_2d → E2D → D2D → x̂_2d
        
        Returns:
            x_2d_recon: Reconstructed 2D flow
            (mu, logvar): Latent distribution parameters for KL loss
        """
        z, (mu, logvar) = self.encode_2d(x_2d)
        x_2d_recon = self.decode_2d(z)
        return x_2d_recon, (mu, logvar)
    
    def forward_3d(
        self,
        x_3d: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full 3D reconstruction: x_3d → E3D → D3D → x̂_3d
        
        Returns:
            x_3d_recon: Reconstructed 3D flow
            (mu, logvar): Latent distribution parameters for KL loss
        """
        z, (mu, logvar) = self.encode_3d(x_3d)
        x_3d_recon = self.decode_3d(z)
        return x_3d_recon, (mu, logvar)
    
    def forward_cross_2d_to_3d(
        self,
        x_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-reconstruction: x_2d → E2D → D3D → x̂_3d
        
        This forces E2D to learn features sufficient for D3D to predict w.
        
        Returns:
            x_3d_from_2d: 3D flow predicted from 2D latent
            z_2d: 2D latent (for computing alignment loss)
        """
        z_2d, (mu_2d, _) = self.encode_2d_deterministic(x_2d)
        x_3d_from_2d = self.decode_3d(z_2d)

        return x_3d_from_2d, z_2d
    
    def forward_cross_3d_to_2d(
        self,
        x_3d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optional cross-reconstruction: x_3d → E3D → D2D → x̂_2d
        
        This encourages E3D to preserve 2D information (u, v components).
        
        Returns:
            x_2d_from_3d: 2D flow predicted from 3D latent
            z_3d: 3D latent (for computing alignment loss)
        """
        z_3d, _ = self.encode_3d(x_3d)
        x_2d_from_3d = self.decode_2d(z_3d)
        return x_2d_from_3d, z_3d
    
    def compute_alignment_loss(
        self,
        x_2d: torch.Tensor,
        x_3d: torch.Tensor,
        mode: str = 'symmetric'
    ) -> torch.Tensor:
        """
        Compute latent alignment loss for paired 2D/3D samples.
        
        Args:
            x_2d: 2D flow from microstructure
            x_3d: Corresponding 3D flow from SAME microstructure
            mode: 'symmetric', 'one_way', or 'stop_grad'
                - symmetric: ||z_2d - z_3d||²
                - one_way: ||z_2d - stopgrad(z_3d)||²
                - stop_grad: Forces E2D to match E3D without affecting E3D
        
        Returns:
            Alignment loss (scalar)
        """
        z_2d, (mu_2d, _) = self.encode_2d_deterministic(x_2d)
        z_3d, (mu_3d, _) = self.encode_3d_deterministic(x_3d)

        
        if mode == 'symmetric':
            loss = torch.nn.functional.mse_loss(z_2d, z_3d)
        elif mode == 'one_way':
            loss = torch.nn.functional.mse_loss(z_2d, z_3d.detach())
        elif mode == 'stop_grad':
            # Train E2D to match E3D, but don't affect E3D's gradient
            loss = torch.nn.functional.mse_loss(z_2d, z_3d.detach())
        else:
            raise ValueError(f"Unknown alignment mode: {mode}")
        
        return loss
    
    def predict_2d_to_3d(self, x_2d: torch.Tensor) -> torch.Tensor:
        """
        Predict 3D flow from 2D flow (inference mode).
        
        For diffusion model, you would instead:
        1. z_2d = E2D(x_2d)
        2. z_3d = diffusion_denoise(z_2d, ...)
        3. x_3d = D3D(z_3d)
        
        Args:
            x_2d: 2D velocity field
        
        Returns:
            x_3d: Predicted 3D velocity field
        """
        with torch.no_grad():
            z_2d, _ = self.encode_2d(x_2d)
            x_3d = self.decode_3d(z_2d)
        return x_3d


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Standard VAE KL divergence: KL(q(z|x) || N(0,I))"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# === Example Training Loop ===
def train_dual_vae_example():
    """
    Example training loop showing loss computation.
    
    In practice, you'd integrate this into the main training script with:
    - DataLoader providing paired (x_2d, x_3d) samples
    - Phased training schedule (warmup → alignment → cross-recon)
    - Loss weight scheduling
    - Proper logging and checkpointing
    """
    
    # Initialize model
    model = DualBranchVAE(
        in_channels=3,
        latent_channels=8,
        share_encoders=False,  # Start with separate encoders
        share_decoders=False
    )
    
    # Hyperparameters
    lambda_align = 0.1  # Alignment loss weight
    lambda_cross_2d3d = 1.0  # 2D→3D cross-reconstruction weight
    lambda_cross_3d2d = 0.5  # 3D→2D cross-reconstruction weight (optional)
    beta_kl = 1e-3  # KL weight
    
    # Dummy batch - use smaller tensors for testing (B=1, 3 channels, D=11, H=64, W=64)
    # For real training, use H=256, W=256
    x_2d = torch.randn(1, 3, 11, 64, 64)
    x_2d[:, 2, :, :, :] = 0  # Zero w for 2D
    x_3d = torch.randn(1, 3, 11, 64, 64)  # w ≠ 0 for 3D
    
    print("Testing DualBranchVAE with dummy data...")
    print(f"  Input shape: {x_2d.shape}")
    
    # === Phase 1: Reconstruction Losses ===
    
    # 2D reconstruction
    x_2d_recon, (mu_2d, logvar_2d) = model.forward_2d(x_2d)
    loss_rec_2d = torch.nn.functional.mse_loss(x_2d_recon, x_2d)
    loss_kl_2d = kl_divergence(mu_2d, logvar_2d)
    
    # 3D reconstruction
    x_3d_recon, (mu_3d, logvar_3d) = model.forward_3d(x_3d)
    loss_rec_3d = torch.nn.functional.mse_loss(x_3d_recon, x_3d)
    loss_kl_3d = kl_divergence(mu_3d, logvar_3d)
    
    # === Phase 2: Latent Alignment (for paired data) ===
    loss_align = model.compute_alignment_loss(x_2d, x_3d, mode='symmetric')
    
    # === Phase 3: Cross-Reconstruction ===
    
    # 2D → 3D (KEY: forces E2D to learn w-predictive features)
    x_3d_from_2d, z_2d = model.forward_cross_2d_to_3d(x_2d)
    loss_cross_2d3d = torch.nn.functional.mse_loss(x_3d_from_2d, x_3d)
    
    # 3D → 2D (optional: encourages E3D to preserve 2D info)
    x_2d_from_3d, z_3d = model.forward_cross_3d_to_2d(x_3d)
    loss_cross_3d2d = torch.nn.functional.mse_loss(x_2d_from_3d, x_2d)
    
    # === Total Loss ===
    loss_total = (
        loss_rec_2d + beta_kl * loss_kl_2d +
        loss_rec_3d + beta_kl * loss_kl_3d +
        lambda_align * loss_align +
        lambda_cross_2d3d * loss_cross_2d3d +
        lambda_cross_3d2d * loss_cross_3d2d
    )
    
    print(f"Loss breakdown:")
    print(f"  Rec 2D: {loss_rec_2d.item():.6f}")
    print(f"  Rec 3D: {loss_rec_3d.item():.6f}")
    print(f"  KL 2D:  {loss_kl_2d.item():.6f}")
    print(f"  KL 3D:  {loss_kl_3d.item():.6f}")
    print(f"  Align:  {loss_align.item():.6f}")
    print(f"  Cross 2D→3D: {loss_cross_2d3d.item():.6f}")
    print(f"  Cross 3D→2D: {loss_cross_3d2d.item():.6f}")
    print(f"  TOTAL:  {loss_total.item():.6f}")


if __name__ == '__main__':
    train_dual_vae_example()
