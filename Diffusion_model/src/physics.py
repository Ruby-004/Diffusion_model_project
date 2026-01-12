"""
Physics-informed losses and metrics for latent diffusion model training.

This module provides differentiable physics constraints for fluid flow prediction:
- Mass conservation (divergence penalty)
- Flow-rate consistency (constant flux constraint)
- Smoothness regularization (gradient-based and Laplacian-based)

Note: No-slip boundary condition is NOT included since the masking in the model
already ensures zero velocity in solid regions.

All losses operate on decoded velocity fields and are differentiable through the VAE decoder.

Tuning Recipe for Lambda Weights:
---------------------------------
1. Start with all lambdas = 0 (baseline training)
2. Enable one constraint at a time with small weight (1e-4 to 1e-3)
3. Monitor both physics metrics AND reconstruction quality
4. Scale up lambdas until physics metrics improve without degrading sample quality
5. Recommended starting values:
   - lambda_div: 0.01 (divergence/mass conservation)
   - lambda_flow: 0.001 (flow-rate consistency)
   - lambda_smooth: 0.001 (gradient smoothness regularization)
   - lambda_laplacian: 0.0001 (Laplacian smoothness - reduces high-freq noise)

Note: If reconstruction loss increases by more than 10-20%, reduce lambdas.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

__all__ = [
    'PhysicsLoss',
    'divergence_loss_masked',
    'flow_rate_consistency_loss',
    'smoothness_loss',
    'laplacian_smoothness_loss',
    'compute_physics_metrics',
    'component_weighted_velocity_loss',
    'compute_per_component_metrics'
]


class PhysicsLoss:
    """
    Combined physics-informed loss for fluid flow prediction.
    
    Computes multiple physics constraints on decoded velocity fields and returns
    both the total weighted loss and individual loss components for logging.
    """
    
    def __init__(
        self,
        lambda_div: float = 0.0,
        lambda_flow: float = 0.0,
        lambda_smooth: float = 0.0,
        lambda_laplacian: float = 0.0,
        eps: float = 1e-8,
        normalize_smoothness: bool = True
    ):
        """
        Args:
            lambda_div: Weight for divergence (mass conservation) loss
            lambda_flow: Weight for flow-rate consistency loss
            lambda_smooth: Weight for gradient smoothness regularization
            lambda_laplacian: Weight for Laplacian smoothness (reduces high-freq noise)
            eps: Small constant for numerical stability
            normalize_smoothness: If True, normalize smoothness losses by velocity magnitude
                                  to make them scale-invariant (recommended for small velocities)
        """
        self.lambda_div = lambda_div
        self.lambda_flow = lambda_flow
        self.lambda_smooth = lambda_smooth
        self.lambda_laplacian = lambda_laplacian
        self.eps = eps
        self.normalize_smoothness = normalize_smoothness
    
    def __call__(
        self,
        velocity: torch.Tensor,
        mask: torch.Tensor,
        return_components: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total physics loss.
        
        Args:
            velocity: Decoded velocity field (batch, num_slices, 3, H, W)
            mask: Binary fluid mask, 1=fluid, 0=solid (batch, num_slices, 1, H, W)
            return_components: If True, return individual loss components
            
        Returns:
            total_loss: Weighted sum of physics losses
            components: Dictionary of individual loss values
        """
        components = {}
        total_loss = torch.tensor(0.0, device=velocity.device)
        
        # Reshape for 3D operations: (batch, 3, num_slices, H, W)
        vel_5d = velocity.permute(0, 2, 1, 3, 4)
        mask_5d = mask.permute(0, 2, 1, 3, 4)  # (batch, 1, num_slices, H, W)
        
        # 1. Divergence loss (mass conservation)
        if self.lambda_div > 0:
            loss_div = divergence_loss_masked(vel_5d, mask_5d, eps=self.eps)
            total_loss = total_loss + self.lambda_div * loss_div
            components['divergence'] = loss_div.detach()
        
        # 2. Flow-rate consistency loss
        if self.lambda_flow > 0:
            loss_flow = flow_rate_consistency_loss(vel_5d, mask_5d, eps=self.eps)
            total_loss = total_loss + self.lambda_flow * loss_flow
            components['flow_rate'] = loss_flow.detach()
        
        # 3. Gradient smoothness regularization
        if self.lambda_smooth > 0:
            loss_smooth = smoothness_loss(vel_5d, mask_5d, eps=self.eps, normalize=self.normalize_smoothness)
            total_loss = total_loss + self.lambda_smooth * loss_smooth
            components['smoothness'] = loss_smooth.detach()
        
        # 4. Laplacian smoothness (better for reducing high-frequency noise)
        if self.lambda_laplacian > 0:
            loss_laplacian = laplacian_smoothness_loss(vel_5d, mask_5d, eps=self.eps, normalize=self.normalize_smoothness)
            total_loss = total_loss + self.lambda_laplacian * loss_laplacian
            components['laplacian'] = loss_laplacian.detach()
        
        if return_components:
            return total_loss, components
        return total_loss
    
    def is_active(self) -> bool:
        """Check if any physics constraint is enabled."""
        return (self.lambda_div > 0 or self.lambda_flow > 0 or 
                self.lambda_smooth > 0 or self.lambda_laplacian > 0)


def divergence_loss_masked(
    velocity: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute divergence penalty in fluid region only (mass conservation).
    
    For incompressible flow: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
    
    Args:
        velocity: (batch, 3, D, H, W) tensor with (u, v, w) velocity components
        mask: (batch, 1, D, H, W) binary mask, 1=fluid, 0=solid
        eps: Small constant for numerical stability
        
    Returns:
        Scalar divergence loss
    """
    assert velocity.dim() == 5, f"Expected 5D tensor (B,3,D,H,W), got {velocity.dim()}D"
    assert velocity.shape[1] == 3, f"Expected 3 velocity channels, got {velocity.shape[1]}"
    
    u = velocity[:, 0:1, :, :, :]  # (B, 1, D, H, W)
    v = velocity[:, 1:2, :, :, :]
    w = velocity[:, 2:3, :, :, :]
    
    # Compute spatial gradients using central differences
    # du/dx (gradient along width, dim=-1)
    du_dx = (u[:, :, :, :, 2:] - u[:, :, :, :, :-2]) / 2.0
    # dv/dy (gradient along height, dim=-2)  
    dv_dy = (v[:, :, :, 2:, :] - v[:, :, :, :-2, :]) / 2.0
    # dw/dz (gradient along depth, dim=-3)
    dw_dz = (w[:, :, 2:, :, :] - w[:, :, :-2, :, :]) / 2.0
    
    # Crop to common size (interior points only)
    D, H, W = velocity.shape[2:]
    du_dx = du_dx[:, :, 1:-1, 1:-1, :]  # (B, 1, D-2, H-2, W-2)
    dv_dy = dv_dy[:, :, 1:-1, :, 1:-1]
    dw_dz = dw_dz[:, :, :, 1:-1, 1:-1]
    
    # Crop mask to match
    mask_interior = mask[:, :, 1:-1, 1:-1, 1:-1]
    
    # Compute divergence
    divergence = du_dx + dv_dy + dw_dz  # (B, 1, D-2, H-2, W-2)
    
    # Masked divergence penalty (only in fluid region)
    divergence_masked = divergence * mask_interior
    
    # L2 norm of divergence in fluid region
    num_fluid_points = mask_interior.sum() + eps
    loss = (divergence_masked ** 2).sum() / num_fluid_points
    
    return loss


def flow_rate_consistency_loss(
    velocity: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Penalize variation in flow rate along the flow direction.
    
    For steady-state incompressible flow, the volumetric flow rate Q should be
    constant at all cross-sections: Q(x) = ∫∫ u dA = const
    
    We compute Q at each x-position and penalize the variance (deviation from mean).
    This is more robust than comparing to inlet only, as the mean is more stable.
    
    Args:
        velocity: (batch, 3, D, H, W) tensor with (u, v, w) velocity components
        mask: (batch, 1, D, H, W) binary mask, 1=fluid, 0=solid
        eps: Small constant for numerical stability
        
    Returns:
        Scalar flow-rate consistency loss
    """
    assert velocity.dim() == 5, f"Expected 5D tensor (B,3,D,H,W), got {velocity.dim()}D"
    
    # Extract u-velocity (flow direction assumed along x/width)
    u = velocity[:, 0:1, :, :, :]  # (B, 1, D, H, W)
    
    # Masked velocity (zero in solid)
    u_masked = u * mask
    
    # Compute flow rate at each cross-section (sum over depth and height)
    # Flow rate Q(x) = sum of (u * area) at each x position
    # Here we sum over D and H dimensions
    Q = u_masked.sum(dim=(2, 3), keepdim=True)  # (B, 1, 1, 1, W)
    Q = Q.squeeze(dim=(2, 3))  # (B, 1, W)
    
    # Also track number of fluid pixels at each cross-section for proper weighting
    fluid_area = mask.sum(dim=(2, 3), keepdim=True).squeeze(dim=(2, 3)) + eps  # (B, 1, W)
    
    # Normalize by fluid area to get mean velocity (more robust)
    Q_normalized = Q / fluid_area
    
    # Use mean flow rate as target (more stable than inlet-only)
    Q_mean = Q_normalized.mean(dim=-1, keepdim=True)  # (B, 1, 1)
    
    # Compute variance of flow rate across x-positions
    # This directly penalizes non-constant flow rate
    Q_variance = ((Q_normalized - Q_mean) ** 2).mean(dim=-1)  # (B, 1)
    
    # Normalize by mean squared to get relative variance (coefficient of variation squared)
    Q_mean_sq = Q_mean.squeeze(-1) ** 2 + eps  # (B, 1)
    relative_variance = Q_variance / Q_mean_sq
    
    # Mean across batch
    loss = relative_variance.mean()
    
    return loss


def no_slip_loss(
    velocity: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Penalize non-zero velocity in solid regions (no-slip boundary condition).
    
    At solid boundaries: u = v = w = 0
    
    Args:
        velocity: (batch, 3, D, H, W) tensor with velocity components
        mask: (batch, 1, D, H, W) binary mask, 1=fluid, 0=solid
        eps: Small constant for numerical stability
        
    Returns:
        Scalar no-slip loss
    """
    assert velocity.dim() == 5, f"Expected 5D tensor (B,3,D,H,W), got {velocity.dim()}D"
    
    # Solid region is where mask = 0
    solid_mask = 1.0 - mask  # (B, 1, D, H, W)
    
    # Velocity in solid region should be zero
    velocity_in_solid = velocity * solid_mask  # (B, 3, D, H, W)
    
    # L2 penalty on velocity magnitude in solid
    num_solid_points = solid_mask.sum() + eps
    loss = (velocity_in_solid ** 2).sum() / (num_solid_points * 3)  # divide by 3 for 3 components
    
    return loss


def smoothness_loss(
    velocity: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    normalize: bool = True
) -> torch.Tensor:
    """
    Penalize high-frequency noise in velocity field (Tikhonov regularization).
    
    L_smooth = mean(|∇u|^2) in fluid region
    
    This prevents overly noisy predictions while preserving physical gradients.
    
    Args:
        velocity: (batch, 3, D, H, W) tensor with velocity components
        mask: (batch, 1, D, H, W) binary mask, 1=fluid, 0=solid
        eps: Small constant for numerical stability
        normalize: If True, normalize by velocity magnitude squared (scale-invariant)
        
    Returns:
        Scalar smoothness loss
    """
    assert velocity.dim() == 5, f"Expected 5D tensor (B,3,D,H,W), got {velocity.dim()}D"
    
    total_grad_sq = torch.tensor(0.0, device=velocity.device)
    count = 0
    
    for c in range(3):  # For each velocity component
        vel_c = velocity[:, c:c+1, :, :, :]
        
        # Compute gradients in each direction
        # dx (along width)
        grad_x = vel_c[:, :, :, :, 1:] - vel_c[:, :, :, :, :-1]
        mask_x = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]  # Both points must be fluid
        
        # dy (along height)
        grad_y = vel_c[:, :, :, 1:, :] - vel_c[:, :, :, :-1, :]
        mask_y = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
        
        # dz (along depth)
        grad_z = vel_c[:, :, 1:, :, :] - vel_c[:, :, :-1, :, :]
        mask_z = mask[:, :, 1:, :, :] * mask[:, :, :-1, :, :]
        
        # Sum squared gradients in fluid region
        total_grad_sq = total_grad_sq + (grad_x ** 2 * mask_x).sum()
        total_grad_sq = total_grad_sq + (grad_y ** 2 * mask_y).sum()
        total_grad_sq = total_grad_sq + (grad_z ** 2 * mask_z).sum()
        
        count += mask_x.sum() + mask_y.sum() + mask_z.sum()
    
    loss = total_grad_sq / (count + eps)
    
    # Normalize by velocity magnitude squared to make scale-invariant
    if normalize:
        vel_mag_sq = ((velocity * mask) ** 2).sum() / (mask.sum() * 3 + eps)
        loss = loss / (vel_mag_sq + eps)
    
    return loss


def laplacian_smoothness_loss(
    velocity: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    normalize: bool = True
) -> torch.Tensor:
    """
    Penalize Laplacian magnitude in velocity field for smoother flow.
    
    L_laplacian = mean(|∇²u|^2) in fluid region
    
    The Laplacian (∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²) measures the local curvature
    of the velocity field. Minimizing this produces smoother, more diffused fields
    while preserving large-scale flow patterns better than gradient-based smoothness.
    
    This is particularly effective for reducing high-frequency noise/oscillations
    in the predicted velocity field.
    
    Args:
        velocity: (batch, 3, D, H, W) tensor with velocity components
        mask: (batch, 1, D, H, W) binary mask, 1=fluid, 0=solid
        eps: Small constant for numerical stability
        normalize: If True, normalize by velocity magnitude squared (scale-invariant)
        
    Returns:
        Scalar Laplacian smoothness loss
    """
    assert velocity.dim() == 5, f"Expected 5D tensor (B,3,D,H,W), got {velocity.dim()}D"
    
    total_laplacian_sq = torch.tensor(0.0, device=velocity.device)
    count = 0
    
    for c in range(3):  # For each velocity component
        vel_c = velocity[:, c:c+1, :, :, :]
        
        # Compute second derivatives using central differences
        # d²u/dx² = u[i+1] - 2*u[i] + u[i-1]
        
        # Along width (x direction)
        d2_dx2 = vel_c[:, :, :, :, 2:] - 2 * vel_c[:, :, :, :, 1:-1] + vel_c[:, :, :, :, :-2]
        
        # Along height (y direction)  
        d2_dy2 = vel_c[:, :, :, 2:, :] - 2 * vel_c[:, :, :, 1:-1, :] + vel_c[:, :, :, :-2, :]
        
        # Along depth (z direction)
        d2_dz2 = vel_c[:, :, 2:, :, :] - 2 * vel_c[:, :, 1:-1, :, :] + vel_c[:, :, :-2, :, :]
        
        # Crop to interior region (need 1 pixel border for second derivative)
        d2_dx2 = d2_dx2[:, :, 1:-1, 1:-1, :]   # (B, 1, D-2, H-2, W-2)
        d2_dy2 = d2_dy2[:, :, 1:-1, :, 1:-1]
        d2_dz2 = d2_dz2[:, :, :, 1:-1, 1:-1]
        
        # Laplacian = sum of second derivatives
        laplacian = d2_dx2 + d2_dy2 + d2_dz2
        
        # Mask for interior fluid region (all three points in each direction must be fluid)
        mask_interior = mask[:, :, 1:-1, 1:-1, 1:-1]
        
        # Also check neighbors for valid Laplacian computation
        mask_valid = (
            mask[:, :, 1:-1, 1:-1, :-2] * mask[:, :, 1:-1, 1:-1, 1:-1] * mask[:, :, 1:-1, 1:-1, 2:] *  # x neighbors
            mask[:, :, 1:-1, :-2, 1:-1] * mask[:, :, 1:-1, 2:, 1:-1] *  # y neighbors  
            mask[:, :, :-2, 1:-1, 1:-1] * mask[:, :, 2:, 1:-1, 1:-1]    # z neighbors
        )
        
        # Sum squared Laplacian in valid fluid region
        laplacian_masked = laplacian * mask_valid
        total_laplacian_sq = total_laplacian_sq + (laplacian_masked ** 2).sum()
        count += mask_valid.sum()
    
    loss = total_laplacian_sq / (count + eps)
    
    # Normalize by velocity magnitude squared to make scale-invariant
    if normalize:
        vel_mag_sq = ((velocity * mask) ** 2).sum() / (mask.sum() * 3 + eps)
        loss = loss / (vel_mag_sq + eps)
    
    return loss


def compute_physics_metrics(
    velocity: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute physics metrics for logging (detached, no gradients).
    
    Args:
        velocity: (batch, num_slices, 3, H, W) or (batch, 3, D, H, W) velocity field
        mask: Binary fluid mask matching velocity shape
        eps: Small constant for numerical stability
        verbose: If True, print debug information about velocity values
        
    Returns:
        Dictionary of physics metrics
    """
    with torch.no_grad():
        # Reshape if needed
        if velocity.dim() == 5 and velocity.shape[2] == 3:
            # Already in (B, D, 3, H, W) format, permute
            vel_5d = velocity.permute(0, 2, 1, 3, 4)
            mask_5d = mask.permute(0, 2, 1, 3, 4) if mask.shape[2] == 1 else mask
        else:
            vel_5d = velocity
            mask_5d = mask
        
        # Ensure mask is float for computations
        mask_5d = mask_5d.float()
        
        metrics = {}
        
        # Debug: print velocity statistics
        if verbose:
            print(f"  [DEBUG] velocity shape: {vel_5d.shape}, mask shape: {mask_5d.shape}")
            print(f"  [DEBUG] vel u: min={vel_5d[:,0].min():.2e}, max={vel_5d[:,0].max():.2e}, mean={vel_5d[:,0].mean():.2e}")
            print(f"  [DEBUG] vel v: min={vel_5d[:,1].min():.2e}, max={vel_5d[:,1].max():.2e}, mean={vel_5d[:,1].mean():.2e}")
            print(f"  [DEBUG] vel w: min={vel_5d[:,2].min():.2e}, max={vel_5d[:,2].max():.2e}, mean={vel_5d[:,2].mean():.2e}")
            print(f"  [DEBUG] mask sum: {mask_5d.sum():.0f}, fluid fraction: {mask_5d.mean():.3f}")
        
        # 1. Mean divergence magnitude in fluid
        try:
            u = vel_5d[:, 0:1, :, :, :]
            v = vel_5d[:, 1:2, :, :, :]
            w = vel_5d[:, 2:3, :, :, :]
            
            du_dx = (u[:, :, :, :, 2:] - u[:, :, :, :, :-2]) / 2.0
            dv_dy = (v[:, :, :, 2:, :] - v[:, :, :, :-2, :]) / 2.0
            dw_dz = (w[:, :, 2:, :, :] - w[:, :, :-2, :, :]) / 2.0
            
            # Crop to common size
            du_dx = du_dx[:, :, 1:-1, 1:-1, :]
            dv_dy = dv_dy[:, :, 1:-1, :, 1:-1]
            dw_dz = dw_dz[:, :, :, 1:-1, 1:-1]
            
            mask_interior = mask_5d[:, :, 1:-1, 1:-1, 1:-1]
            
            divergence = du_dx + dv_dy + dw_dz
            div_masked = divergence * mask_interior
            
            num_points = mask_interior.sum() + eps
            metrics['div_mean'] = (div_masked.abs().sum() / num_points).item()
            metrics['div_std'] = div_masked[mask_interior > 0.5].std().item() if mask_interior.sum() > 0 else 0.0
        except Exception:
            metrics['div_mean'] = 0.0
            metrics['div_std'] = 0.0
        
        # 2. Flow-rate variation (coefficient of variation)
        try:
            u = vel_5d[:, 0:1, :, :, :]
            u_masked = u * mask_5d
            Q = u_masked.sum(dim=(2, 3)).squeeze()  # (B, W)
            if Q.dim() == 1:
                Q = Q.unsqueeze(0)
            fluid_area = mask_5d.sum(dim=(2, 3)).squeeze() + eps
            if fluid_area.dim() == 1:
                fluid_area = fluid_area.unsqueeze(0)
            Q_normalized = Q / fluid_area
            Q_mean = Q_normalized.mean(dim=-1, keepdim=True)
            Q_mean_abs = Q_mean.abs().mean()
            # Only compute CV if mean is significantly non-zero (> 1e-6)
            if Q_mean_abs > 1e-6:
                Q_std = ((Q_normalized - Q_mean) ** 2).mean().sqrt()
                Q_variation = Q_std / (Q_mean_abs + eps)
                metrics['flow_rate_cv'] = Q_variation.item()
            else:
                # Mean flow too small, CV not meaningful
                metrics['flow_rate_cv'] = 0.0
        except Exception:
            metrics['flow_rate_cv'] = 0.0
        
        # 3. Mean velocity in solid region
        try:
            solid_mask = 1.0 - mask_5d
            velocity_in_solid = vel_5d * solid_mask
            num_solid = solid_mask.sum() + eps
            metrics['vel_in_solid'] = (velocity_in_solid ** 2).sum().sqrt().item() / (num_solid.item() ** 0.5)
        except Exception:
            metrics['vel_in_solid'] = 0.0
        
        # 4. Mean velocity magnitude in fluid
        try:
            vel_mag = (vel_5d ** 2).sum(dim=1, keepdim=True).sqrt()
            vel_mag_fluid = vel_mag * mask_5d
            num_fluid = mask_5d.sum() + eps
            metrics['vel_mean_fluid'] = (vel_mag_fluid.sum() / num_fluid).item()
        except Exception:
            metrics['vel_mean_fluid'] = 0.0
        
        # 5. Gradient smoothness (mean |∇u|² in fluid)
        try:
            total_grad_sq = 0.0
            count = 0
            for c in range(3):
                vel_c = vel_5d[:, c:c+1, :, :, :]
                grad_x = vel_c[:, :, :, :, 1:] - vel_c[:, :, :, :, :-1]
                mask_x = mask_5d[:, :, :, :, 1:] * mask_5d[:, :, :, :, :-1]
                grad_y = vel_c[:, :, :, 1:, :] - vel_c[:, :, :, :-1, :]
                mask_y = mask_5d[:, :, :, 1:, :] * mask_5d[:, :, :, :-1, :]
                grad_z = vel_c[:, :, 1:, :, :] - vel_c[:, :, :-1, :, :]
                mask_z = mask_5d[:, :, 1:, :, :] * mask_5d[:, :, :-1, :, :]
                
                total_grad_sq += (grad_x ** 2 * mask_x).sum()
                total_grad_sq += (grad_y ** 2 * mask_y).sum()
                total_grad_sq += (grad_z ** 2 * mask_z).sum()
                count += mask_x.sum() + mask_y.sum() + mask_z.sum()
            
            metrics['gradient_smooth'] = (total_grad_sq / (count + eps)).item()
        except Exception:
            metrics['gradient_smooth'] = 0.0
        
        # 6. Laplacian smoothness (mean |∇²u|² in fluid)
        try:
            total_laplacian_sq = 0.0
            count = 0
            for c in range(3):
                vel_c = vel_5d[:, c:c+1, :, :, :]
                d2_dx2 = vel_c[:, :, :, :, 2:] - 2 * vel_c[:, :, :, :, 1:-1] + vel_c[:, :, :, :, :-2]
                d2_dy2 = vel_c[:, :, :, 2:, :] - 2 * vel_c[:, :, :, 1:-1, :] + vel_c[:, :, :, :-2, :]
                d2_dz2 = vel_c[:, :, 2:, :, :] - 2 * vel_c[:, :, 1:-1, :, :] + vel_c[:, :, :-2, :, :]
                
                d2_dx2 = d2_dx2[:, :, 1:-1, 1:-1, :]
                d2_dy2 = d2_dy2[:, :, 1:-1, :, 1:-1]
                d2_dz2 = d2_dz2[:, :, :, 1:-1, 1:-1]
                
                laplacian = d2_dx2 + d2_dy2 + d2_dz2
                mask_interior = mask_5d[:, :, 1:-1, 1:-1, 1:-1]
                mask_valid = (
                    mask_5d[:, :, 1:-1, 1:-1, :-2] * mask_5d[:, :, 1:-1, 1:-1, 1:-1] * mask_5d[:, :, 1:-1, 1:-1, 2:] *
                    mask_5d[:, :, 1:-1, :-2, 1:-1] * mask_5d[:, :, 1:-1, 2:, 1:-1] *
                    mask_5d[:, :, :-2, 1:-1, 1:-1] * mask_5d[:, :, 2:, 1:-1, 1:-1]
                )
                
                laplacian_masked = laplacian * mask_valid
                total_laplacian_sq += (laplacian_masked ** 2).sum()
                count += mask_valid.sum()
            
            metrics['laplacian_smooth'] = (total_laplacian_sq / (count + eps)).item()
        except Exception:
            metrics['laplacian_smooth'] = 0.0
        
        # 7. Per-component velocity statistics (helps debug scaling issues)
        try:
            for c, name in enumerate(['vel_u', 'vel_v', 'vel_w']):
                vel_c = vel_5d[:, c:c+1, :, :, :] * mask_5d
                num_fluid = mask_5d.sum() + eps
                metrics[f'{name}_mean'] = (vel_c.abs().sum() / num_fluid).item()
                metrics[f'{name}_max'] = vel_c.abs().max().item()
        except Exception:
            for name in ['vel_u', 'vel_v', 'vel_w']:
                metrics[f'{name}_mean'] = 0.0
                metrics[f'{name}_max'] = 0.0
        
        return metrics


def reconstruct_velocity_from_noise_pred(
    noise_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    scheduler,
    vae_decoder,
    normalizer_output,
    batch_size: int,
    latent_depth: int,
    latent_channels: int,
    latent_h: int,
    latent_w: int,
    num_slices: int,
    img: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruct velocity field from predicted noise for physics loss computation.
    
    Uses the DDPM posterior mean estimate: x̂_0 = (x_t - sqrt(1-ᾱ_t)*ε_θ) / sqrt(ᾱ_t)
    
    Args:
        noise_pred: Predicted noise from U-Net
        x_t: Current noisy latent
        t: Current timestep
        scheduler: DiffusionScheduler instance
        vae_decoder: VAE decoder function
        normalizer_output: Output normalizer for denormalization
        batch_size, latent_depth, latent_channels, latent_h, latent_w: Shape info
        num_slices: Original number of slices
        img: Microstructure mask
        
    Returns:
        velocity_3d: Reconstructed velocity field (batch, num_slices, 3, H, W)
    """
    device = noise_pred.device
    
    # Get scheduler parameters for the timesteps
    sqrt_alphas_cumprod_t = scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    
    # Reconstruct x_0 estimate: x̂_0 = (x_t - sqrt(1-ᾱ_t)*ε) / sqrt(ᾱ_t)
    x0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / (sqrt_alphas_cumprod_t + 1e-8)
    
    # Reshape for VAE decoder: (batch, latent_depth, latent_channels, H, W) -> (batch, channels, depth, H, W)
    x0_pred_reshaped = x0_pred.reshape(batch_size, latent_depth, latent_channels, latent_h, latent_w)
    x0_pred_5d = x0_pred_reshaped.permute(0, 2, 1, 3, 4)
    
    # Decode with VAE (no gradient through frozen VAE, but we need gradients through x0_pred)
    velocity_5d = vae_decoder(x0_pred_5d)
    
    # Permute back to (batch, depth, 3, H, W)
    velocity_3d = velocity_5d.permute(0, 2, 1, 3, 4)
    
    # Denormalize
    batch, depth, channels, height, width = velocity_3d.shape
    velocity_flat = velocity_3d.reshape(batch * depth, channels, height, width)
    velocity_flat = normalizer_output.inverse(velocity_flat)
    velocity_3d = velocity_flat.reshape(batch, depth, channels, height, width)
    
    # Interpolate to match original slices if needed
    if depth != num_slices:
        velocity_3d = F.interpolate(
            velocity_3d.permute(0, 2, 1, 3, 4),
            size=(num_slices, height, width),
            mode='trilinear',
            align_corners=False
        ).permute(0, 2, 1, 3, 4)
    
    # Apply mask
    velocity_3d = velocity_3d * img
    
    return velocity_3d


def component_weighted_velocity_loss(
    velocity_pred: torch.Tensor,
    velocity_target: torch.Tensor,
    mask: torch.Tensor,
    weight_u: float = 1.0,
    weight_v: float = 1.0,
    weight_w: float = 1.0,
    eps: float = 1e-8,
    normalize_per_component: bool = True
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute velocity reconstruction loss with per-component weighting.
    
    This loss allows boosting the importance of the w (vz) component, which
    typically has much smaller magnitude than u (vx) and v (vy) in 3D flow.
    
    The w component often has:
    - 9x smaller median values than u
    - Much higher sparsity (90% of values near zero)
    
    Without component weighting, the model learns to predict near-zero w.
    
    Args:
        velocity_pred: Predicted velocity (batch, num_slices, 3, H, W)
        velocity_target: Target velocity (batch, num_slices, 3, H, W)
        mask: Binary fluid mask, 1=fluid, 0=solid (batch, num_slices, 1, H, W)
        weight_u: Weight for u (vx) component. Default 1.0.
        weight_v: Weight for v (vy) component. Default 1.0.
        weight_w: Weight for w (vz) component. Default 1.0 (increase to boost w learning).
        eps: Small constant for numerical stability.
        normalize_per_component: If True, normalize each component by its target std.
        
    Returns:
        total_loss: Weighted sum of per-component losses
        components: Dictionary with individual component losses
    """
    assert velocity_pred.dim() == 5, f"Expected 5D tensor, got {velocity_pred.dim()}D"
    assert velocity_pred.shape[2] == 3, f"Expected 3 velocity channels, got {velocity_pred.shape[2]}"
    
    weights = torch.tensor([weight_u, weight_v, weight_w], device=velocity_pred.device)
    
    # Apply mask
    velocity_pred = velocity_pred * mask
    velocity_target = velocity_target * mask
    
    components = {}
    total_loss = torch.tensor(0.0, device=velocity_pred.device)
    
    component_names = ['u', 'v', 'w']
    
    for i, (name, w) in enumerate(zip(component_names, [weight_u, weight_v, weight_w])):
        pred_c = velocity_pred[:, :, i, :, :]
        target_c = velocity_target[:, :, i, :, :]
        mask_c = mask[:, :, 0, :, :]
        
        # Compute per-component MAE
        error = (pred_c - target_c).abs()
        
        if normalize_per_component:
            # Normalize by target magnitude to make components comparable
            target_scale = (target_c.abs() * mask_c).sum() / (mask_c.sum() + eps)
            loss_c = (error * mask_c).sum() / (mask_c.sum() * target_scale + eps)
        else:
            loss_c = (error * mask_c).sum() / (mask_c.sum() + eps)
        
        components[f'loss_{name}'] = loss_c.detach()
        total_loss = total_loss + w * loss_c
    
    # Normalize by sum of weights
    total_loss = total_loss / (weight_u + weight_v + weight_w)
    
    return total_loss, components


def compute_per_component_metrics(
    velocity_pred: torch.Tensor,
    velocity_target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    Compute per-component velocity statistics for logging.
    
    Args:
        velocity_pred: Predicted velocity (batch, num_slices, 3, H, W)
        velocity_target: Target velocity (batch, num_slices, 3, H, W)
        mask: Binary fluid mask (batch, num_slices, 1, H, W)
        
    Returns:
        Dictionary of per-component metrics
    """
    with torch.no_grad():
        # Apply mask
        velocity_pred = velocity_pred * mask
        velocity_target = velocity_target * mask
        
        mask_flat = mask[:, :, 0, :, :].bool()
        
        metrics = {}
        component_names = ['u', 'v', 'w']
        
        for i, name in enumerate(component_names):
            pred_c = velocity_pred[:, :, i, :, :]
            target_c = velocity_target[:, :, i, :, :]
            
            # Get fluid region values only
            pred_vals = pred_c[mask_flat]
            target_vals = target_c[mask_flat]
            
            # MAE
            mae = (pred_vals - target_vals).abs().mean().item()
            
            # Relative error (normalized by target magnitude)
            target_mag = target_vals.abs().mean().item() + eps
            rel_error = mae / target_mag
            
            # Variance ratio (pred std / target std)
            pred_std = pred_vals.std().item()
            target_std = target_vals.std().item() + eps
            var_ratio = pred_std / target_std
            
            metrics[f'{name}_mae'] = mae
            metrics[f'{name}_rel_error'] = rel_error
            metrics[f'{name}_var_ratio'] = var_ratio
            metrics[f'{name}_pred_std'] = pred_std
            metrics[f'{name}_target_std'] = target_std
        
        return metrics
