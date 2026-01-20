"""
Loss functions and metrics for latent diffusion model training.

This module provides:
- Standard loss functions (MAE, MSE, Huber) for noise prediction
- Per-component loss functions for velocity fields
- Normalized variants that scale by target magnitude
- 3D divergence loss for physics-based constraints

Primary loss function used in training: normalized_mae_loss
Alternative per-component variants are available for better w-component learning.

For more sophisticated physics losses with masking and multiple constraints,
see src/physics.py which provides divergence, flow-rate, and smoothness losses.
"""

from typing import Callable

import torch
from torch import linalg


__all__ = [
    'cost_function',
    'mae_loss',
    'mse_loss',
    'huber_loss',
    'normalized_mae_loss',
    'normalized_mse_loss',
    'divergence_loss',
    'mae_loss_per_component',
    'mse_loss_per_component',
    'normalized_mae_loss_per_component',
    'normalized_mse_loss_per_component',
]


def cost_function(name: str) -> Callable[..., torch.Tensor]:
    """
    Factory function to retrieve a loss function by name.
    
    Args:
        name: Name of the loss function (must be defined in this module).
        
    Returns:
        The corresponding loss function callable.
        
    Example:
        criterion = cost_function('normalized_mae_loss')
        loss = criterion(output=preds, target=targets)
    """
    func = eval(name)
    return func


def mse_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    reduce=True
) -> torch.Tensor:
    """
    Mean Squared Error.
    """
    dim = (-3,-2,-1) # everything except batch dimension
    loss = torch.mean(
        (output - target)**2,
        dim=dim
    )
    if reduce:
        loss = loss.mean()
    return loss


def huber_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    reduce=True,
    delta=1.0
) -> torch.Tensor:
    """
    Huber Loss.
    """
    # PyTorch's F.huber_loss reduction is 'mean' or 'sum' by default over all elements.
    # To match the behavior of other losses here (reduce over batch at the end), we can use elementwise.
    loss = torch.nn.functional.huber_loss(output, target, reduction='none', delta=delta)
    
    dim = (-3,-2,-1)
    loss = torch.mean(loss, dim=dim)
    
    if reduce:
        loss = loss.mean()
    return loss


def mae_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    reduce=True
) -> torch.Tensor:
    """
    Mean Absolute Error.

    `output`: output predicted by the ML model with same shape as `target`.\n
    `target`: target value with shape: (batch,channels,height,width).\n
    `reduce`: whether to take batch average.
    """

    dim = (-3,-2,-1) # everthing except batch dimension

    # error for each sample
    loss = torch.mean(
        torch.abs((output - target)),
        dim=dim
    )

    if reduce:
        # average
        loss = loss.mean()

    return loss


def mae_loss_per_component(
    output: torch.Tensor,
    target: torch.Tensor,
    reduce=True,
    weight_per_channel: torch.Tensor = None
) -> torch.Tensor:
    """
    Mean Absolute Error computed per-channel, then averaged.
    
    This gives the model component-specific feedback by computing loss
    separately for each velocity component (u, v, w) before averaging.
    
    Args:
        output: Predicted tensor (batch, channels, height, width) or (batch, channels, depth, height, width)
        target: Target tensor with same shape as output
        reduce: Whether to take batch average
        weight_per_channel: Optional per-channel weights (channels,). Default: equal weights.
    
    Returns:
        Scalar loss
    """
    # Compute error per channel, averaging only over spatial dimensions
    if output.dim() == 4:
        # 2D: (batch, channels, height, width)
        spatial_dims = (-2, -1)
    elif output.dim() == 5:
        # 3D: (batch, channels, depth, height, width)
        spatial_dims = (-3, -2, -1)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {output.dim()}D")
    
    # Compute MAE per channel: (batch, channels)
    loss_per_channel = torch.mean(
        torch.abs(output - target),
        dim=spatial_dims
    )
    
    # Apply channel weights if provided
    if weight_per_channel is not None:
        if weight_per_channel.dim() == 1:
            # Broadcast to (1, channels)
            weight_per_channel = weight_per_channel.unsqueeze(0)
        loss_per_channel = loss_per_channel * weight_per_channel
        # Normalize by sum of weights
        loss_per_channel = loss_per_channel / weight_per_channel.sum()
    
    # Average over channels: (batch,)
    loss = torch.mean(loss_per_channel, dim=-1)
    
    if reduce:
        # Average over batch
        loss = loss.mean()
    
    return loss


def normalized_mae_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    reduce=True
) -> torch.Tensor:
    """
    Normalized Mean Absolute Error.

    `output`: output predicted by the ML model with same shape as `target`.\n
    `target`: target value with shape: (batch,channels,height,width).\n
    `reduce`: whether to take batch average.
    """

    dim = (-3,-2,-1) # everthing except batch dimension

    mae = torch.mean(
        torch.abs((output - target)),
        dim=dim
    )
    weight = torch.mean(
        torch.abs(target),
        dim=dim
    )

    # error for each sample
    error = mae / weight # shape: (B,)

    if reduce:
        # average
        error = error.mean() # shape: (1)

    return error


def mse_loss_per_component(
    output: torch.Tensor,
    target: torch.Tensor,
    reduce=True,
    weight_per_channel: torch.Tensor = None
) -> torch.Tensor:
    """
    Mean Squared Error computed per-channel, then averaged.
    
    This gives the model component-specific feedback by computing loss
    separately for each velocity component (u, v, w) before averaging.
    
    Args:
        output: Predicted tensor (batch, channels, height, width) or (batch, channels, depth, height, width)
        target: Target tensor with same shape as output
        reduce: Whether to take batch average
        weight_per_channel: Optional per-channel weights (channels,). Default: equal weights.
    
    Returns:
        Scalar loss
    """
    # Compute error per channel, averaging only over spatial dimensions
    if output.dim() == 4:
        # 2D: (batch, channels, height, width)
        spatial_dims = (-2, -1)
    elif output.dim() == 5:
        # 3D: (batch, channels, depth, height, width)
        spatial_dims = (-3, -2, -1)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {output.dim()}D")
    
    # Compute MSE per channel: (batch, channels)
    loss_per_channel = torch.mean(
        (output - target)**2,
        dim=spatial_dims
    )
    
    # Apply channel weights if provided
    if weight_per_channel is not None:
        if weight_per_channel.dim() == 1:
            # Broadcast to (1, channels)
            weight_per_channel = weight_per_channel.unsqueeze(0)
        loss_per_channel = loss_per_channel * weight_per_channel
        # Normalize by sum of weights
        loss_per_channel = loss_per_channel / weight_per_channel.sum()
    
    # Average over channels: (batch,)
    loss = torch.mean(loss_per_channel, dim=-1)
    
    if reduce:
        # Average over batch
        loss = loss.mean()
    
    return loss


def normalized_mae_loss_per_component(
    output: torch.Tensor,
    target: torch.Tensor,
    reduce=True,
    weight_per_channel: torch.Tensor = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalized Mean Absolute Error computed per-channel, then averaged.
    
    Each component is normalized by its own target magnitude, preventing
    large-magnitude components from dominating the loss. This is critical
    for 3D flow where vz is typically 9x smaller than vx/vy.
    
    Args:
        output: Predicted tensor (batch, channels, height, width) or (batch, channels, depth, height, width)
        target: Target tensor with same shape as output
        reduce: Whether to take batch average
        weight_per_channel: Optional per-channel weights (channels,). Default: equal weights.
        eps: Small constant for numerical stability
    
    Returns:
        Scalar loss
    """
    # Determine spatial dimensions
    if output.dim() == 4:
        # 2D: (batch, channels, height, width)
        spatial_dims = (-2, -1)
    elif output.dim() == 5:
        # 3D: (batch, channels, depth, height, width)
        spatial_dims = (-3, -2, -1)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {output.dim()}D")
    
    # Compute MAE per channel: (batch, channels)
    mae_per_channel = torch.mean(
        torch.abs(output - target),
        dim=spatial_dims
    )
    
    # Compute normalization weight per channel: (batch, channels)
    weight_per_channel_norm = torch.mean(
        torch.abs(target),
        dim=spatial_dims
    )
    
    # Normalize each channel by its own magnitude
    normalized_error_per_channel = mae_per_channel / (weight_per_channel_norm + eps)
    
    # Apply channel weights if provided
    if weight_per_channel is not None:
        if weight_per_channel.dim() == 1:
            # Broadcast to (1, channels)
            weight_per_channel = weight_per_channel.unsqueeze(0)
        normalized_error_per_channel = normalized_error_per_channel * weight_per_channel
        # Normalize by sum of weights
        normalized_error_per_channel = normalized_error_per_channel / weight_per_channel.sum()
    
    # Average over channels: (batch,)
    error = torch.mean(normalized_error_per_channel, dim=-1)
    
    if reduce:
        # Average over batch
        error = error.mean()
    
    return error


def normalized_mse_loss_per_component(
    output: torch.Tensor,
    target: torch.Tensor,
    reduce=True,
    weight_per_channel: torch.Tensor = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalized Mean Squared Error computed per-channel, then averaged.
    
    Each component is normalized by its own target magnitude squared, preventing
    large-magnitude components from dominating the loss. Similar to normalized_mae_loss_per_component
    but uses squared error instead of absolute error.
    
    Args:
        output: Predicted tensor (batch, channels, height, width) or (batch, channels, depth, height, width)
        target: Target tensor with same shape as output
        reduce: Whether to take batch average
        weight_per_channel: Optional per-channel weights (channels,). Default: equal weights.
        eps: Small constant for numerical stability
    
    Returns:
        Scalar loss
    """
    # Determine spatial dimensions
    if output.dim() == 4:
        # 2D: (batch, channels, height, width)
        spatial_dims = (-2, -1)
    elif output.dim() == 5:
        # 3D: (batch, channels, depth, height, width)
        spatial_dims = (-3, -2, -1)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {output.dim()}D")
    
    # Compute MSE per channel: (batch, channels)
    mse_per_channel = torch.mean(
        (output - target)**2,
        dim=spatial_dims
    )
    
    # Compute normalization weight per channel: mean squared magnitude (batch, channels)
    weight_per_channel_norm = torch.mean(
        target**2,
        dim=spatial_dims
    )
    
    # Normalize each channel by its own magnitude squared
    normalized_error_per_channel = mse_per_channel / (weight_per_channel_norm + eps)
    
    # Apply channel weights if provided
    if weight_per_channel is not None:
        if weight_per_channel.dim() == 1:
            # Broadcast to (1, channels)
            weight_per_channel = weight_per_channel.unsqueeze(0)
        normalized_error_per_channel = normalized_error_per_channel * weight_per_channel
        # Normalize by sum of weights
        normalized_error_per_channel = normalized_error_per_channel / weight_per_channel.sum()
    
    # Average over channels: (batch,)
    error = torch.mean(normalized_error_per_channel, dim=-1)
    
    if reduce:
        # Average over batch
        error = error.mean()
    
    return error


def normalized_mse_loss(
    output: torch.Tensor,
    target: torch.Tensor
):
    """
    Normalize mean-square-error loss.

    `output`: output predicted by the ML model (shape: [B,C,W,H]).\n
    `target`: target value (shape: [B,C,W,H]).\n
    """

    # Sample-wise norms
    # shape: (batch, channels)
    smp_wise_diff_norm = linalg.matrix_norm(
        target - output,
        dim=(-2,-1)
    )**2

    smp_wise_target_norm = linalg.matrix_norm(
        target,
        dim=(-2,-1)
    )**2

    # Avoid division by zero
    epsilon = 1e-8

    # loss for each channel and sample
    normalized_mse = smp_wise_diff_norm / (smp_wise_target_norm + epsilon)

    # Average over batch and channels to get a single scalar representing the percentage of error
    loss = torch.mean(normalized_mse)

    return loss


# =============================================================================
# PHYSICS-BASED LOSS FUNCTIONS
# =============================================================================
# Note: For more sophisticated physics losses with masking and multiple
# constraints, see src/physics.py which provides divergence, flow-rate,
# and smoothness losses computed on decoded velocity fields.

def divergence_loss(
    flow_field: torch.Tensor
) -> torch.Tensor:
    """
    Calculate divergence loss for a flow field.
    
    `flow_field`: (B, 3, D, H, W) tensor representing (u, v, w) velocity components.
    """
    assert flow_field.dim() == 5, f"Expected 5D tensor, got {flow_field.dim()}D"
    assert flow_field.shape[1] == 3, f"Expected 3 channels (u, v, w), got {flow_field.shape[1]}"

    u = flow_field[:, 0, :, :, :]
    v = flow_field[:, 1, :, :, :]
    w = flow_field[:, 2, :, :, :]

    # Compute gradients (central differences)
    # torch.gradient computes the gradient along the given dimension
    # spacing=1.0 assumed
    
    # du/dx (width is last dim)
    du_dx = torch.gradient(u, dim=-1)[0]
    
    # dv/dy (height is 2nd to last)
    dv_dy = torch.gradient(v, dim=-2)[0]
    
    # dw/dz (depth is 3rd to last)
    dw_dz = torch.gradient(w, dim=-3)[0]

    div = du_dx + dv_dy + dw_dz
    
    # L_div = || div ||^2
    # Mean over spatial dims, mean over batch
    loss = torch.mean(div**2)
    
    return loss

