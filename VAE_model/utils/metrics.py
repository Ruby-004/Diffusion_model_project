import torch


def normalized_mae_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    reduce=True,
    eps=1e-8
) -> torch.Tensor:
    """
    Normalized Mean Absolute Error.

    `output`: output predicted by the ML model with same shape as `target`.\n
    `target`: target value with shape: (batch,channels,height,width).\n
    `reduce`: whether to take batch average.\n
    `eps`: small epsilon to prevent division by zero.
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

    # error for each sample - add epsilon to prevent division by zero
    error = mae / (weight + eps) # shape: (B,)

    if reduce:
        # average
        error = error.mean() # shape: (1)

    return error


def mae_loss_per_channel(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    weight_per_channel: torch.Tensor = None,
    reduce=True
) -> torch.Tensor:
    """
    MAE computed per-channel for VAE training.
    
    This gives the model component-specific feedback by computing loss
    separately for each velocity component (u, v, w) before averaging.
    This prevents the larger u/v components from dominating the loss
    and ignoring the smaller w-component.
    
    Args:
        output: (B, C, D, H, W) or (B, C, H, W) predicted velocity
        target: Same shape as output - target velocity
        mask: (B, 1, D, H, W) or (B, 1, H, W) fluid mask (optional)
        weight_per_channel: Optional (C,) weights for each channel
        reduce: Whether to average over batch
        
    Returns:
        Scalar loss if reduce=True, else (B,) tensor
    """
    if mask is not None:
        # Apply mask to both
        mask_expanded = mask.expand_as(output)
        output = output * mask_expanded
        target = target * mask_expanded
    
    # Determine spatial dimensions based on tensor rank
    if output.dim() == 5:
        # 3D: (B, C, D, H, W)
        spatial_dims = (-3, -2, -1)
    elif output.dim() == 4:
        # 2D: (B, C, H, W)
        spatial_dims = (-2, -1)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {output.dim()}D")
    
    # Compute MAE per channel: (B, C)
    loss_per_channel = torch.mean(
        torch.abs(output - target),
        dim=spatial_dims
    )
    
    # Apply channel weights if provided
    if weight_per_channel is not None:
        if weight_per_channel.dim() == 1:
            weight_per_channel = weight_per_channel.unsqueeze(0)
        loss_per_channel = loss_per_channel * weight_per_channel
        loss_per_channel = loss_per_channel / weight_per_channel.sum()
    
    # Average over channels: (B,)
    loss = torch.mean(loss_per_channel, dim=-1)
    
    if reduce:
        loss = loss.mean()
    
    return loss


def normalized_mae_loss_per_channel(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    reduce: bool = True,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalized MAE computed per-channel for VAE training.
    
    This combines the benefits of:
    1. Per-channel computation: Prevents larger u/v components from dominating
       and ignoring the smaller w-component
    2. Normalization: Scale-invariant loss that divides by target magnitude
    
    Args:
        output: (B, C, D, H, W) or (B, C, H, W) predicted velocity
        target: Same shape as output - target velocity
        mask: (B, 1, D, H, W) or (B, 1, H, W) fluid mask (optional)
        reduce: Whether to average over batch
        eps: Small epsilon to prevent division by zero
        
    Returns:
        Scalar loss if reduce=True, else (B,) tensor
    """
    if mask is not None:
        # Apply mask to both
        mask_expanded = mask.expand_as(output)
        output = output * mask_expanded
        target = target * mask_expanded
    
    # Determine spatial dimensions based on tensor rank
    if output.dim() == 5:
        # 3D: (B, C, D, H, W)
        spatial_dims = (-3, -2, -1)
    elif output.dim() == 4:
        # 2D: (B, C, H, W)
        spatial_dims = (-2, -1)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {output.dim()}D")
    
    # Compute MAE per channel: (B, C)
    mae_per_channel = torch.mean(
        torch.abs(output - target),
        dim=spatial_dims
    )
    
    # Compute normalization weight per channel: (B, C)
    weight_per_channel = torch.mean(
        torch.abs(target),
        dim=spatial_dims
    )
    
    # Normalized MAE per channel: (B, C)
    normalized_mae = mae_per_channel / (weight_per_channel + eps)
    
    # Average over channels: (B,)
    loss = torch.mean(normalized_mae, dim=-1)
    
    if reduce:
        loss = loss.mean()
    
    return loss


def normalized_mse_per_channel(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    reduce: bool = True,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalized MSE computed per-channel for VAE training.
    
    Similar to normalized_mae_per_channel but uses squared error.
    This penalizes larger errors more heavily than MAE.
    
    Args:
        output: (B, C, D, H, W) or (B, C, H, W) predicted velocity
        target: Same shape as output - target velocity
        mask: (B, 1, D, H, W) or (B, 1, H, W) fluid mask (optional)
        reduce: Whether to average over batch
        eps: Small epsilon to prevent division by zero
        
    Returns:
        Scalar loss if reduce=True, else (B,) tensor
    """
    if mask is not None:
        # Apply mask to both
        mask_expanded = mask.expand_as(output)
        output = output * mask_expanded
        target = target * mask_expanded
    
    # Determine spatial dimensions based on tensor rank
    if output.dim() == 5:
        # 3D: (B, C, D, H, W)
        spatial_dims = (-3, -2, -1)
    elif output.dim() == 4:
        # 2D: (B, C, H, W)
        spatial_dims = (-2, -1)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {output.dim()}D")
    
    # Compute MSE per channel: (B, C)
    mse_per_channel = torch.mean(
        (output - target).pow(2),
        dim=spatial_dims
    )
    
    # Compute normalization weight per channel (mean squared target): (B, C)
    weight_per_channel = torch.mean(
        target.pow(2),
        dim=spatial_dims
    )
    
    # Normalized MSE per channel: (B, C)
    normalized_mse = mse_per_channel / (weight_per_channel + eps)
    
    # Average over channels: (B,)
    loss = torch.mean(normalized_mse, dim=-1)
    
    if reduce:
        loss = loss.mean()
    
    return loss


def kl_divergence(
    mu: torch.Tensor,
    *,
    logvar: torch.Tensor = None,
    sigma: torch.Tensor = None
):
    """
    KL divergence loss.
    """

    if logvar is not None:
        loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
    elif sigma is not None:
        loss = (-0.5) * torch.sum(
            1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
        )

    return loss