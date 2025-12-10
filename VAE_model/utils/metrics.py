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