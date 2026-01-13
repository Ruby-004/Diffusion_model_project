import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import get_padding, SelfAttention


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for conditioning.
    
    Applies affine transformation: γ * x + β
    where γ (scale) and β (shift) are learned functions of the condition.
    
    This is much more expressive than simple additive bias because:
    1. Scale (γ) can amplify or suppress features
    2. Shift (β) adds bias
    3. Together they can implement complex conditional transformations
    """
    
    def __init__(self, condition_dim: int, feature_channels: int, hidden_dim: int = 128):
        super().__init__()
        
        # MLP to generate γ (scale) and β (shift) from condition
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * feature_channels)  # γ and β
        )
        
        # Initialize: γ starts at 1 (identity scale), β starts at 0 (no shift)
        # But use small random weights for the last layer so gradients can flow
        # and condition=0 vs condition=1 produce different outputs
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.1)
        # Set bias for γ (first half) to 1.0 for identity transform baseline
        self.mlp[-1].bias.data[:feature_channels] = 1.0
        # Set bias for β (second half) to 0.0
        self.mlp[-1].bias.data[feature_channels:] = 0.0
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        Apply FiLM conditioning.
        
        Args:
            x: Feature tensor (B, C, D, H, W) or (B, C, H, W)
            condition: Condition tensor (B, condition_dim) or (B,) for scalar condition
        
        Returns:
            Conditioned features
        """
        # Handle scalar condition (B,) -> (B, 1)
        if condition.dim() == 1:
            condition = condition.float().unsqueeze(-1)
        
        # Get γ and β from condition
        params = self.mlp(condition)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)  # Each: (B, C)
        
        # Reshape for broadcasting
        if x.dim() == 5:  # 3D: (B, C, D, H, W)
            gamma = gamma.view(gamma.shape[0], -1, 1, 1, 1)
            beta = beta.view(beta.shape[0], -1, 1, 1, 1)
        elif x.dim() == 4:  # 2D: (B, C, H, W)
            gamma = gamma.view(gamma.shape[0], -1, 1, 1)
            beta = beta.view(beta.shape[0], -1, 1, 1)
        
        return gamma * x + beta


class ConditionalResidualBlock(nn.Module):
    """
    ResidualBlock with FiLM conditioning support.
    
    If condition is provided, applies FiLM after each conv layer.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        conditional: bool = False,
        condition_dim: int = 1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conditional = conditional
        padding = get_padding(self.kernel_size)

        # layers
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=self.in_channels)
        self.conv1 = nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size, padding=padding)
 
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, self.kernel_size, padding=padding)

        if self.in_channels != self.out_channels:
            self.residual_layer = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
        else:
            self.residual_layer = nn.Identity()
        
        # FiLM conditioning after each conv
        if self.conditional:
            self.film1 = FiLM(condition_dim, out_channels)
            self.film2 = FiLM(condition_dim, out_channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None):
        
        x_in = x

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        # Apply FiLM after first conv
        if self.conditional and condition is not None:
            x = self.film1(x, condition)

        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        # Apply FiLM after second conv
        if self.conditional and condition is not None:
            x = self.film2(x, condition)

        out = x + self.residual_layer(x_in)
        return out


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        padding = get_padding(self.kernel_size)

        # layers
        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=self.in_channels
        )
        self.conv1 = nn.Conv3d(
            self.in_channels, self.out_channels, self.kernel_size, padding=padding
        )
 
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=self.out_channels
        )
        self.conv2 = nn.Conv3d(
            self.out_channels, self.out_channels, self.kernel_size, padding=padding
        )

        if self.in_channels != self.out_channels:
            self.residual_layer = nn.Conv3d(
                self.in_channels, self.out_channels, kernel_size=1, padding=0
            )
        else:
            self.residual_layer = nn.Identity()

    def forward(self, x: torch.Tensor):
        
        x_in = x

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        out = x + self.residual_layer(x_in)
        return out


class AttentionBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 2
    ):
        super().__init__()
    
        self.in_channels = in_channels
        self.num_heads = num_heads


        self.norm = nn.GroupNorm(32, self.in_channels)
        self.attention = SelfAttention(
            num_heads=self.num_heads, embed_dim=self.in_channels
        )

    def forward(self, x: torch.Tensor):

        x_in = x
        n, c, d, h, w = x.shape

        x = self.norm(x)
        # (B, C, D, H, W) -> (B, C, D * H * W)
        x = x.view(n, c, d * h * w)
        
        # (B, C, D * H * W) -> (B, D * H * W, C)
        x = x.transpose(-1, -2)

        x = self.attention(x)

        # Re-arrange back
        # (B, D * H * W, C) -> (B, C, D * H * W)
        x = x.transpose(-1, -2)
        x = x.view((n, c, d, h, w))
        
        x += x_in
        return x