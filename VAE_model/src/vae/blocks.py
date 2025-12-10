import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import get_padding, SelfAttention


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