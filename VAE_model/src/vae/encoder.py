import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock, AttentionBlock
from ..common import get_padding, SelfAttention


class Encoder(nn.Module):

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

        # Use asymmetric stride to preserve depth dimension for z-flow accuracy
        # Only downsample in H and W, NOT in depth D
        # This allows the diffusion model to work accurately in latent space
        self.layers = nn.Sequential(
            nn.Conv3d(self.in_channels, 128, kernel_size=self.kernel_size, padding=padding),

            ResidualBlock(in_channels=128, out_channels=128),

            ResidualBlock(in_channels=128, out_channels=128),

            # (B, C, D, H, W) -> (B, C, D, H/2, W/2) - preserve depth!
            nn.Conv3d(128, 128, kernel_size=self.kernel_size, stride=(1, 2, 2), padding=0),

            ResidualBlock(in_channels=128, out_channels=256),

            ResidualBlock(in_channels=256, out_channels=256),

            # (B, C, D, H/2, W/2) -> (B, C, D, H/4, W/4) - preserve depth!
            nn.Conv3d(256, 256, kernel_size=self.kernel_size, stride=(1, 2, 2), padding=0),

            ResidualBlock(in_channels=256, out_channels=512),

            ResidualBlock(in_channels=512, out_channels=512),


            AttentionBlock(in_channels=512),

            nn.GroupNorm(num_groups=32, num_channels=512),

            nn.SiLU(),

            nn.Conv3d(512, 2*self.out_channels, kernel_size=self.kernel_size, padding=padding),

            # nn.Conv3d(2*self.out_channels, 2*self.out_channels, kernel_size=1, padding=0)
        )

        print(f'Trainable parameters: {self.trainable_params}.')

    def forward(
        self,
        x: torch.Tensor, # (B, in_channels, D, H, W)
        # noise: torch.Tensor # (B, out_channels, D, H/4, W/4) - depth preserved!
    ):

        for module in self.layers:
            if getattr(module, 'stride', None) == (1, 2, 2):
                # for layers where we down-scale with stride (1,2,2),
                # asymmetrically pad height and width for stride-2, and depth for kernel-3
                # Depth padding: 1 on each side to maintain depth dimension with kernel_size=3
                # H/W padding: asymmetric (0,1) for stride-2 downsampling
                left, right, top, bottom, front, back = 0, 1, 0, 1, 1, 1
                x = F.pad(x, (left, right, top, bottom, front, back))

            x = module(x)
        
        # from (B, C, d, h, w) -> 2 tensors of shape (B, C/2, d, h, w)
        mu, log_var = torch.chunk(x, 2, dim=1)

        # var = log_var.exp()
        # std = var.sqrt()
        # out = mu + std * noise
        return mu, log_var

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps * std
    
    @property
    def trainable_params(self):
        total = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        return total



