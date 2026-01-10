import torch
import torch.nn as nn

from .blocks import ResidualBlock, AttentionBlock
from ..common import get_padding



class Decoder(nn.Module):

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


        # Use asymmetric upsampling to match encoder - preserve depth dimension
        # Only upsample in H and W, NOT in depth D
        # This allows the diffusion model to work accurately in latent space
        self.layers = nn.Sequential(
            # nn.Conv3d(self.in_channels, self.in_channels, kernel_size=1, padding=0),

            nn.Conv3d(self.in_channels, 512, kernel_size=self.kernel_size, padding=padding),

            # AttentionBlock removed - with preserved depth (11×64×64 = 45k positions),
            # the attention matrix would need 15+ GB memory
            # ResidualBlocks are sufficient for VAE reconstruction

            ResidualBlock(in_channels=512, out_channels=512),

            ResidualBlock(in_channels=512, out_channels=512),

            # (B, C, D, H, W) -> (B, C, D, 2*H, 2*W) - preserve depth!
            nn.Upsample(scale_factor=(1, 2, 2)),

            nn.Conv3d(512, 256, kernel_size=self.kernel_size, padding=padding),

            ResidualBlock(in_channels=256, out_channels=256),

            ResidualBlock(in_channels=256, out_channels=256),


            # (B, C, D, 2*H, 2*W) -> (B, C, D, 4*H, 4*W) - preserve depth!
            nn.Upsample(scale_factor=(1, 2, 2)),

            nn.Conv3d(256, 128, kernel_size=self.kernel_size, padding=padding),

            ResidualBlock(in_channels=128, out_channels=128),

            ResidualBlock(in_channels=128, out_channels=128),


            # ResidualBlock(in_channels=128, out_channels=128),

            nn.GroupNorm(num_groups=32, num_channels=128),

            nn.SiLU(),

            nn.Conv3d(128, self.out_channels, kernel_size=self.kernel_size, padding=padding),
        )

        print(f'Trainable parameters: {self.trainable_params}.')

    def forward(self, x: torch.Tensor):

        for module in self.layers:
            x = module(x)

        return x

    @property
    def trainable_params(self):
        total = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        return total

