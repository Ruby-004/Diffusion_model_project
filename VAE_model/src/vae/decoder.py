import torch
import torch.nn as nn

from .blocks import ResidualBlock, AttentionBlock
from ..common import get_padding



class Decoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        conditional: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conditional = conditional
        padding = get_padding(self.kernel_size)
        
        # Condition embedding: maps is_3d (0 or 1) to a per-channel bias
        # This allows the decoder to specialize output for 2D vs 3D flow
        if self.conditional:
            # Embed condition into same number of channels as first conv output
            self.cond_embed = nn.Sequential(
                nn.Linear(1, 64),
                nn.SiLU(),
                nn.Linear(64, 512)  # Output matches first conv layer's output channels
            )


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

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor = None  # (B,) boolean tensor: True=3D flow, False=2D flow
    ):
        """
        Forward pass through decoder.
        
        Args:
            x: Latent representation (B, latent_channels, D, H/4, W/4)
            condition: Optional boolean tensor (B,) where True=3D flow (U), False=2D flow (U_2d)
                       Only used if self.conditional=True
        """
        first_layer = True
        for module in self.layers:
            x = module(x)
            
            # Inject condition after first conv layer
            if first_layer and self.conditional and condition is not None:
                first_layer = False
                # condition: (B,) -> (B, 1) -> embed -> (B, 512) -> (B, 512, 1, 1, 1)
                cond_float = condition.float().unsqueeze(-1)  # (B, 1)
                cond_bias = self.cond_embed(cond_float)  # (B, 512)
                cond_bias = cond_bias.view(x.shape[0], -1, 1, 1, 1)  # (B, 512, 1, 1, 1)
                x = x + cond_bias  # Broadcast across D, H, W
        
        # Explicitly zero out w component (channel 2) for 2D flow
        # This ensures w=0 when condition=False (is_2d=True)
        if self.conditional and condition is not None:
            # condition is True for 3D flow, False for 2D flow
            # For 2D flow (condition=False), set w channel to zero
            # x shape: (B, 3, D, H, W) where channels are [u, v, w]
            mask_2d = ~condition  # True where flow is 2D (w should be 0)
            if mask_2d.any():
                # Zero out w channel (index 2) for 2D samples
                x[mask_2d, 2, :, :, :] = 0.0

        return x

    @property
    def trainable_params(self):
        total = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        return total

