import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock, ConditionalResidualBlock, FiLM, AttentionBlock
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
        
        # Use asymmetric upsampling to match encoder - preserve depth dimension
        # Only upsample in H and W, NOT in depth D
        
        # Initial conv from latent
        self.conv_in = nn.Conv3d(self.in_channels, 512, kernel_size=self.kernel_size, padding=padding)
        
        # FiLM conditioning after initial conv (when conditional=True)
        if self.conditional:
            self.film_in = FiLM(condition_dim=1, feature_channels=512)
        
        # Stage 1: 512 channels
        if self.conditional:
            self.res1_1 = ConditionalResidualBlock(512, 512, conditional=True, condition_dim=1)
            self.res1_2 = ConditionalResidualBlock(512, 512, conditional=True, condition_dim=1)
        else:
            self.res1_1 = ResidualBlock(512, 512)
            self.res1_2 = ResidualBlock(512, 512)
        
        # Upsample 1: 2*H, 2*W
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2))
        self.conv_up1 = nn.Conv3d(512, 256, kernel_size=self.kernel_size, padding=padding)
        
        # Stage 2: 256 channels
        if self.conditional:
            self.res2_1 = ConditionalResidualBlock(256, 256, conditional=True, condition_dim=1)
            self.res2_2 = ConditionalResidualBlock(256, 256, conditional=True, condition_dim=1)
        else:
            self.res2_1 = ResidualBlock(256, 256)
            self.res2_2 = ResidualBlock(256, 256)
        
        # Upsample 2: 4*H, 4*W
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2))
        self.conv_up2 = nn.Conv3d(256, 128, kernel_size=self.kernel_size, padding=padding)
        
        # Stage 3: 128 channels
        if self.conditional:
            self.res3_1 = ConditionalResidualBlock(128, 128, conditional=True, condition_dim=1)
            self.res3_2 = ConditionalResidualBlock(128, 128, conditional=True, condition_dim=1)
        else:
            self.res3_1 = ResidualBlock(128, 128)
            self.res3_2 = ResidualBlock(128, 128)
        
        # Final layers
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=128)
        self.conv_out = nn.Conv3d(128, self.out_channels, kernel_size=self.kernel_size, padding=padding)
        
        # Final FiLM before output (can help generate condition-specific outputs)
        if self.conditional:
            self.film_pre_out = FiLM(condition_dim=1, feature_channels=128)

        print(f'Trainable parameters: {self.trainable_params}.')

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor = None  # (B,) boolean tensor: True=3D flow, False=2D flow
    ):
        """
        Forward pass through decoder with FiLM conditioning at multiple layers.
        
        Args:
            x: Latent representation (B, latent_channels, D, H/4, W/4)
            condition: Optional boolean tensor (B,) where True=3D flow (U), False=2D flow (U_2d)
                       Only used if self.conditional=True
        """
        # Initial conv
        x = self.conv_in(x)
        if self.conditional and condition is not None:
            x = self.film_in(x, condition)
        
        # Stage 1
        if self.conditional and condition is not None:
            x = self.res1_1(x, condition)
            x = self.res1_2(x, condition)
        else:
            x = self.res1_1(x)
            x = self.res1_2(x)
        
        # Upsample 1
        x = self.up1(x)
        x = self.conv_up1(x)
        
        # Stage 2
        if self.conditional and condition is not None:
            x = self.res2_1(x, condition)
            x = self.res2_2(x, condition)
        else:
            x = self.res2_1(x)
            x = self.res2_2(x)
        
        # Upsample 2
        x = self.up2(x)
        x = self.conv_up2(x)
        
        # Stage 3
        if self.conditional and condition is not None:
            x = self.res3_1(x, condition)
            x = self.res3_2(x, condition)
        else:
            x = self.res3_1(x)
            x = self.res3_2(x)
        
        # Final layers
        x = self.norm_out(x)
        x = F.silu(x)
        
        # Apply FiLM before final conv for condition-specific output
        if self.conditional and condition is not None:
            x = self.film_pre_out(x, condition)
        
        x = self.conv_out(x)

        return x

    @property
    def trainable_params(self):
        total = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        return total

