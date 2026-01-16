"""
Dual-Branch VAE module for 2D→3D flow prediction.

This module provides separate encoders/decoders for 2D and 3D flow with latent alignment.
"""

from .model import DualBranchVAE, kl_divergence

__all__ = ['DualBranchVAE', 'kl_divergence']
