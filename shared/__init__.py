"""
Shared utilities for VAE and Diffusion model training.

This package provides:
- data_split: Unified train/val/test split mechanism
- statistics: Training set statistics computation
"""

from .data_split import (
    get_or_create_split,
    get_split_indices,
    create_split,
    create_paired_split_for_vae,
    get_3d_only_split,
    load_split,
    save_split,
    verify_split_consistency,
    DEFAULT_SEED,
    DEFAULT_SPLIT_FILENAME
)

__all__ = [
    'get_or_create_split',
    'get_split_indices', 
    'create_split',
    'create_paired_split_for_vae',
    'get_3d_only_split',
    'load_split',
    'save_split',
    'verify_split_consistency',
    'DEFAULT_SEED',
    'DEFAULT_SPLIT_FILENAME'
]
