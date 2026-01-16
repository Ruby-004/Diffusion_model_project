"""
Custom sampler for DualBranchVAE that ensures paired 2D/3D samples in each batch.
"""

import torch
from torch.utils.data import Sampler
from typing import Iterator
import random


class PairedBatchSampler(Sampler):
    """
    Sampler that creates batches with paired 2D and 3D samples from the same microstructure.
    
    For MicroFlowDatasetVAE:
    - Indices 0 to N-1: 2D flow samples
    - Indices N to 2N-1: 3D flow samples
    - Index i and i+N come from the same microstructure
    
    This sampler creates batches like:
    - [i, i+N] → paired 2D and 3D from same microstructure
    - Ensures alignment and cross-reconstruction losses can always be computed
    """
    
    def __init__(self, num_base_samples: int, batch_size: int, shuffle: bool = True, seed: int = None):
        """
        Args:
            num_base_samples: Number of unique microstructures (dataset size / 2)
            batch_size: Must be even (half 2D, half 3D)
            shuffle: Whether to shuffle the order of paired batches
            seed: Random seed for reproducibility
        """
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even for paired sampling, got {batch_size}")
        
        self.num_base_samples = num_base_samples
        self.batch_size = batch_size
        self.pairs_per_batch = batch_size // 2
        self.shuffle = shuffle
        self.seed = seed
        
        # Total number of batches
        self.num_batches = (num_base_samples + self.pairs_per_batch - 1) // self.pairs_per_batch
    
    def __iter__(self) -> Iterator[list[int]]:
        # Create list of base indices (0 to N-1)
        base_indices = list(range(self.num_base_samples))
        
        if self.shuffle:
            if self.seed is not None:
                rng = random.Random(self.seed)
                rng.shuffle(base_indices)
            else:
                random.shuffle(base_indices)
        
        # Generate batches
        for batch_start in range(0, self.num_base_samples, self.pairs_per_batch):
            batch_end = min(batch_start + self.pairs_per_batch, self.num_base_samples)
            batch_base = base_indices[batch_start:batch_end]
            
            # Create paired batch: [2D indices, 3D indices]
            batch_indices = []
            for base_idx in batch_base:
                batch_indices.append(base_idx)  # 2D sample
                batch_indices.append(base_idx + self.num_base_samples)  # 3D sample
            
            yield batch_indices
    
    def __len__(self) -> int:
        return self.num_batches


class StratifiedPairedBatchSampler(Sampler):
    """
    Sampler that creates stratified train/val/test splits with paired batches.
    
    Ensures that:
    1. Same microstructures stay in same split (2D and 3D together)
    2. Each batch contains pairs from the same microstructure
    3. Splits are reproducible with seed
    """
    
    def __init__(
        self, 
        num_base_samples: int, 
        batch_size: int,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
        seed: int = 2024
    ):
        """
        Args:
            num_base_samples: Number of unique microstructures
            batch_size: Must be even
            split: 'train', 'val', or 'test'
            train_ratio, val_ratio, test_ratio: Split ratios
            shuffle: Whether to shuffle within split
            seed: Random seed for reproducibility
        """
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even for paired sampling, got {batch_size}")
        
        self.num_base_samples = num_base_samples
        self.batch_size = batch_size
        self.pairs_per_batch = batch_size // 2
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        
        # Split base indices
        rng = random.Random(seed)
        base_indices = list(range(num_base_samples))
        rng.shuffle(base_indices)
        
        train_size = int(train_ratio * num_base_samples)
        val_size = int(val_ratio * num_base_samples)
        
        if split == 'train':
            self.split_indices = base_indices[:train_size]
        elif split == 'val':
            self.split_indices = base_indices[train_size:train_size + val_size]
        elif split == 'test':
            self.split_indices = base_indices[train_size + val_size:]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")
        
        self.num_batches = (len(self.split_indices) + self.pairs_per_batch - 1) // self.pairs_per_batch
        
        print(f"{split.upper()} split: {len(self.split_indices)} microstructures, {self.num_batches} batches")
    
    def __iter__(self) -> Iterator[list[int]]:
        split_indices = self.split_indices.copy()
        
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(split_indices)
        
        # Generate batches
        for batch_start in range(0, len(split_indices), self.pairs_per_batch):
            batch_end = min(batch_start + self.pairs_per_batch, len(split_indices))
            batch_base = split_indices[batch_start:batch_end]
            
            # Create paired batch
            batch_indices = []
            for base_idx in batch_base:
                batch_indices.append(base_idx)
                batch_indices.append(base_idx + self.num_base_samples)
            
            yield batch_indices
    
    def __len__(self) -> int:
        return self.num_batches
