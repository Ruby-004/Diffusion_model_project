"""
Unified Data Split Module for VAE and Diffusion Training

This module provides a single source of truth for train/val/test splits
across both VAE and diffusion model training pipelines.

Features:
- Deterministic splits with fixed seed
- Persistence to JSON for reproducibility
- Graceful handling of dataset changes (e.g., removing zero samples)
- Support for paired sampling (2D/3D from same microstructure)

Usage:
    # Generate and save split
    python -m shared.data_split --generate --dataset-dir /path/to/dataset --output splits.json

    # Use in training code
    from shared.data_split import get_or_create_split, get_split_indices
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random


# Default configuration
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SEED = 2024
DEFAULT_SPLIT_FILENAME = "splits.json"


def compute_sample_ids(num_samples: int, id_prefix: str = "sample") -> List[str]:
    """
    Generate stable sample IDs based on index.
    
    Args:
        num_samples: Number of samples in dataset
        id_prefix: Prefix for sample IDs
        
    Returns:
        List of sample IDs
    """
    return [f"{id_prefix}_{i:06d}" for i in range(num_samples)]


def create_split(
    num_samples: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
    sample_ids: Optional[List[str]] = None
) -> Dict[str, List[Union[int, str]]]:
    """
    Create a deterministic train/val/test split.
    
    Args:
        num_samples: Total number of samples
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for test (default 0.15)
        seed: Random seed for reproducibility
        sample_ids: Optional list of sample identifiers (for stability under dataset changes)
        
    Returns:
        Dictionary with 'train', 'val', 'test' lists of indices (or IDs if provided)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Generate sample IDs if not provided
    if sample_ids is None:
        sample_ids = list(range(num_samples))
        use_indices = True
    else:
        use_indices = False
        assert len(sample_ids) == num_samples, \
            f"sample_ids length {len(sample_ids)} != num_samples {num_samples}"
    
    # Shuffle with fixed seed
    rng = random.Random(seed)
    shuffled_indices = list(range(num_samples))
    rng.shuffle(shuffled_indices)
    
    # Compute split sizes
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size
    
    # Split indices
    train_indices = sorted(shuffled_indices[:train_size])
    val_indices = sorted(shuffled_indices[train_size:train_size + val_size])
    test_indices = sorted(shuffled_indices[train_size + val_size:])
    
    # Convert to IDs if provided
    if use_indices:
        split = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
            'metadata': {
                'num_samples': num_samples,
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': test_ratio,
                'seed': seed,
                'type': 'index_based'
            }
        }
    else:
        split = {
            'train': [sample_ids[i] for i in train_indices],
            'val': [sample_ids[i] for i in val_indices],
            'test': [sample_ids[i] for i in test_indices],
            'metadata': {
                'num_samples': num_samples,
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': test_ratio,
                'seed': seed,
                'type': 'id_based'
            }
        }
    
    return split


def save_split(split: Dict, filepath: str) -> None:
    """
    Save split to JSON file.
    
    Args:
        split: Split dictionary
        filepath: Path to save JSON
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(split, f, indent=2)
    print(f"Saved split to {filepath}")


def load_split(filepath: str) -> Dict:
    """
    Load split from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Split dictionary
    """
    with open(filepath, 'r') as f:
        split = json.load(f)
    return split


def get_or_create_split(
    dataset_dir: str,
    num_samples: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
    split_filename: str = DEFAULT_SPLIT_FILENAME,
    force_recreate: bool = False,
    filter_indices: Optional[List[int]] = None
) -> Dict[str, List[int]]:
    """
    Get existing split or create new one.
    
    This is the main entry point for both VAE and diffusion training.
    
    Args:
        dataset_dir: Directory containing dataset
        num_samples: Total number of samples in dataset
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed
        split_filename: Name of split file
        force_recreate: If True, recreate even if file exists
        filter_indices: Optional list of indices to keep (e.g., after removing zero samples)
        
    Returns:
        Dictionary with 'train', 'val', 'test' lists of indices
    """
    split_path = os.path.join(dataset_dir, split_filename)
    
    # Try to load existing split
    if os.path.exists(split_path) and not force_recreate:
        print(f"Loading existing split from {split_path}")
        split = load_split(split_path)
        
        # Validate split matches current dataset
        meta = split.get('metadata', {})
        stored_num = meta.get('num_samples', -1)
        
        if filter_indices is not None:
            # Filter split to only include indices that still exist
            filter_set = set(filter_indices)
            split = {
                'train': [i for i in split['train'] if i in filter_set],
                'val': [i for i in split['val'] if i in filter_set],
                'test': [i for i in split['test'] if i in filter_set],
                'metadata': meta
            }
            print(f"Filtered split: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
            
            # Re-index if filter was applied
            if filter_indices:
                # Create mapping from old indices to new indices
                old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(filter_indices))}
                split = {
                    'train': [old_to_new[i] for i in split['train'] if i in old_to_new],
                    'val': [old_to_new[i] for i in split['val'] if i in old_to_new],
                    'test': [old_to_new[i] for i in split['test'] if i in old_to_new],
                    'metadata': meta
                }
        elif stored_num != num_samples:
            print(f"WARNING: Split file has {stored_num} samples but dataset has {num_samples}")
            print("Recreating split...")
            split = create_split(num_samples, train_ratio, val_ratio, test_ratio, seed)
            save_split(split, split_path)
            
        return split
    
    # Create new split
    print(f"Creating new split with seed={seed}")
    effective_num_samples = len(filter_indices) if filter_indices else num_samples
    
    split = create_split(effective_num_samples, train_ratio, val_ratio, test_ratio, seed)
    save_split(split, split_path)
    
    return split


def get_split_indices(
    split: Dict,
    subset: str
) -> List[int]:
    """
    Get indices for a specific subset.
    
    Args:
        split: Split dictionary
        subset: One of 'train', 'val', 'test'
        
    Returns:
        List of indices
    """
    assert subset in ['train', 'val', 'test'], f"Invalid subset: {subset}"
    return split[subset]


def create_paired_split_for_vae(
    num_microstructures: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED
) -> Dict[str, List[int]]:
    """
    Create split for VAE dataset where 2D and 3D samples from same microstructure
    are kept together.
    
    In MicroFlowDatasetVAE:
    - Indices 0 to N-1: 2D samples (U_2d)
    - Indices N to 2N-1: 3D samples (U)
    - Sample i and sample i+N come from the same microstructure
    
    Args:
        num_microstructures: Number of microstructures (N)
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' lists of paired indices
    """
    # First, split the microstructure indices
    base_split = create_split(
        num_microstructures,
        train_ratio,
        val_ratio,
        test_ratio,
        seed
    )
    
    # Create paired indices (both 2D and 3D from same microstructure)
    def expand_to_paired(indices: List[int], n: int) -> List[int]:
        """Expand base indices to include both 2D (idx) and 3D (idx + n) versions"""
        return indices + [i + n for i in indices]
    
    paired_split = {
        'train': expand_to_paired(base_split['train'], num_microstructures),
        'val': expand_to_paired(base_split['val'], num_microstructures),
        'test': expand_to_paired(base_split['test'], num_microstructures),
        'metadata': {
            **base_split['metadata'],
            'type': 'paired_vae',
            'num_microstructures': num_microstructures
        }
    }
    
    return paired_split


def get_3d_only_split(
    paired_split: Dict,
    num_microstructures: int
) -> Dict[str, List[int]]:
    """
    Extract 3D-only indices from a paired split.
    
    Used for Stage 1 VAE training which only uses 3D samples.
    
    Args:
        paired_split: Paired split from create_paired_split_for_vae
        num_microstructures: N (number of microstructures)
        
    Returns:
        Dictionary with indices pointing to 3D samples only (indices >= N)
    """
    def filter_3d_indices(indices: List[int]) -> List[int]:
        """Keep only indices >= num_microstructures (3D samples)"""
        return [i for i in indices if i >= num_microstructures]
    
    return {
        'train': filter_3d_indices(paired_split['train']),
        'val': filter_3d_indices(paired_split['val']),
        'test': filter_3d_indices(paired_split['test']),
        'metadata': {
            **paired_split.get('metadata', {}),
            'type': '3d_only_from_paired'
        }
    }


def verify_split_consistency(
    vae_split_path: str,
    diffusion_split_path: str
) -> bool:
    """
    Verify that VAE and diffusion use consistent splits.
    
    Args:
        vae_split_path: Path to VAE split file
        diffusion_split_path: Path to diffusion split file
        
    Returns:
        True if splits are consistent
    """
    vae_split = load_split(vae_split_path)
    diff_split = load_split(diffusion_split_path)
    
    # Check metadata
    vae_meta = vae_split.get('metadata', {})
    diff_meta = diff_split.get('metadata', {})
    
    if vae_meta.get('seed') != diff_meta.get('seed'):
        print(f"WARNING: Different seeds: VAE={vae_meta.get('seed')}, Diffusion={diff_meta.get('seed')}")
        return False
    
    # For diffusion (which uses raw microstructure indices),
    # extract base indices from VAE paired split
    vae_type = vae_meta.get('type', '')
    if vae_type == 'paired_vae':
        num_micro = vae_meta.get('num_microstructures', 0)
        # Extract base microstructure indices from 3D portion
        vae_train_base = sorted([i - num_micro for i in vae_split['train'] if i >= num_micro])
        vae_val_base = sorted([i - num_micro for i in vae_split['val'] if i >= num_micro])
        vae_test_base = sorted([i - num_micro for i in vae_split['test'] if i >= num_micro])
    else:
        vae_train_base = sorted(vae_split['train'])
        vae_val_base = sorted(vae_split['val'])
        vae_test_base = sorted(vae_split['test'])
    
    diff_train = sorted(diff_split['train'])
    diff_val = sorted(diff_split['val'])
    diff_test = sorted(diff_split['test'])
    
    # Compare
    train_match = vae_train_base == diff_train
    val_match = vae_val_base == diff_val
    test_match = vae_test_base == diff_test
    
    if not all([train_match, val_match, test_match]):
        print("Split mismatch detected!")
        print(f"  Train match: {train_match}")
        print(f"  Val match: {val_match}")
        print(f"  Test match: {test_match}")
        return False
    
    print("✓ Splits are consistent between VAE and diffusion")
    return True


def main():
    """CLI entry point for split generation."""
    parser = argparse.ArgumentParser(
        description="Generate or verify data splits for VAE and diffusion training"
    )
    parser.add_argument(
        '--dataset-dir', type=str, required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--generate', action='store_true',
        help='Generate new split file'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify existing split'
    )
    parser.add_argument(
        '--output', type=str, default=DEFAULT_SPLIT_FILENAME,
        help=f'Output filename (default: {DEFAULT_SPLIT_FILENAME})'
    )
    parser.add_argument(
        '--train-ratio', type=float, default=DEFAULT_TRAIN_RATIO,
        help=f'Training set ratio (default: {DEFAULT_TRAIN_RATIO})'
    )
    parser.add_argument(
        '--val-ratio', type=float, default=DEFAULT_VAL_RATIO,
        help=f'Validation set ratio (default: {DEFAULT_VAL_RATIO})'
    )
    parser.add_argument(
        '--test-ratio', type=float, default=DEFAULT_TEST_RATIO,
        help=f'Test set ratio (default: {DEFAULT_TEST_RATIO})'
    )
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_SEED,
        help=f'Random seed (default: {DEFAULT_SEED})'
    )
    parser.add_argument(
        '--num-samples', type=int, default=None,
        help='Number of samples (auto-detected if not provided)'
    )
    parser.add_argument(
        '--paired-vae', action='store_true',
        help='Create paired split for VAE (keeps 2D/3D from same microstructure together)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force recreate even if split exists'
    )
    
    args = parser.parse_args()
    
    if args.generate:
        # Auto-detect num_samples from dataset if not provided
        if args.num_samples is None:
            # Try to load domain.pt to count samples
            domain_path = os.path.join(args.dataset_dir, 'x', 'domain.pt')
            if os.path.exists(domain_path):
                import torch
                domain = torch.load(domain_path)
                args.num_samples = domain.shape[0]
                print(f"Auto-detected {args.num_samples} samples from {domain_path}")
            else:
                raise ValueError("Could not auto-detect num_samples. Please provide --num-samples")
        
        if args.paired_vae:
            split = create_paired_split_for_vae(
                args.num_samples,
                args.train_ratio,
                args.val_ratio,
                args.test_ratio,
                args.seed
            )
        else:
            split = create_split(
                args.num_samples,
                args.train_ratio,
                args.val_ratio,
                args.test_ratio,
                args.seed
            )
        
        output_path = os.path.join(args.dataset_dir, args.output)
        save_split(split, output_path)
        
        print(f"\nSplit summary:")
        print(f"  Train: {len(split['train'])} samples")
        print(f"  Val: {len(split['val'])} samples")
        print(f"  Test: {len(split['test'])} samples")
        
    elif args.verify:
        split_path = os.path.join(args.dataset_dir, args.output)
        if not os.path.exists(split_path):
            print(f"ERROR: Split file not found: {split_path}")
            return 1
        
        split = load_split(split_path)
        print(f"Split file: {split_path}")
        print(f"  Train: {len(split['train'])} samples")
        print(f"  Val: {len(split['val'])} samples")
        print(f"  Test: {len(split['test'])} samples")
        print(f"  Metadata: {split.get('metadata', {})}")
        
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
