"""
Statistics Generation Script for VAE/Diffusion Training

Computes normalization statistics from the TRAINING SET ONLY to prevent data leakage.
The generated statistics.json is used by both VAE and diffusion training pipelines.

Usage:
    python -m shared.generate_statistics \\
        --dataset-dir /path/to/dataset_3d \\
        --output statistics_train.json \\
        --use-split  # Use existing splits.json

    # Or generate both split and statistics:
    python -m shared.generate_statistics \\
        --dataset-dir /path/to/dataset_3d \\
        --generate-split \\
        --seed 2024

Output Format (statistics.json):
    {
        "U": {
            "max": float,
            "mean": float,
            "std": float
        },
        "U_per_component": {
            "max_u": float, "max_v": float, "max_w": float,
            "mean_u": float, "mean_v": float, "mean_w": float,
            "std_u": float, "std_v": float, "std_w": float,
            "p1_u": float, "p5_u": float, "p50_u": float, "p95_u": float, "p99_u": float,
            ...
        },
        "U_2d": { ... },
        "U_2d_per_component": { ... },
        "p": { ... },
        "metadata": {
            "computed_from": "training_set_only",
            "num_train_samples": int,
            "split_seed": int
        }
    }
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch


def compute_percentiles(tensor: torch.Tensor, percentiles: List[float] = [1, 5, 50, 95, 99]) -> Dict[str, float]:
    """
    Compute percentiles of a tensor.
    
    Args:
        tensor: Input tensor (flattened for computation)
        percentiles: List of percentile values to compute
        
    Returns:
        Dictionary mapping percentile names to values
    """
    # Convert to numpy for large tensors (torch.quantile can fail on very large tensors)
    flat = tensor.flatten().float().cpu().numpy()
    result = {}
    for p in percentiles:
        val = float(np.percentile(flat, p))
        result[f'p{p}'] = val
    return result


def compute_velocity_statistics(
    velocity: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    prefix: str = 'U'
) -> Dict:
    """
    Compute comprehensive statistics for velocity field.
    
    Args:
        velocity: Velocity tensor, shape (N, slices, 3, H, W) for 3D or (N, 3, H, W) for 2D
        mask: Optional mask tensor for fluid region
        prefix: Prefix for stats keys ('U' or 'U_2d')
        
    Returns:
        Dictionary with global and per-component statistics
    """
    stats = {}
    
    # Determine if 3D (has slices dimension)
    is_3d = velocity.dim() == 5
    
    if mask is not None:
        # Expand mask to match velocity shape
        if is_3d:
            # mask: (N, slices, 1, H, W) -> (N, slices, 3, H, W)
            mask_expanded = mask.expand_as(velocity)
        else:
            mask_expanded = mask.expand_as(velocity)
        
        # Apply mask (only compute stats for fluid region)
        velocity_masked = velocity * mask_expanded
    else:
        velocity_masked = velocity
    
    # Global statistics
    stats[prefix] = {
        'max': velocity_masked.abs().max().item(),
        'mean': velocity_masked.mean().item(),
        'std': velocity_masked.std().item(),
        'min': velocity_masked.min().item(),
    }
    
    # Per-component statistics (critical for w-component training)
    pc_stats = {}
    components = ['u', 'v', 'w']
    
    for c_idx, c_name in enumerate(components):
        if is_3d:
            # Shape: (N, slices, 3, H, W) -> select channel c_idx
            component = velocity_masked[:, :, c_idx, :, :]
        else:
            component = velocity_masked[:, c_idx, :, :]
        
        # Basic stats
        pc_stats[f'max_{c_name}'] = component.abs().max().item()
        pc_stats[f'mean_{c_name}'] = component.mean().item()
        pc_stats[f'std_{c_name}'] = component.std().item()
        pc_stats[f'min_{c_name}'] = component.min().item()
        
        # Percentiles for robust normalization
        percentiles = compute_percentiles(component)
        for p_key, p_val in percentiles.items():
            pc_stats[f'{p_key}_{c_name}'] = p_val
        
        # Robust stats (median, MAD)
        flat = component.flatten().cpu().numpy()
        median = float(np.median(flat))
        pc_stats[f'median_{c_name}'] = median
        mad = float(np.median(np.abs(flat - median)))
        pc_stats[f'mad_{c_name}'] = mad
    
    stats[f'{prefix}_per_component'] = pc_stats
    
    return stats


def compute_statistics_from_dataset(
    dataset_dir: str,
    train_indices: List[int],
    use_3d: bool = True
) -> Dict:
    """
    Compute statistics from training samples only.
    
    Args:
        dataset_dir: Path to dataset directory
        train_indices: List of indices for training samples
        use_3d: Whether to load 3D velocity data
        
    Returns:
        Statistics dictionary
    """
    print(f"Loading dataset from {dataset_dir}")
    print(f"Computing statistics from {len(train_indices)} training samples")
    
    # Load raw data files
    velocity_3d_path = os.path.join(dataset_dir, 'x', 'U.pt')
    velocity_2d_path = os.path.join(dataset_dir, 'x', 'U_2d.pt')
    domain_path = os.path.join(dataset_dir, 'x', 'domain.pt')
    pressure_path = os.path.join(dataset_dir, 'x', 'p.pt')
    dxyz_path = os.path.join(dataset_dir, 'x', 'dxyz.pt')
    
    stats = {}
    
    # Load domain/mask for training samples
    if os.path.exists(domain_path):
        domain = torch.load(domain_path)
        domain_train = domain[train_indices]
        print(f"  Domain shape: {domain_train.shape}")
    else:
        domain_train = None
    
    # 3D velocity statistics (U)
    if os.path.exists(velocity_3d_path):
        velocity_3d = torch.load(velocity_3d_path)
        velocity_3d_train = velocity_3d[train_indices]
        print(f"  U (3D velocity) shape: {velocity_3d_train.shape}")
        
        u_stats = compute_velocity_statistics(velocity_3d_train, mask=domain_train, prefix='U')
        stats.update(u_stats)
    
    # 2D velocity statistics (U_2d)
    if os.path.exists(velocity_2d_path):
        velocity_2d = torch.load(velocity_2d_path)
        velocity_2d_train = velocity_2d[train_indices]
        print(f"  U_2d (2D velocity) shape: {velocity_2d_train.shape}")
        
        u2d_stats = compute_velocity_statistics(velocity_2d_train, mask=domain_train, prefix='U_2d')
        stats.update(u2d_stats)
    
    # Pressure statistics
    if os.path.exists(pressure_path):
        pressure = torch.load(pressure_path)
        pressure_train = pressure[train_indices]
        print(f"  Pressure shape: {pressure_train.shape}")
        
        stats['p'] = {
            'max': pressure_train.abs().max().item(),
            'mean': pressure_train.mean().item(),
            'std': pressure_train.std().item(),
        }
    
    # dxyz statistics
    if os.path.exists(dxyz_path):
        dxyz = torch.load(dxyz_path)
        dxyz_train = dxyz[train_indices]
        print(f"  dxyz shape: {dxyz_train.shape}")
        
        stats['dxyz'] = {
            'max': dxyz_train.abs().max().item(),
            'mean': dxyz_train.mean().item(),
        }
    
    return stats


def main():
    """CLI entry point for statistics generation."""
    parser = argparse.ArgumentParser(
        description="Generate normalization statistics from training set only"
    )
    parser.add_argument(
        '--dataset-dir', type=str, required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output', type=str, default='statistics.json',
        help='Output filename (default: statistics.json)'
    )
    parser.add_argument(
        '--split-file', type=str, default='splits.json',
        help='Name of split file to use (default: splits.json)'
    )
    parser.add_argument(
        '--use-split', action='store_true',
        help='Use existing split file'
    )
    parser.add_argument(
        '--generate-split', action='store_true',
        help='Generate new split file before computing statistics'
    )
    parser.add_argument(
        '--seed', type=int, default=2024,
        help='Random seed for split generation (default: 2024)'
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.70,
        help='Training set ratio (default: 0.70)'
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Overwrite existing statistics file'
    )
    
    args = parser.parse_args()
    
    output_path = os.path.join(args.dataset_dir, args.output)
    split_path = os.path.join(args.dataset_dir, args.split_file)
    
    # Check if output already exists
    if os.path.exists(output_path) and not args.force:
        print(f"Statistics file already exists: {output_path}")
        print("Use --force to overwrite")
        return 1
    
    # Determine number of samples
    domain_path = os.path.join(args.dataset_dir, 'x', 'domain.pt')
    if not os.path.exists(domain_path):
        print(f"ERROR: Could not find {domain_path}")
        return 1
    
    domain = torch.load(domain_path)
    num_samples = domain.shape[0]
    print(f"Found {num_samples} samples in dataset")
    
    # Get or create split
    try:
        from shared.data_split import create_split, load_split, save_split
    except ModuleNotFoundError:
        # Running as direct script, use relative import
        from data_split import create_split, load_split, save_split
    
    if args.generate_split or not os.path.exists(split_path):
        print(f"Generating new split with seed={args.seed}")
        split = create_split(
            num_samples,
            args.train_ratio,
            args.val_ratio,
            1.0 - args.train_ratio - args.val_ratio,
            args.seed
        )
        save_split(split, split_path)
    elif args.use_split:
        print(f"Loading existing split from {split_path}")
        split = load_split(split_path)
    else:
        print("ERROR: No split specified. Use --use-split or --generate-split")
        return 1
    
    train_indices = split['train']
    print(f"Training set: {len(train_indices)} samples")
    print(f"Validation set: {len(split['val'])} samples")
    print(f"Test set: {len(split['test'])} samples")
    
    # Compute statistics
    stats = compute_statistics_from_dataset(args.dataset_dir, train_indices)
    
    # Add metadata
    stats['metadata'] = {
        'computed_from': 'training_set_only',
        'num_train_samples': len(train_indices),
        'num_total_samples': num_samples,
        'split_seed': split.get('metadata', {}).get('seed', args.seed),
        'split_file': args.split_file,
        'description': 'Statistics computed from training set only to prevent data leakage'
    }
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Saved statistics to {output_path}")
    
    # Print summary
    print("\nStatistics summary:")
    if 'U_per_component' in stats:
        pc = stats['U_per_component']
        print(f"  U (3D velocity):")
        print(f"    max_u={pc['max_u']:.6f}, max_v={pc['max_v']:.6f}, max_w={pc['max_w']:.6f}")
        print(f"    std_u={pc['std_u']:.6f}, std_v={pc['std_v']:.6f}, std_w={pc['std_w']:.6f}")
    
    if 'U_2d_per_component' in stats:
        pc = stats['U_2d_per_component']
        print(f"  U_2d (2D velocity):")
        print(f"    max_u={pc['max_u']:.6f}, max_v={pc['max_v']:.6f}, max_w={pc['max_w']:.6f}")
    
    return 0


if __name__ == '__main__':
    exit(main())
