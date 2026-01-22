#!/usr/bin/env python3
"""
End-to-end evaluation script for latent diffusion pipeline.

This script supports TWO evaluation modes:

1. TRUE END-TO-END DIFFUSION EVALUATION (default):
    2D input → E2D encoder → diffusion sampling → D3D decoder → compare vs GT
    This measures REAL model performance.

2. VAE-ONLY SANITY CHECK (--sanity-mode):
    3D GT → E3D encoder → D3D decoder → compare vs GT
    This measures VAE reconstruction quality ONLY, NOT diffusion performance.
    ⚠️  WARNING: Sanity mode does NOT evaluate the diffusion model!

Metrics computed:
    - Per-component MAE/MSE/RMSE for u, v, w velocity components
    - Normalized MAE/MSE (using dataset statistics)
    - Cosine similarity between velocity vectors
    - IoU of top-k magnitude voxels (structure agreement)
    - Combined accuracy score (only meaningful in diffusion mode)

Outputs:
    - JSON file with aggregated metrics (mean ± std)
    - Optional CSV file with per-sample metrics
    - Print summary with clear mode labeling

Usage (End-to-End Diffusion):
    python scripts/eval_testset_end2end.py \\
        --diffusion-model-path Diffusion_model/trained/normal_mse \\
        --vae-encoder-path VAE_model/trained/dual_vae_stage2_2d \\
        --vae-decoder-path VAE_model/trained/dual_vae_stage1_3d \\
        --dataset-dir /path/to/dataset_3d \\
        --sampler ddim --steps 50

Usage (VAE Sanity Check):
    python scripts/eval_testset_end2end.py \\
        --diffusion-model-path Diffusion_model/trained/normal_mse \\
        --vae-encoder-path ... --vae-decoder-path ... \\
        --dataset-dir /path/to/dataset_3d \\
        --sanity-mode

Author: GitHub Copilot
Date: January 2026
"""

import argparse
import sys
import os
import os.path as osp
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root and Diffusion_model to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

diffusion_model_path = os.path.join(project_root, 'Diffusion_model')
if diffusion_model_path not in sys.path:
    sys.path.insert(0, diffusion_model_path)

from Diffusion_model.src.helper import set_model
from Diffusion_model.utils.dataset import get_loader


# ============================================================================
# Metric Computation Functions
# ============================================================================

def compute_mae_per_component(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[float, float, float]:
    """
    Compute Mean Absolute Error per velocity component.
    
    Args:
        y_pred: Predicted velocity, shape (batch, slices, 3, H, W) or (slices, 3, H, W)
        y_true: Ground truth velocity, same shape as y_pred
        mask: Optional binary mask for fluid region, shape (batch, slices, 1, H, W)
        
    Returns:
        mae_u, mae_v, mae_w: Per-component MAE values
    """
    # Ensure 5D tensors
    if y_pred.dim() == 4:
        y_pred = y_pred.unsqueeze(0)
        y_true = y_true.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    
    # Extract components: (batch, slices, 3, H, W) -> component is dim 2
    err = (y_pred - y_true).abs()
    
    if mask is not None:
        # Expand mask to match velocity channels
        mask = mask.expand_as(err)
        # Compute mean only over fluid region
        num_fluid = mask.sum()
        if num_fluid > 0:
            mae_u = (err[:, :, 0, :, :] * mask[:, :, 0, :, :]).sum() / (mask[:, :, 0, :, :].sum() + 1e-8)
            mae_v = (err[:, :, 1, :, :] * mask[:, :, 1, :, :]).sum() / (mask[:, :, 1, :, :].sum() + 1e-8)
            mae_w = (err[:, :, 2, :, :] * mask[:, :, 2, :, :]).sum() / (mask[:, :, 2, :, :].sum() + 1e-8)
        else:
            mae_u = mae_v = mae_w = 0.0
    else:
        mae_u = err[:, :, 0, :, :].mean()
        mae_v = err[:, :, 1, :, :].mean()
        mae_w = err[:, :, 2, :, :].mean()
    
    return float(mae_u), float(mae_v), float(mae_w)


def compute_mse_per_component(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[float, float, float]:
    """
    Compute Mean Squared Error per velocity component.
    
    Args:
        y_pred: Predicted velocity, shape (batch, slices, 3, H, W)
        y_true: Ground truth velocity, same shape
        mask: Optional binary mask for fluid region
        
    Returns:
        mse_u, mse_v, mse_w: Per-component MSE values
    """
    if y_pred.dim() == 4:
        y_pred = y_pred.unsqueeze(0)
        y_true = y_true.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    
    err_sq = (y_pred - y_true).pow(2)
    
    if mask is not None:
        mask = mask.expand_as(err_sq)
        mse_u = (err_sq[:, :, 0, :, :] * mask[:, :, 0, :, :]).sum() / (mask[:, :, 0, :, :].sum() + 1e-8)
        mse_v = (err_sq[:, :, 1, :, :] * mask[:, :, 1, :, :]).sum() / (mask[:, :, 1, :, :].sum() + 1e-8)
        mse_w = (err_sq[:, :, 2, :, :] * mask[:, :, 2, :, :]).sum() / (mask[:, :, 2, :, :].sum() + 1e-8)
    else:
        mse_u = err_sq[:, :, 0, :, :].mean()
        mse_v = err_sq[:, :, 1, :, :].mean()
        mse_w = err_sq[:, :, 2, :, :].mean()
    
    return float(mse_u), float(mse_v), float(mse_w)


def compute_rmse_per_component(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[float, float, float]:
    """
    Compute Root Mean Squared Error per velocity component.
    """
    mse_u, mse_v, mse_w = compute_mse_per_component(y_pred, y_true, mask)
    return np.sqrt(mse_u), np.sqrt(mse_v), np.sqrt(mse_w)


def compute_normalized_mae(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    norm_factors: Tuple[float, float, float],
    mask: Optional[torch.Tensor] = None
) -> Tuple[float, float, float, float]:
    """
    Compute Normalized MAE (nMAE) per component.
    
    nMAE = MAE / max_component, where max_component is from statistics.json
    
    Returns:
        nmae_u, nmae_v, nmae_w, nmae_total: Normalized MAE values
    """
    mae_u, mae_v, mae_w = compute_mae_per_component(y_pred, y_true, mask)
    
    nmae_u = mae_u / (norm_factors[0] + 1e-8)
    nmae_v = mae_v / (norm_factors[1] + 1e-8)
    nmae_w = mae_w / (norm_factors[2] + 1e-8)
    
    # Total is average of normalized components (equally weighted)
    nmae_total = (nmae_u + nmae_v + nmae_w) / 3.0
    
    return nmae_u, nmae_v, nmae_w, nmae_total


def compute_normalized_mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    norm_factors: Tuple[float, float, float],
    mask: Optional[torch.Tensor] = None
) -> Tuple[float, float, float, float]:
    """
    Compute Normalized MSE (nMSE) per component.
    
    nMSE = MSE / (max_component^2)
    """
    mse_u, mse_v, mse_w = compute_mse_per_component(y_pred, y_true, mask)
    
    nmse_u = mse_u / (norm_factors[0]**2 + 1e-8)
    nmse_v = mse_v / (norm_factors[1]**2 + 1e-8)
    nmse_w = mse_w / (norm_factors[2]**2 + 1e-8)
    
    nmse_total = (nmse_u + nmse_v + nmse_w) / 3.0
    
    return nmse_u, nmse_v, nmse_w, nmse_total


def compute_cosine_similarity(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute average cosine similarity between velocity vectors.
    
    For each voxel, compute cos_sim(v_pred, v_true), then average over all fluid voxels.
    
    Returns:
        avg_cosine_similarity: Average cosine similarity in [−1, 1]
    """
    if y_pred.dim() == 4:
        y_pred = y_pred.unsqueeze(0)
        y_true = y_true.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    
    # Flatten spatial dimensions: (batch, slices, 3, H, W) -> (batch*slices*H*W, 3)
    batch, slices, channels, H, W = y_pred.shape
    y_pred_flat = y_pred.permute(0, 1, 3, 4, 2).reshape(-1, channels)  # (N, 3)
    y_true_flat = y_true.permute(0, 1, 3, 4, 2).reshape(-1, channels)  # (N, 3)
    
    # Compute dot product and magnitudes
    dot_prod = (y_pred_flat * y_true_flat).sum(dim=1)  # (N,)
    mag_pred = y_pred_flat.norm(dim=1)  # (N,)
    mag_true = y_true_flat.norm(dim=1)  # (N,)
    
    # Cosine similarity with numerical stability
    denom = mag_pred * mag_true + 1e-8
    cos_sim = dot_prod / denom
    
    if mask is not None:
        # Flatten mask
        mask_flat = mask[:, :, 0, :, :].reshape(-1)  # (N,)
        # Weighted average
        num_valid = mask_flat.sum()
        if num_valid > 0:
            avg_cos_sim = (cos_sim * mask_flat).sum() / num_valid
        else:
            avg_cos_sim = 0.0
    else:
        avg_cos_sim = cos_sim.mean()
    
    return float(avg_cos_sim)


def compute_iou_topk(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    k_percent: float = 10.0,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute IoU (Intersection over Union) of top-k% magnitude voxels.
    
    This measures structure agreement: whether high-velocity regions align between
    prediction and ground truth. Useful for detecting "spike blobs" behavior.
    
    Args:
        y_pred: Predicted velocity
        y_true: Ground truth velocity
        k_percent: Percentage of top magnitude voxels to consider (default 10%)
        mask: Optional fluid region mask
        
    Returns:
        iou: Intersection over Union of top-k voxels
    """
    if y_pred.dim() == 4:
        y_pred = y_pred.unsqueeze(0)
        y_true = y_true.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    
    # Compute velocity magnitude
    mag_pred = y_pred.norm(dim=2, keepdim=True)  # (batch, slices, 1, H, W)
    mag_true = y_true.norm(dim=2, keepdim=True)
    
    # Flatten
    mag_pred_flat = mag_pred.reshape(-1)
    mag_true_flat = mag_true.reshape(-1)
    
    if mask is not None:
        mask_flat = mask[:, :, 0:1, :, :].reshape(-1)
        # Only consider fluid voxels
        valid_idx = mask_flat > 0.5
        mag_pred_flat = mag_pred_flat[valid_idx]
        mag_true_flat = mag_true_flat[valid_idx]
    
    if len(mag_pred_flat) == 0:
        return 0.0
    
    # Compute k-th percentile threshold
    k_idx = int(len(mag_pred_flat) * (100 - k_percent) / 100)
    
    # Sort and get threshold
    threshold_pred = torch.sort(mag_pred_flat, descending=True)[0][min(k_idx, len(mag_pred_flat) - 1)]
    threshold_true = torch.sort(mag_true_flat, descending=True)[0][min(k_idx, len(mag_true_flat) - 1)]
    
    # Binary masks for top-k voxels
    topk_pred = mag_pred_flat >= threshold_pred
    topk_true = mag_true_flat >= threshold_true
    
    # Compute IoU
    intersection = (topk_pred & topk_true).sum().float()
    union = (topk_pred | topk_true).sum().float()
    
    iou = intersection / (union + 1e-8)
    
    return float(iou)


def compute_sanity_stats(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, float]:
    """
    Compute sanity statistics for a tensor.
    
    Returns:
        dict with min, max, mean, std values
    """
    return {
        f"{name}_min": float(tensor.min()),
        f"{name}_max": float(tensor.max()),
        f"{name}_mean": float(tensor.mean()),
        f"{name}_std": float(tensor.std())
    }


def compute_all_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    norm_factors: Tuple[float, float, float],
    mask: Optional[torch.Tensor] = None,
    compute_optional: bool = True
) -> Dict[str, float]:
    """
    Compute all metrics for a single sample.
    
    Args:
        y_pred: Predicted 3D velocity field
        y_true: Ground truth 3D velocity field
        norm_factors: (max_u, max_v, max_w) for normalization
        mask: Optional fluid mask
        compute_optional: Whether to compute cosine similarity and IoU
        
    Returns:
        Dictionary with all metric values
    """
    metrics = {}
    
    # A) MAE per component
    mae_u, mae_v, mae_w = compute_mae_per_component(y_pred, y_true, mask)
    metrics['mae_u'] = mae_u
    metrics['mae_v'] = mae_v
    metrics['mae_w'] = mae_w
    metrics['mae_total'] = (mae_u + mae_v + mae_w) / 3.0
    
    # B) MSE per component
    mse_u, mse_v, mse_w = compute_mse_per_component(y_pred, y_true, mask)
    metrics['mse_u'] = mse_u
    metrics['mse_v'] = mse_v
    metrics['mse_w'] = mse_w
    metrics['mse_total'] = (mse_u + mse_v + mse_w) / 3.0
    
    # C) RMSE per component
    metrics['rmse_u'] = np.sqrt(mse_u)
    metrics['rmse_v'] = np.sqrt(mse_v)
    metrics['rmse_w'] = np.sqrt(mse_w)
    metrics['rmse_total'] = np.sqrt(metrics['mse_total'])
    
    # D) Normalized MAE and MSE
    nmae_u, nmae_v, nmae_w, nmae_total = compute_normalized_mae(y_pred, y_true, norm_factors, mask)
    metrics['nmae_u'] = nmae_u
    metrics['nmae_v'] = nmae_v
    metrics['nmae_w'] = nmae_w
    metrics['nmae_total'] = nmae_total
    
    nmse_u, nmse_v, nmse_w, nmse_total = compute_normalized_mse(y_pred, y_true, norm_factors, mask)
    metrics['nmse_u'] = nmse_u
    metrics['nmse_v'] = nmse_v
    metrics['nmse_w'] = nmse_w
    metrics['nmse_total'] = nmse_total
    
    if compute_optional:
        # E) Cosine similarity
        metrics['cosine_similarity'] = compute_cosine_similarity(y_pred, y_true, mask)
        
        # F) IoU of top-k magnitude voxels
        metrics['iou_top10'] = compute_iou_topk(y_pred, y_true, k_percent=10.0, mask=mask)
        metrics['iou_top5'] = compute_iou_topk(y_pred, y_true, k_percent=5.0, mask=mask)
    
    return metrics


def compute_accuracy_score(nmae_total: float) -> float:
    """
    Compute combined accuracy score.
    
    Accuracy = 1 / (1 + normalized_MAE_total)
    
    This is bounded in (0, 1], where higher is better.
    When nMAE=0, accuracy=1 (perfect).
    When nMAE=1, accuracy=0.5.
    """
    return 1.0 / (1.0 + nmae_total)


# ============================================================================
# Model Loading (Reusing Inference Code)
# ============================================================================

def resolve_path(path: str, project_root: str) -> Optional[str]:
    """Resolve a path to absolute, trying several strategies."""
    if not path:
        return None
    if os.path.exists(path):
        return os.path.abspath(path)
    if not os.path.isabs(path):
        abs_path = os.path.join(project_root, path)
        if os.path.exists(abs_path):
            return abs_path
    if 'VAE_model' in path:
        idx = path.find('VAE_model')
        abs_path = os.path.join(project_root, path[idx:])
        if os.path.exists(abs_path):
            return abs_path
    return path


def load_model_and_config(
    model_path: str,
    vae_encoder_path: Optional[str],
    vae_decoder_path: Optional[str],
    vae_path: Optional[str],
    dataset_dir: Optional[str],
    device: str
) -> Tuple:
    """
    Load diffusion model and VAE following inference.py logic.
    
    Returns:
        predictor: Loaded LatentDiffusionPredictor
        config: Model configuration dict
        norm_factors: Tuple of (max_u, max_v, max_w) for normalization
        dataset_root: Resolved dataset root directory
    """
    # Determine model directory
    if os.path.isfile(model_path):
        model_dir = os.path.dirname(model_path)
        weights_file = model_path
    else:
        model_dir = model_path
        best_model_path = os.path.join(model_dir, 'best_model.pt')
        model_pt_path = os.path.join(model_dir, 'model.pt')
        if os.path.exists(best_model_path):
            weights_file = best_model_path
        elif os.path.exists(model_pt_path):
            weights_file = model_pt_path
        else:
            raise FileNotFoundError(f"No model weights found in {model_dir}")
    
    # Load config
    log_path = os.path.join(model_dir, 'log.json')
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"log.json not found in {model_dir}")
    
    with open(log_path, 'r') as f:
        log_data = json.load(f)
        config = log_data.get('params', log_data)
    
    predictor_kwargs = config['training']['predictor']
    dataset_root = config['dataset']['root_dir']
    
    # Override dataset root if provided
    if dataset_dir:
        dataset_root = dataset_dir
        print(f"Using dataset directory from command-line: {dataset_root}")
    
    # Locate statistics.json
    norm_file = None
    if os.path.exists(os.path.join(dataset_root, 'statistics.json')):
        norm_file = os.path.join(dataset_root, 'statistics.json')
    else:
        # Try default data paths
        default_paths = [
            os.path.join(project_root, 'Diffusion_model', 'data', 'dataset', 'statistics.json'),
            os.path.join(project_root, 'data', 'dataset', 'statistics.json'),
        ]
        for p in default_paths:
            if os.path.exists(p):
                norm_file = p
                break
    
    if norm_file is None:
        raise FileNotFoundError("statistics.json not found. Required for normalization.")
    
    print(f"Using statistics from: {norm_file}")
    
    # Load norm_factors from statistics.json
    with open(norm_file, 'r') as f:
        stats = json.load(f)
    
    if 'U_per_component' in stats:
        pc = stats['U_per_component']
        norm_factors = (pc['max_u'], pc['max_v'], pc.get('max_w', pc['max_u']))
    elif 'U' in stats:
        max_v = stats['U']['max']
        norm_factors = (max_v, max_v, max_v)
    else:
        print("WARNING: Could not find velocity statistics. Using default (1, 1, 1).")
        norm_factors = (1.0, 1.0, 1.0)
    
    print(f"Norm factors: max_u={norm_factors[0]:.6f}, max_v={norm_factors[1]:.6f}, max_w={norm_factors[2]:.6f}")
    
    # Handle VAE paths
    if vae_path:
        vae_path = resolve_path(vae_path, project_root)
        predictor_kwargs['vae_path'] = vae_path
        print(f"Using VAE path: {vae_path}")
    
    if vae_encoder_path:
        vae_encoder_path = resolve_path(vae_encoder_path, project_root)
        predictor_kwargs['vae_encoder_path'] = vae_encoder_path
        print(f"Using VAE encoder path: {vae_encoder_path}")
    
    if vae_decoder_path:
        vae_decoder_path = resolve_path(vae_decoder_path, project_root)
        predictor_kwargs['vae_decoder_path'] = vae_decoder_path
        print(f"Using VAE decoder path: {vae_decoder_path}")
    
    # Initialize predictor
    print("Initializing models...")
    predictor = set_model(
        type='latent-diffusion',
        kwargs=predictor_kwargs,
        norm_file=norm_file
    )
    
    # Load diffusion model weights
    print(f"Loading weights from {weights_file}")
    loaded_state = torch.load(weights_file, map_location=device)
    
    # Filter out VAE weights (already loaded separately)
    model_state = {k: v for k, v in loaded_state.items() if not k.startswith('vae.')}
    
    missing, unexpected = predictor.load_state_dict(model_state, strict=False)
    non_vae_missing = [k for k in missing if not k.startswith('vae.')]
    non_vae_unexpected = [k for k in unexpected if not k.startswith('vae.')]
    
    if non_vae_missing:
        print(f"Warning: Missing keys (non-VAE): {non_vae_missing}")
    if non_vae_unexpected:
        print(f"Warning: Unexpected keys (non-VAE): {non_vae_unexpected}")
    
    predictor.to(device)
    predictor.eval()
    
    return predictor, config, norm_factors, dataset_root


# ============================================================================
# Main Evaluation Function
# ============================================================================

def run_evaluation(
    predictor,
    test_loader: DataLoader,
    norm_factors: Tuple[float, float, float],
    device: str,
    sampler: str = 'ddim',
    num_steps: int = 50,
    seed: int = 42,
    sanity_mode: bool = False,
    num_samples: Optional[int] = None,
    single_index: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Run evaluation over test set.
    
    Args:
        predictor: Loaded LatentDiffusionPredictor
        test_loader: DataLoader for test set
        norm_factors: (max_u, max_v, max_w) for normalization
        device: Device to run on
        sampler: 'ddpm' or 'ddim'
        num_steps: Number of diffusion steps (default 50 for DDIM, 1000 for DDPM)
        seed: Random seed for reproducibility
        sanity_mode: If True, bypass diffusion and use VAE reconstruction (VAE-ONLY)
        num_samples: Optional limit on number of samples
        single_index: If set, evaluate ONLY this single index (efficient direct access)
        
    Returns:
        per_sample_results: List of dicts with per-sample metrics
        sanity_stats: Dictionary with prediction/target statistics
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    per_sample_results = []
    sanity_stats = {
        'pred_min': [], 'pred_max': [], 'pred_mean': [], 'pred_std': [],
        'target_min': [], 'target_max': [], 'target_mean': [], 'target_std': []
    }
    
    # Determine which samples to evaluate
    total_available = len(test_loader.dataset)
    
    if single_index is not None:
        # Validate index
        if single_index < 0 or single_index >= total_available:
            raise ValueError(f"Index {single_index} out of range [0, {total_available-1}]")
        sample_indices = [single_index]
        print(f"\n  Evaluating SINGLE sample at index {single_index}")
    elif num_samples is not None:
        sample_indices = list(range(min(num_samples, total_available)))
        print(f"\n  Evaluating first {len(sample_indices)} samples")
    else:
        sample_indices = list(range(total_available))
        print(f"\n  Evaluating ALL {len(sample_indices)} samples")
    
    total_samples = len(sample_indices)
    
    # Print evaluation mode header
    print(f"\n{'='*60}")
    if sanity_mode:
        print(f"⚠️  VAE-ONLY SANITY CHECK on {total_samples} sample(s)")
        print(f"    Pipeline: GT → E3D → D3D → compare")
    else:
        print(f"🔬 END-TO-END DIFFUSION EVALUATION on {total_samples} sample(s)")
        print(f"    Pipeline: 2D input → E2D → {sampler.upper()} ({num_steps} steps) → D3D → compare")
    print(f"    Seed: {seed}, Device: {device}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    with torch.no_grad():
        for eval_idx, sample_idx in enumerate(sample_indices):
            sample_start = time.time()
            
            # Direct access to sample from dataset (efficient!)
            data = test_loader.dataset[sample_idx]
            
            # Add batch dimension and move to device
            img = data['microstructure'].unsqueeze(0).to(device)  # (1, slices, 1, H, W)
            velocity_2d = data['velocity_input'].unsqueeze(0).to(device)  # (1, slices, 3, H, W)
            target_velocity = data['velocity'].unsqueeze(0).to(device)  # (1, slices, 3, H, W)
            
            # Generate prediction based on mode
            if sanity_mode:
                # VAE-ONLY: bypass diffusion, use E3D→D3D on ground truth
                # This tests VAE reconstruction quality, NOT diffusion
                prediction = _run_vae_reconstruction(predictor, target_velocity, img, device)
            else:
                # TRUE END-TO-END: 2D input → E2D → diffusion → D3D
                prediction = _run_diffusion_prediction(
                    predictor, img, velocity_2d, device, 
                    sampler, num_steps, seed + sample_idx
                )
            
            sample_time = time.time() - sample_start
            
            # Verify shape
            assert prediction.shape == target_velocity.shape, \
                f"Shape mismatch: pred {prediction.shape} vs target {target_velocity.shape}"
            
            # Compute metrics
            mask = img  # Use microstructure as fluid mask
            metrics = compute_all_metrics(
                y_pred=prediction,
                y_true=target_velocity,
                norm_factors=norm_factors,
                mask=mask,
                compute_optional=True
            )
            
            # Add sample info
            metrics['sample_id'] = sample_idx
            metrics['time_sec'] = sample_time
            metrics['accuracy_score'] = compute_accuracy_score(metrics['nmae_total'])
            
            per_sample_results.append(metrics)
            
            # Collect sanity stats
            sanity_stats['pred_min'].append(float(prediction.min()))
            sanity_stats['pred_max'].append(float(prediction.max()))
            sanity_stats['pred_mean'].append(float(prediction.mean()))
            sanity_stats['pred_std'].append(float(prediction.std()))
            sanity_stats['target_min'].append(float(target_velocity.min()))
            sanity_stats['target_max'].append(float(target_velocity.max()))
            sanity_stats['target_mean'].append(float(target_velocity.mean()))
            sanity_stats['target_std'].append(float(target_velocity.std()))
            
            # Progress
            elapsed = time.time() - start_time
            samples_done = eval_idx + 1
            samples_per_sec = samples_done / elapsed if elapsed > 0 else 0
            
            mode_prefix = "[VAE]" if sanity_mode else "[DIFF]"
            print(f"{mode_prefix} Sample {sample_idx:4d} ({eval_idx+1}/{total_samples}) | "
                  f"nMAE={metrics['nmae_total']:.4f} | "
                  f"Acc={metrics['accuracy_score']:.4f} | "
                  f"Time={sample_time:.2f}s | "
                  f"Speed={samples_per_sec:.2f} samples/sec")
    
    total_time = time.time() - start_time
    print(f"\nTotal evaluation time: {total_time:.2f}s")
    print(f"Average time per sample: {total_time / max(1, len(per_sample_results)):.2f}s")
    
    return per_sample_results, sanity_stats


def _run_vae_reconstruction(predictor, target_velocity: torch.Tensor, img: torch.Tensor, device: str) -> torch.Tensor:
    """
    VAE-ONLY reconstruction: GT → E3D → D3D → output
    
    ⚠️  This does NOT evaluate diffusion! Only VAE reconstruction quality.
    """
    velocity_permuted = target_velocity.permute(0, 2, 1, 3, 4)  # (batch, 3, slices, H, W)
    
    # Normalize for VAE
    batch_size, channels, depth, H, W = velocity_permuted.shape
    velocity_flat = velocity_permuted.permute(0, 2, 1, 3, 4).contiguous().reshape(
        batch_size * depth, channels, H, W
    )
    velocity_flat_norm = predictor.normalizer['output'](velocity_flat)
    velocity_norm_5d = velocity_flat_norm.reshape(
        batch_size, depth, channels, H, W
    ).permute(0, 2, 1, 3, 4)
    
    # Encode with E3D
    if predictor.vae_is_dual:
        latent_5d, _ = predictor.vae.encode_3d_deterministic(velocity_norm_5d)
    else:
        condition = torch.ones(batch_size, dtype=torch.bool, device=device) if predictor.vae_conditional else None
        latent_5d, _ = predictor.vae.encode(velocity_norm_5d, condition=condition)
    
    # Decode with D3D
    if predictor.vae_is_dual:
        velocity_recon_5d = predictor.vae.decode_3d(latent_5d)
    else:
        condition = torch.ones(batch_size, dtype=torch.bool, device=device) if predictor.vae_conditional else None
        velocity_recon_5d = predictor.vae.decode(latent_5d, condition=condition)
    
    # Back to original shape and denormalize
    velocity_3d = velocity_recon_5d.permute(0, 2, 1, 3, 4)  # (batch, slices, 3, H, W)
    batch, depth, channels, H, W = velocity_3d.shape
    velocity_flat = velocity_3d.reshape(batch * depth, channels, H, W)
    velocity_flat = predictor.normalizer['output'].inverse(velocity_flat)
    prediction = velocity_flat.reshape(batch, depth, channels, H, W)
    
    # Mask
    prediction = prediction * img
    
    return prediction


def _run_diffusion_prediction(
    predictor, 
    img: torch.Tensor, 
    velocity_2d: torch.Tensor, 
    device: str,
    sampler: str,
    num_steps: int,
    sample_seed: int
) -> torch.Tensor:
    """
    TRUE END-TO-END diffusion prediction: 2D input → E2D → diffusion → D3D → output
    
    This is the REAL evaluation of the diffusion model.
    """
    # Get latent dimensions for fixed noise
    num_slices = velocity_2d.shape[1]
    with torch.no_grad():
        dummy_5d = torch.zeros(1, 3, num_slices, img.shape[3], img.shape[4]).to(device)
        if predictor.vae_is_dual:
            latent_shape = predictor.vae.encoder_2d(dummy_5d)[0].shape
        else:
            dummy_condition = torch.zeros(1, dtype=torch.bool, device=device) if predictor.vae_conditional else None
            latent_shape = predictor.vae.encoder(dummy_5d, condition=dummy_condition)[0].shape
        latent_channels = latent_shape[1]
        latent_depth = latent_shape[2]
        latent_h, latent_w = latent_shape[3], latent_shape[4]
    
    batch_size = img.shape[0]
    
    # Fix noise based on sample seed for reproducibility
    torch.manual_seed(sample_seed)
    fixed_noise = torch.randn(batch_size * latent_depth, latent_channels, latent_h, latent_w, device=device)
    
    # Run diffusion sampling
    if sampler == 'ddim':
        prediction = predictor.predict_ddim(img, velocity_2d, num_steps=num_steps, eta=0.0, noise=fixed_noise)
    else:
        prediction = predictor.predict(img, velocity_2d, noise=fixed_noise)
    
    return prediction


def aggregate_results(per_sample_results: List[Dict]) -> Dict:
    """
    Aggregate per-sample results into summary statistics.
    
    Returns:
        Dictionary with mean ± std for each metric
    """
    if not per_sample_results:
        return {}
    
    # Get all metric keys (exclude non-numeric)
    metric_keys = [k for k in per_sample_results[0].keys() if k not in ['sample_id']]
    
    aggregated = {}
    for key in metric_keys:
        values = [r[key] for r in per_sample_results if key in r]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_min"] = float(np.min(values))
            aggregated[f"{key}_max"] = float(np.max(values))
    
    return aggregated


def print_summary(aggregated: Dict, sanity_stats: Dict, sanity_mode: bool = False, sampler: str = 'ddim', steps: int = 50):
    """Print formatted summary of results with clear mode labeling."""
    
    print("\n" + "="*70)
    if sanity_mode:
        print("⚠️  VAE-ONLY SANITY CHECK RESULTS ⚠️")
        print("    (NOT diffusion model performance!)")
    else:
        print(f"END-TO-END DIFFUSION EVALUATION RESULTS")
        print(f"    (Sampler: {sampler.upper()}, Steps: {steps})")
    print("="*70)
    
    print("\n--- Per-Component MAE ---")
    print(f"  MAE_u:  {aggregated.get('mae_u_mean', 0):.6f} ± {aggregated.get('mae_u_std', 0):.6f}")
    print(f"  MAE_v:  {aggregated.get('mae_v_mean', 0):.6f} ± {aggregated.get('mae_v_std', 0):.6f}")
    print(f"  MAE_w:  {aggregated.get('mae_w_mean', 0):.6f} ± {aggregated.get('mae_w_std', 0):.6f}")
    
    print("\n--- Per-Component Normalized MAE ---")
    print(f"  nMAE_u: {aggregated.get('nmae_u_mean', 0):.6f} ± {aggregated.get('nmae_u_std', 0):.6f}")
    print(f"  nMAE_v: {aggregated.get('nmae_v_mean', 0):.6f} ± {aggregated.get('nmae_v_std', 0):.6f}")
    print(f"  nMAE_w: {aggregated.get('nmae_w_mean', 0):.6f} ± {aggregated.get('nmae_w_std', 0):.6f}")
    
    print("\n--- Total Metrics ---")
    print(f"  nMAE_total: {aggregated.get('nmae_total_mean', 0):.6f} ± {aggregated.get('nmae_total_std', 0):.6f}")
    print(f"  RMSE_total: {aggregated.get('rmse_total_mean', 0):.6f} ± {aggregated.get('rmse_total_std', 0):.6f}")
    
    print("\n--- Optional Metrics ---")
    print(f"  Cosine Similarity: {aggregated.get('cosine_similarity_mean', 0):.4f} ± {aggregated.get('cosine_similarity_std', 0):.4f}")
    print(f"  IoU (top 10%):     {aggregated.get('iou_top10_mean', 0):.4f} ± {aggregated.get('iou_top10_std', 0):.4f}")
    print(f"  IoU (top 5%):      {aggregated.get('iou_top5_mean', 0):.4f} ± {aggregated.get('iou_top5_std', 0):.4f}")
    
    print("\n--- Combined Accuracy Score ---")
    accuracy = aggregated.get('accuracy_score_mean', 0)
    if sanity_mode:
        print(f"  [VAE-only] Reconstruction Accuracy: {accuracy:.4f} ± {aggregated.get('accuracy_score_std', 0):.4f}")
        print(f"  ⚠️  This is VAE quality, NOT diffusion model performance!")
    else:
        print(f"  Diffusion Model Accuracy = 1 / (1 + nMAE_total) = {accuracy:.4f} ± {aggregated.get('accuracy_score_std', 0):.4f}")
        print(f"  Score (negative nMAE) = {-aggregated.get('nmae_total_mean', 0):.4f}")
    
    print("\n--- Sanity Statistics ---")
    print(f"  Prediction range: [{np.mean(sanity_stats['pred_min']):.6f}, {np.mean(sanity_stats['pred_max']):.6f}]")
    print(f"  Prediction mean:  {np.mean(sanity_stats['pred_mean']):.6f} ± {np.std(sanity_stats['pred_mean']):.6f}")
    print(f"  Target range:     [{np.mean(sanity_stats['target_min']):.6f}, {np.mean(sanity_stats['target_max']):.6f}]")
    print(f"  Target mean:      {np.mean(sanity_stats['target_mean']):.6f} ± {np.std(sanity_stats['target_mean']):.6f}")
    
    print("\n" + "="*70)


def save_results(
    per_sample_results: List[Dict],
    aggregated: Dict,
    sanity_stats: Dict,
    args,
    output_dir: str
):
    """Save results to JSON and optionally CSV."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine mode for filename and metadata
    mode_str = "vae_only" if args.sanity_mode else f"diffusion_{args.sampler}_{args.steps}steps"
    
    # Prepare full results dict with clear mode indication
    results = {
        'timestamp': timestamp,
        'evaluation_mode': 'VAE_ONLY_SANITY_CHECK' if args.sanity_mode else 'END_TO_END_DIFFUSION',
        'warning': '⚠️ This is VAE reconstruction quality, NOT diffusion model performance!' if args.sanity_mode else None,
        'pipeline': 'GT → E3D → D3D → compare' if args.sanity_mode else f'2D input → E2D → {args.sampler.upper()} ({args.steps} steps) → D3D → compare',
        'args': vars(args),
        'summary': aggregated,
        'sanity_stats': {
            k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
            for k, v in sanity_stats.items()
        },
        'accuracy_definition': "Accuracy = 1 / (1 + normalized_MAE_total), bounded in (0, 1], higher is better",
        'per_sample_results': per_sample_results
    }
    
    # Remove None warning if not sanity mode
    if results['warning'] is None:
        del results['warning']
    
    # Save JSON with mode in filename
    json_path = os.path.join(output_dir, f'eval_results_{mode_str}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Save CSV if requested
    if args.save_csv:
        import csv
        csv_path = args.save_csv if os.path.isabs(args.save_csv) else os.path.join(output_dir, args.save_csv)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
        
        fieldnames = ['sample_id', 'mae_u', 'mae_v', 'mae_w', 'nmae_total', 
                      'rmse_total', 'cosine_similarity', 'iou_top10', 'time_sec', 'accuracy_score']
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(per_sample_results)
        
        print(f"CSV saved to: {csv_path}")
    
    return json_path


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation of latent diffusion pipeline on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EVALUATION MODES:
  Default mode (no --sanity-mode): TRUE END-TO-END DIFFUSION EVALUATION
      Pipeline: 2D input → E2D → diffusion sampling → D3D → compare vs GT
      
  With --sanity-mode: VAE-ONLY RECONSTRUCTION CHECK
      Pipeline: 3D GT → E3D → D3D → compare vs GT
      ⚠️  This does NOT evaluate diffusion! Use only for debugging VAE.

Example (end-to-end diffusion, recommended):
    python scripts/eval_testset_end2end.py \\
        --diffusion-model-path Diffusion_model/trained/normal_mse \\
        --vae-encoder-path VAE_model/trained/dual_vae_stage2_2d \\
        --vae-decoder-path VAE_model/trained/dual_vae_stage1_3d \\
        --dataset-dir /path/to/dataset_3d \\
        --sampler ddim --steps 50

Example (single sample):
    python scripts/eval_testset_end2end.py ... --index 5

Example (VAE sanity check only):
    python scripts/eval_testset_end2end.py ... --sanity-mode
        """
    )
    
    # Required paths
    parser.add_argument('--diffusion-model-path', type=str, required=True,
                        help='Path to trained diffusion model directory')
    
    # VAE paths (at least one must be provided)
    parser.add_argument('--vae-path', type=str, default=None,
                        help='Path to VAE model (if single VAE, not dual)')
    parser.add_argument('--vae-encoder-path', type=str, default=None,
                        help='Path to VAE encoder (E2D, stage 2)')
    parser.add_argument('--vae-decoder-path', type=str, default=None,
                        help='Path to VAE decoder (D3D, stage 1)')
    
    # Dataset
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='Dataset split to evaluate on (default: test)')
    
    # Sampling options
    parser.add_argument('--index', type=int, default=None,
                        help='Evaluate ONLY this single sample index (skips all others)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Limit evaluation to first N samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--sampler', type=str, default='ddim', choices=['ddpm', 'ddim'],
                        help='Diffusion sampler: ddim (fast, default) or ddpm (slow, 1000 steps)')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of diffusion steps (default: 50 for DDIM, use 1000 for DDPM)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation (default: 1)')
    
    # Output options
    parser.add_argument('--save-csv', type=str, default=None,
                        help='Path to save per-sample CSV results')
    parser.add_argument('--save-npz-preds', action='store_true',
                        help='Save predictions as NPZ files for inspection')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: model directory)')
    
    # Sanity mode
    parser.add_argument('--sanity-mode', action='store_true',
                        help='⚠️  VAE-ONLY mode: bypass diffusion, test VAE reconstruction only')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: cuda or cpu (default: cuda if available)')
    
    return parser.parse_args()


def validate_device(requested_device: str) -> str:
    """Validate and resolve device, with clear error messages."""
    if requested_device is None:
        # Auto-detect
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("⚠ No CUDA available, using CPU (this will be slow)")
    elif requested_device == 'cuda':
        if not torch.cuda.is_available():
            print("="*70)
            print("ERROR: --device cuda requested but CUDA is not available!")
            print("Please use --device cpu or ensure CUDA is properly installed.")
            print("="*70)
            sys.exit(1)
        device = 'cuda'
        print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU device (this may be slow)")
    
    return device


def print_mode_banner(sanity_mode: bool, sampler: str, steps: int):
    """Print a clear banner indicating evaluation mode."""
    print("\n" + "="*70)
    if sanity_mode:
        print("⚠️  " + "="*20 + " VAE-ONLY SANITY CHECK " + "="*20 + " ⚠️")
        print("="*70)
        print("  MODE: VAE reconstruction quality check")
        print("  Pipeline: 3D Ground Truth → E3D encoder → D3D decoder → compare")
        print("")
        print("  ⚠️  WARNING: This does NOT evaluate the diffusion model!")
        print("  ⚠️  Metrics reflect VAE quality only, NOT end-to-end performance.")
        print("  ⚠️  Do NOT report these as diffusion model accuracy!")
        print("="*70)
    else:
        print("🔬 " + "="*18 + " END-TO-END DIFFUSION EVALUATION " + "="*17 + " 🔬")
        print("="*70)
        print("  MODE: True end-to-end diffusion evaluation")
        print(f"  Pipeline: 2D input → E2D → {sampler.upper()} ({steps} steps) → D3D → compare")
        print("")
        print("  ✓ This evaluates the FULL inference pipeline.")
        print("  ✓ Metrics reflect actual model performance.")
        print("="*70)


def main():
    args = parse_args()
    
    # Validate and set device
    device = validate_device(args.device)
    args.device = device  # Update args for saving
    
    print("\n" + "="*70)
    print("LATENT DIFFUSION MODEL EVALUATION")
    print("="*70)
    print(f"  Diffusion model: {args.diffusion_model_path}")
    print(f"  Dataset: {args.dataset_dir}")
    print(f"  Split: {args.split}")
    print(f"  Device: {device}")
    if args.index is not None:
        print(f"  Single sample index: {args.index}")
    elif args.num_samples is not None:
        print(f"  Number of samples: {args.num_samples}")
    print("="*70)
    
    # Print mode banner with clear warnings
    print_mode_banner(args.sanity_mode, args.sampler, args.steps)
    
    # Validate arguments
    if args.vae_path is None and (args.vae_encoder_path is None or args.vae_decoder_path is None):
        # Check if paths are in the model config
        log_path = os.path.join(args.diffusion_model_path, 'log.json')
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                config = json.load(f)
                predictor_kwargs = config.get('params', config).get('training', {}).get('predictor', {})
                if predictor_kwargs.get('vae_path') or (predictor_kwargs.get('vae_encoder_path') and predictor_kwargs.get('vae_decoder_path')):
                    print("VAE paths found in model config")
                else:
                    print("WARNING: No VAE paths provided and none found in model config")
    
    # Load model
    predictor, config, norm_factors, dataset_root = load_model_and_config(
        model_path=args.diffusion_model_path,
        vae_encoder_path=args.vae_encoder_path,
        vae_decoder_path=args.vae_decoder_path,
        vae_path=args.vae_path,
        dataset_dir=args.dataset_dir,
        device=device
    )
    
    # Load test data
    print(f"\nLoading {args.split} set from {dataset_root}...")
    loaders = get_loader(
        root_dir=dataset_root,
        batch_size=args.batch_size,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=False,
        seed=2024,  # Match training seed
        augment=False,
        use_3d=True,
        num_workers=0
    )
    
    train_loader, val_loader, test_loader = loaders[0]
    
    if args.split == 'train':
        data_loader = train_loader
    elif args.split == 'valid':
        data_loader = val_loader
    else:
        data_loader = test_loader
    
    print(f"Loaded {len(data_loader)} batches from {args.split} set")
    
    # Run evaluation
    per_sample_results, sanity_stats = run_evaluation(
        predictor=predictor,
        test_loader=data_loader,
        norm_factors=norm_factors,
        device=device,
        sampler=args.sampler,
        num_steps=args.steps,
        seed=args.seed,
        sanity_mode=args.sanity_mode,
        num_samples=args.num_samples,
        single_index=args.index
    )
    
    # Aggregate results
    aggregated = aggregate_results(per_sample_results)
    
    # Print summary with mode indication
    print_summary(aggregated, sanity_stats, args.sanity_mode, args.sampler, args.steps)
    
    # Save results
    output_dir = args.output_dir if args.output_dir else args.diffusion_model_path
    os.makedirs(output_dir, exist_ok=True)
    
    save_results(per_sample_results, aggregated, sanity_stats, args, output_dir)
    
    # Final message with mode-appropriate wording
    print("\n" + "="*70)
    if args.sanity_mode:
        print("⚠️  VAE-ONLY SANITY CHECK COMPLETE")
        print(f"  VAE Reconstruction nMAE: {aggregated.get('nmae_total_mean', 0):.4f}")
        print("  (This is NOT diffusion model accuracy!)")
    else:
        print("✓ END-TO-END DIFFUSION EVALUATION COMPLETE")
        print(f"  Diffusion Model Accuracy Score: {aggregated.get('accuracy_score_mean', 0):.4f}")
        print(f"  (Accuracy = 1 / (1 + nMAE_total), higher is better)")
    print("="*70)


if __name__ == "__main__":
    main()
