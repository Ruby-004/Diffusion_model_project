"""
VAE Inference Script: Visualize Encoder/Decoder Input/Output

This script loads a trained VAE model and visualizes:
1. Original velocity field (input to encoder)
2. Latent space representation (encoder output)
3. Reconstructed velocity field (decoder output)
4. Error maps between input and reconstruction

Supports both standard VAE and Dual-Branch VAE models.

Usage:
    python inference_vae.py [vae_path] [options]
    
Examples:
    # Standard VAE with first test sample
    python inference_vae.py trained/vae_norm_8 --index 0
    
    # Dual VAE Stage 1 (3D branch test)
    python inference_vae.py trained/dual_vae_stage1_3d --mode 3d
    
    # Dual VAE Stage 2 (2D branch test)
    python inference_vae.py trained/dual_vae_stage2_2d --mode 2d
    
    # Dual VAE cross-reconstruction (2D input -> 3D output)
    python inference_vae.py trained/dual_vae_stage2_2d --mode cross
    
    # Custom dataset directory
    python inference_vae.py trained/vae_norm_8 --dataset-dir path/to/dataset
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.vae.autoencoder import VariationalAutoencoder
from src.dual_vae.model import DualBranchVAE
from utils.dataset import get_loader


def _map_encoder_keys(state_dict: dict, prefix: str = 'encoder.') -> dict:
    """
    Map encoder keys from named-layer format to sequential format.
    
    Named format: encoder.conv_in, encoder.res1_1, encoder.down1, etc.
    Sequential format: encoder.layers.0, encoder.layers.1, etc.
    """
    key_mapping = {
        f'{prefix}conv_in': f'{prefix}layers.0',
        f'{prefix}res1_1': f'{prefix}layers.1',
        f'{prefix}res1_2': f'{prefix}layers.2',
        f'{prefix}down1': f'{prefix}layers.3',
        f'{prefix}res2_1': f'{prefix}layers.4',
        f'{prefix}res2_2': f'{prefix}layers.5',
        f'{prefix}down2': f'{prefix}layers.6',
        f'{prefix}res3_1': f'{prefix}layers.7',
        f'{prefix}res3_2': f'{prefix}layers.8',
        f'{prefix}norm_out': f'{prefix}layers.9',
        f'{prefix}conv_out': f'{prefix}layers.11',
    }
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in key_mapping.items():
            if key.startswith(old_prefix):
                new_key = key.replace(old_prefix, new_prefix, 1)
                break
        new_state_dict[new_key] = value
    
    return new_state_dict


def _map_decoder_keys(state_dict: dict, prefix: str = 'decoder.') -> dict:
    """
    Map decoder keys from named-layer format to sequential format.
    
    Named format: decoder.conv_in, decoder.res1_1, decoder.conv_up1, etc.
    Sequential format: decoder.layers.0, decoder.layers.1, etc.
    """
    key_mapping = {
        f'{prefix}conv_in': f'{prefix}layers.0',
        f'{prefix}res1_1': f'{prefix}layers.1',
        f'{prefix}res1_2': f'{prefix}layers.2',
        f'{prefix}conv_up1': f'{prefix}layers.4',
        f'{prefix}res2_1': f'{prefix}layers.5',
        f'{prefix}res2_2': f'{prefix}layers.6',
        f'{prefix}conv_up2': f'{prefix}layers.8',
        f'{prefix}res3_1': f'{prefix}layers.9',
        f'{prefix}res3_2': f'{prefix}layers.10',
        f'{prefix}norm_out': f'{prefix}layers.11',
        f'{prefix}conv_out': f'{prefix}layers.13',
    }
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in key_mapping.items():
            if key.startswith(old_prefix):
                new_key = key.replace(old_prefix, new_prefix, 1)
                break
        new_state_dict[new_key] = value
    
    return new_state_dict


def _needs_key_mapping(state_dict: dict) -> bool:
    """Check if state dict uses named-layer format (needs mapping)."""
    return any(
        '.conv_in.' in k or '.res1_1.' in k or '.down1.' in k 
        for k in state_dict.keys()
    )


def detect_model_type(vae_path: str) -> str:
    """
    Detect whether the model is a standard VAE or DualBranchVAE.
    
    Returns:
        'standard': Standard VariationalAutoencoder
        'dual_stage1': DualBranchVAE Stage 1 (encoder.*, decoder.* -> E3D, D3D)
        'dual_stage2': DualBranchVAE Stage 2 (encoder_2d.*, decoder_2d.*, etc.)
    """
    # Check for model files
    model_files = ['vae.pt', 'model.pt', 'best_model.pt']
    model_path = None
    for f in model_files:
        p = os.path.join(vae_path, f)
        if os.path.exists(p):
            model_path = p
            break
    
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {vae_path}")
    
    # Load state dict to check keys
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    keys = list(state_dict.keys())
    
    # Check for dual VAE stage 2 keys (encoder_2d, decoder_2d, encoder_3d, decoder_3d all present)
    if any(k.startswith('encoder_2d.') for k in keys) and any(k.startswith('encoder_3d.') for k in keys):
        return 'dual_stage2'
    
    # Check for 3D-only model (encoder_3d/decoder_3d but no encoder_2d) - this is dual_stage1
    if any(k.startswith('encoder_3d.') for k in keys) and not any(k.startswith('encoder_2d.') for k in keys):
        return 'dual_stage1_3d_only'
    
    # Check for standard/stage1 keys (encoder., decoder.)
    if any(k.startswith('encoder.') for k in keys):
        # Could be standard VAE or dual_stage1 (trained as standard VAE for 3D)
        # Check log file for more info
        log_path = os.path.join(vae_path, 'vae_log.json')
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log = json.load(f)
            # Check if it was trained as part of dual VAE pipeline
            if 'dual_stage' in log or 'stage' in log:
                return 'dual_stage1'
        return 'standard'
    
    # Default to standard if we can't determine
    return 'standard'


def load_vae(vae_path: str, device: str, model_type: str = None) -> tuple:
    """
    Load VAE model and its configuration.
    
    Args:
        vae_path: Path to VAE model directory
        device: Device to load model on
        model_type: Override auto-detected model type ('standard', 'dual_stage1', 'dual_stage2')
    
    Returns:
        vae: VariationalAutoencoder or DualBranchVAE model
        log: Configuration/training log dictionary
        norm_factors: Normalization factors [max_u, max_v, max_w]
        model_type: Detected or specified model type
    """
    # Auto-detect model type if not specified
    if model_type is None:
        model_type = detect_model_type(vae_path)
    
    # Load log to get configuration
    log_path = os.path.join(vae_path, 'vae_log.json')
    with open(log_path, 'r') as f:
        log = json.load(f)
    
    in_channels = log.get('in_channels', 3)
    latent_channels = log.get('latent_channels', 8)
    conditional = log.get('conditional', False)
    norm_factors = log.get('norm_factors', [1.0, 1.0, 1.0])
    
    print(f"Loading VAE from: {vae_path}")
    print(f"  Model type: {model_type}")
    print(f"  in_channels: {in_channels}")
    print(f"  latent_channels: {latent_channels}")
    print(f"  conditional: {conditional}")
    print(f"  norm_factors: {norm_factors}")
    
    if model_type in ['standard', 'dual_stage1']:
        # Create model instance
        vae = VariationalAutoencoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            conditional=conditional
        )
        
        # Load weights with key mapping if needed
        model_files = ['vae.pt', 'best_model.pt', 'model.pt']
        for f in model_files:
            model_path = os.path.join(vae_path, f)
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                
                # Apply key mapping if checkpoint uses named-layer format
                if _needs_key_mapping(state_dict):
                    print(f"  Applying key mapping for checkpoint format...")
                    state_dict = _map_encoder_keys(state_dict)
                    state_dict = _map_decoder_keys(state_dict)
                
                vae.load_state_dict(state_dict)
                print(f"  Loaded weights from: {f}")
                break
    elif model_type == 'dual_stage1_3d_only':
        # Stage 1 3D model saved with encoder_3d/decoder_3d prefixes
        # Need to remap keys to standard encoder/decoder format
        vae = VariationalAutoencoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            conditional=conditional
        )
        
        model_files = ['vae.pt', 'best_model.pt', 'model.pt']
        for f in model_files:
            model_path = os.path.join(vae_path, f)
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                
                # Remap encoder_3d -> encoder and decoder_3d -> decoder
                print(f"  Remapping encoder_3d/decoder_3d keys to standard format...")
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('encoder_3d.'):
                        new_key = k.replace('encoder_3d.', 'encoder.', 1)
                    elif k.startswith('decoder_3d.'):
                        new_key = k.replace('decoder_3d.', 'decoder.', 1)
                    else:
                        new_key = k
                    new_state_dict[new_key] = v
                
                vae.load_state_dict(new_state_dict)
                print(f"  Loaded weights from: {f}")
                break
    elif model_type == 'dual_stage2':
        # Stage 2: Full DualBranchVAE
        vae = DualBranchVAE(
            in_channels=in_channels,
            latent_channels=latent_channels,
            share_encoders=False,
            share_decoders=False
        )
        # Load weights - DualBranchVAE uses named layer format directly
        model_files = ['model.pt', 'best_model.pt', 'vae.pt']
        for f in model_files:
            model_path = os.path.join(vae_path, f)
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                # DualBranchVAE already expects named-layer format (conv_in, res1_1, etc.)
                # No key mapping needed
                vae.load_state_dict(state_dict)
                print(f"  Loaded weights from: {f}")
                break
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    vae.to(device)
    vae.eval()
    
    return vae, log, norm_factors, model_type


def visualize_velocity_slice(
    velocity: np.ndarray,
    mask: np.ndarray = None,
    title: str = "Velocity",
    figsize: tuple = (15, 4),
    vmin_vmax: tuple = None,
    show_colorbar: bool = True
):
    """
    Visualize a single velocity slice with u, v, w components.
    
    Args:
        velocity: Shape (3, H, W) - velocity components [u, v, w]
        mask: Shape (H, W) - optional binary mask for display
        title: Title prefix for the plots
        figsize: Figure size
        vmin_vmax: Optional (vmin, vmax) for colorbar normalization
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    component_names = ['u (vx)', 'v (vy)', 'w (vz)']
    
    for i, (ax, name) in enumerate(zip(axes, component_names)):
        data = velocity[i]
        
        # Apply mask for visualization if provided
        if mask is not None:
            # Show solid regions as gray
            masked_data = np.ma.masked_where(mask == 0, data)
            im = ax.imshow(masked_data, cmap='coolwarm', origin='lower')
            # Overlay solid region
            ax.imshow(1 - mask, cmap='gray', alpha=0.3, origin='lower', vmin=0, vmax=1)
        else:
            im = ax.imshow(data, cmap='coolwarm', origin='lower')
        
        ax.set_title(f"{title} - {name}")
        ax.axis('off')
        
        if show_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def visualize_latent_space(
    latent: np.ndarray,
    depth_slice: int = 0,
    title: str = "Latent Space",
    figsize: tuple = (15, 4)
):
    """
    Visualize latent space channels for a specific depth slice.
    
    Args:
        latent: Shape (latent_channels, D, H, W)
        depth_slice: Which depth slice to visualize
        title: Title for the plot
    """
    latent_channels = latent.shape[0]
    
    # Show up to 8 channels
    n_show = min(latent_channels, 8)
    n_cols = min(4, n_show)
    n_rows = (n_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_show):
        row, col = i // n_cols, i % n_cols
        data = latent[i, depth_slice]
        im = axes[row, col].imshow(data, cmap='viridis', origin='lower')
        axes[row, col].set_title(f"Ch {i}")
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(n_show, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f"{title} (depth slice {depth_slice})")
    plt.tight_layout()
    return fig


def visualize_reconstruction_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray = None,
    depth_slice: int = 0,
    figsize: tuple = (15, 12)
):
    """
    Compare original and reconstructed velocity fields with error maps.
    
    Args:
        original: Shape (D, 3, H, W) or (3, D, H, W)
        reconstructed: Same shape as original
        mask: Shape (D, H, W) or (H, W)
        depth_slice: Which depth slice to visualize
    """
    # Ensure shape is (D, 3, H, W)
    if original.shape[1] != 3:
        original = original.transpose(1, 0, 2, 3)
        reconstructed = reconstructed.transpose(1, 0, 2, 3)
    
    orig_slice = original[depth_slice]  # (3, H, W)
    recon_slice = reconstructed[depth_slice]
    error = np.abs(orig_slice - recon_slice)
    
    if mask is not None:
        if mask.ndim == 3:
            mask_slice = mask[depth_slice]
        else:
            mask_slice = mask
    else:
        mask_slice = None
    
    component_names = ['u (vx)', 'v (vy)', 'w (vz)']
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    
    for i, name in enumerate(component_names):
        # Compute shared vmin/vmax for original and reconstructed
        orig_data = orig_slice[i]
        recon_data = recon_slice[i]
        if mask_slice is not None:
            orig_masked = np.ma.masked_where(mask_slice == 0, orig_data)
            recon_masked = np.ma.masked_where(mask_slice == 0, recon_data)
            vmin = min(orig_masked.min(), recon_masked.min())
            vmax = max(orig_masked.max(), recon_masked.max())
        else:
            vmin = min(orig_data.min(), recon_data.min())
            vmax = max(orig_data.max(), recon_data.max())
        
        # Make symmetric around zero for coolwarm colormap
        vabs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vabs, vabs
        
        # Original
        ax = axes[0, i]
        data = orig_data
        if mask_slice is not None:
            data = np.ma.masked_where(mask_slice == 0, data)
        im = ax.imshow(data, cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"Original {name}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Reconstructed
        ax = axes[1, i]
        data = recon_data
        if mask_slice is not None:
            data = np.ma.masked_where(mask_slice == 0, data)
        im = ax.imshow(data, cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"Reconstructed {name}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Error
        ax = axes[2, i]
        err_data = error[i]
        if mask_slice is not None:
            err_data = np.ma.masked_where(mask_slice == 0, err_data)
        im = ax.imshow(err_data, cmap='Reds', origin='lower')
        ax.set_title(f"Error |Δ{name}|")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Velocity Reconstruction Comparison (depth slice {depth_slice})")
    plt.tight_layout()
    return fig


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor, mask: torch.Tensor = None):
    """
    Compute reconstruction metrics.
    
    Args:
        original: Shape (B, 3, D, H, W)
        reconstructed: Shape (B, 3, D, H, W)
        mask: Shape (B, 1, D, H, W)
    
    Returns:
        Dictionary with metrics
    """
    if mask is not None:
        # Expand mask to match channels
        mask_expanded = mask.expand_as(original)
        error = torch.abs(original - reconstructed) * mask_expanded
        num_fluid_pixels = mask_expanded.sum() + 1e-8
        
        # Per-component metrics
        mae_u = (error[:, 0:1] * mask).sum() / (mask.sum() + 1e-8)
        mae_v = (error[:, 1:2] * mask).sum() / (mask.sum() + 1e-8)
        mae_w = (error[:, 2:3] * mask).sum() / (mask.sum() + 1e-8)
        
        # Overall
        mae = error.sum() / num_fluid_pixels
        
        # Relative error
        orig_magnitude = torch.abs(original * mask_expanded).sum() / num_fluid_pixels + 1e-8
        relative_error = mae / orig_magnitude
    else:
        error = torch.abs(original - reconstructed)
        mae = error.mean()
        mae_u = error[:, 0].mean()
        mae_v = error[:, 1].mean()
        mae_w = error[:, 2].mean()
        relative_error = mae / (torch.abs(original).mean() + 1e-8)
    
    return {
        'mae': mae.item(),
        'mae_u': mae_u.item(),
        'mae_v': mae_v.item(),
        'mae_w': mae_w.item(),
        'relative_error': relative_error.item()
    }


def encode_decode(vae, velocity_normalized, model_type: str, mode: str, is_3d: torch.Tensor = None):
    """
    Encode and decode velocity field based on model type and mode.
    
    Args:
        vae: VAE model (VariationalAutoencoder or DualBranchVAE)
        velocity_normalized: Normalized velocity field (B, 3, D, H, W)
        model_type: 'standard', 'dual_stage1', 'dual_stage2'
        mode: '2d', '3d', 'cross' (for dual VAE modes)
        is_3d: Boolean tensor for conditional VAE
    
    Returns:
        latent: Latent representation
        mean, logvar: VAE distribution parameters
        reconstructed: Reconstructed velocity field
    """
    with torch.no_grad():
        if model_type == 'standard':
            latent, (mean, logvar) = vae.encode(velocity_normalized, condition=is_3d)
            reconstructed = vae.decode(latent, condition=is_3d)
        
        elif model_type in ['dual_stage1', 'dual_stage1_3d_only']:
            # Stage 1 is a standard VAE trained on 3D data
            latent, (mean, logvar) = vae.encode(velocity_normalized, condition=None)
            reconstructed = vae.decode(latent, condition=None)
        
        elif model_type == 'dual_stage2':
            if mode == '2d':
                latent, (mean, logvar) = vae.encode_2d(velocity_normalized)
                reconstructed = vae.decode_2d(latent)
            elif mode == '3d':
                latent, (mean, logvar) = vae.encode_3d(velocity_normalized)
                reconstructed = vae.decode_3d(latent)
            elif mode == 'cross':
                # Cross-reconstruction: 2D input -> 3D output
                latent, (mean, logvar) = vae.encode_2d(velocity_normalized)
                reconstructed = vae.decode_3d(latent)
            else:
                raise ValueError(f"Unknown mode for dual_stage2: {mode}")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    return latent, (mean, logvar), reconstructed


def main():
    parser = argparse.ArgumentParser(description="VAE Inference: Visualize Encoder/Decoder")
    parser.add_argument('vae_path', type=str, nargs='?', default='trained/vae_norm_8',
                        help='Path to the trained VAE model directory')
    parser.add_argument('--dataset-dir', type=str, default=None,
                        help='Path to the dataset directory')
    parser.add_argument('--index', type=int, default=0,
                        help='Index of the test sample to visualize')
    parser.add_argument('--depth-slice', type=int, default=5,
                        help='Which depth slice to visualize (default: middle slice)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save output figures (if not provided, displays interactively)')
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', '2d', '3d', 'cross'],
                        help='Mode for dual VAE: 2d (E2D->D2D), 3d (E3D->D3D), cross (E2D->D3D)')
    parser.add_argument('--model-type', type=str, default=None, 
                        choices=['standard', 'dual_stage1', 'dual_stage1_3d_only', 'dual_stage2'],
                        help='Override model type detection')
    
    args = parser.parse_args()
    device = args.device
    
    print(f"\n{'='*60}")
    print("VAE INFERENCE: Visualize Encoder/Decoder Input/Output")
    print(f"{'='*60}\n")
    
    # --- Resolve paths ---
    vae_path = args.vae_path
    if not os.path.isabs(vae_path):
        # Try relative to script directory first
        if not os.path.exists(vae_path):
            vae_path = os.path.join(current_dir, vae_path)
    
    if not os.path.exists(vae_path):
        print(f"ERROR: VAE path not found: {vae_path}")
        sys.exit(1)
    
    # --- Load VAE ---
    vae, log, norm_factors, model_type = load_vae(vae_path, device, args.model_type)
    norm_factors_tensor = torch.tensor(norm_factors, dtype=torch.float32, device=device)
    conditional = log.get('conditional', False)
    
    # Determine mode
    mode = args.mode
    if mode == 'auto':
        if model_type in ['dual_stage1', 'dual_stage1_3d_only']:
            mode = '3d'
        elif model_type == 'dual_stage2':
            mode = '2d'  # Default to 2D branch for stage 2
        else:
            mode = '3d'  # Standard VAE, test with 3D data
    
    print(f"  Testing mode: {mode}")
    
    # --- Load Dataset ---
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        # Try common dataset locations
        possible_paths = [
            os.path.join(project_root, 'data', 'rve_5k_xy'),
            os.path.join(project_root, 'Diffusion_model', 'data', 'dataset'),
            os.path.join(project_root, 'data', 'dataset'),
            r'C:\Users\alexd\Downloads\dataset_3d',  # Common download location
        ]
        for path in possible_paths:
            if os.path.exists(path):
                dataset_dir = path
                break
    
    if dataset_dir is None or not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory not found. Tried: {possible_paths}")
        print("Please specify --dataset-dir")
        sys.exit(1)
    
    print(f"Using dataset: {dataset_dir}")
    
    # Get data loader - use regular dataset for cross mode (needs both U_2d and U)
    # Use VAE dataset for 2d/3d modes (alternating samples)
    use_vae_dataset = mode != 'cross'
    _, _, test_loader = get_loader(
        root_dir=dataset_dir,
        batch_size=1,
        use_vae_dataset=use_vae_dataset,
        augment=False
    )
    
    # Get sample
    test_dataset = test_loader.dataset
    if hasattr(test_dataset, 'dataset'):
        # Handle Subset wrapper
        base_dataset = test_dataset.dataset
        indices = test_dataset.indices
        actual_idx = indices[args.index]
    else:
        base_dataset = test_dataset
        actual_idx = args.index
    
    print(f"Loading test sample {args.index} (actual index: {actual_idx})")
    
    # Get sample data
    sample = test_dataset[args.index]
    
    mask = sample['microstructure'].to(device)  # (1, D, H, W) or (D, 1, H, W)
    
    # For dual VAE modes, we need to select appropriate velocity field
    if 'velocity_input' in sample and 'velocity' in sample:
        # Full dataset with both U_2d and U
        if mode in ['2d', 'cross']:
            velocity = sample['velocity_input'].to(device)  # U_2d for 2D encoder
            print("Using velocity_input (2D velocity) as input")
        else:
            velocity = sample['velocity'].to(device)  # U for 3D encoder
            print("Using velocity (3D velocity) as input")
        velocity_target = sample['velocity'].to(device)  # Always compare to 3D target
    else:
        # VAE dataset (alternating 2D/3D samples)
        velocity = sample['velocity'].to(device)
        velocity_target = velocity
        is_2d_sample = sample.get('is_2d', torch.tensor(False))
        print(f"Sample is {'2D' if is_2d_sample else '3D'} flow data")
    
    # Ensure consistent shape: (1, 3, D, H, W) for batch processing
    if velocity.dim() == 4:
        # Check if shape is (3, D, H, W) or (D, 3, H, W)
        if velocity.shape[0] == 3:
            velocity = velocity.unsqueeze(0)  # (1, 3, D, H, W)
        else:
            velocity = velocity.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, D, H, W)
    
    if velocity_target.dim() == 4:
        if velocity_target.shape[0] == 3:
            velocity_target = velocity_target.unsqueeze(0)
        else:
            velocity_target = velocity_target.permute(1, 0, 2, 3).unsqueeze(0)
    
    if mask.dim() == 4:
        if mask.shape[0] == 1:
            mask = mask.unsqueeze(0)  # (1, 1, D, H, W)
        else:
            # (D, 1, H, W) -> (1, 1, D, H, W)
            mask = mask.permute(1, 0, 2, 3).unsqueeze(0)
    
    print(f"Input velocity shape: {velocity.shape}")
    print(f"Target velocity shape: {velocity_target.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Check actual w component statistics
    w_component = velocity[:, 2, :, :, :]  # Extract w (vz) component
    w_max = w_component.abs().max().item()
    w_mean = w_component.abs().mean().item()
    print(f"Input w component - max: {w_max:.6e}, mean: {w_mean:.6e}")
    
    # Normalize input
    nf = norm_factors_tensor.view(1, 3, 1, 1, 1)
    velocity_normalized = velocity / nf
    
    # --- Encode & Decode ---
    print("\n--- Encoding & Decoding ---")
    
    # For conditional VAE
    is_3d = None
    if conditional:
        is_3d = torch.tensor([mode == '3d'], dtype=torch.bool, device=device)
        print(f"Using conditional VAE with is_3d={is_3d.item()}")
    
    latent, (mean, logvar), reconstructed_normalized = encode_decode(
        vae, velocity_normalized, model_type, mode, is_3d
    )
    
    print(f"Input shape: {velocity_normalized.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Latent mean range: [{latent.min().item():.4f}, {latent.max().item():.4f}]")
    print(f"Latent std: {latent.std().item():.4f}")
    
    # Apply mask (in image space, not latent space!)
    reconstructed_normalized_masked = reconstructed_normalized * mask
    
    # Denormalize
    reconstructed = reconstructed_normalized_masked * nf
    
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Check reconstructed w component
    w_recon = reconstructed[:, 2, :, :, :]
    w_recon_max = w_recon.abs().max().item()
    w_recon_mean = w_recon.abs().mean().item()
    print(f"Reconstructed w component - max: {w_recon_max:.6e}, mean: {w_recon_mean:.6e}")
    
    # --- Compute Metrics ---
    print("\n--- Reconstruction Metrics ---")
    # Also mask the velocity for fair comparison
    velocity_masked = velocity * mask
    velocity_target_masked = velocity_target * mask
    
    # Compute metrics against input (reconstruction accuracy)
    metrics_input = compute_metrics(velocity_masked, reconstructed, mask)
    print("Compared to INPUT:")
    print(f"  MAE (overall): {metrics_input['mae']:.6f}")
    print(f"  MAE (u): {metrics_input['mae_u']:.6f}")
    print(f"  MAE (v): {metrics_input['mae_v']:.6f}")
    print(f"  MAE (w): {metrics_input['mae_w']:.6f}")
    print(f"  Relative error: {metrics_input['relative_error']*100:.2f}%")
    
    # For cross-reconstruction mode, also compare to 3D target
    if mode == 'cross' and not torch.equal(velocity, velocity_target):
        metrics_target = compute_metrics(velocity_target_masked, reconstructed, mask)
        print("\nCompared to 3D TARGET (cross-reconstruction):")
        print(f"  MAE (overall): {metrics_target['mae']:.6f}")
        print(f"  MAE (u): {metrics_target['mae_u']:.6f}")
        print(f"  MAE (v): {metrics_target['mae_v']:.6f}")
        print(f"  MAE (w): {metrics_target['mae_w']:.6f}")
        print(f"  Relative error: {metrics_target['relative_error']*100:.2f}%")
    
    # --- Visualize ---
    print("\n--- Generating Visualizations ---")
    
    # Convert to numpy for plotting
    velocity_np = velocity_masked[0].cpu().numpy()  # (3, D, H, W)
    reconstructed_np = reconstructed[0].cpu().numpy()
    latent_np = latent[0].cpu().numpy()  # (latent_channels, D', H', W')
    mask_np = mask[0, 0].cpu().numpy()  # (D, H, W)
    
    # Determine depth slice
    depth = velocity_np.shape[1]
    depth_slice = min(args.depth_slice, depth - 1)
    depth_slice = max(0, depth_slice)
    
    # 1. Latent space visualization
    fig_latent = visualize_latent_space(
        latent_np,
        depth_slice=min(depth_slice, latent_np.shape[1] - 1),
        title=f"Latent Space Representation ({model_type}, mode={mode})"
    )
    
    # 2. Reconstruction comparison
    fig_comparison = visualize_reconstruction_comparison(
        velocity_np.transpose(1, 0, 2, 3),  # (D, 3, H, W)
        reconstructed_np.transpose(1, 0, 2, 3),
        mask_np,
        depth_slice=depth_slice
    )
    
    # 3. All depth slices for w component (often the most challenging)
    n_depths = min(velocity_np.shape[1], 6)
    depth_indices = np.linspace(0, velocity_np.shape[1] - 1, n_depths, dtype=int)
    
    # Compute global vmin/vmax for w component across all depth slices
    all_orig_w = velocity_np[2]  # (D, H, W)
    all_recon_w = reconstructed_np[2]
    all_orig_w_masked = np.ma.masked_where(mask_np == 0, all_orig_w)
    all_recon_w_masked = np.ma.masked_where(mask_np == 0, all_recon_w)
    w_vmin = min(all_orig_w_masked.min(), all_recon_w_masked.min())
    w_vmax = max(all_orig_w_masked.max(), all_recon_w_masked.max())
    w_vabs = max(abs(w_vmin), abs(w_vmax))
    w_vmin, w_vmax = -w_vabs, w_vabs
    
    fig_w, axes = plt.subplots(2, n_depths, figsize=(3*n_depths, 6))
    for i, d in enumerate(depth_indices):
        # Original w
        orig_w = velocity_np[2, d]
        mask_d = mask_np[d]
        orig_w_masked = np.ma.masked_where(mask_d == 0, orig_w)
        
        im = axes[0, i].imshow(orig_w_masked, cmap='coolwarm', origin='lower', vmin=w_vmin, vmax=w_vmax)
        axes[0, i].set_title(f"Input w (d={d})")
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)
        
        # Reconstructed w
        recon_w = reconstructed_np[2, d]
        recon_w_masked = np.ma.masked_where(mask_d == 0, recon_w)
        
        im = axes[1, i].imshow(recon_w_masked, cmap='coolwarm', origin='lower', vmin=w_vmin, vmax=w_vmax)
        axes[1, i].set_title(f"Recon w (d={d})")
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)
    
    plt.suptitle(f"W-Component Across Depth Slices ({model_type}, mode={mode})")
    plt.tight_layout()
    
    # Save or show
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        suffix = f"_{model_type}_{mode}"
        fig_latent.savefig(os.path.join(args.save_dir, f'latent_space_sample{args.index}{suffix}.png'), dpi=150)
        fig_comparison.savefig(os.path.join(args.save_dir, f'reconstruction_sample{args.index}{suffix}.png'), dpi=150)
        fig_w.savefig(os.path.join(args.save_dir, f'w_component_sample{args.index}{suffix}.png'), dpi=150)
        print(f"\nFigures saved to: {args.save_dir}")
    else:
        plt.show()
    
    print("\n--- Done ---")


if __name__ == '__main__':
    main()