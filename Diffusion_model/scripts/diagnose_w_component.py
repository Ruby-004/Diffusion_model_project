"""
Diagnostic script to investigate why the w (z-direction velocity) component
looks like a uniform block while u/v components look correct.

This script performs the following investigations:
1. VAE sanity check: encode/decode ground truth and check if w is preserved
2. Channel mapping audit: verify consistent ordering end-to-end
3. Per-component statistics: compare GT vs prediction stats
4. Channel swap test: check if blocky artifact follows channel index
"""

import argparse
import sys
import os
import os.path as osp
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

diffusion_model_path = os.path.join(project_root, 'Diffusion_model')
if diffusion_model_path not in sys.path:
    sys.path.append(diffusion_model_path)

from Diffusion_model.src.helper import set_model
from Diffusion_model.utils.dataset import get_loader
from VAE_model.src.vae.autoencoder import VariationalAutoencoder


def compute_component_stats(velocity: torch.Tensor, name: str = ""):
    """
    Compute and print per-component statistics.
    
    Args:
        velocity: Shape (batch, slices, 3, H, W) or (slices, 3, H, W)
    """
    if velocity.dim() == 4:
        velocity = velocity.unsqueeze(0)
    
    # Flatten spatial dimensions
    v_flat = velocity.view(velocity.shape[0], velocity.shape[1], 3, -1)
    
    components = ['u (vx)', 'v (vy)', 'w (vz)']
    print(f"\n{'='*60}")
    print(f"Statistics for: {name}")
    print(f"{'='*60}")
    print(f"Shape: {velocity.shape}")
    
    for i, comp in enumerate(components):
        comp_data = velocity[:, :, i, :, :]
        print(f"\n{comp}:")
        print(f"  Mean:    {comp_data.mean().item():12.6f}")
        print(f"  Std:     {comp_data.std().item():12.6f}")
        print(f"  Min:     {comp_data.min().item():12.6f}")
        print(f"  Max:     {comp_data.max().item():12.6f}")
        print(f"  |Mean|:  {comp_data.abs().mean().item():12.6f}")
    
    return {
        'mean': [velocity[:, :, i].mean().item() for i in range(3)],
        'std': [velocity[:, :, i].std().item() for i in range(3)],
        'min': [velocity[:, :, i].min().item() for i in range(3)],
        'max': [velocity[:, :, i].max().item() for i in range(3)],
    }


def test_vae_reconstruction(vae, velocity_gt, mask=None, device='cuda'):
    """
    Test 1: VAE sanity check - encode and decode ground truth.
    
    If w_rec is already blocky, the issue is in the VAE/data.
    If w_rec looks correct, the issue is in diffusion/conditioning.
    """
    print("\n" + "="*80)
    print("TEST 1: VAE RECONSTRUCTION SANITY CHECK")
    print("="*80)
    
    # velocity_gt shape: (batch, slices, 3, H, W)
    if velocity_gt.dim() == 4:
        velocity_gt = velocity_gt.unsqueeze(0)
    
    velocity_gt = velocity_gt.to(device)
    
    # Permute to VAE format: (batch, channels, depth, H, W)
    velocity_5d = velocity_gt.permute(0, 2, 1, 3, 4)  # (batch, 3, slices, H, W)
    
    print(f"Input shape to VAE: {velocity_5d.shape}")
    
    with torch.no_grad():
        # Encode
        latent, (mean, logvar) = vae.encode(velocity_5d)
        print(f"Latent shape: {latent.shape}")
        
        # Decode
        recon_5d = vae.decode(latent)
        print(f"Reconstruction shape: {recon_5d.shape}")
    
    # Permute back: (batch, slices, 3, H, W)
    recon = recon_5d.permute(0, 2, 1, 3, 4)
    
    # Apply mask if provided
    if mask is not None:
        if mask.dim() == 4:
            mask = mask.unsqueeze(0)
        mask = mask.to(device)
        recon = recon * mask
        velocity_gt = velocity_gt * mask
    
    # Compute stats
    gt_stats = compute_component_stats(velocity_gt, "Ground Truth")
    recon_stats = compute_component_stats(recon, "VAE Reconstruction")
    
    # Compute per-component reconstruction error
    print("\n" + "-"*60)
    print("Per-component reconstruction error (MAE):")
    print("-"*60)
    
    components = ['u (vx)', 'v (vy)', 'w (vz)']
    recon_errors = []
    for i, comp in enumerate(components):
        error = (velocity_gt[:, :, i] - recon[:, :, i]).abs().mean().item()
        recon_errors.append(error)
        print(f"  {comp}: {error:.6f}")
    
    # Check if w reconstruction is significantly worse
    if recon_errors[2] > 2 * max(recon_errors[0], recon_errors[1]):
        print("\n⚠️  WARNING: w reconstruction error is significantly higher!")
        print("    This suggests the issue may be in the VAE or data.")
    else:
        print("\n✓ w reconstruction error is comparable to u/v.")
        print("  Issue is likely in the diffusion model or conditioning.")
    
    return recon, gt_stats, recon_stats


def test_channel_consistency(vae, velocity_gt, device='cuda'):
    """
    Test 2: Channel mapping consistency check.
    
    Create a synthetic velocity where each component has distinct values:
    - u = 0.1 (low)
    - v = 0.5 (medium) 
    - w = 0.9 (high)
    
    Then verify the ordering is preserved through encode/decode.
    """
    print("\n" + "="*80)
    print("TEST 2: CHANNEL MAPPING CONSISTENCY")
    print("="*80)
    
    if velocity_gt.dim() == 4:
        velocity_gt = velocity_gt.unsqueeze(0)
    
    # Create synthetic velocity with distinct component values
    batch, slices, channels, H, W = velocity_gt.shape
    synthetic = torch.zeros(1, slices, 3, H, W, device=device)
    synthetic[:, :, 0, :, :] = 0.1  # u
    synthetic[:, :, 1, :, :] = 0.5  # v
    synthetic[:, :, 2, :, :] = 0.9  # w
    
    print("Synthetic input values: u=0.1, v=0.5, w=0.9")
    
    # Encode/decode
    synthetic_5d = synthetic.permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        latent, _ = vae.encode(synthetic_5d)
        recon_5d = vae.decode(latent)
    recon = recon_5d.permute(0, 2, 1, 3, 4)
    
    # Check mean values after reconstruction
    print("\nReconstruction mean values per component:")
    for i, comp in enumerate(['u', 'v', 'w']):
        mean_val = recon[:, :, i].mean().item()
        expected = [0.1, 0.5, 0.9][i]
        print(f"  {comp}: {mean_val:.4f} (expected ~{expected})")
        
        if abs(mean_val - expected) > 0.2:
            print(f"    ⚠️  WARNING: Large deviation from expected value!")
    
    # Check if channels are swapped
    means = [recon[:, :, i].mean().item() for i in range(3)]
    expected_order = sorted(range(3), key=lambda i: [0.1, 0.5, 0.9][i])
    actual_order = sorted(range(3), key=lambda i: means[i])
    
    if expected_order == actual_order:
        print("\n✓ Channel ordering is preserved through VAE.")
    else:
        print(f"\n⚠️  WARNING: Channel ordering may be swapped!")
        print(f"   Expected order: {expected_order}")
        print(f"   Actual order: {actual_order}")
    
    return recon


def test_channel_swap(prediction, name="Prediction"):
    """
    Test 3: Channel swap test.
    
    Check which channel index has the blocky/uniform artifact.
    """
    print("\n" + "="*80)
    print("TEST 3: CHANNEL CHARACTERISTICS")
    print("="*80)
    
    if prediction.dim() == 4:
        prediction = prediction.unsqueeze(0)
    
    components = ['Channel 0 (u/vx)', 'Channel 1 (v/vy)', 'Channel 2 (w/vz)']
    
    print(f"\n{name} - Per-channel variance analysis:")
    print("-"*60)
    
    for i, comp in enumerate(components):
        comp_data = prediction[:, :, i, :, :]
        
        # Compute spatial variance (variance across H, W)
        spatial_var = comp_data.var(dim=(-2, -1)).mean().item()
        
        # Compute depth variance (variance across slices)
        depth_var = comp_data.var(dim=1).mean().item()
        
        # Compute overall variance
        overall_var = comp_data.var().item()
        
        print(f"\n{comp}:")
        print(f"  Overall variance: {overall_var:.8f}")
        print(f"  Spatial variance: {spatial_var:.8f}")
        print(f"  Depth variance:   {depth_var:.8f}")
        
        # Flag if variance is suspiciously low
        if overall_var < 1e-6:
            print(f"  ⚠️  WARNING: Very low variance - may be nearly constant!")


def test_diffusion_prediction(predictor, img, velocity_2d, velocity_gt, device='cuda'):
    """
    Test 4: Run diffusion prediction and compare to GT.
    """
    print("\n" + "="*80)
    print("TEST 4: DIFFUSION MODEL PREDICTION")
    print("="*80)
    
    if img.dim() == 4:
        img = img.unsqueeze(0)
    if velocity_2d.dim() == 4:
        velocity_2d = velocity_2d.unsqueeze(0)
    if velocity_gt.dim() == 4:
        velocity_gt = velocity_gt.unsqueeze(0)
    
    img = img.to(device)
    velocity_2d = velocity_2d.to(device)
    velocity_gt = velocity_gt.to(device)
    
    print(f"Input shapes:")
    print(f"  Microstructure: {img.shape}")
    print(f"  Velocity 2D: {velocity_2d.shape}")
    print(f"  Velocity GT: {velocity_gt.shape}")
    
    with torch.no_grad():
        prediction = predictor.predict(img, velocity_2d)
    
    print(f"  Prediction: {prediction.shape}")
    
    # Apply mask
    prediction = prediction * img
    velocity_gt = velocity_gt * img
    
    # Compare statistics
    gt_stats = compute_component_stats(velocity_gt, "Ground Truth (masked)")
    pred_stats = compute_component_stats(prediction, "Prediction (masked)")
    
    # Per-component error
    print("\n" + "-"*60)
    print("Per-component prediction error (MAE):")
    print("-"*60)
    
    components = ['u (vx)', 'v (vy)', 'w (vz)']
    for i, comp in enumerate(components):
        error = (velocity_gt[:, :, i] - prediction[:, :, i]).abs().mean().item()
        print(f"  {comp}: {error:.6f}")
    
    # Check variance ratio (pred vs GT)
    print("\n" + "-"*60)
    print("Variance ratio (Prediction / GT):")
    print("-"*60)
    
    for i, comp in enumerate(components):
        gt_var = velocity_gt[:, :, i].var().item()
        pred_var = prediction[:, :, i].var().item()
        ratio = pred_var / (gt_var + 1e-10)
        print(f"  {comp}: {ratio:.4f}")
        
        if ratio < 0.1:
            print(f"    ⚠️  WARNING: Prediction variance is much lower than GT!")
    
    return prediction


def visualize_components(velocity, title, save_path=None, slice_idx=5):
    """Visualize velocity components at a specific slice."""
    if velocity.dim() == 5:
        velocity = velocity[0]  # Remove batch dim
    
    velocity = velocity.cpu().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    components = ['u (vx)', 'v (vy)', 'w (vz)', 'magnitude']
    
    for i, (ax, comp) in enumerate(zip(axes, components)):
        if i < 3:
            data = velocity[slice_idx, i]
        else:
            data = np.sqrt(np.sum(velocity[slice_idx]**2, axis=0))
        
        im = ax.imshow(data, cmap='RdBu_r' if i < 3 else 'magma')
        ax.set_title(f'{comp}\nmin={data.min():.4f}, max={data.max():.4f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'{title} (slice {slice_idx})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Diagnose w component issues")
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained diffusion model')
    parser.add_argument('--vae-path', type=str, required=True, help='Path to VAE model')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--sample-index', type=int, default=0, help='Sample index to use')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-figs', action='store_true', help='Save figures to disk')
    
    args = parser.parse_args()
    device = args.device
    
    print(f"Running diagnostics on device: {device}")
    
    # Load VAE - need to specify channels since they're not in log
    print("\nLoading VAE...")
    # VAE config: 3 input channels (u,v,w velocity), 8 latent channels
    in_channels = 3
    latent_channels = 8
    vae = VariationalAutoencoder.from_directory(
        args.vae_path, 
        device=device,
        in_channels=in_channels,
        latent_channels=latent_channels
    )
    vae.to(device)
    vae.eval()
    print(f"VAE: {in_channels} input channels, {latent_channels} latent channels")
    
    # Load dataset
    print("\nLoading dataset...")
    loaders = get_loader(
        root_dir=args.dataset_dir,
        batch_size=1,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=False,
        seed=2024,
        augment=False,
        use_3d=True
    )
    train_loader, val_loader, test_loader = loaders[0]
    
    # Get sample
    for i, data in enumerate(test_loader):
        if i == args.sample_index:
            img = data['microstructure']
            velocity_2d = data['velocity_input']
            velocity_gt = data['velocity']
            break
    
    print(f"\nLoaded sample {args.sample_index}:")
    print(f"  Microstructure shape: {img.shape}")
    print(f"  Velocity 2D shape: {velocity_2d.shape}")
    print(f"  Velocity GT shape: {velocity_gt.shape}")
    
    # Test 1: VAE reconstruction
    vae_recon, gt_stats, recon_stats = test_vae_reconstruction(
        vae, velocity_gt, mask=img, device=device
    )
    
    # Visualize VAE reconstruction
    if args.save_figs:
        save_dir = osp.join(osp.dirname(args.model_path), 'diagnostics')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None
    
    visualize_components(
        velocity_gt.to(device) * img.to(device), 
        "Ground Truth",
        save_path=osp.join(save_dir, 'gt.png') if save_dir else None
    )
    
    visualize_components(
        vae_recon * img.to(device), 
        "VAE Reconstruction",
        save_path=osp.join(save_dir, 'vae_recon.png') if save_dir else None
    )
    
    # Test 2: Channel consistency
    test_channel_consistency(vae, velocity_gt, device=device)
    
    # Test 3: Channel characteristics
    test_channel_swap(velocity_gt.to(device), "Ground Truth")
    
    # Load diffusion model
    print("\nLoading diffusion model...")
    log_path = osp.join(args.model_path, 'log.json')
    with open(log_path) as f:
        config = json.load(f)
    
    if 'params' in config:
        config = config['params']
    
    predictor_kwargs = config['training']['predictor']
    predictor_kwargs['vae_path'] = args.vae_path
    
    norm_file = osp.join(args.dataset_dir, 'statistics.json')
    predictor = set_model(
        type='latent-diffusion',
        kwargs=predictor_kwargs,
        norm_file=norm_file
    )
    
    weights_file = osp.join(args.model_path, 'model.pt')
    predictor.load_state_dict(torch.load(weights_file, map_location=device))
    predictor.to(device)
    predictor.eval()
    
    # Test 4: Diffusion prediction
    prediction = test_diffusion_prediction(
        predictor, img, velocity_2d, velocity_gt, device=device
    )
    
    test_channel_swap(prediction, "Diffusion Prediction")
    
    visualize_components(
        prediction,
        "Diffusion Prediction",
        save_path=osp.join(save_dir, 'diffusion_pred.png') if save_dir else None
    )
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print("""
Next steps based on results:
1. If VAE reconstruction w is blocky → Issue is in VAE or data
2. If VAE reconstruction w is fine but prediction w is blocky → Issue is in diffusion
3. Check per-component variance ratios
4. Look for channel ordering inconsistencies
    """)


if __name__ == '__main__':
    main()
