"""
Diagnostic script to analyze the source of noisy predictions in the diffusion model.

This script tests:
1. VAE reconstruction quality (encode -> decode)
2. Diffusion scheduler behavior
3. UNet output at different timesteps
4. Full inference loop step by step
"""

import os
import sys
import json
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

from utils.dataset import get_loader
from src.helper import set_model, select_input_output
from src.predictor import LatentDiffusionPredictor
from src.diffusion import DiffusionScheduler
from VAE_model.src.vae.autoencoder import VariationalAutoencoder


def test_vae_reconstruction(vae, velocity_3d, device, conditional=True, norm_factors=None):
    """Test VAE encode/decode cycle."""
    print("\n" + "="*60)
    print("TEST 1: VAE RECONSTRUCTION QUALITY")
    print("="*60)
    
    batch, num_slices, channels, H, W = velocity_3d.shape
    
    # Permute to (batch, channels, depth, H, W)
    vel_5d = velocity_3d.permute(0, 2, 1, 3, 4)
    
    # Normalize velocity if norm_factors provided
    if norm_factors is not None:
        nf = torch.tensor(norm_factors, device=device).view(1, 3, 1, 1, 1)
        vel_5d_norm = vel_5d / nf
        print(f"Normalized input using factors: {norm_factors}")
        print(f"Input (raw) range: [{vel_5d.min().item():.6f}, {vel_5d.max().item():.6f}]")
        print(f"Input (normalized) range: [{vel_5d_norm.min().item():.4f}, {vel_5d_norm.max().item():.4f}]")
    else:
        vel_5d_norm = vel_5d
        print(f"Input range: [{vel_5d.min().item():.6f}, {vel_5d.max().item():.6f}]")
    
    with torch.no_grad():
        # Encode with condition=True (3D flow)
        condition = torch.ones(batch, dtype=torch.bool, device=device) if conditional else None
        latent, _ = vae.encode(vel_5d_norm, condition=condition)
        print(f"Latent shape: {latent.shape}")
        print(f"Latent range: [{latent.min().item():.4f}, {latent.max().item():.4f}]")
        
        # Decode
        recon_norm = vae.decode(latent, condition=condition)
        print(f"Reconstruction (normalized) range: [{recon_norm.min().item():.4f}, {recon_norm.max().item():.4f}]")
        
        # Denormalize if needed
        if norm_factors is not None:
            recon = recon_norm * nf
            print(f"Reconstruction (denormalized) range: [{recon.min().item():.6f}, {recon.max().item():.6f}]")
        else:
            recon = recon_norm
        
        # Compute error on normalized scale (what training uses)
        error_norm = (vel_5d_norm - recon_norm).abs()
        mae_norm = error_norm.mean().item()
        max_err_norm = error_norm.max().item()
        
        print(f"\nNormalized space MAE: {mae_norm:.6f}")
        print(f"Normalized space Max Error: {max_err_norm:.6f}")
        
        # Per-channel stats (normalized)
        for c in range(channels):
            c_error = (vel_5d_norm[:, c] - recon_norm[:, c]).abs()
            print(f"  Channel {c} MAE: {c_error.mean().item():.6f}, Max: {c_error.max().item():.6f}")
    
    return latent, recon


def test_scheduler_behavior(scheduler, latent, device):
    """Test diffusion scheduler forward/backward process."""
    print("\n" + "="*60)
    print("TEST 2: DIFFUSION SCHEDULER BEHAVIOR")
    print("="*60)
    
    # Test forward diffusion at different timesteps
    batch, channels, depth, H, W = latent.shape
    latent_flat = latent.reshape(batch * depth, channels, H, W)
    
    print(f"Latent stats: min={latent_flat.min():.4f}, max={latent_flat.max():.4f}, mean={latent_flat.mean():.4f}")
    
    timesteps = [0, 100, 250, 500, 750, 999]
    for t in timesteps:
        t_tensor = torch.full((latent_flat.shape[0],), t, device=device, dtype=torch.long)
        noise = torch.randn_like(latent_flat)
        
        # Forward diffusion
        noisy = scheduler.q_sample(latent_flat, t_tensor, noise)
        
        # Signal-to-noise ratio
        alpha_bar = scheduler.alphas_cumprod[t].item()
        snr = alpha_bar / (1 - alpha_bar)
        
        print(f"t={t:4d}: alpha_bar={alpha_bar:.4f}, SNR={snr:.4f}, "
              f"noisy range=[{noisy.min().item():.3f}, {noisy.max().item():.3f}]")
    
    # Test reverse process with perfect noise prediction
    print("\n--- Testing reverse diffusion with PERFECT noise prediction ---")
    noise = torch.randn_like(latent_flat)
    t_tensor = torch.full((latent_flat.shape[0],), 999, device=device, dtype=torch.long)
    x_t = scheduler.q_sample(latent_flat, t_tensor, noise)
    
    # If we know the exact noise, we should recover the original
    x_recovered = scheduler.p_sample(noise, x_t, 999, clip_denoised=False)
    recovery_error = (latent_flat - x_recovered).abs().mean().item()
    print(f"Single-step recovery error (t=999 -> t=998): {recovery_error:.6f}")
    
    return True


def test_unet_output_range(predictor, img, velocity_2d, target_latents, device):
    """Test UNet output characteristics."""
    print("\n" + "="*60)
    print("TEST 3: UNET OUTPUT CHARACTERISTICS")
    print("="*60)
    
    batch = img.shape[0]
    num_slices = img.shape[1]
    
    # Get latent dimensions
    latent_depth = target_latents.shape[1]
    latent_channels = target_latents.shape[2]
    latent_h = target_latents.shape[3]
    latent_w = target_latents.shape[4]
    
    # Prepare conditioning
    velocity_2d_permuted = velocity_2d.permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        condition_2d = torch.zeros(batch, dtype=torch.bool, device=device)
        velocity_2d_latent_5d, _ = predictor.vae.encode(velocity_2d_permuted, condition=condition_2d)
    velocity_2d_latent = velocity_2d_latent_5d.permute(0, 2, 1, 3, 4)
    velocity_2d_latent_flat = velocity_2d_latent.reshape(batch * latent_depth, latent_channels, latent_h, latent_w)
    
    # Prepare microstructure features
    img_flat = img.view(batch * num_slices, 1, img.shape[3], img.shape[4])
    feats_flat = predictor.pre_process(img_flat)
    feats_downsampled = torch.nn.functional.interpolate(
        feats_flat, size=(latent_h, latent_w), mode='bilinear', align_corners=False
    )
    feats_3d = feats_downsampled.reshape(batch, num_slices, 1, latent_h, latent_w)
    feats_3d_interp = torch.nn.functional.interpolate(
        feats_3d.permute(0, 2, 1, 3, 4),
        size=(latent_depth, latent_h, latent_w),
        mode='trilinear',
        align_corners=False
    ).permute(0, 2, 1, 3, 4)
    feats_latent_flat = feats_3d_interp.reshape(batch * latent_depth, 1, latent_h, latent_w)
    
    # Test UNet output at different timesteps
    target_latents_flat = target_latents.reshape(batch * latent_depth, latent_channels, latent_h, latent_w)
    
    timesteps = [0, 100, 500, 999]
    for t in timesteps:
        t_tensor = torch.full((batch * latent_depth,), t, device=device, dtype=torch.long)
        noise = torch.randn_like(target_latents_flat)
        
        predictor.scheduler.to(device)
        x_t = predictor.scheduler.q_sample(target_latents_flat, t_tensor, noise)
        
        unet_input = torch.cat([x_t, velocity_2d_latent_flat, feats_latent_flat], dim=1)
        
        with torch.no_grad():
            noise_pred = predictor.model(unet_input, t_tensor)
        
        # Compare predicted noise to actual noise
        noise_error = (noise_pred - noise).abs()
        print(f"t={t:4d}: predicted noise range=[{noise_pred.min().item():.3f}, {noise_pred.max().item():.3f}], "
              f"MAE to true noise={noise_error.mean().item():.4f}")
    
    return True


def test_full_inference_stepwise(predictor, img, velocity_2d, target_latents, device, num_steps=10):
    """Test inference loop and track denoising progress."""
    print("\n" + "="*60)
    print("TEST 4: STEPWISE INFERENCE ANALYSIS")
    print("="*60)
    
    batch = img.shape[0]
    num_slices = img.shape[1]
    
    # Get latent dimensions
    latent_depth = target_latents.shape[1]
    latent_channels = target_latents.shape[2]
    latent_h = target_latents.shape[3]
    latent_w = target_latents.shape[4]
    
    # Prepare conditioning (same as in forward)
    velocity_2d_permuted = velocity_2d.permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        condition_2d = torch.zeros(batch, dtype=torch.bool, device=device)
        velocity_2d_latent_5d, _ = predictor.vae.encode(velocity_2d_permuted, condition=condition_2d)
    velocity_2d_latent = velocity_2d_latent_5d.permute(0, 2, 1, 3, 4)
    velocity_2d_latent_flat = velocity_2d_latent.reshape(batch * latent_depth, latent_channels, latent_h, latent_w)
    
    # Prepare microstructure
    img_flat = img.view(batch * num_slices, 1, img.shape[3], img.shape[4])
    feats_flat = predictor.pre_process(img_flat)
    feats_downsampled = torch.nn.functional.interpolate(
        feats_flat, size=(latent_h, latent_w), mode='bilinear', align_corners=False
    )
    feats_3d = feats_downsampled.reshape(batch, num_slices, 1, latent_h, latent_w)
    feats_3d_interp = torch.nn.functional.interpolate(
        feats_3d.permute(0, 2, 1, 3, 4),
        size=(latent_depth, latent_h, latent_w),
        mode='trilinear',
        align_corners=False
    ).permute(0, 2, 1, 3, 4)
    feats_latent_flat = feats_3d_interp.reshape(batch * latent_depth, 1, latent_h, latent_w)
    
    # Start from pure noise
    x = torch.randn(batch * latent_depth, latent_channels, latent_h, latent_w, device=device)
    
    predictor.scheduler.to(device)
    
    # Track progress at intervals
    target_latents_flat = target_latents.reshape(batch * latent_depth, latent_channels, latent_h, latent_w)
    
    check_points = [999, 900, 700, 500, 300, 100, 50, 10, 1, 0]
    print(f"\nStarting inference from pure noise...")
    print(f"Initial x range: [{x.min().item():.3f}, {x.max().item():.3f}]")
    
    for t in reversed(range(0, predictor.num_timesteps)):
        t_batch = torch.full((batch * latent_depth,), t, device=device, dtype=torch.long)
        
        unet_input = torch.cat([x, velocity_2d_latent_flat, feats_latent_flat], dim=1)
        
        with torch.no_grad():
            noise_pred = predictor.model(unet_input, t_batch)
        
        x = predictor.scheduler.p_sample(noise_pred, x, t)
        
        if t in check_points:
            # Compare to target latent
            error_to_target = (x - target_latents_flat).abs().mean().item()
            print(f"t={t:4d}: x range=[{x.min().item():.4f}, {x.max().item():.4f}], "
                  f"error to target latent={error_to_target:.4f}")
    
    print("\nFinal latent comparison:")
    print(f"Predicted latent range: [{x.min().item():.4f}, {x.max().item():.4f}]")
    print(f"Target latent range: [{target_latents_flat.min().item():.4f}, {target_latents_flat.max().item():.4f}]")
    
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True, help='Path to trained model directory')
    parser.add_argument('--vae-path', type=str, required=True, help='Path to VAE')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model config
    log_path = os.path.join(args.model_dir, 'log.json')
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    params = log_data['params']
    predictor_kwargs = params['training']['predictor']
    
    # Override VAE path
    predictor_kwargs['vae_path'] = args.vae_path
    
    # Load predictor
    print("\nLoading model...")
    predictor = LatentDiffusionPredictor(**predictor_kwargs)
    predictor.to(device)
    
    model_path = os.path.join(args.model_dir, 'model.pt')
    state_dict = torch.load(model_path, map_location=device)
    
    # Filter out scheduler keys if format mismatch (old vs new scheduler)
    scheduler_keys_new = [k for k in state_dict.keys() if k.startswith('scheduler.')]
    model_scheduler_keys = [k for k in predictor.state_dict().keys() if k.startswith('scheduler.')]
    
    if set(scheduler_keys_new) != set(model_scheduler_keys):
        print("Scheduler format mismatch - filtering scheduler keys and reinitializing")
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('scheduler.')}
    
    predictor.load_state_dict(state_dict, strict=False)
    predictor.eval()
    
    print(f"Model parameters: {predictor.trainable_params:,}")
    
    # Load a test sample
    print("\nLoading test data...")
    loaders = get_loader(
        root_dir=args.dataset_dir,
        batch_size=1,
        augment=False,
        shuffle=False,
        use_3d=True
    )
    # get_loader returns a list of tuples [(train_loader, val_loader, test_loader)]
    train_loader, val_loader, test_loader = loaders[0]
    loader = test_loader
    
    data = next(iter(loader))
    imgs = data['microstructure'].to(device)
    velocity_2d = data['velocity_input'].to(device)
    velocity_3d = data['velocity'].to(device)
    
    print(f"Microstructure shape: {imgs.shape}")
    print(f"Velocity 2D shape: {velocity_2d.shape}")
    print(f"Velocity 3D target shape: {velocity_3d.shape}")
    print(f"Velocity 3D range: [{velocity_3d.min().item():.6f}, {velocity_3d.max().item():.6f}]")
    
    # Encode target for comparison
    target_latents = predictor.encode_target(velocity_3d, velocity_2d)
    print(f"Target latent shape: {target_latents.shape}")
    print(f"Target latent range: [{target_latents.min().item():.4f}, {target_latents.max().item():.4f}]")
    
    # Get VAE norm_factors from predictor's resolved VAE path
    # The predictor converts relative VAE paths to absolute, so get the resolved path
    # If relative path, convert to absolute from project root (one level up from Diffusion_model/)
    if not os.path.isabs(args.vae_path):
        # __file__ = Diffusion_model/scripts/diagnose_noise.py
        # Go up 3 levels: scripts -> Diffusion_model -> project_root
        script_dir = os.path.dirname(os.path.abspath(__file__))  # .../Diffusion_model/scripts
        diffusion_dir = os.path.dirname(script_dir)  # .../Diffusion_model
        project_root = os.path.dirname(diffusion_dir)  # ...project_root
        vae_path_abs = os.path.join(project_root, args.vae_path)
    else:
        vae_path_abs = args.vae_path
    
    vae_log_path = os.path.join(vae_path_abs, 'vae_log.json')
    
    vae_norm_factors = None
    if os.path.exists(vae_log_path):
        print(f"Loading VAE log from: {vae_log_path}")
        with open(vae_log_path, 'r') as f:
            vae_log = json.load(f)
        print(f"VAE log keys: {list(vae_log.keys())}")
        if 'norm_factors' in vae_log:
            vae_norm_factors = vae_log['norm_factors']
            print(f"VAE norm_factors found: {vae_norm_factors}")
        else:
            print("WARNING: 'norm_factors' key not found in VAE log")
    else:
        print(f"WARNING: VAE log file not found at {vae_log_path}")
    
    # Run tests
    test_vae_reconstruction(predictor.vae, velocity_3d, device, conditional=predictor.vae_conditional, norm_factors=vae_norm_factors)
    test_scheduler_behavior(predictor.scheduler, target_latents.permute(0, 2, 1, 3, 4), device)
    test_unet_output_range(predictor, imgs, velocity_2d, target_latents, device)
    test_full_inference_stepwise(predictor, imgs, velocity_2d, target_latents, device)
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("\nKey findings:")
    print("- If VAE normalized MAE > 0.01: VAE may need retraining")
    print("- If final latent error > 2x target range: Model needs retraining with fixed scheduler")
    print("- If latent values explode during inference: Old scheduler bug (now fixed)")
    print("\nRecommendations:")
    print("1. Retrain model with the fixed diffusion scheduler")
    print("2. Use deeper UNet: --features 64 128 256 512 1024")
    print("3. Add attention: --attention '3..2'")
    print("4. Use DDIM sampling for faster inference: predictor.predict_ddim()")


if __name__ == '__main__':
    main()
