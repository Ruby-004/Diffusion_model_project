"""
Stage 1: Train only 3D VAE (E3D + D3D)

This trains a standard VAE on 3D velocity fields only.
Stage 2 will load this and add 2D components.
"""

import time
import os
import json
import os.path as osp
import argparse
import sys

import torch
import torch.optim as optim

from src.vae.autoencoder import VariationalAutoencoder
from utils.dataset import get_loader, MicroFlowDatasetVAE
from utils.metrics import normalized_mae_loss, kl_divergence, mae_loss_per_channel, normalized_mae_loss_per_channel, normalized_mse_per_channel
from torch.utils.data import DataLoader, Subset

# Force unbuffered output to see prints before crashes
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D VAE (Stage 1)')
    
    parser.add_argument('--dataset-dir', type=str, default='C:/Users/alexd/Downloads/dataset_3d')
    parser.add_argument('--save-dir', type=str, default='trained/dual_vae_stage1_3d')
    parser.add_argument('--in-channels', type=int, default=3)
    parser.add_argument('--latent-channels', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--conditional', action='store_true', help='Use conditional VAE')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--loss-function', type=str, default='normalized_mae_per_channel', 
                        choices=['mae_per_channel', 'normalized_mae_per_channel', 'normalized_mse_per_channel'],
                        help='Reconstruction loss function (default: normalized_mae_per_channel)')
    parser.add_argument('--debug-latent', action='store_true', help='Log mu/logvar statistics for first few batches')
    parser.add_argument('--debug-batches', type=int, default=5, help='Number of batches to log debug stats (default: 5)')
    parser.add_argument('--use-split-file', type=str, default=None, help='Path to splits.json for reproducible train/val/test split')
    parser.add_argument('--split-seed', type=int, default=2024, help='Seed for data split (default: 2024)')
    parser.add_argument('--norm-mode', type=str, default='max', choices=['max', 'mean'],
                        help='Normalization mode: max (default) or mean velocity per component')
    
    args = parser.parse_args()
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def check_tensor_health(tensor: torch.Tensor, name: str, fail_on_bad: bool = True) -> bool:
    """
    Check tensor for NaN/Inf values and optionally fail with clear message.
    
    Args:
        tensor: Tensor to check
        name: Name for error messages
        fail_on_bad: If True, raise exception on NaN/Inf
        
    Returns:
        True if tensor is healthy, False otherwise
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        msg = f"FATAL: {name} contains {'NaN' if has_nan else ''}{'/' if has_nan and has_inf else ''}{'Inf' if has_inf else ''}"
        msg += f"\n  Shape: {tensor.shape}"
        msg += f"\n  Min: {tensor.min().item():.6e}, Max: {tensor.max().item():.6e}"
        msg += f"\n  NaN count: {torch.isnan(tensor).sum().item()}, Inf count: {torch.isinf(tensor).sum().item()}"
        
        if fail_on_bad:
            raise RuntimeError(msg)
        else:
            print(f"WARNING: {msg}")
            return False
    return True


def log_latent_stats(mu: torch.Tensor, logvar: torch.Tensor, prefix: str = ""):
    """
    Log statistics of mu and logvar for debugging KL divergence issues.
    
    Args:
        mu: Mean tensor from encoder
        logvar: Log-variance tensor from encoder  
        prefix: Prefix for log messages (e.g., "Train" or "Val")
    """
    print(f"  {prefix} Latent Stats:")
    print(f"    mu    - min: {mu.min().item():+.4f}, max: {mu.max().item():+.4f}, "
          f"mean: {mu.mean().item():+.4f}, std: {mu.std().item():.4f}")
    print(f"    logvar - min: {logvar.min().item():+.4f}, max: {logvar.max().item():+.4f}, "
          f"mean: {logvar.mean().item():+.4f}, std: {logvar.std().item():.4f}")
    
    # Check for potential KL explosion indicators
    exp_logvar_max = torch.exp(logvar).max().item()
    if exp_logvar_max > 1e6:
        print(f"    WARNING: exp(logvar) max = {exp_logvar_max:.2e} - potential KL explosion!")
    
    # Compute raw KL components for diagnostics
    kl_mu_term = (mu.pow(2)).mean().item()
    kl_logvar_term = logvar.mean().item()
    kl_exp_term = logvar.exp().mean().item()
    print(f"    KL components: mu²={kl_mu_term:.4f}, logvar={kl_logvar_term:.4f}, exp(logvar)={kl_exp_term:.4f}")


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("STAGE 1: Training 3D VAE Only")
    print(f"{'='*60}\n")
    
    device = args.device
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    
    # Verify dataset exists
    if not os.path.exists(args.dataset_dir):
        print(f"ERROR: Dataset directory not found: {args.dataset_dir}")
        exit(1)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # KL annealing parameters
    kl_warmup_epochs = 10  # Gradually increase KL weight over first 20 epochs
    max_kl_coeff = 1e-3  # Maximum KL coefficient
    gradient_accumulation_steps = 10  # Accumulate gradients to simulate larger batch
    
    # Clear CUDA cache before training
    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

    # Load data - first get full dataset to filter 3D samples only
    print(f"\nLoading dataset from: {args.dataset_dir}")
    full_dataset = MicroFlowDatasetVAE(root_dir=args.dataset_dir, augment=args.augment)
    
    # Filter to keep only 3D samples (is_2d=False)
    print("Filtering for 3D samples only...")
    indices_3d = []
    for i in range(len(full_dataset)):
        sample = full_dataset[i]
        if not sample['is_2d']:
            indices_3d.append(i)
    
    print(f"Total samples: {len(full_dataset)}, 3D samples: {len(indices_3d)}")
    
    # Create subset with only 3D samples
    dataset_3d = Subset(full_dataset, indices_3d)
    
    # 70/15/15 split on 3D samples only
    num_samples = len(indices_3d)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size
    
    print(f"\nDataset split (3D samples only):")
    print(f"  Total 3D samples: {num_samples}")
    print(f"  Train: {train_size} (70%)")
    print(f"  Val: {val_size} (15%)")
    print(f"  Test: {test_size} (15%)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Val batches: ~{val_size // args.batch_size} (may be only 1 batch if val_size < 2*batch_size)")
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset_3d, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(2024)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Verify we're only getting 3D data
    print("\nVerifying dataset contains only 3D samples...")
    sample_batch = next(iter(train_loader))
    if sample_batch['is_2d'].any():
        print("WARNING: Found 2D samples in training data!")
    else:
        print("✓ All samples are 3D (is_2d=False)")

    # Load normalization statistics
    stats_file = osp.join(args.dataset_dir, 'statistics.json')
    if not os.path.exists(stats_file):
        print(f"ERROR: statistics.json not found at {stats_file}")
        exit(1)

    with open(stats_file, 'r') as f:
        statistics = json.load(f)

    # Per-component normalization for better w-component learning (always enabled)
    use_per_component = True
    norm_mode = args.norm_mode
    
    if use_per_component and 'U_per_component' in statistics:
        pc = statistics['U_per_component']
        pc_2d = statistics.get('U_2d_per_component', {})
        
        # Use max or mean of U and U_2d for each component based on norm_mode
        if norm_mode == 'max':
            norm_u = max(pc['max_u'], pc_2d.get('max_u', 0))
            norm_v = max(pc['max_v'], pc_2d.get('max_v', 0))
            norm_w = max(pc['max_w'], pc_2d.get('max_w', 0))
            stat_key = 'max'
        else:  # mean
            norm_u = max(pc.get('mean_u', pc['max_u']), pc_2d.get('mean_u', pc_2d.get('max_u', 0)))
            norm_v = max(pc.get('mean_v', pc['max_v']), pc_2d.get('mean_v', pc_2d.get('max_v', 0)))
            norm_w = max(pc.get('mean_w', pc['max_w']), pc_2d.get('mean_w', pc_2d.get('max_w', 0)))
            stat_key = 'mean'
        
        # Create per-component normalization tensor [3] for (u, v, w)
        norm_factors = torch.tensor([norm_u, norm_v, norm_w], dtype=torch.float32)
        
        print(f"\n=== Per-Component Normalization ({norm_mode.upper()}) ===")
        print(f"  {stat_key}_u (vx): {norm_u:.6f}")
        print(f"  {stat_key}_v (vy): {norm_v:.6f}")
        print(f"  {stat_key}_w (vz): {norm_w:.6f}")
        print(f"  Ratio {stat_key}_u/{stat_key}_w: {norm_u/norm_w:.2f}x")
        print(f"===================================\n")
    else:
        # Fallback to global max (legacy behavior)
        if 'U_2d' in statistics:
            max_U_2d = statistics['U_2d']['max']
        else:
            print("WARNING: 'U_2d' key not found in statistics.json. Using 'U' for max_U_2d.")
            max_U_2d = statistics['U']['max']

        max_U_3d = statistics['U']['max']
        max_velocity = max(max_U_2d, max_U_3d)
        
        # Create uniform normalization tensor
        norm_factors = torch.tensor([max_velocity, max_velocity, max_velocity], dtype=torch.float32)
        
        print(f"\n=== Global Normalization (Legacy) ===")
        print(f"  max_U_2d={max_U_2d:.6f}, max_U_3d={max_U_3d:.6f}")
        print(f"  Using max_velocity={max_velocity:.6f} for all components")
        if 'U_per_component' in statistics:
            print(f"  TIP: Use --per-component-norm for better w-component training")
        print(f"======================================\n")
    
    # Conditional VAE mode
    use_conditional = args.conditional
    if use_conditional:
        print(f"\n=== Conditional VAE Mode ===")
        print(f"  WARNING: Stage 1 trains only on 3D samples, conditional mode not needed")
        print(f"  Setting conditional=False for stage 1")
        print(f"==============================\n")
        use_conditional = False  # Override for stage 1

    # Create VAE with consistent naming (encoder_3d/decoder_3d)
    # This allows Stage 2 to load weights directly without key remapping
    print(f"\nCreating 3D VAE (E3D + D3D)...")
    base_vae = VariationalAutoencoder(
        in_channels=args.in_channels,
        latent_channels=args.latent_channels,
        conditional=use_conditional  # False for stage 1
    )
    
    # Wrap VAE to use consistent naming with Stage 2 (encoder_3d/decoder_3d)
    class VAE3DWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.encoder_3d = vae.encoder
            self.decoder_3d = vae.decoder
            # Logvar clamping bounds to prevent KL explosion
            self.logvar_min = -10.0
            self.logvar_max = 10.0
        
        def forward(self, x, condition=None):
            mean, logvar = self.encoder_3d(x, condition)
            # CRITICAL: Clamp logvar to prevent KL explosion
            # Without this, large logvar values cause exp(logvar) -> Inf -> KL = Inf
            # This was the root cause of KL explosion in validation (training had external clamping)
            logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
            z = self.encoder_3d.sample(mean, logvar)
            recon = self.decoder_3d(z, condition)
            return recon, (mean, logvar)
        
        def save_model(self, folder, log=None):
            """Save model checkpoint and training log."""
            import os
            os.makedirs(folder, exist_ok=True)
            
            # Save state dict
            model_path = os.path.join(folder, 'vae.pt')
            torch.save(self.state_dict(), model_path)
            
            # Save log if provided
            if log is not None:
                log_path = os.path.join(folder, 'vae_log.json')
                import json
                with open(log_path, 'w') as f:
                    json.dump(log, f, indent=2)
    
    vae = VAE3DWrapper(base_vae).to(device)
    
    # Enable multi-GPU training if available
    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        vae = torch.nn.DataParallel(vae)
    
    print(f"Model loaded on {device}")
    if device == "cuda":
        print(f"GPU Memory after model load: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    # Select loss function
    loss_functions = {
        'mae_per_channel': mae_loss_per_channel,
        'normalized_mae_per_channel': normalized_mae_loss_per_channel,
        'normalized_mse_per_channel': normalized_mse_per_channel
    }
    reconstruction_loss_fn = loss_functions[args.loss_function]
    print(f"Using reconstruction loss: {args.loss_function}")
    
    log_dict = {
        'loss': {'recons_train': [], 'recons_val': [], 'kl_train':[], 'kl_val':[], 'kl_coeff': []},
        'in_channels': args.in_channels,
        'latent_channels': args.latent_channels,
        'per_component_norm': use_per_component,
        'norm_mode': norm_mode,  # 'max' or 'mean'
        'norm_factors': norm_factors.tolist(),  # [norm_u, norm_v, norm_w] for decoding
        'conditional': use_conditional,  # Whether VAE uses conditioning
        'loss_function': args.loss_function,  # Which reconstruction loss function is used
    }
    
    best_val_loss = float('inf')  # Track best validation loss for saving best model
    
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")
    
    for epoch in range(args.num_epochs):
        
        start_time = time.time()
        
        # KL annealing: gradually increase KL weight
        # Start from small value (1e-5) instead of 0 to prevent KL explosion
        min_kl_coeff = 1e-5
        if epoch < kl_warmup_epochs:
            kl_coeff = min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (epoch / kl_warmup_epochs)
        else:
            kl_coeff = max_kl_coeff
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs} - KL coefficient: {kl_coeff:.6f}")

        """Training set"""
        running_recons = 0
        running_kl = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        i = -1  # Initialize in case loop doesn't run
        for i, data in enumerate(train_loader):
            
            print(f'Training, batch {i}')

            # Data shape: [batch, channels, depth, height, width]
            mask = data['microstructure'].to(device)
            velocity = data['velocity'].to(device)  # 3D flow only (U)
            is_2d = data['is_2d']  # Should all be False in stage 1
            
            # No condition needed - all samples are 3D
            is_3d = None
            
            # Keep 3D structure for Conv3d layers: [batch, 3, depth, height, width]
            inputs = velocity
            targets = inputs.clone()  # Autoencoder: reconstruct the same input
            
            # Normalize to [0,1] using per-component or global normalization
            # norm_factors shape: [3] -> reshape to [1, 3, 1, 1, 1] for broadcasting
            nf = norm_factors.to(device).view(1, 3, 1, 1, 1)
            inputs = inputs / nf
            targets = targets / nf
            
            # Don't pass condition if None (DataParallel issue)
            if is_3d is None:
                preds, (mean, logvar) = vae(inputs)
            else:
                preds, (mean, logvar) = vae(inputs, condition=is_3d)
            
            # Note: logvar is now clamped inside VAE3DWrapper.forward()
            # The external clamping here is kept for extra safety but shouldn't be needed
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            
            # Robust NaN/Inf checking with clear error messages
            if not check_tensor_health(mean, f"mu (batch {i})", fail_on_bad=False):
                print(f"  Skipping batch {i} due to bad mu values")
                continue
            if not check_tensor_health(logvar, f"logvar (batch {i})", fail_on_bad=False):
                print(f"  Skipping batch {i} due to bad logvar values")
                continue
            
            # Debug logging for first few batches of first epoch
            if args.debug_latent and epoch == 0 and i < args.debug_batches:
                log_latent_stats(mean, logvar, prefix=f"Train batch {i}")
            
            preds = preds * mask
            targets = targets * mask

            # Compute reconstruction loss using selected function
            reconstruction_loss = reconstruction_loss_fn(preds, targets, mask=mask)
            kl_loss = kl_divergence(mu=mean, logvar=logvar)
            loss = (reconstruction_loss + kl_coeff * kl_loss) / gradient_accumulation_steps
            
            # All samples should be 3D
            num_2d = is_2d.sum().item()
            num_3d = len(is_2d) - num_2d
            if num_2d > 0:
                print(f'WARNING: Found {num_2d} 2D samples in batch (should be 0)')
            
            # Print every 10 batches to reduce output
            if i % 10 == 0:
                print(f'Batch {i}: {num_3d} 3D samples - Recons/KL: {reconstruction_loss.item():.6f}/{kl_loss.item():.6f}')
                # Monitor if KL is exploding
                if kl_loss.item() > 1000:
                    print(f'  ERROR: KL loss exploded to {kl_loss.item():.2f}! Training unstable.')
                    print(f'  This indicates the VAE latent space is collapsing.')
                    print(f'  Stopping training to prevent further instability.')
                    sys.exit(1)

            loss.backward()
            
            # Clip gradients immediately after backward to prevent explosion
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            
            # Update weights every gradient_accumulation_steps batches
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_recons += reconstruction_loss.item()
            running_kl += kl_loss.item()
            
            # Clear CUDA cache to prevent memory buildup
            if device == "cuda":
                torch.cuda.empty_cache()

        # Apply any remaining accumulated gradients at end of epoch
        if (i + 1) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if i == -1:  # No training batches
            print("ERROR: No training batches found!")
            continue
            
        avg_recons_train = running_recons / (i+1)
        avg_kl_train = running_kl / (i+1)

        log_dict['loss']['recons_train'].append(avg_recons_train)
        log_dict['loss']['kl_train'].append(avg_kl_train)
        log_dict['loss']['kl_coeff'].append(kl_coeff)

        """Validation set"""
        
        print(f"\nStarting validation... Expected ~{len(val_dataset) // args.batch_size} batches")
        sys.stdout.flush()

        with torch.no_grad():

            running_recons = 0
            running_kl = 0
            j = -1  # Initialize in case loop doesn't run
            for j, data in enumerate(val_loader):
                
                print(f'Val batch {j}')
                sys.stdout.flush()
                
                # Data shape: [batch, channels, depth, height, width]
                mask = data['microstructure'].to(device)
                velocity = data['velocity'].to(device)  # 3D flow only (U)
                is_2d = data['is_2d']  # Should all be False in stage 1
                
                # No condition needed - all samples are 3D
                is_3d = None
                
                # Keep 3D structure for Conv3d layers: [batch, 3, depth, height, width]
                inputs = velocity
                targets = inputs.clone()  # Autoencoder: reconstruct the same input
                
                # Normalize to [0,1] using per-component or global normalization
                nf = norm_factors.to(device).view(1, 3, 1, 1, 1)
                inputs = inputs / nf
                targets = targets / nf

                # Don't pass condition if None (DataParallel issue)
                if is_3d is None:
                    preds, (mean, logvar) = vae(inputs)
                else:
                    preds, (mean, logvar) = vae(inputs, condition=is_3d)
                
                # Note: logvar is clamped inside VAE3DWrapper.forward()
                # This was the ROOT CAUSE of KL explosion - validation path was missing clamping
                # Now the model handles it internally, but we add extra safety here
                logvar = torch.clamp(logvar, min=-10.0, max=10.0)
                
                # Check for NaN/Inf in validation too
                if not check_tensor_health(mean, f"val mu (batch {j})", fail_on_bad=False):
                    print(f"  Skipping val batch {j} due to bad mu values")
                    continue
                if not check_tensor_health(logvar, f"val logvar (batch {j})", fail_on_bad=False):
                    print(f"  Skipping val batch {j} due to bad logvar values")
                    continue
                
                # Debug logging for first few validation batches of first epoch
                if args.debug_latent and epoch == 0 and j < args.debug_batches:
                    log_latent_stats(mean, logvar, prefix=f"Val batch {j}")
                
                preds = preds * mask
                targets = targets * mask
                
                # Compute reconstruction loss using selected function
                reconstruction_loss = reconstruction_loss_fn(preds, targets, mask=mask)
                kl_loss = kl_divergence(mu=mean, logvar=logvar)
                
                # Additional KL sanity check
                if kl_loss.item() > 1e6 or torch.isinf(kl_loss):
                    print(f"  WARNING: KL explosion in val batch {j}: {kl_loss.item():.2e}")
                    log_latent_stats(mean, logvar, prefix=f"  EXPLOSION DEBUG")
                    continue
                    
                loss = reconstruction_loss + kl_coeff * kl_loss
                
                # All samples should be 3D
                num_2d = is_2d.sum().item()
                num_3d = len(is_2d) - num_2d
                print(f'Val batch {j}: {num_3d} 3D samples - Recons/KL: {reconstruction_loss.item():.6f}/{kl_loss.item():.6f}')
                sys.stdout.flush()

                running_recons += reconstruction_loss.item()
                running_kl += kl_loss.item()

        # Calculate validation averages (outside the with block for clarity)
        if j == -1:  # No validation batches
            print("ERROR: No validation batches found!")
            avg_recons_val = 0.0
            avg_kl_val = 0.0
        else:
            avg_recons_val = running_recons / (j+1)
            avg_kl_val = running_kl / (j+1)

        log_dict['loss']['recons_val'].append(avg_recons_val)
        log_dict['loss']['kl_val'].append(avg_kl_val)
        
        print(f"DEBUG: Validation complete. avg_recons_val={avg_recons_val:.6f}, avg_kl_val={avg_kl_val:.6f}")
        sys.stdout.flush()

        dtime = time.time() - start_time
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}]: recons/kl_train=[{avg_recons_train:.6f}/{avg_kl_train:.6f}] | recons/kl_val=[{avg_recons_val:.6f}/{avg_kl_val:.6f}] | kl_coeff={kl_coeff:.6f} | time={dtime:.2f} s")
        sys.stdout.flush()
        
        if device == "cuda":
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
        
        print("DEBUG: About to save model...")
        sys.stdout.flush()

        # save model
        # Unwrap DataParallel before saving
        model_to_save = vae.module if isinstance(vae, torch.nn.DataParallel) else vae
        model_to_save.save_model(args.save_dir, log=log_dict)
        
        # Track best model based on validation reconstruction loss
        current_val_loss = avg_recons_val + kl_coeff * avg_kl_val
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            print(f"  ✓ New best model! Val loss: {current_val_loss:.6f}")
            # Save best model as separate file
            best_model_path = osp.join(args.save_dir, 'best_model.pt')
            torch.save(model_to_save.state_dict(), best_model_path)
            print(f"  Saved to: {best_model_path}")
        
        print(f"DEBUG: Model saved successfully to {args.save_dir}")
        sys.stdout.flush()

    # Final test evaluation after all epochs
    print("\n" + "="*50)
    print("Running final test set evaluation...")
    print("="*50)
    
    with torch.no_grad():
        running_recons = 0
        running_kl = 0
        k = -1  # Initialize in case loop doesn't run
        for k, data in enumerate(test_loader):
            mask = data['microstructure'].to(device)
            velocity = data['velocity'].to(device)
            is_2d = data['is_2d']
            
            # No condition needed - all samples are 3D
            is_3d = None
            
            nf = norm_factors.to(device).view(1, 3, 1, 1, 1)
            inputs = velocity / nf
            targets = inputs.clone()
            
            # Don't pass condition if None (DataParallel issue)
            if is_3d is None:
                preds, (mean, logvar) = vae(inputs)
            else:
                preds, (mean, logvar) = vae(inputs, condition=is_3d)
            # logvar is clamped in model
            
            preds = preds * mask
            targets = targets * mask
            
            # Use selected reconstruction loss function (same as training)
            reconstruction_loss = reconstruction_loss_fn(preds, targets, mask=mask)
            kl_loss = kl_divergence(mu=mean, logvar=logvar)
            
            num_2d = is_2d.sum().item()
            num_3d = len(is_2d) - num_2d
            print(f'Test batch {k}: {num_2d} 2D and {num_3d} 3D samples - Recons/KL: {reconstruction_loss.item():.6f}/{kl_loss.item():.6f}')
            
            running_recons += reconstruction_loss.item()
            running_kl += kl_loss.item()
        
        if k == -1:  # No test batches
            print("WARNING: No test batches found!")
            avg_recons_test = 0.0
            avg_kl_test = 0.0
        else:
            avg_recons_test = running_recons / (k+1)
            avg_kl_test = running_kl / (k+1)
        
        log_dict['loss']['recons_test'] = avg_recons_test
        log_dict['loss']['kl_test'] = avg_kl_test
        
        print(f"\nFinal Test Results: recons={avg_recons_test:.6f} | kl={avg_kl_test:.6f}")
        print("="*50)
    
    # Save final model with test results
    model_to_save = vae.module if isinstance(vae, torch.nn.DataParallel) else vae
    model_to_save.save_model(args.save_dir, log=log_dict)
    
    print(f"\n{'='*60}")
    print("STAGE 1 COMPLETE")
    print(f"Model saved to: {args.save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"FATAL ERROR: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
