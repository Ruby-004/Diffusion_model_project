"""
Stage 2: Train 2D VAE (E2D + D2D) with alignment and cross-reconstruction

This loads the stage 1 checkpoint (E3D + D3D) and trains:
- E2D + D2D on 2D samples only (reconstruction)
- Alignment loss between E2D and E3D latents (paired samples)
- Cross-reconstruction: E2D → D3D (2D→3D prediction with frozen D3D)
"""

import time
import os
import json
import os.path as osp
import argparse
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.dual_vae.model import DualBranchVAE, kl_divergence
from utils.dataset import get_loader, MicroFlowDatasetVAE
from utils.metrics import normalized_mae_loss, kl_divergence as kl_div_metric, mae_loss_per_channel, normalized_mae_loss_per_channel, normalized_mse_per_channel
from torch.utils.data import DataLoader, Subset, Dataset

# Force unbuffered output to see prints before crashes
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


class PairedDataset(Dataset):
    """
    Dataset that returns paired 2D and 3D samples from the same microstructure.
    """
    def __init__(self, base_dataset, paired_indices):
        self.base_dataset = base_dataset
        self.paired_indices = paired_indices
    
    def __len__(self):
        return len(self.paired_indices)
    
    def __getitem__(self, idx):
        idx_2d, idx_3d = self.paired_indices[idx]
        sample_2d = self.base_dataset[idx_2d]
        sample_3d = self.base_dataset[idx_3d]
        
        # Verify pairing is correct
        assert sample_2d['original_idx'] == sample_3d['original_idx'], \
            f"Pairing mismatch: 2D={sample_2d['original_idx']}, 3D={sample_3d['original_idx']}"
        
        return {'2d': sample_2d, '3d': sample_3d}


def parse_args():
    parser = argparse.ArgumentParser(description='Train 2D VAE with alignment and cross-reconstruction (Stage 2)')
    
    parser.add_argument('--dataset-dir', type=str, default='C:/Users/alexd/Downloads/dataset_3d')
    parser.add_argument('--save-dir', type=str, default='trained/dual_vae_stage2_2d')
    parser.add_argument('--stage1-checkpoint', type=str, required=True, help='Path to stage 1 checkpoint (E3D+D3D)')
    parser.add_argument('--in-channels', type=int, default=3)
    parser.add_argument('--latent-channels', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size (reduce to 1 if OOM)')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--loss-function', type=str, default='normalized_mae_per_channel', 
                        choices=['mae_per_channel', 'normalized_mae_per_channel', 'normalized_mse_per_channel'],
                        help='Reconstruction loss function (default: normalized_mae_per_channel)')
    
    # Loss weights
    parser.add_argument('--beta-kl', type=float, default=1e-3, help='KL divergence weight')
    parser.add_argument('--lambda-align', type=float, default=0.1, help='Alignment loss weight')
    parser.add_argument('--lambda-cross', type=float, default=1.0, help='Cross-reconstruction (2D->3D) loss weight')
    parser.add_argument('--norm-mode', type=str, default='max', choices=['max', 'mean'],
                        help='Normalization mode: max (default) or mean velocity per component')
    
    args = parser.parse_args()
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("STAGE 2: Training 2D VAE with Alignment & Cross-Reconstruction")
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
    
    # Verify stage 1 checkpoint exists
    if not os.path.exists(args.stage1_checkpoint):
        print(f"ERROR: Stage 1 checkpoint not found: {args.stage1_checkpoint}")
        exit(1)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # KL annealing parameters
    kl_warmup_epochs = 10  # Gradually increase KL weight over first 10 epochs
    max_kl_coeff = args.beta_kl  # Maximum KL coefficient
    gradient_accumulation_steps = 5  # Reduced accumulation steps for memory
    
    print(f"\nMemory optimization settings:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * gradient_accumulation_steps}")
    
    # Clear CUDA cache before training
    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

    # Load data - create properly paired 2D/3D dataset
    print(f"\nLoading dataset from: {args.dataset_dir}")
    full_dataset = MicroFlowDatasetVAE(root_dir=args.dataset_dir, augment=args.augment)
    
    # Create paired indices: match 2D and 3D from same microstructure
    # Dataset structure: indices 0 to N-1 are 2D, indices N to 2N-1 are 3D
    num_microstructures = full_dataset.num_samples_per_field
    print(f"\nCreating paired dataset...")
    print(f"  Total microstructures: {num_microstructures}")
    print(f"  2D samples: indices 0-{num_microstructures-1}")
    print(f"  3D samples: indices {num_microstructures}-{2*num_microstructures-1}")
    
    paired_indices = []
    for i in range(num_microstructures):
        idx_2d = i  # First half of dataset
        idx_3d = i + num_microstructures  # Second half of dataset
        paired_indices.append((idx_2d, idx_3d))
    
    print(f"  Created {len(paired_indices)} paired samples")
    
    # Create paired dataset
    paired_dataset = PairedDataset(full_dataset, paired_indices)
    
    # 70/15/15 split
    num_samples = len(paired_dataset)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        paired_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(2024)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"\nDataset split (paired samples):")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Verify pairing is correct
    print("\nVerifying paired dataset...")
    sample_pair = next(iter(train_loader))
    print(f"  ✓ 2D original_idx: {sample_pair['2d']['original_idx'][:min(3, args.batch_size)]}")
    print(f"  ✓ 3D original_idx: {sample_pair['3d']['original_idx'][:min(3, args.batch_size)]}")
    if torch.equal(sample_pair['2d']['original_idx'], sample_pair['3d']['original_idx']):
        print(f"  ✓ Pairing is CORRECT - indices match!")
    else:
        print(f"  ❌ ERROR: Pairing is WRONG - indices don't match!")
        exit(1)

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

    # Create DualBranchVAE
    print(f"\nCreating DualBranchVAE...")
    model = DualBranchVAE(
        in_channels=args.in_channels,
        latent_channels=args.latent_channels,
        share_encoders=False,
        share_decoders=False
    ).to(device)
    
    # Load stage 1 checkpoint (encoder_3d + decoder_3d) into E3D + D3D
    print(f"\nLoading stage 1 checkpoint: {args.stage1_checkpoint}")
    
    # Handle both file and directory paths
    if os.path.isdir(args.stage1_checkpoint):
        # Try best_model.pt first, then vae.pt
        checkpoint_file = os.path.join(args.stage1_checkpoint, 'best_model.pt')
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(args.stage1_checkpoint, 'vae.pt')
        print(f"  Loading from: {checkpoint_file}")
    else:
        checkpoint_file = args.stage1_checkpoint
    
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Stage 1 now saves with encoder_3d/decoder_3d naming - no remapping needed!
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    
    # Count loaded parameters
    loaded_params = sum(1 for k in checkpoint.keys() if k.startswith('encoder_3d.') or k.startswith('decoder_3d.'))
    print(f"  Loaded {loaded_params} parameters into E3D+D3D (direct load, no remapping)")
    
    if missing:
        # E2D and D2D parameters should be missing (randomly initialized)
        e2d_missing = sum(1 for k in missing if 'encoder_2d' in k or 'decoder_2d' in k)
        print(f"  E2D+D2D parameters (randomly initialized): {e2d_missing}")
    
    # Freeze E3D encoder and D3D decoder - fixed latent space from Stage 1
    # E2D learns to encode into E3D's fixed latent distribution
    for param in model.encoder_3d.parameters():
        param.requires_grad = False
    for param in model.decoder_3d.parameters():
        param.requires_grad = False
    print("  ✓ Frozen E3D encoder and D3D decoder")
    
    # Store checksums of E3D/D3D weights to verify they don't change
    e3d_checksum = sum(p.sum().item() for p in model.encoder_3d.parameters())
    d3d_checksum = sum(p.sum().item() for p in model.decoder_3d.parameters())
    print(f"  E3D weight checksum: {e3d_checksum:.6f}")
    print(f"  D3D weight checksum: {d3d_checksum:.6f}")
    
    # Enable multi-GPU training if available
    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    
    model_module = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    print(f"Model loaded on {device}")
    if device == "cuda":
        print(f"GPU Memory after model load: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} (E2D + D2D only)")
    print(f"  Frozen: {frozen_params:,} (E3D + D3D)")

    # Optimizer - only trainable parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # Select loss function
    loss_functions = {
        'mae_per_channel': mae_loss_per_channel,
        'normalized_mae_per_channel': normalized_mae_loss_per_channel,
        'normalized_mse_per_channel': normalized_mse_per_channel
    }
    reconstruction_loss_fn = loss_functions[args.loss_function]
    print(f"\nUsing reconstruction loss: {args.loss_function}")

    log_dict = {
        'loss': {
            'recons_2d_train': [], 'recons_2d_val': [],
            'kl_2d_train': [], 'kl_2d_val': [],
            'align_train': [], 'align_val': [],
            'cross_2d3d_train': [], 'cross_2d3d_val': [],
            'kl_coeff': []
        },
        'in_channels': args.in_channels,
        'latent_channels': args.latent_channels,
        'per_component_norm': use_per_component,
        'norm_mode': norm_mode,  # 'max' or 'mean'
        'norm_factors': norm_factors.tolist(),
        'stage1_checkpoint': args.stage1_checkpoint,
        'lambda_align': args.lambda_align,
        'lambda_cross': args.lambda_cross,
        'loss_function': args.loss_function,
    }
    
    best_val_loss = float('inf')  # Track best validation loss for saving best model
    
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")
    print(f"Loss configuration:")
    print(f"  Reconstruction (2D): E2D → D2D on 2D samples")
    print(f"  Alignment: ||z_2d - z_3d|| on PROPERLY PAIRED samples")
    print(f"  Cross-recon: E2D → D3D (frozen) for 2D→3D prediction")
    print(f"  KL coefficient: {max_kl_coeff} (with annealing)")
    print(f"  Alignment weight: {args.lambda_align}")
    print(f"  Cross-recon weight: {args.lambda_cross}")
    print()
    
    for epoch in range(args.num_epochs):
        
        start_time = time.time()
        
        # KL annealing: gradually increase KL weight
        min_kl_coeff = 1e-5
        if epoch < kl_warmup_epochs:
            kl_coeff = min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (epoch / kl_warmup_epochs)
        else:
            kl_coeff = max_kl_coeff
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs} - KL coefficient: {kl_coeff:.6f}")

        """Training set"""
        running_recons_2d = 0
        running_kl_2d = 0
        running_align = 0
        running_cross = 0
        optimizer.zero_grad()
        
        i = -1
        # Iterate over paired samples
        for i, paired_batch in enumerate(train_loader):
            
            if i % 10 == 0:
                print(f'Training, batch {i}/{len(train_loader)}')
            
            # Unpack paired data
            data_2d = paired_batch['2d']
            data_3d = paired_batch['3d']
            
            # Print pairing verification on first batch
            if i == 0 and epoch == 0:
                print(f"\n  Pairing verification (batch 0):")
                print(f"    2D original_idx: {data_2d['original_idx'][:min(3, args.batch_size)]}")
                print(f"    3D original_idx: {data_3d['original_idx'][:min(3, args.batch_size)]}")
                if torch.equal(data_2d['original_idx'], data_3d['original_idx']):
                    print(f"    ✓ Pairing CORRECT\n")
                else:
                    print(f"    ❌ ERROR: Pairing WRONG!\n")
                    exit(1)

            # 2D data
            mask_2d = data_2d['microstructure'].to(device)
            velocity_2d = data_2d['velocity'].to(device)  # 2D flow (w=0)
            
            # 3D data (from SAME microstructure)
            mask_3d = data_3d['microstructure'].to(device)
            velocity_3d = data_3d['velocity'].to(device)  # 3D flow
            
            # Normalize
            nf = norm_factors.to(device).view(1, 3, 1, 1, 1)
            inputs_2d = velocity_2d / nf
            inputs_3d = velocity_3d / nf
            targets_2d = inputs_2d.clone()
            targets_3d = inputs_3d.clone()
            
            # === Loss 1: 2D Reconstruction (E2D → D2D) ===
            preds_2d, mean_2d = model_module.forward_2d_deterministic(inputs_2d)
            
            if torch.isnan(mean_2d).any() or torch.isinf(mean_2d).any():
                print(f"WARNING: NaN/Inf in mean_2d at batch {i}")
                continue
            
            preds_2d = preds_2d * mask_2d
            targets_2d = targets_2d * mask_2d
            
            reconstruction_loss_2d = reconstruction_loss_fn(preds_2d, targets_2d, mask=mask_2d)
            kl_loss_2d = torch.tensor(0.0, device=device)
            
            # === Loss 2: Latent Alignment (||z_2d - z_3d||) ===
            # Encode 3D samples with frozen E3D
            with torch.no_grad():
                _, (mean_3d, _) = model_module.encode_3d_deterministic(inputs_3d)
            
            # Alignment: minimize distance between 2D and 3D latents
            alignment_loss = (
                F.mse_loss(mean_2d, mean_3d) +
                0.1 * (1 - F.cosine_similarity(mean_2d, mean_3d, dim=1).mean())
            )

            
            # === Loss 3: Cross-Reconstruction (E2D → D3D) ===
            # Use E2D latents with frozen D3D to predict 3D velocity
            # D3D is frozen but gradients flow back to E2D through mean_2d
            preds_3d_from_2d = model_module.decoder_3d(mean_2d)
            
            # CRITICAL FIX: Use 3D mask from the SAME microstructure, not 2D mask
            preds_3d_from_2d = preds_3d_from_2d * mask_3d
            targets_3d_masked = targets_3d * mask_3d
            
            # Cross-loss trains E2D to encode info useful for 3D reconstruction
            cross_loss = reconstruction_loss_fn(preds_3d_from_2d, targets_3d_masked, mask=mask_3d)
            
            # Total loss - cross_loss trains E2D even though D3D is frozen
            loss = (reconstruction_loss_2d +
                    args.lambda_align * alignment_loss +
                    args.lambda_cross * cross_loss) / gradient_accumulation_steps
            
            if i % 10 == 0:
                print(f'  Recons2D/KL2D/Align/Cross: {reconstruction_loss_2d.item():.6f}/{kl_loss_2d.item():.6f}/{alignment_loss.item():.6f}/{cross_loss.item():.6f}')
                if kl_loss_2d.item() > 1000:
                    print(f'  ERROR: KL loss exploded to {kl_loss_2d.item():.2f}!')
                    sys.exit(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_recons_2d += reconstruction_loss_2d.item()
            running_kl_2d += kl_loss_2d.item()
            running_align += alignment_loss.item()
            running_cross += cross_loss.item()
            
            # Aggressive memory cleanup
            del inputs_2d, inputs_3d, targets_2d, targets_3d
            del preds_2d, mean_2d, mean_3d
            del preds_3d_from_2d, targets_3d_masked
            del reconstruction_loss_2d, kl_loss_2d, alignment_loss, cross_loss, loss
            del mask_2d, mask_3d, velocity_2d, velocity_3d
            
            if device == "cuda":
                torch.cuda.empty_cache()

        # Apply any remaining accumulated gradients
        if (i + 1) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if i == -1:
            print("ERROR: No training batches found!")
            continue
            
        avg_recons_2d_train = running_recons_2d / (i+1)
        avg_kl_2d_train = running_kl_2d / (i+1)
        avg_align_train = running_align / (i+1)
        avg_cross_train = running_cross / (i+1)

        log_dict['loss']['recons_2d_train'].append(avg_recons_2d_train)
        log_dict['loss']['kl_2d_train'].append(avg_kl_2d_train)
        log_dict['loss']['align_train'].append(avg_align_train)
        log_dict['loss']['cross_2d3d_train'].append(avg_cross_train)
        log_dict['loss']['kl_coeff'].append(kl_coeff)

        """Validation set"""
        
        print(f"\nStarting validation... {len(val_loader)} batches")
        sys.stdout.flush()
        
        with torch.no_grad():
            running_recons_2d = 0
            running_kl_2d = 0
            running_align = 0
            running_cross = 0
            j = -1
            
            for j, paired_batch in enumerate(val_loader):
                
                # Unpack paired data
                data_2d = paired_batch['2d']
                data_3d = paired_batch['3d']
                
                print(f'Val batch {j}/{len(val_loader)}')
                sys.stdout.flush()
                
                mask_2d = data_2d['microstructure'].to(device)
                mask_3d = data_3d['microstructure'].to(device)
                velocity_2d = data_2d['velocity'].to(device)
                velocity_3d = data_3d['velocity'].to(device)
                
                nf = norm_factors.to(device).view(1, 3, 1, 1, 1)
                inputs_2d = velocity_2d / nf
                inputs_3d = velocity_3d / nf
                targets_2d = inputs_2d.clone()
                targets_3d = inputs_3d.clone()
                
                # 2D Reconstruction
                preds_2d, mean_2d = model_module.forward_2d_deterministic(inputs_2d)
                preds_2d = preds_2d * mask_2d
                targets_2d = targets_2d * mask_2d
                
                reconstruction_loss_2d = reconstruction_loss_fn(preds_2d, targets_2d, mask=mask_2d)
                kl_loss_2d = torch.tensor(0.0, device=device)
                
                # Alignment
                mean_3d, _ = model_module.encoder_3d(inputs_3d)
                alignment_loss = (
                    F.mse_loss(mean_2d, mean_3d) +
                    0.1 * (1 - F.cosine_similarity(mean_2d, mean_3d, dim=1).mean())
                )

                
                # Cross-reconstruction
                preds_3d_from_2d = model_module.decoder_3d(mean_2d)
                preds_3d_from_2d = preds_3d_from_2d * mask_3d  # Use 3D mask
                targets_3d_masked = targets_3d * mask_3d
                cross_loss = reconstruction_loss_fn(preds_3d_from_2d, targets_3d_masked, mask=mask_3d)
                
                print(f'Val batch {j}: Recons2D/Align/Cross: {reconstruction_loss_2d.item():.6f}/{alignment_loss.item():.6f}/{cross_loss.item():.6f}')
                sys.stdout.flush()

                running_recons_2d += reconstruction_loss_2d.item()
                running_kl_2d += kl_loss_2d.item()
                running_align += alignment_loss.item()
                running_cross += cross_loss.item()
                
                # Memory cleanup for validation
                del inputs_2d, inputs_3d, targets_2d, targets_3d
                del mask_2d, mask_3d, velocity_2d, velocity_3d
                del preds_2d, mean_2d, mean_3d
                del preds_3d_from_2d, targets_3d_masked
                del reconstruction_loss_2d, kl_loss_2d, alignment_loss, cross_loss
                
                if device == "cuda" and j % 5 == 0:
                    torch.cuda.empty_cache()

        if j == -1:
            print("ERROR: No validation batches found!")
            avg_recons_2d_val = 0.0
            avg_kl_2d_val = 0.0
            avg_align_val = 0.0
            avg_cross_val = 0.0
        else:
            avg_recons_2d_val = running_recons_2d / (j+1)
            avg_kl_2d_val = running_kl_2d / (j+1)
            avg_align_val = running_align / (j+1)
            avg_cross_val = running_cross / (j+1)

        log_dict['loss']['recons_2d_val'].append(avg_recons_2d_val)
        log_dict['loss']['kl_2d_val'].append(avg_kl_2d_val)
        log_dict['loss']['align_val'].append(avg_align_val)
        log_dict['loss']['cross_2d3d_val'].append(avg_cross_val)
        
        print(f"DEBUG: Validation complete.")
        sys.stdout.flush()

        dtime = time.time() - start_time
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}]:")
        print(f"  Train: recons_2d={avg_recons_2d_train:.6f}, kl_2d={avg_kl_2d_train:.6f}, align={avg_align_train:.6f}, cross={avg_cross_train:.6f}")
        print(f"  Val:   recons_2d={avg_recons_2d_val:.6f}, kl_2d={avg_kl_2d_val:.6f}, align={avg_align_val:.6f}, cross={avg_cross_val:.6f}")
        print(f"  kl_coeff={kl_coeff:.6f}, time={dtime:.2f}s")
        sys.stdout.flush()
        
        if device == "cuda":
            print(f"GPU Memory: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        
        print("DEBUG: Saving model...")
        sys.stdout.flush()

        # Verify frozen weights haven't changed
        current_e3d_checksum = sum(p.sum().item() for p in model_module.encoder_3d.parameters())
        current_d3d_checksum = sum(p.sum().item() for p in model_module.decoder_3d.parameters())
        if abs(current_e3d_checksum - e3d_checksum) > 1e-5:
            print(f"  WARNING: E3D weights changed! {e3d_checksum:.6f} → {current_e3d_checksum:.6f}")
        if abs(current_d3d_checksum - d3d_checksum) > 1e-5:
            print(f"  WARNING: D3D weights changed! {d3d_checksum:.6f} → {current_d3d_checksum:.6f}")

        # Save model
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        save_path = osp.join(args.save_dir, 'model.pt')
        torch.save(model_to_save.state_dict(), save_path)
        
        # Track best model based on validation loss (reconstruction + alignment + cross)
        current_val_loss = avg_recons_2d_val + kl_coeff * avg_kl_2d_val + args.lambda_align * avg_align_val + args.lambda_cross * avg_cross_val
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            print(f"  ✓ New best model! Val loss: {current_val_loss:.6f}")
            # Save best model as separate file
            best_model_path = osp.join(args.save_dir, 'best_model.pt')
            torch.save(model_to_save.state_dict(), best_model_path)
            print(f"  Saved best model to: {best_model_path}")
        
        # Save log
        log_path = osp.join(args.save_dir, 'vae_log.json')
        with open(log_path, 'w') as f:
            json.dump(log_dict, f, indent=2)
        
        print(f"DEBUG: Model saved to {args.save_dir}")
        sys.stdout.flush()

    # Final test evaluation
    print("\n" + "="*50)
    print("Running final test set evaluation...")
    print("="*50)
    
    with torch.no_grad():
        running_recons_2d = 0
        running_kl_2d = 0
        running_align = 0
        running_cross = 0
        k = -1
        
        for k, paired_batch in enumerate(test_loader):
            data_2d = paired_batch['2d']
            data_3d = paired_batch['3d']
            
            mask_2d = data_2d['microstructure'].to(device)
            mask_3d = data_3d['microstructure'].to(device)
            velocity_2d = data_2d['velocity'].to(device)
            velocity_3d = data_3d['velocity'].to(device)
            
            nf = norm_factors.to(device).view(1, 3, 1, 1, 1)
            inputs_2d = velocity_2d / nf
            inputs_3d = velocity_3d / nf
            targets_2d = inputs_2d.clone()
            targets_3d = inputs_3d.clone()
            
            # 2D Reconstruction
            preds_2d, mean_2d = model_module.forward_2d_deterministic(inputs_2d)
            preds_2d = preds_2d * mask_2d
            targets_2d = targets_2d * mask_2d
            reconstruction_loss_2d = reconstruction_loss_fn(preds_2d, targets_2d, mask=mask_2d)
            kl_loss_2d = torch.tensor(0.0, device=device)
            
            # Alignment
            mean_3d, _ = model_module.encoder_3d(inputs_3d)
            alignment_loss = (
                F.mse_loss(mean_2d, mean_3d) +
                0.1 * (1 - F.cosine_similarity(mean_2d, mean_3d, dim=1).mean())
            )

            
            # Cross-reconstruction
            preds_3d_from_2d = model_module.decoder_3d(mean_2d)
            preds_3d_from_2d = preds_3d_from_2d * mask_3d  # Use 3D mask
            targets_3d_masked = targets_3d * mask_3d
            cross_loss = reconstruction_loss_fn(preds_3d_from_2d, targets_3d_masked, mask=mask_3d)
            
            print(f'Test batch {k}: Recons2D/Align/Cross: {reconstruction_loss_2d.item():.6f}/{alignment_loss.item():.6f}/{cross_loss.item():.6f}')
            
            running_recons_2d += reconstruction_loss_2d.item()
            running_kl_2d += kl_loss_2d.item()
            running_align += alignment_loss.item()
            running_cross += cross_loss.item()
        
        if k == -1:
            print("WARNING: No test batches found!")
        else:
            avg_recons_2d_test = running_recons_2d / (k+1)
            avg_kl_2d_test = running_kl_2d / (k+1)
            avg_align_test = running_align / (k+1)
            avg_cross_test = running_cross / (k+1)
            
            log_dict['loss']['recons_2d_test'] = avg_recons_2d_test
            log_dict['loss']['kl_2d_test'] = avg_kl_2d_test
            log_dict['loss']['align_test'] = avg_align_test
            log_dict['loss']['cross_2d3d_test'] = avg_cross_test
            
            print(f"\nFinal Test Results:")
            print(f"  recons_2d={avg_recons_2d_test:.6f}, kl_2d={avg_kl_2d_test:.6f}")
            print(f"  align={avg_align_test:.6f}, cross_2d3d={avg_cross_test:.6f}")
            print("="*50)
    
    # Save final model with test results
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    save_path = osp.join(args.save_dir, 'model.pt')
    torch.save(model_to_save.state_dict(), save_path)
    
    log_path = osp.join(args.save_dir, 'vae_log.json')
    with open(log_path, 'w') as f:
        json.dump(log_dict, f, indent=2)
    
    print(f"\n{'='*60}")
    print("STAGE 2 COMPLETE")
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
        sys.exit(1)
