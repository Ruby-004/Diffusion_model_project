import time
import os
import json
import os.path as osp

import torch
import torch.optim as optim

# from src.vae.encoder import Encoder
# from src.vae.decoder import Decoder
from src.vae.autoencoder import VariationalAutoencoder
from utils.dataset import get_loader
from utils.metrics import normalized_mae_loss, kl_divergence, mae_loss_per_channel

from config.vae import parser

args = parser.parse_args()
if args.device is None:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

# Force CUDA for this 3D training
if args.device == "cuda" and not torch.cuda.is_available():
    print("ERROR: CUDA requested but not available!")
    print("Please ensure CUDA is properly installed.")
    exit(1)

print(f"Using device: {args.device}")
if args.device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

# Verify dataset exists
import os
if not os.path.exists(args.dataset_dir):
    print(f"ERROR: Dataset directory not found: {args.dataset_dir}")
    print("Please ensure the dataset is downloaded to the correct location.")
    exit(1)

# Create save directory if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)


# root_dir = '/home/jimmy/Documents/data/unet/rve_5k_xy'
# save_dir = 'trained'

# IN_CHANNELS = 6  # 6 channels: 3 from U_2d + 3 from U
# MID_CHANNELS = 4
# device = 'cuda'
# num_epochs = 100
# batch_size = 1  # Reduced to 1 for 3D Conv memory management
# learning_rate = 1e-5
seed = None
# latent_shape = (1, MID_CHANNELS, 64, 64)

gradient_accumulation_steps = 10  # Accumulate gradients to simulate larger batch


def main():
    # KL annealing parameters
    kl_warmup_epochs = 10  # Gradually increase KL weight over first 10 epochs
    max_kl_coeff = 1e-3  # Maximum KL coefficient
    
    # Clear CUDA cache before training
    if args.device == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

    # generator = torch.Generator(device=args.device)
    # if seed is None:
    #     generator.seed()
    # else:
    #     generator.manual_seed(seed)

    """Load data"""

    train_loader, val_loader, test_loader = get_loader(
        root_dir=args.dataset_dir, batch_size=args.batch_size, use_vae_dataset=True, augment=args.augment
    )
    
    # Verify augmentation status
    print(f"Data augmentation requested: {args.augment}")
    if hasattr(train_loader.dataset, 'dataset'):
        print(f"Train dataset augmentation: {train_loader.dataset.dataset.augment}")
    if hasattr(val_loader.dataset, 'dataset'):
        print(f"Val dataset augmentation: {val_loader.dataset.dataset.augment}")

    # Load normalization statistics
    stats_file = osp.join(args.dataset_dir, 'statistics.json')
    if not os.path.exists(stats_file):
        print(f"ERROR: statistics.json not found at {stats_file}")
        print("Please ensure the dataset has been processed and statistics computed.")
        exit(1)

    with open(stats_file, 'r') as f:
        statistics = json.load(f)

    # Per-component normalization for better w-component learning
    # The w (vz) component is typically much sparser than u/v, requiring separate normalization
    use_per_component = args.per_component_norm
    
    if use_per_component and 'U_per_component' in statistics:
        pc = statistics['U_per_component']
        pc_2d = statistics.get('U_2d_per_component', {})
        
        # Use max of U and U_2d for each component
        max_u = max(pc['max_u'], pc_2d.get('max_u', 0))
        max_v = max(pc['max_v'], pc_2d.get('max_v', 0))
        max_w = max(pc['max_w'], pc_2d.get('max_w', 0))
        
        # Create per-component normalization tensor [3] for (u, v, w)
        norm_factors = torch.tensor([max_u, max_v, max_w], dtype=torch.float32)
        
        print(f"\n=== Per-Component Normalization ===")
        print(f"  max_u (vx): {max_u:.6f}")
        print(f"  max_v (vy): {max_v:.6f}")
        print(f"  max_w (vz): {max_w:.6f}")
        print(f"  Ratio max_u/max_w: {max_u/max_w:.2f}x")
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
        print(f"  The VAE will learn to distinguish 2D flow (U_2d, w=0) from 3D flow (U, w≠0)")
        print(f"  Condition signal: is_3d = True for U samples, False for U_2d samples")
        print(f"==============================\n")

    """Model"""

    vae = VariationalAutoencoder(
        in_channels=args.in_channels,
        latent_channels=args.latent_channels,
        conditional=use_conditional
    )
    # encoder = Encoder(
    #     in_channels=IN_CHANNELS, out_channels=MID_CHANNELS
    # )
    # decoder = Decoder(
    #     in_channels=MID_CHANNELS, out_channels=IN_CHANNELS
    # )
    vae.to(args.device)
    # encoder.to(device)
    # decoder.to(device)
    
    # Enable multi-GPU training if available
    if args.device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        vae = torch.nn.DataParallel(vae)
    
    print(f"Model loaded on {args.device}")
    if args.device == "cuda":
        print(f"GPU Memory after model load: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    # params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(
        vae.parameters(),
        lr=args.learning_rate
    )

    log_dict = {
        'loss': {'recons_train': [], 'recons_val': [], 'kl_train':[], 'kl_val':[], 'kl_coeff': []},
        'in_channels': args.in_channels,
        'latent_channels': args.latent_channels,
        'per_component_norm': use_per_component,
        'norm_factors': norm_factors.tolist(),  # [max_u, max_v, max_w] for decoding
        'conditional': use_conditional,  # Whether VAE uses conditioning
        'vz_weight': args.vz_weight,  # Weight for V_z component in loss
    }
    
    # Channel weights for loss: [vx_weight, vy_weight, vz_weight]
    # Higher weight on V_z helps the model learn the sparse z-component better
    channel_weights = torch.tensor([1.0, 1.0, args.vz_weight], dtype=torch.float32, device=args.device)
    if args.vz_weight != 1.0:
        print(f"\n=== Channel Weighting ===")
        print(f"  V_x weight: 1.0")
        print(f"  V_y weight: 1.0")
        print(f"  V_z weight: {args.vz_weight}")
        print(f"=========================\n")
    
    for epoch in range(args.num_epochs):
        
        start_time = time.time()
        
        # KL annealing: gradually increase KL weight
        if epoch < kl_warmup_epochs:
            kl_coeff = max_kl_coeff * (epoch / kl_warmup_epochs)
        else:
            kl_coeff = max_kl_coeff
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs} - KL coefficient: {kl_coeff:.6f}")

        """Training set"""
        running_recons = 0
        running_kl = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for i, data in enumerate(train_loader):
            
            print(f'Training, batch {i}')

            # Data shape: [batch, channels, depth, height, width]
            mask = data['microstructure'].to(args.device)
            velocity = data['velocity'].to(args.device)  # Either U_2d or U (3 channels)
            is_2d = data['is_2d']  # Boolean flag indicating if this is 2D flow
            
            # Condition for conditional VAE: is_3d = NOT is_2d
            # True = 3D flow (U, w≠0), False = 2D flow (U_2d, w=0)
            is_3d = (~is_2d).to(args.device) if use_conditional else None
            
            # Keep 3D structure for Conv3d layers: [batch, 3, depth, height, width]
            inputs = velocity
            targets = inputs.clone()  # Autoencoder: reconstruct the same input
            
            # Normalize to [0,1] using per-component or global normalization
            # norm_factors shape: [3] -> reshape to [1, 3, 1, 1, 1] for broadcasting
            nf = norm_factors.to(args.device).view(1, 3, 1, 1, 1)
            inputs = inputs / nf
            targets = targets / nf
            
            preds, (mean, logvar) = vae(inputs, condition=is_3d)
            
            # logvar is already clamped inside the model's encode method
            
            preds = preds * mask
            targets = targets * mask

            # Per-channel loss: computes MAE separately for u, v, w then averages
            # This prevents larger u/v components from dominating, ensuring w-component is learned
            # Channel weights give higher priority to V_z which has smaller magnitude
            reconstruction_loss = mae_loss_per_channel(preds, targets, mask=mask, weight_per_channel=channel_weights)
            kl_loss = kl_divergence(mu=mean, logvar=logvar)
            loss = (reconstruction_loss + kl_coeff * kl_loss) / gradient_accumulation_steps
            
            # Print which type of flow is being processed
            num_2d = is_2d.sum().item()
            num_3d = len(is_2d) - num_2d
            print(f'Batch contains {num_2d} 2D and {num_3d} 3D samples')
            print(f'Recons/KL loss: {reconstruction_loss.item():.6f}/{kl_loss.item():.6f}')

            loss.backward()
            
            # Update weights every gradient_accumulation_steps batches
            if (i + 1) % gradient_accumulation_steps == 0:
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_recons += reconstruction_loss.item()
            running_kl += kl_loss.item()
            
            # Clear CUDA cache to prevent memory buildup
            if args.device == "cuda":
                torch.cuda.empty_cache()

        # Apply any remaining accumulated gradients at end of epoch
        if (i + 1) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_recons_train = running_recons / (i+1)
        avg_kl_train = running_kl / (i+1)

        log_dict['loss']['recons_train'].append(avg_recons_train)
        log_dict['loss']['kl_train'].append(avg_kl_train)
        log_dict['loss']['kl_coeff'].append(kl_coeff)

        """Validation set"""

        with torch.no_grad():

            running_recons = 0
            running_kl = 0
            for j, data in enumerate(val_loader):
                
                # Data shape: [batch, channels, depth, height, width]
                mask = data['microstructure'].to(args.device)
                velocity = data['velocity'].to(args.device)  # Either U_2d or U (3 channels)
                is_2d = data['is_2d']  # Boolean flag indicating if this is 2D flow
                
                # Condition for conditional VAE: is_3d = NOT is_2d
                is_3d = (~is_2d).to(args.device) if use_conditional else None
                
                # Keep 3D structure for Conv3d layers: [batch, 3, depth, height, width]
                inputs = velocity
                targets = inputs.clone()  # Autoencoder: reconstruct the same input
                
                # Normalize to [0,1] using per-component or global normalization
                nf = norm_factors.to(args.device).view(1, 3, 1, 1, 1)
                inputs = inputs / nf
                targets = targets / nf

                # noise = torch.randn(
                #     latent_shape, generator=generator, device=device
                # )
                preds, (mean, logvar) = vae(inputs, condition=is_3d)
                
                # logvar is already clamped
                
                preds = preds * mask
                targets = targets * mask
                
                # Per-channel loss: computes MAE separately for u, v, w then averages
                reconstruction_loss = mae_loss_per_channel(preds, targets, mask=mask, weight_per_channel=channel_weights)
                kl_loss = kl_divergence(mu=mean, logvar=logvar)
                loss = reconstruction_loss + kl_coeff * kl_loss
                
                # Print which type of flow is being processed
                num_2d = is_2d.sum().item()
                num_3d = len(is_2d) - num_2d
                print(f'Val batch {j}: {num_2d} 2D and {num_3d} 3D samples')
                print(f'Recons/KL loss: {reconstruction_loss.item():.6f}/{kl_loss.item():.6f}')

                running_recons += reconstruction_loss.item()
                running_kl += kl_loss.item()

            avg_recons_val = running_recons / (j+1)
            avg_kl_val = running_kl / (j+1)

            log_dict['loss']['recons_val'].append(avg_recons_val)
            log_dict['loss']['kl_val'].append(avg_kl_val)

        dtime = time.time() - start_time
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}]: recons/kl_train=[{avg_recons_train:.6f}/{avg_kl_train:.6f}] | recons/kl_val=[{avg_recons_val:.6f}/{avg_kl_val:.6f}] | kl_coeff={kl_coeff:.6f} | time={dtime:.2f} s")
        
        if args.device == "cuda":
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

        # log_path = osp.join(save_dir, 'log.json')
        # with open(log_path, 'w') as f:
        #     json.dump(log_dict, f, indent=4)

        # save model
        # Unwrap DataParallel before saving
        model_to_save = vae.module if isinstance(vae, torch.nn.DataParallel) else vae
        model_to_save.save_model(args.save_dir, log=log_dict)

    # Final test evaluation after all epochs
    print("\n" + "="*50)
    print("Running final test set evaluation...")
    print("="*50)
    
    with torch.no_grad():
        running_recons = 0
        running_kl = 0
        for k, data in enumerate(test_loader):
            mask = data['microstructure'].to(args.device)
            velocity = data['velocity'].to(args.device)
            is_2d = data['is_2d']
            
            # Condition for conditional VAE: is_3d = NOT is_2d
            is_3d = (~is_2d).to(args.device) if use_conditional else None
            
            nf = norm_factors.to(args.device).view(1, 3, 1, 1, 1)
            inputs = velocity / nf
            targets = inputs.clone()
            
            preds, (mean, logvar) = vae(inputs, condition=is_3d)
            # logvar is clamped in model
            
            preds = preds * mask
            targets = targets * mask
            
            # Per-channel loss: computes MAE separately for u, v, w then averages
            reconstruction_loss = mae_loss_per_channel(preds, targets, mask=mask, weight_per_channel=channel_weights)
            kl_loss = kl_divergence(mu=mean, logvar=logvar)
            
            num_2d = is_2d.sum().item()
            num_3d = len(is_2d) - num_2d
            print(f'Test batch {k}: {num_2d} 2D and {num_3d} 3D samples - Recons/KL: {reconstruction_loss.item():.6f}/{kl_loss.item():.6f}')
            
            running_recons += reconstruction_loss.item()
            running_kl += kl_loss.item()
        
        avg_recons_test = running_recons / (k+1)
        avg_kl_test = running_kl / (k+1)
        
        log_dict['loss']['recons_test'] = avg_recons_test
        log_dict['loss']['kl_test'] = avg_kl_test
        
        print(f"\nFinal Test Results: recons={avg_recons_test:.6f} | kl={avg_kl_test:.6f}")
        print("="*50)
    
    # Save final model with test results
    model_to_save = vae.module if isinstance(vae, torch.nn.DataParallel) else vae
    model_to_save.save_model(args.save_dir, log=log_dict)
    print(f"\nTraining complete! Model saved to {args.save_dir}")



if __name__=='__main__':

    main()