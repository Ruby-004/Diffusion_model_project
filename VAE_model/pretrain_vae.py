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
from utils.metrics import normalized_mae_loss, kl_divergence

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
# Load normalization statistics
stats_file = osp.join(args.dataset_dir, 'statistics.json')
if not os.path.exists(stats_file):
    print(f"ERROR: statistics.json not found at {stats_file}")
    print("Please ensure the dataset has been processed and statistics computed.")
    exit(1)

with open(stats_file, 'r') as f:
    statistics = json.load(f)

# Compute max value across both velocity fields for [0,1] normalization
max_U_2d = statistics['U_2d']['max']
max_U_3d = statistics['U']['max']
max_velocity = max(max_U_2d, max_U_3d)

print(f"Loaded statistics: max_U_2d={max_U_2d:.6f}, max_U_3d={max_U_3d:.6f}")
print(f"Using max_velocity={max_velocity:.6f} for [0,1] normalization")

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
        root_dir=args.dataset_dir, batch_size=args.batch_size, use_vae_dataset=True
    )

    """Model"""

    vae = VariationalAutoencoder(
        in_channels=args.in_channels,
        latent_channels=args.latent_channels
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

    print(f"Model loaded on {args.device}")
    if args.device == "cuda":
        print(f"GPU Memory after model load: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    # params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(
        vae.parameters(),
        lr=args.learning_rate
    )

    log_dict = {
        'loss': {'recons_train': [], 'recons_val': [], 'kl_train':[], 'kl_val':[], 'kl_coeff': []}
    }
    
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
            
            # Keep 3D structure for Conv3d layers: [batch, 3, depth, height, width]
            inputs = velocity
            targets = inputs.clone()  # Autoencoder: reconstruct the same input
            
            # Normalize both inputs and targets to [0,1]
            inputs = inputs / max_velocity
            targets = targets / max_velocity
            
            preds, (mean, logvar) = vae(inputs)
            
            # Clamp logvar to prevent explosion
            logvar = torch.clamp(logvar, min=-10, max=10)
            
            preds = preds * mask
            targets = targets * mask

            # Expand mask to match channel dimension [B,1,D,H,W] -> [B,3,D,H,W]
            mask_expanded = mask.expand_as(preds)
            # Use simple MAE since inputs are normalized to [0,1]
            fluid_preds = preds[mask_expanded > 0.5]
            fluid_targets = targets[mask_expanded > 0.5]
            reconstruction_loss = torch.mean(torch.abs(fluid_preds - fluid_targets))
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
                
                # Keep 3D structure for Conv3d layers: [batch, 3, depth, height, width]
                inputs = velocity
                targets = inputs.clone()  # Autoencoder: reconstruct the same input
                
                # Normalize both inputs and targets to [0,1]
                inputs = inputs / max_velocity
                targets = targets / max_velocity

                # noise = torch.randn(
                #     latent_shape, generator=generator, device=device
                # )
                preds, (mean, logvar) = vae(inputs)
                
                # Clamp logvar to prevent explosion
                logvar = torch.clamp(logvar, min=-10, max=10)
                
                preds = preds * mask
                targets = targets * mask
                
                # mean, logvar = encoder(targets)
                # # sample
                # latents = encoder.sample(mu=mean, logvar=logvar)
                # preds = decoder(latents) * mask

                # Expand mask to match channel dimension [B,1,D,H,W] -> [B,3,D,H,W]
                mask_expanded = mask.expand_as(preds)
                # Use simple MAE since inputs are normalized to [0,1]
                fluid_preds = preds[mask_expanded > 0.5]
                fluid_targets = targets[mask_expanded > 0.5]
                reconstruction_loss = torch.mean(torch.abs(fluid_preds - fluid_targets))
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
        vae.save_model(args.save_dir, log=log_dict)

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
            
            inputs = velocity / max_velocity
            targets = inputs.clone()
            
            preds, (mean, logvar) = vae(inputs)
            logvar = torch.clamp(logvar, min=-10, max=10)
            
            preds = preds * mask
            targets = targets * mask
            
            mask_expanded = mask.expand_as(preds)
            fluid_preds = preds[mask_expanded > 0.5]
            fluid_targets = targets[mask_expanded > 0.5]
            reconstruction_loss = torch.mean(torch.abs(fluid_preds - fluid_targets))
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
    vae.save_model(args.save_dir, log=log_dict)
    print(f"\nTraining complete! Model saved to {args.save_dir}")



if __name__=='__main__':

    main()