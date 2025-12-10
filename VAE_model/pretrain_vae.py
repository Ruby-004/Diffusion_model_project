import time
import os

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
scale_factor = 0.004389363341033459  # May need recalculation for 3D data
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

    train_loader, val_loader = get_loader(
        root_dir=args.dataset_dir, batch_size=args.batch_size
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
            velocity_2d = data['velocity_input'].to(args.device)  # U_2d.pt - 2D flow
            velocity_3d = data['velocity'].to(args.device)        # U.pt - 3D flow
            
            # Concatenate both velocity fields: [batch, 6 channels (3 from U_2d + 3 from U), depth, height, width]
            velocity_combined = torch.cat([velocity_2d, velocity_3d], dim=1)  # [batch, 6, depth, height, width]
            
            # Keep 3D structure for Conv3d layers: [batch, channels, depth, height, width]
            inputs = velocity_combined
            targets = inputs.clone()  # Autoencoder: reconstruct the same combined input
            
            # Scale both inputs and targets
            inputs = inputs / scale_factor
            targets = targets / scale_factor
            
            preds, (mean, logvar) = vae(inputs)
            
            # Clamp logvar to prevent explosion
            logvar = torch.clamp(logvar, min=-10, max=10)
            
            preds = preds * mask
            targets = targets * mask

            reconstruction_loss = normalized_mae_loss(
                output=preds, target=targets
            )
            kl_loss = kl_divergence(mu=mean, logvar=logvar)
            loss = (reconstruction_loss + kl_coeff * kl_loss) / gradient_accumulation_steps
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
                velocity_2d = data['velocity_input'].to(args.device)  # U_2d.pt - 2D flow
                velocity_3d = data['velocity'].to(args.device)        # U.pt - 3D flow
                
                # Concatenate both velocity fields: [batch, 6 channels (3 from U_2d + 3 from U), depth, height, width]
                velocity_combined = torch.cat([velocity_2d, velocity_3d], dim=1)  # [batch, 6, depth, height, width]
                
                # Keep 3D structure for Conv3d layers: [batch, channels, depth, height, width]
                inputs = velocity_combined
                targets = inputs.clone()  # Autoencoder: reconstruct the same combined input
                
                # Scale both inputs and targets
                inputs = inputs / scale_factor
                targets = targets / scale_factor

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

                reconstruction_loss = normalized_mae_loss(
                    output=preds, target=targets
                )
                kl_loss = kl_divergence(mu=mean, logvar=logvar)
                loss = reconstruction_loss + kl_coeff * kl_loss
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

        # # save sample
        # sample = {
        #     'target': targets,
        #     'prediction': preds.detach(),
        #     'recons': reconstruction_loss.item(),
        #     'kl': kl_loss.item()
        # }
        # torch.save(sample, osp.join(save_dir, 'sample_final.pt'))



if __name__=='__main__':

    main()