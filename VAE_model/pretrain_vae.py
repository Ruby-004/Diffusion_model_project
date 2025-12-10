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

print(f"Using device: {args.device}")
if args.device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Create save directory if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)


# root_dir = '/home/jimmy/Documents/data/unet/rve_5k_xy'
# save_dir = 'trained'

# IN_CHANNELS = 2
# MID_CHANNELS = 4
# device = 'cuda'
# num_epochs = 100
# batch_size = 10
# learning_rate = 1e-4
seed = None
# latent_shape = (1, MID_CHANNELS, 64, 64)
scale_factor = 0.004389363341033459


def main():
    coeff = 1e-3

    # Clear CUDA cache before training
    if args.device == "cuda":
        torch.cuda.empty_cache()

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

    # params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(
        vae.parameters(),
        lr=args.learning_rate
    )

    log_dict = {
        'loss': {'recons_train': [], 'recons_val': [], 'kl_train':[], 'kl_val':[]}
    }
    for epoch in range(args.num_epochs):
        
        start_time = time.time()

        """Training set"""
        running_recons = 0
        running_kl = 0
        for i, data in enumerate(train_loader):
            
            print(f'Training, batch {i}')

            mask = data['microstructure'].to(args.device)
            inputs = data['velocity_input'].to(args.device)
            targets = data['velocity'].to(args.device)
            
            # Scale both inputs and targets
            inputs = inputs / scale_factor
            targets = targets / scale_factor

            # noise = torch.randn(
            #     latent_shape, generator=generator, device=device
            # )
            
            preds, (mean, logvar) = vae(inputs)
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
            loss = reconstruction_loss + coeff * kl_loss
            print(f'Recons/KL loss: {reconstruction_loss.item()}/{kl_loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients more aggressively for large model stability
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=0.5)
            
            optimizer.step()

            running_recons += reconstruction_loss.item()
            running_kl += kl_loss.item()
            
            # Clear CUDA cache to prevent memory buildup
            if args.device == "cuda":
                torch.cuda.empty_cache()

            # # save sample
            # sample = {
            #     'target': targets,
            #     'prediction': preds.detach(),
            #     'recons': reconstruction_loss.item(),
            #     'kl': kl_loss.item()
            # }
            # torch.save(sample, osp.join(save_dir,'sample.pt'))

        avg_recons_train = running_recons / (i+1)
        avg_kl_train = running_kl / (i+1)

        log_dict['loss']['recons_train'].append(avg_recons_train)
        log_dict['loss']['kl_train'].append(avg_kl_train)

        """Validation set"""

        with torch.no_grad():

            running_recons = 0
            running_kl = 0
            for j, data in enumerate(val_loader):
                
                mask = data['microstructure'].to(args.device)
                inputs = data['velocity_input'].to(args.device)
                targets = data['velocity'].to(args.device)
                
                # Scale both inputs and targets
                inputs = inputs / scale_factor
                targets = targets / scale_factor

                # noise = torch.randn(
                #     latent_shape, generator=generator, device=device
                # )
                preds, (mean, logvar) = vae(inputs)
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
                loss = reconstruction_loss + coeff * kl_loss
                print(f'Recons/KL loss: {reconstruction_loss.item()}/{kl_loss.item()}')

                running_recons += reconstruction_loss.item()
                running_kl += kl_loss.item()

            avg_recons_val = running_recons / (j+1)
            avg_kl_val = running_kl / (j+1)

            log_dict['loss']['recons_val'].append(avg_recons_val)
            log_dict['loss']['kl_val'].append(avg_kl_val)

        dtime = time.time() - start_time
        print(f"Epoch [{epoch}/{args.num_epochs}]: recons/kl_train=[{avg_recons_train}/{avg_kl_train}] | recons/kl_val=[{avg_recons_val}/{avg_kl_val}] | time={dtime:.2f} s")

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