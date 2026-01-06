from typing import Callable, Literal, Union
import json
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from src.predictor import LatentDiffusionPredictor
from utils.zenodo import download_data, unzip_data, is_url


def get_norm_params(
    file: str,
    option: Literal['latent-diffusion']
):
    """
    Retrieve normalization parameters for dataset.

    Args:
        file: path to file with parameters.
    """

    stats = json.load(open(file))

    if option == 'latent-diffusion':
        # Try to get max velocity from either 'U' or 'velocity' key
        if 'U' in stats:
            max_velocity = stats['U']['max']
        elif 'velocity' in stats:
            max_velocity = stats['velocity']['max']
        else:
            # Fallback: check for U_2d or U_3d
            if 'U_2d' in stats and 'U_3d' in stats:
                max_velocity = max(stats['U_2d']['max'], stats['U_3d']['max'])
            elif 'U_2d' in stats:
                max_velocity = stats['U_2d']['max']
            elif 'U_3d' in stats:
                max_velocity = stats['U_3d']['max']
            else:
                # No velocity statistics found - use default normalization
                print(f"WARNING: No velocity statistics found in {file}.")
                print(f"Available keys: {list(stats.keys())}")
                print("Using default max_velocity=1.0 for normalization.")
                print("This means velocity data should already be normalized to [0,1] range.")
                max_velocity = 1.0

        out = {
            'input': None,
            'output': (max_velocity, max_velocity, max_velocity)  # 3 channels: vx, vy, vz
        }
    else:
        raise ValueError(f'Unknown option: {option}')

    return out


def set_model(
    type: Literal['latent-diffusion'],
    kwargs: dict,
    norm_file: str
):

    if type=='latent-diffusion':
        predictor = LatentDiffusionPredictor(**kwargs)
    else:
        raise ValueError(f'Unknown model type: {type}')

    norm_params = get_norm_params(
        file=norm_file,
        option=type
    )
    predictor.set_normalizer(norm_params)

    return predictor


def get_model(
    type: Literal['latent-diffusion'],
    kwargs: dict,
    model_path: str,
    device: str = None
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if type=='latent-diffusion':
        predictor = LatentDiffusionPredictor(**kwargs)
    else:
        raise ValueError(f'Unknown model type: {type}')

    predictor.to(device)

    predictor.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device(device)
        )
    )
    return predictor


def select_input_output(
    data: dict[str, torch.Tensor],
    option: Literal['latent-diffusion'],
    device: str
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Select appropriate input & output for ML models.

    Args:
        data: dictionary returned by data loader.
        option: case into consideration.
        device: device on which to put data.
    """
    
    if option=='latent-diffusion':
        # For latent diffusion: input is 2D velocity (U_2d), target is 3D velocity (U)
        # Microstructure used as mask - shape: (batch, num_slices, 1, H, W)
        imgs = data['microstructure'].to(device)  # Shape: (batch, num_slices, 1, H, W)
        velocity_2d = data['velocity_input'].to(device)  # Shape: (batch, num_slices, 3, H, W)
        
        input = (imgs, velocity_2d)
        targets = data['velocity'].to(device)  # Shape: (batch, num_slices, 3, H, W)
    else:
        raise ValueError(f'Unknown option: {option}')

    return input, targets


def run_epoch(
    loaders: tuple[DataLoader, DataLoader],
    predictor: Union[LatentDiffusionPredictor],
    optimizer: optim.Optimizer,
    criterion: Callable[..., torch.Tensor],
    device: str = 'cuda'
):
    """
    Optimize model for 1 epoch over training set, then evaluate over validation set.
    
    Args:
        loaders: data loaders for training and validation sets.
        predictor: ML model.
        optimizer: optimizer for tuning ML model parameters.
        criterion: cost function.
        device: device on which to train ML model.
    """

    train_loader, val_loader = loaders
    num_train_batch = len(train_loader)
    num_val_batch = len(val_loader)

    if isinstance(predictor, LatentDiffusionPredictor):
        option = 'latent-diffusion'
    else:
        raise ValueError(f'Unknown predictor type: {type(predictor)}')

    """1. Training Set"""
    predictor.train()

    running_loss = 0
    for i, data in enumerate(train_loader):
        print(f"Training set: batch [{i+1}/{num_train_batch}]")

        input, targets = select_input_output(data, option, device)

        if option == 'latent-diffusion':
            # For latent diffusion: encode target to latent space and add noise
            # Input: microstructure + 2D velocity
            img = input[0]
            velocity_2d = input[1]
            
            # Encode 3D velocity target to latent space using 2D velocity for VAE
            target_latents = predictor.encode_target(targets, velocity_2d)
            
            # Generate random noise with same shape as latent
            noise = torch.randn_like(target_latents)
            
            # Predict denoised latent (one-step: predict clean 3D from noise + 2D velocity)
            preds = predictor(img, velocity_2d, noise)
            
            # Loss in latent space (no normalization needed, already in latent)
            loss = criterion(output=preds, target=target_latents)
        else:
            raise ValueError(f"Unknown option: {option}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / (i+1)


    """2. Validation Set"""
    predictor.eval()

    with torch.no_grad():
        val_loss = 0
        for j, data in enumerate(val_loader):
            print(f"Validation set: batch [{j+1}/{num_val_batch}]")


            input, targets = select_input_output(data, option, device)

            if option == 'latent-diffusion':
                # For latent diffusion validation
                img = input[0]
                velocity_2d = input[1]
                
                # Encode target to latent space
                target_latents = predictor.encode_target(targets, velocity_2d)
                
                # Generate random noise
                noise = torch.randn_like(target_latents)
                
                # Predict denoised latent
                preds = predictor(img, velocity_2d, noise)
                
                # Loss in latent space
                loss = criterion(output=preds, target=target_latents)
            else:
                raise ValueError(f"Unknown option: {option}")
            
            val_loss += loss.item()

        avg_val_loss = val_loss / (j+1)

    return (avg_train_loss, avg_val_loss)


def retrieve_model_path(directory_or_url: str, filename: str = 'model.pt') -> str:
    """
    Retrieve path to pre-trained model.
    
    Args:
        directory_or_url: either local directory or URL of the pre-trained model.
        filename: name of the model file.
    """

    if is_url(directory_or_url):
        # Use pre-trained model in repo

        _folder = 'pretrained'
        if not osp.exists(_folder): os.mkdir(_folder)

        # download pre-trained weights
        zip_path = download_data(url=directory_or_url, save_dir=_folder)

        # unzip data
        folder_path = unzip_data(zip_path=zip_path, save_dir=_folder)

        model_path = osp.join(folder_path, filename)

    else:
        # Use trained model in local directory
        model_path = osp.join(directory_or_url, filename)

    return model_path
