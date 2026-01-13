from typing import Callable, Literal, Union, Dict
import json
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from src.predictor import LatentDiffusionPredictor
from src.unet.metrics import divergence_loss
from src.physics import PhysicsLoss, compute_physics_metrics, reconstruct_velocity_from_noise_pred, component_weighted_velocity_loss, compute_per_component_metrics
from utils.zenodo import download_data, unzip_data, is_url


def get_norm_params(
    file: str,
    option: Literal['latent-diffusion']
):
    """
    Retrieve normalization parameters for dataset.
    
    Uses per-component max values (max_u, max_v, max_w) when available
    for proper normalization across velocity components.

    Args:
        file: path to file with parameters.
    """

    stats = json.load(open(file))

    if option == 'latent-diffusion':
        # Check for per-component normalization (preferred for 3D)
        if 'U_per_component' in stats:
            pc = stats['U_per_component']
            max_u = pc['max_u']
            max_v = pc['max_v']
            max_w = pc.get('max_w', max_u)  # Fallback to max_u for 2D
            
            print(f"Using per-component normalization: max_u={max_u:.6f}, max_v={max_v:.6f}, max_w={max_w:.6f}")
            
            out = {
                'input': None,
                'output': (max_u, max_v, max_w)  # 3 channels: vx, vy, vz
            }
        else:
            # Fallback to global max (legacy behavior)
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

            print(f"WARNING: Using global max normalization ({max_velocity:.6f}) for all components.")
            print("         This may cause w component to be undertrained.")
            print("         Re-run dataset loading to generate per-component stats.")
            
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
    device: str = 'cuda',
    lambda_div: float = 0.0,
    lambda_flow: float = 0.0,
    lambda_smooth: float = 0.0,
    lambda_laplacian: float = 0.0,
    physics_loss_freq: int = 1,
    lambda_velocity: float = 0.0,
    weight_u: float = 1.0,
    weight_v: float = 1.0,
    weight_w: float = 1.0,
    velocity_loss_primary: bool = False
) -> tuple[float, float, Dict[str, float]]:
    """
    Optimize model for 1 epoch over training set, then evaluate over validation set.
    
    Args:
        loaders: data loaders for training and validation sets.
        predictor: ML model.
        optimizer: optimizer for tuning ML model parameters.
        criterion: cost function.
        device: device on which to train ML model.
        lambda_div: Weight for divergence (mass conservation) loss.
        lambda_flow: Weight for flow-rate consistency loss.
        lambda_smooth: Weight for gradient smoothness regularization.
        lambda_laplacian: Weight for Laplacian smoothness (reduces high-freq noise).
        physics_loss_freq: Compute physics loss every N batches.
        lambda_velocity: Weight for auxiliary velocity reconstruction loss.
        weight_u: Weight for u (vx) component in velocity loss.
        weight_v: Weight for v (vy) component in velocity loss.
        weight_w: Weight for w (vz) component in velocity loss.
        velocity_loss_primary: If True, use velocity loss as primary instead of noise prediction.
        
    Returns:
        avg_train_loss: Average training loss.
        avg_val_loss: Average validation loss.
        physics_metrics: Dictionary of physics metrics from validation.
    """

    train_loader, val_loader = loaders
    num_train_batch = len(train_loader)
    num_val_batch = len(val_loader)

    if isinstance(predictor, LatentDiffusionPredictor):
        option = 'latent-diffusion'
    else:
        raise ValueError(f'Unknown predictor type: {type(predictor)}')

    # Initialize physics loss calculator
    physics_loss_calc = PhysicsLoss(
        lambda_div=lambda_div,
        lambda_flow=lambda_flow,
        lambda_smooth=lambda_smooth,
        lambda_laplacian=lambda_laplacian,
        normalize_smoothness=True  # Enable scale-invariant smoothness losses
    )
    use_physics_loss = physics_loss_calc.is_active()
    use_velocity_loss = lambda_velocity > 0 or velocity_loss_primary

    # Track physics metrics
    accumulated_physics_metrics = {
        'div_mean': 0.0,
        'div_std': 0.0,
        'flow_rate_cv': 0.0,
        'vel_in_solid': 0.0,
        'vel_mean_fluid': 0.0,
        'gradient_smooth': 0.0,
        'laplacian_smooth': 0.0,
        'vel_u_mean': 0.0,
        'vel_v_mean': 0.0,
        'vel_w_mean': 0.0,
        'vel_u_max': 0.0,
        'vel_v_max': 0.0,
        'vel_w_max': 0.0
    }
    physics_loss_components = {
        'divergence': 0.0,
        'flow_rate': 0.0,
        'smoothness': 0.0,
        'laplacian': 0.0
    }
    
    # Track per-component velocity metrics
    component_metrics = {
        'loss_u': 0.0,
        'loss_v': 0.0,
        'loss_w': 0.0
    }

    """1. Training Set"""
    predictor.train()

    running_loss = 0
    running_physics_loss = 0
    for i, data in enumerate(train_loader):
        print(f"Training set: batch [{i+1}/{num_train_batch}]")

        input, targets = select_input_output(data, option, device)

        if option == 'latent-diffusion':
            # For latent diffusion: encode target to latent space and add noise
            img = input[0]  # (batch, num_slices, 1, H, W)
            velocity_2d = input[1]  # (batch, num_slices, 3, H, W)
            
            # Encode 3D velocity target to latent space
            target_latents = predictor.encode_target(targets, velocity_2d)
            
            # Get latent dimensions for reconstruction
            batch_size = img.shape[0]
            num_slices = img.shape[1]
            latent_depth = target_latents.shape[1]
            latent_channels = target_latents.shape[2]
            latent_h = target_latents.shape[3]
            latent_w = target_latents.shape[4]
            
            # Sample noise and timesteps
            noise = torch.randn_like(target_latents)
            
            # Flatten for diffusion process
            target_latents_flat = target_latents.reshape(
                batch_size * latent_depth, latent_channels, latent_h, latent_w
            )
            noise_flat = noise.reshape(
                batch_size * latent_depth, latent_channels, latent_h, latent_w
            )
            
            # Sample timesteps
            t = torch.randint(0, predictor.num_timesteps, (batch_size * latent_depth,), device=device).long()
            
            # Forward diffusion: add noise
            predictor.scheduler.to(device)
            x_t = predictor.scheduler.q_sample(target_latents_flat, t, noise_flat)
            
            # Predict noise using forward pass
            preds, target_noise = predictor(img, velocity_2d, x_start=target_latents, noise=noise)
            
            # Primary loss: either noise prediction OR velocity-based
            if velocity_loss_primary:
                # Use per-channel velocity loss as primary
                # Reconstruct velocity from noise prediction
                velocity_pred = reconstruct_velocity_from_noise_pred(
                    noise_pred=preds,
                    x_t=x_t,
                    t=t,
                    scheduler=predictor.scheduler,
                    vae_decoder=predictor.vae.decode,
                    normalizer_output=predictor.normalizer['output'],
                    batch_size=batch_size,
                    latent_depth=latent_depth,
                    latent_channels=latent_channels,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    num_slices=num_slices,
                    img=img
                )
                
                # Per-channel velocity loss as PRIMARY
                loss, vel_components = component_weighted_velocity_loss(
                    velocity_pred=velocity_pred,
                    velocity_target=targets,
                    mask=img,
                    weight_u=weight_u,
                    weight_v=weight_v,
                    weight_w=weight_w
                )
                
                # Accumulate component metrics for logging
                for key in component_metrics:
                    if key in vel_components:
                        component_metrics[key] += vel_components[key].item()
            else:
                # Standard noise prediction loss
                loss = criterion(output=preds, target=target_noise)
            
            # Physics-informed loss (computed periodically for efficiency)
            physics_loss = torch.tensor(0.0, device=device)
            velocity_loss = torch.tensor(0.0, device=device)
            velocity_pred_for_physics = None
            
            # For physics loss or auxiliary velocity loss, reconstruct velocity if not already done
            if (use_physics_loss or (use_velocity_loss and not velocity_loss_primary)) and (physics_loss_freq > 0) and (i % physics_loss_freq == 0):
                try:
                    # Reconstruct velocity if not already done for primary loss
                    if velocity_loss_primary:
                        velocity_pred_for_physics = velocity_pred
                    else:
                        velocity_pred_for_physics = reconstruct_velocity_from_noise_pred(
                            noise_pred=preds,
                            x_t=x_t,
                            t=t,
                            scheduler=predictor.scheduler,
                            vae_decoder=predictor.vae.decode,
                            normalizer_output=predictor.normalizer['output'],
                            batch_size=batch_size,
                            latent_depth=latent_depth,
                            latent_channels=latent_channels,
                            latent_h=latent_h,
                            latent_w=latent_w,
                            num_slices=num_slices,
                            img=img
                        )
                    
                    # Compute physics loss
                    if use_physics_loss:
                        physics_loss, components = physics_loss_calc(velocity_pred_for_physics, img)
                        
                        # Accumulate component losses for logging
                        for key in physics_loss_components:
                            if key in components:
                                physics_loss_components[key] += components[key].item()
                    
                    # Compute auxiliary component-weighted velocity loss (only if not primary)
                    if use_velocity_loss and not velocity_loss_primary and lambda_velocity > 0:
                        aux_velocity_loss, vel_components = component_weighted_velocity_loss(
                            velocity_pred=velocity_pred_for_physics,
                            velocity_target=targets,
                            mask=img,
                            weight_u=weight_u,
                            weight_v=weight_v,
                            weight_w=weight_w
                        )
                        velocity_loss = lambda_velocity * aux_velocity_loss
                        
                        # Accumulate component metrics for logging
                        for key in component_metrics:
                            if key in vel_components:
                                component_metrics[key] += vel_components[key].item()
                    
                except Exception as e:
                    print(f"Warning: Physics/velocity loss computation failed: {e}")
                    physics_loss = torch.tensor(0.0, device=device)
                    velocity_loss = torch.tensor(0.0, device=device)
            
            # Total loss
            total_loss = loss + physics_loss + velocity_loss

        else:
            raise ValueError(f"Unknown option: {option}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if use_physics_loss:
            running_physics_loss += physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss

        # Explicitly clean up heavy memory after physics loss buffer
        if (use_physics_loss or (use_velocity_loss and not velocity_loss_primary)) and (physics_loss_freq > 0) and (i % physics_loss_freq == 0):
            # Delete large tensors holding computational graphs
            del velocity_pred_for_physics
            del physics_loss
            del velocity_loss
            del total_loss
            
            # Force empty cache to release VAE decoder memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_train_loss = running_loss / (i+1)
    avg_physics_loss = running_physics_loss / (i+1) if use_physics_loss else 0.0
    
    if use_physics_loss:
        print(f"  Train physics loss: {avg_physics_loss:.6f}")
        # Normalize component losses
        for key in physics_loss_components:
            physics_loss_components[key] /= max(1, (i+1) // physics_loss_freq)
    
    if use_velocity_loss:
        # Normalize component metrics
        count = max(1, (i+1) // physics_loss_freq)
        print(f"  Train velocity loss components: u={component_metrics['loss_u']/count:.6f}, "
              f"v={component_metrics['loss_v']/count:.6f}, w={component_metrics['loss_w']/count:.6f}")


    """2. Validation Set"""
    predictor.eval()

    with torch.no_grad():
        val_loss = 0
        val_physics_count = 0
        
        for j, data in enumerate(val_loader):
            print(f"Validation set: batch [{j+1}/{num_val_batch}]")

            input, targets = select_input_output(data, option, device)

            if option == 'latent-diffusion':
                img = input[0]
                velocity_2d = input[1]
                
                # Encode target to latent space
                target_latents = predictor.encode_target(targets, velocity_2d)
                
                # Predict noise
                noise = torch.randn_like(target_latents)
                preds, target_noise = predictor(img, velocity_2d, x_start=target_latents, noise=noise)
                
                # Loss in latent space
                loss = criterion(output=preds, target=target_noise)
                
                # Compute physics metrics on validation set
                if use_physics_loss or use_velocity_loss:
                    try:
                        # Use full prediction for metrics (not just noise pred)
                        batch_size = img.shape[0]
                        num_slices = img.shape[1]
                        latent_depth = target_latents.shape[1]
                        latent_channels = target_latents.shape[2]
                        latent_h = target_latents.shape[3]
                        latent_w = target_latents.shape[4]
                        
                        # Get timesteps for reconstruction
                        t = torch.randint(0, predictor.num_timesteps, (batch_size * latent_depth,), device=device).long()
                        target_latents_flat = target_latents.reshape(
                            batch_size * latent_depth, latent_channels, latent_h, latent_w
                        )
                        noise_flat = noise.reshape(
                            batch_size * latent_depth, latent_channels, latent_h, latent_w
                        )
                        x_t = predictor.scheduler.q_sample(target_latents_flat, t, noise_flat)
                        
                        velocity_pred = reconstruct_velocity_from_noise_pred(
                            noise_pred=preds,
                            x_t=x_t,
                            t=t,
                            scheduler=predictor.scheduler,
                            vae_decoder=predictor.vae.decode,
                            normalizer_output=predictor.normalizer['output'],
                            batch_size=batch_size,
                            latent_depth=latent_depth,
                            latent_channels=latent_channels,
                            latent_h=latent_h,
                            latent_w=latent_w,
                            num_slices=num_slices,
                            img=img
                        )
                        
                        # Compute physics metrics (verbose on first batch only)
                        batch_metrics = compute_physics_metrics(velocity_pred, img, verbose=(j == 0))
                        for key in accumulated_physics_metrics:
                            if key in batch_metrics:
                                accumulated_physics_metrics[key] += batch_metrics[key]
                        val_physics_count += 1
                        
                    except Exception as e:
                        print(f"Warning: Physics metrics computation failed: {e}")
                        import traceback
                        traceback.print_exc()

            else:
                raise ValueError(f"Unknown option: {option}")
            
            val_loss += loss.item()

        avg_val_loss = val_loss / (j+1)
        
        # Average physics metrics
        if val_physics_count > 0:
            for key in accumulated_physics_metrics:
                accumulated_physics_metrics[key] /= val_physics_count

    # Combine all metrics
    all_metrics = {
        **accumulated_physics_metrics,
        **{f'loss_{k}': v for k, v in physics_loss_components.items()}
    }

    return (avg_train_loss, avg_val_loss, all_metrics)


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
