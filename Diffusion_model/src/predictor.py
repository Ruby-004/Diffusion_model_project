from abc import ABC, abstractmethod
from typing import Union, Any
import json
import os
import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

from .normalizer import Normalizer, MaxNormalizer
from .unet.models import UNet
from utils.zenodo import download_data, unzip_data, is_url

import sys
sys.path.append(osp.join(osp.dirname(__file__), '..', '..'))
from VAE_model.src.vae.autoencoder import VariationalAutoencoder


_model_type = Union[UNet, Any]


class Predictor(ABC, nn.Module):
    
    def __init__(
        self,
        model_name: str,
        model_kwargs: dict,
        distance_transform: bool,
    ):
        super().__init__()

        self.model_name = model_name
        self.model: _model_type = eval(model_name)(**model_kwargs)

        in_channels = model_kwargs['in_channels']
        out_channels = model_kwargs['out_channels']
        self.normalizer: dict[str, Normalizer] = self.init_normalizer(
            in_channels=in_channels,
            out_channels=out_channels
        )

        self.distance_transform = nn.Parameter(
            torch.Tensor([distance_transform]),
            requires_grad=False
        )
    
    @property
    @abstractmethod
    def type(self) -> str:
        pass

    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def predict(self, *args):
        pass

    @abstractmethod
    def pre_process(self, *args):
        pass

    @property
    def trainable_params(self):
        total = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        return total
    
    def init_normalizer(self, in_channels: int, out_channels: int):
        """Initialize normalizers for model input & output"""

        self.normalizer = nn.ModuleDict({
            'input': MaxNormalizer(scale_factors=[1 for _ in range(in_channels)]),
            'output': MaxNormalizer(scale_factors=[1 for _ in range(out_channels)])
        })
        return self.normalizer
    
    def set_normalizer(
        self,
        norm_dict: dict[str, Union[tuple, list, None]]
    ):
        """Set parameters for normalizers."""

        for key, val in norm_dict.items():
            if val is not None:
                self.normalizer[key] = MaxNormalizer(scale_factors=val)

        return self.normalizer

    def load_weights(self, model_path: str, device: str):
        """Load model parameters from `.pt` file."""
        self.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device(device)
            )
        )
        print(f'Loaded weights from "{model_path}".')

    @classmethod
    def from_directory(cls, folder: str, device: str) -> 'Predictor':
        """
        Load trained ML model from folder.
        
        Args:
            folder: Directory containing `model.pt` and `log.json` files created during training.
            device: Device (e.g. "cuda:0" or "cpu") to map the model to.
        """

        log_file = osp.join(folder, 'log.json')
        model_path = osp.join(folder, 'model.pt')

        with open(log_file) as fp:
            log_data = json.load(fp)
        
        param_dict = log_data['params']
        predictor_type = param_dict['training']['predictor_type']
        predictor_kwargs = param_dict['training']['predictor']

        if predictor_type == 'latent-diffusion':
            predictor_class = LatentDiffusionPredictor
        else:
            raise ValueError(f'Unknown or unsupported predictor type: {predictor_type}')

        predictor = predictor_class(**predictor_kwargs)
        predictor.to(device)
        predictor.load_weights(model_path, device=device)
        return predictor

    @classmethod
    def from_url(cls, url: str, device: str) -> 'Predictor':
        """
        Load trained ML model from URL. Pre-trained models are hosted here: https://doi.org/10.5281/zenodo.17306446.
        
        Args:
            url: URL pointing to a zipped folder containing `model.pt` and `log.json` files created during training.
            device: Device (e.g. "cuda:0" or "cpu") to map the model to.
        """

        _folder = 'pretrained'
        if not osp.exists(_folder): os.mkdir(_folder)

        # download pre-trained weights
        zip_path = download_data(url=url, save_dir=_folder)

        # unzip data
        folder_path = unzip_data(zip_path=zip_path, save_dir=_folder)

        predictor = cls.from_directory(folder_path, device=device)
        return predictor

    @classmethod
    def from_directory_or_url(
        cls,
        directory_or_url: str,
        device: str
    ) -> 'Predictor':
        """
        Load trained ML model from local directory or URL.
        
        Args:
            directory_or_url: either local directory or URL of the pre-trained model.
            device: Device (e.g. "cuda:0" or "cpu") to map the model to.
        """

        if is_url(directory_or_url):
            predictor = cls.from_url(url=directory_or_url, device=device)
        else:
            predictor = cls.from_directory(folder=directory_or_url, device=device)
        return predictor


class LatentDiffusionPredictor(Predictor):
    """
    Model for 3D velocity field prediction using multi-step diffusion in VAE latent space.
    """
    type: str = 'latent-diffusion'

    def __init__(
        self,
        model_name='UNet',
        model_kwargs: dict = {},
        distance_transform = True,
        vae_path: str = None,
        num_slices: int = 11,
    ) -> None:
        
        # Helper imports
        from .diffusion import DiffusionScheduler

        # Add time embedding dimension to model kwargs if not present
        if 'time_embedding_dim' not in model_kwargs:
            model_kwargs['time_embedding_dim'] = 64
            
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            distance_transform=distance_transform
        )
        
        self.num_slices = num_slices
        self.num_timesteps = 1000
        
        # Init scheduler
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scheduler = DiffusionScheduler(num_timesteps=self.num_timesteps, device=device)
        
        # Override normalizer for latent diffusion:
        # - Input: 1 channel (binary microstructure)
        # - Output: latent_channels (4 by default, set from model out_channels)
        latent_channels = model_kwargs.get('out_channels', 4)
        self.normalizer = nn.ModuleDict({
            'input': MaxNormalizer(scale_factors=[1]),  # 1 channel for microstructure
            'output': MaxNormalizer(scale_factors=[1 for _ in range(latent_channels)])
        })
        
        # Load pre-trained VAE
        if vae_path is None:
            raise ValueError("VAE path must be provided for latent diffusion")
        
        # Convert to absolute path if relative
        if not osp.isabs(vae_path):
            # Get the project root directory (2 levels up from this file)
            project_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
            vae_path = osp.join(project_root, vae_path)
        
        # Load VAE with correct architecture (3 input channels from velocity only)
        # Use latent_channels from model_kwargs if provided (should match output channels)
        self.vae = VariationalAutoencoder.from_directory(
            vae_path, 
            in_channels=3,  # 3 channels: velocity (vx, vy, vz)
            latent_channels=latent_channels
        )
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        
        print(f'Initialized {self.type} predictor with {self.trainable_params} parameters.')
        print(f'Loaded VAE from {vae_path} (frozen).')
        
    def to(self, device):
        super().to(device)
        self.scheduler.to(device)
        return self

    def forward(self, img: torch.Tensor, velocity_2d: torch.Tensor, x_start: torch.Tensor = None, noise: torch.Tensor = None):
        """
        Forward pass for multi-step diffusion model in latent space.
        
        Args:
            x_start: Target latents (clean data). Required for training step.
        """

        batch_size = img.shape[0]
        device = img.device
        num_slices = velocity_2d.shape[1]
        
        # img has shape (batch, num_slices, 1, H, W) - flatten to (batch*num_slices, 1, H, W)
        img_flat = img.view(batch_size * num_slices, 1, img.shape[3], img.shape[4])
        
        # Get latent dimensions from VAE encoder (uses 3D conv, expects 5D input)
        with torch.no_grad():
            # Create dummy input: (batch, channels, depth, height, width)
            dummy_5d = torch.zeros(1, 3, num_slices, img.shape[3], img.shape[4]).to(device)
            latent_shape = self.vae.encoder(dummy_5d)[0].shape  # (1, latent_channels, depth/4, H/4, W/4)
            latent_channels = latent_shape[1]
            latent_depth = latent_shape[2]
            latent_h, latent_w = latent_shape[3], latent_shape[4]
        
        # Encode velocity_2d to latent space to use as conditioning
        # Permute to (batch, channels, num_slices, H, W) for 3D VAE
        velocity_2d_permuted = velocity_2d.permute(0, 2, 1, 3, 4)  # (batch, 3, num_slices, H, W)
        
        # Encode using 3D VAE
        with torch.no_grad():
            velocity_2d_latent_5d, _ = self.vae.encode(velocity_2d_permuted)  # (batch, latent_channels, depth/4, H/4, W/4)
        
        # Permute to (batch, depth, latent_channels, H, W)
        velocity_2d_latent = velocity_2d_latent_5d.permute(0, 2, 1, 3, 4)
        
        # Preprocess microstructure slices (apply distance transform if needed)
        mask_flat = img_flat.clone()  # Binary mask for fluid/solid regions (batch*num_slices, 1, H, W)
        feats_flat = self.pre_process(img_flat)  # Shape: (batch*num_slices, 1, H, W) at original resolution
        
        # Downsample mask to latent resolution
        mask_downsampled = torch.nn.functional.interpolate(
            mask_flat, 
            size=(latent_h, latent_w), 
            mode='nearest'
        )  # (batch*num_slices, 1, latent_h, latent_w)
        
        # Downsample microstructure features to latent resolution
        feats_downsampled = torch.nn.functional.interpolate(
            feats_flat,
            size=(latent_h, latent_w),
            mode='bilinear',
            align_corners=False
        )  # (batch*num_slices, 1, latent_h, latent_w)
        
        # Flatten latents: (batch * latent_depth, latent_channels, latent_h, latent_w)
        # noise_flat = noise.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w) # We don't have noise here yet
        velocity_2d_latent_flat = velocity_2d_latent.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w)
        
        # Repeat microstructure features to match latent depth
        # We have num_slices microstructure slices but latent_depth latent slices
        # Interpolate feats to match latent_depth
        feats_3d = feats_downsampled.reshape(batch_size, num_slices, 1, latent_h, latent_w)
        mask_3d = mask_downsampled.reshape(batch_size, num_slices, 1, latent_h, latent_w)
        
        # Interpolate along depth dimension to match latent_depth
        feats_3d_interp = torch.nn.functional.interpolate(
            feats_3d.permute(0, 2, 1, 3, 4),  # (batch, 1, num_slices, H, W)
            size=(latent_depth, latent_h, latent_w),
            mode='trilinear',
            align_corners=False
        ).permute(0, 2, 1, 3, 4)  # (batch, latent_depth, 1, H, W)
        
        mask_3d_interp = torch.nn.functional.interpolate(
            mask_3d.permute(0, 2, 1, 3, 4),
            size=(latent_depth, latent_h, latent_w),
            mode='nearest'
        ).permute(0, 2, 1, 3, 4)
        
        feats_latent_flat = feats_3d_interp.reshape(batch_size * latent_depth, 1, latent_h, latent_w)
        mask_latent_flat = mask_3d_interp.reshape(batch_size * latent_depth, 1, latent_h, latent_w)
        
        # Training Logic
        if x_start is not None:
             # Flatten x_start to match latent_depth
             # x_start shape from encode_target: (batch, latent_depth, latent_channels, H, W)
             x_start_flat = x_start.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w)
             
             if noise is None:
                 noise = torch.randn_like(x_start_flat)
             else:
                 # Ensure noise is also flattened
                 noise = noise.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w)
                 
             # Sample timesteps
             t = torch.randint(0, self.num_timesteps, (batch_size * latent_depth,), device=device).long()
             
             # Add noise
             self.scheduler.to(device)
             x_t = self.scheduler.q_sample(x_start_flat, t, noise)
             
             # Network Input: Concatenate Noisy Latent + Conditioning
             unet_input = torch.cat([x_t, velocity_2d_latent_flat, feats_latent_flat], dim=1)
             
             # Predict noise
             noise_pred = self.model(unet_input, t)
             
             return noise_pred, noise
        
        else:
             raise ValueError("forward() requires x_start (target latents) for training. Use predict() for inference.")


    def predict(self, img: torch.Tensor, velocity_2d: torch.Tensor, noise: torch.Tensor = None):
        """
        Predict 3D velocity field using iterative denoising.
        """
        batch_size = img.shape[0]
        device = img.device
        num_slices = velocity_2d.shape[1]
        
        img_flat = img.view(batch_size * num_slices, 1, img.shape[3], img.shape[4])
        
        # Get dimensions
        with torch.no_grad():
             dummy_5d = torch.zeros(1, 3, num_slices, img.shape[3], img.shape[4]).to(device)
             latent_shape = self.vae.encoder(dummy_5d)[0].shape
             latent_channels = latent_shape[1]
             latent_depth = latent_shape[2]
             latent_h, latent_w = latent_shape[3], latent_shape[4]
             
        # Prepare conditioning (Copy from forward)
        velocity_2d_permuted = velocity_2d.permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            velocity_2d_latent_5d, _ = self.vae.encode(velocity_2d_permuted)
        velocity_2d_latent = velocity_2d_latent_5d.permute(0, 2, 1, 3, 4)
        
        feats_flat = self.pre_process(img_flat)
        feats_downsampled = torch.nn.functional.interpolate(feats_flat, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
        
        velocity_2d_latent_flat = velocity_2d_latent.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w)
        
        feats_3d = feats_downsampled.reshape(batch_size, num_slices, 1, latent_h, latent_w)
        feats_3d_interp = torch.nn.functional.interpolate(
            feats_3d.permute(0, 2, 1, 3, 4),
            size=(latent_depth, latent_h, latent_w),
            mode='trilinear',
            align_corners=False
        ).permute(0, 2, 1, 3, 4)
        feats_latent_flat = feats_3d_interp.reshape(batch_size * latent_depth, 1, latent_h, latent_w)
        
        # Sampling Loop
        if noise is None:
            noise = torch.randn(batch_size * latent_depth, latent_channels, latent_h, latent_w, device=device)
        else:
            noise = noise.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w)
            
        x = noise
        self.scheduler.to(device)
        
        for t in reversed(range(0, self.num_timesteps)):
             t_batch = torch.full((batch_size * latent_depth,), t, device=device, dtype=torch.long)
             
             unet_input = torch.cat([x, velocity_2d_latent_flat, feats_latent_flat], dim=1)
             
             # Predict noise
             with torch.no_grad():
                  noise_pred = self.model(unet_input, t_batch)
             
             # Step
             x = self.scheduler.p_sample(noise_pred, x, t)
        
        predicted_latent_flat = x
        
        # Decode
        predicted_latents = predicted_latent_flat.reshape(batch_size, latent_depth, latent_channels, latent_h, latent_w)
        predicted_latents_5d = predicted_latents.permute(0, 2, 1, 3, 4)
        
        with torch.no_grad():
            velocity_5d = self.vae.decode(predicted_latents_5d)
        
        # Permute back to (batch, num_slices, 3, H, W)
        velocity_3d = velocity_5d.permute(0, 2, 1, 3, 4)
        
        # Denormalize
        batch, depth, channels, height, width = velocity_3d.shape
        velocity_flat = velocity_3d.reshape(batch * depth, channels, height, width)
        velocity_flat = self.normalizer['output'].inverse(velocity_flat)
        velocity_3d = velocity_flat.reshape(batch, depth, channels, height, width)
        
        return velocity_3d

    def pre_process(self, img: torch.Tensor):
        """
        Pre-process microstructure inputs.

        Args:
            img: (binary) microstructure images. Shape: (batch, 1, height, width).
        """
        assert img.dim() == 4
        assert img.shape[1] == 1

        if self.distance_transform:
            img = apply_distance_transform(img)

        features = self.normalizer['input'](img)

        return features
    
    def encode_target(self, velocity_3d: torch.Tensor, velocity_2d: torch.Tensor = None):
        """
        Encode 3D velocity target into latent space using VAE.
        
        Args:
            velocity_3d: Target 3D velocity field. Shape: (batch, num_slices, 3, H, W)
            velocity_2d: Input 2D velocity field (optional). Shape: (batch, num_slices, 3, H, W)
        
        Returns:
            latents: Encoded latent representations. Shape: (batch, num_slices, latent_channels, latent_h, latent_w)
        """
        # VAE expects 5D input: (batch, channels, depth, height, width)
        # Our data is: (batch, num_slices, channels, height, width)
        # Permute to match VAE input: (batch, channels, num_slices, height, width)
        velocity_permuted = velocity_3d.permute(0, 2, 1, 3, 4)  # (batch, 3, num_slices, H, W)
        
        # Normalize - VAE expects normalized input
        # Flatten to 4D for normalizer: (batch, 3, num_slices*H, W) won't work
        # Instead, process each sample separately or reshape properly
        batch_size = velocity_permuted.shape[0]
        channels = velocity_permuted.shape[1]
        depth = velocity_permuted.shape[2]
        height = velocity_permuted.shape[3]
        width = velocity_permuted.shape[4]
        
        # Reshape to (batch*depth, channels, height, width) for normalization
        velocity_flat = velocity_permuted.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, channels, height, width)
        velocity_flat_norm = self.normalizer['output'](velocity_flat)
        # Reshape back to 5D: (batch, depth, channels, H, W) then permute to (batch, channels, depth, H, W)
        velocity_norm_5d = velocity_flat_norm.reshape(batch_size, depth, channels, height, width).permute(0, 2, 1, 3, 4)
        
        # Encode with VAE (no gradient through VAE)
        with torch.no_grad():
            latent_5d, _ = self.vae.encode(velocity_norm_5d)  # (batch, latent_channels, depth/4, H/4, W/4)
        
        # Permute back to (batch, depth, latent_channels, H, W) to match expected output
        latents = latent_5d.permute(0, 2, 1, 3, 4)  # (batch, depth/4, latent_channels, H/4, W/4)
        
        return latents
        
        return latents










def apply_distance_transform(imgs: torch.Tensor):
    """
    Perform distance transform of input images.

    Args:
        imgs: batch of images, with shape (n_img, 1, height, width)
    """
    device = imgs.device

    imgs = imgs.cpu().numpy()

    tmp_list = []
    for im in imgs:
        im = im[0]
        im_tr = ndimage.distance_transform_edt(im)
        tmp_list.append([[im_tr]])
    
    out = torch.from_numpy(
        np.concatenate(tmp_list)
    ).float()
    return out.to(device)
