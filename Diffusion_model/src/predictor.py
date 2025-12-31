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

        if predictor_type == 'velocity':
            predictor_class = VelocityPredictor
        elif predictor_type == 'pressure':
            predictor_class = PressurePredictor
        else:
            raise ValueError(f'Unknown predictor type: {predictor_type}')

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


class VelocityPredictor(Predictor):
    """
    Model for velocity field prediction in microstructures.
    """
    type: str = 'velocity'

    def __init__(
        self,
        model_name='UNet',
        model_kwargs: dict = {},
        distance_transform = True,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            distance_transform=distance_transform
        )
        print(f'Initialized {self.type} predictor with {self.trainable_params} parameters.')

    def forward(self, img: torch.Tensor):
        """
        Forward pass for the model.

        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
        """

        mask = img.clone()

        feats = self.pre_process(img)
        out = self.model(feats)

        out = out * mask # multiply by mask

        return out

    def predict(self, img: torch.Tensor):
        """
        Predict velocity field from input microstructure images.

        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
        """

        pred = self(img)
        out = self.normalizer['output'].inverse(pred)
        return out

    def pre_process(self, img: torch.Tensor):
        """
        Pre-process inputs.

        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
        """
        assert img.dim() == 4
        assert img.shape[1] == 1 # only 1 channel

        if self.distance_transform:
            img = apply_distance_transform(img)

        features = self.normalizer['input'](img)

        return features


class PressurePredictor(Predictor):
    """
    Model for pressure field prediction in microstructures.
    """
    type: str = 'pressure'

    def __init__(
        self,
        model_name='UNet',
        model_kwargs: dict = {},
        distance_transform = False,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            distance_transform=distance_transform
        )
        print(f'Initialized {self.type} predictor with {self.trainable_params} parameters.')

    def forward(
        self,
        img: torch.Tensor,
        x_length: torch.Tensor
    ):
        """
        Forward pass for the model.

        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
            x_length: Microstructure (physical) length. Shape: (batch, 1) or (batch, 1, height, width).
        """

        mask = img.clone()

        feats = self.pre_process(img, x_length)
        out = self.model(feats)

        out = out * mask # multiply by mask

        return out

    def predict(
        self,
        img: torch.Tensor,
        x_length: torch.Tensor
    ):
        """
        Predict pressure field from input microstructure images.
        
        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
            x_length: Microstructure (physical) length. Shape: (batch, 1) or (batch, 1, height, width).
        """

        pred = self(img, x_length)
        out = self.normalizer['output'].inverse(pred) # ρ-normalized pressure
        return out
    

    def pre_process(
        self,
        img: torch.Tensor,
        x_length: torch.Tensor
    ):
        """
        Pre-process inputs.

        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
            x_length: Microstructure (physical) length. Shape: (batch, 1) or (batch, 1, height, width).
        """

        x_length = self._process_microstructure_length(x_length, img.shape)
        assert img.dim() == 4
        assert x_length.dim() == 4
        img_copy = img.clone()

        if self.distance_transform:
            img = apply_distance_transform(img)

        features = torch.cat(
            (img, x_length),
            dim=1
        ) # shape: (samples, 2, height, width)

        features = self.normalizer['input'](features)


        """Additional modifications"""

        # multiply microstructure by fiber volume fraction
        fiber_vf = self._compute_fiber_vf(img_copy)
        features[:, [0]] = features[:, [0]] * fiber_vf

        # use inverse of microstructure length
        features[:, [1]] = 1 / features[:, [1]]

        return features


    @staticmethod
    def _process_microstructure_length(
        x_length: torch.Tensor,
        shape: tuple
    ):
        """
        Evaluate whether microstructures' length is passed as scalars or matrices. Returns a 4D tensor.
        
        Args:
            x_length: microstructure (physical) length.
            shape: shape (samples, 1, height, width) of microstructure images. Used to expand `x_length` if needed.
        """
        if x_length.dim() == 4:
            # shape: (samples, 1, height, width)
            pass
        else:
            x_length = x_length.squeeze() # shape: (samples,)
            x_length = x_length.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # shape: (samples, 1, 1, 1)
            x_length = x_length * torch.ones(shape, device=x_length.device)

        return x_length

    @staticmethod
    def _compute_fiber_vf(img: torch.Tensor):
        """
        Compute fiber volume fraction

        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
        """

        fluid_vf = torch.sum(
            img,
            dim=[-1,-2]
        ) / (img.shape[-1] * img.shape[-2])

        fiber_vf = 1 - fluid_vf
        fiber_vf = fiber_vf.unsqueeze(-1).unsqueeze(-1) # shape: (samples, 1, height, width)

        return fiber_vf
        


class LatentDiffusionPredictor(Predictor):
    """
    Model for 3D velocity field prediction using one-step diffusion in VAE latent space.
    Takes 2D microstructure as input, predicts 3D flow (stack of 2D slices) in latent space.
    """
    type: str = 'latent-diffusion'

    def __init__(
        self,
        model_name='UNet',
        model_kwargs: dict = {},
        distance_transform = True,
        vae_path: str = None,
        num_slices: int = 10,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            distance_transform=distance_transform
        )
        
        self.num_slices = num_slices
        
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
        self.vae = VariationalAutoencoder.from_directory(
            vae_path, 
            in_channels=3,  # 3 channels: velocity (vx, vy, vz)
            latent_channels=4
        )
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        
        print(f'Initialized {self.type} predictor with {self.trainable_params} parameters.')
        print(f'Loaded VAE from {vae_path} (frozen).')

    def forward(self, img: torch.Tensor, velocity_2d: torch.Tensor, noise: torch.Tensor = None):
        """
        Forward pass for one-step diffusion model in latent space.

        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. 
                 Shape: (batch, num_slices, 1, height, width) for 3D case.
            velocity_2d: 2D velocity input (where vz=0). Shape: (batch, num_slices, 3, H, W)
            noise: Optional noise for training. If None, uses random noise.
                 Shape: (batch, num_slices, latent_channels, latent_height, latent_width).
        
        Returns:
            predicted_latents: Shape (batch, num_slices, latent_channels, latent_height, latent_width)
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
        
        # Generate or use provided noise  
        if noise is None:
            noise = torch.randn(batch_size, latent_depth, latent_channels, latent_h, latent_w).to(device)
        
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
        noise_flat = noise.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w)
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
        
        # Concatenate: noise + velocity_2d_latent + microstructure_features
        # Shape: (batch * latent_depth, latent_channels*2 + 1, latent_h, latent_w)
        unet_input = torch.cat([noise_flat, velocity_2d_latent_flat, feats_latent_flat], dim=1)
        
        # Predict denoised latent (one-step: predict clean 3D velocity latent from noise + 2D velocity)
        predicted_latent_flat = self.model(unet_input)
        
        # Apply mask in latent space (zero out solid regions)
        predicted_latent_flat = predicted_latent_flat * mask_latent_flat
        
        # Reshape back to 3D: (batch, latent_depth, latent_channels, latent_h, latent_w)
        predicted_latents = predicted_latent_flat.reshape(batch_size, latent_depth, latent_channels, latent_h, latent_w)
        
        return predicted_latents

    def predict(self, img: torch.Tensor, velocity_2d: torch.Tensor, noise: torch.Tensor = None):
        """
        Predict 3D velocity field from 2D velocity input and microstructure.

        Args:
            img: (binary) microstructure images. Shape: (batch, num_slices, 1, height, width).
            velocity_2d: 2D velocity input. Shape: (batch, num_slices, 3, H, W)
            noise: Optional starting noise. If None, uses random noise.
        
        Returns:
            velocity_3d: Predicted 3D velocity field. Shape: (batch, num_slices, 3, height, width)
        """
        
        # Get predicted latents: (batch, latent_depth, latent_channels, latent_h, latent_w)
        predicted_latents = self(img, velocity_2d, noise)
        
        batch_size = predicted_latents.shape[0]
        latent_depth = predicted_latents.shape[1]
        
        # Decode using 3D VAE
        # Permute to (batch, latent_channels, latent_depth, latent_h, latent_w) for VAE decoder
        predicted_latents_5d = predicted_latents.permute(0, 2, 1, 3, 4)
        
        with torch.no_grad():
            # Decode: (batch, 3, num_slices, H, W) <- note VAE upsamples depth dimension
            velocity_5d = self.vae.decode(predicted_latents_5d)
        
        # Permute back to (batch, num_slices, 3, H, W)
        velocity_3d = velocity_5d.permute(0, 2, 1, 3, 4)
        
        # Denormalize - reshape to 4D for normalizer
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
