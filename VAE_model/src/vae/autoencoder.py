import os.path as osp
import json

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class VariationalAutoencoder(nn.Module):
    """
    Variational autoencoder model combining Encoder and Decoder.
    """
    _model_filename = 'vae.pt'
    _log_filename = 'vae_log.json'

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        kernel_size: int = 3
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            kernel_size=kernel_size
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=in_channels,
            kernel_size=kernel_size
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the autoencoder.
        """

        # Encode
        latent, (mean, logvar) = self.encode(x)

        # Decode
        recons = self.decode(latent)

        return recons, (mean, logvar)

    def encode(self, x: torch.Tensor):
        """
        Encode input into latent representation.
        """
        # encoding
        mean, logvar = self.encoder(x)

        # Clamping logvar to prevent numerical instability during sampling
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        # sampling
        latent = self.encoder.sample(mu=mean, logvar=logvar)

        return latent, (mean, logvar)

    def decode(self, z: torch.Tensor):
        """
        Decode latent representation into original space.
        """
        # decode
        recons = self.decoder(z)
        return recons

    def save_model(self, folder, log: dict = None):
        """
        Save model parameters.
        """
        model_path = osp.join(folder, self._model_filename)
        torch.save(self.state_dict(), model_path)

        if log is not None:
            log_path = osp.join(folder, self._log_filename)
            with open(log_path, 'w') as f:
                json.dump(log, f, indent=4)
                
    def load_model(self, folder, device=None):
        """
        Load model parameters.
        """
        model_path = osp.join(folder, self._model_filename)
        self.load_state_dict(torch.load(model_path, map_location=device))
    
    @classmethod
    def from_directory(cls, folder, device=None, in_channels=None, latent_channels=None, kernel_size=3):
        """
        Create model instance from saved parameters.
        
        Args:
            folder: Directory containing vae.pt and optionally vae_log.json
            device: Device to load model on
            in_channels: Number of input channels (if None, tries to read from log or defaults to 2)
            latent_channels: Number of latent channels (if None, tries to read from log or defaults to 4)
            kernel_size: Kernel size (default 3)
        """
        # Try to load log if it exists
        log_path = osp.join(folder, cls._log_filename)
        if osp.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    log = json.load(f)
                
                # Use log values if parameters not provided
                if in_channels is None and 'in_channels' in log:
                    in_channels = log['in_channels']
                if latent_channels is None and 'latent_channels' in log:
                    latent_channels = log['latent_channels']
                if 'kernel_size' in log:
                    kernel_size = log['kernel_size']
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Use defaults if still None
        if in_channels is None:
            in_channels = 2  # Default for velocity fields (vx, vy)
        if latent_channels is None:
            latent_channels = 4

        # create model
        model = cls(
            in_channels=in_channels,
            latent_channels=latent_channels,
            kernel_size=kernel_size
        )

        # load parameters
        model.load_model(folder, device=device)

        return model