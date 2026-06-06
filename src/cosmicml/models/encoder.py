"""Encoder module for dimensionality reduction in spectral data."""

import torch.nn as nn
from typing import List


class Encoder(nn.Module):
    """
    Encoder for reducing high-dimensional spectral data to latent representation.

    Maps transmission spectra (512 wavelength bins) to lower-dimensional latent space
    that captures the essential atmospheric information.
    """

    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 64,
        hidden_layers: List[int] = None,
    ):
        """
        Initialize encoder.

        Args:
            input_dim: Dimension of input spectrum
            latent_dim: Dimension of latent space
            hidden_layers: Sizes of hidden layers
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, spectrum):
        """Encode spectrum to latent representation."""
        return self.network(spectrum)
