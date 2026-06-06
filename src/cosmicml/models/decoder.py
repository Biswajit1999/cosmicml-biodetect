"""Decoder module for reconstructing atmospheric composition from latent space."""

import torch.nn as nn
from typing import List


class Decoder(nn.Module):
    """
    Decoder for reconstructing atmospheric composition from latent representation.

    Maps latent vectors back to atmospheric composition space (mixing ratios).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        output_dim: int = 32,
        hidden_layers: List[int] = None,
    ):
        """
        Initialize decoder.

        Args:
            latent_dim: Dimension of latent space
            output_dim: Dimension of output (number of species)
            hidden_layers: Sizes of hidden layers
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 256]

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        # Soft normalize to ensure physical validity
        layers.append(nn.Softplus())

        self.network = nn.Sequential(*layers)

    def forward(self, latent):
        """Decode latent representation to atmospheric composition."""
        return self.network(latent)
