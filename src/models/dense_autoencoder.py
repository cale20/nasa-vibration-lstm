"""Dense autoencoder for sequence reconstruction on flattened windows."""

from __future__ import annotations

import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    """Small MLP autoencoder used as an intermediate baseline."""

    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        # Keep hidden width conservative so training remains practical
        # on CPU-only environments.
        hidden = max(latent_dim * 2, 64)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return reconstruction in the same shape as input batch."""
        z = self.encoder(x)
        return self.decoder(z)
