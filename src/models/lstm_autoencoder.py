"""LSTM autoencoder for sequence reconstruction."""

from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """Sequence autoencoder that reconstructs each timestep signal."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        # PyTorch ignores LSTM dropout when num_layers=1; make it explicit
        # to avoid confusion when users tune hyperparameters.
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence and decode from repeated latent context."""
        _, (h_n, _) = self.encoder(x)
        batch_size, seq_len, _ = x.shape
        latent = h_n[-1]
        # Repeat final encoder state across timesteps for simple decoder input.
        repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(repeated)
        return self.output_layer(decoded)
