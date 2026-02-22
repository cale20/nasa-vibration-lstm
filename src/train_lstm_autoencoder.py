"""Train an LSTM autoencoder on healthy vibration sequences."""

from __future__ import annotations

import argparse
import logging
import os

import torch
from torch import nn

from .config import CONFIG, configure_logging, ensure_output_dirs
from .dataset import make_torch_dataloaders
from .models import LSTMAutoencoder


def _run_epoch(model, loader, criterion, optimizer, device, train: bool, max_batches: int | None = None):
    """Run one epoch and return average reconstruction loss."""
    model.train(mode=train)
    total = 0.0
    count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = batch.to(device)
        if train:
            optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * x.shape[0]
        count += x.shape[0]
    return total / max(count, 1)


def train(
    epochs: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> str:
    """Train LSTM AE on healthy sequences and save best validation checkpoint."""
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # LSTM AE expects unflattened sequence tensors (batch, seq_len, features).
    train_loader, val_loader = make_torch_dataloaders(flatten=False)
    model = LSTMAutoencoder(
        input_size=1,
        hidden_size=CONFIG["lstm_hidden_size"],
        num_layers=CONFIG["lstm_num_layers"],
        dropout=CONFIG["lstm_dropout"],
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    max_epochs = epochs or CONFIG["epochs"]
    patience = CONFIG["early_stopping_patience"]
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    # Sequence models are expensive to train on CPU; early stopping keeps
    # iterative experiments practical while preserving best validation state.
    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            max_batches=max_train_batches,
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            max_batches=max_val_batches,
        )
        logging.info(
            "[lstm-ae] epoch=%s train_loss=%.6f val_loss=%.6f",
            epoch,
            train_loss,
            val_loss,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": 1,
                    "hidden_size": CONFIG["lstm_hidden_size"],
                    "num_layers": CONFIG["lstm_num_layers"],
                    "dropout": CONFIG["lstm_dropout"],
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                },
                CONFIG["lstm_autoencoder_model_file"],
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(
                    "[lstm-ae] early stopping at epoch %s (best epoch %s)",
                    epoch,
                    best_epoch,
                )
                break

    return os.path.abspath(CONFIG["lstm_autoencoder_model_file"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM autoencoder")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Cap train batches per epoch for quick iteration")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Cap validation batches per epoch for quick iteration")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    model_path = train(
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    logging.info("LSTM autoencoder saved to %s", model_path)


if __name__ == "__main__":
    main()
