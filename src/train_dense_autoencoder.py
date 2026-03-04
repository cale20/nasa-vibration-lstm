"""Train a dense autoencoder on healthy vibration sequences."""

from __future__ import annotations

import argparse
import logging
import os
import time

import torch
from torch import nn

from .config import CONFIG, configure_logging, ensure_output_dirs
from .dataset import make_torch_dataloaders
from .logging_utils import fmt_seconds, log_note, log_progress
from .models import DenseAutoencoder


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
    max_batches: int | None = None,
    log_interval_batches: int | None = None,
):
    """Run one epoch and return average reconstruction loss."""
    model.train(mode=train)
    total = 0.0
    count = 0
    target_batches = len(loader)
    if max_batches is not None:
        target_batches = min(target_batches, max_batches)
    start = time.perf_counter()
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
        if log_interval_batches and (batch_idx + 1) % log_interval_batches == 0:
            elapsed = time.perf_counter() - start
            avg_batch_sec = elapsed / max(batch_idx + 1, 1)
            eta_sec = avg_batch_sec * max(target_batches - (batch_idx + 1), 0)
            stage = "train" if train else "val"
            log_progress(
                f"Dense AE {stage}: batch {batch_idx + 1}/{target_batches} | "
                f"avg_loss={total / max(count, 1):.6f} | elapsed={fmt_seconds(elapsed)} | eta={fmt_seconds(eta_sec)}"
            )
    return total / max(count, 1)


def train(
    epochs: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    log_interval_batches: int | None = None,
) -> str:
    """Train dense AE on healthy windows and save best validation checkpoint."""
    ensure_output_dirs()
    torch.manual_seed(int(CONFIG["random_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(CONFIG["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dense AE consumes flattened windows (shape: batch, seq_len * features).
    train_loader, val_loader = make_torch_dataloaders(flatten=True)
    log_note(
        f"Dense AE context: device={device}, train_batches={len(train_loader)}, "
        f"val_batches={len(val_loader)}, caps=({max_train_batches},{max_val_batches})"
    )
    input_dim = CONFIG["sequence_length"] * 1
    model = DenseAutoencoder(
        input_dim=input_dim, latent_dim=CONFIG["dense_latent_dim"]
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

    # Early stopping protects against overfitting and saves CPU/GPU time
    # during iterative experimentation.
    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            max_batches=max_train_batches,
            log_interval_batches=log_interval_batches,
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            max_batches=max_val_batches,
            log_interval_batches=log_interval_batches,
        )
        logging.info(
            "[dense-ae] epoch=%s train_loss=%.6f val_loss=%.6f",
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
                    "input_dim": input_dim,
                    "latent_dim": CONFIG["dense_latent_dim"],
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "random_seed": int(CONFIG["random_seed"]),
                },
                CONFIG["dense_autoencoder_model_file"],
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(
                    "[dense-ae] early stopping at epoch %s (best epoch %s)",
                    epoch,
                    best_epoch,
                )
                break

    return os.path.abspath(CONFIG["dense_autoencoder_model_file"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dense autoencoder")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Cap train batches per epoch for quick iteration")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Cap validation batches per epoch for quick iteration")
    parser.add_argument("--log-interval-batches", type=int, default=CONFIG["log_interval_batches"])
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    model_path = train(
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        log_interval_batches=args.log_interval_batches,
    )
    logging.info("Dense autoencoder saved to %s", model_path)


if __name__ == "__main__":
    main()
