"""Evaluate dense/LSTM autoencoders with reconstruction-error diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import joblib

from .config import CONFIG, configure_logging
from .dataset import load_memmap_dataset, load_split_metadata
from .models import DenseAutoencoder, LSTMAutoencoder
from .preprocessing import create_sequences


def _load_model(model_type: str, device: torch.device):
    """Load trained dense or LSTM autoencoder checkpoint."""
    if model_type == "dense":
        checkpoint = torch.load(CONFIG["dense_autoencoder_model_file"], map_location=device)
        model = DenseAutoencoder(
            input_dim=int(checkpoint["input_dim"]),
            latent_dim=int(checkpoint["latent_dim"]),
        ).to(device)
    elif model_type == "lstm":
        checkpoint = torch.load(CONFIG["lstm_autoencoder_model_file"], map_location=device)
        model = LSTMAutoencoder(
            input_size=int(checkpoint["input_size"]),
            hidden_size=int(checkpoint["hidden_size"]),
            num_layers=int(checkpoint["num_layers"]),
            dropout=float(checkpoint["dropout"]),
        ).to(device)
    else:
        raise ValueError("model_type must be one of: dense, lstm")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _reconstruction_errors(model, data: np.ndarray, device: torch.device, flatten: bool) -> np.ndarray:
    """Compute per-sample reconstruction MSE in batches."""
    batch_size = CONFIG["torch_batch_size"]
    errs = []
    with torch.no_grad():
        for start in range(0, data.shape[0], batch_size):
            end = min(start + batch_size, data.shape[0])
            # Ensure writable backing memory before converting to torch tensor.
            batch_np = np.array(data[start:end], dtype=np.float32, copy=True)
            x = torch.from_numpy(batch_np).to(device)
            recon = model(x)
            if flatten:
                per_sample = torch.mean((recon - x) ** 2, dim=1)
            else:
                per_sample = torch.mean((recon - x) ** 2, dim=(1, 2))
            errs.append(per_sample.detach().cpu().numpy())
    return np.concatenate(errs, axis=0)


def evaluate(model_type: str) -> dict:
    """Evaluate AE model and persist diagnostics/threshold artifacts."""
    flatten = model_type == "dense"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_type=model_type, device=device)

    X_val = load_memmap_dataset(flatten_for_tree=flatten, split="healthy_val")
    # Validation errors define the anomaly threshold so deployment behavior
    # is anchored to healthy holdout statistics.
    val_errors = _reconstruction_errors(model, X_val, device=device, flatten=flatten)
    threshold = float(np.percentile(val_errors, CONFIG["ae_error_threshold_percentile"]))

    split_meta = load_split_metadata() or {}
    file_records = split_meta.get("file_records", [])
    all_memmap_enabled = bool(split_meta.get("all_memmap_enabled", True))
    scaler = joblib.load(CONFIG["scaler_file"])
    file_metrics = []
    if all_memmap_enabled:
        # Fast path when full memmap exists.
        X_all = load_memmap_dataset(flatten_for_tree=flatten, split="all")
        all_errors = _reconstruction_errors(model, X_all, device=device, flatten=flatten)
        for record in file_records:
            start = int(record.get("global_start_idx", 0))
            end = int(record.get("global_end_idx", 0))
            if end <= start or end > all_errors.shape[0]:
                continue
            file_errs = all_errors[start:end]
            file_metrics.append(
                {
                    "file_idx": int(record["file_idx"]),
                    "split": record.get("split", "unknown"),
                    "mean_reconstruction_error": float(np.mean(file_errs)),
                    "anomaly_rate": float(np.mean(file_errs >= threshold)),
                }
            )
    else:
        # Fallback path for environments where full "all" memmap is skipped.
        all_error_parts = []
        for record in file_records:
            signal = np.loadtxt(record["file_path"], dtype=np.float32).reshape(-1, 1)
            scaled = scaler.transform(signal)
            seqs = create_sequences(scaled, CONFIG["sequence_length"], CONFIG["stride"])
            if len(seqs) <= 0:
                continue
            model_in = seqs.reshape(len(seqs), -1) if flatten else seqs
            file_errs = _reconstruction_errors(model, model_in, device=device, flatten=flatten)
            all_error_parts.append(file_errs)
            file_metrics.append(
                {
                    "file_idx": int(record["file_idx"]),
                    "split": record.get("split", "unknown"),
                    "mean_reconstruction_error": float(np.mean(file_errs)),
                    "anomaly_rate": float(np.mean(file_errs >= threshold)),
                }
            )
        all_errors = np.concatenate(all_error_parts, axis=0) if all_error_parts else np.array([], dtype=np.float32)

    diagnostics_dir = os.path.join(CONFIG["processed_folder"], "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    prefix = f"{model_type}_autoencoder"
    np.save(os.path.join(diagnostics_dir, f"{prefix}_all_errors.npy"), all_errors)
    np.save(os.path.join(diagnostics_dir, f"{prefix}_val_errors.npy"), val_errors)
    with open(os.path.join(diagnostics_dir, f"{prefix}_threshold.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "threshold": threshold,
                "percentile": CONFIG["ae_error_threshold_percentile"],
                "model_type": model_type,
            },
            fh,
        )
    with open(os.path.join(diagnostics_dir, f"{prefix}_file_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(file_metrics, fh)

    # Persist plots so notebook and README workflows can reference static artifacts.
    hist_fig, hist_ax = plt.subplots(figsize=(10, 4))
    hist_ax.hist(all_errors, bins=80)
    hist_ax.axvline(threshold, linestyle="--", label=f"val p{CONFIG['ae_error_threshold_percentile']:.1f}")
    hist_ax.set_title(f"{model_type.upper()} AE Reconstruction Error Distribution")
    hist_ax.set_xlabel("Reconstruction Error (MSE)")
    hist_ax.set_ylabel("Count")
    hist_ax.legend()
    hist_fig.savefig(os.path.join(diagnostics_dir, f"{prefix}_error_distribution.png"), bbox_inches="tight")

    err_curve = [m["mean_reconstruction_error"] for m in file_metrics]
    rate_curve = [m["anomaly_rate"] for m in file_metrics]
    curve_fig, curve_ax = plt.subplots(figsize=(12, 5))
    curve_ax.plot(err_curve, marker="o", markersize=3)
    curve_ax.set_title(f"{model_type.upper()} AE Mean Error by File")
    curve_ax.set_xlabel("File Order (Time)")
    curve_ax.set_ylabel("Mean Reconstruction Error")
    curve_ax.grid(True)
    curve_fig.savefig(os.path.join(diagnostics_dir, f"{prefix}_mean_error_curve.png"), bbox_inches="tight")

    rate_fig, rate_ax = plt.subplots(figsize=(12, 5))
    rate_ax.plot(rate_curve, marker="o", markersize=3)
    rate_ax.set_title(f"{model_type.upper()} AE Per-file Anomaly Rate")
    rate_ax.set_xlabel("File Order (Time)")
    rate_ax.set_ylabel("Anomaly Rate")
    rate_ax.grid(True)
    rate_fig.savefig(os.path.join(diagnostics_dir, f"{prefix}_anomaly_rate_curve.png"), bbox_inches="tight")

    return {"threshold": threshold, "diagnostics_dir": diagnostics_dir}


def main():
    parser = argparse.ArgumentParser(description="Evaluate dense or LSTM autoencoder")
    parser.add_argument("--model-type", choices=["dense", "lstm"], required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    result = evaluate(model_type=args.model_type)
    logging.info(
        "[%s-ae] diagnostics=%s threshold=%.8f",
        args.model_type,
        result["diagnostics_dir"],
        result["threshold"],
    )


if __name__ == "__main__":
    main()
