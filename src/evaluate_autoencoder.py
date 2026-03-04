"""Evaluate dense/LSTM autoencoders with reconstruction-error diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import joblib

from .config import CONFIG, configure_logging
from .dataset import load_memmap_dataset, load_split_metadata
from .logging_utils import fmt_seconds, log_note, log_progress
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


def _reconstruction_errors(
    model,
    data: np.ndarray,
    device: torch.device,
    flatten: bool,
    progress_label: str | None = None,
    log_interval_batches: int | None = None,
) -> np.ndarray:
    """Compute per-sample reconstruction MSE in batches."""
    batch_size = CONFIG["torch_batch_size"]
    errs = []
    total_batches = (data.shape[0] + batch_size - 1) // batch_size
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, data.shape[0], batch_size), start=1):
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
            if progress_label and log_interval_batches and batch_idx % log_interval_batches == 0:
                elapsed = time.perf_counter() - start_time
                eta_sec = (elapsed / batch_idx) * max(total_batches - batch_idx, 0)
                log_progress(
                    f"{progress_label}: batch {batch_idx}/{total_batches} | "
                    f"elapsed={fmt_seconds(elapsed)} | eta={fmt_seconds(eta_sec)}"
                )
    return np.concatenate(errs, axis=0)


def evaluate(model_type: str, log_interval_files: int | None = None) -> dict:
    """Evaluate AE model and persist diagnostics/threshold artifacts."""
    flatten = model_type == "dense"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_type=model_type, device=device)
    log_interval_batches = CONFIG.get("log_interval_batches", 100)

    X_val = load_memmap_dataset(flatten_for_tree=flatten, split="healthy_val")
    # Validation errors define the anomaly threshold so deployment behavior
    # is anchored to healthy holdout statistics.
    val_errors = _reconstruction_errors(
        model,
        X_val,
        device=device,
        flatten=flatten,
        progress_label=f"{model_type.upper()} AE val reconstruction",
        log_interval_batches=log_interval_batches,
    )
    threshold = float(np.percentile(val_errors, CONFIG["ae_error_threshold_percentile"]))

    split_meta = load_split_metadata() or {}
    file_records = split_meta.get("file_records", [])
    all_memmap_enabled = bool(split_meta.get("all_memmap_enabled", True))
    log_note(f"{model_type.upper()} AE eval context: files={len(file_records)}, all_memmap={all_memmap_enabled}")
    eval_start = time.perf_counter()
    scaler = joblib.load(CONFIG["scaler_file"])
    file_metrics = []
    if all_memmap_enabled:
        # Fast path when full memmap exists.
        X_all = load_memmap_dataset(flatten_for_tree=flatten, split="all")
        all_errors = _reconstruction_errors(
            model,
            X_all,
            device=device,
            flatten=flatten,
            progress_label=f"{model_type.upper()} AE all reconstruction",
            log_interval_batches=log_interval_batches,
        )
        for file_pos, record in enumerate(file_records):
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
            if log_interval_files and (file_pos + 1) % log_interval_files == 0:
                elapsed = time.perf_counter() - eval_start
                eta_sec = (elapsed / (file_pos + 1)) * max(len(file_records) - (file_pos + 1), 0)
                log_progress(
                    f"{model_type.upper()} AE aggregate (all memmap): file {file_pos + 1}/{len(file_records)} | "
                    f"elapsed={fmt_seconds(elapsed)} | eta={fmt_seconds(eta_sec)}"
                )
    else:
        # Fallback path for environments where full "all" memmap is skipped.
        all_error_parts = []
        for file_pos, record in enumerate(file_records):
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
            if log_interval_files and (file_pos + 1) % log_interval_files == 0:
                elapsed = time.perf_counter() - eval_start
                eta_sec = (elapsed / (file_pos + 1)) * max(len(file_records) - (file_pos + 1), 0)
                log_progress(
                    f"{model_type.upper()} AE eval (stream): file {file_pos + 1}/{len(file_records)} | "
                    f"elapsed={fmt_seconds(elapsed)} | eta={fmt_seconds(eta_sec)}"
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
                "threshold_source_split": "healthy_val",
                "file_alert_threshold": 0.0,
                "file_score_name": "anomaly_rate",
            },
            fh,
        )
    with open(os.path.join(diagnostics_dir, f"{prefix}_file_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(file_metrics, fh)
    file_scores_arr = np.asarray([row["anomaly_rate"] for row in file_metrics], dtype=np.float32)
    file_alerts_arr = (file_scores_arr > 0.0).astype(np.uint8)
    np.save(os.path.join(diagnostics_dir, f"{prefix}_file_scores.npy"), file_scores_arr)
    np.save(os.path.join(diagnostics_dir, f"{prefix}_file_alerts.npy"), file_alerts_arr)

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
    parser.add_argument("--log-interval-files", type=int, default=CONFIG["log_interval_files"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    result = evaluate(model_type=args.model_type, log_interval_files=args.log_interval_files)
    logging.info(
        "[%s-ae] diagnostics=%s threshold=%.8f",
        args.model_type,
        result["diagnostics_dir"],
        result["threshold"],
    )


if __name__ == "__main__":
    main()
