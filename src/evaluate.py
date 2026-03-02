# --------------------------------------------------
# evaluate.py
# --------------------------------------------------
"""Compute anomaly scores and produce the Machine Health Curve.

This script loads a trained IsolationForest and the memmap dataset,
computes per-sequence anomaly scores, aggregates them per file, and
either returns or saves a plot for downstream consumption.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from typing import List

import joblib
import numpy as np
import matplotlib.pyplot as plt

from .config import CONFIG, configure_logging
from .dataset import load_memmap_dataset, load_split_metadata
from .preprocessing import create_sequences
from .utils import plot_health_curve, list_ims_files


def machine_health_curve(limit: int | None = None, save_path: str | None = None):
    """Score chronological files and generate baseline diagnostics.

    The function supports two data access modes:
    1) Fast path with prebuilt full memmap ("all")
    2) Streaming fallback from raw files when full memmap is unavailable
       due to platform/storage constraints.
    """
    model_file = os.path.join(CONFIG["processed_folder"], "isolation_forest.model")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}. Run training first.")

    model = joblib.load(model_file)

    scaler = joblib.load(CONFIG["scaler_file"])
    # Threshold must come from healthy holdout data, not mixed/test data.
    X_val = load_memmap_dataset(flatten_for_tree=True, split="healthy_val")
    if X_val.shape[0] == 0:
        raise ValueError("healthy_val split contains no sequences; cannot compute threshold")
    val_scores = model.decision_function(X_val)
    threshold = float(np.percentile(val_scores, CONFIG["score_threshold_percentile"]))

    split_meta = load_split_metadata()
    file_mean_scores: List[float] = []
    file_anomaly_rates: List[float] = []
    seq_len = CONFIG["sequence_length"]
    file_records = []
    all_scores_parts = []

    if split_meta and split_meta.get("file_records"):
        file_records = split_meta["file_records"]
        if limit:
            file_records = file_records[:limit]
    else:
        files: List[str] = list_ims_files(CONFIG["data_folder"], seq_length=seq_len)
        if limit:
            files = files[:limit]
        for file_idx, f in enumerate(files[: CONFIG["num_files_to_process"]]):
            try:
                signal = np.loadtxt(f, dtype=np.float32).reshape(-1, 1)
            except Exception:
                logging.exception("Failed to read file %s", f)
                continue
            n_seqs = (len(signal) - seq_len) // CONFIG["stride"] + 1
            if n_seqs <= 0:
                continue
            file_records.append(
                {"file_idx": file_idx, "file_path": f, "num_sequences": int(n_seqs)}
            )

    use_all_memmap = bool((split_meta or {}).get("all_memmap_enabled", True))
    if use_all_memmap:
        X_flat = load_memmap_dataset(flatten_for_tree=True, split="all")
        scores = model.decision_function(X_flat)
        seq_counter = 0
        for record in file_records:
            n_seqs = int(record["num_sequences"])
            if n_seqs <= 0:
                continue
            file_scores = scores[seq_counter : seq_counter + n_seqs]
            seq_counter += n_seqs
            all_scores_parts.append(file_scores)
            file_mean_scores.append(float(np.nanmean(file_scores)))
            file_anomaly_rates.append(float(np.mean(file_scores <= threshold)))
    else:
        # First pass: collect scores to derive threshold.
        for record in file_records:
            signal = np.loadtxt(record["file_path"], dtype=np.float32).reshape(-1, 1)
            scaled = scaler.transform(signal)
            seqs = create_sequences(scaled, seq_len, CONFIG["stride"])
            if len(seqs) <= 0:
                continue
            file_scores = model.decision_function(seqs.reshape(len(seqs), -1))
            all_scores_parts.append(file_scores)
        scores = np.concatenate(all_scores_parts, axis=0) if all_scores_parts else np.array([], dtype=np.float32)

        # Second pass: per-file metrics. This keeps thresholding consistent
        # with the global score distribution built in pass one.
        for record in file_records:
            signal = np.loadtxt(record["file_path"], dtype=np.float32).reshape(-1, 1)
            scaled = scaler.transform(signal)
            seqs = create_sequences(scaled, seq_len, CONFIG["stride"])
            if len(seqs) <= 0:
                continue
            file_scores = model.decision_function(seqs.reshape(len(seqs), -1))
            file_mean_scores.append(float(np.nanmean(file_scores)))
            file_anomaly_rates.append(float(np.mean(file_scores <= threshold)))

    diagnostics_dir = os.path.join(CONFIG["processed_folder"], "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    np.save(os.path.join(diagnostics_dir, "isolation_forest_scores.npy"), scores)
    file_scores_arr = np.asarray(file_anomaly_rates, dtype=np.float32)
    file_alerts_arr = (file_scores_arr > 0.0).astype(np.uint8)
    np.save(os.path.join(diagnostics_dir, "isolation_forest_file_scores.npy"), file_scores_arr)
    np.save(os.path.join(diagnostics_dir, "isolation_forest_file_alerts.npy"), file_alerts_arr)
    with open(
        os.path.join(diagnostics_dir, "isolation_forest_threshold.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(
            {
                "threshold": threshold,
                "percentile": CONFIG["score_threshold_percentile"],
                "num_scores": int(scores.shape[0]),
                "num_val_scores": int(val_scores.shape[0]),
                "threshold_source_split": "healthy_val",
                "file_alert_threshold": 0.0,
                "file_score_name": "anomaly_rate",
            },
            fh,
        )

    hist_fig, hist_ax = plt.subplots(figsize=(10, 4))
    hist_ax.hist(scores, bins=80)
    hist_ax.axvline(threshold, linestyle="--", label=f"p{CONFIG['score_threshold_percentile']:.1f}")
    hist_ax.set_title("Isolation Forest Score Distribution")
    hist_ax.set_xlabel("Decision Function Score")
    hist_ax.set_ylabel("Count")
    hist_ax.legend()
    hist_fig.savefig(
        os.path.join(diagnostics_dir, "isolation_forest_score_distribution.png"),
        bbox_inches="tight",
    )

    fig = plot_health_curve(file_mean_scores, title="Machine Health Curve (Mean IF Score)")
    rate_fig, rate_ax = plt.subplots(figsize=(12, 5))
    rate_ax.plot(file_anomaly_rates, marker="o", markersize=3)
    rate_ax.set_title("Per-file Anomaly Rate")
    rate_ax.set_xlabel("File Order (Time)")
    rate_ax.set_ylabel("Anomaly Rate")
    rate_ax.grid(True)
    rate_fig.savefig(
        os.path.join(diagnostics_dir, "isolation_forest_anomaly_rate_curve.png"),
        bbox_inches="tight",
    )

    # File-level JSON is intentionally lightweight for easy downstream parsing
    metrics = []
    for idx, value in enumerate(file_mean_scores):
        metrics.append(
            {
                "file_order_idx": idx,
                "mean_score": value,
                "anomaly_rate": file_anomaly_rates[idx],
            }
        )
    with open(
        os.path.join(diagnostics_dir, "isolation_forest_file_metrics.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(metrics, fh)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logging.info("Saved Machine Health Curve to %s", save_path)
        return {"threshold": threshold, "diagnostics_dir": diagnostics_dir}
    # Caller (or notebook) may decide how to display the figure. Return it so
    # callers can either show or further manipulate the figure.
    logging.info("Machine Health Curve computed; returning figure object.")
    return {
        "figure": fig,
        "threshold": threshold,
        "diagnostics_dir": diagnostics_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Machine Health Curve from trained model and memmap dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to aggregate (demo)")
    parser.add_argument("--save", type=str, default=None, help="Save plot to given path instead of returning/displaying")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    try:
        output = machine_health_curve(limit=args.limit, save_path=args.save)
        # If no save path was provided and we got a figure back, save it to processed/figures
        if args.save is None and output is not None and output.get("figure") is not None:
            try:
                figures_dir = os.path.join(CONFIG["processed_folder"], "figures")
                os.makedirs(figures_dir, exist_ok=True)
                fname = f"health_curve_{uuid.uuid4().hex}.png"
                out_path = os.path.join(figures_dir, fname)
                output["figure"].savefig(out_path, bbox_inches="tight")
                logging.info("Saved Machine Health Curve to %s", out_path)
                logging.info(
                    "Diagnostics saved under %s | threshold=%.6f",
                    output["diagnostics_dir"],
                    output["threshold"],
                )
            except Exception:
                logging.exception("Failed to save Machine Health Curve interactively")
    except Exception as exc:
        logging.exception("Failed to compute Machine Health Curve: %s", exc)
        raise


if __name__ == "__main__":
    main()
