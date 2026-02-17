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
import logging
import os
import uuid
from typing import List

import joblib
import numpy as np
import matplotlib.pyplot as plt

from .config import CONFIG, configure_logging
from .dataset import load_memmap_dataset
from .utils import plot_health_curve, list_ims_files


def machine_health_curve(limit: int | None = None, save_path: str | None = None):
    model_file = os.path.join(CONFIG["processed_folder"], "isolation_forest.model")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}. Run training first.")

    model = joblib.load(model_file)

    X_flat = load_memmap_dataset(flatten_for_tree=True)

    # Score
    scores = model.decision_function(X_flat)

    # Discover files in chronological order using the project helper
    files: List[str] = list_ims_files(CONFIG["data_folder"], seq_length=CONFIG["sequence_length"])
    if limit:
        files = files[:limit]

    file_mean_scores: List[float] = []
    seq_counter = 0
    seq_len = CONFIG["sequence_length"]

    for f in files[: CONFIG["num_files_to_process"]]:
        try:
            signal = np.loadtxt(f, dtype=np.float32).reshape(-1, 1)
        except Exception:
            logging.exception("Failed to read file %s", f)
            continue
        n_seqs = (len(signal) - seq_len) // CONFIG["stride"] + 1
        if n_seqs <= 0:
            continue
        file_scores = scores[seq_counter : seq_counter + n_seqs]
        seq_counter += n_seqs
        file_mean_scores.append(float(np.nanmean(file_scores)))

    fig = plot_health_curve(file_mean_scores)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logging.info("Saved Machine Health Curve to %s", save_path)
        return None
    # Caller (or notebook) may decide how to display the figure. Return it so
    # callers can either show or further manipulate the figure.
    logging.info("Machine Health Curve computed; returning figure object.")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Machine Health Curve from trained model and memmap dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to aggregate (demo)")
    parser.add_argument("--save", type=str, default=None, help="Save plot to given path instead of returning/displaying")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    try:
        fig = machine_health_curve(limit=args.limit, save_path=args.save)
        # If no save path was provided and we got a figure back, save it to processed/figures
        if args.save is None and fig is not None:
            try:
                figures_dir = os.path.join(CONFIG["processed_folder"], "figures")
                os.makedirs(figures_dir, exist_ok=True)
                fname = f"health_curve_{uuid.uuid4().hex}.png"
                out_path = os.path.join(figures_dir, fname)
                fig.savefig(out_path, bbox_inches="tight")
                logging.info("Saved Machine Health Curve to %s", out_path)
            except Exception:
                logging.exception("Failed to save Machine Health Curve interactively")
    except Exception as exc:
        logging.exception("Failed to compute Machine Health Curve: %s", exc)
        raise


if __name__ == "__main__":
    main()
