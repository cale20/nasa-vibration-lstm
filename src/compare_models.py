"""Compare Isolation Forest, Dense AE, and LSTM AE file-level trends."""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from .config import CONFIG, configure_logging
from .utils import annotate_chronological_splits


def _safe_read_json(path: str):
    """Return parsed JSON when available, else None."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize(values):
    """Min-max normalize for shape-only trend comparison across models."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v - min_v < 1e-12:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def _plot_if_absolute_rate(if_metrics, diagnostics_dir: str) -> str | None:
    """Rebuild Isolation Forest absolute anomaly-rate curve with split markers."""
    if not if_metrics:
        return None
    vals = [row["anomaly_rate"] for row in if_metrics]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(vals, marker="o", markersize=3, label="Isolation Forest", zorder=3)
    ax.set_title("Per-file Anomaly Rate (Isolation Forest)")
    ax.set_xlabel("File Order (Time)")
    ax.set_ylabel("Anomaly Rate")
    ax.grid(True)
    annotate_chronological_splits(ax, x_max=max(len(vals) - 1, 0))
    ax.legend(loc="upper right")
    out_path = os.path.join(diagnostics_dir, "isolation_forest_anomaly_rate_curve.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run() -> dict[str, str]:
    """Build normalized anomaly-rate comparison figure across available models."""
    diagnostics_dir = os.path.join(CONFIG["processed_folder"], "diagnostics")
    if_metrics = _safe_read_json(os.path.join(diagnostics_dir, "isolation_forest_file_metrics.json"))
    dense_metrics = _safe_read_json(os.path.join(diagnostics_dir, "dense_autoencoder_file_metrics.json"))
    lstm_metrics = _safe_read_json(os.path.join(diagnostics_dir, "lstm_autoencoder_file_metrics.json"))

    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = False
    n_files = 0
    if if_metrics:
        vals = [row["anomaly_rate"] for row in if_metrics]
        n_files = max(n_files, len(vals))
        ax.plot(_normalize(vals), label="Isolation Forest", zorder=3)
        plotted = True
    if dense_metrics:
        vals = [row["anomaly_rate"] for row in dense_metrics]
        n_files = max(n_files, len(vals))
        ax.plot(_normalize(vals), label="Dense AE", zorder=3)
        plotted = True
    if lstm_metrics:
        vals = [row["anomaly_rate"] for row in lstm_metrics]
        n_files = max(n_files, len(vals))
        ax.plot(_normalize(vals), label="LSTM AE", zorder=3)
        plotted = True

    if not plotted:
        raise FileNotFoundError(
            "No diagnostics found. Run baseline and autoencoder evaluations first."
        )

    ax.set_title("Normalized Per-file Anomaly Trend Comparison")
    ax.set_xlabel("File Order (Time)")
    ax.set_ylabel("Normalized Anomaly Rate")
    ax.grid(True)
    annotate_chronological_splits(ax, x_max=max(n_files - 1, 0))
    ax.legend(loc="upper right")
    out_path = os.path.join(diagnostics_dir, "model_comparison_anomaly_rate.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    if_path = _plot_if_absolute_rate(if_metrics, diagnostics_dir)
    outputs = {"comparison": out_path}
    if if_path:
        outputs["isolation_forest_rate"] = if_path
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Compare model anomaly trends")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    outputs = run()
    logging.info("Saved model comparison figure to %s", outputs["comparison"])
    if "isolation_forest_rate" in outputs:
        logging.info("Saved Isolation Forest rate figure to %s", outputs["isolation_forest_rate"])


if __name__ == "__main__":
    main()
