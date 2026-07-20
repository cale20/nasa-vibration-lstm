# --------------------------------------------------
# utils.py
# --------------------------------------------------
"""Helper functions for file management and plotting.

Replace heavy side-effect
operations (like `plt.show`) with returnable figures so calling scripts
can decide how to display or save the output.
"""
import logging
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def _has_min_rows(file_path, min_rows):
    """Return True if text file has at least `min_rows` non-empty lines."""
    rows = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows += 1
            if rows >= min_rows:
                return True
    return False


def list_ims_files(folder, seq_length=100):
    """Return IMS file paths that can produce at least one sequence.

    Files that cannot be read are skipped with a logged warning.
    """
    files = []
    for root, dirs, filenames in os.walk(folder):
        for f in filenames:
            # ends with .## where ## are digits
            if len(f) >= 3 and f[-3] == "." and f[-2:].isdigit():
                files.append(os.path.join(root, f))
    files.sort()

    valid_files = []
    for fpath in files:
        try:
            # Quick pass: avoid loading full numeric arrays just to ensure
            # enough timesteps exist for one sequence.
            if _has_min_rows(fpath, seq_length):
                valid_files.append(fpath)
        except Exception as exc:
            logging.warning("Unable to read %s: %s", fpath, exc)

    return valid_files

def annotate_chronological_splits(ax, x_max=None):
    """Shade healthy_train / healthy_val / test_mixed regions on a file-order axis.

    Boundaries come from CONFIG split sizes. Regions are policy labels for
    training and thresholding—not independent proof that every early file is
    physically healthy.
    """
    from .config import CONFIG

    train_end = int(CONFIG["healthy_train_files"])
    val_end = train_end + int(CONFIG["healthy_val_files"])
    if x_max is None:
        x_max = ax.get_xlim()[1]
    x_max = float(max(x_max, val_end))

    ax.axvspan(0, train_end, color="#2ca02c", alpha=0.08, zorder=0)
    ax.axvspan(train_end, val_end, color="#1f77b4", alpha=0.10, zorder=0)
    ax.axvspan(val_end, x_max, color="#ff7f0e", alpha=0.06, zorder=0)
    ax.axvline(train_end, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
    ax.axvline(val_end, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
    ymin, ymax = ax.get_ylim()
    y_text = ymin + 0.92 * (ymax - ymin)
    ax.text(train_end / 2.0, y_text, "healthy_train", ha="center", va="top", fontsize=8, color="#2ca02c")
    ax.text((train_end + val_end) / 2.0, y_text, "healthy_val", ha="center", va="top", fontsize=8, color="#1f77b4")
    ax.text((val_end + x_max) / 2.0, y_text, "test_mixed (monitor)", ha="center", va="top", fontsize=8, color="#d62728")
    return train_end, val_end


def plot_health_curve(scores, title="Machine Health Curve"):
    """Plot file-level mean anomaly scores"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(scores, marker="o", markersize=3)
    ax.set_title(title)
    ax.set_xlabel("File Order (Time)")
    ax.set_ylabel("Mean Anomaly Score")
    ax.grid(True)
    annotate_chronological_splits(ax, x_max=max(len(scores) - 1, 0))
    return fig


def write_memmap_metadata(memmap_path, meta):
    """Write JSON metadata next to a memmap file.

    `meta` should be a JSON-serializable dict.
    Metadata file path is memmap_path + '.meta.json'.
    """
    meta_path = f"{memmap_path}.meta.json"
    meta = dict(meta)
    meta.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)


def read_memmap_metadata(memmap_path):
    """Read metadata JSON for a memmap file. Returns dict or None if missing."""
    meta_path = f"{memmap_path}.meta.json"
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        logging.exception("Failed reading memmap metadata %s", meta_path)
        return None

