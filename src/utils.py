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
            size = len(np.loadtxt(fpath))
        except Exception as exc:
            logging.warning("Unable to read %s: %s", fpath, exc)
            continue
        if size >= seq_length:
            valid_files.append(fpath)

    return valid_files

def plot_health_curve(scores, title="Machine Health Curve"):
    """Plot file-level mean anomaly scores"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(scores, marker="o", markersize=3)
    ax.set_title(title)
    ax.set_xlabel("File Order (Time)")
    ax.set_ylabel("Mean Anomaly Score")
    ax.grid(True)
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

