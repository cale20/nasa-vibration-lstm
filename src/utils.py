# --------------------------------------------------
# utils.py
# --------------------------------------------------
"""
Helper functions for file management and plotting.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def list_ims_files(folder, seq_length=100):
    """Return numeric IMS files that can produce at least one sequence"""
    files = []
    for root, dirs, filenames in os.walk(folder):
        for f in filenames:
            if f[-3] == "." and f[-2:].isdigit():
                files.append(os.path.join(root, f))
    files.sort()
    valid_files = [f for f in files if len(np.loadtxt(f)) >= seq_length]
    return valid_files

def plot_health_curve(scores, title="Machine Health Curve"):
    """Plot file-level mean anomaly scores"""
    plt.figure(figsize=(12,5))
    plt.plot(scores, marker='o', markersize=3)
    plt.title(title)
    plt.xlabel("File Order (Time)")
    plt.ylabel("Mean Anomaly Score")
    plt.grid(True)
    plt.show()


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
        return None

