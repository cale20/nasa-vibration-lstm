# --------------------------------------------------
# dataset.py
# --------------------------------------------------
"""Load the memmap dataset used by training and evaluation.

This module exposes `load_memmap_dataset` which reads the on-disk
np.memmap produced by the preprocessing step. The function validates
the presence of the memmap and its metadata and returns a NumPy
array view (optionally flattened for tree-based models).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from .config import CONFIG
from .utils import read_memmap_metadata


def load_memmap_dataset(flatten_for_tree: bool = True) -> np.ndarray:
    """Load the memmap dataset from disk.

    Args:
        flatten_for_tree: If True, return a 2D array (n_samples, features)
            suitable for tree-based models. If False, return the raw
            memmap shape (n_sequences, seq_length, 1).

    Raises:
        FileNotFoundError: If the memmap file is missing.
        ValueError: If metadata is inconsistent with file size.

    Returns:
        np.ndarray: The dataset, memory-mapped from disk.
    """
    memmap_path = CONFIG["memmap_file"]

    if not os.path.exists(memmap_path):
        raise FileNotFoundError(
            f"Memmap file not found: {memmap_path}. Run preprocessing to create it."
        )

    meta = read_memmap_metadata(memmap_path)
    if meta is not None:
        try:
            num_sequences = int(meta.get("num_sequences"))
            seq_length = int(meta.get("sequence_length"))
            dtype = meta.get("dtype", "float32")
        except Exception as exc:
            logging.exception("Invalid memmap metadata for %s", memmap_path)
            raise ValueError("Corrupt or invalid memmap metadata") from exc

        dataset = np.memmap(memmap_path, dtype=dtype, mode="r", shape=(num_sequences, seq_length, 1))
    else:
        # Fallback: compute from file size (assumes contiguous float32 values)
        filesize = os.path.getsize(memmap_path)
        itemsize = np.dtype("float32").itemsize
        total_values = filesize // itemsize
        seq_length = CONFIG["sequence_length"]
        num_sequences = total_values // (seq_length * 1)
        if num_sequences <= 0:
            raise ValueError("Memmap file exists but contains no sequences")
        dataset = np.memmap(memmap_path, dtype="float32", mode="r", shape=(num_sequences, seq_length, 1))

    if flatten_for_tree:
        return dataset.reshape(dataset.shape[0], -1)
    return dataset
