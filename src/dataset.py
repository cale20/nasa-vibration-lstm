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

import json
import logging
import os
from typing import Optional

import numpy as np

from .config import CONFIG
from .utils import read_memmap_metadata


def _memmap_path_for_split(split: str) -> str:
    """Resolve split name to configured memmap path."""
    if split == "all":
        return CONFIG["memmap_file"]
    if split == "healthy_train":
        return CONFIG["healthy_train_memmap_file"]
    if split == "healthy_val":
        return CONFIG["healthy_val_memmap_file"]
    raise ValueError(f"Unknown split: {split}")


def load_memmap_dataset(flatten_for_tree: bool = True, split: str = "all") -> np.ndarray:
    """Load the memmap dataset from disk.

    Args:
        flatten_for_tree: If True, return a 2D array (n_samples, features)
            suitable for tree-based models. If False, return the raw
            memmap shape (n_sequences, seq_length, 1).
        split: one of {"all", "healthy_train", "healthy_val"}.

    Raises:
        FileNotFoundError: If the memmap file is missing.
        ValueError: If metadata is inconsistent with file size.

    Returns:
        np.ndarray: The dataset, memory-mapped from disk.
    """
    memmap_path = _memmap_path_for_split(split)

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


def load_split_metadata() -> Optional[dict]:
    """Load split metadata generated during preprocessing, if present."""
    path = CONFIG["split_metadata_file"]
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class MemmapTorchDataset:
    """PyTorch dataset backed by project memmaps.

    This dataset is intentionally thin: split policy and sequence shapes
    are defined upstream in preprocessing/config so training scripts can
    stay focused on optimization logic.
    """

    def __init__(self, split: str, flatten: bool):
        try:
            import torch  # noqa: F401
            from torch.utils.data import Dataset  # noqa: F401
        except Exception as exc:
            raise ImportError("PyTorch is required for torch datasets. Install `torch`.") from exc
        self._split = split
        self._flatten = flatten
        self._data = load_memmap_dataset(flatten_for_tree=flatten, split=split)

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        import torch

        x = self._data[idx]
        # Memmap-backed slices are read-only; return a writable copy for PyTorch.
        return torch.from_numpy(np.array(x, dtype=np.float32, copy=True))


def make_torch_dataloaders(flatten: bool = False):
    """Create healthy-only train/validation dataloaders for AE training."""
    try:
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise ImportError("PyTorch is required for dataloaders. Install `torch`.") from exc

    train_ds = MemmapTorchDataset(split="healthy_train", flatten=flatten)
    val_ds = MemmapTorchDataset(split="healthy_val", flatten=flatten)
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["torch_batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["torch_batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    return train_loader, val_loader
