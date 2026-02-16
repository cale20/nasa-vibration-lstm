# --------------------------------------------------
# dataset.py
# --------------------------------------------------
"""
Load memmap dataset for training or evaluation
"""
import os
import numpy as np
from .config import CONFIG
from .utils import read_memmap_metadata

def load_memmap_dataset(flatten_for_tree=True):
    memmap_path = CONFIG["memmap_file"]

    meta = read_memmap_metadata(memmap_path)
    if meta is not None:
        num_sequences = int(meta.get("num_sequences"))
        seq_length = int(meta.get("sequence_length"))
        dtype = meta.get("dtype", "float32")
        dataset = np.memmap(memmap_path, dtype=dtype, mode='r', shape=(num_sequences, seq_length, 1))
    else:
        # Fallback: compute from file size (assumes contiguous float32 values)
        filesize = os.path.getsize(memmap_path)
        itemsize = np.dtype('float32').itemsize
        total_values = filesize // itemsize
        seq_length = CONFIG["sequence_length"]
        num_sequences = total_values // (seq_length * 1)
        dataset = np.memmap(memmap_path, dtype='float32', mode='r', shape=(num_sequences, seq_length, 1))

    if flatten_for_tree:
        return dataset.reshape(dataset.shape[0], -1)
    return dataset
