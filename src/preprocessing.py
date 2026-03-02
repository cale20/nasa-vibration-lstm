# --------------------------------------------------
# preprocessing.py
# --------------------------------------------------
"""
Preprocessing pipeline:
- Sequence creation
- Global scaler fitting
- Memmap dataset creation
"""
import os
import json
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from .config import CONFIG
from .utils import list_ims_files, write_memmap_metadata

def create_sequences(signal, seq_length, stride=5):
    """Convert 1D vibration signal into overlapping sequences"""
    sequences = []
    # include the final window so counts match arithmetic formula
    for i in range(0, len(signal) - seq_length + 1, stride):
        sequences.append(signal[i:i+seq_length])
    return np.array(sequences, dtype=np.float32)


def count_sequences(signal_length, seq_length, stride):
    """Return number of sliding windows without materializing sequences."""
    if signal_length < seq_length:
        return 0
    return ((signal_length - seq_length) // stride) + 1

def fit_global_scaler(files):
    """Fit a single scaler on early-life healthy files only.

    Using one global scaler preserves absolute amplitude shifts that
    anomaly models rely on during later-life scoring.
    """
    healthy_cutoff = CONFIG["healthy_files"]
    sample_files = files[:healthy_cutoff]

    if len(sample_files) == 0:
        raise ValueError(
            "No files were found. Check your data path."
        )

    scaler = StandardScaler()
    total_rows = 0
    fit_files = 0

    for file_path in sample_files:
        try:
            signal = np.loadtxt(file_path, dtype=np.float32).reshape(-1, 1)
            scaler.partial_fit(signal)
            total_rows += int(signal.shape[0])
            fit_files += 1
        except Exception as e:
            logging.warning("Skipping %s: %s", file_path, e)

    if fit_files == 0:
        raise ValueError(
            "Files were discovered but none could be loaded. Check file format."
        )

    logging.info("Global scaler fitted")
    logging.info("Files used: %s | Samples used: %s", fit_files, total_rows)

    os.makedirs(CONFIG["processed_folder"], exist_ok=True)
    joblib.dump(scaler, os.path.join(CONFIG["processed_folder"], "global_scaler.save"))

    return scaler


def _split_name_for_file_idx(file_idx):
    """Map chronological file index to split label."""
    healthy_train = CONFIG["healthy_train_files"]
    healthy_val = CONFIG["healthy_val_files"]
    healthy_total = healthy_train + healthy_val
    if file_idx < healthy_train:
        return "healthy_train"
    if file_idx < healthy_total:
        return "healthy_val"
    return "test_mixed"


def _validate_split_config():
    """Fail fast if configured healthy splits are inconsistent."""
    healthy_files = CONFIG["healthy_files"]
    healthy_total = CONFIG["healthy_train_files"] + CONFIG["healthy_val_files"]
    if healthy_total > healthy_files:
        raise ValueError(
            "healthy_train_files + healthy_val_files must be <= healthy_files"
        )


def _save_split_metadata(file_records, split_counts):
    """Persist split metadata used by downstream evaluation and reporting."""
    create_all = bool(CONFIG.get("create_all_memmap", True))
    all_bytes = (
        int(split_counts["all"])
        * int(CONFIG["sequence_length"])
        * np.dtype("float32").itemsize
    )
    all_memmap_enabled = create_all and all_bytes <= int(CONFIG.get("max_all_memmap_bytes", 3_500_000_000))
    payload = {
        "healthy_train_files": int(CONFIG["healthy_train_files"]),
        "healthy_val_files": int(CONFIG["healthy_val_files"]),
        "healthy_files": int(CONFIG["healthy_files"]),
        "num_files_to_process": int(CONFIG["num_files_to_process"]),
        "file_records": file_records,
        "split_sequence_counts": split_counts,
        "all_memmap_enabled": bool(all_memmap_enabled),
        "estimated_all_memmap_bytes": int(all_bytes),
    }
    with open(CONFIG["split_metadata_file"], "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def create_memmap_dataset(files, scaler):
    """Save split-aware sequences to memory-mapped datasets.

    Returns:
        dict: Paths to generated memmaps and metadata.
    """
    _validate_split_config()
    seq_length = CONFIG["sequence_length"]
    stride = CONFIG["stride"]
    files_to_process = files[: CONFIG["num_files_to_process"]]

    file_records = []
    split_counts = {"healthy_train": 0, "healthy_val": 0, "test_mixed": 0, "all": 0}
    # First pass computes exact per-file sequence counts before any
    # allocation, which keeps memmap shapes deterministic.
    for file_idx, fpath in enumerate(files_to_process):
        signal = np.loadtxt(fpath, dtype=np.float32).reshape(-1, 1)
        n_seqs = count_sequences(len(signal), seq_length, stride)
        if n_seqs <= 0:
            continue
        split_name = _split_name_for_file_idx(file_idx)
        file_records.append(
            {
                "file_idx": int(file_idx),
                "file_path": fpath,
                "num_sequences": int(n_seqs),
                "split": split_name,
            }
        )
        split_counts[split_name] += n_seqs
        split_counts["all"] += n_seqs

    # Healthy train/val memmaps are always materialized because training
    # depends on them. The full "all" memmap is optional on constrained
    # platforms (e.g. Windows) and may fall back to streaming evaluation.
    memmaps = {
        "healthy_train": np.memmap(
            CONFIG["healthy_train_memmap_file"],
            dtype="float32",
            mode="w+",
            shape=(split_counts["healthy_train"], seq_length, 1),
        ),
        "healthy_val": np.memmap(
            CONFIG["healthy_val_memmap_file"],
            dtype="float32",
            mode="w+",
            shape=(split_counts["healthy_val"], seq_length, 1),
        ),
    }
    all_bytes = int(split_counts["all"]) * int(seq_length) * np.dtype("float32").itemsize
    allow_all_memmap = bool(CONFIG.get("create_all_memmap", True)) and all_bytes <= int(
        CONFIG.get("max_all_memmap_bytes", 3_500_000_000)
    )
    if allow_all_memmap:
        memmaps["all"] = np.memmap(
            CONFIG["memmap_file"],
            dtype="float32",
            mode="w+",
            shape=(split_counts["all"], seq_length, 1),
        )
    else:
        logging.warning(
            "Skipping full 'all' memmap allocation (%s bytes). "
            "Evaluations will stream from raw files using split metadata.",
            all_bytes,
        )

    write_index = {"healthy_train": 0, "healthy_val": 0}
    if allow_all_memmap:
        write_index["all"] = 0
    global_cursor = 0

    # Second pass writes scaled sequences and records global/split indices
    # so evaluators can aggregate per-file metrics deterministically.
    for record in file_records:
        fpath = record["file_path"]
        signal = np.loadtxt(fpath, dtype=np.float32).reshape(-1, 1)
        scaled = scaler.transform(signal)
        seqs = create_sequences(scaled, seq_length, stride)
        n_seqs = len(seqs)
        if n_seqs <= 0:
            continue

        start = global_cursor
        end = start + n_seqs
        record["global_start_idx"] = int(start)
        record["global_end_idx"] = int(end)
        global_cursor = end
        if allow_all_memmap:
            memmaps["all"][start:end] = seqs
            write_index["all"] = end

        split_name = record["split"]
        if split_name in ("healthy_train", "healthy_val"):
            local_start = write_index[split_name]
            local_end = local_start + n_seqs
            memmaps[split_name][local_start:local_end] = seqs
            write_index[split_name] = local_end
            record["split_start_idx"] = int(local_start)
            record["split_end_idx"] = int(local_end)

    # Flush and persist metadata once writes are complete.
    for split_name, mmap_obj in memmaps.items():
        mmap_obj.flush()
        path_key = "memmap_file" if split_name == "all" else f"{split_name}_memmap_file"
        write_memmap_metadata(
            CONFIG[path_key],
            {
                "num_sequences": int(write_index[split_name]),
                "sequence_length": int(seq_length),
                "dtype": str(mmap_obj.dtype),
                "stride": int(stride),
                "split_name": split_name,
            },
        )

    _save_split_metadata(file_records, split_counts)
    logging.info(
        "Memmaps created | all=%s (%s) healthy_train=%s healthy_val=%s",
        split_counts["all"] if allow_all_memmap else 0,
        "enabled" if allow_all_memmap else "streaming-fallback",
        split_counts["healthy_train"],
        split_counts["healthy_val"],
    )
    return {
        "all": CONFIG["memmap_file"] if allow_all_memmap else None,
        "healthy_train": CONFIG["healthy_train_memmap_file"],
        "healthy_val": CONFIG["healthy_val_memmap_file"],
        "split_metadata": CONFIG["split_metadata_file"],
    }
