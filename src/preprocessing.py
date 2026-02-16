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

def fit_global_scaler(files):
    healthy_cutoff = CONFIG["healthy_files"]
    sample_files = files[:healthy_cutoff]

    if len(sample_files) == 0:
        raise ValueError(
            "No files were found. Check your data path."
        )

    all_samples = []

    for file_path in sample_files:
        try:
            signal = np.loadtxt(file_path, dtype=np.float32).reshape(-1, 1)
            all_samples.append(signal)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    if len(all_samples) == 0:
        raise ValueError(
            "Files were discovered but none could be loaded. Check file format."
        )

    all_samples = np.vstack(all_samples)

    scaler = StandardScaler()
    scaler.fit(all_samples)

    print("✅ Global scaler fitted")
    print("Samples used:", all_samples.shape)

    joblib.dump(scaler, os.path.join(CONFIG["processed_folder"], "global_scaler.save"))

    return scaler


def create_memmap_dataset(files, scaler):
    """Save sequences to memory-mapped dataset"""
    seq_length = CONFIG["sequence_length"]
    stride = CONFIG["stride"]
    # initial arithmetic estimate (fast)
    total_sequences = sum(max((len(np.loadtxt(f)) - seq_length)//stride + 1, 0) for f in files[:CONFIG["num_files_to_process"]])
    # verify exact count using the same sequence logic (safe). If mismatch, use exact count.
    exact_total = 0
    for f in files[:CONFIG["num_files_to_process"]]:
        signal = np.loadtxt(f)
        exact_total += len(create_sequences(signal.reshape(-1,1), seq_length, stride))
    if exact_total != total_sequences:
        print(f"ℹ️ Adjusting memmap allocation: estimated {total_sequences} -> exact {exact_total}")
        total_sequences = exact_total
    memmap_file = CONFIG["memmap_file"]
    dataset = np.memmap(memmap_file, dtype='float32', mode='w+', shape=(total_sequences, seq_length, 1))
    
    idx = 0
    for f in files[:CONFIG["num_files_to_process"]]:
        signal = np.loadtxt(f).reshape(-1,1)
        if len(signal) < seq_length:
            print(f"⚠️ Skipping {f} — too short for sequences")
            continue
        scaled = scaler.transform(signal)
        seqs = create_sequences(scaled, seq_length, stride)
        dataset[idx:idx+len(seqs)] = seqs
        idx += len(seqs)
    dataset.flush()
    # actual number of sequences written (may be <= total_sequences if some files were skipped)
    actual_sequences = idx

    # Write metadata for robust loading
    meta = {
        "num_sequences": int(actual_sequences),
        "sequence_length": int(seq_length),
        "dtype": str(dataset.dtype),
        "stride": int(stride),
    }
    write_memmap_metadata(memmap_file, meta)

    print(f"✅ Memmap dataset created: {memmap_file} | total sequences (allocated): {total_sequences} | written: {actual_sequences}")
    return memmap_file
