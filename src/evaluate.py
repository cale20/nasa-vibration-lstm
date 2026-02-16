# --------------------------------------------------
# evaluate.py
# --------------------------------------------------
"""
Compute anomaly scores and plot Machine Health Curve
"""
import numpy as np
import os
import joblib
from .config import CONFIG
from .utils import plot_health_curve
from .dataset import load_memmap_dataset

def machine_health_curve():
    model_file = os.path.join(CONFIG["processed_folder"], "isolation_forest.model")
    model = joblib.load(model_file)

    X_flat = load_memmap_dataset(flatten_for_tree=True)

    # Score
    scores = model.decision_function(X_flat)

    # Aggregate per file
    files = sorted([os.path.join(CONFIG["data_folder"], f) for f in os.listdir(CONFIG["data_folder"])])
    file_mean_scores = []
    seq_counter = 0
    seq_len = CONFIG["sequence_length"]

    for f in files[:CONFIG["num_files_to_process"]]:
        signal = np.loadtxt(f, dtype=np.float32).reshape(-1,1)
        n_seqs = (len(signal) - seq_len)//CONFIG["stride"] + 1
        if n_seqs <= 0:
            continue
        file_scores = scores[seq_counter:seq_counter+n_seqs]
        seq_counter += n_seqs
        file_mean_scores.append(np.nanmean(file_scores))

    plot_health_curve(file_mean_scores)

if __name__ == "__main__":
    machine_health_curve()
