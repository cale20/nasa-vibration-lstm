# --------------------------------------------------
# config.py
# --------------------------------------------------
"""
Global configuration for anomaly detection project.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    # Sequence & sliding window
    "sequence_length": 100,
    "stride": 5,
    
    # Preprocessing
    "batch_size": 50,
    "healthy_files": 100,
    "num_files_to_process": 400,
    
    # Data paths
    "data_folder": os.path.join(BASE_DIR, "data/raw/IMS/1st_test"),
    "processed_folder": os.path.join(BASE_DIR, "data/processed"),
    "memmap_file": os.path.join(BASE_DIR, "data/processed/all_sequences.dat"),
    "scaler_file": os.path.join(BASE_DIR, "data/processed/global_scaler.save"),
    
    # Isolation Forest
    "max_train_samples": 50000,
    "contamination": 0.01,
    "n_estimators": 50
}
