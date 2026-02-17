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


def configure_logging(level=None):
    """Configure basic logging for CLI/entry scripts."""
    import logging

    if level is None:
        level = logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def ensure_output_dirs():
    """Create expected output directories (safe to call repeatedly)."""
    os.makedirs(CONFIG["processed_folder"], exist_ok=True)
    memmap_dir = os.path.dirname(CONFIG["memmap_file"])
    if memmap_dir:
        os.makedirs(memmap_dir, exist_ok=True)
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
