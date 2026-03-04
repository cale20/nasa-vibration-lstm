# --------------------------------------------------
# config.py
# --------------------------------------------------
"""
Global configuration for anomaly detection project.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    # Sequence/window parameters used by all models
    "sequence_length": 100,
    "stride": 5,
    
    # Preprocessing and split policy
    "batch_size": 50,
    "healthy_files": 100,
    "num_files_to_process": 400,
    "healthy_train_files": 80,
    "healthy_val_files": 20,
    
    # Data paths
    "data_folder": os.path.join(BASE_DIR, "data/raw/IMS/1st_test"),
    "processed_folder": os.path.join(BASE_DIR, "data/processed"),
    "memmap_file": os.path.join(BASE_DIR, "data/processed/all_sequences.dat"),
    "healthy_train_memmap_file": os.path.join(BASE_DIR, "data/processed/healthy_train_sequences.dat"),
    "healthy_val_memmap_file": os.path.join(BASE_DIR, "data/processed/healthy_val_sequences.dat"),
    "split_metadata_file": os.path.join(BASE_DIR, "data/processed/split_metadata.json"),
    "scaler_file": os.path.join(BASE_DIR, "data/processed/global_scaler.save"),
    "create_all_memmap": True,
    "max_all_memmap_bytes": 3_500_000_000,
    
    # Isolation Forest baseline parameters
    "max_train_samples": 50000,
    "contamination": 0.01,
    "n_estimators": 50,
    "score_threshold_percentile": 1.0,
    "ae_error_threshold_percentile": 99.0,
    "random_seed": 42,

    # Autoencoder training parameters
    "torch_batch_size": 512,
    "num_workers": 0,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "epochs": 20,
    "early_stopping_patience": 5,
    "dense_latent_dim": 32,
    "lstm_hidden_size": 64,
    "lstm_num_layers": 1,
    "lstm_dropout": 0.0,

    # Unsupervised evaluation policy
    "healthy_reference_split": "healthy_val",
    "late_life_split": "test_mixed",
    "unsup_alert_percentile": 99.0,
    "persistence_k": 3,
    "persistence_m": 5,
    "max_false_alarm_rate_healthy": 0.05,
    "trend_min_spearman": 0.5,
    "log_interval_batches": 100,
    "log_interval_files": 10,

    # Model artifact paths
    "dense_autoencoder_model_file": os.path.join(BASE_DIR, "models/dense_autoencoder.pt"),
    "lstm_autoencoder_model_file": os.path.join(BASE_DIR, "models/lstm_autoencoder.pt"),
}


def configure_logging(level=None):
    """Configure logging for CLI/entry scripts."""
    import logging

    if level is None:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def ensure_output_dirs():
    """Create expected output directories (safe to call repeatedly)."""
    os.makedirs(CONFIG["processed_folder"], exist_ok=True)
    memmap_dir = os.path.dirname(CONFIG["memmap_file"])
    if memmap_dir:
        os.makedirs(memmap_dir, exist_ok=True)
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
