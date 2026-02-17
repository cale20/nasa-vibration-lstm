# --------------------------------------------------
# train_isolation_forest.py
# --------------------------------------------------
"""Train and save an Isolation Forest model.

This script loads the memmap dataset, optionally subsamples it for
fast training, fits an IsolationForest, and saves the model file.
"""
import argparse
import logging
import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from .dataset import load_memmap_dataset
from .config import CONFIG, ensure_output_dirs, configure_logging


def train(limit: int | None = None) -> str:
    """Train IsolationForest and return path to saved model.

    Args:
        limit: if provided and smaller than dataset size, limit training samples.
    """
    X = load_memmap_dataset(flatten_for_tree=True)

    # Optionally limit dataset for demo/troubleshooting
    if limit is not None and limit > 0 and X.shape[0] > limit:
        X = X[:limit]

    # Subsample deterministically for reproducible runs
    if X.shape[0] > CONFIG["max_train_samples"]:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], CONFIG["max_train_samples"], replace=False)
        X_train = X[idx]
    else:
        X_train = X

    model = IsolationForest(
        n_estimators=CONFIG["n_estimators"],
        contamination=CONFIG["contamination"],
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    ensure_output_dirs()
    model_path = os.path.join(CONFIG["processed_folder"], "isolation_forest.model")
    joblib.dump(model, model_path)
    logging.info("Isolation Forest trained | Training samples: %s", X_train.shape)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Isolation Forest model")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training examples (demo)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    try:
        model_file = train(limit=args.limit)
        logging.info("Model saved to %s", model_file)
    except Exception as exc:
        logging.exception("Training failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
