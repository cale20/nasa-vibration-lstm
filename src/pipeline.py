"""Simple orchestrator that runs preprocessing, training and evaluation in order.

This is intentionally minimal: it calls the existing `main()` entrypoints
from the corresponding scripts. No CLI args are parsed here (by design
for the initial implementation).
"""
from __future__ import annotations

from . import run_preprocessing
from . import train_isolation_forest
from . import evaluate
from .config import configure_logging, ensure_output_dirs
import logging


def run() -> dict:
    """Run the full pipeline: preprocess -> train -> evaluate.

    Returns a small dict summarizing outcome (keeps signature simple for now).
    """
    configure_logging()
    ensure_output_dirs()

    logging.info("[pipeline] Starting preprocessing...")
    run_preprocessing.main()

    logging.info("[pipeline] Starting training...")
    train_isolation_forest.main()

    logging.info("[pipeline] Starting evaluation...")
    evaluate.main()

    logging.info("[pipeline] Pipeline finished.")
    return {"status": "ok"}


if __name__ == "__main__":
    run()
