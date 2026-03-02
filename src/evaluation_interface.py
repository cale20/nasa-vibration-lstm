"""Evaluation interface to keep adapters swappable by data availability."""

from __future__ import annotations

import argparse
import logging

from .config import configure_logging
from .evaluate_unsupervised import evaluate_all_models, evaluate_model


def run_evaluation(kind: str, model_type: str | None = None, all_models: bool = False):
    """Dispatch evaluation request to selected adapter."""
    if kind == "unsupervised":
        if all_models:
            return evaluate_all_models()
        if model_type is None:
            raise ValueError("model_type is required for unsupervised single-model evaluation")
        return evaluate_model(model_type)
    if kind == "change_windows":
        raise NotImplementedError(
            "change_windows adapter is intentionally not implemented in this dataset-only phase."
        )
    raise ValueError(f"Unknown evaluation kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation through adapter interface")
    parser.add_argument("--kind", choices=["unsupervised", "change_windows"], required=True)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    result = run_evaluation(kind=args.kind, model_type=args.model_type, all_models=args.all_models)
    logging.info("Evaluation completed: %s", result)


if __name__ == "__main__":
    main()
