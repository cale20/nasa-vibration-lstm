"""Configurable orchestrator for preprocessing, training, and evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from datetime import datetime
from typing import Any

from .config import CONFIG, configure_logging, ensure_output_dirs
from .evaluate import machine_health_curve
from .evaluate_autoencoder import evaluate as evaluate_autoencoder
from .evaluate_unsupervised import evaluate_all_models
from .logging_utils import fmt_seconds, log_note, log_ok, log_progress, log_section, log_step
from .preprocessing import create_memmap_dataset, fit_global_scaler
from .train_dense_autoencoder import train as train_dense_autoencoder
from .train_isolation_forest import train as train_isolation_forest
from .train_lstm_autoencoder import train as train_lstm_autoencoder
from .utils import list_ims_files


def _run_preprocessing(preprocess_limit: int | None = None, data_folder: str | None = None) -> dict[str, Any]:
    folder = data_folder or CONFIG["data_folder"]
    files = list_ims_files(folder, seq_length=CONFIG["sequence_length"])
    if preprocess_limit is not None and preprocess_limit > 0:
        files = files[:preprocess_limit]
    scaler = fit_global_scaler(files)
    outputs = create_memmap_dataset(files, scaler)
    return {"num_files": len(files), "outputs": outputs}


def _detect_device() -> str:
    try:
        import torch
    except Exception:
        return "unknown"
    if torch.cuda.is_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    return "cpu"


def _copy_if_exists(src_path: str, dst_path: str) -> None:
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    elif os.path.isfile(src_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)


def run(
    preprocess: bool = True,
    preprocess_limit: int | None = None,
    data_folder: str | None = None,
    run_if: bool = True,
    if_train_limit: int | None = None,
    if_eval_limit: int | None = None,
    run_dense: bool = True,
    dense_epochs: int | None = None,
    dense_max_train_batches: int | None = None,
    dense_max_val_batches: int | None = None,
    run_lstm: bool = True,
    lstm_epochs: int | None = None,
    lstm_max_train_batches: int | None = None,
    lstm_max_val_batches: int | None = None,
    log_interval_batches: int | None = None,
    log_interval_files: int | None = None,
    run_unsupervised_eval: bool = True,
    run_tag: str | None = None,
    save_run_artifacts: bool = True,
    log_path: str | None = None,
    cli_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run selected pipeline steps with configurable limits."""
    ensure_output_dirs()
    started_at = datetime.now().isoformat()
    step_times: dict[str, float] = {}
    summary: dict[str, Any] = {
        "status": "ok",
        "run_tag": run_tag,
        "started_at": started_at,
        "random_seed": int(CONFIG["random_seed"]),
        "device": _detect_device(),
    }
    log_section(f"[{run_tag}] PIPELINE START")
    log_step("Starting pipeline run")
    log_note(f"Seed: {CONFIG['random_seed']}")
    log_note(f"Device: {summary['device']}")
    if log_path:
        log_note(f"Log file: {log_path}")

    def _run_step(step_name: str, fn):
        log_step(f"Starting {step_name}")
        step_start = time.perf_counter()
        result = fn()
        duration_sec = time.perf_counter() - step_start
        step_times[step_name] = duration_sec
        log_ok(f"Completed {step_name} in {fmt_seconds(duration_sec)}")
        return result

    if preprocess:
        summary["preprocessing"] = _run_step(
            "preprocessing",
            lambda: _run_preprocessing(
                preprocess_limit=preprocess_limit,
                data_folder=data_folder,
            ),
        )

    if run_if:
        model_path = _run_step(
            "if_train",
            lambda: train_isolation_forest(limit=if_train_limit),
        )
        if_result = _run_step(
            "if_eval",
            lambda: machine_health_curve(limit=if_eval_limit, log_interval_files=log_interval_files),
        )
        summary["isolation_forest"] = {
            "model_path": model_path,
            "threshold": if_result.get("threshold"),
            "diagnostics_dir": if_result.get("diagnostics_dir"),
        }

    if run_dense:
        dense_path = _run_step(
            "dense_train",
            lambda: train_dense_autoencoder(
                epochs=dense_epochs,
                max_train_batches=dense_max_train_batches,
                max_val_batches=dense_max_val_batches,
                log_interval_batches=log_interval_batches,
            ),
        )
        dense_eval = _run_step(
            "dense_eval",
            lambda: evaluate_autoencoder(model_type="dense", log_interval_files=log_interval_files),
        )
        summary["dense_autoencoder"] = {
            "model_path": dense_path,
            "threshold": dense_eval.get("threshold"),
            "diagnostics_dir": dense_eval.get("diagnostics_dir"),
        }

    if run_lstm:
        lstm_path = _run_step(
            "lstm_train",
            lambda: train_lstm_autoencoder(
                epochs=lstm_epochs,
                max_train_batches=lstm_max_train_batches,
                max_val_batches=lstm_max_val_batches,
                log_interval_batches=log_interval_batches,
            ),
        )
        lstm_eval = _run_step(
            "lstm_eval",
            lambda: evaluate_autoencoder(model_type="lstm", log_interval_files=log_interval_files),
        )
        summary["lstm_autoencoder"] = {
            "model_path": lstm_path,
            "threshold": lstm_eval.get("threshold"),
            "diagnostics_dir": lstm_eval.get("diagnostics_dir"),
        }

    if run_unsupervised_eval:
        summary["unsupervised"] = _run_step("unsupervised_eval", evaluate_all_models)

    finished_at = datetime.now().isoformat()
    summary["finished_at"] = finished_at
    summary["step_durations_sec"] = step_times
    summary["total_duration_sec"] = float(sum(step_times.values()))
    summary["log_path"] = log_path
    summary["args"] = cli_args or {}

    runs_dir = os.path.join(CONFIG["processed_folder"], "runs")
    run_dir = os.path.join(runs_dir, str(run_tag))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh)
    with open(os.path.join(run_dir, "timings.json"), "w", encoding="utf-8") as fh:
        json.dump(step_times, fh)

    if save_run_artifacts:
        diagnostics_src = os.path.join(CONFIG["processed_folder"], "diagnostics")
        _copy_if_exists(diagnostics_src, os.path.join(run_dir, "diagnostics"))
        _copy_if_exists(CONFIG["split_metadata_file"], os.path.join(run_dir, "split_metadata.json"))
        _copy_if_exists(CONFIG["scaler_file"], os.path.join(run_dir, "global_scaler.save"))

    log_section(f"[{run_tag}] PIPELINE COMPLETE")
    log_note(f"Total runtime: {fmt_seconds(summary['total_duration_sec'])}")
    for step_name, seconds in step_times.items():
        log_note(f"{step_name}: {fmt_seconds(seconds)}")
    log_ok(f"Run outputs saved to {run_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end anomaly pipeline")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Override CONFIG random_seed for this run")
    parser.add_argument("--run-tag", type=str, default=None, help="Run identifier used for logs and artifacts")
    parser.add_argument("--no-save-run-artifacts", action="store_true", help="Do not snapshot diagnostics under data/processed/runs/<run_tag>")
    parser.add_argument("--log-interval-batches", type=int, default=CONFIG["log_interval_batches"])
    parser.add_argument("--log-interval-files", type=int, default=CONFIG["log_interval_files"])
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--preprocess-limit", type=int, default=None)
    parser.add_argument("--data-folder", type=str, default=None)

    parser.add_argument("--skip-if", action="store_true")
    parser.add_argument("--if-train-limit", type=int, default=None)
    parser.add_argument("--if-eval-limit", type=int, default=None)

    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--dense-epochs", type=int, default=None)
    parser.add_argument("--dense-max-train-batches", type=int, default=None)
    parser.add_argument("--dense-max-val-batches", type=int, default=None)

    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--lstm-epochs", type=int, default=None)
    parser.add_argument("--lstm-max-train-batches", type=int, default=None)
    parser.add_argument("--lstm-max-val-batches", type=int, default=None)

    parser.add_argument("--skip-unsupervised", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    if args.seed is not None:
        CONFIG["random_seed"] = int(args.seed)
    run_tag = args.run_tag or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{CONFIG['random_seed']}"

    logs_dir = os.path.join(CONFIG["processed_folder"], "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"run_{run_tag}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)
    log_note(f"Logging configured (batches={args.log_interval_batches}, files={args.log_interval_files})")

    run(
        preprocess=not args.skip_preprocess,
        preprocess_limit=args.preprocess_limit,
        data_folder=args.data_folder,
        run_if=not args.skip_if,
        if_train_limit=args.if_train_limit,
        if_eval_limit=args.if_eval_limit,
        run_dense=not args.skip_dense,
        dense_epochs=args.dense_epochs,
        dense_max_train_batches=args.dense_max_train_batches,
        dense_max_val_batches=args.dense_max_val_batches,
        run_lstm=not args.skip_lstm,
        lstm_epochs=args.lstm_epochs,
        lstm_max_train_batches=args.lstm_max_train_batches,
        lstm_max_val_batches=args.lstm_max_val_batches,
        log_interval_batches=args.log_interval_batches,
        log_interval_files=args.log_interval_files,
        run_unsupervised_eval=not args.skip_unsupervised,
        run_tag=run_tag,
        save_run_artifacts=not args.no_save_run_artifacts,
        log_path=log_path,
        cli_args=vars(args),
    )


if __name__ == "__main__":
    main()
