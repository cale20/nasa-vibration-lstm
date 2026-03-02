"""Unsupervised model quality evaluation across file-level diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from typing import Dict, List

import numpy as np

from .config import CONFIG, configure_logging
from .dataset import load_split_metadata


MODEL_ALIASES = {
    "if": "isolation_forest",
    "isolation_forest": "isolation_forest",
    "dense": "dense_autoencoder",
    "dense_autoencoder": "dense_autoencoder",
    "lstm": "lstm_autoencoder",
    "lstm_autoencoder": "lstm_autoencoder",
}


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks (1-indexed) for 1D array."""
    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < values.shape[0]:
        j = i
        while j + 1 < values.shape[0] and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation without SciPy dependency."""
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        return float("nan")
    rx = _rankdata(x.astype(np.float64))
    ry = _rankdata(y.astype(np.float64))
    cx = rx - np.mean(rx)
    cy = ry - np.mean(ry)
    denom = np.sqrt(np.sum(cx**2) * np.sum(cy**2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(cx * cy) / denom)


def _cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Return effect size between two groups."""
    if group_a.size == 0 or group_b.size == 0:
        return float("nan")
    var_a = float(np.var(group_a, ddof=1)) if group_a.size > 1 else 0.0
    var_b = float(np.var(group_b, ddof=1)) if group_b.size > 1 else 0.0
    pooled_num = (group_a.size - 1) * var_a + (group_b.size - 1) * var_b
    pooled_den = group_a.size + group_b.size - 2
    if pooled_den <= 0:
        return 0.0
    pooled_std = float(np.sqrt(max(pooled_num / pooled_den, 0.0)))
    if pooled_std <= 1e-12:
        return 0.0
    return float((np.mean(group_b) - np.mean(group_a)) / pooled_std)


def _persistent_mask(alerts: np.ndarray, k: int, m: int, start_idx: int = 0) -> np.ndarray:
    """Mark timesteps where at least k alerts occur in trailing m-length window."""
    n = alerts.shape[0]
    mask = np.zeros(n, dtype=np.uint8)
    for idx in range(start_idx, n):
        left = max(start_idx, idx - m + 1)
        if int(np.sum(alerts[left : idx + 1])) >= k:
            mask[idx] = 1
    return mask


def _first_persistent_alert_idx(alerts: np.ndarray, k: int, m: int, start_idx: int) -> int | None:
    mask = _persistent_mask(alerts, k=k, m=m, start_idx=start_idx)
    found = np.where(mask == 1)[0]
    if found.size == 0:
        return None
    return int(found[0])


def _count_persistent_segments(alerts: np.ndarray, k: int, m: int, start_idx: int) -> int:
    mask = _persistent_mask(alerts, k=k, m=m, start_idx=start_idx)
    active = False
    segments = 0
    for idx in range(start_idx, mask.shape[0]):
        if mask[idx] and not active:
            segments += 1
            active = True
        elif not mask[idx]:
            active = False
    return int(segments)


def _resolve_model_name(model_type: str) -> str:
    key = model_type.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unknown model_type: {model_type}")
    return MODEL_ALIASES[key]


def _load_file_artifacts(model_name: str, diagnostics_dir: str) -> tuple[np.ndarray, np.ndarray]:
    scores_path = os.path.join(diagnostics_dir, f"{model_name}_file_scores.npy")
    alerts_path = os.path.join(diagnostics_dir, f"{model_name}_file_alerts.npy")
    if not os.path.exists(scores_path) or not os.path.exists(alerts_path):
        raise FileNotFoundError(
            f"Missing standardized artifacts for {model_name}. "
            f"Expected {scores_path} and {alerts_path}. Run model evaluation first."
        )
    scores = np.load(scores_path).astype(np.float32)
    alerts = np.load(alerts_path).astype(np.uint8)
    if scores.shape[0] != alerts.shape[0]:
        raise ValueError(f"Shape mismatch for {model_name}: scores={scores.shape}, alerts={alerts.shape}")
    return scores, alerts


def _indices_for_split(file_records: List[dict], split_name: str) -> np.ndarray:
    indices = [i for i, rec in enumerate(file_records) if rec.get("split") == split_name]
    return np.asarray(indices, dtype=np.int64)


def evaluate_model(model_type: str) -> Dict[str, object]:
    """Evaluate unsupervised behavior for one model."""
    model_name = _resolve_model_name(model_type)
    diagnostics_dir = os.path.join(CONFIG["processed_folder"], "diagnostics")
    scores, alerts = _load_file_artifacts(model_name, diagnostics_dir)
    split_meta = load_split_metadata() or {}
    file_records = split_meta.get("file_records", [])
    if len(file_records) != int(scores.shape[0]):
        raise ValueError(
            f"split_metadata file_records ({len(file_records)}) does not match file artifacts ({scores.shape[0]})."
        )

    healthy_split = str(CONFIG["healthy_reference_split"])
    late_split = str(CONFIG["late_life_split"])
    healthy_idx = _indices_for_split(file_records, healthy_split)
    late_idx = _indices_for_split(file_records, late_split)
    late_start = int(late_idx[0]) if late_idx.size > 0 else 0

    healthy_far = float(np.mean(alerts[healthy_idx])) if healthy_idx.size > 0 else float("nan")
    late_life_alert_rate = float(np.mean(alerts[late_idx])) if late_idx.size > 0 else float("nan")
    trend_spearman = _spearman_rank_correlation(
        np.arange(scores.shape[0], dtype=np.float64),
        scores.astype(np.float64),
    )
    first_persistent = _first_persistent_alert_idx(
        alerts=alerts,
        k=int(CONFIG["persistence_k"]),
        m=int(CONFIG["persistence_m"]),
        start_idx=late_start,
    )
    persistent_count = _count_persistent_segments(
        alerts=alerts,
        k=int(CONFIG["persistence_k"]),
        m=int(CONFIG["persistence_m"]),
        start_idx=late_start,
    )
    separation = _cohens_d(
        scores[healthy_idx] if healthy_idx.size > 0 else np.array([], dtype=np.float32),
        scores[late_idx] if late_idx.size > 0 else np.array([], dtype=np.float32),
    )

    metrics = {
        "model": model_name,
        "num_files": int(scores.shape[0]),
        "healthy_reference_split": healthy_split,
        "late_life_split": late_split,
        "healthy_false_alarm_rate": healthy_far,
        "late_life_alert_rate": late_life_alert_rate,
        "trend_spearman_r": trend_spearman,
        "first_persistent_alert_file_idx": first_persistent,
        "persistent_alert_count": persistent_count,
        "signal_separation": separation,
        "passes_healthy_far_cap": bool(
            np.isfinite(healthy_far) and healthy_far <= float(CONFIG["max_false_alarm_rate_healthy"])
        ),
        "passes_trend_gate": bool(
            np.isfinite(trend_spearman) and trend_spearman >= float(CONFIG["trend_min_spearman"])
        ),
        "config_used": {
            "healthy_reference_split": healthy_split,
            "late_life_split": late_split,
            "persistence_k": int(CONFIG["persistence_k"]),
            "persistence_m": int(CONFIG["persistence_m"]),
            "max_false_alarm_rate_healthy": float(CONFIG["max_false_alarm_rate_healthy"]),
            "trend_min_spearman": float(CONFIG["trend_min_spearman"]),
        },
    }

    json_path = os.path.join(diagnostics_dir, f"{model_name}_unsupervised_metrics.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh)

    csv_path = os.path.join(diagnostics_dir, f"{model_name}_unsupervised_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model",
                "num_files",
                "healthy_false_alarm_rate",
                "late_life_alert_rate",
                "trend_spearman_r",
                "first_persistent_alert_file_idx",
                "persistent_alert_count",
                "signal_separation",
                "passes_healthy_far_cap",
                "passes_trend_gate",
            ],
        )
        writer.writeheader()
        writer.writerow({k: metrics.get(k) for k in writer.fieldnames})

    return {
        "model": model_name,
        "metrics_path": json_path,
        "summary_path": csv_path,
        "healthy_false_alarm_rate": healthy_far,
        "trend_spearman_r": trend_spearman,
    }


def evaluate_all_models() -> Dict[str, object]:
    diagnostics_dir = os.path.join(CONFIG["processed_folder"], "diagnostics")
    models = ["isolation_forest", "dense_autoencoder", "lstm_autoencoder"]
    rows = [evaluate_model(model) for model in models]
    out_json = os.path.join(diagnostics_dir, "unsupervised_model_comparison.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)

    out_csv = os.path.join(diagnostics_dir, "unsupervised_model_comparison.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model",
                "healthy_false_alarm_rate",
                "trend_spearman_r",
                "metrics_path",
                "summary_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    return {"rows": rows, "json_path": out_json, "csv_path": out_csv}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate unsupervised anomaly metrics")
    parser.add_argument(
        "--model-type",
        choices=["if", "isolation_forest", "dense", "dense_autoencoder", "lstm", "lstm_autoencoder"],
        default=None,
    )
    parser.add_argument("--all-models", action="store_true", help="Evaluate all supported models")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    if args.all_models:
        result = evaluate_all_models()
        logging.info("Saved unsupervised comparison to %s and %s", result["json_path"], result["csv_path"])
        return
    if args.model_type is None:
        raise ValueError("Provide --model-type or use --all-models")
    result = evaluate_model(args.model_type)
    logging.info(
        "[%s] unsupervised metrics saved: %s",
        result["model"],
        result["metrics_path"],
    )


if __name__ == "__main__":
    main()
