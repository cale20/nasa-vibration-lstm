"""Human-readable logging helpers for long-running pipeline jobs."""

from __future__ import annotations

import logging


def log_section(title: str) -> None:
    line = "=" * 70
    logging.info("")
    logging.info(line)
    logging.info("%s", title)
    logging.info(line)


def log_step(message: str) -> None:
    logging.info("→ %s", message)


def log_note(message: str) -> None:
    logging.info("• %s", message)


def log_progress(message: str) -> None:
    logging.info("↻ %s", message)


def log_ok(message: str) -> None:
    logging.info("✓ %s", message)


def fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, sec = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {sec}s"
    return f"{mins}m {sec}s"
