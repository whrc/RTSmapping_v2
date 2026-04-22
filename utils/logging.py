"""Logging setup. Use this instead of print() per CLAUDE.md §Code Style."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure the root logger with stdout (and optional file) handler.

    Idempotent: subsequent calls replace existing handlers so re-running scripts
    in the same Python session doesn't produce duplicate lines.

    Args:
        level: logging level name (DEBUG, INFO, WARNING, ERROR).
        log_file: optional path to also log to a file.

    Returns:
        The root logger.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return root
