# =========================================================
# Logging Configuration
# =========================================================
"""
Standardized logging factory for the pipeline.
Replaces bare print() calls with structured, level-aware logging.
"""

import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create or retrieve a named logger with console and optional file output.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Minimum logging level (default: INFO).
        log_file: Optional path to a log file for persistent output.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handler attachment on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
