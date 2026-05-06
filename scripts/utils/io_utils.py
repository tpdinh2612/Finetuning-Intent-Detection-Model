# =========================================================
# I/O Utilities
# =========================================================
"""
Configuration loading, path validation, and directory management.
All pipeline scripts ingest their parameters through these functions.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from scripts.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary.

    Fails fast if the file does not exist, preventing the pipeline from
    proceeding with invalid state (e.g., loading a 4-bit model before
    discovering an output path is missing).

    Args:
        config_path: Absolute or relative path to the YAML config file.

    Returns:
        Dictionary of configuration key-value pairs.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.resolve()}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from: %s", config_path.resolve())
    return config


def validate_paths(paths: List[str]) -> None:
    """Validate that all provided paths exist on disk.

    Called early in each pipeline stage to fail fast on missing
    data directories or model checkpoints.

    Args:
        paths: List of filesystem paths to verify.

    Raises:
        FileNotFoundError: If any path does not exist.
    """
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")
    logger.info("All %d paths validated successfully.", len(paths))


def ensure_dir(directory: str) -> Path:
    """Create a directory (and parents) if it does not already exist.

    Args:
        directory: Path to the directory to create.

    Returns:
        Resolved ``Path`` object for the directory.
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path.resolve()
