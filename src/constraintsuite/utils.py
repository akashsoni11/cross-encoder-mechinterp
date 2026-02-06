"""
Utility functions for ConstraintSuite.

This module provides common utilities for:
- Configuration loading and validation
- JSONL I/O operations
- Logging setup
- Random seed management
"""

from pathlib import Path
from typing import Any, Iterator
import json
import logging
import random

import yaml
import numpy as np


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.

    Example:
        >>> config = load_config("configs/negation_v0.yaml")
        >>> print(config["dataset"]["source_corpus"])
        'msmarco-passage'
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Basic validation
    if config is None:
        raise ValueError(f"Empty config file: {config_path}")

    return config


def save_jsonl(data: list[dict], output_path: str | Path) -> None:
    """
    Save a list of dictionaries to a JSONL file.

    Args:
        data: List of dictionaries to save.
        output_path: Path to output JSONL file.

    Example:
        >>> examples = [{"id": "001", "query": "test"}]
        >>> save_jsonl(examples, "data/output.jsonl")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(input_path: str | Path) -> list[dict]:
    """
    Load a JSONL file into a list of dictionaries.

    Args:
        input_path: Path to input JSONL file.

    Returns:
        List of dictionaries, one per line.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If any line is invalid JSON.

    Example:
        >>> examples = load_jsonl("data/release/negation_v0/main.jsonl")
        >>> print(len(examples))
        2000
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON on line {line_num}: {e.msg}",
                        e.doc,
                        e.pos
                    )
    return data


def iter_jsonl(input_path: str | Path) -> Iterator[dict]:
    """
    Iterate over a JSONL file line by line (memory efficient).

    Args:
        input_path: Path to input JSONL file.

    Yields:
        Dictionary for each line.

    Example:
        >>> for example in iter_jsonl("data/large_file.jsonl"):
        ...     process(example)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logging("DEBUG", "logs/run.log")
        >>> logger.info("Starting pipeline")
    """
    # Get the root logger for constraintsuite
    logger = logging.getLogger("constraintsuite")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Sets seed for:
    - Python's random module
    - NumPy
    - PyTorch (if available)

    Args:
        seed: Random seed value.

    Example:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Try to set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # For MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have a separate seed function, torch.manual_seed covers it
            pass
    except ImportError:
        pass  # PyTorch not installed


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.

    Example:
        >>> output_dir = ensure_dir("data/intermediate/candidates")
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name: str = "constraintsuite") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
