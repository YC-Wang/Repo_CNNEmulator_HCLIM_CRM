from __future__ import annotations

import logging
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)


def add_file_handler(
    logger: logging.Logger,
    log_file: str | Path,
    level: int = logging.INFO,
) -> Path:
    path = Path(log_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == path:
            handler.setLevel(level)
            return path

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(_build_formatter())
    logger.addHandler(file_handler)
    return path


def setup_logging(
    log_file: str | Path,
    level: int = logging.INFO,
    logger_name: str = "paper_revision_cnn",
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(_build_formatter())
        logger.addHandler(stream_handler)

    add_file_handler(logger, log_file, level=level)
    return logger
