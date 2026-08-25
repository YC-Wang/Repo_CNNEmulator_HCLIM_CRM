from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.logging_utils import setup_logging


def close_logger_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


class LoggingUtilsTests(unittest.TestCase):
    def test_setup_logging_creates_file_and_avoids_duplicate_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "logs" / "diagnostic.log"
            logger = setup_logging(log_file, logger_name="logging_utils_test")
            logger.info("first message")
            logger = setup_logging(log_file, logger_name="logging_utils_test")
            logger.info("second message")

            for handler in logger.handlers:
                handler.flush()

            content = log_file.read_text(encoding="utf-8")
            exists = log_file.exists()
            close_logger_handlers(logger)

        self.assertTrue(exists)
        self.assertEqual(content.count("first message"), 1)
        self.assertEqual(content.count("second message"), 1)

    def test_setup_logging_records_exception_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "traceback.log"
            logger = setup_logging(log_file, logger_name="logging_utils_traceback")
            try:
                raise RuntimeError("boom")
            except RuntimeError:
                logger.exception("Training failed")

            for handler in logger.handlers:
                handler.flush()

            content = log_file.read_text(encoding="utf-8")
            close_logger_handlers(logger)

        self.assertIn("Training failed", content)
        self.assertIn("Traceback", content)
        self.assertIn("RuntimeError: boom", content)


if __name__ == "__main__":
    unittest.main()
