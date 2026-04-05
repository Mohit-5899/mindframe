"""
MindFrame - Video analysis platform for transcripts, key frames, and content patterns.

Logging is configured here once, used everywhere via:
    import logging
    logger = logging.getLogger(__name__)
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for the project.

    Call once at app startup. All modules use logging.getLogger(__name__).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    root = logging.getLogger("src")
    root.setLevel(level)

    if not root.handlers:
        root.addHandler(handler)
