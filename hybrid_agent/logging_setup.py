"""
Basic logging setup for the Hybrid HRM+EBM Agent.
"""
from __future__ import annotations
import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", *, fmt: Optional[str] = None) -> None:
    """Configure root logger with a simple stdout formatter.

    Parameters
    ----------
    level: str
        Log level name (e.g., "DEBUG", "INFO", "WARNING"). Defaults to INFO.
    fmt: Optional[str]
        Optional custom format string. If not provided, a sensible default is used.
    """
    fmt = fmt or "%(asctime)s %(levelname)s %(name)s: %(message)s"
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Reset basicConfig only if no handlers are configured yet
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=numeric_level,
            format=fmt,
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )
    else:
        logging.getLogger().setLevel(numeric_level)

    # Quiet noisy libraries a bit by default
    logging.getLogger("urllib3").setLevel(max(logging.WARNING, numeric_level))
    logging.getLogger("httpx").setLevel(max(logging.WARNING, numeric_level))
