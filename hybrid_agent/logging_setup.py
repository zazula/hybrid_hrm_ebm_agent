"""
Simple structured logging setup for the agent.

Usage:
    from hybrid_agent.logging_setup import setup_logging
    log = setup_logging(__name__)
    log.info("hello")
"""
from __future__ import annotations
import logging
import os
import sys


def setup_logging(name: str | None = None) -> logging.Logger:
    """Configure root logging once and return a module-specific logger.

    Level is controlled via LOG_LEVEL env var (default INFO).
    """
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    # Configure root handler to stdout with a concise format
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # idempotent setup
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)

    # Quiet common noisy libs
    for noisy in ("urllib3", "httpx", "asyncio", "matplotlib"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, level))

    return logging.getLogger(name) if name else root
