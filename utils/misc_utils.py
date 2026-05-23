"""
utils/misc_utils.py

Shared logging utility used across all inference and debiasing scripts.
"""
import logging


def setup_logging():
    """Configure and return a module-level logger with timestamped console output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)
