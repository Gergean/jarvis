"""Logging configuration for the Jarvis trading system."""

import logging
import multiprocessing


def get_logger() -> logging.Logger:
    _logger = logging.getLogger("jarvis")

    # Prevent adding handlers multiple times
    if _logger.handlers:
        return _logger

    _logger.propagate = False
    _logger.setLevel(logging.DEBUG)

    # Skip logging in multiprocessing worker processes
    is_worker = multiprocessing.current_process().name != "MainProcess"

    if not is_worker:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(stream_handler)
    else:
        _logger.addHandler(logging.NullHandler())

    return _logger


logger = get_logger()
