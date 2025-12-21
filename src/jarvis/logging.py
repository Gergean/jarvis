"""Logging configuration for the Jarvis trading system."""

import logging
from logging.handlers import RotatingFileHandler
from os.path import isfile


def get_logger(filename: str = "logs/backtest.log") -> logging.Logger:
    _logger = logging.getLogger(__name__)
    _logger.propagate = False
    _logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(filename, mode="a", backupCount=5)
    if isfile(filename):
        file_handler.doRollover()
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(levelname)s: %(funcName)s : %(message)s"))

    _logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(stream_handler)
    return _logger


logger = get_logger()
