import logging

# one-stop entry point for logging
from typing import Any, Dict


def create_logger(level: int = logging.INFO):
    """
    Create a logger that logs to both file (in debug mode) and terminal (info).

    Note: this should be called inside a Ray actor. A logging actor has to create an instance
    of a logger (created by default by the logging singleton) within its python process. Making one earlier
    will not make the logs distributed and cause bottlenecks.

    See https://docs.ray.io/en/master/ray-logging.html#how-to-set-up-loggers for details.

    """
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=level)


class LoggableMixin:
    """A mixin to store variables when recording log entries."""
    def __init__(self):
        self.logged_data: Dict[str, Any] = {}

    def log(self, log_message: str, *args, **kwargs):
        """Wrapper on the logger function that logs a message and saves these for visualisation purposes"""
        self.logged_data.update(kwargs)
        logging.debug(log_message, *args, *(kwargs.values()))

    def clear_log(self):
        self.logged_data.clear()
