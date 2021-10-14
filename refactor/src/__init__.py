import logging

# one-stop entry point for logging
from typing import Any, Dict

logger = logging.getLogger("SeaLiceManagementGame")

def create_logger():
    """
    Create a logger that logs to both file (in debug mode) and terminal (info).
    """
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("SeaLiceManagementGame.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    term_handler = logging.StreamHandler()
    term_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    term_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(term_handler)
    logger.addHandler(file_handler)

class LoggableMixin:
    """A mixin to store variables when recording log entries."""
    def __init__(self):
        self.logged_data: Dict[str, Any] = {}

    def log(self, log_message: str, *args, **kwargs):
        """Wrapper on the logger function that logs a message and saves these for visualisation purposes"""
        self.logged_data.update(kwargs)
        logger.debug(log_message, *args, *(kwargs.values()))

    def clear_log(self):
        self.logged_data.clear()