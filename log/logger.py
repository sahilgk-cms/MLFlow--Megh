import logging
import os
import sys
from config.filepaths import LOGS_DIRECTORY

# Ensure logs directory exists
os.makedirs(LOGS_DIRECTORY, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOGS_DIRECTORY, "ml_training.log")


def get_logger(name: str) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s - %(lineno)s - %(levelname)s: %(message)s"
    )

    # File handler 
    file_handler = logging.FileHandler(
        LOG_FILE_PATH,
        mode="a",
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger