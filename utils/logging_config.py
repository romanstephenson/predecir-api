import logging
import os
from logging.handlers import RotatingFileHandler
from core.config import settings
import time

LOG_DIR = settings.LOG_DIR or "logs"
LOG_FILE_NAME = settings.LOG_FILE_NAME
LOG_FILE = os.path.join(LOG_DIR, LOG_FILE_NAME)
MAX_LOG_SIZE =settings.LOG_MAX_SIZE_MB  # 5 MB
BACKUP_COUNT = settings.LOG_BACKUP_COUNT

# Parse log level from string
LOG_LEVEL = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

# Convert MB to bytes
LOG_MAX_SIZE_BYTES = MAX_LOG_SIZE * 1024 * 1024

os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)  # Internal log filtering

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.Formatter.converter = time.localtime # For local timestamps
    
    # File handler with full DEBUG logs
    rf_handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_SIZE_BYTES, backupCount=BACKUP_COUNT)
    rf_handler.setFormatter(formatter)
    rf_handler.setLevel(LOG_LEVEL)

    # Console handler with env-based log level
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(LOG_LEVEL)

    logger.addHandler(rf_handler)
    logger.addHandler(console_handler)

    return logger
