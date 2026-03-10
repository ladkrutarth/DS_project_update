import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config.config import settings

def setup_logger(name: str = "veriscan"):
    """
    Sets up a structured logger that outputs to both console and a rotating file.
    """
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = settings.LOGS_DIR / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL)

    # Avoid adding multiple handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    # Formatting
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (10MB max per file, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Global default logger
logger = setup_logger()
