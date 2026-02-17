def suppress_console_logging():
    """
    Temporarily suppress all logging output to the console (StreamHandler), but keep file logging.
    """
    import logging
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.CRITICAL + 1)

def restore_console_logging(level=None):
    """
    Restore logging output to the console (StreamHandler) at the given level (default: INFO).
    """
    import logging
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level if level is not None else logging.INFO)
"""
Logging setup utilities for DAQ acquisition system.
Provides consistent logging configuration across all modules.
"""
import logging
import datetime
import os
from pathlib import Path


TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_level=logging.INFO, log_to_file=True, log_folder="logs"):
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (default: logging.INFO)
        log_to_file: Whether to log to file in addition to console (default: True)
        log_folder: Folder to store log files (default: "logs")
    
    Returns:
        logging.Logger: Configured root logger
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        log_folder_path = Path(log_folder)
        log_folder_path.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        log_file = log_folder_path / f"daq_log_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_logger(name):
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the module (typically __name__)
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


def create_data_file(prefix="data", folder="data", extension="csv"):
    """
    Create a timestamped data file path for storing acquisition data.
    
    Args:
        prefix: File name prefix (default: "data")
        folder: Folder to store data files (default: "data")
        extension: File extension without dot (default: "csv")
    
    Returns:
        Path: Path object for the data file
    """
    folder_path = Path(folder)
    folder_path.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
    filename = f"{prefix}_{timestamp}.{extension}"
    
    return folder_path / filename

