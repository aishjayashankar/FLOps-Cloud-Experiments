import logging
import os
import sys

def configure_logging(log_file: str = None):
    """Configures the root logger to write to a file and console."""
    if log_file is None:
        log_file = os.getenv("LOG_FILE", "flops_infra_drift.log")

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    # Check if handlers already exist to avoid duplicate logs
    if not logger.hasHandlers():
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    else:
        # If handlers exist, we might want to ensure our file handler is added if not present
        # For simplicity in this context, we'll just add it if it's not there, 
        # but typically we might want to clear existing handlers or be more careful.
        # Let's just append for now as it's safer than clearing.
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        if not has_file_handler:
             logger.addHandler(f_handler)

    logging.info(f"Logging configured. Writing to {log_file}")
