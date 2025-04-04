"""
Centralized logging configuration for the climate policy extractor.
"""
import logging
import colorlog
from scrapy.utils.project import get_project_settings

def setup_colored_logging(logger=None):
    """Set up colored logging for both custom and Scrapy loggers."""
    settings = get_project_settings()
    
    # If LOG_FILE is set and we're not supposed to log to stdout/stderr,
    # don't add any console handlers
    if settings.get('LOG_FILE') and not (settings.get('LOG_STDOUT') or settings.get('LOG_STDERR')):
        return
    
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(name)s] %(levelname)s:%(reset)s %(white)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Configure specific logger if provided
    if logger:
        logger.handlers = []
        logger.addHandler(handler)

def get_logger(name):
    """
    Get a logger with colored formatting.
    Use this instead of logging.getLogger() throughout the project.
    """
    logger = logging.getLogger(name)
    setup_colored_logging(logger)
    return logger 