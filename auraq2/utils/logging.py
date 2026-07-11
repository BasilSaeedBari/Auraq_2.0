"""
Auraq 2.0 — Logging Utility
Provides a consistent, color-coded logger for both CLI output and GUI integration.
"""
import logging
import sys
import os

_LOGGER_NAME = "auraq2"
_logger: logging.Logger | None = None


class _ColorFormatter(logging.Formatter):
    """ANSI-colored formatter for terminal output."""
    COLORS = {
        logging.DEBUG:   "\033[90m",   # Grey
        logging.INFO:    "\033[0m",    # Default
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR:   "\033[31m",   # Red
        logging.CRITICAL:"\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"


def setup_logger(verbose_level: int = 0, log_to_file: bool = False) -> logging.Logger:
    """
    Configure and return the shared Auraq 2.0 logger.

    Args:
        verbose_level: 0 = INFO, 1 = DEBUG.
        log_to_file: If True, also write to %APPDATA%/auraq2/auraq2.log.
    """
    global _logger
    logger = logging.getLogger(_LOGGER_NAME)

    if logger.handlers:
        # Already configured — just adjust level.
        logger.setLevel(logging.DEBUG if verbose_level >= 1 else logging.INFO)
        _logger = logger
        return logger

    level = logging.DEBUG if verbose_level >= 1 else logging.INFO
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    fmt = "%(levelname)s %(asctime)s — %(message)s"
    date_fmt = "%H:%M:%S"
    if sys.stdout.isatty():
        ch.setFormatter(_ColorFormatter(fmt, datefmt=date_fmt))
    else:
        ch.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    logger.addHandler(ch)

    # Optional file handler
    if log_to_file:
        log_dir = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "auraq2")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "auraq2.log"), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        logger.addHandler(fh)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Return the shared logger, initialising it at INFO level if not yet set up."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger
