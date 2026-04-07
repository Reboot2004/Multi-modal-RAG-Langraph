import logging
import sys

from config.settings import PIPELINE_DEBUG


_LOGGER_NAME = "indic_rag"


def get_logger(name: str = None) -> logging.Logger:
    logger_name = _LOGGER_NAME if not name else f"{_LOGGER_NAME}.{name}"
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    logger.setLevel(logging.DEBUG if PIPELINE_DEBUG else logging.INFO)
    return logger


def set_debug_enabled(enabled: bool):
    root = logging.getLogger(_LOGGER_NAME)
    root.setLevel(logging.DEBUG if enabled else logging.INFO)

    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith(f"{_LOGGER_NAME}."):
            logging.getLogger(name).setLevel(logging.DEBUG if enabled else logging.INFO)