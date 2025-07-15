# app/core/logging.py
import logging
import sys
from pythonjsonlogger import json as jsonlogger
from .config import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON log formatter.
    """

    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get("timestamp"):
            log_record["timestamp"] = record.created
        if log_record.get("level"):
            log_record["level"] = log_record["level"].upper()
        else:
            log_record["level"] = record.levelname


def setup_logging():
    """
    Configures logging for the application.
    """
    log = logging.getLogger()
    log.setLevel(settings.LOG_LEVEL.upper())

    # Use a stream handler to output to stdout
    handler = logging.StreamHandler(sys.stdout)

    # Set the custom formatter
    formatter = CustomJsonFormatter("%(timestamp)s %(level)s %(name)s %(message)s")
    handler.setFormatter(formatter)

    # Clear existing handlers and add the new one
    if log.hasHandlers():
        log.handlers.clear()
    log.addHandler(handler)
