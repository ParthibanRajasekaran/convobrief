"""Structured logging configuration using structlog.

Provides JSON-formatted logs with context propagation, PII redaction,
and performance tracking.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from insightsvc.config import get_settings


def add_log_level(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add log level to event dict.

    Args:
        logger: Logger instance.
        method_name: Log method name.
        event_dict: Event dictionary.

    Returns:
        Updated event dictionary with level.
    """
    event_dict["level"] = method_name.upper()
    return event_dict


def censor_pii(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Redact PII from logs if enabled.

    Args:
        logger: Logger instance.
        method_name: Log method name.
        event_dict: Event dictionary.

    Returns:
        Event dictionary with PII redacted.
    """
    settings = get_settings()
    if not settings.redact_pii:
        return event_dict

    # Redact transcript text if configured
    if not settings.log_transcript_text and "transcript" in event_dict:
        event_dict["transcript"] = "[REDACTED]"

    # Hash audio URIs if configured
    if settings.hash_audio_uris and "audio_uri" in event_dict:
        import hashlib

        uri = event_dict["audio_uri"]
        hashed = hashlib.sha256(uri.encode()).hexdigest()[:16]
        event_dict["audio_uri"] = f"[HASHED:{hashed}]"

    return event_dict


def setup_logging() -> None:
    """Configure structured logging for the application.

    Sets up structlog with JSON output, context propagation, and PII redaction.
    """
    settings = get_settings()

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )

    # Build processor chain
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        censor_pii,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured structlog logger.
    """
    return structlog.get_logger(name)
