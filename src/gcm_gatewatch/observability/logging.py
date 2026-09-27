"""Structured logging setup.

v1 used bare `print`. For a process that runs unattended for weeks, logs need to be
parseable and queryable. structlog emits JSON (configurable) so events can be shipped
and searched. Call `configure()` once at startup.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure(*, level: str = "INFO", json: bool = True) -> None:
    """Configure structlog for the process.

    JSON is the default because the deployed process writes to journald or to a file that
    something else has to read. Pass json=False for the console renderer when working
    locally, where a person is the reader.

    Safe to call more than once; the last call wins.
    """
    numeric_level = logging.getLevelNamesMapping().get(level.upper())
    if numeric_level is None:
        raise ValueError(f"unknown log level {level!r}")

    # stdlib logging underneath, so uvicorn's and ultralytics' own loggers land in the
    # same stream at the same level instead of escaping to their own default handlers.
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=numeric_level, force=True)

    renderer = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # UTC ISO-8601, so timestamps from the gate box compare cleanly with anything
            # else no matter how the machine's clock is configured
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,  # so a later configure() call actually takes
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for a module."""
    return structlog.get_logger(name)
