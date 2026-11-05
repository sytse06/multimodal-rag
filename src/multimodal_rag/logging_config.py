"""Shared logging setup: plain text locally, JSON on Cloud Run, correlation IDs."""

import functools
import json
import logging
import os
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import cast

from multimodal_rag.models.config import AppSettings

TEXT_FORMAT = "%(asctime)s %(name)s %(levelname)s [%(correlation_id)s] %(message)s"

# Per-request chatter from HTTP stacks drowns our own records at DEBUG/INFO.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "hpack", "asyncio", "PIL")

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")


def get_correlation_id() -> str:
    return _correlation_id.get()


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


@contextmanager
def correlation_scope(correlation_id: str | None = None) -> Iterator[str]:
    """Bind a correlation ID for the duration of a block of plain code."""
    cid = correlation_id or new_correlation_id()
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


def correlated(fn: Callable[..., object]) -> Callable[..., Iterator[object]]:
    """Give each call of a generator handler its own correlation ID.

    Gradio may advance a generator from different worker threads, and each hop runs
    in a copied context, so the ID is re-bound around every ``next()`` instead of
    being set once.
    """

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> Iterator[object]:
        cid = new_correlation_id()
        events = cast(Iterator[object], fn(*args, **kwargs))
        try:
            while True:
                token = _correlation_id.set(cid)
                try:
                    item = next(events)
                except StopIteration:
                    return
                finally:
                    _correlation_id.reset(token)
                yield item
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()

    return wrapper


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line with the fields Cloud Logging understands."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        payload = {
            "severity": record.levelname,
            "message": message,
            "logger": record.name,
            "correlation_id": getattr(record, "correlation_id", "-"),
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)


def _use_json(log_format: str) -> bool:
    if log_format == "json":
        return True
    if log_format == "text":
        return False
    return bool(os.environ.get("K_SERVICE"))  # set by Cloud Run


def configure_logging(settings: AppSettings) -> None:
    """Configure the root logger once; safe to call again (replaces its own handler)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_CorrelationFilter())
    handler.setFormatter(
        JsonFormatter()
        if _use_json(settings.log_format)
        else logging.Formatter(TEXT_FORMAT)
    )
    handler._multimodal_rag = True  # type: ignore[attr-defined]

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, "_multimodal_rag", False):
            root.removeHandler(existing)
    root.addHandler(handler)
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
