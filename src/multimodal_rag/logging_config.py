"""Shared logging setup: plain text locally, JSON on Cloud Run, correlation IDs."""

import functools
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import cast

from pydantic import BaseModel, SecretStr

from multimodal_rag.models.config import AppSettings

TEXT_FORMAT = "%(asctime)s %(name)s %(levelname)s [%(correlation_id)s] %(message)s"

# Per-request chatter from HTTP stacks drowns our own records at DEBUG/INFO.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "hpack", "asyncio", "PIL")

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")

REDACTED = "***"
_MIN_SECRET_LENGTH = 8

# Shapes of credentials that can end up in exception text. Configured secrets are
# also masked by exact value (see ``_collect_secrets``), which covers formats
# not listed here.
_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\bbearer\s+[\w.~+/=-]{8,}"), f"Bearer {REDACTED}"),
    (re.compile(r"(?<=://)[^/\s:@\"']+:[^/\s@\"']+@"), f"{REDACTED}@"),
    (
        re.compile(
            r"(?i)(\b(?:api[_-]?key|key|access[_-]?token|token|secret|password"
            r"|authorization)\b[\\\"']*\s*[:=]\s*[\\\"']*)[^\s\\\"',;&)}\]]+"
        ),
        rf"\1{REDACTED}",
    ),
    (
        re.compile(
            r"\b(?:sk-[\w-]{8,}|AIza[\w-]{20,}|nvapi-[\w-]{8,}|fc-[\w]{16,}"
            r"|hf_[A-Za-z0-9]{20,})"
        ),
        REDACTED,
    ),
)


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


def text_fingerprint(text: str) -> str:
    """Short stable hash so repeated questions can be correlated without the text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def redact(text: str, secrets: tuple[str, ...] = ()) -> str:
    """Mask credentials in free text: configured secrets first, then known shapes."""
    for secret in secrets:
        text = text.replace(secret, REDACTED)
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _collect_secrets(model: object) -> tuple[str, ...]:
    """Collect non-trivial ``SecretStr`` values from a (nested) settings model."""
    found: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, SecretStr):
            secret = value.get_secret_value()
            if len(secret) >= _MIN_SECRET_LENGTH:
                found.add(secret)
        elif isinstance(value, BaseModel):
            for name in type(value).model_fields:
                walk(getattr(value, name, None))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item)

    walk(model)
    # Longest first so a secret containing another is masked whole.
    return tuple(sorted(found, key=len, reverse=True))


class _RedactingFormatter(logging.Formatter):
    """Mask credentials in the final line, including formatted tracebacks."""

    def __init__(self, inner: logging.Formatter, secrets: tuple[str, ...]) -> None:
        super().__init__()
        self._inner = inner
        self._secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        return redact(self._inner.format(record), self._secrets)


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
        payload = {**(getattr(record, "fields", None) or {}), **payload}
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
    formatter: logging.Formatter = (
        JsonFormatter()
        if _use_json(settings.log_format)
        else logging.Formatter(TEXT_FORMAT)
    )
    handler.setFormatter(_RedactingFormatter(formatter, _collect_secrets(settings)))
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
