"""One structured log record per generation, without prompt or answer text."""

import functools
import inspect
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any

from multimodal_rag.models.inference import ContextStats, InferenceEvent

logger = logging.getLogger(__name__)

_TERMINAL = {"complete", "error", "cancelled"}

StatsFn = Callable[[Mapping[str, Any]], ContextStats]


def estimate_context_stats(
    context: str, chunks: int, budget_tokens: int | None
) -> ContextStats:
    """Describe how much of the budget a context uses (4 characters ~ 1 token)."""
    tokens = len(context) // 4
    truncated = budget_tokens is not None and tokens > budget_tokens
    used = min(tokens, budget_tokens) if budget_tokens is not None else tokens
    return ContextStats(
        chunks=chunks,
        context_tokens=used,
        budget_tokens=budget_tokens,
        truncated=truncated,
    )


def log_inference(kind: str, event: InferenceEvent, context: ContextStats) -> None:
    """Log the terminal event of a generation. Never includes text or error detail."""
    usage = event.usage
    fields: dict[str, Any] = {
        "event": "inference",
        "kind": kind,
        "provider": event.provider,
        "model": event.model,
        "status": event.event_type,
        "finish_reason": event.finish_reason,
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "usage_reported": usage is not None,
        "time_to_first_token_ms": _round(event.time_to_first_token_ms),
        "latency_ms": _round(event.elapsed_ms),
        "chunks": context.chunks,
        "context_tokens": context.context_tokens,
        "context_budget_tokens": context.budget_tokens,
        "context_truncated": context.truncated,
    }
    level = logging.WARNING if event.event_type == "error" else logging.INFO
    logger.log(
        level,
        "Inference %s kind=%s provider=%s model=%s status=%s latency_ms=%s "
        "tokens=%s ttft_ms=%s",
        event.event_type,
        kind,
        event.provider,
        event.model,
        event.event_type,
        fields["latency_ms"],
        fields["total_tokens"] if usage else "unreported",
        fields["time_to_first_token_ms"],
        extra={"fields": fields},
    )


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def _is_generation(event: InferenceEvent) -> bool:
    # The "no sources" short-circuit completes without calling a model.
    return event.event_type in _TERMINAL and event.elapsed_ms is not None


def logged_stream(
    kind: str, stats: StatsFn
) -> Callable[
    [Callable[..., Iterator[InferenceEvent]]], Callable[..., Iterator[InferenceEvent]]
]:
    """Log one record when a sync inference stream reaches its terminal event."""

    def decorate(
        fn: Callable[..., Iterator[InferenceEvent]],
    ) -> Callable[..., Iterator[InferenceEvent]]:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Iterator[InferenceEvent]:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            for event in fn(*args, **kwargs):
                if _is_generation(event):
                    log_inference(kind, event, stats(bound.arguments))
                yield event

        return wrapper

    return decorate


def logged_astream(
    kind: str, stats: StatsFn
) -> Callable[
    [Callable[..., AsyncIterator[InferenceEvent]]],
    Callable[..., AsyncIterator[InferenceEvent]],
]:
    """Async twin of :func:`logged_stream`."""

    def decorate(
        fn: Callable[..., AsyncIterator[InferenceEvent]],
    ) -> Callable[..., AsyncIterator[InferenceEvent]]:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[InferenceEvent]:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            async for event in fn(*args, **kwargs):
                if _is_generation(event):
                    log_inference(kind, event, stats(bound.arguments))
                yield event

        return wrapper

    return decorate
