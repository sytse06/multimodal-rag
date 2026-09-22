"""Typed events emitted by streaming model inference."""

from typing import Literal

from pydantic import BaseModel

from multimodal_rag.models.query import CitedAnswer


class TokenUsage(BaseModel):
    """Provider-normalized token accounting."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reported: bool = True


class InferenceEvent(BaseModel):
    """A progress, completion, failure, or cancellation event."""

    event_type: Literal["progress", "complete", "error", "cancelled"]
    delta: str = ""
    text: str = ""
    provider: str | None = None
    model: str | None = None
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    answer: CitedAnswer | None = None
    article: str | None = None
    error: str | None = None
