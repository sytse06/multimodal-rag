"""Tests for the one-record-per-generation inference log."""

import json
import logging
from threading import Event
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessageChunk

from multimodal_rag.logging_config import JsonFormatter
from multimodal_rag.models.chunks import SourceType
from multimodal_rag.models.query import Citation, CitedAnswer, SearchResult
from multimodal_rag.query.generator import (
    astream_cited_answer,
    stream_cited_answer,
    stream_kb_article,
)
from multimodal_rag.query.metrics import estimate_context_stats

QUESTION = "How do I reset my unique-question-marker password?"
ANSWER_TEXT = "unique-answer-marker"


def _result() -> SearchResult:
    return SearchResult(
        text="Click File > New to create a project",
        source_type=SourceType.VIDEO,
        source_url="https://yt.com/watch?v=abc",
        source_name="Quickstart",
        timestamp_seconds=42,
        relevance_score=0.9,
    )


def _inference_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if getattr(r, "fields", {}).get("event") == "inference"
    ]


def _fields(record: logging.LogRecord) -> dict[str, Any]:
    return record.fields  # type: ignore[attr-defined, no-any-return]


@pytest.fixture(autouse=True)
def _capture(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="multimodal_rag.query.metrics")


def _llm(*chunks: AIMessageChunk) -> MagicMock:
    llm = MagicMock()
    llm.stream.return_value = list(chunks)
    return llm


class TestAnswerRecord:
    def test_one_record_for_a_completed_generation(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = _llm(
            AIMessageChunk(content=ANSWER_TEXT),
            AIMessageChunk(
                content=" [1].",
                usage_metadata={
                    "input_tokens": 30,
                    "output_tokens": 5,
                    "total_tokens": 35,
                },
                response_metadata={"finish_reason": "stop"},
            ),
        )

        list(
            stream_cited_answer(
                QUESTION, [_result()], llm, provider="gemini", model="m1"
            )
        )

        (record,) = _inference_records(caplog)
        fields = _fields(record)
        assert fields["kind"] == "answer"
        assert fields["provider"] == "gemini"
        assert fields["model"] == "m1"
        assert fields["status"] == "complete"
        assert fields["finish_reason"] == "stop"
        assert (fields["input_tokens"], fields["output_tokens"]) == (30, 5)
        assert fields["total_tokens"] == 35
        assert fields["usage_reported"] is True
        assert fields["latency_ms"] is not None
        assert fields["time_to_first_token_ms"] is not None
        assert fields["chunks"] == 1
        assert fields["context_budget_tokens"] == 10000
        assert fields["context_truncated"] is False

    def test_record_never_contains_question_or_answer_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = _llm(AIMessageChunk(content=ANSWER_TEXT))

        list(stream_cited_answer(QUESTION, [_result()], llm))

        (record,) = _inference_records(caplog)
        rendered = record.getMessage() + json.dumps(_fields(record), default=str)
        assert "unique-question-marker" not in rendered
        assert ANSWER_TEXT not in rendered
        assert "Click File" not in rendered

    def test_unreported_usage_is_explicit(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = _llm(AIMessageChunk(content="x"))

        list(stream_cited_answer(QUESTION, [_result()], llm))

        (record,) = _inference_records(caplog)
        fields = _fields(record)
        assert fields["usage_reported"] is False
        assert fields["total_tokens"] is None
        assert "tokens=unreported" in record.getMessage()

    def test_error_is_logged_as_a_warning_without_detail(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = MagicMock()
        llm.stream.side_effect = RuntimeError("sk-secret-token")

        list(stream_cited_answer(QUESTION, [_result()], llm, provider="p", model="m"))

        (record,) = _inference_records(caplog)
        assert record.levelno == logging.WARNING
        assert _fields(record)["status"] == "error"
        assert "sk-secret-token" not in record.getMessage()
        assert "sk-secret-token" not in json.dumps(_fields(record), default=str)

    def test_cancellation_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        cancel = Event()
        cancel.set()

        list(
            stream_cited_answer(
                QUESTION,
                [_result()],
                _llm(AIMessageChunk(content="partial")),
                cancel_event=cancel,
            )
        )

        (record,) = _inference_records(caplog)
        assert _fields(record)["status"] == "cancelled"

    def test_no_sources_short_circuit_is_not_a_generation(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = MagicMock()

        list(stream_cited_answer(QUESTION, [], llm))

        assert _inference_records(caplog) == []
        llm.stream.assert_not_called()

    def test_truncation_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        big = _result().model_copy(update={"text": "word " * 2000})

        list(
            stream_cited_answer(
                QUESTION,
                [big],
                _llm(AIMessageChunk(content="x")),
                max_context_tokens=100,
            )
        )

        (record,) = _inference_records(caplog)
        fields = _fields(record)
        assert fields["context_truncated"] is True
        assert fields["context_tokens"] == 100
        assert fields["context_budget_tokens"] == 100

    @pytest.mark.anyio
    async def test_async_stream_logs_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = MagicMock()

        async def chunks() -> Any:
            yield AIMessageChunk(content="Use [1].")

        llm.astream.return_value = chunks()

        async for _ in astream_cited_answer(
            QUESTION, [_result()], llm, provider="p", model="m"
        ):
            pass

        (record,) = _inference_records(caplog)
        assert _fields(record)["status"] == "complete"
        assert _fields(record)["kind"] == "answer"


class TestArticleRecord:
    def test_article_stream_logs_kind_and_chunk_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        answer = CitedAnswer(
            answer="a",
            citations=[
                Citation(
                    label="L",
                    url="https://x",
                    source_type=SourceType.WEB,
                    relevance_score=0.5,
                )
            ],
        )

        list(
            stream_kb_article(
                answer,
                _llm(AIMessageChunk(content="# Title")),
                [_result(), _result()],
                QUESTION,
                provider="p",
                model="m",
            )
        )

        (record,) = _inference_records(caplog)
        assert _fields(record)["kind"] == "kb_article"
        assert _fields(record)["chunks"] == 2


class TestEstimateContextStats:
    def test_within_budget(self) -> None:
        stats = estimate_context_stats("a" * 400, 3, 1000)
        assert (stats.context_tokens, stats.truncated) == (100, False)

    def test_over_budget_caps_used_tokens(self) -> None:
        stats = estimate_context_stats("a" * 4000, 3, 100)
        assert (stats.context_tokens, stats.truncated) == (100, True)

    def test_no_budget_is_never_truncated(self) -> None:
        stats = estimate_context_stats("a" * 4000, 3, None)
        assert (stats.context_tokens, stats.truncated) == (1000, False)


class TestJsonFields:
    def test_extra_fields_become_top_level_keys(self) -> None:
        record = logging.LogRecord("app", logging.INFO, "f.py", 1, "hi", (), None)
        record.fields = {"model": "m1", "latency_ms": 12.5}  # type: ignore[attr-defined]

        payload = json.loads(JsonFormatter().format(record))

        assert payload["model"] == "m1"
        assert payload["latency_ms"] == 12.5
        assert payload["severity"] == "INFO"

    def test_fields_cannot_overwrite_core_keys(self) -> None:
        record = logging.LogRecord("app", logging.INFO, "f.py", 1, "hi", (), None)
        record.fields = {"severity": "DEBUG", "event": "inference"}  # type: ignore[attr-defined]

        payload = json.loads(JsonFormatter().format(record))

        assert payload["severity"] == "INFO"
        assert payload["event"] == "inference"
