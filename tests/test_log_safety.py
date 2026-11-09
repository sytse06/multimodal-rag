"""Tests that logs never carry secrets or user query text."""

import json
import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from multimodal_rag.logging_config import (
    REDACTED,
    configure_logging,
    redact,
    text_fingerprint,
)
from multimodal_rag.models import AppSettings
from multimodal_rag.models.chunks import SourceType
from multimodal_rag.models.query import SearchResult
from multimodal_rag.query.generator import stream_cited_answer
from multimodal_rag.query.retriever import retrieve

QUESTION = "customer Jansen cannot reset password secret-question-marker"

# One realistic leak per provider error path.
PROVIDER_LEAKS = {
    "openai": "401 Incorrect API key provided: sk-proj-AbCdEf1234567890XyZ",
    "openrouter": "403 {'error': 'key sk-or-v1-0123456789abcdef0123456789abcdef'}",
    "gemini": (
        "400 for url https://generativelanguage.googleapis.com/v1beta/models/"
        "m:streamGenerateContent?key=AIzaSyA1234567890abcdefghijklmnopqrstu"
    ),
    "nvidia": "401 Authorization: Bearer nvapi-AbCdEf1234567890XyZ",
    "ollama": "connection failed to http://admin:hunter2pass@ollama.internal:11434",
}


@pytest.fixture
def clean_root() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield root
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)


def _settings(log_format: str = "text", log_level: str = "INFO") -> Any:
    return SimpleNamespace(log_level=log_level, log_format=log_format)


def _result() -> SearchResult:
    return SearchResult(
        text="Click File > New",
        source_type=SourceType.VIDEO,
        source_url="https://yt.com/watch?v=abc",
        source_name="Quickstart",
        timestamp_seconds=42,
        relevance_score=0.9,
    )


class TestRedact:
    @pytest.mark.parametrize(
        ("text", "leaked"),
        [
            ("Authorization: Bearer abc123def456ghi", "abc123def456ghi"),
            ("failed https://user:p4ssw0rd@host/db", "p4ssw0rd"),
            ("api_key=supersecretvalue1", "supersecretvalue1"),
            ('{"api_key": "supersecretvalue1"}', "supersecretvalue1"),
            ('\\"token\\": \\"supersecretvalue1\\"', "supersecretvalue1"),
            ("sk-proj-AbCdEf1234567890", "AbCdEf1234567890"),
            ("AIzaSyA1234567890abcdefghijklmnopqrstu", "AIzaSyA1234567890"),
            ("nvapi-AbCdEf1234567890", "AbCdEf1234567890"),
            ("url?key=AIzaSyA1234567890abcdefghijklmnopqrstu&alt=sse", "AIzaSyA"),
        ],
    )
    def test_masks_known_credential_shapes(self, text: str, leaked: str) -> None:
        result = redact(text)

        assert leaked not in result
        assert REDACTED in result

    def test_masks_configured_secret_values_verbatim(self) -> None:
        assert redact("oops custom-format-secret-77", ("custom-format-secret-77",)) == (
            f"oops {REDACTED}"
        )

    def test_leaves_ordinary_messages_and_metrics_alone(self) -> None:
        text = "Inference complete total_tokens=35 tokens=unreported ttft_ms=12.5"

        assert redact(text) == text

    def test_fingerprint_is_stable_and_not_the_text(self) -> None:
        assert text_fingerprint("abc") == text_fingerprint("abc")
        assert text_fingerprint("abc") != text_fingerprint("abd")
        assert "abc" not in text_fingerprint("abc")
        assert len(text_fingerprint("abc")) == 8


class TestConfiguredHandler:
    @pytest.mark.parametrize("log_format", ["text", "json"])
    def test_exception_traceback_and_message_are_redacted(
        self,
        capsys: pytest.CaptureFixture[str],
        clean_root: logging.Logger,
        log_format: str,
    ) -> None:
        configure_logging(_settings(log_format))

        try:
            raise RuntimeError("bad key: sk-proj-AbCdEf1234567890XyZ")
        except RuntimeError as exc:
            logging.getLogger("t").exception("failed: %s", exc)

        err = capsys.readouterr().err
        assert "AbCdEf1234567890" not in err
        assert REDACTED in err
        if log_format == "json":
            json.loads(err.strip().splitlines()[-1])  # still valid JSON

    def test_secrets_from_settings_are_masked_by_exact_value(
        self, capsys: pytest.CaptureFixture[str], clean_root: logging.Logger
    ) -> None:
        settings = AppSettings(
            _env_file=None,
            llm_provider="ollama",
            embedding_provider="ollama",
            firecrawl_api_key=SecretStr("fc-odd-format-credential"),
            weaviate_api_key=SecretStr("wv-odd-format-credential"),
        )
        configure_logging(settings)

        logging.getLogger("t").error(
            "calls failed with fc-odd-format-credential and wv-odd-format-credential"
        )

        err = capsys.readouterr().err
        assert "odd-format-credential" not in err

    def test_short_secrets_are_not_used_for_masking(
        self, capsys: pytest.CaptureFixture[str], clean_root: logging.Logger
    ) -> None:
        settings = AppSettings(
            _env_file=None,
            llm_provider="ollama",
            embedding_provider="ollama",
            firecrawl_api_key=SecretStr("a"),
        )
        configure_logging(settings)

        logging.getLogger("t").error("a banana")

        assert "a banana" in capsys.readouterr().err


class TestQueryTextPolicy:
    def _store(self) -> MagicMock:
        store = MagicMock()
        store.search.return_value = []
        return store

    def test_info_log_has_length_and_hash_but_not_the_query(
        self, capsys: pytest.CaptureFixture[str], clean_root: logging.Logger
    ) -> None:
        configure_logging(_settings(log_level="INFO"))

        retrieve(QUESTION, self._store(), top_k=5)

        err = capsys.readouterr().err
        assert "secret-question-marker" not in err
        assert "Jansen" not in err
        assert f"query_len={len(QUESTION)}" in err
        assert f"query_hash={text_fingerprint(QUESTION)}" in err

    def test_full_text_only_appears_at_debug(
        self, capsys: pytest.CaptureFixture[str], clean_root: logging.Logger
    ) -> None:
        configure_logging(_settings(log_level="DEBUG"))

        retrieve(QUESTION, self._store(), top_k=5)

        assert "secret-question-marker" in capsys.readouterr().err


class TestProviderErrorPaths:
    @pytest.mark.parametrize("provider", sorted(PROVIDER_LEAKS))
    def test_stream_error_never_logs_the_secret(
        self,
        capsys: pytest.CaptureFixture[str],
        clean_root: logging.Logger,
        provider: str,
    ) -> None:
        configure_logging(_settings("json"))
        llm = MagicMock()
        llm.stream.side_effect = RuntimeError(PROVIDER_LEAKS[provider])

        events = list(
            stream_cited_answer(
                QUESTION, [_result()], llm, provider=provider, model="m"
            )
        )

        err = capsys.readouterr().err
        for leaked in (
            "sk-proj-AbCdEf",
            "0123456789abcdef0123",
            "AIzaSyA1234567890",
            "nvapi-AbCdEf",
            "hunter2pass",
        ):
            assert leaked not in err
        assert "secret-question-marker" not in err
        assert events[-1].event_type == "error"
        assert events[-1].error is not None
        for line in err.strip().splitlines():
            json.loads(line)
