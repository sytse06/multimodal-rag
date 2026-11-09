"""Tests for shared logging configuration and correlation IDs."""

import inspect
import json
import logging
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from multimodal_rag.logging_config import (
    JsonFormatter,
    configure_logging,
    correlated,
    correlation_scope,
    get_correlation_id,
)
from multimodal_rag.models import AppSettings
from multimodal_rag.query.generator import _truncate_context


def _settings(log_level: str = "INFO", log_format: str = "auto") -> Any:
    return SimpleNamespace(log_level=log_level, log_format=log_format)


def _last_line(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().err.strip().splitlines()[-1]


@pytest.fixture
def clean_root() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield root
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)


class TestJsonFormatter:
    def test_emits_cloud_logging_fields(self) -> None:
        record = logging.LogRecord(
            "app", logging.WARNING, "f.py", 1, "hi %s", ("x",), None
        )
        record.correlation_id = "abc123"

        payload = json.loads(JsonFormatter().format(record))

        assert payload["severity"] == "WARNING"
        assert payload["message"] == "hi x"
        assert payload["logger"] == "app"
        assert payload["correlation_id"] == "abc123"
        assert "time" in payload

    def test_missing_correlation_id_defaults(self) -> None:
        record = logging.LogRecord("app", logging.INFO, "f.py", 1, "hi", (), None)

        assert json.loads(JsonFormatter().format(record))["correlation_id"] == "-"

    def test_includes_traceback_in_message(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                "app", logging.ERROR, "f.py", 1, "failed", (), sys.exc_info()
            )

        message = json.loads(JsonFormatter().format(record))["message"]

        assert message.startswith("failed\n")
        assert "ValueError: boom" in message


class TestCorrelation:
    def test_scope_binds_and_resets(self) -> None:
        assert get_correlation_id() == "-"
        with correlation_scope("run-1") as cid:
            assert cid == "run-1"
            assert get_correlation_id() == "run-1"
        assert get_correlation_id() == "-"

    def test_decorator_keeps_one_id_across_threads(self) -> None:
        seen: list[str] = []

        @correlated
        def handler(count: int) -> Iterator[int]:
            for i in range(count):
                seen.append(get_correlation_id())
                yield i

        events = handler(3)
        results: list[object] = []

        def step() -> None:
            results.append(next(events))

        for _ in range(3):
            worker = threading.Thread(target=step)
            worker.start()
            worker.join()

        assert results == [0, 1, 2]
        assert len(set(seen)) == 1
        assert seen[0] != "-"
        assert get_correlation_id() == "-"

    def test_each_call_gets_its_own_id(self) -> None:
        @correlated
        def handler() -> Iterator[str]:
            yield get_correlation_id()

        assert next(handler()) != next(handler())

    def test_decorator_preserves_generator_introspection(self) -> None:
        def handler(message: str, history: list[str]) -> Iterator[str]:
            yield message

        wrapped = correlated(handler)

        assert inspect.isgeneratorfunction(wrapped)
        assert inspect.signature(wrapped) == inspect.signature(handler)

    def test_closing_the_wrapper_closes_the_handler(self) -> None:
        closed: list[bool] = []

        @correlated
        def handler() -> Iterator[int]:
            try:
                yield 1
                yield 2
            finally:
                closed.append(True)

        events = handler()
        next(events)
        events.close()  # type: ignore[attr-defined]

        assert closed == [True]

    def test_handler_errors_propagate_and_reset_the_id(self) -> None:
        @correlated
        def handler() -> Iterator[int]:
            yield 1
            raise RuntimeError("provider failed")

        events = handler()
        next(events)
        with pytest.raises(RuntimeError, match="provider failed"):
            next(events)
        assert get_correlation_id() == "-"


class TestConfigureLogging:
    def test_auto_is_json_on_cloud_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        clean_root: logging.Logger,
    ) -> None:
        monkeypatch.setenv("K_SERVICE", "multimodal-rag")
        configure_logging(_settings())

        logging.getLogger("t").info("hello")

        payload = json.loads(_last_line(capsys))
        assert payload["severity"] == "INFO"
        assert payload["message"] == "hello"

    def test_auto_is_plain_text_locally(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        clean_root: logging.Logger,
    ) -> None:
        monkeypatch.delenv("K_SERVICE", raising=False)
        configure_logging(_settings())

        logging.getLogger("t").info("hello")

        line = _last_line(capsys)
        assert line.endswith("INFO [-] hello")

    @pytest.mark.parametrize(
        ("log_format", "k_service", "expect_json"),
        [("json", None, True), ("text", "multimodal-rag", False)],
    )
    def test_explicit_format_overrides_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        clean_root: logging.Logger,
        log_format: str,
        k_service: str | None,
        expect_json: bool,
    ) -> None:
        if k_service:
            monkeypatch.setenv("K_SERVICE", k_service)
        else:
            monkeypatch.delenv("K_SERVICE", raising=False)
        configure_logging(_settings(log_format=log_format))

        logging.getLogger("t").info("hello")

        line = _last_line(capsys)
        assert line.startswith("{") is expect_json

    def test_includes_the_active_correlation_id(
        self,
        capsys: pytest.CaptureFixture[str],
        clean_root: logging.Logger,
    ) -> None:
        configure_logging(_settings(log_format="text"))

        with correlation_scope("q-42"):
            logging.getLogger("t").info("retrieving")

        assert "[q-42] retrieving" in _last_line(capsys)

    def test_respects_log_level(
        self,
        capsys: pytest.CaptureFixture[str],
        clean_root: logging.Logger,
    ) -> None:
        configure_logging(_settings(log_level="WARNING", log_format="text"))

        logging.getLogger("t").info("quiet")
        logging.getLogger("t").warning("loud")

        output = capsys.readouterr().err
        assert "quiet" not in output
        assert "loud" in output

    def test_quiets_http_library_chatter_even_at_debug(
        self, clean_root: logging.Logger
    ) -> None:
        configure_logging(_settings(log_level="DEBUG", log_format="text"))

        assert clean_root.level == logging.DEBUG
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING

    def test_drops_google_genai_afc_notices_but_keeps_other_records(
        self,
        capsys: pytest.CaptureFixture[str],
        clean_root: logging.Logger,
    ) -> None:
        configure_logging(_settings(log_format="text"))
        genai = logging.getLogger("google_genai.models")

        genai.info("AFC is enabled with max remote calls: 10.")
        genai.warning(
            "Direct use of automatic function calling (AFC) in Models.generate"
        )
        genai.warning("quota nearly exhausted")
        logging.getLogger("t").warning("AFC is enabled in our own message")

        err = capsys.readouterr().err
        assert "max remote calls" not in err
        assert "automatic function calling" not in err
        assert "quota nearly exhausted" in err
        assert "our own message" in err

    def test_is_idempotent_and_keeps_foreign_handlers(
        self, clean_root: logging.Logger
    ) -> None:
        foreign = logging.NullHandler()
        clean_root.addHandler(foreign)

        configure_logging(_settings())
        configure_logging(_settings())

        ours = [h for h in clean_root.handlers if hasattr(h, "_multimodal_rag")]
        assert len(ours) == 1
        assert foreign in clean_root.handlers


class TestLogFormatSetting:
    def _make(self, **values: str) -> AppSettings:
        return AppSettings(
            _env_file=None,
            llm_provider="ollama",
            embedding_provider="ollama",
            **values,
        )

    def test_default_is_auto(self) -> None:
        assert self._make().log_format == "auto"

    def test_value_is_normalised(self) -> None:
        assert self._make(log_format=" JSON ").log_format == "json"

    def test_unknown_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="log_format"):
            self._make(log_format="yaml")


class TestTruncationLogging:
    def test_logs_when_context_is_cut(self, caplog: pytest.LogCaptureFixture) -> None:
        context = "a" * 100 + "\n\n" + "b" * 100

        with caplog.at_level(logging.INFO, logger="multimodal_rag.query.generator"):
            result = _truncate_context(context, 10)

        assert "omitted" in result
        messages = [r.getMessage() for r in caplog.records]
        assert any("Context truncated" in m and "budget 10" in m for m in messages)

    def test_silent_when_context_fits(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="multimodal_rag.query.generator"):
            assert _truncate_context("short", 100) == "short"

        assert not any("truncated" in r.getMessage() for r in caplog.records)
