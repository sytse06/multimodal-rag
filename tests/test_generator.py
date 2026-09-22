"""Tests for cited answer generation."""

from threading import Event
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from multimodal_rag.models.chunks import SourceType
from multimodal_rag.models.query import Citation, CitedAnswer, SearchResult
from multimodal_rag.query.generator import (
    KB_ARTICLE_PROMPT,
    SYSTEM_PROMPT,
    _build_citations,
    _content_to_text,
    _format_context_budgeted,
    _replace_refs_with_links,
    _strip_code_fence,
    astream_cited_answer,
    generate_cited_answer,
    generate_kb_article,
    stream_cited_answer,
    stream_kb_article,
)


def _video_result(score: float = 0.9) -> SearchResult:
    return SearchResult(
        text="Click File > New to create a project",
        source_type=SourceType.VIDEO,
        source_url="https://yt.com/watch?v=abc",
        source_name="Quickstart",
        timestamp_seconds=42,
        relevance_score=score,
    )


def _web_result(score: float = 0.8) -> SearchResult:
    return SearchResult(
        text="Install via pip install hydrosym",
        source_type=SourceType.WEB,
        source_url="https://docs.example.com/install",
        source_name="Docs",
        section_heading="Installation",
        relevance_score=score,
    )


class TestBuildCitations:
    def test_video_citation(self) -> None:
        citations = _build_citations([_video_result()])
        assert len(citations) == 1
        assert citations[0].source_type == SourceType.VIDEO
        assert "&t=42s" in citations[0].url
        assert "00:42" in citations[0].label

    def test_short_url_uses_question_mark(self) -> None:
        result = SearchResult(
            text="clip",
            source_type=SourceType.VIDEO,
            source_url="https://youtu.be/abc123",
            source_name="Short",
            timestamp_seconds=60,
            relevance_score=0.9,
        )
        citations = _build_citations([result])
        assert citations[0].url == "https://youtu.be/abc123?t=60s"

    def test_web_citation(self) -> None:
        citations = _build_citations([_web_result()])
        assert citations[0].source_type == SourceType.WEB
        assert citations[0].url == "https://docs.example.com/install"
        assert "Installation" in citations[0].label

    def test_preserves_order(self) -> None:
        citations = _build_citations([_video_result(), _web_result()])
        assert len(citations) == 2
        assert citations[0].source_type == SourceType.VIDEO
        assert citations[1].source_type == SourceType.WEB


class TestReplaceRefsWithLinks:
    def test_replaces_numbered_refs(self) -> None:
        results = [_video_result(0.9), _web_result(0.8)]
        text = "See [1] for video and [2] for docs."
        replaced = _replace_refs_with_links(text, results)
        assert "[1]" not in replaced
        assert "[2]" not in replaced
        assert "Quickstart @ 00:42" in replaced
        assert "(90%)" in replaced
        assert "Docs" in replaced
        assert "(80%)" in replaced

    def test_no_refs_unchanged(self) -> None:
        text = "No references here."
        assert _replace_refs_with_links(text, []) == text

    def test_unmatched_ref_kept(self) -> None:
        text = "See [1] and [99]."
        replaced = _replace_refs_with_links(text, [_video_result()])
        assert "[99]" in replaced
        assert "[1]" not in replaced


class TestContentToText:
    def test_plain_text_is_unchanged(self) -> None:
        assert _content_to_text("Answer.") == "Answer."

    def test_extracts_text_from_structured_blocks(self) -> None:
        content = [
            {
                "type": "text",
                "text": "First part. ",
                "extras": {"signature": "ignored"},
            },
            {"type": "text", "text": "Second part."},
        ]
        assert _content_to_text(content) == "First part. Second part."

    def test_ignores_non_text_blocks(self) -> None:
        content = [
            {"type": "image", "data": "..."},
            {"type": "text", "text": "Answer."},
        ]
        assert _content_to_text(content) == "Answer."

    def test_context_budget_omits_tail_predictably(self) -> None:
        results = [
            _video_result(),
            SearchResult(
                text="A very long source that should be omitted from the tail.",
                source_type=SourceType.WEB,
                source_url="https://docs.example.com/long",
                source_name="Long source",
                relevance_score=0.7,
            ),
        ]

        context = _format_context_budgeted(results, max_context_tokens=8)

        assert "Additional source context omitted" in context


class TestGenerateCitedAnswer:
    def test_empty_results_returns_fallback(self) -> None:
        mock_llm = MagicMock()
        result = generate_cited_answer("test?", [], llm=mock_llm)
        assert "couldn't find" in result.answer.lower()
        assert result.citations == []
        mock_llm.invoke.assert_not_called()

    def test_calls_llm_and_returns_answer(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="Use [1] to get started."
        )

        results = [_video_result()]
        answer = generate_cited_answer(
            "How do I start?", results, llm=mock_llm
        )

        assert "Quickstart @ 00:42" in answer.answer
        assert len(answer.citations) == 1
        mock_llm.invoke.assert_called_once()

    def test_normalizes_structured_provider_content(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content=[
                {
                    "type": "text",
                    "text": "Use [1] to zoom.",
                    "extras": {"signature": "provider metadata"},
                }
            ]
        )

        answer = generate_cited_answer(
            "How do I zoom?", [_video_result()], llm=mock_llm
        )

        assert "Use " in answer.answer
        assert "provider metadata" not in answer.answer
        assert "'type': 'text'" not in answer.answer

    def test_streams_progress_and_final_citations(self) -> None:
        mock_llm = MagicMock()
        mock_llm.stream.return_value = [
            AIMessageChunk(content="Use "),
            AIMessageChunk(
                content="[1].",
                usage_metadata={
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                },
                response_metadata={"finish_reason": "stop"},
            ),
        ]

        events = list(
            stream_cited_answer(
                "How do I start?",
                [_video_result()],
                mock_llm,
                provider="ollama",
                model="test-model",
            )
        )

        assert [event.event_type for event in events] == [
            "progress",
            "progress",
            "complete",
        ]
        assert events[1].text == "Use [1]."
        assert events[-1].answer is not None
        assert "Quickstart @ 00:42" in events[-1].text
        assert events[-1].usage is not None
        assert events[-1].usage.total_tokens == 5
        assert events[-1].finish_reason == "stop"
        assert events[-1].elapsed_ms is not None
        assert events[-1].time_to_first_token_ms is not None
        assert events[-1].time_to_first_token_ms <= events[-1].elapsed_ms

    def test_stream_error_is_safe(self) -> None:
        mock_llm = MagicMock()
        mock_llm.stream.side_effect = RuntimeError("provider secret must not leak")

        events = list(stream_cited_answer("question", [_video_result()], mock_llm))

        assert events[-1].event_type == "error"
        assert events[-1].error is not None
        assert "provider secret" not in events[-1].error

    def test_stream_cancellation_does_not_complete(self) -> None:
        mock_llm = MagicMock()
        mock_llm.stream.return_value = [
            AIMessageChunk(content="Partial answer."),
            AIMessageChunk(content="More text."),
        ]
        cancel_event = Event()
        cancel_event.set()

        events = list(
            stream_cited_answer(
                "question", [_video_result()], mock_llm, cancel_event=cancel_event
            )
        )

        assert [event.event_type for event in events] == ["cancelled"]
        assert events[-1].answer is None

    def test_passes_context_to_llm(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Answer.")

        results = [_video_result(), _web_result()]
        generate_cited_answer("question?", results, llm=mock_llm)

        messages = mock_llm.invoke.call_args[0][0]
        assert messages[0].content == SYSTEM_PROMPT
        assert "## Sources" in messages[1].content
        assert "## Question" in messages[1].content


class TestGenerateKbArticle:
    def _answer(self) -> CitedAnswer:
        return CitedAnswer(
            answer="Use File > New to create a project. See [1].",
            citations=[
                Citation(
                    label="Quickstart @ 00:42",
                    url="https://yt.com/watch?v=abc&t=42s",
                    relevance_score=0.9,
                    source_type=SourceType.VIDEO,
                )
            ],
        )

    def test_calls_llm(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="# Article\n\nBody text.")
        generate_kb_article(self._answer(), llm=mock_llm)
        mock_llm.invoke.assert_called_once()

    def test_uses_kb_article_prompt(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Draft.")
        generate_kb_article(self._answer(), llm=mock_llm)
        messages = mock_llm.invoke.call_args[0][0]
        assert messages[0].content == KB_ARTICLE_PROMPT

    def test_includes_citation_label_in_user_message(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Draft.")
        generate_kb_article(self._answer(), llm=mock_llm, results=[_video_result()])
        messages = mock_llm.invoke.call_args[0][0]
        assert "Quickstart @ 00:42" in messages[1].content

    def test_source_chunk_text_included_when_results_provided(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Draft.")
        generate_kb_article(self._answer(), llm=mock_llm, results=[_video_result()])
        messages = mock_llm.invoke.call_args[0][0]
        assert "Click File > New to create a project" in messages[1].content

    def test_returns_llm_content(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="# KB Article\n\nDetails.")
        result = generate_kb_article(self._answer(), llm=mock_llm)
        assert result.startswith("# KB Article\n\nDetails.")
        assert "## Sources" in result
        assert "https://yt.com/watch?v=abc&t=42s" in result

    def test_no_citations(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Draft.")
        answer = CitedAnswer(answer="Simple answer.", citations=[])
        result = generate_kb_article(answer, llm=mock_llm)
        assert result == "Draft."
        mock_llm.invoke.assert_called_once()

    def test_strips_code_fence_from_output(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="```markdown\n# Article\n\nBody.\n```"
        )
        result = generate_kb_article(self._answer(), llm=mock_llm)
        assert "```" not in result
        assert "# Article" in result
        assert "## Sources" in result

    def test_question_included_in_user_message(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Draft.")
        generate_kb_article(self._answer(), llm=mock_llm, question="How do I export?")
        messages = mock_llm.invoke.call_args[0][0]
        assert "## Question" in messages[1].content
        assert "How do I export?" in messages[1].content

    def test_sources_appended_programmatically(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="# Article\n\nBody.")
        result = generate_kb_article(self._answer(), llm=mock_llm)
        assert "## Sources" in result
        assert "https://yt.com/watch?v=abc&t=42s" in result

    def test_streams_article_and_appends_sources_on_completion(self) -> None:
        mock_llm = MagicMock()
        mock_llm.stream.return_value = [
            AIMessageChunk(content="# Article\n\nBody."),
        ]

        events = list(stream_kb_article(self._answer(), mock_llm))

        assert events[-1].event_type == "complete"
        assert events[-1].article is not None
        assert "## Sources" in events[-1].article

    @pytest.mark.anyio
    async def test_async_stream_matches_sync_contract(self) -> None:
        mock_llm = MagicMock()

        async def chunks() -> object:
            yield AIMessageChunk(content="Use [1].")

        mock_llm.astream.return_value = chunks()
        events = [
            event
            async for event in astream_cited_answer(
                "question", [_video_result()], mock_llm
            )
        ]

        assert events[-1].event_type == "complete"
        assert events[-1].answer is not None


class TestStripCodeFence:
    def test_strips_markdown_fence(self) -> None:
        assert _strip_code_fence("```markdown\n# Hi\n```") == "# Hi"

    def test_strips_plain_fence(self) -> None:
        assert _strip_code_fence("```\n# Hi\n```") == "# Hi"

    def test_no_fence_unchanged(self) -> None:
        assert _strip_code_fence("# Hi\n\nBody.") == "# Hi\n\nBody."

    def test_strips_surrounding_whitespace(self) -> None:
        assert _strip_code_fence("  ```markdown\n# Hi\n```  ") == "# Hi"
