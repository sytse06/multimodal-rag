"""Tests for cited answer generation."""

from unittest.mock import MagicMock, patch

from multimodal_rag.models.chunks import SourceType
from multimodal_rag.models.query import SearchResult
from multimodal_rag.query.generator import (
    _build_citations,
    _replace_refs_with_links,
    generate_cited_answer,
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


class TestGenerateCitedAnswer:
    def test_empty_results_returns_fallback(self) -> None:
        result = generate_cited_answer("test?", [], api_key="fake")
        assert "couldn't find" in result.answer.lower()
        assert result.citations == []

    @patch("multimodal_rag.query.generator.OpenAI")
    def test_calls_llm_and_returns_answer(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Use [1] to get started."
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        results = [_video_result()]
        answer = generate_cited_answer(
            "How do I start?",
            results,
            api_key="fake",
            base_url="http://test",
            model="test-model",
        )

        assert "Quickstart @ 00:42" in answer.answer
        assert len(answer.citations) == 1
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test-model"

    @patch("multimodal_rag.query.generator.OpenAI")
    def test_passes_context_to_llm(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Answer."
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        results = [_video_result(), _web_result()]
        generate_cited_answer("question?", results, api_key="fake")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "Paro Software" in messages[0]["content"]
        assert "## Sources" in messages[1]["content"]
        assert "## Question" in messages[1]["content"]
