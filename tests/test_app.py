"""Tests for the Gradio chat app."""

from multimodal_rag.app import _format_citations_block
from multimodal_rag.models.chunks import SourceType
from multimodal_rag.models.query import Citation, CitedAnswer


class TestFormatCitationsBlock:
    def test_no_citations(self) -> None:
        answer = CitedAnswer(answer="No sources found.", citations=[])
        result = _format_citations_block(answer)
        assert result == "No sources found."
        assert "Sources" not in result

    def test_video_citation_has_icon(self) -> None:
        answer = CitedAnswer(
            answer="Use File > New. See [1].",
            citations=[
                Citation(
                    label="Quickstart @ 00:42",
                    url="https://yt.com/watch?v=abc&t=42s",
                    relevance_score=0.9,
                    source_type=SourceType.VIDEO,
                )
            ],
        )
        result = _format_citations_block(answer)
        assert "**Sources:**" in result
        assert "Quickstart @ 00:42" in result
        assert "(90%)" in result

    def test_web_citation_has_icon(self) -> None:
        answer = CitedAnswer(
            answer="Install via pip.",
            citations=[
                Citation(
                    label="Docs — Installation",
                    url="https://docs.example.com/install",
                    relevance_score=0.8,
                    source_type=SourceType.WEB,
                )
            ],
        )
        result = _format_citations_block(answer)
        assert "Docs — Installation" in result
        assert "(80%)" in result

    def test_multiple_citations(self) -> None:
        answer = CitedAnswer(
            answer="Combined answer.",
            citations=[
                Citation(
                    label="Video A",
                    url="https://yt.com/a",
                    relevance_score=0.9,
                    source_type=SourceType.VIDEO,
                ),
                Citation(
                    label="Page B",
                    url="https://docs.com/b",
                    relevance_score=0.7,
                    source_type=SourceType.WEB,
                ),
            ],
        )
        result = _format_citations_block(answer)
        assert result.count("- ") == 2

    def test_answer_text_preserved(self) -> None:
        answer = CitedAnswer(
            answer="The answer is 42.",
            citations=[
                Citation(
                    label="Source",
                    url="https://example.com",
                    relevance_score=0.5,
                    source_type=SourceType.WEB,
                )
            ],
        )
        result = _format_citations_block(answer)
        assert result.startswith("The answer is 42.")
