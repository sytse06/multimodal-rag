"""Tests for the Gradio chat app."""

import re
from pathlib import Path

from multimodal_rag.app import (
    _format_citations_block,
    _format_step1,
    _format_step2,
    _slugify,
    save_kb_article,
)
from multimodal_rag.models.chunks import SourceType
from multimodal_rag.models.query import Citation, CitedAnswer, SearchResult


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


class TestSlugify:
    def test_lowercase_and_spaces_to_dashes(self) -> None:
        assert _slugify("Hello World") == "hello-world"

    def test_special_chars_removed(self) -> None:
        result = _slugify("File > New: Setup!")
        assert re.match(r"^[a-z0-9-]+$", result)

    def test_long_title_truncated(self) -> None:
        long = "a" * 100
        assert len(_slugify(long)) <= 60

    def test_empty_returns_article(self) -> None:
        assert _slugify("") == "article"
        assert _slugify("   ") == "article"


class TestFormatStep1:
    def test_no_citations(self) -> None:
        answer = CitedAnswer(answer="The answer.", citations=[])
        a_text, c_text = _format_step1(answer)
        assert a_text == "The answer."
        assert "_No citations._" in c_text

    def test_with_citations(self) -> None:
        answer = CitedAnswer(
            answer="See the docs.",
            citations=[
                Citation(
                    label="Quickstart @ 00:42",
                    url="https://yt.com/abc",
                    relevance_score=0.9,
                    source_type=SourceType.VIDEO,
                )
            ],
        )
        a_text, c_text = _format_step1(answer)
        assert a_text == "See the docs."
        assert "Quickstart @ 00:42" in c_text
        assert "(90%)" in c_text

    def test_multiple_citations(self) -> None:
        answer = CitedAnswer(
            answer="Answer.",
            citations=[
                Citation(
                    label="A",
                    url="https://a.com",
                    relevance_score=0.9,
                    source_type=SourceType.VIDEO,
                ),
                Citation(
                    label="B",
                    url="https://b.com",
                    relevance_score=0.7,
                    source_type=SourceType.WEB,
                ),
            ],
        )
        _, c_text = _format_step1(answer)
        assert c_text.count("- [") == 2


class TestFormatStep2:
    def test_empty_results(self) -> None:
        result = _format_step2([])
        assert "_No sources retrieved._" in result

    def test_includes_source_text(self) -> None:
        results = [
            SearchResult(
                text="Click File > New to create a project.",
                source_type=SourceType.VIDEO,
                source_url="https://yt.com/watch?v=abc",
                source_name="Quickstart",
                timestamp_seconds=42,
                relevance_score=0.9,
            )
        ]
        result = _format_step2(results)
        assert "Click File > New" in result
        assert "(90%)" in result
        assert "Quickstart" in result

    def test_multiple_results_separated(self) -> None:
        results = [
            SearchResult(
                text="Text A",
                source_type=SourceType.VIDEO,
                source_url="https://yt.com/a",
                source_name="Video A",
                timestamp_seconds=0,
                relevance_score=0.9,
            ),
            SearchResult(
                text="Text B",
                source_type=SourceType.WEB,
                source_url="https://docs.com/b",
                source_name="Docs B",
                relevance_score=0.7,
            ),
        ]
        result = _format_step2(results)
        assert "---" in result
        assert "[1]" in result
        assert "[2]" in result


class TestSaveKbArticle:
    def test_writes_file(self, tmp_path: Path) -> None:
        path_str = save_kb_article("My Article", "Body text.", output_dir=tmp_path)
        path = Path(path_str)
        assert path.exists()
        content = path.read_text()
        assert "# My Article" in content
        assert "Body text." in content

    def test_slug_in_filename(self, tmp_path: Path) -> None:
        path_str = save_kb_article("Hello World", "Body.", output_dir=tmp_path)
        assert "hello-world" in Path(path_str).name

    def test_empty_title_uses_untitled(self, tmp_path: Path) -> None:
        path_str = save_kb_article("", "Body.", output_dir=tmp_path)
        assert "untitled-article" in Path(path_str).name

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub" / "kb"
        path_str = save_kb_article("Test", "Body.", output_dir=nested)
        assert Path(path_str).exists()
