"""Cited answer generation via LangChain chat model."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from multimodal_rag.models.query import Citation, CitedAnswer, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a support assistant for Paro Software. Answer the user's question \
using ONLY the provided source chunks. Follow these rules strictly:

1. Base your answer entirely on the provided sources. Do NOT invent information.
2. Cite sources using bracket notation like [1], [2], etc. matching the chunk numbers.
3. If the sources don't contain enough information, say so honestly.
4. Keep answers concise and actionable for support staff.
5. Use markdown formatting for readability."""

USER_TEMPLATE = """\
## Sources

{context}

## Question

{question}"""


def generate_cited_answer(
    question: str,
    results: list[SearchResult],
    llm: BaseChatModel,
) -> CitedAnswer:
    """Generate a cited answer from retrieved search results."""
    from multimodal_rag.query.retriever import format_context

    if not results:
        return CitedAnswer(
            answer="I couldn't find any relevant sources to answer your question.",
            citations=[],
        )

    context = format_context(results)
    user_message = USER_TEMPLATE.format(context=context, question=question)

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    raw_answer = str(response.content) if response.content else ""

    citations = _build_citations(results)
    answer_with_links = _replace_refs_with_links(raw_answer, results)

    logger.info("Generated answer with %d citations", len(citations))
    return CitedAnswer(answer=answer_with_links, citations=citations)


def _build_citations(results: list[SearchResult]) -> list[Citation]:
    """Build Citation objects from search results."""
    return [
        Citation(
            label=r.citation_label,
            url=r.citation_url,
            relevance_score=r.relevance_score,
            source_type=r.source_type,
        )
        for r in results
    ]


def _replace_refs_with_links(
    text: str, results: list[SearchResult]
) -> str:
    """Replace [1], [2] etc. with clickable markdown citation links."""
    for i, r in enumerate(results, 1):
        score_pct = round(r.relevance_score * 100)
        link = f"[{r.citation_label}]({r.citation_url}) ({score_pct}%)"
        text = text.replace(f"[{i}]", link)
    return text


KB_ARTICLE_PROMPT = """\
You are a technical writer for Paro Software.
Write a comprehensive, standalone knowledge base article based on the source \
material below.

Rules:
1. Write for support staff who may not have seen the original question.
2. Open with a short summary sentence stating what the article covers.
3. Use markdown: headers, bullet points, numbered steps where appropriate.
4. Draw on ALL details in the source material — include specific steps, settings, \
menu paths, keyboard shortcuts, and field names. Do not summarise away detail.
5. Do not invent information that is not present in the sources.
6. End with a "Sources" section listing the references."""


def generate_kb_article(
    answer: CitedAnswer,
    llm: BaseChatModel,
    results: list[SearchResult] | None = None,
) -> str:
    """Generate a comprehensive KB article from a cited answer and its source chunks."""
    source_blocks: list[str] = []
    if results:
        for i, r in enumerate(results, 1):
            source_blocks.append(
                f"### Source [{i}]: {r.citation_label}\n\n{r.text}"
            )

    parts = ["## Source Material\n"]
    if source_blocks:
        parts.append("\n\n".join(source_blocks))
    else:
        parts.append(answer.answer)

    if answer.citations:
        refs = "\n".join(
            f"- [{c.label}]({c.url})" for c in answer.citations
        )
        parts.append(f"\n\n## References\n\n{refs}")

    user_message = "\n".join(parts)
    response = llm.invoke([
        SystemMessage(content=KB_ARTICLE_PROMPT),
        HumanMessage(content=user_message),
    ])
    return str(response.content) if response.content else ""
