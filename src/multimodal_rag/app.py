"""Gradio chat interface for the support knowledge base."""

import logging
import re
import warnings
from datetime import datetime
from pathlib import Path

import gradio as gr
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from multimodal_rag.models.chunks import SourceType
from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.llm import create_embeddings
from multimodal_rag.models.query import CitedAnswer, SearchResult
from multimodal_rag.query.generator import generate_cited_answer, generate_kb_article
from multimodal_rag.query.retriever import retrieve
from multimodal_rag.store.weaviate import WeaviateStore

logger = logging.getLogger(__name__)

OPENROUTER_MODELS = [
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
    "mistralai/ministral-14b-2512",
    "qwen/qwen3-32b",
]

KB_OUTPUT_DIR = Path("kb_output")


def _format_citations_block(answer: CitedAnswer) -> str:
    """Append a citations summary block below the answer."""
    if not answer.citations:
        return answer.answer

    lines = [answer.answer, "", "---", "**Sources:**"]
    for c in answer.citations:
        icon = "\U0001f3ac" if c.source_type == SourceType.VIDEO else "\U0001f4c4"
        score_pct = round(c.relevance_score * 100)
        lines.append(f"- {icon} [{c.label}]({c.url}) ({score_pct}%)")
    return "\n".join(lines)


def _slugify(title: str) -> str:
    """Convert a title into a filename-safe slug (max 60 chars)."""
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "-", slug).strip("-")
    return slug[:60] or "article"


def _format_step1(answer: CitedAnswer) -> tuple[str, str]:
    """Return (answer_markdown, citations_markdown) for Step 1 display."""
    if not answer.citations:
        return answer.answer, "_No citations._"
    lines = [
        f"- [{c.label}]({c.url}) ({round(c.relevance_score * 100)}%)"
        for c in answer.citations
    ]
    return answer.answer, "\n".join(lines)


def _format_step2(results: list[SearchResult]) -> str:
    """Return full source chunks markdown for Step 2 display."""
    if not results:
        return "_No sources retrieved._"
    blocks = []
    for i, r in enumerate(results, 1):
        icon = "\U0001f3ac" if r.source_type == SourceType.VIDEO else "\U0001f4c4"
        score_pct = round(r.relevance_score * 100)
        blocks.append(
            f"**[{i}] {icon} {r.citation_label}** ({score_pct}%)\n\n{r.text}"
        )
    return "\n\n---\n\n".join(blocks)


def save_kb_article(
    title: str, body: str, output_dir: Path = KB_OUTPUT_DIR
) -> str:
    """Write article to output_dir/{slug}-{timestamp}.md. Returns file path."""
    if not title.strip():
        title = "Untitled Article"
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(title)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"{slug}-{ts}.md"
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    logger.info("Saved KB article: %s", path)
    return str(path)


def _is_ollama_model(model_name: str) -> bool:
    """Ollama models are bare names (e.g. 'llama3.2'); OpenRouter uses 'provider/model'."""  # noqa: E501
    return "/" not in model_name


def _make_llm(model_name: str, settings: AppSettings) -> BaseChatModel:
    """Route to Ollama or OpenRouter based on model name format."""
    if _is_ollama_model(model_name):
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name,
            base_url=settings.ollama_base_url,
            temperature=0.3,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        api_key=SecretStr(settings.openrouter_api_key),
        base_url=settings.openrouter_base_url,
        temperature=0.3,
    )


def main() -> None:
    # Suppress Pandas deprecation warnings emitted by Gradio internals.
    # gradio/queueing.py calls df.infer_objects(copy=False) and uses
    # future.no_silent_downcasting — both deprecated in pandas 3.0.
    warnings.filterwarnings("ignore", message=".*no_silent_downcasting.*")
    warnings.filterwarnings("ignore", message=".*copy keyword is deprecated.*")

    settings = AppSettings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    embeddings = create_embeddings(settings)
    store = WeaviateStore(
        weaviate_url=settings.weaviate_url,
        embeddings=embeddings,
    )

    def _respond(
        message: str,
        model: str,
    ) -> tuple[str, CitedAnswer, list[SearchResult]]:
        llm = _make_llm(model, settings)
        results = retrieve(message, store, top_k=settings.top_k)
        answer = generate_cited_answer(
            question=message,
            results=results,
            llm=llm,
        )
        return _format_citations_block(answer), answer, results

    _css = ".align-bottom { align-self: flex-end; }"
    with gr.Blocks(title="Paro Support KB", css=_css) as demo:
        gr.Markdown("# Paro Support Knowledge Base")
        gr.Markdown(
            "Ask questions about Paro software products."
            "Answers include cited sources with clickable links."
            "Workflow to generate knowledge base articles based on the sources."
        )

        ollama_models = [
            m for m in [settings.llm_model] if _is_ollama_model(m)
        ]
        all_models = ollama_models + OPENROUTER_MODELS
        default_model = settings.llm_model
        if default_model not in all_models:
            default_model = all_models[0]
        model_dropdown = gr.Dropdown(
            choices=all_models,
            value=default_model,
            label="Model",
            interactive=True,
        )

        last_answer_state: gr.State = gr.State(None)
        last_results_state: gr.State = gr.State([])

        chatbot = gr.Chatbot(label="Chat", height=500)

        with gr.Column(visible=True) as input_row:
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a support question...",
                    label="Question",
                    show_label=False,
                    scale=4,
                )
                submit_btn = gr.Button(
                    "Submit", variant="primary", scale=1, elem_classes=["align-bottom"]
                )
            with gr.Row():
                gr.ClearButton([msg, chatbot], value="Clear conversation")
                review_btn = gr.Button(
                    "Review & save as article", variant="secondary"
                )

        with gr.Column(visible=False) as walkthrough_col:
            with gr.Walkthrough(selected=1) as walkthrough:
                with gr.Step("Review answer", id=1):
                    gr.Markdown("### Generated answer")
                    step1_answer = gr.Markdown()
                    gr.Markdown("### Citations")
                    step1_citations = gr.Markdown()
                    with gr.Row():
                        cancel_btn = gr.Button(
                            "Cancel — back to chat", variant="stop"
                        )
                        next1_btn = gr.Button("Next →", variant="primary")

                with gr.Step("Inspect sources", id=2):
                    gr.Markdown("### Retrieved source chunks")
                    step2_sources = gr.Markdown()
                    with gr.Row():
                        back2_btn = gr.Button("← Back")
                        next2_btn = gr.Button("Next →", variant="primary")

                with gr.Step("Edit draft", id=3):
                    gr.Markdown("### Edit the KB article draft")
                    article_editor = gr.Textbox(
                        lines=20,
                        label="Article draft",
                        placeholder="Generating draft...",
                    )
                    with gr.Row():
                        back3_btn = gr.Button("← Back")
                        next3_btn = gr.Button("Next →", variant="primary")

                with gr.Step("Save", id=4):
                    gr.Markdown("### Save article")
                    title_input = gr.Textbox(
                        label="Article title",
                        placeholder="Enter a descriptive title...",
                    )
                    save_btn = gr.Button("Save article", variant="primary")
                    save_msg = gr.Markdown()
                    back4_btn = gr.Button("← Back")

        # --- Chat submit ---

        def user_submit(
            message: str,
            history: list[dict[str, str]],
            model: str,
        ) -> tuple[str, list[dict[str, str]], CitedAnswer | None, list[SearchResult]]:
            if not message.strip():
                return "", history, None, []
            history = history + [{"role": "user", "content": message}]
            formatted, answer, results = _respond(message, model)
            history = history + [{"role": "assistant", "content": formatted}]
            return "", history, answer, results

        msg.submit(
            user_submit,
            inputs=[msg, chatbot, model_dropdown],
            outputs=[msg, chatbot, last_answer_state, last_results_state],
        )
        submit_btn.click(
            user_submit,
            inputs=[msg, chatbot, model_dropdown],
            outputs=[msg, chatbot, last_answer_state, last_results_state],
        )

        # --- Review workflow ---

        def enter_review(
            answer: CitedAnswer | None,
        ) -> tuple:
            if answer is None:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
            a_text, c_text = _format_step1(answer)
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(height=300),
                gr.Walkthrough(selected=1),
                a_text,
                c_text,
            )

        review_btn.click(
            enter_review,
            inputs=[last_answer_state],
            outputs=[
                input_row,
                walkthrough_col,
                chatbot,
                walkthrough,
                step1_answer,
                step1_citations,
            ],
        )

        def cancel_review() -> tuple:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(height=500),
            )

        cancel_btn.click(
            cancel_review,
            outputs=[input_row, walkthrough_col, chatbot],
        )

        # Step navigation — Step 2 sources loaded lazily on first visit
        def go_to_step2(results: list[SearchResult]) -> tuple[object, str]:
            return gr.Walkthrough(selected=2), _format_step2(results)

        next1_btn.click(
            go_to_step2,
            inputs=[last_results_state],
            outputs=[walkthrough, step2_sources],
        )
        back2_btn.click(lambda: gr.Walkthrough(selected=1), outputs=walkthrough)

        def go_to_step3(
            answer: CitedAnswer | None,
            results: list[SearchResult],
            model: str,
        ) -> tuple[object, str]:
            if answer is None:
                return gr.Walkthrough(selected=3), ""
            llm = _make_llm(model, settings)
            draft = generate_kb_article(answer, llm, results=results)
            return gr.Walkthrough(selected=3), draft

        next2_btn.click(
            go_to_step3,
            inputs=[last_answer_state, last_results_state, model_dropdown],
            outputs=[walkthrough, article_editor],
        )

        back3_btn.click(lambda: gr.Walkthrough(selected=2), outputs=walkthrough)
        next3_btn.click(lambda: gr.Walkthrough(selected=4), outputs=walkthrough)
        back4_btn.click(lambda: gr.Walkthrough(selected=3), outputs=walkthrough)

        def do_save(title: str, body: str) -> str:
            path = save_kb_article(title, body)
            return f"Saved to `{path}`"

        save_btn.click(
            do_save,
            inputs=[title_input, article_editor],
            outputs=[save_msg],
        )

    demo.launch(share=True)


if __name__ == "__main__":
    main()
