"""Gradio chat interface for the support knowledge base."""

import logging
import re
import warnings
from datetime import datetime
from pathlib import Path

import gradio as gr

from multimodal_rag.models.chunks import SourceType
from multimodal_rag.models.config import AppSettings, ChatProvider, ModelSelection
from multimodal_rag.models.llm import create_chat_model, create_embeddings
from multimodal_rag.models.providers import (
    model_choices,
    provider_choices,
    selection_from_key,
)
from multimodal_rag.models.query import CitedAnswer, SearchResult
from multimodal_rag.query.generator import (
    generate_cited_answer,
    stream_cited_answer,
    stream_kb_article,
)
from multimodal_rag.query.retriever import retrieve
from multimodal_rag.store.weaviate import WeaviateStore

logger = logging.getLogger(__name__)

KB_OUTPUT_DIR = Path("kb_output")


def resolve_selection(
    provider: str, model_key: str, settings: AppSettings
) -> ModelSelection:
    """Validate that the selected model belongs to the selected provider."""
    selection = selection_from_key(model_key, settings)
    if selection.provider != provider:
        raise ValueError("The selected model does not belong to the selected provider")
    return selection


def normalize_stored_selection(
    selection: ModelSelection | dict[str, object] | None,
    settings: AppSettings,
) -> ModelSelection:
    """Restore a selection from Gradio state or use the configured active model."""
    if isinstance(selection, ModelSelection):
        return selection
    if isinstance(selection, dict):
        return ModelSelection.model_validate(selection)
    return selection_from_key(
        f"{settings.chat.provider}:{settings.chat.model}", settings
    )


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
        weaviate_mode=settings.weaviate_mode,
        weaviate_api_key=settings.weaviate_api_key.get_secret_value(),
    )

    def _respond(
        message: str,
        selection_key: str,
    ) -> tuple[str, CitedAnswer, list[SearchResult]]:
        selection = selection_from_key(selection_key, settings)
        llm = create_chat_model(settings, selection)
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

        choices = model_choices(settings, settings.chat.provider)
        default_model = f"{settings.chat.provider}:{settings.chat.model}"
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=choices,
                value=default_model,
                label="Model",
                interactive=True,
                scale=1,
            )
            provider_dropdown = gr.Dropdown(
                choices=provider_choices(settings),
                value=settings.chat.provider,
                label="Provider",
                interactive=True,
                scale=1,
            )
        provider_status = gr.Markdown()

        last_answer_state: gr.State = gr.State(None)
        last_results_state: gr.State = gr.State([])
        last_question_state: gr.State = gr.State("")
        last_selection_state: gr.State = gr.State(None)

        chatbot = gr.Chatbot(label="Chat", height=500)

        with gr.Column(visible=True) as input_row:
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a support question...",
                    label="Question",
                    show_label=False,
                    scale=4,
                    submit_btn=True,
                    stop_btn=True,
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

        def update_models(provider: ChatProvider) -> tuple[object, str]:
            choices = model_choices(settings, provider)
            if not choices:
                return gr.update(choices=[], value=None), (
                    f"No models are configured for **{provider}**."
                )
            return gr.update(choices=choices, value=choices[0][1]), ""

        provider_dropdown.change(
            update_models,
            inputs=[provider_dropdown],
            outputs=[model_dropdown, provider_status],
            queue=False,
        )

        def user_submit(
            message: str,
            history: list[dict[str, str]],
            provider: str,
            model_key: str,
            previous_answer: CitedAnswer | None,
            previous_results: list[SearchResult],
            previous_question: str,
            previous_selection: ModelSelection | None,
        ) -> object:
            if not message.strip():
                yield (
                    "",
                    history,
                    previous_answer,
                    previous_results,
                    previous_question,
                    previous_selection,
                    "",
                )
                return
            try:
                selection = resolve_selection(provider, model_key, settings)
            except ValueError as exc:
                yield (
                    "",
                    history,
                    previous_answer,
                    previous_results,
                    previous_question,
                    previous_selection,
                    f"⚠️ {exc}",
                )
                return
            question = message
            history = history + [{"role": "user", "content": message}]
            history = history + [
                {"role": "assistant", "content": "_Retrieving sources..._"}
            ]
            yield (
                "",
                history,
                previous_answer,
                previous_results,
                previous_question,
                previous_selection,
                "Retrieving sources...",
            )
            try:
                llm = create_chat_model(settings, selection)
                results = retrieve(message, store, top_k=settings.top_k)
            except Exception as exc:
                logger.exception("Retrieval setup failed: %s", exc)
                history[-1] = {
                    "role": "assistant",
                    "content": (
                        "⚠️ Unable to start this request. Check the provider "
                        "configuration."
                    ),
                }
                yield (
                    "",
                    history,
                    previous_answer,
                    previous_results,
                    previous_question,
                    previous_selection,
                    "Request failed.",
                )
                return

            history[-1] = {"role": "assistant", "content": "_Generating..._"}
            yield (
                "",
                history,
                previous_answer,
                previous_results,
                previous_question,
                previous_selection,
                "Generating...",
            )
            for event in stream_cited_answer(
                question=question,
                results=results,
                llm=llm,
                provider=selection.provider,
                model=selection.model,
                max_context_tokens=settings.chat.max_context_tokens,
            ):
                if event.event_type == "progress":
                    history[-1] = {"role": "assistant", "content": event.text}
                    yield (
                        "",
                        history,
                        previous_answer,
                        previous_results,
                        previous_question,
                        previous_selection,
                        "Generating...",
                    )
                elif event.event_type == "complete" and event.answer is not None:
                    history[-1] = {
                        "role": "assistant",
                        "content": _format_citations_block(event.answer),
                    }
                    yield (
                        "",
                        history,
                        event.answer,
                        results,
                        question,
                        selection,
                        "Completed",
                    )
                elif event.event_type in {"error", "cancelled"}:
                    status = (
                        "Cancelled"
                        if event.event_type == "cancelled"
                        else event.error or "Request failed."
                    )
                    history[-1] = {
                        "role": "assistant",
                        "content": f"⚠️ {event.error or status}",
                    }
                    yield (
                        "",
                        history,
                        previous_answer,
                        previous_results,
                        previous_question,
                        previous_selection,
                        status,
                    )

        outputs_submit = [
            msg,
            chatbot,
            last_answer_state,
            last_results_state,
            last_question_state,
            last_selection_state,
            provider_status,
        ]
        submit_event = msg.submit(
            user_submit,
            inputs=[
                msg,
                chatbot,
                provider_dropdown,
                model_dropdown,
                last_answer_state,
                last_results_state,
                last_question_state,
                last_selection_state,
            ],
            outputs=outputs_submit,
            trigger_mode="always_last",
            concurrency_limit=1,
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
            queue=False,
            show_progress="hidden",
            cancels=[submit_event],
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
            queue=False,
            show_progress="hidden",
            cancels=[submit_event],
        )

        # Step navigation — Step 2 sources loaded lazily on first visit
        def go_to_step2(results: list[SearchResult]) -> tuple[object, str]:
            return gr.Walkthrough(selected=2), _format_step2(results)

        next1_btn.click(
            go_to_step2,
            inputs=[last_results_state],
            outputs=[walkthrough, step2_sources],
            queue=False,
            show_progress="hidden",
        )
        back2_btn.click(
            lambda: gr.Walkthrough(selected=1),
            outputs=walkthrough,
            queue=False,
            show_progress="hidden",
        )

        def go_to_step3(
            answer: CitedAnswer | None,
            results: list[SearchResult],
            selection: ModelSelection | dict[str, object] | None,
            question: str,
        ) -> object:
            if answer is None:
                yield gr.Walkthrough(selected=3), "", "No answer available."
                return
            selection = normalize_stored_selection(selection, settings)
            try:
                llm = create_chat_model(settings, selection)
            except Exception as exc:
                logger.exception("Article draft setup failed: %s", exc)
                yield (
                    gr.Walkthrough(selected=3),
                    "",
                    "⚠️ Unable to start the draft. Check the provider configuration.",
                )
                return
            for event in stream_kb_article(
                answer,
                llm,
                results=results,
                question=question,
                provider=selection.provider,
                model=selection.model,
                max_context_tokens=settings.chat.max_context_tokens,
            ):
                if event.event_type == "progress":
                    yield gr.Walkthrough(selected=3), event.text, "Generating draft..."
                elif event.event_type == "complete":
                    yield (
                        gr.Walkthrough(selected=3),
                        event.article or event.text,
                        "Draft ready",
                    )
                else:
                    yield (
                        gr.Walkthrough(selected=3),
                        event.text,
                        event.error or "Draft cancelled",
                    )

        def start_draft() -> tuple[object, str, str]:
            return gr.Walkthrough(selected=3), "", "Generating draft..."

        draft_transition = next2_btn.click(
            start_draft,
            outputs=[walkthrough, article_editor, provider_status],
            queue=False,
            show_progress="hidden",
        )
        draft_transition.then(
            go_to_step3,
            inputs=[
                last_answer_state,
                last_results_state,
                last_selection_state,
                last_question_state,
            ],
            outputs=[walkthrough, article_editor, provider_status],
        )

        back3_btn.click(
            lambda: gr.Walkthrough(selected=2),
            outputs=walkthrough,
            queue=False,
            show_progress="hidden",
        )

        def go_to_step4(article: str) -> tuple[object, str]:
            h1 = next(
                (ln for ln in article.splitlines() if ln.startswith("# ")), ""
            )
            return gr.Walkthrough(selected=4), h1[2:].strip()

        next3_btn.click(
            go_to_step4,
            inputs=[article_editor],
            outputs=[walkthrough, title_input],
            queue=False,
            show_progress="hidden",
        )
        back4_btn.click(
            lambda: gr.Walkthrough(selected=3),
            outputs=walkthrough,
            queue=False,
            show_progress="hidden",
        )

        def do_save(title: str, body: str) -> str:
            path = save_kb_article(title, body)
            return f"Saved to `{path}`"

        save_btn.click(
            do_save,
            inputs=[title_input, article_editor],
            outputs=[save_msg],
        )

    demo.launch(share=settings.gradio_share)


if __name__ == "__main__":
    main()
