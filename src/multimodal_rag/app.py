"""Gradio chat interface for the support knowledge base."""

import logging

import gradio as gr

from multimodal_rag.models.chunks import SourceType
from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.query import CitedAnswer
from multimodal_rag.query.generator import generate_cited_answer
from multimodal_rag.query.retriever import retrieve
from multimodal_rag.store.weaviate import WeaviateStore

logger = logging.getLogger(__name__)


def _format_citations_block(answer: CitedAnswer) -> str:
    """Append a citations summary block below the answer."""

    if not answer.citations:
        return answer.answer

    lines = [answer.answer, "", "---", "**Sources:**"]
    for c in answer.citations:
        icon = "🎬" if c.source_type == SourceType.VIDEO else "📄"
        score_pct = round(c.relevance_score * 100)
        lines.append(f"- {icon} [{c.label}]({c.url}) ({score_pct}%)")
    return "\n".join(lines)


def main() -> None:
    settings = AppSettings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    store = WeaviateStore(
        weaviate_url=settings.weaviate_url,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_base_url=settings.openrouter_base_url,
        embedding_model=settings.embedding_model,
    )

    def respond(
        message: str,
        history: list[dict[str, str]],
        model: str,
    ) -> str:
        if not message.strip():
            return "Please enter a question."

        results = retrieve(message, store, top_k=settings.top_k)
        answer = generate_cited_answer(
            question=message,
            results=results,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=model,
        )
        return _format_citations_block(answer)

    with gr.Blocks(title="Paro Support KB") as demo:
        gr.Markdown("# Paro Support Knowledge Base")
        gr.Markdown(
            "Ask questions about HydroSym products."
            " Answers include cited sources with clickable links."
        )

        model_dropdown = gr.Dropdown(
            choices=[
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-pro-1.5",
            ],
            value=settings.llm_model,
            label="Model",
            interactive=True,
        )

        chatbot = gr.Chatbot(label="Chat", height=500)
        msg = gr.Textbox(
            placeholder="Ask a support question...",
            label="Question",
            show_label=False,
        )
        gr.ClearButton([msg, chatbot], value="Clear conversation")

        def user_submit(
            message: str,
            history: list[dict[str, str]],
            model: str,
        ) -> tuple[str, list[dict[str, str]]]:
            history = history + [
                {"role": "user", "content": message},
            ]
            reply = respond(message, history, model)
            history = history + [
                {"role": "assistant", "content": reply},
            ]
            return "", history

        msg.submit(
            user_submit,
            inputs=[msg, chatbot, model_dropdown],
            outputs=[msg, chatbot],
        )

    demo.launch()


if __name__ == "__main__":
    main()
