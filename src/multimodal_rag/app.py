"""Gradio chat interface for the support knowledge base."""

import logging

import gradio as gr
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from multimodal_rag.models.chunks import SourceType
from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.llm import create_embeddings
from multimodal_rag.models.query import CitedAnswer
from multimodal_rag.query.generator import generate_cited_answer
from multimodal_rag.query.retriever import retrieve
from multimodal_rag.store.weaviate import WeaviateStore

logger = logging.getLogger(__name__)

OPENROUTER_MODELS = [
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
    "mistralai/ministral-14b-2512",
    "qwen/qwen3-32b",
]


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


def _make_llm(model_name: str, settings: AppSettings) -> BaseChatModel:
    """Create a chat model for the given model name and provider."""
    if settings.llm_provider == "ollama":
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

    def respond(
        message: str,
        history: list[dict[str, str]],
        model: str,
    ) -> str:
        if not message.strip():
            return "Please enter a question."

        llm = _make_llm(model, settings)
        results = retrieve(message, store, top_k=settings.top_k)
        answer = generate_cited_answer(
            question=message,
            results=results,
            llm=llm,
        )
        return _format_citations_block(answer)

    with gr.Blocks(title="Paro Support KB") as demo:
        gr.Markdown("# Paro Support Knowledge Base")
        gr.Markdown(
            "Ask questions about HydroSym products."
            " Answers include cited sources with clickable links."
        )

        if settings.llm_provider == "openrouter":
            model_dropdown = gr.Dropdown(
                choices=OPENROUTER_MODELS,
                value=settings.llm_model,
                label="Model",
                interactive=True,
            )
        else:
            model_dropdown = gr.Dropdown(
                choices=[settings.llm_model],
                value=settings.llm_model,
                label="Model (Ollama)",
                interactive=False,
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
