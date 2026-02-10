"""Gradio chat interface for the support knowledge base."""

import gradio as gr


def main() -> None:
    demo = gr.Blocks(title="Multimodal RAG - Support Knowledge Base")

    with demo:
        gr.Markdown("# Support Knowledge Base")
        gr.Markdown(
            "Ask questions about our products."
            " Answers include citations with clickable links."
        )
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Ask a support question...", label="Question")
        gr.ClearButton([msg, chatbot])

    demo.launch()


if __name__ == "__main__":
    main()
