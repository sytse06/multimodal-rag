from unittest.mock import Mock

import pytest

from multimodal_rag.app import build_app
from multimodal_rag.models.config import AppSettings


def test_preview_has_real_ui_without_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    forbidden = Mock(side_effect=AssertionError("Preview must not use a backend"))
    for name in ("create_embeddings", "WeaviateStore", "create_chat_model"):
        monkeypatch.setattr(f"multimodal_rag.app.{name}", forbidden)
    settings = AppSettings(
        _env_file=None,
        llm_provider="openai",
        llm_model="gpt-6-luna",
        openai_api_key="test-openai",
        gemini_api_key="test-gemini",
        embedding_provider="ollama",
        weaviate_mode="local",
        weaviate_url="http://localhost:8080",
    )
    demo = build_app(settings, preview=True)
    try:
        components = demo.config["components"]
        assert any(component["type"] == "chatbot" for component in components)
        question = next(
            component for component in components
            if component["props"].get("label") == "Question"
        )
        assert question["props"]["interactive"] is False
        review = next(
            component for component in components
            if component["props"].get("value") == "Review & save as article"
        )
        assert review["props"]["interactive"] is False
        handlers = {fn.name: fn for fn in demo.fns.values() if fn.fn is not None}
        assert set(handlers) == {"update_models"}
        update, status = handlers["update_models"].fn("gemini")
        assert update["value"].startswith("gemini:")
        assert status == ""
        forbidden.assert_not_called()
    finally:
        demo.close()
