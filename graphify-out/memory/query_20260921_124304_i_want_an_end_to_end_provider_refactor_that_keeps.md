---
type: "query"
date: "2026-09-21T12:43:04.978776+00:00"
question: "I want an end-to-end provider refactor that keeps the Gradio interface functioning"
contributor: "graphify"
outcome: "useful"
source_nodes: ["app.py", "_make_llm()", "_is_ollama_model()", "Query Pipeline", "Gradio Chat Interface", "TestCreateChatModel"]
---

# Q: I want an end-to-end provider refactor that keeps the Gradio interface functioning

## Answer

Expanded from original query via vocab: [provider, model, factory, gradio, interface, streaming, configuration]. The graph confirms the current seam is app.py: _make_llm and _is_ollama_model own provider routing, while query/generator.py remains provider-neutral and tests/test_llm.py covers only factory types. The safest refactor is vertical: first introduce a provider/model registry and explicit factory contract, then route one non-streaming provider through it while keeping the current Gradio callbacks, then add provider selectors and availability validation, then add streaming generator callbacks, and finally migrate the KB article workflow. Keep retrieval, CitedAnswer, SearchResult, and citation formatting stable throughout; retain a blocking fallback until streaming is qualified. Add interface-level tests before removing the legacy path.

## Outcome

- Signal: useful

## Source Nodes

- app.py
- _make_llm()
- _is_ollama_model()
- Query Pipeline
- Gradio Chat Interface
- TestCreateChatModel