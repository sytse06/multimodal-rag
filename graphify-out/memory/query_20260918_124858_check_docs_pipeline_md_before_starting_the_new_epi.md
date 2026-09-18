---
type: "query"
date: "2026-09-18T12:48:58.242608+00:00"
question: "Check docs/pipeline.md before starting the new epic"
contributor: "graphify"
outcome: "useful"
source_nodes: ["Query Pipeline", "app.py", "create_chat_model()", "generate_cited_answer()", "WeaviateStore"]
---

# Q: Check docs/pipeline.md before starting the new epic

## Answer

Expanded from original query via vocab: [pipeline, provider, configuration, storage, weaviate]. docs/pipeline.md needs an Epic 8 refresh before implementation. It currently documents a blocking app.py path using _respond(message, model), _make_llm(model, settings), and final-only state updates; it also recreates the LLM for article generation and describes queue behavior around blocking calls. The new design needs provider/model selection, a centralized provider-neutral factory, streaming inference events with cumulative Gradio updates, final citation replacement after chunk accumulation, typed error and completion state, and chat/embedding separation. Epic 9 only needs a short operational note here: retrieval requires the embedding model compatible with the restored SupportChunk snapshot; backup/restore procedures belong in a deployment or storage runbook, not this query-pipeline document.

## Outcome

- Signal: useful

## Source Nodes

- Query Pipeline
- app.py
- create_chat_model()
- generate_cited_answer()
- WeaviateStore