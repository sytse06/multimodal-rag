---
type: "query"
date: "2026-09-18T12:00:13.350160+00:00"
question: "Gradio seems bolted on. Plse check."
contributor: "graphify"
outcome: "useful"
source_nodes: ["SourceType", "app.py", "AppSettings", "create_chat_model()", "retrieve()", "generate_cited_answer()"]
---

# Q: Gradio seems bolted on. Plse check.

## Answer

Expanded from original query via graph vocab: [gradio, application, source, type, config, settings, ingestion, query, interface]. Verdict: Gradio is bolted on at the orchestration boundary, but it has not contaminated the domain layer. app.py directly imports SourceType only to select video versus web icons. The real coupling is that app.py constructs settings, embeddings, WeaviateStore and chat models; invokes retrieval and answer/article generation; manages workflow state; writes articles; and defines all Gradio components in one 389-line module. Its _make_llm also duplicates models/llm.py factory logic and routes providers by whether a model name contains a slash instead of using AppSettings.llm_provider. Retrieval and generation are already separate and no core module imports Gradio, so the dependency direction remains sound. The next model-settings/inference epic should centralize runtime model selection in models/llm.py, extract an application-level query/article service, and leave app.py as a thin Gradio adapter. Avoid a broad UI rewrite until that boundary exists.

## Outcome

- Signal: useful

## Source Nodes

- SourceType
- app.py
- AppSettings
- create_chat_model()
- retrieve()
- generate_cited_answer()