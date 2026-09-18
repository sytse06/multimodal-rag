---
type: "query"
date: "2026-09-18T12:39:46.052938+00:00"
question: "Do we need to update docs/PRD.md for the provider configuration work and fixed Weaviate snapshot epic?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["Multimodal RAG Support Knowledge Base PRD", "Provider-agnostic Model Configuration", "AppSettings", "WeaviateStore"]
---

# Q: Do we need to update docs/PRD.md for the provider configuration work and fixed Weaviate snapshot epic?

## Answer

Expanded from original query via vocab: [prd, provider, configuration, storage, weaviate]. The graph connects docs/PRD.md to AppSettings, provider-agnostic model configuration, WeaviateStore, and the ingestion/query pipelines. Direct PRD inspection shows it is now stale in four places: provider scope omits OpenAI and Gemini; the Gradio model selector is described as OpenRouter-only; the completed query factory still says model-name routing; and the PRD has no roadmap entries for the planned inference/configuration or fixed Weaviate snapshot epics. Update the PRD before implementing Epic 8/9, but do not rewrite the stable ingestion and data-model sections.

## Outcome

- Signal: useful

## Source Nodes

- Multimodal RAG Support Knowledge Base PRD
- Provider-agnostic Model Configuration
- AppSettings
- WeaviateStore