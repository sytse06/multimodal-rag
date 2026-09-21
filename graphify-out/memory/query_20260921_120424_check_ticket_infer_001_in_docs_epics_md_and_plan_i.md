---
type: "query"
date: "2026-09-21T12:04:24.110136+00:00"
question: "Check ticket INFER-001 in docs/epics.md and plan its implementation"
contributor: "graphify"
outcome: "useful"
source_nodes: ["AppSettings", "SourceConfig", "SupportChunk", "create_chat_model()", "Provider-agnostic Model Configuration"]
---

# Q: Check ticket INFER-001 in docs/epics.md and plan its implementation

## Answer

Expanded from original query via vocab: [provider, configuration, AppSettings, SupportChunk, SourceConfig, settings]. INFER-001 should be implemented as a typed configuration migration, not only a config.py edit. The current flat AppSettings loads .env, has string credentials, only OpenRouter/Ollama providers, and no explicit validation; app.py, llm.py, ingestion, Makefile, README, and tests consume those fields. SourceConfig uses mutable list defaults, and SupportChunk declares UUID = None with a type ignore. The plan is: establish typed Pydantic settings and provider-specific models with SecretStr and explicit discriminators; define env aliases and precedence; validate credentials/endpoints/model/timeout/retry/temperature and GRADIO_SHARE=false; fix model defaults and UUID typing; migrate all consumers; add a complete root .env.example and onboarding docs; add offline tests for provider matrix, precedence, validation, secret redaction/serialization, mutable defaults, and compatibility. Existing local .env contains OPENAI_* and GEMINI_* keys that currently cause four LLM tests to fail as extra inputs, so the migration must address those fields deliberately. Run quality and tests before commit.

## Outcome

- Signal: useful

## Source Nodes

- AppSettings
- SourceConfig
- SupportChunk
- create_chat_model()
- Provider-agnostic Model Configuration