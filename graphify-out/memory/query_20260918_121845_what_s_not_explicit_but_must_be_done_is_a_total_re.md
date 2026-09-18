---
type: "query"
date: "2026-09-18T12:18:45.413152+00:00"
question: "what's not explicit but must be done is a total refresh of the langchain and gradio libs. Maybe do a check on pydantic as well."
contributor: "graphify"
outcome: "useful"
source_nodes: ["AppSettings", "SupportChunk", "SourceConfig", "create_chat_model()", "_make_llm()", "app.py"]
---

# Q: what's not explicit but must be done is a total refresh of the langchain and gradio libs. Maybe do a check on pydantic as well.

## Answer

Expanded from original query via graph vocab: [lang, gradio, pydantic, settings, config, configuration, model, provider]. Dependency audit on 2026-09-18: installed versus current stable is gradio 6.9.0 to 6.28.0, langchain 1.2.12 to 1.4.1, langchain-core 1.2.18 to 1.6.3, langchain-openai 1.1.11 to 1.6.2, langchain-ollama 1.0.1 to 1.1.0, langchain-weaviate 0.0.6 to 0.0.8, pydantic 2.12.5 to 2.13.5, and pydantic-settings 2.13.1 to 2.15.0. Add langchain-google-genai 4.4.0 and langchain-openrouter 0.2.8. Remove unused top-level langchain if direct provider factories remain, archived and unused langchain-community, unused langchain-weaviate, and unused python-dotenv. Do not perform a single blind upgrade: prune, clean Pydantic configuration, upgrade LangChain providers, implement streaming, then upgrade Gradio and test UI behavior. Gradio 6.28.0 was released today, so test it but retain 6.27.0 as fallback until qualification. Pydantic cleanup: use SecretStr for keys, discriminated provider configs, Field(default_factory=list), eliminate UUID-with-None type ignores, validate provider/model/credential combinations, add an env prefix and complete .env.example, and add serialization tests proving secrets cannot leak. The current repository has no .env.example, which blocks colleague onboarding.

## Outcome

- Signal: useful

## Source Nodes

- AppSettings
- SupportChunk
- SourceConfig
- create_chat_model()
- _make_llm()
- app.py