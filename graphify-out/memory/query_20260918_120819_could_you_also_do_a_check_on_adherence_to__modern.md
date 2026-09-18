---
type: "query"
date: "2026-09-18T12:08:19.801728+00:00"
question: "Could you also do a check on adherence to 'modern' langchain streaming patterns as in this textbook example? https://docs.langchain.com/oss/python/langchain/models"
contributor: "graphify"
outcome: "useful"
source_nodes: ["generate_cited_answer()", "generate_kb_article()", "_make_llm()", "create_chat_model()", "app.py", "generator.py"]
---

# Q: Could you also do a check on adherence to 'modern' langchain streaming patterns as in this textbook example? https://docs.langchain.com/oss/python/langchain/models

## Answer

Expanded from original query via graph vocab: [lang, chat, model, response, message, generator, generation, gradio, answer, chunk]. Verdict: the project uses the current BaseChatModel abstraction and message objects, but the interactive query path does not implement LangChain streaming. All four model call sites use blocking invoke(); the two ingestion vision calls can remain blocking, while generate_cited_answer and generate_kb_article should gain streaming variants. The modern generic pattern is model.stream or model.astream, consuming AIMessageChunk objects, adding chunks into a full message, and reading chunk.text/full.text. Gradio should receive a generator or async generator that yields cumulative chatbot state. Citation replacement currently requires the complete answer, so stream numbered references during generation and perform link replacement/final CitedAnswer construction once the final accumulated message exists. Model initialization is partly modern: direct ChatOpenAI and ChatOllama classes are supported, but current LangChain documentation recommends init_chat_model as the easiest provider-neutral entry point and has a dedicated langchain-openrouter ChatOpenRouter integration. The existing OpenRouter-via-ChatOpenAI(base_url=...) approach and duplicate app.py _make_llm factory should be replaced during the model-settings epic. Add stream tests using AIMessageChunk values, final-message accumulation, empty streams, provider errors, and final citation replacement. Do not stream offline vision ingestion merely for stylistic consistency.

## Outcome

- Signal: useful

## Source Nodes

- generate_cited_answer()
- generate_kb_article()
- _make_llm()
- create_chat_model()
- app.py
- generator.py