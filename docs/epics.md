# Development Epics

## Epic 1: Ingestion Pipeline (completed)

Batch CLI process that fetches, chunks, embeds, and stores all source material into Weaviate.

### CORE-001 — Data Models

**Branch:** `feature/CORE-001-data-models`

- `TranscriptChunk` — video transcript segment with timestamp metadata and computed `timestamp_url`/`timestamp_display`
- `WebChunk` — web page segment with source URL and optional section heading
- `SupportChunk` — unified Weaviate model with `from_transcript_chunk`/`from_web_chunk` factory methods, auto-generated `url_hash` (SHA256)
- `SourceConfig` — parsed `sources.yaml` with `YouTubeSource` and `KnowledgeBaseSource` lists
- `SearchResult` — retrieved chunk with relevance score and `citation_markdown` property
- `Citation`, `CitedAnswer` — structured LLM response models
- `AppSettings` — Pydantic BaseSettings for all env-based configuration

**Files:** `src/multimodal_rag/models/` (config.py, sources.py, chunks.py, query.py)
**Tests:** 17

### CORE-002 — YouTube Transcript Ingestion

**Branch:** `feature/CORE-002-youtube-ingest`

- `extract_video_id()` — regex extraction from standard, short, and embed YouTube URLs
- `fetch_transcript()` — retrieves timestamped segments via `youtube-transcript-api`
- `chunk_segments()` — groups consecutive segments into ~400-token chunks, preserving start timestamp of first segment
- `fetch_transcript_chunks()` — orchestrator with error handling

**Files:** `src/multimodal_rag/ingest/youtube.py`
**Tests:** 15 (13 unit + 2 integration against real YouTube API)

### CORE-003 — Web Knowledge Base Ingestion

**Branch:** `feature/CORE-003-web-ingest`

- `crawl_knowledge_base()` — crawls a root URL via Firecrawl, returns pages with markdown content
- `split_by_sections()` — splits markdown by `#`/`##`/`###` headers, falls back to token-based splitting for headerless content
- `_split_by_tokens()` — word-based chunking at configurable target token count
- `fetch_web_chunks()` — orchestrator with error handling

**Files:** `src/multimodal_rag/ingest/web.py`
**Tests:** 15

### CORE-004 — Weaviate Store + Embeddings

**Branch:** `feature/CORE-004-weaviate-store`

- `embed_texts()` — batched embedding via OpenRouter (OpenAI-compatible API), configurable model
- `WeaviateStore` — context manager wrapping weaviate-client v4:
  - `ensure_collection()` — creates `SupportChunk` collection with `Vectorizer.none()` (we supply our own vectors)
  - `add_chunks()` — embeds and batch-inserts SupportChunk objects
  - `search()` — near_vector query returning properties + cosine distance
  - `count()` — aggregate object count
  - `delete_collection()` — teardown for re-ingestion

**Files:** `src/multimodal_rag/store/` (embeddings.py, weaviate.py)
**Tests:** 5

### CORE-005 — Ingest CLI Orchestrator

**Branch:** `feature/CORE-005-ingest-cli`

- Reads `config/sources.yaml` for YouTube and knowledge base URLs
- Iterates YouTube sources → `fetch_transcript_chunks()` → `SupportChunk.from_transcript_chunk()`
- Iterates KB sources → `fetch_web_chunks()` → `SupportChunk.from_web_chunk()`
- Opens `WeaviateStore`, ensures collection, batch-inserts all chunks
- Runnable via `make ingest` or `uv run python -m multimodal_rag.ingest`

**Files:** `src/multimodal_rag/ingest/__main__.py`
**Tests:** 3

---

## Epic 2: Query Pipeline + Gradio UI (completed)

Retrieval-augmented generation chain with a chat interface for support staff.

### QUERY-001 — Retrieval Chain

**Branch:** `feature/QUERY-001-retrieval-chain`

- `retrieve()` — embeds a user query via WeaviateStore, performs near_vector search, converts raw hits to `SearchResult` objects with cosine distance mapped to 0–1 relevance scores
- `format_context()` — renders numbered context blocks (`[1] Source Label\nchunk text`) for LLM prompt injection
- `_distance_to_score()` — Weaviate cosine distance (0=identical, 2=opposite) → relevance score

**Files:** `src/multimodal_rag/query/retriever.py`
**Tests:** 11

### QUERY-002 — Cited Answer Generation

**Branch:** `feature/QUERY-002-cited-answer`

- `generate_cited_answer()` — calls OpenRouter LLM with a system prompt that enforces source-only citing via bracket notation `[1]`, `[2]`
- `_replace_refs_with_links()` — post-processes LLM output, replacing `[N]` references with clickable markdown links including relevance percentages
- `_build_citations()` — extracts structured `Citation` objects from search results
- Graceful fallback when no results are retrieved

**Files:** `src/multimodal_rag/query/generator.py`
**Tests:** 9

### QUERY-003 — Gradio Chat Interface

**Branch:** `feature/QUERY-003-gradio-ui`

- `gr.Blocks` layout: title, model selector dropdown, chat area, text input, clear button
- Model selector: OpenRouter models (gpt-4o-mini, gpt-4o, claude-3.5-sonnet, gemini-pro-1.5)
- `_format_citations_block()` — appends a "Sources" section below each answer with source type icons and relevance percentages
- Messages rendered as markdown — citation links are clickable natively in Gradio
- Runnable via `make run` or `uv run python -m multimodal_rag.app`

**Files:** `src/multimodal_rag/app.py`
**Tests:** 5

### QUERY-004 — LangChain Model Client (pending)

**Branch:** `feature/QUERY-004-langchain-model-client`

Replace direct `openai` SDK usage with LangChain's `BaseChatModel` and `Embeddings` interfaces. This removes the hard lock-in to OpenRouter and enables provider diversity.

**Scope:**
- Create a model client factory (`src/multimodal_rag/models/llm.py`) that returns LangChain model instances based on config
- Support at minimum: OpenRouter (via `ChatOpenAI`), Ollama (via `ChatOllama` / `OllamaEmbeddings`)
- New env vars: `LLM_PROVIDER` (openrouter|ollama), `EMBEDDING_PROVIDER` (openrouter|ollama), `OLLAMA_BASE_URL`
- Refactor `store/embeddings.py` to use LangChain `Embeddings` interface instead of `openai.OpenAI`
- Refactor `query/generator.py` to use LangChain `BaseChatModel` instead of `openai.OpenAI`
- Refactor `store/weaviate.py` to accept a LangChain `Embeddings` instance instead of embedding internally
- Update Gradio app model selector to work with LangChain model switching
- Add `langchain-ollama` to dependencies

**Why:** Never lock into a single provider. LangChain's abstraction lets you swap OpenRouter for Ollama (local, free, private) by changing an env var. Nomic embed models via Ollama are not on OpenRouter — this unblocks local embedding workflows.

---

## Summary

| Epic | Features | Total Tests |
|------|----------|-------------|
| 1 — Ingestion Pipeline | 5 | 55 |
| 2 — Query + UI | 4 | 80+ (after QUERY-004) |
| **Total** | **9** | **80+** |
