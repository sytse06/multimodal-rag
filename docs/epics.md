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

## Epic 3: Per-source Ingestion Pipeline (completed)

Restructured the ingestion pipeline from monolithic batch processing to per-source granularity. Each YouTube video and each crawled KB page is independently scraped → chunked → embedded → stored. Failures are isolated and logged, not fatal.

### INGEST-001 — Per-source Ingest Loop

**Branch:** `feature/per-source-ingest`

- `__main__.py` opens `WeaviateStore` and embeddings once, then processes per unit
- YouTube: after fetching+chunking each video, immediately `store.add_chunks()` for that video
- Web KBs: `crawl_knowledge_base()` per KB, then `split_by_sections()` + `store.add_chunks()` per page
- Removed `fetch_web_chunks()` monolith — orchestrator calls `crawl_knowledge_base()` and `split_by_sections()` directly
- Extracted `_ingest_chunks()` helper with per-unit logging (source name, page URL, chunk count)

**Files:** `src/multimodal_rag/ingest/__main__.py`, `src/multimodal_rag/ingest/web.py`, `src/multimodal_rag/ingest/__init__.py`
**Tests:** 6

### INGEST-002 — Error Isolation

**Branch:** `feature/per-source-ingest`

- Each video ingest wrapped in try/except — logs error with video name, continues
- Each KB page ingest wrapped in try/except — logs error with page URL, continues
- KB crawl failure skips entire KB, continues to next
- Summary log at end: total added, total failed, total in store

**Files:** `src/multimodal_rag/ingest/__main__.py`

### INGEST-003 — Embedding Safety

**Branch:** `feature/per-source-ingest`

- Lowered `_MAX_WORDS` from 800 → 400 (matches chunk target, safer for URL-heavy markdown)
- On "context length" error: catch per-batch, re-truncate to `_RETRY_MAX_WORDS = 200`, retry once
- Logs warning when truncation or retry kicks in (includes original word count)
- Non-context-length errors propagate normally

**Files:** `src/multimodal_rag/store/embeddings.py`
**Tests:** 2

**Note:** `_MAX_WORDS` (400) is independent of the `CHUNK_SIZE` env var (default 400 tokens). If `CHUNK_SIZE` is increased above ~500 tokens, embeddings will silently truncate chunks to 400 words. This is intentional — the embedding model's context window is the hard limit, not the chunk target.

---

## Epic 4: Voxtral Audio Transcription Fallback (completed)

Automatic fallback to Mistral Voxtral Mini audio transcription when `youtube-transcript-api` cannot retrieve captions. Produces segment-level timestamps fed into the unchanged `chunk_segments` pipeline. `IpBlocked` (transient network error) is explicitly excluded from the fallback path.

### VOXTRAL-001 — Audio Download

**Branch:** `feature/voxtral-transcription-fallback`

- `download_audio()` — invokes `yt-dlp` with `bestaudio[ext=m4a]/bestaudio` format preference, writes to a caller-supplied `output_dir`
- Falls back to directory scan when `yt-dlp` adjusts the output filename after download

**Files:** `src/multimodal_rag/ingest/voxtral.py`

### VOXTRAL-002 — Voxtral Transcription

**Branch:** `feature/voxtral-transcription-fallback`

- `transcribe_with_voxtral()` — calls `mistralai` SDK (`voxtral-mini-latest`) with `timestamp_granularities=["segment"]`
- Normalises Mistral segment objects to `{"text", "start", "duration"}` dicts — same schema as `youtube-transcript-api` output
- Returns empty list (with warning log) when Voxtral returns no segments

**Files:** `src/multimodal_rag/ingest/voxtral.py`

### VOXTRAL-003 — Fallback Integration

**Branch:** `feature/voxtral-transcription-fallback`

- `fetch_voxtral_transcript()` — orchestrator: creates `tempfile.TemporaryDirectory`, calls `download_audio` + `transcribe_with_voxtral`, guarantees cleanup on exit
- `fetch_transcript_chunks()` gains `mistral_api_key: str = ""` parameter
- Catches `TranscriptsDisabled` and `NoTranscriptFound` → invokes Voxtral fallback if key is set; logs warning and returns `[]` if key is absent
- `IpBlocked` bypasses fallback — logged as warning, returns `[]`
- Voxtral failure propagates as `[]` (exception logged, pipeline continues)
- `AppSettings.mistral_api_key` added; `__main__.py` passes it through to `fetch_transcript_chunks`

**Files:** `src/multimodal_rag/ingest/youtube.py`, `src/multimodal_rag/ingest/__main__.py`, `src/multimodal_rag/models/config.py`
**Tests:** 13 (all mocked — no real API calls or audio downloads)

---

## Summary

| Epic | Features | Total Tests |
|------|----------|-------------|
| 1 — Ingestion Pipeline | 5 | 51 |
| 2 — Query + UI | 4 | 80+ (after QUERY-004) |
| 3 — Per-source Ingest | 3 | 8 |
| 4 — Voxtral Fallback | 3 | 13 |
| **Total** | **15** | **93+** |
