# Development Epics

When and where each story reached users (code releases, Cloud Run revisions and content
promotions) is recorded in [`CHANGELOG.md`](../CHANGELOG.md). This file describes what a story
is; the changelog says when it shipped.

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

- `gr.Blocks` layout: title, model selector dropdown, chat area, text input, Submit button, Clear + Review buttons
- Model selector: unified dropdown — Ollama models (bare names, e.g. `llama3.2`) listed first, then OpenRouter models (`openai/gpt-5.4-mini`, `qwen/qwen3.5-35b-a3b`, `deepseek/deepseek-v3.2`, `mistralai/ministral-14b-2512`)
- `_format_citations_block()` — appends a "Sources" section below each answer with source type icons and relevance percentages
- Messages rendered as markdown — citation links are clickable natively in Gradio
- Runnable via `make run` or `uv run python -m multimodal_rag.app`

**Files:** `src/multimodal_rag/app.py`
**Tests:** 5

### QUERY-004 — LangChain Model Client (completed)

**Branch:** `feature/QUERY-004-langchain-model-client`

Replaced direct `openai` SDK usage with LangChain's `BaseChatModel` and `Embeddings` interfaces.

- `create_chat_model()` and `create_embeddings()` factory functions in `models/llm.py`
- `_make_llm()` in `app.py` routes by model name format: bare name (no `/`) → `ChatOllama`, `provider/model` format → `ChatOpenAI` pointed at OpenRouter
- Supports mixing Ollama and OpenRouter models in the same session via the dropdown — no restart required
- `store/embeddings.py` uses LangChain `Embeddings` interface
- `query/generator.py` accepts `BaseChatModel`
- Env vars: `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `OLLAMA_BASE_URL`

**Files:** `src/multimodal_rag/models/llm.py`, `src/multimodal_rag/app.py`, `src/multimodal_rag/store/embeddings.py`, `src/multimodal_rag/query/generator.py`

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

## Epic 4: Voxtral Audio Transcription Fallback (completed; demoted 2026-11-23)

> **Decision (2026-11-23, maintainer): Voxtral is demoted and no longer part of video
> ingestion.** A spoken video's transcript now comes from its **manual English captions**
> when it has them and from a **Gemini transcription** (`gemini-3.8-flash`) otherwise;
> automatic YouTube captions are never used (`ingest/transcripts.py`, built in SYNC-011).
> `make ingest` and `make sync-videos` no longer call Voxtral and need no Mistral key.
>
> **Why.** (1) With a Mistral key set the default path used Voxtral for every video and
> never tried captions. (2) The key in use was rejected (HTTP 401), and the preflight, which
> only checked that a key was set, could not see it; the default path was broken without
> warning. (3) `voxtral-mini-latest` is a floating alias, the same silent-drift exposure that
> cost us the Ollama vectors. (4) A measured comparison on three existing videos, with their
> manual captions as ground truth, gave Gemini the lower word error rate on all three
> (0.010 to 0.040 against 0.028 to 0.054) and every product name correct, while the stored
> Voxtral text writes "HydroSim" for HydroSym (8 of 8 mentions). Gemini also keeps one
> vendor, key and quota for chat, vision and transcription.
>
> **What stays.** The code is not deleted yet: `voxtral.py`, the `mistralai` dependency and
> the `MISTRAL_API_KEY` setting, `fetch_transcript_chunks` (a captions path that still
> imports Voxtral and is no longer reached by ingest), and the `skip_voxtral` flag in the
> registry, which now only means "silent recording, no transcript". Deleting them is a
> follow-up.
>
> **Existing data.** The 25 videos already stored were transcribed by Voxtral and are
> unchanged, "HydroSim" included. Re-transcribing them changes their text and vectors
> (re-embed, then a promotion) and is a separate decision. The stories below describe
> what was built in March and are kept as the record.

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

## Epic 5: Visual Grounding (completed)

Adds a "describe-then-retrieve" ingestion path for visual content. A vision LLM converts images to text descriptions that are embedded with the configured embedding model — no embedding model change is required. This makes the project genuinely multimodal: it indexes not just what was said or written, but what was shown on screen.

### VIS-001 — Video Frame Extraction and Description

**Branch:** `feature/VIS-001-video-frames`

- `extract_keyframes()` — invokes `ffmpeg` to extract frames at regular intervals from a video downloaded by `yt-dlp`
- `describe_frame()` — sends a keyframe image to a vision LLM via OpenRouter, returns a text description
- `fetch_frame_chunks()` — orchestrator: extracts frames, describes each, returns `list[TranscriptChunk]` with stable chunk IDs keyed on `hash(source_url + timestamp_seconds)` to prevent duplicate accumulation on re-ingest

**Files:** `src/multimodal_rag/ingest/video_frames.py`

### VIS-002 — Web Screenshot Extraction and Description

**Branch:** `feature/VIS-002-web-images`

- `extract_image_urls()` — parses crawled markdown from Firecrawl, extracts all embedded image URLs
- `describe_image()` — downloads each image, sends it to a vision LLM via OpenRouter, returns a text description
- `fetch_image_chunks()` — orchestrator: extracts image URLs, describes each, returns `list[WebChunk]` with stable chunk IDs keyed on `hash(image_url)`

**Files:** `src/multimodal_rag/ingest/web_images.py`

### VIS-003 — Vision LLM Factory and Config

**Branch:** `feature/VIS-003-vision-llm-factory`

- `create_vision_llm()` — returns a LangChain-compatible OpenRouter vision model configured from `AppSettings`; visual grounding is disabled when `VISION_MODEL` is empty
- `AppSettings.vision_model` — new `VISION_MODEL` env var; consistent with existing `LLM_MODEL` / `EMBEDDING_MODEL` pattern

**Files:** `src/multimodal_rag/models/llm.py`, `src/multimodal_rag/models/config.py`

### VIS-004 — Stable Chunk ID for Frame and Screenshot Chunks

**Branch:** `feature/VIS-004-chunk-id-stability`

- `SupportChunk.from_frame_chunk()` generates frame IDs from `source_url + timestamp_seconds`, not non-deterministic LLM output
- `SupportChunk.from_screenshot_chunk()` generates screenshot IDs from `image_url`
- Ensures idempotent re-ingestion — re-running `make ingest` does not accumulate duplicate visual chunks

**Files:** `src/multimodal_rag/models/chunks.py`

### VIS-005 — Ingestion Orchestrator Wiring and Tests

**Branch:** `feature/VIS-005-ingest-wiring`

- `ingest/__main__.py` wires both new ingestion paths alongside existing YouTube and web paths
- Frame chunk path: `fetch_frame_chunks()` → `SupportChunk.from_frame_chunk()` → `store.add_chunks()`
- Screenshot chunk path: `fetch_image_chunks()` → `SupportChunk.from_screenshot_chunk()` → `store.add_chunks()`
- Tests for all new code — vision LLM calls, `ffmpeg`/`yt-dlp` invocations, and image downloads are fully mocked

**Files:** `src/multimodal_rag/ingest/__main__.py`
**Tests:** approximately 15 across the five VIS stories

**Not modified:** `store/`, `query/`, `app.py`, embedding pipeline

---

## Epic 6: Multimodal Chunk Fusion (completed)

Combines audio transcript and visual frame description into a single chunk per time window, producing richer embeddings that capture both what was said and what was shown. Adds per-source-type purge tooling to support selective reingest without wiping the entire vector store.

### FUSION-001 — Combined Audio + Visual Chunks

**Branch:** `feature/FUSION-001-combined-chunks`

Before this story, transcript chunks and frame descriptions were stored as separate, competing objects. For a narrated tutorial, the same time window could exist twice: once as a transcript fragment and once as a generic frame description. The completed fusion path replaces those competing objects with one combined chunk per window.

This story merges them into one chunk per time window:

- Spoken videos use Voxtral when `MISTRAL_API_KEY` is configured; `youtube-transcript-api` remains the fallback when it is absent (**superseded 2026-11-23**: manual captions, else Gemini; see the note under Epic 4)
- Align Voxtral segments to the frame extraction interval (e.g. 30s windows)
- For each window: concatenate transcript text for that interval with the vision LLM description of the corresponding keyframe
- Combined chunk format: `"[Transcript] {speech_text}\n[Visual] {frame_description}"`
- If no transcript is available for a window (silence), store visual-only
- If no vision LLM is configured, store transcript-only (Voxtral text only)
- Stable chunk ID keyed on `hash(source_url + window_start_seconds)` — same as current frame chunk ID scheme, ensuring idempotent reingest
- `transcribe_with_voxtral()` is wrapped with `tenacity.retry` (exponential backoff, max 3 attempts, wait 2→60s) — Mistral API rate limits and transient 5xx errors trigger retry; `MistralAPIException` with 4xx status codes are not retried

**Scope:** `src/multimodal_rag/ingest/video_frames.py`, `src/multimodal_rag/ingest/voxtral.py`, `src/multimodal_rag/ingest/youtube.py`, `src/multimodal_rag/ingest/__main__.py`

### FUSION-002 — Purge by Source Type

**Branch:** `feature/FUSION-002-purge-source-type`

Before this story, the only purge options were `make purge` (the entire collection) and `make purge-source URL=...` (one video or page). The completed source-type targets allow video and web chunks to be purged independently.

- Add `delete_by_source_type(source_type: str) -> int` to `WeaviateStore` — filters on the `source_type` property (`"video"` or `"web"`)
- Add `make purge-video` Makefile target — calls `delete_by_source_type("video")` with a confirmation prompt
- Add `make purge-web` Makefile target — calls `delete_by_source_type("web")` with a confirmation prompt
- Log count of deleted objects

**Scope:** `src/multimodal_rag/store/weaviate.py`, `Makefile`

---

## Epic 7: Answer-to-Knowledge-Base Pipeline (completed)

Closes the loop between retrieval quality and knowledge base growth. After the RAG system generates an answer, a structured review workflow lets a support engineer validate the answer against its source chunks, polish it, and promote it directly into the knowledge base as a new article. This serves two purposes: it surfaces retrieval gaps (a bad answer flags missing or low-quality source material) and it turns good answers into reusable, citable KB content.

### WALK-001 — Answer Review UI

**Branch:** `feature/WALK-001-answer-review`

**Starting point:** The existing chat UI (`make run`) has a text input "Ask a support question", a "Clear conversation" button, and a new **"Review & save as article"** button alongside it. Clicking "Review & save as article" triggers a layout transition:

- The chat history stays visible above as a fixed reference (height capped via `gr.Chatbot(max_height=...)` so it doesn't dominate the viewport)
- The chat input row (text input + "Clear conversation" + "Review & save as article") disappears
- The `gr.Walkthrough` panel appears directly below the chat history
- A **"Cancel — back to chat"** button is visible in Step 1 as an escape hatch; it reverses the transition

The walkthrough has four steps, pre-populated from the most recent chat answer and its retrieved source chunks:

1. **Review answer** — display the generated answer alongside its source citations; user assesses quality; Cancel button present
2. **Inspect sources** — show full text of each retrieved chunk with relevance score; cross-reference against the answer above
3. **Edit draft** — editable text area pre-filled with the generated answer; user polishes into a standalone KB article
4. **Save** — user assigns a title and confirms; triggers WALK-003 file export; success message shown

Step navigation driven by Next/Back buttons returning `gr.Walkthrough(selected=N)`:

```python
with gr.Walkthrough(selected=1) as walkthrough:
    with gr.Step("Review answer", id=1): ...
    with gr.Step("Inspect sources", id=2): ...
    with gr.Step("Edit draft", id=3): ...
    with gr.Step("Save", id=4): ...
```

**Layout note:** The combined chat + walkthrough takes significant vertical space but is scrollable. Validate with a realistic 3–4 exchange chat history before finalising layout constants.

**Scope:** `src/multimodal_rag/app.py`

### WALK-002 — KB Article Generation

**Branch:** `feature/WALK-002-kb-article-generation`

Generates a structured KB article draft from the original question, RAG answer, and full source chunks. The existing `SYSTEM_PROMPT` is unchanged — WALK-002 adds a separate `KB_ARTICLE_PROMPT` used only in the editorial workflow.

**`generate_kb_article(answer, llm, results, question)`** — key design decisions:

- `question` is prepended as `## Question` in the user message so the LLM stays scoped to what was asked, not what the sources happen to emphasise
- Full source chunk texts (`r.text`) are included as `### Source [N]: label` blocks — the LLM writes from the raw material, not just the already-summarised answer
- `KB_ARTICLE_PROMPT` instructs a technical writer persona: open with a summary sentence, use markdown structure, draw on ALL source detail, output raw markdown
- `_strip_code_fence()` defensive post-processing removes ` ```markdown ``` ` wrappers that some models add despite the prompt instruction
- **Sources section is appended programmatically** from `answer.citations` after the LLM response — never generated by the LLM. This preserves exact URLs; LLM-generated sources sections drop the URLs.

**Files:** `src/multimodal_rag/query/generator.py`
**Tests:** 12

### WALK-003 — KB Article Export

**Branch:** `feature/WALK-003-kb-article-export`

Saves the approved article as a markdown file in `kb_output/`:

- `save_kb_article(title, body)` — writes `kb_output/{slug}-{timestamp}.md`; `_slugify()` lowercases, strips special chars, truncates to 60 chars
- Step 4 auto-suggests a filename by extracting the first `# ` heading from the draft (`go_to_step4`)
- Confirmation message shown in UI with the saved file path
- `kb_output/` is gitignored

**Out of scope for this iteration:** re-ingest hook, Firecrawl POST, external KB integration.

**Files:** `src/multimodal_rag/app.py`, `kb_output/` (gitignored)
**Tests:** 10

---

## Epic 8: Production-ready, Configurable AI Inference Experience (completed)

Makes the application dependable and self-explanatory for colleagues who did not
build it. A colleague can clone the repository, configure credentials safely, start
the application predictably, select an available AI provider and model, and receive
streamed cited answers without editing source code or understanding provider-specific
implementation details.

**Primary user story:**

> As a colleague using or developing the project, I can configure the application
> safely, start it predictably, select an available AI provider and model, and receive
> streamed cited answers without modifying source code or understanding
> provider-specific implementation details.

**End-to-end acceptance criteria:**

- A fresh clone can be installed with `uv sync`, configured from `.env.example`, and
  started using the documented Make targets
- Missing or invalid configuration produces an actionable error naming the affected
  setting and provider; secrets are never printed or serialized
- Gradio lists only providers and models that are usable with the current configuration
- A colleague can switch chat provider and model without restarting the application
- OpenRouter, OpenAI, Gemini, NVIDIA NIM, and Ollama use the same cited-answer workflow
- Answers and article drafts appear progressively while generation is running
- A running generation can be cancelled cleanly, and context is reduced predictably
  when the configured generation budget would otherwise be exceeded
- Changing the chat provider never changes the embedding provider or invalidates the
  vectors already stored in Weaviate
- Provider, model, completion status, token usage, latency, and safe error context are
  available for operational logging
- Transient provider capacity errors and timeouts are presented as actionable retry or
  provider-switch messages rather than misleading configuration errors

### INFER-001 — Typed Configuration and Colleague Onboarding

**Branch:** `feature/INFER-001-typed-configuration`

- Replace the flat settings collection with explicit Pydantic settings and data models
  for secrets, infrastructure, chat inference, embeddings, ingestion, and application
  runtime configuration
- Use `SecretStr` for credentials and exclude secrets from serialization and logs
- Represent provider-specific configuration as a discriminated union with an explicit
  provider field; never infer the provider from model-name formatting
- Validate provider, model, credentials, endpoint, timeout, retry, and temperature
  combinations at startup
- Add a complete `.env.example` containing safe placeholders and documented defaults
- Define and document configuration precedence and environment-variable naming
- Make public Gradio sharing an explicit opt-in setting; it must be disabled by default
- Clean up existing Pydantic models: use explicit default factories, remove false
  non-null annotations and related type ignores, and normalize validation behaviour
- Use Pydantic models whenever this work introduces data or configuration classes;
  keep stateless orchestration as functions rather than unnecessary service classes

**Scope:** `src/multimodal_rag/models/config.py`,
`src/multimodal_rag/models/chunks.py`, `src/multimodal_rag/models/sources.py`,
`.env.example`, configuration tests and setup documentation

### INFER-002 — Qualified Dependency and Runtime Refresh

**Branch:** `feature/INFER-002-dependency-refresh`

- Refresh LangChain, provider integrations, Gradio, Pydantic, and Pydantic Settings to
  qualified stable releases; regenerate `uv.lock`
- Add the dedicated Gemini and OpenRouter LangChain integrations
- Remove unused or obsolete dependencies, including the archived
  `langchain-community`, unused `langchain-weaviate`, and redundant `python-dotenv`
- Retain the top-level `langchain` package only if the final implementation imports it
  directly; otherwise depend on `langchain-core` and the provider packages explicitly
- Replace historical `>=0.3` dependency floors with ranges representing versions the
  project actually tests and supports
- Qualify the supported Python range on Python 3.12, 3.13, and 3.14 rather than leaving
  `requires-python` open-ended
- Run dependency upgrades in isolated steps so LangChain, Pydantic, and Gradio
  regressions can be attributed to the package that caused them

**Scope:** `pyproject.toml`, `uv.lock`, imports affected by upstream API changes,
dependency and Python-version documentation

### INFER-003 — Provider Registry and Model Factory

**Branch:** `feature/INFER-003-provider-registry`

- Add typed provider and model registry entries for OpenRouter, OpenAI, Gemini, NVIDIA
  NIM, and Ollama, including display name and streaming, vision, and reasoning
  capabilities
- Extend `src/multimodal_rag/models/config.py` with the typed provider/model selection
  contract consumed by the registry and factory; provider identity must remain explicit
- Keep `.env.example` and `config/development.env` aligned with the supported provider
  keys, endpoints, active provider, and active model so a colleague can configure the
  application without editing source code
- Centralize chat-model construction in one factory returning LangChain's
  `BaseChatModel`
- Use the dedicated provider integrations (`ChatOpenRouter`, `ChatOpenAI`,
  `ChatGoogleGenerativeAI`, and `ChatOllama`) plus NVIDIA's OpenAI-compatible endpoint
  behind the shared factory
- Remove `_make_llm()` and the hardcoded `OPENROUTER_MODELS` list from `app.py`
- Determine provider availability from validated configuration without exposing keys
- Keep chat inference and embedding configuration independent; provider switching in
  Gradio affects chat inference only
- Make adding a model a configuration change and adding a provider a contained factory
  extension, not a Gradio rewrite

**Acceptance criteria:**

- The active provider and model are represented as an explicit typed selection; a model
  name or slash format never determines the provider
- `.env.example` documents every supported chat provider's credential, endpoint, active
  provider, and active model settings without containing real secrets
- The registry exposes only providers with valid configuration and never exposes API
  keys through model data, UI choices, serialization, or logs
- The factory returns the correct dedicated LangChain chat integration for OpenRouter,
  OpenAI, Gemini, NVIDIA NIM, and Ollama, with provider-specific settings applied
  consistently
- The existing cited-answer and KB-article paths work through the central factory
  without provider-specific branches in `app.py`
- Gradio model choices come from the registry; adding a model does not require editing
  the Gradio component code
- Changing chat provider/model leaves embedding provider, embedding model, and existing
  Weaviate vector compatibility unchanged
- Factory, registry, configuration, and application-routing tests pass without network
  calls, and the existing Gradio smoke test succeeds with the configured provider

**Scope:** `src/multimodal_rag/models/config.py`,
`src/multimodal_rag/models/llm.py`, `.env.example`, `config/development.env`,
provider-registry models, application-routing tests, and factory tests

### INFER-004 — Provider-neutral Streaming Inference

**Branch:** `feature/INFER-004-streaming-inference`

- Add provider-neutral streaming APIs for cited answers and KB article drafts using
  LangChain `stream()`/`astream()` semantics and `AIMessageChunk` accumulation
- Emit typed Pydantic progress and completion events containing cumulative text and
  safe response metadata
- Build the final message by combining chunks so completion metadata and token usage
  are retained
- Normalize provider usage metadata into a common shape containing prompt tokens,
  completion tokens, total tokens, and finish reason when the provider reports them;
  do not invent estimates when usage is unavailable
- Record time-to-first-token, total generation latency, provider, model, and final
  completion status without logging secrets or prompt contents
- Enforce a predictable context budget before generation; reduce or truncate
  retrieved context explicitly when the configured budget would be exceeded
- Support cancellation and interrupted streams without returning a misleading
  completed answer or leaving the request in a running state
- Stream numbered citation references during generation, then perform citation-link
  replacement once on the completed answer; never rewrite partial token chunks
- Preserve blocking `invoke()` where progressive output provides no user benefit, such
  as offline vision-description ingestion
- Normalize provider errors into safe, actionable application errors without hiding
  the original cause from logs
- Test multi-chunk responses, empty streams, interrupted streams, provider errors,
  structured provider content, usage metadata, context-budget handling, cancellation,
  latency events, and final citation construction without network calls

**Scope:** `src/multimodal_rag/query/generator.py`, inference event models,
streaming and error-handling tests

### INFER-005 — Gradio Provider Selection and Streaming UX

**Branch:** `feature/INFER-005-gradio-provider-streaming`

- Add separate provider and model selectors; changing the provider filters the model
  choices to compatible configured entries
- Use provider/model capability flags to hide or disable unsupported streaming and
  multimodal selections, with an explanatory status instead of a request that is
  known to fail
- Clearly explain unavailable providers instead of allowing a request that is known to
  fail
- Adapt streaming inference events into cumulative Gradio chatbot and walkthrough
  updates using generator callbacks
- Show immediate progress while retrieval and generation are running
- Show generation status, cancellation state, and normalized token/latency metadata
  after completion when the provider reports it
- Preserve the final `CitedAnswer`, retrieved chunks, question, selected provider, and
  selected model in state for the review-and-save workflow
- Keep provider construction, routing, inference orchestration, and article persistence
  outside the Gradio component definitions
- Preserve clickable citations and the existing answer-to-KB walkthrough across all
  supported providers

**Scope:** `src/multimodal_rag/app.py`, Gradio adapter helpers and UI tests

### INFER-006 — Team Qualification and Operating Documentation

**Branch:** `feature/INFER-006-team-qualification`

- Document first-time setup, configuration, provider credentials, local Ollama use,
  ingestion prerequisites, startup, and common failure recovery
- Document how to add a model and how to implement another provider
- Add mocked contract tests proving each provider satisfies the same streaming and
  metadata behaviour, including structured content blocks and normalized usage
- Qualify the provider capability registry for streaming, vision, reasoning, and
  tool-calling support; capability flags are metadata only and do not add tool use
  to Epic 8
- Add startup tests for configured, unavailable, and misconfigured providers
- Run the full quality and test suite across the supported Python versions
- Manually smoke-test OpenRouter, OpenAI, Gemini, NVIDIA NIM, and Ollama through Gradio,
  including provider switching, streaming, citations, article drafting, and safe
  failures
- Verify a colleague can complete the documented fresh-clone workflow without
  undocumented local knowledge

**Scope:** `README.md`, `CLAUDE.md`, `.env.example`, test configuration, provider
contract tests and release checklist

**Out of scope:** changing embedding providers from the Gradio interface, migrating the
existing Weaviate collection to a different embedding model, adding agent/tool-calling
workflows, or replacing Gradio with another frontend.

---

## Epic 9: Shareable Weaviate Inference Storage (completed)

Provides the simplest MVP way for colleagues to run inference against the same fixed
`SupportChunk` dataset. The application supports either a local single-node Docker
instance or a hosted Weaviate cluster selected through configuration. The collection is
exported once, validated with a compatibility manifest, and stored outside Git. This
epic deliberately excludes application-side ingestion, live synchronization, shared
writes, automatic re-ingestion, and embedding-model migration.

### DATA-001 — Provider-neutral Weaviate Connection

**Branch:** `feature/DATA-001-weaviate-connection`

- Add typed Weaviate deployment settings for `local` and `cloud` modes
- Use an unauthenticated local Docker connection in local mode
- Use HTTPS and an API key in hosted mode
- Keep collection name and optional tenant explicit
- Return actionable connection and authorization errors
- Test both connection paths without network calls

### DATA-002 — Fixed Collection Export and Manifest

- Export the `SupportChunk` schema, properties, stored vectors, and object count
- Produce a manifest containing the Weaviate version, collection name, object count,
  embedding provider/model, vector dimension, creation date, source commit, and SHA-256
  checksum
- Store the export and manifest in an approved location outside Git
- Verify that the configured query embedding model is compatible with the stored vectors

### DATA-003 — Local Docker Restore

- Start a clean single-node Weaviate instance with the project Docker configuration
- Restore the fixed collection artifact
- Verify the collection schema and object count
- Verify one known retrieval query followed by an inference smoke test

### DATA-004 — Hosted Weaviate Bootstrap

- Provide a one-time procedure to load the fixed collection into a hosted Weaviate
  cluster
- Configure the application to query the hosted collection without write access
- Keep hosted credentials out of Git and use read-only access for colleagues where
  supported
- Verify hosted retrieval and inference with the same known query as the local path

### DATA-005 — Runtime Compatibility Validation

- Validate that the configured collection exists before inference
- Validate the expected schema and vector dimension (since SYNC-000 the eight original
  properties are required with matching types, and extra properties are allowed)
- Validate optional tenant availability and read permissions
- Fail with a clear configuration or compatibility message instead of an opaque query
  error

### DATA-006 — Deployment and Restore Documentation

- Update `.env.example` with local and hosted Weaviate settings
- Document local startup, fixed-collection restore, and hosted bootstrap procedures
- Document the embedding-model and vector-dimension compatibility requirement
- Update `docs/PRD.md` and `docs/pipeline.md` with the deployment contract

**Completion:** Local Docker and hosted Weaviate onboarding, restore, compatibility,
viewer-access, and troubleshooting guidance are documented in the README, PRD, and
pipeline documentation. The fixed snapshot remains outside Git.

**Out of scope:** application-side ingestion, live synchronization, shared writes,
incremental synchronization, automatic re-ingestion, schema migration, embedding-model
migration, multi-tenant UI, and collection version management. Maintainer-run
synchronization of content changes into the hosted collection is planned in Epic 11.

---

## Epic 10: Query-Only Ollama Embedding and Cloud Deployment (completed)

Deploys the existing inference application to a managed cloud platform using a local
Ollama embedding service. The `nomic-embed-text` model is baked into the Docker image
and used only for query embeddings. The app reads from the fixed `SupportChunk`
collection and uses configured cloud chat providers. This remains a static-collection
inference MVP: it does not ingest, reindex, mutate, or migrate the collection.

### DATA-007 — Ollama Query Embeddings in a Deployed Container

**Branch:** `feature/DATA-007-hf-space-boot`

- Run the production Gradio application as a Docker container on a managed platform
- Start Ollama as a private, local service in the container and use its embedding API
  for query vectors
- Bake the `nomic-embed-text` model into the Docker image at build time, so container
  restarts need no runtime download and no persistent storage volume
- Configure query embeddings as Ollama `nomic-embed-text` and preserve the 768-dimensional
  collection contract
- Connect to the hosted Weaviate collection with the viewer key and perform read-only
  retrieval
- Keep chat inference selectable between configured OpenAI and Gemini models
- Validate the collection and query-embedding dimension at app startup; surface clear
  errors for missing secrets, missing model, or incompatible vectors
- Add tests for container entrypoint settings and startup behavior without requiring
  credentials
- Smoke-test one known retrieval query and an inference request against the deployed
  service
- Document the deployment platform, image build, and provider settings

**Deployment platform:** Hugging Face Spaces (Docker SDK) was the original target and
is where the Dockerfile, entrypoint script, and baked-in embedding model were built and
first verified end-to-end. The Space was deleted after Hugging Face changed pricing
policy mid-deployment to require a PRO subscription for any compute-backed Space
(Gradio or Docker, public or private — previously only GPU tiers were restricted).
The application redeployed to **Google Cloud Run** with no changes to the Dockerfile,
entrypoint script, or embedding design — only the deployment target changed. Full
setup, IAM, secrets, and a documented browser-access gotcha (Gradio's client computes
its own canonical origin for a heartbeat connection, which breaks when accessed through
an origin-changing proxy) are in `docs/gcp-deployment.txt`.

**Acceptance criteria:**

- The existing collection is never modified by this epic — met
- The deployed container starts Ollama and serves the production Gradio interface — met
- The model ships inside the Docker image and is available immediately on every
  restart, with no runtime download — met
- Ollama produces 768-dimensional query vectors compatible with the current collection —
  met, verified via `validate_compatibility()` at startup in both the local Docker test
  and the deployed Cloud Run service
- A known retrieval query and one chat inference complete against the existing collection
  without Weaviate writes — met, verified against the live Weaviate Cloud collection
  through the deployed service in a browser
- OpenAI and Gemini are selectable when their secrets are configured — met; both serve
  live requests on Cloud Run (verified during Epic 11, November 2026)
- Missing runtime configuration and incompatible collection/vector dimensions produce
  actionable startup errors — met (`space-entrypoint.sh` fails fast on missing
  `WEAVIATE_URL`/`WEAVIATE_VIEWER_API_KEY`; `validate_compatibility()` fails fast on
  dimension mismatch)
- Deployment setup and test outcome are documented — met, `docs/gcp-deployment.txt`
- Misconfiguration or incompatible dimensions produce an actionable error — met

**Out of scope:** ingestion, reindexing, collection migration, embedding fingerprint
storage, other embedding providers, production-scale autoscaling or high availability, and
any new vector database functionality. Maintainer-run content synchronization is planned
separately in Epic 11.

---

## Epic 11: Operational Observability, Hosted Content Sync and Production Hardening (in progress)

Makes the live tool observable and safe to operate, then lets a maintainer propagate
source content changes into the hosted `SupportChunk` collection without a rebuild or an
app redeploy. The embedding contract does not change (Ollama `nomic-embed-text`, 768
dimensions). Sync writes happen only from a maintainer machine with the admin key; Cloud
Run keeps the read-only viewer key.

The tool is in use, so Phase 1 (observability) ships first. It is small, low-risk, and
shippable on its own. Phase 2 (sync) writes to production data and reuses the Phase 1
logging.

**Status:** Phase 1 (OBS-001 to OBS-003) is complete, merged to `main`, deployed and
verified against live traffic on Cloud Run revision 00021 (2026-11-13). Verification
found and fixed three defects after the first deploy: Gemini token usage (reported per
chunk as deltas, now summed), the Gemini function-calling log noise (filtered), and
OpenAI streamed usage (now requested explicitly). Phase 2 (sync) has started: SYNC-000
(schema readiness) is done and deployed (revision 00024, 2026-11-13) and the local
Weaviate now matches the hosted version. Phase 2 was reworked on 2026-11-16 so that
automating the sync (SYNC-009) needs a scheduler and a policy, not a rewrite: plan and
apply are separate artifacts, state lives in a ledger in the bucket, and the first
syncs need no change to the hosted schema (see "Design rules for an automatable sync" and
"Order of work"). SYNC-002 is next; SYNC-008 is partly done; SYNC-001 and SYNC-010 are
deferred; SYNC-005 and SYNC-006 are dissolved. Phase 3 (production hardening, OPS-001 to
OPS-008) was added after an audit of the live service on 2026-11-13 and waits until the
sync works; none of it has started. The design and the supporting evidence
are in `docs/chunk-parent-design.md`; the operating model is in
`docs/content-update-workflow.md`.

### Personas and update workflow

Full detail, facts and assumptions: `docs/content-update-workflow.md`.

| Persona | Role in Phase 2 |
|---|---|
| Support colleague | Uses the hosted app with the read-only viewer key. Needs no outage while content is updated. Never runs ingestion. |
| Maintainer | Owns the sources, runs the refresh and the promotion, holds the Firecrawl, Mistral, vision and admin keys, and decides what goes live. |
| Operator | Deploys the app, keeps the schema compatible, reads the Phase 1 logs. The same person as the maintainer today. |
| Content owner | Edits the documentation and videos. Not a system user; their change rate sizes the workload (about 11 edits, 7 new pages and 1 removal in eight months). |

The sync is **two stages**, because the local Docker collection and the hosted collection
are identical today (922 UUIDs, identical text) and play different roles:

- **Stage A, refresh local** (SYNC-002 to SYNC-004, SYNC-008): the maintainer captures the
  sitemaps, plans the change against the page ledger, applies the reviewed plan to the
  local collection only, and runs the retrieval gate. Local is a disposable staging copy.
- **Stage B, promote to hosted** (SYNC-007): the maintainer backs up the hosted
  collection, diffs local against hosted, copies the difference (vectors included) with
  the admin key without deleting the collection, and verifies that the two are identical
  again.

Prerequisites for a run, and the gaps found today (an untracked source list that omits
the three knowledge bases, four duplicate `VISION_MODEL` lines, one `.env` shared by both
targets, local Weaviate was 1.28.4 against hosted 1.39.13 until aligned on 2026-11-13), are in section 4 of the workflow
note and in SYNC-008.

### Phase 1: Observability

#### OBS-001 — Central Logging Configuration and Correlation (done, deployed, verified)

**Branch:** `feature/OBS-001-logging-config`

- Replace the two separate `logging.basicConfig` calls (`app.py`, `ingest/__main__.py`)
  with one `configure_logging(settings)` shared by the app, ingestion, and sync
- JSON output on Cloud Run (detected via the `K_SERVICE` environment variable, or
  `LOG_FORMAT=json`) with a `severity` field Cloud Logging understands; readable plain
  text locally
- A correlation ID set per question (context variable) and included on every log line
  from retrieval through inference
- Add DEBUG-level logging at decision points: provider/model selection, retrieval
  counts, and context truncation (log when `_truncate_context` actually cuts, with
  before/after size)

**Scope:** new `src/multimodal_rag/logging_config.py`, `app.py`, `ingest/__main__.py`,
`query/generator.py`

#### OBS-002 — Inference Metrics Logging (done, deployed, verified)

**Branch:** `feature/OBS-002-inference-metrics`

- Emit exactly one structured INFO record per generation, whether completed, errored, or
  cancelled: provider, model, status, finish reason, input/output/total tokens (or an
  explicit "unreported"), time-to-first-token, total latency, chunks retrieved, context
  tokens used vs the budget, and whether truncation occurred
- Never include prompt, question, or answer text
- Closes the Epic 8 requirement that provider, model, completion status, token usage, and
  latency be available for operational logging. `InferenceEvent` already carries these
  values, but nothing writes them to the logs.
- Document two Cloud Logging queries: error rate by model (provider capacity errors) and
  p95 latency by model. Documented in the README; the filter queries were run, and Log
  Analytics is enabled on `_Default`, but the p95 SQL itself has not been run

**Scope:** `src/multimodal_rag/query/generator.py`, `app.py`, `models/inference.py`

#### OBS-003 — Logging Privacy and Secret Safety (done, deployed, verified)

**Branch:** `feature/OBS-003-logging-safety`

- Decide and document the policy for user query text. Default proposal: do not log it;
  log only its length and a short hash, with full text at DEBUG in local runs only.
  Change `query/retriever.py`, which currently logs the first 60 characters of every
  query at INFO.
- Redact API keys, bearer tokens, and credentialed URLs from exception text before
  `logger.exception` calls
- Mocked tests for each provider's error path asserting no secret appears in captured
  log output
- Document log retention for the Cloud Logging bucket

**Scope:** `query/retriever.py`, `query/generator.py`, logging helpers, tests

### Phase 2: Hosted content sync

Current behaviour that motivates this phase: web chunk IDs derive from either
`url|index` or `url|text` (`SupportChunk.from_web_chunk`), `WeaviateStore.add_chunks`
uses a plain batch insert, and no change detection exists. A spike on Weaviate 1.28.4 and
1.39.13 (the hosted version) showed that re-ingesting a chunk in the same position
overwrites it completely, but stale chunks still arise when positions shift, when content
is removed, and for screenshot chunks keyed on image URLs that change. Removed content
lingers. Because an overwrite replaces the whole object, a plain re-ingest also erases
any property it does not write, such as backfilled provenance fields.

#### Design rules for an automatable sync

The first runs are human-run, but the stories are built so that automating them
(SYNC-009) needs a scheduler and a policy, not a rewrite. These rules apply to every
story below.

1. **Plan and apply are separate artifacts.** `sync-plan` writes a JSON plan with a
   content hash, bound to the registry generation and the ledger generation it was built
   from. `sync-apply` takes that plan file and executes exactly it; it refuses when
   either generation has moved. What a person reviewed is what runs.
2. **Policy is configuration.** `config/sync-policy.yaml` holds the limits that decide
   whether a plan may proceed: deletion share, changed pages, new pages, Firecrawl pages,
   embedding calls. A person approves by running apply; automation may approve only a
   plan inside policy, and anything outside it stops for review.
3. **State is a ledger, not chunk properties.** The bucket holds a page ledger (URL,
   `lastmod`, text fingerprint, chunk count) written only after a verified run. The next
   run diffs against it. It needs no change to the hosted schema.
4. **Every run writes a manifest** to `runs/<timestamp>/`: registry and ledger
   generations, plan hash, fetched page content, per-page results, gate result, snapshot
   reference, counts. It is both the audit trail and the reproducible input of the apply.
5. **Machine-readable by default.** Every sync command has `--json` and distinct exit
   codes: 0 done or nothing to do, 10 plan outside policy (needs review), 1 failure. The
   existing preflight and gate are retrofitted (SYNC-008).
6. **One run at a time.** A lock object in the bucket, taken with a generation
   precondition (as the registry push already does), blocks overlapping runs.
7. **The core is target-agnostic.** Apply and verify take a target. Local Docker is the
   staging target today and the write guard keeps hosted writes inside promotion, but
   nothing in the core assumes Docker, so a hosted staging collection can be added later.
8. **Two lanes.** The web lane (sitemap and `lastmod` driven) is the automatable one. The
   video lane (new URL driven) stays maintainer-triggered: YouTube blocks datacenter IPs
   and the cookies expire.
9. **Determinism.** Hash inputs, never regenerated outputs; fingerprints exclude
   `ingested_at`; a run never re-describes an image that did not change.
10. **`make ingest` and `make sync` are the same pipeline.** They share one normaliser
    and one chunker (`chunk_page`), so a page chunks identically whichever path wrote
    it. A test pins that ingest uses the shared function.
11. **Designed for template change.** The normaliser recognises the site template by its
    wording, so every run checks that it still does (`check_template`): the share of
    fetched pages whose byline and footer rules still match must stay at 90% or more; a
    footer cut larger than 1,500 characters is not trusted and the page is flagged and
    left whole; and any paragraph that repeats on at least half of the fetched pages
    (three or more) and was not removed is reported as probable new boilerplate. A
    finding makes the plan need review (exit code 10) instead of silently chunking the
    new boilerplate into every page. Ingest logs the same findings as warnings. A short
    page is reported but is not a template failure

#### Order of work

The story numbers are stable identifiers, not the order. The order they were built in, and
what is left:

| Order | Story | What | Status | Hosted impact |
|-------|-------|------|--------|---------------|
| 1 | SYNC-002 | Page ledger, inventory and plan (read-only) | done | None |
| 2 | SYNC-003 | Apply a plan to the local collection, then the retrieval gate | done | None |
| 3 | SYNC-004 | Removals, behind explicit confirmation | done | None |
| 4 | SYNC-007 | Promote to hosted: snapshot, copy vectors, verify | done | The only hosted write |
| 5 | SYNC-008 | Preflight, registry, runbook | done for manual operation | None |
| 6 | SYNC-001 | Stored identity and conversion | done, deployed | Schema change, done |
| 7 | SYNC-012 | Video frame descriptions: prompt and chunk structure | part 1 built and merged (89d989a); parts 2 and 3 are SYNC-013 and later | Local, then promotion |
| 8 | SYNC-011 | Video lane: new videos first, changed videos later | new-video half built and merged; first real run is next (needs the `.env` change and the registry entry) |
| 9 | SYNC-013 | Split the existing video chunks into narration and description | ticket written 2026-11-23; four decisions open; not started | Local, then promotion |
| Later | SYNC-010 | Screenshot re-description (an experiment behind the gate) | deferred | Replaces of screenshot chunks |
| After OPS-001, 003, 004 | SYNC-009 | Automate the web lane | stub | Via promotion |

SYNC-005 and SYNC-006 are dissolved into the stories that need them.

#### SYNC-000 — Schema Readiness (done, deployed 2026-11-13)

**Branch:** `feature/SYNC-000-schema-readiness`

Gate for every later sync story. The collection schema is checked for an exact property
match: `WeaviateStore.validate_compatibility` runs at app startup and rejects any extra
property. Adding a property to the hosted collection before the deployed app is updated
would make Cloud Run fail its startup check and return 503.

- Change `validate_compatibility` to require the 8 original properties and allow extra
  ones; new properties are optional until a feature needs them
- Add the planned optional properties to `ensure_collection`, so freshly created
  collections match (`restore_snapshot` already builds its schema from the manifest)
- Update the schema tests, including the `EXPECTED_PROPERTIES` fixture
- Deploy this to Cloud Run and verify startup **before** any property is added to the
  hosted collection
- Spike on throwaway collections in local Docker, not on the hosted one: what
  `batch.add_object` does when the UUID already exists; whether adding a property needs a
  reindex; and whether a partial `data.update` leaves the stored vector untouched. Done on
  Weaviate 1.28.4 and 1.39.13 (the hosted version), 18 of 18 checks on both. Results:
  `batch.add_object` on an existing UUID **replaces the whole object** (text, vector, and
  any property not supplied becomes null); `data.insert` rejects a duplicate (422);
  `data.replace` swaps an object by UUID; `data.update` merges and keeps the vector;
  `add_property` works on a populated collection with no reindex for reads or filters.
  Recorded in `docs/chunk-parent-design.md`

**Scope:** `src/multimodal_rag/store/weaviate.py`, `tests/test_store.py`

#### SYNC-001 — Stored Identity, Conversion and Chunk Quality (done, deployed 2026-11-17)

**Branch:** `feature/SYNC-001-chunk-identity`

Depends on SYNC-000. **Not a prerequisite for the first syncs.** The plan computes a
fingerprint from the stored text on both sides, which is deterministic at this scale
(160 pages, 922 chunks) and needs no schema change; embeddings run locally, so skipping
unchanged chunks saves no cost. Take this story when scale, a verification need, or
SYNC-010 requires stored values.

- Add `document_id` (`uuid5(source_url)`), `content_hash` and `source_lastmod` as
  additive properties, written on **every** insert. This must ship before any backfill:
  a re-ingest replaces whole objects, so a chunk written without these fields would erase
  backfilled values
- Hash inputs, never generated outputs (design note D7)
- Convert the existing chunks without a crawl (922 when this was written, 943 when it
  ran): `document_id` and a `content_hash`
  from stored text for every chunk, `source_lastmod` where the page is unchanged, `legacy`
  markers for fields that cannot be known. Run it on the local collection with
  metadata-only `data.update`, verify texts and vectors are unchanged, then promote
  (design note, Conversion of the existing chunks)
- Keep the embedded video IDs as page metadata. The header and footer boilerplate is
  now removed by the normaliser in SYNC-002; the YouTube player text that March captured
  is no longer returned by fresh scrapes, and the existing chunks keep it until a
  rebaseline
- Write every property on each object, because an overwrite is a full replace
  (`data.replace` or `batch.add_object` by explicit UUID); use `data.update` for
  metadata-only changes

**Scope:** `models/chunks.py`, `store/weaviate.py`, `ingest/web.py`

**Built (2026-11-17), on the epic's bullets only** (the design note's screenshot and video
fields wait for SYNC-010 and SYNC-011, which read them):

- `models/identity.py`: `document_id` is `uuid5` of a **canonical** URL, so
  `https://youtu.be/<id>` and `https://www.youtube.com/watch?v=<id>` are one document (the 9
  `youtu.be` videos would otherwise have hashed differently). The stored `source_url` is
  never rewritten, because chunk UUIDs derive from it. `content_hash` is the full SHA-256 of
  the stored text. For a screenshot chunk that text is a generated description, so this hash
  describes what is stored and **must never decide a re-description** (D7).
- `embedded_video_ids` is a list of the **`document_id`s** of the videos a page links, on
  every web chunk of the page (the maintainer's point: a video id is a document id, not a
  second identity). All 19 videos embedded in pages resolve to an ingested video.
- `source_lastmod` (date): set on write by `chunk_page`/apply, and by the conversion from the
  sitemap for pages the inventory finds unchanged; a stored value is never overwritten with
  nothing. Apply also gives every **other** chunk of an applied page (unchanged text and
  screenshots) the page's new `source_lastmod` and videos with a metadata-only update, so a
  page's chunks never disagree. Videos are the **union** of what the stored page already
  links and what the fresh scrape finds, because fresh scrapes mostly drop the links the
  March text still has; a plain overwrite would have erased them on the first sync.
- Every writer emits all four fields (`SupportChunk`, `add_chunks`, `chunk_page`, apply);
  `put_objects`, the promotion digest, snapshot export and restore were already
  property-generic. A new property reads `None` until set and an explicit `[]` reads back as
  `[]`, so writers always send the list.
- `WeaviateStore.extend_schema()` adds missing optional properties with their declared
  types; writes never create them (a date would be guessed as text and cannot be changed on a
  populated collection). `update_metadata()` is a metadata-only `data.update`
  (verified: the vector is bit-identical afterwards).
- `make sync-convert` (`DRY=1`): local only; snapshot, extend schema, update, then verify
  object count and a checksum of every text and vector.
- **Done on local:** 943 chunks converted (`runs/convert-20261117T132848Z`), 729 web chunks
  with `source_lastmod`, 118 web chunks on pages that link a video, 25 videos with 25
  distinct `document_id`s. The promotion plan sees 943 replaces and 0 vector refreshes.
- **Startup check:** the app's `validate_compatibility` tolerates unknown properties; the
  code of commit `d471a57` (in the deployed revision `00024`, inferred from timestamps)
  passes against the converted collection.
- **Promotion creates the target's properties:** the plan lists the properties the target
  lacks and needs review for it (a property cannot be removed again); `promote` calls
  `extend_schema()` after the backup and before the canary, so a write never auto-creates a
  property with a guessed type. `promote-rollback` leaves the added properties in place.
- **Checked against the deployed code:** commit `d471a57`'s `validate_compatibility`,
  `search` and retriever all work against the converted collection.
- **Rehearsed (2026-11-17)** on a copy seeded from hosted backup
  `snapshots/20261117T133802Z-cloud`: the plan matched the hosted plan in every count; the
  canary created the five properties with the right types (`date`, `text[]`) and read back
  equal; the full promotion wrote 942, the gate kept its ranks (1, 1, 4) and a new plan
  showed 943 unchanged; the app's startup check passed on the result. **The rehearsal found
  a bug:** rolling back to a snapshot from before the schema grew restored the data but
  exited 1, because the check treated a property the object never set (absent in the
  snapshot, null once the schema has it) as a difference. Verification now compares only
  properties that hold a value; the plan still compares strictly. After the fix the
  rollback reports clean and a second promotion from the rolled-back state works.
- **Hosted promoted (2026-11-17):** the canary (backup `snapshots/20261117T134631Z-promote-before`)
  created the five properties on hosted and wrote one object; the full run (backup
  `...T134836Z-promote-before`) wrote the other 942, verified every object, kept the gate
  ranks (1, 1, 4) and took `snapshots/20261117T134901Z-promote-after`. Checked independently:
  a new plan shows 943 unchanged, the preflight shows 26 ok, hosted has 943 chunks that all
  carry `document_id` and a 64-character `content_hash`, 729 web chunks carry
  `source_lastmod`, 118 web chunks link a video (19 distinct videos, all resolving to an
  ingested one), and the app answers.
  **Deployed (2026-11-17):** revision `multimodal-rag-00025-juf` (main at `6129516`) was
  deployed with `--no-traffic --tag`, so it cold-started against the new schema while
  `00024` kept serving. It became ready in 2m5s, its logs show the startup check running
  (Ollama 0.40.14, same weights), a real question through the tagged URL returned a cited
  answer from the right HydroMan article, and there were no errors; traffic was then
  shifted (`--to-latest`) and the tag removed. Production answered a second question
  afterwards. Rollback: `gcloud run services update-traffic multimodal-rag
  --to-revisions multimodal-rag-00024-ndg=100`. **Restore only from `20261117T134901Z-promote-after` or later**
  (older backups lack the identity fields; the schema stays either way). The promote policy does not limit a plan of property-only
  replaces, so the plan must be read.

#### SYNC-002 — Page Ledger, Inventory and Plan (done, 2026-11-16; read-only)

**Branch:** `feature/SYNC-002-sync-plan`

Writes nothing to Weaviate. It answers, before any change is built, how much has really
changed and what the crawl missed.

- **Baseline ledger.** Build it once from the local collection (verified identical to
  hosted by the preflight): per stored page its URL, knowledge base, text-chunk count,
  frozen-chunk count and `ingested_at`, with the text fingerprint left **null**. Write it
  to the bucket under `ledger/`, created exactly once (`--if-generation-match=0`). Today
  the only record of the 160 stored pages (hydrosym 99, hydroman 48, hydrocam 13) is
  `source_url` on each chunk. **Telling the layers apart:** screenshot description chunks
  are stored with `source_type="web"` like page text, so no property separates them. A
  chunk is page text only if its UUID re-derives as `uuid5(url|index)`; all 327 text
  chunks do, with contiguous positions on every page, and the other 381 (the screenshots)
  do not. Sync treats every non-derivable chunk as frozen
- **Inventory.** Capture each knowledge-base sitemap (URL and `lastmod`) to
  `captures/<timestamp>/`. The registry holds only root URLs; the pages come from the
  sitemap, never from the crawl
- **Classify** each page: modified (`lastmod` newer than the ledger, or than
  `ingested_at` for the baseline), new (in the sitemap, not stored), removal candidate
  (stored, not in the sitemap), unchanged. Pages without a `lastmod` are always
  refetched; `--verify` refetches everything. Measured 2026-11-16 (in-scope `/article/`
  pages, exact host match): 11 modified, 12 not stored, 1 removal candidate, 148
  unchanged and not fetched. Two cautions: the plan must filter the sitemap to the same
  `/article/` scope as the crawler, and it must match on the exact host (hydroman article
  slugs contain "hydrosym")
- **The crawl cap is a defect.** `ingest/__main__.py` crawls `limit=100` per knowledge
  base and hydrosym holds 99 pages against 109 in its sitemap. Of the 12 pages not stored, 5
  are old build release notes (`lastmod` 2025-11-10) that existed at the March ingest, which
  is consistent with the cap but could equally be empty or failed scrapes; 7 are newer. Sync never uses the cap: the plan lists the pages, and only
  those are fetched. The plan reports sitemap pages the old crawl never reached
- **Stored text is not a change signal (checked 2026-11-16).** Re-scraping pages that
  predate the March ingest does not reproduce their stored chunks: the site template
  changed (a new title and "Updated … min read" header, a "Still stuck?" footer, and the
  YouTube player text March captured is gone), and the chunk counts differ. So the
  baseline fingerprint is null, `lastmod` is the only trigger, and a fetched page's text
  chunks are replaced as a whole set. The fingerprint is recorded from the first fresh,
  normalised fetch and is comparable from then on. A one-off `--rebaseline` that fetches
  every page (about 171 Firecrawl credits) would give all pages a comparable fingerprint
  and one consistent extraction; it is optional and decided at the first run
- **Normalise before chunking.** The template adds page furniture that is not
  documentation: a byline (`HydroManUpdated 27 November 20252 min readPrint`) at the top
  and, from the feedback widget down, "Did this answer your question?", "Last updated
  on", "No results found" and a "Still have questions? ... support@paro.nl" call to
  action. Decided 2026-11-16 and built (`ingest/web.py`): remove the byline; cut
  everything from `Did this answer your question?` to the end; keep the page title in the
  first chunk; keep images in the fetched content (the screenshot pipeline reads them)
  with an optional `strip_images` for the text chunks, off by default. The fingerprint is
  the SHA-256 of the normalised page plus `NORMALISER_VERSION`, so it ignores the byline
  and footer dates; bump the version when a rule changes. All 160 stored pages already
  end with the feedback widget, and 46 chunks hold embedded YouTube player text, so the
  index carries this noise today and only fetched pages get cleaned until a rebaseline.
  Without the cut, the new call to action would sit in the last chunk of every refetched
  page and match any "contact support" question equally. `make ingest` still uses the
  old chunker and is not changed yet
- **Fetch and chunk** each modified and new page once, store the fetched content under
  `runs/<timestamp>/pages/`, and compute the per-position difference against the stored
  chunks. Apply then uses this stored content and fetches nothing, so it is
  reproducible. Built as `make sync-preview` (`FROM_RUN=<dir>` reuses a fetch, `ONLY=<url>`
  limits it); the fetched pages stay under `runs/`, which is git-ignored. A position is
  identical, **template-only** (it differs only by the new title or the removed footer),
  changed, added, or removed (trailing positions); frozen screenshot chunks are never part
  of the diff
- **First preview, run 2026-11-16 against local Docker:** 23 of 23 pages fetched, the
  byline and footer rules matched all 23. Of the 11 pages `lastmod` marks modified, **6
  are identical to the stored text once the title and footer are set aside** and only 5
  have content that differs: two are links that moved from the staging domain
  `paro.1902.studio` to `paro.nl`, one lost the YouTube player text March had captured,
  one has a real edit (new text and images), and one was not inspected. So `lastmod`
  over-reports, and the stored index holds 4 links to the staging domain on 3 pages. The
  12 pages not stored chunk into 1 to 7 chunks each. One page (article 276) is a
  one-sentence stub whose content is probably an embedded video
- **Plan artifact (built):** canonical JSON with a SHA-256 hash, the registry and ledger
  generations it was built from, and per page: kind, reason, whether the content really
  changed, the title, the fingerprint, stored and fresh chunk counts, the fetched file
  with its checksum, and every position as write, skip or delete with its UUID. It also
  carries the removal candidates (with the screenshot chunks a delete would orphan), the
  fetch failures, the template findings, the canaries for pages whose content changed
  (`retrieval_gate.from_changed_pages` form) and the policy result. `staleness()` tells
  an apply why it must refuse a plan: a hash that does not match, or a registry or
  ledger that moved. Template-only positions are written, so the index gets clean
  chunks, but such a page does not count as content-changed
- **Policy (built):** `config/sync-policy.yaml` limits pages to fetch, new pages, pages
  with changed content and the share of page-text chunks deleted (30, 30, 30 and 10%).
  A plan outside a limit, with fetch failures, with template findings, or with a
  registry that moved since the pull needs review. Removal candidates never decide the
  status: they always wait for an explicit confirmation (SYNC-004)
- **`make sync-plan` (built)** fetches (or reuses `FROM_RUN=`), builds the plan, writes
  `plan.json` and `manifest.json` next to the pages, and uploads the run to
  `runs/<run id>/` in the bucket (`NO_UPLOAD=1` to skip); `--json` prints the plan, and
  the exit code is 0 within policy, 10 needs review, 1 failure. It writes nothing to
  Weaviate. The run lock is taken by the apply (SYNC-003), where it matters
- **First plan, 2026-11-16:** the baseline ledger is saved (`ledger/pages.json`, created
  once). The plan for the 23 pages: 12 new and 5 with changed content, 6 template-only,
  42 positions to write, 9 skipped, no deletes, 61 screenshot chunks on those pages left
  untouched, 17 canaries, policy within limits, one removal candidate waiting for
  confirmation. The kept run was read back from the bucket: the hash is valid and every
  page matches its checksum
- Reuse per-source error isolation: one failed source never aborts the run

Tests and docs ship with this story.

**Scope:** new `src/multimodal_rag/sync/` package (ledger, inventory, plan, policy,
manifest), `config/sync-policy.yaml`, `Makefile`

#### SYNC-003 — Apply a Plan to the Target Collection (done, 2026-11-17; reaches hosted through SYNC-007)

**Branch:** `feature/SYNC-003-sync-apply`

**Built and merged (2026-11-17):** `make sync-apply RUN=runs/<id>` validates the plan
(hash, format 2, normaliser and chunk size, registry and ledger generations, policy again
at apply time), takes a create-only lock in the bucket, re-derives every chunk from the
stored pages and proves each write matches the plan's text hash, checks the collection
still looks as the plan expects (a page already applied is skipped, so an unfinished run
can be repeated), snapshots local into the run, writes every position and only then
deletes by explicit UUID, verifies per page and in total (screenshots and videos
unchanged), runs the retrieval gate with the plan's canaries, and updates the ledger only
if all of that held. Also fixed along the way: `add_chunks` never checked for rejected
batch objects and now raises (ingest and sync share that path). The plan now carries the
sitemap `lastmod` and a text hash per write (format 2).

**First real apply, 2026-11-17: written correctly, stopped by the gate.** 25 inserted, 17
replaced, 0 deleted, 922 to 947 chunks, verification passed, then the gate failed on the
weakest canary (the "import into HydroMan" question): after the apply the whole top 10 was
different and its expected sources left the top 40. The ledger was not updated and local
holds the applied state; hosted is untouched. **Root cause: vector drift.** The chunks
written today match their own stored vectors exactly (cosine 1.0000), but the March
vectors do not reproduce from their own text: mean cosine 0.83 (range 0.64 to 0.92, on
short chunks through the production embedding helper, either input format, equal norms).
Today's embedding function differs from the one that made the March vectors, for web
text, video and screenshot chunks alike, and the `nomic-embed-text` digest is unchanged
(`0a109f422b47`). Fresh vectors therefore score about 0.1 higher against the same query
than old ones and outrank them regardless of relevance, so any partial apply biases the
ranking toward the new chunks. The embedding input changed on 2026-03-13 (`text` to
`source_name: text`), but that does not explain it. Local Ollama is 0.40.16 and the
production image pins 0.40.14; what generated the March vectors is not known

**Cause, as far as it is known:** most likely Ollama version drift. Upstream issue
ollama/ollama#14449 documents `nomic-embed-text` output differing across versions (about
0.96 cosine on medium text, 0.84 on long, a casing bug from 0.13.1 fixed in the 0.30
series). The March ingest ran on about v0.17.7 and the Mac app has since auto-updated to
v0.40.16. Capital-heavy chunks match worst (0.88 at 1 to 3% capitals, 0.77 at 6% and up),
but even the least capitalised only reach 0.88, so casing is a part and not the whole.
The model weights are not the cause: the registry's current `:latest`, the Mac and
production all serve the same blob (`970aa74c0a90`). Not proven: running v0.17.7 side by
side was blocked by the permission classifier.

**Built in response (2026-11-17):** (1) the embedding runtime is pinned in `compose.yaml`
(`ollama/ollama:0.40.14`, the Dockerfile's version, kept equal by a test, bound to
localhost:11435, restarted with Docker; `make ollama-up` starts it and pulls the model),
and `EMBEDDING_BASE_URL` points embeddings at it without moving the chat models. (2) The
pinned container and the Mac's 0.40.16 agree to cosine 0.999992 or better on 70 real
chunks, so production's version family is safe to rely on. (3) `make sync-reembed`
(`DRY=1` to measure) recomputes every vector from its stored text with the runtime and
replaces only the vector, snapshots first, and reads everything back to prove properties
and counts did not change. The dry run over all 947 chunks found 905 off-spec (web mean
0.86, min 0.67; video mean 0.85, min 0.76). (4) An embedding fingerprint (Ollama version
and weights digest) is recorded in the ledger. The preflight now checks the runtime against
production's pin and samples stored vectors against fresh embeddings (OK at 0.9999 or
better, warn from 0.99, fail below), for local and, as information, for hosted; the apply
refuses on a wrong runtime, on drifted vectors, or on a runtime that differs from the one
the ledger was last verified against. Production's Linux build was checked on
2026-11-17 (below)

**Production check (2026-11-17):** Cloud Run revision 00024's logs show it loading the
same weights blob (`sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6`,
F16, GGUF V3), running the llama-server runner on the `libggml-cpu-haswell` backend (x86
AVX2) with a context of 2048 (Ollama asks for 8192 and logs a warning). Docker on the Mac
runs `linux/arm64`, so the same 0.40.14 was also run as `linux/amd64` under emulation: it
loaded the same haswell backend with the same CPU feature line, and its vectors agree with
the arm64 container to cosine 0.999992 or better and with the Mac's 0.40.16 to 1.000000
(40 real chunks, 8 to 707 words, none below 0.9999). Production's logs never print the
Ollama version, so 0.40.14 rests on the Dockerfile pin and the log format, not on a log
line. Copying the locally made vectors to hosted is therefore safe at the precision any
two runtimes of this family give

**Outcome (2026-11-17):** after `make sync-reembed` (947 vectors rewritten, nothing else
changed, an independent sample of 120 chunks at cosine 1.000000) the gate passed on all 20
canaries. The weak "import into HydroMan" question went from rank 6 on hosted to **rank 4**
on local, so consistent vectors improved that answer. The apply then finished its
"nothing to write" pass and updated the ledger (generation 1794914873468520, 172 pages,
embedding runtime recorded). Local holds 947 chunks, 25 more than hosted; hosted is
untouched and its vectors are still the March ones (lowest 0.76, mean 0.83 against a fresh
embedding), which the promotion (SYNC-007) replaces. Also found and fixed: a module run
with `python -m` is loaded twice, so the preflight's `Status` and the one `embedding_check`
imported were different enums and `is` comparisons failed silently, which could have made
the preflight exit 0 on a FAIL; `Status` and `Check` now live in `checks.py`

**Environment note:** the first write attempt of the re-embed failed because Docker
Desktop's disk was 99% full and Weaviate set the shard read-only at 90%. Freeing the build
cache and three obsolete images brought it to 63% and Weaviate recovered by itself. Keep
the Docker disk below 80%: a full disk fails writes, and the pinned Ollama image alone is
4.3 GB
- `make sync-apply PLAN=<file>` verifies the plan hash and that the registry and ledger
  generations still match, takes the run lock, and refuses a stale plan. The core has no
  prompt; the `make` wrapper asks for confirmation unless `--yes`
- The target is a parameter and defaults to local Docker. A hosted target is refused by
  the store's write guard; hosted writes happen only in SYNC-007
- Per page, replace the chunks as a set: write changed and new positions with their
  vectors (embedded now, from the stored content), leave unchanged chunks untouched,
  and delete leftover trailing positions last. Insert and update first, delete last, so a
  page is never empty or partial mid-run. Write every property on each object
- **Frozen layers:** the web lane never changes or deletes a screenshot or video chunk,
  and a test says so
- Enforce the policy limits again at apply time, including an abort when the plan would
  delete more than the allowed share of the collection
- Refuse to run if the embedding model or dimension does not match the collection
- Verify counts against the plan, then run `make sync-gate` with the plan's changed-page
  canaries and the fixed canaries. The ledger is updated **only** after verification and
  the gate pass; the manifest records the result

Tests and docs ship with this story.

**Scope:** `src/multimodal_rag/sync/`, `store/weaviate.py`, `Makefile`

#### SYNC-004 — Removed Pages and Sources (done, 2026-11-17; reaches hosted through SYNC-007)

**Branch:** `feature/SYNC-004-removals`

- **Candidates.** A stored page is a removal candidate when it is missing from its
  source's sitemap, or when its whole source (a knowledge base) has been dropped from the
  registry. Video sources are the video lane's (SYNC-011)
- **The site decides, not the sitemap.** A page missing from a sitemap is only a hint, so
  the plan asks the page's own address, without following redirects: 404 or 410 is
  **gone**, a redirect is **moved** (deletable only if its target is held), a 200 is
  **still served** (the sitemap simply omitted it, never deletable), and a timeout or 5xx
  is **unknown** (never deletable). The result, with the HTTP status and any redirect, is
  in the plan and its hash. Real case: article 437 returns 404 with no redirect, while its
  probable replacement 525 returns 200
- **Confirmation by name.** `make sync-apply` deletes nothing unless a person passes
  `CONFIRM_REMOVALS='<url> ...'`, and every URL must be a candidate in the plan (a typo is
  refused) whose evidence supports it. Pages of a dropped source are confirmed by source
  with `CONFIRM_SOURCE='<host>'`, not page by page. `ApplyOptions` defaults to confirming
  nothing, so an automated run can add and change pages but never delete one. A page whose
  slug resembles a stored or new page is shown with that hint, and if the plan names a
  replacement it must already be held (or be written by the same plan) before the old page
  may go
- **Exact UUIDs.** The plan lists every chunk of the page, screenshot chunks included, and
  the apply deletes exactly those, last, after all writes. If the page has gained a chunk
  since the plan it refuses; if it already lost some (an unfinished run) it deletes the
  rest. It verifies that no chunk of the page is left, then drops the page from the
  ledger. `delete_by_source` is never used, because it would take a page's screenshots
  with it unseen
- **Limits.** The deletion-share limit counts confirmed removals too, and a source that
  has more than 10% of its stored pages missing from its sitemap needs review, because that
  looks like a sitemap fault and not a clean-up (`max_removal_share_per_source`)
- **A failed or implausible sitemap never deletes** (SYNC-002's guard: a host whose sitemap
  failed, is empty, or lists under half its stored pages produces no candidates at all)
- Also changed: the plan format is 3, an apply that changes nothing no longer rewrites the
  ledger or moves its generation, and a rename hint now searches every page the sitemap
  still lists instead of only new ones
- **Real data, 2026-11-17:** article 437 (4 text chunks, 0 screenshot chunks) returned 404
  and its probable replacement 525 is held. `make sync-apply` without confirmation left it
  and the total of 947 chunks untouched and listed it as waiting. A plan built before an
  unconfirmed apply was then refused as stale (the ledger had moved), which is the check
  working; a rebuilt plan with `CONFIRM_REMOVALS` naming 437 deleted its 4 chunks (947 to
  943, videos still 214) and dropped it from the ledger, which now holds the 171 in-scope
  pages of the three sources. The inventory afterwards shows nothing modified, new or
  removed, and the preflight's difference between local and hosted is exactly what the
  promotion has to do: 25 only local, 4 only hosted, 17 changed

**Caveat:** removing a *dropped source* (`CONFIRM_SOURCE`) has only run against test
fakes; the real-data path is removal by page (article 437).

**Scope:** `src/multimodal_rag/sync/` (probe, classify, plan, apply, policy), `Makefile`

#### SYNC-005 — Backup, Verification and Credentials (dissolved)

Folded into the stories that need it. The snapshot before a hosted write, and the
post-write verification, belong to SYNC-007 (the backup tool, `make snapshot-backup`,
already exists). Verification of a local apply belongs to SYNC-003. Run reports and
structured logs are the manifest of SYNC-002. The key rule (the admin key is used only by
sync and promotion; the serving app keeps the viewer key) is enforced by the write guard
and by OPS-003 and SYNC-009.

#### SYNC-006 — Documentation and Tests (dissolved)

Tests and documentation ship with each story, not at the end. The consolidated
procedure is the maintainer runbook in SYNC-008. The README, `docs/pipeline.md` and
`docs/gcp-deployment.txt` updates, and the "out of scope" lists in Epics 9 and 10, are
done at the end of SYNC-007.

#### SYNC-007 — Promote Local to Hosted (done, 2026-11-17)

**Branch:** `feature/SYNC-007-promote`

**Built and merged (2026-11-17):** `make promote-plan` (`TARGET=hosted` or `rehearsal`)
reads local and the target (hosted with the viewer key) and writes nothing. It classifies
every object: add, replace (properties differ), refresh the vector (only the vector
differs), delete, unchanged. The plan holds ids and counts, and a digest of every object
on both sides (id, properties, vector), because hosted has no generation number.
`make promote RUN=<dir>` recomputes both digests and refuses if either side moved. Then it
takes a backup of the target, writes **one canary object first** (read back and compared;
any mismatch aborts before anything else is written), writes the rest in batches of 100,
deletes last by UUID, and verifies **every object** (ids, properties, vectors, cosine
0.999999), runs the retrieval gate with local as the baseline, and takes a backup after.
Everything is written by one primitive, `WeaviateStore.put_objects`, which copies a local
object verbatim by UUID with its own vector (a full replace, checked for rejected objects)
and exists only behind `allow_hosted_write`. Vector refreshes are written before replaces
and adds, so a run that stops part-way leaves the old content on current vectors, a better
state than any other. A stopped run changes the target, so its plan is refused as stale;
a new plan is the remaining difference. A policy limit (`max_vector_only_updates`, 25)
makes a promotion that only refreshes vectors need review, because after this one-time
migration that would mean local drifted. `make promote-rollback SNAPSHOT_DIR=<dir>`
writes a snapshot's objects back verbatim and deletes the rest, in place

**The real plan (2026-11-17):** local 943, hosted 922: 25 to add, 17 to replace, **901
vector refreshes**, 4 to delete (article 437), 0 unchanged. It needs review because of the
901 vector-only updates, which is this migration (March vectors against the pinned runtime).
Hosted's exported vectors are byte-identical to local's earlier export, so vector equality
is checkable (hosted returns the stored full-precision vectors, not the 1-bit RQ copies)

**Rehearsal, off production:** a second local Weaviate 1.39.13 (`weaviate-rehearsal`
profile, ports 8081 and 50052, `make rehearsal-up`, `rehearsal-seed`, `rehearsal-down`)
seeded from the hosted backup of 2026-11-17 with the `hfresh` index (the open-source
server accepts it). Its digest equals hosted's. The real promotion wrote 943 objects and
deleted 4, verification and the gate passed, and a new plan shows 943 unchanged. The
rollback restored the pre-promotion snapshot and the digest equals the original hosted
digest, with the identical plan hash. Not rehearsable off production: the exact hosted
index, which uses RQ 1-bit quantization, which is why the first write on hosted is a
canary (`CANARY_ONLY=1`), and the network latency of 947 writes

**Production sequence, each step waits for a go-ahead:** (1) `make promote-plan`, review;
(2) `make promote RUN=<plan> CANARY_ONLY=1 ACCEPT_REVIEW=1` writes and verifies one
object, then stops (the plan is then stale, so plan again); (3) `make promote RUN=<new
plan> ACCEPT_REVIEW=1` writes the rest, verifies, gates and takes the after-backup. Pick a
quiet moment: during the batch some hosted chunks hold new vectors and some old, which
biases ranking toward the new ones. Every backup taken before this promotion holds the
March vectors, so the after-backup is the one to restore from later

**Production result (2026-11-17):** the canary wrote one object (backup
`snapshots/20261117T125131Z-promote-before`, verified equal to local, nothing else
changed). The full run then wrote 942 objects and deleted 4 (article 437), after its own
backup (`...125404Z-promote-before`), verified every object, passed the gate, and took the
after-backup (`snapshots/20261117T125431Z-promote-after`). Independent checks afterwards:
a new plan shows 943 unchanged and local's digest equals hosted's, the preflight shows 26
ok and no warning (hosted vectors match a fresh embedding at 1.0000), the app answers, and
hosted holds 943 chunks with 214 video chunks and none of the removed article. **Restore
only from `20261117T125431Z-promote-after` or later**: the three older backups
(`20261113T130911Z-cloud`, `20261113T131338Z-cloud`, `20261117T120502Z-cloud`) and both
`promote-before` backups hold the March vectors, so restoring one would bring the drift
back. Hosted now equals local, which is the state every later sync starts from

**Checked by a person (2026-11-17):** the maintainer asked the three canary questions in the
production app after the promotion and found the last one, the "import a HydroSym schematic
into HydroMan" question, improved. That matches the gate: its best source ranked 6 on
hosted before and 4 on local, which hosted now equals


- `make promote-plan` compares the local and hosted collections by UUID and text
  fingerprint (the same fingerprint as the preflight) and writes a promotion plan in the
  same plan format, with counts of added, changed and removed objects. It writes nothing
  to either collection
- `make promote PLAN=<file>` applies that plan to the hosted collection with the admin
  key. This module is the only caller that passes `allow_hosted_write`. It takes the run
  lock and refuses a plan whose hosted generation has moved
- **Copy the vectors from the local collection; never re-embed.** Local and hosted then
  hold identical vectors, and the promotion needs no embedding service
- Take a snapshot of the hosted collection first (`make snapshot-backup TARGET=cloud`)
  and record it in the manifest. Insert and update first, delete last, never delete the
  collection, and abort above the policy's deletion share. A pure addition (for example
  a new video) is the simplest case: nothing existing is touched
- Verify afterwards that the UUID set and fingerprints of local and hosted are
  identical, the count matches, and the retrieval gate passes against hosted
- Document the rollback: restore the snapshot with `--replace`, which deletes and
  recreates the hosted collection and so causes an outage; it is for failures only. A
  promotion that stops part-way is safe to repeat, because the diff is recomputed
- **Hosted tier constraint (checked 2026-11-16):** the hosted cluster is on the free
  plan, which allows **one collection** (a second one is refused with
  `USAGE_LIMIT_EXCEEDED … collections count limit of 1`) and the `hfresh` vector index
  only. A candidate collection with an alias flip, which is the vendor's blue/green
  pattern, therefore cannot run on this cluster, and that is why promotion mutates the
  live collection in place. Aliases do work on the local 1.39.13 server and are
  available on the paid Flex plan (1,000 collections). Multi-tenancy (up to 3 tenants on
  free) is not used: `SupportChunk` is not multi-tenant, enabling it means recreating the
  collection, and a tenant cannot be restored on its own. Revisit if the plan changes

Tests and docs ship with this story.

**Scope:** `src/multimodal_rag/sync/`, `store/weaviate.py`, `Makefile`

#### SYNC-008 — Maintainer Preflight, Sources of Truth, and Release Gate (done for manual operation, 2026-11-17; leftovers moved to SYNC-009, SYNC-011 and OPS-007)

**Branch:** `feature/SYNC-008-preflight`

- Keep the source registry (the three knowledge bases and the 25 videos) together with
  run state, sitemap captures and snapshots in one dedicated private, versioned bucket,
  with a prefix per kind of data, in a region other than Cloud Run's, accessible to the
  admin and operator only, and a lifecycle rule keeping the last 2 snapshots (**not built**: the bucket's only rule deletes
  superseded object versions, so `snapshots/` grows by about three per promotion; decision moved to OPS-007); each run manifest
  records the registry generation it used. The repository is public, so only an example
  file lives there. Keep credentials, including the YouTube cookies, in Secret Manager
  (workflow note, section 10)
- Done: `make sources-pull` and `make sources-push` move the registry between the bucket
  and a local cache; the push validates the schema and refuses if the remote changed
  since the pull (object generation check). The first registry is in the bucket: 25 videos
  and the 3 knowledge-base roots, restored from the March original because the working
  file had lost them. The registry holds roots only, not pages (see SYNC-002). Cloud
  Run's service account gets no access to the bucket. The run manifest that records
  the registry generation is written by SYNC-002
- Done: the tracked `Makefile` no longer hardcodes the GCP project; `GCP_PROJECT` comes
  from the environment or `.env`, and `make deploy` stops with a message when it is unset
- Done: the YouTube cookies are in Secret Manager (`YOUTUBE_COOKIES`), fetched into a
  private temporary file for each ingest run via `YOUTUBE_COOKIES_SECRET`, and explicit
  ignore rules cover cookie files; fresh cookies were uploaded on 2026-11-13 (version 2,
  verified with yt-dlp) and the March files were deleted
- Done: one `VISION_MODEL` value, and local and hosted settings kept apart. Found on the
  way: `VISION_MODEL` in `.env` had never taken effect (pydantic-settings treated the name
  as part of the nested `vision` field and dropped it), so local ingestion ran with visual
  grounding off whatever `.env` said; fixed with a regression test. `.env` also defined
  `VISION_MODEL` and `LLM_MODEL` four times each (the last one silently won); ingestion
  now warns about duplicate keys, naming lines and never values. `WeaviateStore` refuses
  every write (create, add, delete) when the target is hosted unless the caller passes
  `allow_hosted_write`, which only the promotion step will do; ingestion and the purge
  targets stop with a message instead. The local default now matches production
  (`LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-3.6-flash`, one line each), `VISION_MODEL` is
  `google/gemini-3.8-flash`, and the app logs a warning at startup when the default model
  is not in the model registry. Moved to SYNC-012: a Gemini vision path (the vision
  client always uses OpenRouter) and thinking and output limits on that client
- Align the compose image with the hosted Weaviate version (done 2026-11-13: both 1.39.13)
- Done: `make sync-preflight` (`LOCAL_ONLY=1` skips the cloud checks) reports missing
  tools (Docker, ffmpeg, gcloud, yt-dlp, Ollama with `nomic-embed-text`), missing keys
  without printing them, duplicate `.env` keys, whether the default and vision models
  exist, YouTube cookie expiry, whether the local registry copy matches the bucket, the
  age of the latest backup, Weaviate version drift, and whether local and hosted are
  identical (UUID and content fingerprint per chunk, read-only key). It changes nothing,
  turns any crashing probe into a failed check, and exits non-zero on a failure. The
  Gemini quota and spend cap cannot be queried, so it prints a manual line. A dedicated
  `WEAVIATE_HOSTED_URL` setting now holds the cluster address, separate from the local
  `WEAVIATE_URL`; backup and restore with `TARGET=cloud` use it, so no environment
  override is needed
- Done (decision 2026-11-16): the release gate between Stage A and Stage B is a
  retrieval gate, not an answer review. `make sync-gate` retrieves for each question in
  `config/canaries.yaml` against hosted (before) and local (after) and compares the best
  rank of an expected source: it fails when no expected source is in the top 10 and warns
  when the rank falls by 3 or more. It calls no LLM, so it is deterministic and free.
  Three canaries, chosen by the maintainer: zoom out in a schematic, separate multiple
  oil flows, and the advantages and steps of importing a HydroSym schematic into
  HydroMan. A person reads answers only when a canary fails. Reason: the pilot's one
  clear regression (library manager window) was a ranking displacement, visible at
  retrieval, and 12 hand-judged answers per run would not be run. Changed-page
  self-retrieval (each page SYNC-002 reports as changed is queried by its title) reuses
  the same code and arrives with SYNC-002. Known weak canary: the import question ranks
  its best source at 6 of 10, so it is the first to flip
- Moved to SYNC-009 (precondition 7): retrofit `--json` output and the shared exit codes
  (0, 10, 1) to `make sync-preflight` and `make sync-gate`. Neither has them today (exit 0
  or 1 only); only an unattended caller needs them, and the maintainer runs both by hand
- Done: the maintainer runbook, `docs/maintainer-runbook.md` (rules, who decides, the
  checks before a run, Stage A, Stage B, rollback, a failure table, routine, a command
  reference). `tests/test_runbook.py` fails if it names a `make` target, a variable or a
  file that does not exist, so it cannot drift from the Makefile unnoticed. Found while
  writing it: `make snapshot-bootstrap-cloud` passes no `--replace`, so it cannot be used
  to replace an existing hosted collection

**Scope:** `config/`, `Makefile`, `compose.yaml`, `docs/`, preflight module

#### SYNC-009 — Automate the Web Lane (stub; not started)

**Branch:** `feature/SYNC-009-automate`

Goal: a scheduled run of the web lane that plans, checks the plan against the policy,
applies, runs the gate and promotes only when the policy and the gate both pass; anything
else stops and notifies a person. This is a stub. It is not built until its preconditions
are met, and the first one is unresolved.

Preconditions, each a decision or a ticket of its own:

1. **Where it runs.** A job (for example a Cloud Run Job) with its own service account,
   with the admin key readable only by that job. The serving app keeps the viewer key. The
   epic's rule "Cloud Run never holds write credentials" becomes "the serving app never
   holds them" and must be reworded when this is built
2. **Embeddings.** Vectors come from local Ollama `nomic-embed-text`, and a cloud job has
   none. Either the job ships an embedding service that produces vectors equivalent to the
   local ones (to be measured by embedding the same texts in both and comparing), or the
   job cannot run apply. **This is the blocker and needs a spike first**
3. **Staging and the Weaviate plan.** The two-stage model uses local Docker, which a
   scheduled job cannot. On the free plan (one collection, no native backups, no
   replication, best-effort availability, suspension after 7 idle days) the only options
   are direct in-place promotion with a snapshot and the gate as the protection, or
   moving to the Flex plan (from $45 per month: 1,000 collections, so aliases and
   blue/green work, plus 7-day backups, replication and a 99.5% availability promise).
   Decide the plan before building; an unattended sync on the free plan is not
   recommended. If aliases are adopted, `SupportChunk` would have to become an alias
   name, which means moving the data once and changing the app and the snapshot tooling
4. **Alerting and cost.** OPS-004 (failure and needs-review notifications), OPS-001 (spend
   caps) and OPS-003 (dedicated service accounts) are done before any schedule is enabled
5. **Policy tuned from experience.** At least three consecutive manual runs inside the
   policy with no gate failure before it is trusted unattended
6. **Overlap and repeatability.** A test shows two concurrent runs cannot both proceed, and
   that a repeated run is a no-op
7. **Machine-readable checks.** `make sync-preflight` and `make sync-gate` gain `--json`
   output and the shared exit codes (0, 10, 1); they have neither today (moved from
   SYNC-008, because only an unattended caller needs it)
8. **The bucket leads for the registry.** The state bucket, not a file on a laptop, is the
   source of truth for what is in scope, because it is shared, versioned, recoverable and
   already holds the ledger, locks and run records that every plan is checked against. The
   manual phase hides this: `make sync-videos` reads the local `config/sources.yaml`, and
   when that file has unpushed edits it warns and carries on. An unattended run must never
   read a local file. Before SYNC-009 is built:
   - the lanes read the registry from the bucket, and the local file is only an editing
     buffer (a run that finds the two different uses the bucket and says so);
   - `make sources-edit` fetches, lets the maintainer edit, validates and pushes in one
     step, so there is no separate pull to forget. The push already refuses when the
     bucket changed since the fetch (a generation check), which also protects an edit
     against a scheduled run; `FORCE=1` on a pull is the way an edit gets lost and should
     go away;
   - the "local → hosted" flow describes the manual phase only. Where a run builds and
     verifies content before hosted changes (precondition 3) is the open design question
     here; the manual runbook is not the design

9. **Video discovery (notify only).** Nothing watches the Paro Software YouTube channel
   today: a video is ingested only when a person adds its URL to the registry. A scheduled
   check may list the channel's uploads (its public RSS feed, or yt-dlp, which the video lane
   already uses), compare them with the registry and the stored videos, and notify the
   maintainer through OPS-004 of any upload that is in neither. It never writes to the
   registry and never ingests. The maintainer reviews the video, adds it to the registry
   (`make sources-pull`, edit, `make sources-push`) and runs the video lane by hand.
   Reasons: each video costs vision and transcription calls, its transcript and
   descriptions need a person's check, and a video that has chunks is never looked at again,
   so a bad first ingest stays bad. The registry in the bucket stays the single list of
   what is in scope; the channel is only a source of suggestions for it. Decide before
   building: which uploads to skip (Shorts, announcements, live streams), and how a
   replaced tutorial is flagged (the changed-video half of SYNC-011 is not built)

Limits that stay: the video lane is never scheduled and discovery never ingests; removals
are never confirmed by automation; screenshots are never regenerated by an automated run.

**Scope:** to be set once preconditions are met

#### SYNC-010 — Screenshot Provenance and Re-description (experiment; deferred)

**Branch:** `feature/SYNC-010-screenshots`

Depends on SYNC-001 for the stored fields. Not a prerequisite for sync: the pilot found
that screenshot descriptions did not improve any answer and two got worse, so this is an
experiment to run behind the retrieval gate, not a precondition.

- In the hosted collection 381 of 943 chunks (40%) are keyed on `uuid5(image_url)` while
  `image_url` is not stored. SYNC-001 added the `image_url` property but left it empty;
  add `image_hash`, `description_model` and `prompt_version`, and backfill `image_url`
  by re-deriving `uuid5(image_url)` from each
  live page (a one-page test matched 3 of 4 stored chunks, and 19 of 19 and 7 of 7 on two
  more pages)
- Add the Gemini provider path to the vision factory (it goes through OpenRouter only)
  with `thinking_level="low"` and a `max_output_tokens` cap, and record the model that
  wrote each description. SYNC-012 builds this for video frames; do it once and share it
- Re-describe with the page-anchored prompt, with a repetition guard, decorative images
  skipped and the application named only if visible; replace the three runaway chunks
  (81k to 229k characters). Needs Gemini quota for roughly 380 images; run a 5-image
  trial first
- Re-measure with the retrieval gate and the 57-query retrieval measurement; keep the
  change only if retrieval does not get worse

Evidence: `docs/screenshot-description-pilot.md`.

**Scope:** `models/chunks.py`, `ingest/web_images.py`, `models/llm.py`

#### SYNC-011 — Video Lane: New and Changed Videos (new videos built; first real run pending)

**Branch:** `feature/SYNC-011-video-lane`

Today `make ingest` reprocesses every video and re-crawls every knowledge base on each
run, with no per-source option. Both cases below write to the local collection and reach
hosted through SYNC-007.

- **New videos (first):** ingest only registry sources with no stored chunks. Chunk IDs
  (`url|fused|start_seconds`) are new, so nothing collides and nothing existing changes.
  Promotion treats it as a pure addition
- **Changed videos (later):** a fingerprint (video ID, duration, title, caption hash) and
  a pipeline version; reprocess a video only when either changes, and sanity-check the
  stored windows against current durations
- Maintainer-triggered only; never part of an automated run
- **Match on the canonical `document_id`, not the raw URL.** Chunk IDs derive from the
  URL string, so the same video registered as `youtu.be/<id>` and as `watch?v=<id>` would
  otherwise be ingested twice (`models/identity.py` already makes both one document)
- `DRY=1` lists the videos that would be ingested and estimates the vision calls (one
  frame per 30 seconds) before anything is spent; the runbook gains an "Adding a video"
  section from the first real run (first customer: `yWuBQik8KPs`, "HydroMan Tutorials:
  Cross-Over Pilot Lines and Angled Connections", 2:04, auto captions, about 5 frames)
- Moved on to SYNC-012 (frame descriptions), after coming from SYNC-008: a Gemini vision
  path (the vision client always uses OpenRouter) and thinking and output limits on that
  client

**Built, new-video half (2026-11-23), not yet run on real data:** `make sync-videos`
(`DRY=1`, `ROLLBACK=runs/videos-<id>`, `YES=1`, `ALLOW_NO_VISION=1`), local only.

- `ingest/videos.py::ingest_video` is the one per-video function; `make ingest` now calls
  it too, so the two cannot drift (the existing tests cover all three paths through it).
- **Selection** (`sync/videos.py`): new means no stored chunk carries the video's
  `document_id`. It refuses, instead of guessing, an address that is not a watch, youtu.be
  or embed link, two registry entries for one video, a stored video without a
  `document_id`, and a registry that differs from the bucket's. On the real data all 25
  registered videos match their stored chunks, including the nine `youtu.be` ones.
- **All or nothing per video:** a video that already has chunks is never looked at again,
  so a half-described one would stay that way. `strict` mode (new in `video_frames.py`)
  fails the whole video when one frame cannot be described, before anything is stored, and
  the lane refuses to run without `VISION_MODEL`. (Voxtral already raised on failure.)
- **Transcript source (2026-11-23, replaces Voxtral):** the video's **manual English
  captions** if it has them, otherwise a **Gemini transcription** (`gemini-3.8-flash`,
  mono 16 kHz audio, temperature 0, JSON segments); automatic YouTube captions are never
  used. `ingest/transcripts.py` returns segments in the shape the fused 30-second windows
  already consume, so new videos look like the existing 25. The run record keeps each
  video's source and the model version Gemini reported. A failed transcription fails the
  video. Why: the default path always used Voxtral, the Mistral key in `.env` turned out to
  be rejected (401) and the preflight could not see it, and `voxtral-mini-latest` is a
  floating alias. **Spike** (3 existing videos with manual captions as ground truth, plus
  the new one): word error rate 0.010 to 0.040 for Gemini against 0.028 to 0.054 for the
  stored Voxtral text; every product name correct (Voxtral wrote "HydroSim" for HydroSym
  in 8 of 8 mentions); timestamp offset median 0.4 to 0.6 s; coverage 0.99 to 1.02. By the
  criteria written beforehand it missed three marginal checks (window agreement 0.901
  against 0.967 on the 113 s video, one 6.1 s offset against a 5 s limit, run-to-run word
  error rate 0.032 against 0.03). 18 of 26 videos have manual captions, 8 only automatic.
  `Mistral` and `voxtral.py` are no longer used by video ingest; deleting them and the
  `mistralai` package is a follow-up.
- **Estimate** per real path: spoken videos one frame per 30 s plus the minutes of audio
  Gemini must transcribe (none when manual captions exist), silent recordings one frame
  per 5 s. For `yWuBQik8KPs` (2:04, automatic captions only): 5 vision calls, 2.07 minutes
  of Gemini transcription. The preflight now checks that the Gemini key is *accepted*, not
  just set. `yt-dlp` was upgraded to 2026.10.26: the locked version got HTTP 403 on some
  videos.
- **Known:** the existing 25 videos' transcripts say "HydroSim"; re-transcribing them
  would change text and vectors (re-embed and a promotion) and is a separate decision.
  Gemini's model name is a floating alias (no dated ID was listed), so each run records the
  version it reported.
- **Proof and rollback:** after the run every earlier object must be unchanged and every
  new one must belong to a new video; the exact UUIDs are written to `videos.json` and
  `ROLLBACK=` deletes only those (never `make purge-video`, which deletes all 214 video
  chunks). The snapshot is the backstop; the retrieval gate runs afterwards.
- **Hardened after review:** a real run refuses while `config/sources.yaml` has edits not
  pushed to the bucket (the dry run only warns), so the registry stays the source of truth;
  the planned videos are written to `videos.json` **before** ingesting, and the rollback
  deletes every object of those videos, so a crashed run can still be undone; a video that
  yields no chunks counts as failed (it would otherwise be retried silently); an unknown
  duration is shown as unknown and the total is marked incomplete.
- Runbook section 3b "Adding a video" is written from this code. **Not done:** the first
  real run, which is the maintainer's acceptance test of the runbook; changed videos
  (fingerprint and pipeline version); the Gemini vision path.

**Scope:** `ingest/__main__.py`, `ingest/youtube.py`, `Makefile`

#### SYNC-012 — Video Frame Descriptions: Prompt and Chunk Structure (measured 2026-11-23; part 1 built, not yet merged)

**Branch:** `feature/SYNC-012-frame-descriptions`

Do this **before the first new video is ingested**: a video that has chunks is never looked
at again, so a video stored with the current descriptions keeps them. It is independent of
the lane's mechanics (SYNC-011) and shares the vision path and the repetition guard with
SYNC-010 (web screenshots), which should be built once.

**Evidence** (read-only analysis of the 214 stored video chunks of the 25 videos, local
collection, 2026-11-23; the scripts were exploratory and are not in the repository):

- **Length.** All 214 descriptions exceed the 80 words the prompt in `video_frames.py` asks
  for: median 476, and one with **42,843 words** (HydroSym Explainer: Dimensions at 60 s,
  distinct-word ratio 0.014, a repetition loop). Retrieved, that chunk goes into the LLM's
  context whole. The description is a median 87% of a chunk's words; the narration is about
  70.
- **Boilerplate.** The prompt says not to describe static UI, but 86% of the descriptions
  describe the menu bar, 92% the toolbar, 71% the title bar and 78% the status bar; 62%
  list the identical menu ("File, Edit, Draw, Modify, Format, Assist, Window, Print,
  Tools, Library, Help"). 194 hedge ("likely", "appears to"), 191 open with "This video
  frame displays" or "The image displays", and 212 are markdown with headings and bullets.
  Which prompt produced them is not known: the one in the code cannot yield 476 words.
- **What the vector sees.** `embed_texts` cuts every text to its first 400 words, and a
  chunk is the narration followed by the description, so 206 of 214 chunks are cut and a
  median 32% of the description is lost. The words that remain are mostly the window
  inventory (median 6 chrome mentions inside them); the part that says what is being
  demonstrated comes later. The stored vector is closer to the description alone than to
  the narration alone in 39 of 40 sampled chunks (cosine 0.967 against 0.918).
- **Dilution.** Chunks of different videos have a mean cosine of 0.841 (0.739 when embedded
  from the narration alone), and chunks of the same video 0.938 (0.872).
- **Diagnostic retrieval.** A sentence the narrator says, used as the query, finds its own
  chunk among the 214 at top-1 in 32% and top-3 in 51% of cases with the stored vectors,
  against 47% and 75% with narration-only vectors. **Limit:** this measures dilution and
  favours narration-only by construction. It says nothing about questions that depend on
  what is on screen, so it cannot show that the descriptions are useless.

**Measurement (2026-11-23).** About 82 vision calls on `gemini-3.8-flash` and local
re-embedding with the pinned Ollama; the scripts were exploratory and are not in the
repository. Frames came from the project's own `extract_keyframes`; every claim below was
checked against the frame image, not the log.

- **Today's prompt is not the cause.** The prompt in `video_frames.py` (P0), run today,
  writes 38 to 72 words with almost no window inventory. The stored 300 to 757-word
  descriptions came from an older prompt or model. P0's real flaws are markdown and thin
  detail.
- **Stage 1, three prompts on five frames.** Candidate A (anchored on the video title and
  the window's narration) and B (unanchored) both gave 110 to 134 plain-text words and
  answered "No relevant on-screen content." on a title card. **Neither passed "every value
  correct":** A said the Join and DV Bottom toggles were inactive when they are on, and in
  a first run claimed a right-click because the narrator says "right-click" while the mouse
  graphic showed the left button; B garbled a quoted field as `ø<><X<>` for `ø<>x<>`.
  Anchoring on the narration added a way to be wrong and no demonstrated benefit, so A was
  dropped. Errors came from enumerating many fields, which a higher word cap invites.
- **Stage 2, prompt C on 52 frames of 6 videos.** C is today's prompt with its last
  sentence changed to "Write plain text in one paragraph of at most 100 words. No markdown,
  no lists, no bold." Result: 64 to 100 words (median 90), no markdown or hedging, no loop
  (distinct-word ratio at least 0.63), 7 of 52 still mention the title, menu, tool or status
  bar. A rebuilt chunk is at most 195 words against a stored median of 540, so nothing is
  cut at the embedding limit. Of 8 descriptions drawn at random, none had a wrong dialog
  name or value; one overreached ("actively clicking" a button the cursor only hovered on).
  Control: re-embedding the stored text reproduces the stored vectors (cosine 1.0000).

  | Swapped into the stored vectors | Narrator diagnostic top-1 / top-3 | On-screen questions in the top 3 | Canaries |
  |---|---|---|---|
  | stored | 15% / 37% | 1 of 6 | 1, 1, 4 |
  | C, one hybrid vector | 27% / 44% | 4 of 6 | 1, 5, 4 |
  | C, narration and visual as two vectors or two chunks | 46% / 63% | 5 of 6 | 1, 5, 4 |
  | narration only | 48% / 67% | 2 of 6 | 1, 5, 4 |

  Against the criteria fixed beforehand, C passed length, loops and chunk size, the
  diagnostic against stored, the on-screen questions and accuracy, and **failed** two: the
  diagnostic within 5 points of the narration-only baseline (27% against 48%, a ceiling
  that favours narration by construction), and "canaries rank no worse" (the oilflow canary
  fell from 1 to 5; the top six sit within 0.023, narration alone also gives 5, and the
  gate would WARN, not FAIL). The 6 on-screen questions were written by the assistant from
  frames it had viewed, then shown to and accepted by the maintainer before any run; with 6
  questions one question is a 17-point swing.
- **Length.** Cutting C's descriptions to the first 20, 40, 60 or 90 words (re-embedded,
  no new calls) moves the diagnostic top-1 from 37% to 27% and the on-screen questions from
  3 to 4 of 6; about 60 words looks like the knee. This is truncation of 90-word text, not
  what a 60-word prompt would write.
- **Why two vectors help.** One embedding averages two kinds of content, what is said and
  what is shown, so a hybrid sits between both kinds of query. Scoring a query against a
  narration vector and a visual vector and taking the better match closes most of the gap
  to narration-only (46% against 48%) and keeps the screen-detail gains. The simulation is
  optimistic: only the 52 swapped windows had two chances to match.
- **Splitting the existing 213 chunks needs no vision calls.** Splitting each stored chunk
  into its narration and its stored description and embedding both raised the narrator
  diagnostic over 213 windows from 14% / 30% (median rank 12) to 40% / 65% (median rank 2),
  and fixed the "weak" import canary (rank 4 to 1), while the oilflow canary fell from 1 to
  6. The stored descriptions add nothing for screen-detail questions (2 of 6, the same as
  narration alone), and trimming them changes nothing: the on-screen gains need new
  descriptions. The diagnostic overstates the real gain because the queries come from the
  narration.
- **The frame labelled "30 s" is not the video's 30 s.** The pipeline's frame for a window
  matches about 42 to 54 s of the video for the window starting at 30 s: inside the window,
  later than its label. Feeding a window's narration to a frame at its start made the
  narration and the picture disagree.
- **Two chunks per window share the top results (retrieval note).** The production answer
  to "What advantages will I have when I import my HydroSym schematic into HydroMan?" listed
  10 sources: 5 were windows of one video (0:00, 0:30, 1:00, 2:00, 2:30) and 5 were articles.
  HydroMan has little content (about 48 articles; HydroSym has 99), so such a question
  reaches weak matches quickly and several windows of the one relevant video are partly
  correct, a content gap more than a ranking fault. A local simulation with the existing
  chunks split gave a top 10 of 6 video chunks (5 windows of that video plus one other
  video) and 4 articles, against 2 video chunks and 8 articles for the stored hybrid
  vectors. The hosted list differs in its tail from the local ranking (294, 444 and 276
  against three more windows of the video), so the exact makeup on hosted cannot be read
  from local. With new descriptions both chunks of a window can rank, taking two slots and
  listing one timestamp twice. A mitigation, not built: fetch extra results and merge a
  window's chunks into one source (best score, narration first) before the answer is made.
- **8 of the 25 registered videos cannot be downloaded now,** though YouTube's oEmbed
  endpoint serves all 25 as public. `yt-dlp` answers "This video is not available" with or
  without cookies, for videos that downloaded this morning. The cause is unknown and may be
  transient. They include the Dimensions video behind the 42,843-word chunk, so that chunk
  cannot be re-described and is fixed by cutting its stored text.

**Work, in three parts**

1. **New videos** (the build). Prompt C with a hard output limit and a cap of **100
   words** (60 was chosen first, tried on real output and set back on 2026-11-23, see the
   checks below), a low thinking level, a **repetition guard** (reject a description with a low
   distinct-word ratio or over a length cap, retry once, fail the video in strict mode), a
   recorded `prompt_version`, and **two chunks per window** written at ingest: the
   narration, and the visual description under its own id (`<url>|visual|<start>`), both
   with the same `source_url` and timestamp. The embedded text is planned, not cut by
   accident: a warning or a failure when a chunk would exceed the embedding limit. Add the
   Gemini provider to the vision factory (it goes through OpenRouter only), shared with
   SYNC-010. Not feeding the narration to the vision call. Two chunks need no schema change
   and use the pattern the web screenshots already follow; named vectors would need a schema
   change and changes to the store, search, promotion, snapshot and app, tested with
   `make weaviate-compat`.
2. **The existing chunks:** split each into a narration chunk and a description chunk by
   re-embedding locally, with no vision calls. Now its own ticket, **SYNC-013**.
3. **Re-describing the existing chunks** with the new prompt, for the on-screen gains.
   Deferred: it needs the 8 videos that cannot be downloaded now, and it changes text and
   vectors again.

**Decided 2026-11-23:** the cap is 100 words; the oilflow canary stays as it is and its fall
to rank 5 is accepted (the maintainer calls it a glitch). **Checked 2026-11-23 on the real `gemini:` path (13 calls, then 12 more after two fixes):**
the mocked tests could not see two defects, both fixed. (1) Gemini 3 returns a list of
content blocks, and `str()` of it would have put `[{'type': 'text', ...}]` at the start of
every stored description, so a `message_text` helper now reads only the text blocks (it
also serves `web_images`). (2) Every measurement sent the image before the prompt and the
code sent it after; with the image first the LangChain path reproduces the earlier SDK
output nearly word for word (same opening, same quoted names, a couple of words differ,
as Gemini is not fully deterministic at temperature 0). On the final code, 12 frames at the
60-word cap gave 37 to 56 words (median 52), none over the cap, none rejected, no markdown,
one chrome mention, and no wrong claim in the four frames behind the on-screen questions.
**The cap costs detail.** The four on-screen questions' frames lost exactly what some
questions ask for (the machine and material checkboxes, "Hold Ctrl to Copy"). On those
12 windows the on-screen questions in the top 3 went from 4 of 6 (100 words, one vector) to
3 of 6, and from 5 of 6 to 3 of 6 with two chunks (still above the stored 1 of 6), while
the narrator diagnostic did not move. The truncation test had predicted no loss. Six
questions are a thin sample, and an in-between cap of 80 words is untested. **The maintainer
set the cap back to 100.** What 100 costs over 60: about 70 more output tokens per frame
(1.73 tokens per word, measured in Stage 1; input is about 1,300 tokens per call, dominated
by the image), so about 350 per 2-minute video; about 40 more words in the context of each
retrieved visual chunk; no narration dilution with two chunks; and a possibly higher risk of
a wrong detail, on thin evidence (two errors in 5 frames at 110 to 134 words in Stage 1, none
in the 8 sampled at 100 words in Stage 2). The price of the model was not looked up.
**Deferred until after the first video:** how a window's two chunks share the top results
(see the retrieval note above; the merge at retrieval is not built), and whether the old
visual chunks should be kept or left out of retrieval.

**Acceptance:** for part 1, no description over the cap and none with a repetition loop, no
chunk cut at the embedding limit, accuracy checked on a sample against the frame images
(none wrong about a dialog name or a value), and the three canaries not worse than a WARN.
For part 2, the narrator diagnostic and the canaries measured before and after on the real
collection, the object count equal to the plan, and a rehearsal first. The first video of
SYNC-011 is ingested only after part 1.

**Scope:** `ingest/video_frames.py`, `ingest/videos.py`, `models/llm.py`, `models/chunks.py`

#### SYNC-013 — Split the Existing Video Chunks into Narration and Description (not started; ticket written 2026-11-23)

**Branch:** `feature/SYNC-013-split-video-chunks`

Part 2 of SYNC-012. SYNC-012 part 1 (merged, `89d989a`) makes every **new** video store a
window as two chunks: the narration (`[Transcript] ...`, id `<url>|fused|<t>`) and the
on-screen description (`[Visual] ...`, id `<url>|visual|<t>`). The 214 video chunks stored
today are still one hybrid chunk each: a median 70 words of narration followed by a median
476 words of description, cut at the embedder's 400 words. This story converts them, with
no vision calls and only local embedding. It is **not a prerequisite for the first new
video**: the two formats can coexist.

**Why.** Measured on the real collection with 213 narrator queries (a sentence the narrator
says, which should find its own chunk): the hybrid finds it first 14% of the time at median
rank 12; the narration alone 40% at median rank 2, and splitting gives the same 40% / 65%.
The import canary improved from rank 4 to 1; the oilflow canary fell from 1 to 6 (a WARN by
the gate's rule; the maintainer called it a glitch and accepts it). The old descriptions add
nothing for questions about what is on screen (2 of 6, the same as narration alone), and
trimming them to 100 or 60 words changes nothing: those gains need new descriptions, which
is SYNC-012 part 3. **Limits:** the diagnostic favours narration vectors by construction, so
14% to 40% overstates the real gain; the canaries (three questions) are the only
real-question evidence; the on-screen set is six questions.

**What is stored today:** 214 video chunks, every one with a description, one with **no
narration** and one description of **42,843 words** (HydroSym Explainer: Dimensions at 60 s,
distinct-word ratio 0.014). The ticket text of SYNC-012 said 213 chunks and 1,156; the
counts below are the checked ones.

**Work**

- A command in the style of `make sync-convert`, say `make sync-split-videos` (`DRY=1`,
  `YES=1`), local only: lock, snapshot of local first, refuse unless the embedding runtime
  is the pinned one, then write and verify.
- **Narration chunk:** the existing object, same id, text `[Transcript] <narration>`, content
  hash recomputed, a new vector from `"<video name>: <text>"`.
- **Description chunk** (if kept, see decision 1): a new object with the id
  `uuid5(<url>|visual|<t>)` from `SupportChunk.from_visual_chunk`, text `[Visual]
  <description>`, the same `source_url`, name, timestamp, `url_hash` and `ingested_at`, the
  identity fields, and its own vector.
- **Verify:** the object count equals the plan; every old id is still present; for every
  window the narration text plus the description text equals the stored text (a partition,
  nothing lost); re-embedding the new text reproduces the new vectors (cosine 1.0000, the
  control used in the earlier tests); a second run changes nothing.
- **Promote** the usual way: `make promote-plan`, a rehearsal first, a canary write, the full
  promotion, independent checks, the before and after backups. Rollback is the snapshot and
  `make promote-rollback`.

**Counts.** 214 stored chunks, 213 with narration. If the descriptions are kept: 213
narration replaces, 214 description adds, 1,157 chunks in total. If they are left out:
213 replaces and no adds, 943 chunks.

**Decisions needed before building**

1. **Keep the old descriptions as chunks, or leave them out?** Kept, they add 214 chunks of
   mostly menu and toolbar inventory that almost never rank, take slots in the top results
   when they do, and will be replaced again by SYNC-012 part 3. Left out, the collection
   stays at 943 and the old text stays only in the snapshot. *Recommendation: leave them
   out until they are re-described,* since the measurements gave them no retrieval value.
   This is the maintainer's call.
2. **The 42,843-word description:** if descriptions are kept, drop this one, since it is a
   repetition loop; its opening words are window inventory anyway.
3. **The window with no narration:** its stored chunk holds only a description under the
   `|fused|` id. *Recommendation: leave it unchanged.*
4. **Bump local and the rehearsal copy to Weaviate 1.40.5 first?** Hosted runs 1.40.5 and
   local 1.39.13, so a rehearsal is not an exact copy (runbook section 7a; run
   `make weaviate-compat VERSION=1.40.5 UPGRADE=1` first).

**Risks.** The promotion changes the text of 213 chunks in place on production, so it needs
the full procedure and a verified rollback. A window can then appear twice in the top
results (narration and description of one moment) once new descriptions exist; the merge at
retrieval was deferred by the maintainer until after the first video (SYNC-012 retrieval
note). Hosted must be awake: the free cluster paused after six days on 2026-11-23 and no
keep-alive exists (OPS-007).

**Acceptance (fix these before the run).** On the real local collection before and after:
narrator diagnostic over all windows top-1 at least 35% (14% today) and no worse top-3; the
three canaries not worse than the accepted oilflow WARN, the import canary not worse than 4;
the six on-screen questions in the top 3 at least as often as now (1 of 6); the object count
equal to the plan; every old id present; a second run a no-op. After the promotion: a fresh
`make promote-plan` shows nothing to do, `make sync-preflight` shows local and hosted
identical, the app answers a real question with a cited answer.

**Not in scope:** re-describing the existing chunks (SYNC-012 part 3), the merge at retrieval,
the 8 videos that cannot be downloaded now.

**Scope:** `sync/` (a split module and command), `Makefile`, `docs/maintainer-runbook.md`

### Phase 3: Production hardening

An audit of the live service on 2026-11-13 found the application layer sound (non-root
container, read-only Weaviate key at runtime, secrets in Secret Manager, structured and
redacted logs, a verified off-machine backup) and the operating layer mostly unbuilt. The
findings behind each story are in the evidence line. Priority order: OPS-001 to OPS-004
first, because they bound cost and blast radius; the rest follow. None of these change the
embedding contract or the collection.

| Story | Gap | Evidence (2026-11-13) | Priority |
|-------|-----|-----------------------|----------|
| OPS-001 | No spend cap or budget alert | Billing Budgets API not enabled, no budget found | 1 |
| OPS-002 | Public endpoint calls paid LLM APIs | `allUsers` holds `roles/run.invoker`; no length cap, queue cap or per-client limit; decision: open with limits | 1 |
| OPS-003 | Runtime uses the default compute service account | `71323613645-compute@developer.gserviceaccount.com` | 1 |
| OPS-004 | No monitoring or alerting | Cloud Monitoring API not enabled | 1 |
| OPS-005 | No liveness probe, service config not in code, stale `RESTART` env var | Startup probe only; env and secrets set by hand | 2 |
| OPS-006 | No CI | No `.github/workflows` directory | 2 |
| OPS-007 | Backups are manual and the cluster tier is unchecked | `make snapshot-backup` is not scheduled | 2 |
| OPS-008 | Ingested web content reaches the LLM context unfiltered | Crawl output goes straight into prompts | 3 |

#### OPS-001 — Spend Cap and Budget Alerts

**Branch:** `feature/OPS-001-spend-guard`

- Set a hard monthly limit on the Gemini key in AI Studio (the "Hydroman" project, which is
  separate from the Google Cloud project) and a usage limit on the OpenAI project; record
  both values in the runbook
- Enable the Billing Budgets API and create a budget on the Cloud project with alerts at
  50, 90 and 100 percent, sent to the maintainer and the operator
- A log-based metric on the existing `event="inference"` records, so token totals per
  provider per day are visible without new instrumentation
- Document what happens at the cap: the app answers with an error, not silently with a
  different provider

**Scope:** cloud configuration, `docs/gcp-deployment.txt`, runbook; no application code
unless the cap needs a friendly error message

#### OPS-002 — Public Endpoint Abuse Control

**Branch:** `feature/OPS-002-abuse-control`

The service is open to the internet (`allUsers`) and every question costs LLM tokens. Max
instances 1 limits throughput, not spend.

- **Decision (2026-11-13):** keep the endpoint open and add limits first. Sign-in stays an
  option if the limits prove insufficient; the trigger is unexplained spend or a burst the
  limits did not stop (watched through OPS-001 and OPS-004)
- Verified today: `GRADIO_SHARE` is off and not set on Cloud Run. The question box has no
  length cap. The Gradio queue has no maximum size. The only limit is
  `concurrency_limit=1` on the question event, which serialises all visitors through one
  slot: it bounds spend but makes a second user wait behind the first. No per-client rate
  limit exists, and the Gradio API can be called directly, bypassing the interface, so
  limits must be enforced server-side
- Cap the question length (agreed: 1,000 characters) and reject longer input with a
  clear message before any embedding or LLM call
- Cap the queue (`max_size`, agreed: 10) so a burst gets a "busy, try again" answer
  instead of an unbounded wait
- A per-client sliding-window rate limit keyed on the forwarded client address (agreed:
  10 questions per minute and 100 per day), in memory per instance, which is enough while
  max instances is 1; log each rejection as a structured event without the question
- Raise the question concurrency only after the rate limit is in place, and record the
  value and the reason
- Test the limits with a scripted burst against a staging revision, not production

**Scope:** `app.py`, Cloud Run IAM, runbook, tests for the length and queue limits

#### OPS-003 — Dedicated Runtime Service Account

**Branch:** `feature/OPS-003-service-account`

- Create a runtime service account with `secretmanager.secretAccessor` on the three
  secrets the service reads, and the log writer role, nothing else
- Switch the service to it (`--service-account`, set in the `Makefile` deploy target so a
  deploy cannot drift back) and remove the default compute account from the service
- Report the roles the default compute account holds on the project and remove any it
  does not need; this is a read first, a change only with the maintainer's approval
- Confirm after the switch: the app starts, reads its secrets and writes logs; the
  `YOUTUBE_COOKIES` and bucket stay inaccessible to it

**Scope:** IAM, `Makefile`, runbook

#### OPS-004 — Monitoring and Alerting

**Branch:** `feature/OPS-004-monitoring`

- Enable the Cloud Monitoring API
- An uptime check on the service URL, with an alert after consecutive failures
- Alerts on: the 5xx rate, instance startup failures, `event="inference"` records with a
  failed status, and p95 inference latency above an agreed threshold
- One notification channel to the maintainer and the operator; name who acts on what
- The p95 latency SQL from OBS-002, run once against Log Analytics and the result recorded

**Scope:** cloud configuration, runbook

#### OPS-005 — Probes, Service Config as Code and Rollback

**Branch:** `feature/OPS-005-service-config`

- Add a liveness probe next to the existing startup probe, using a cheap health route
  that does not call Weaviate or an LLM
- Export the service definition (env var names and secret references, never values) to
  the repository and deploy from it, so env and secrets stop being set by hand. Remove the
  stale `RESTART` variable
- Write down and rehearse the rollback: send traffic to the previous revision with one
  command, and keep at least two earlier revisions
- Record the minimum-instance decision (scale to zero costs a cold start; one warm
  instance costs money) with the reason

**Scope:** service config file, `Makefile`, `app.py` (health route), runbook

#### OPS-006 — Continuous Integration

**Branch:** `feature/OPS-006-ci`

- A GitHub Actions workflow on pull requests and pushes to `main`: `ruff`, `mypy --strict`,
  the test suite, and `uv lock --check`
- Dependency vulnerability scanning (Dependabot alerts or `pip-audit`); GitGuardian already
  covers secrets
- Branch protection on `main` requiring the checks, with no automatic deploy: a person
  still runs `make deploy`
- Keep the tests free of network and cloud calls, as they are now

**Scope:** `.github/workflows/`, repository settings, `README`

#### OPS-007 — Backup Cadence, Restore Drill and Cluster Tier

**Branch:** `feature/OPS-007-backup-cadence`

- State the recovery point and who runs the backup. The backup needs the admin key, which
  Cloud Run must never hold, so it runs from the maintainer machine: before every
  promotion (already required by SYNC-007) and on a fixed cadence in between
- Rehearse a restore from the bucket into a fresh local volume and compare it with the
  hosted collection. Done once on 2026-11-13: 922 objects, identical UUIDs, text and
  vectors. Repeat it on the cadence and after any server upgrade
- **Free-plan lifecycle (checked 2026-11-16, pricing page and cluster docs):** a free
  cluster is suspended after 7 days of inactivity (reactivated in the Weaviate Cloud
  console, and until then the app cannot read its data) and permanently deleted after 30
  days of total inactivity, with one warning email a day before. The docs do not say what
  counts as activity, so whether colleagues' questions keep the cluster alive is unknown.
  The free plan has no native backups, no replication and best-effort availability, so
  the GCS snapshot is the only recovery path
- Do now, ahead of the rest of Phase 3: confirm which address receives Weaviate's
  warning emails; add the cluster state to `make sync-preflight`; test whether a
  scheduled read-only query counts as activity before relying on it; keep the newest
  snapshot younger than the suspension window
- **Incident evidence (2026-11-23):** the free cluster was **paused for inactivity** after
  about six days without traffic (the last hosted query was the 2026-11-17 verification).
  Its endpoint answered `503 no healthy upstream`, so the app failed its startup check and
  returned 503 too; Cloud Run logged failed requests at 09:25 UTC and the maintainer woke
  the cluster in the Weaviate Cloud console at about 11:30 UTC, so the outage lasted at
  least two hours and may have started earlier. The assumption above that colleagues use
  the app daily did not hold, and the preflight reported the failure only after the fact.
  Afterwards all 943 chunks were intact and identical to local, the vectors matched, the
  app answered and a real question returned a cited answer. **The wake-up also moved the
  cluster to Weaviate 1.40.5**; local Docker and the rehearsal copy run 1.39.13, so the
  rehearsal is no longer an exact copy of hosted (`make sync-preflight` warns). This makes
  the keep-alive query above a priority: test that a scheduled read-only query keeps the
  cluster awake, and add the cluster's state to the preflight
  Version changes are now checked with `make weaviate-compat VERSION=<x.y.z>` (a throwaway
  server, the newest hosted backup, optionally `CLIENT=` and `UPGRADE=1`); on 2026-11-23
  it passed for 1.40.5 and failed as expected for 1.28.4, which predates the `hfresh`
  index. The procedure is runbook section 7a. Local and the client are not bumped yet.
- Record the decision on moving to the Flex plan ($45 per month minimum: 7-day backups,
  replication, 99.5% availability, 1,000 collections) and the trigger for it, which is
  SYNC-009 or the first suspension
- **Snapshot retention (moved here from SYNC-008, checked 2026-11-23):** the keep-last-2
  rule does not exist. The bucket is versioned and its one lifecycle rule deletes
  superseded versions of an object (`numNewerVersions: 1`), which does not cap the snapshot
  folders (10 on 2026-11-23). A cap by count needs a small cleanup command or a different
  rule, and must never remove the newest backup that holds the identity fields
  (`20261117T134901Z-promote-after`). Decide, then verify after several backups

**Scope:** runbook, `docs/content-update-workflow.md`; no application code

#### OPS-008 — Trust Boundary for Ingested Content

**Branch:** `feature/OPS-008-ingest-trust`

Crawled pages and transcripts are placed in the LLM context. Anyone who can edit a source
page can plant instructions that reach an answer.

- Treat retrieved text as data in the prompt: delimit it and instruct the model to ignore
  instructions inside it, covered by a prompt test with a planted instruction
- List the trusted sources and record who can edit each one; only the maintainer's
  registry decides what gets crawled
- Review the SYNC-002 diff report for unexpected new instructions-like text before
  promotion, as part of the release gate

**Scope:** `query/generator.py`, prompt tests, runbook

**Acceptance criteria:**

Observability

- Every generation produces one structured log record with provider, model, status,
  tokens, and latency, and no question or answer text — met, verified live for Gemini and
  OpenAI on revision 00021 (real token counts, totals equal input plus output). NVIDIA
  NIM and OpenRouter usage reporting is not verified. A stream abandoned without
  cancelling logs no record.
- Cloud Run logs are JSON with severity and a correlation ID; local logs stay
  human-readable — met, one correlation ID traced from retrieval through inference in
  Cloud Logging
- No user query text or secret appears in INFO logs, verified by tests — met. Retrieval
  logs `query_len` and `query_hash` only (checked live); redaction is covered by tests for
  each provider's error path. Limit: a provider error that echoes prompt text is not
  detected as a secret (documented in the README).
- Context truncation is logged whenever it happens — met by tests; no live truncation has
  occurred yet (largest observed context 7.3k of the 10k budget)

Operations

- A spend cap and a budget alert exist for every paid provider, and the caps are in the
  runbook (OPS-001)
- The public endpoint is either gated or limited, and a scripted burst stays within the
  limits (OPS-002)
- The runtime uses a dedicated service account with only the roles it needs (OPS-003)
- A failed uptime check or an error-rate breach alerts a named person (OPS-004)
- A bad revision can be rolled back with one documented command that has been rehearsed
  (OPS-005)
- Every pull request runs lint, types and tests, and `main` is protected (OPS-006)
- A restore from the bucket into a fresh volume has been rehearsed and matches the hosted
  collection (OPS-007)
- A planted instruction in a retrieved page does not change an answer (OPS-008)

Sync

- An edited page results in updated chunks and no stale duplicates in the hosted
  collection
- A page removed upstream is removed only when explicitly confirmed
- A failed crawl or failed source never deletes existing content
- An unchanged source costs no embedding calls
- The deployed app serves new content without a redeploy
- A pre-sync snapshot of the hosted collection exists before every promotion
- The serving app never holds write credentials
- Apply executes exactly the plan that was reviewed, and refuses a plan whose registry or
  ledger generation has moved
- Every run leaves a manifest, and the page ledger changes only after a verified run
- A promotion copies vectors from local and makes no embedding calls
- Two runs cannot overlap
- Every sync command has machine-readable output and distinct exit codes
- A maintainer can run a preflight that reports missing tools, keys, quota, version drift
  and local-versus-hosted differences before anything changes
- The local and hosted collections are verified identical after every promotion
- Colleagues see no outage while a promotion runs
- A failed promotion can be rolled back from the pre-promotion snapshot

**Out of scope:** running the sync unattended on a schedule (SYNC-009 is a stub that waits
for its preconditions; the video lane is never scheduled), distributed tracing or
an external log platform beyond Cloud Logging, embedding-model migration, multi-tenant
sync, and live webhooks from the content source.

## Epic 12: Answer and Source Quality Calibration (planned)

Epic 11 got the content right: the sync lanes keep hosted equal to what a maintainer
reviewed. This epic is about what a support colleague sees when they ask a question: the
form of the answer, the sources it cites, and how far both can be trusted. It starts from
one observed case and measures before it changes anything.

**The observed case (2026-11-24).** On the first new video (`yWuBQik8KPs`) the question
"How can I work with angled connections in Hydroman?" retrieved the right chunks on local
and on hosted, with identical sources and scores. The answers had four problems:

1. **A list of facts, not a procedure.** The system prompt already asks for "the complete
   numbered procedure" (rule 4) and for citing "once per group of steps" (rule 2). The
   answer ignored both. First find out whether the sources hold a procedure at all: the
   tutorial chunks are narration of a two-minute demonstration, not numbered steps
2. **Duplicate and repeated sources.** The Sources list showed `@ 01:00` twice (the
   `[Transcript]` and `[Visual]` chunk of one 30-second window) and one article twice (two
   chunks of article 312). The body repeated the same two or three citations after nearly
   every sentence
3. **A weak tail.** The default `top_k` is 10, so the list reached 73 to 76% relevance:
   two cross-section articles and a HydroSym page about drawing a slanted connection line,
   which is a different feature
4. **Claims attached to the wrong source.** The local answer credited angled connections
   with a step that belongs to the pilot-lines function at 0:30, and said both functions
   are checked for wall thickness where the transcript says only the pilot lines are. The
   hosted answer of the same question did not repeat either; that is one sample each, not
   a result

Local and hosted differ in one setting that is not in the repo: hosted runs with
`LLM_TEMPERATURE=1` (set on the Cloud Run service), local defaults to `0.3`
(`.env.example`). Same model, same code, same chunks. So answers cannot be compared across
the two, and any prompt change has to be judged at the production setting.

**Status:** not started. All of it is query-side: nothing stored changes, so none of it
goes through a promotion. It needs an app deploy to reach colleagues. The last deploy is
Cloud Run revision `multimodal-rag-00025-juf` (2026-11-17); no query-side commit has
landed since, so the live app and `main` agree on answer behaviour. Follow "Before each
deploy" in `docs/gcp-deployment.txt` (hosted Weaviate is 1.40.5, never tested against the
app's client on purpose).

#### QUAL-001 — Calibration Set and Baseline (first)

**Branch:** `feature/QUAL-001-calibration-set`

Nothing else in this epic is judged without it. The retrieval gate (`config/canaries.yaml`)
checks only that three questions keep their rank; it says nothing about the answer.

- A fixed set of about 10 to 15 questions in `config/`, each with the facts a good answer
  must contain, the sources it should cite, and the form it should take (procedure,
  factual, "the sources do not say"). Include the angled-connection question, the
  reducing-valve question (its answer is in a `[Visual]` chunk that ranked 5th), the three
  canaries, and at least two questions the knowledge base cannot answer
- A command that runs the set against a chosen store and settings, **3 times per question**
  at the production temperature, and reports per question: distinct sources cited, repeated
  citations, required facts present, answer form, and the spread between runs. Output goes
  to `runs/` and is not part of `make test`
- Record the baseline before any change in this epic. A change is kept only if it improves
  the measures it targets and makes none of the others worse

**Scope:** a new module under `src/multimodal_rag/`, `config/`, tests for the scoring only
(the model calls are not tested)

#### QUAL-002 — Temperature and Setting Parity

**Branch:** `feature/QUAL-002-setting-parity`

- Decide the answer temperature and write it down. Whether `1` on Cloud Run was deliberate
  is not recorded anywhere; Gemini 3 guidance suggests leaving it at 1.0, which may be why.
  The decision is the maintainer's
- Whichever value wins is set the same on local and hosted, in `.env.example` and in
  `docs/gcp-deployment.txt`, and `make sync-preflight` warns when the deployed value
  differs from the local one (it already reports the Weaviate version difference)
- If the chosen value is 1.0, QUAL-001's three runs per question are what show the spread

**Scope:** `.env.example`, `docs/gcp-deployment.txt`, `preflight.py`

#### QUAL-003 — One Source per Idea

**Branch:** `feature/QUAL-003-merge-sources`

The merge at retrieval that SYNC-012 deferred until the first video had run. It has now run.

- Group retrieved video chunks by `(source_url, timestamp_seconds)` before the context is
  built: one source per window, taking the better score and keeping both texts, so the
  model sees the narration and what was on screen together
- Group web chunks of one article the same way, so one article is not two sources
- The citation list and the inline links show each source once
- Judge by QUAL-001: fewer repeated citations, no loss of required facts. Check the
  reducing-valve question in particular, where merging should help a value that lives
  only in the visual chunk

**Scope:** `query/retriever.py`, `models/query.py`, the context formatter in
`query/generator.py`

#### QUAL-004 — Relevance Floor and Source Count

**Branch:** `feature/QUAL-004-relevance-floor`

- Measure, do not guess: with QUAL-001 run `top_k` 5, 7 and 10 and a minimum relevance
  score, and look at which tail sources the model used and which only filled the list
- Keep `retrieve()`'s guarantee that at least half the results are video chunks unless the
  measurement shows it adds weak ones; a floor must not remove the only video source
- Possibly different floors for video and web, since the two score differently

**Scope:** `query/retriever.py`, `models/config.py`

#### QUAL-005 — Answer Form: Procedure First

**Branch:** `feature/QUAL-005-answer-form`

The observed case, point 1. The prompt already asks for the right thing, so the cause comes
before the fix.

- Diagnose with QUAL-001 on three groups: questions where the sources contain a numbered
  procedure (does the model produce one?), questions where they hold only narration of a
  demonstration (what should an honest answer look like?), and questions they do not
  answer
- Produce **two or three prompt variants** and compare them on the calibration set at the
  production setting; the maintainer picks and mixes. Candidates: lead with the best
  video link and its timestamp; order the steps as the video shows them; one citation per
  group of steps, enforced rather than requested; say plainly when the sources show a
  demonstration but no written procedure
- Prompt changes alter every answer, so the whole calibration set must not get worse, not
  just the question that triggered this

**Scope:** `query/generator.py` (`SYSTEM_PROMPT`)

#### QUAL-006 — Claims Match Their Sources

**Branch:** `feature/QUAL-006-faithfulness`

The observed case, point 4: a claim true of one function attached to the neighbouring one.

- Add a faithfulness measure to QUAL-001: for each cited claim, is it supported by the
  chunk it cites? Start with the questions that have a known trap (neighbouring windows
  that describe different functions) and decide whether an automated judge is worth its
  cost or whether the required-facts and forbidden-facts lists are enough
- If prompt wording can reduce it (cite the chunk that says it, do not combine two
  functions' steps), test it as a QUAL-005 variant. Otherwise record it as a known limit
  of narration-based sources

**Scope:** the QUAL-001 module; `query/generator.py` only if a variant wins

#### QUAL-007 — Noise in Stored Chunks (measure here, fix in Epic 11)

**Branch:** `docs/QUAL-007-stored-noise`

Some of what hurts an answer is stored, so a prompt cannot fix it.

- Known: March knowledge-base chunks contain the page-footer widget text ("Did this answer
  your question? Thanks for the feedback … There was a problem submitting your feedback"),
  near-empty heading fragments, and several chunks of one article; the 25 older videos
  carry long frame descriptions cut at the embedder's 400-word limit (SYNC-013)
- QUAL-001 counts how often such chunks reach an answer. If it is often, a ticket in the
  sync lanes fixes it through the normal route (normaliser in `ingest/template.py`, then
  a plan and a promotion), and this story ends with that ticket written. No stored data is
  changed here

**Scope:** documentation and a measurement; the fix belongs to Epic 11

### Order of work

| Order | Story | Why here |
|---|---|---|
| 1 | QUAL-001 | Everything else is judged against it |
| 2 | QUAL-002 | The baseline and every later run need the production setting |
| 3 | QUAL-003, QUAL-004 | Retrieval-side, cheap, and they change what the prompt sees |
| 4 | QUAL-005, QUAL-006 | Prompt work, judged after the sources are clean |
| 5 | QUAL-007 | Needs the measurement; may produce a sync ticket |

Deploy once at the end of a batch that passed the calibration set, not per story.

### Epic 12 acceptance criteria

- A calibration set and a command that scores answers exist, and a baseline from before
  any change is recorded
- Local and hosted use the same answer temperature, and the preflight warns when they differ
- A window's narration and description, and an article's chunks, appear as one source
- The angled-connection question gives an ordered answer with one citation per group of
  steps, or says plainly that the sources show a demonstration and no procedure
- No change is kept that makes any question in the calibration set worse
- The deployed revision and the calibration run it was approved on are recorded together

**Out of scope:** changing the embedding model, re-ranking with a second model (revisit if
QUAL-003 and QUAL-004 are not enough), changing stored content (Epic 11 routes), and
automated answer evaluation in production.

---

## Summary

| Epic | Features | Status | Tests |
|------|----------|--------|-------|
| 1 — Ingestion Pipeline | 5 | completed | 51 |
| 2 — Query + UI | 4 | completed | ~80 |
| 3 — Per-source Ingest | 3 | completed | 8 |
| 4 — Voxtral Fallback | 3 | completed; demoted 2026-11-23 | 13 |
| 5 — Visual Grounding | 5 | completed | ~15 |
| 6 — Multimodal Chunk Fusion | 2 | completed | ~8 |
| 7 — Answer-to-KB Pipeline | 3 | completed | 22 |
| 8 — Configurable AI Inference | 6 | completed | 63+ |
| 9 — Shareable Weaviate Inference Storage | 6 | completed | 241+ |
| 10 — Query-Only Ollama Embedding and Cloud Deployment | 1 | completed | TBD |
| 11 — Observability, Hosted Content Sync and Production Hardening | 21 | in progress (Phase 1 and SYNC-000 done and deployed; SYNC-002 next; 17 open) | ~148 |
| 12 — Answer and Source Quality Calibration | 7 | planned (not started) | TBD |
| **Total** | **66** | **10 completed, 1 in progress, 1 planned** | **395 current** |

Per-epic test counts are approximate and overlap where later epics extend earlier modules.
The total is the current non-integration test count, not the sum of the rows.
