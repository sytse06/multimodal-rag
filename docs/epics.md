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

- Spoken videos use Voxtral when `MISTRAL_API_KEY` is configured; `youtube-transcript-api` remains the fallback when it is absent
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
- Validate the expected schema and vector dimension
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
migration, multi-tenant UI, and collection version management.

---

## Summary

| Epic | Features | Status | Tests |
|------|----------|--------|-------|
| 1 — Ingestion Pipeline | 5 | completed | 51 |
| 2 — Query + UI | 4 | completed | ~80 |
| 3 — Per-source Ingest | 3 | completed | 8 |
| 4 — Voxtral Fallback | 3 | completed | 13 |
| 5 — Visual Grounding | 5 | completed | ~15 |
| 6 — Multimodal Chunk Fusion | 2 | completed | ~8 |
| 7 — Answer-to-KB Pipeline | 3 | completed | 22 |
| 8 — Configurable AI Inference | 6 | completed | 63+ |
| 9 — Shareable Weaviate Inference Storage | 6 | completed | 241+ |
| **Total** | **37** | **9 completed** | **241 current** |

Per-epic test counts are approximate and overlap where later epics extend earlier modules.
The total is the current non-integration test count, not the sum of the rows.
