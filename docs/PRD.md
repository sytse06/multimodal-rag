# PRD: Multimodal RAG Support Knowledge Base

## 1. Overview

**Project name:** multimodal-rag

A RAG-powered support knowledge base for Paro Software. Support staff ask
natural language questions and receive synthesized answers grounded in the company's
YouTube tutorial library and web-published knowledge bases, with clickable citations
pointing to exact video timestamps and source pages.

### Objectives

- Enable support staff to quickly find answers across video tutorials and web documentation
- Provide clickable citations: YouTube timestamp links (`&t=Ns`) and knowledge base page URLs
- Show relevance scores per source chunk so staff can judge confidence
- Run locally first, with a path to HuggingFace Spaces deployment

## 2. Target Audience

**Primary:** Paro Software support staff answering customer questions about HydroSym.

**Pain point:** Support knowledge is scattered across dozens of YouTube tutorials and
multiple web knowledge bases. Videos are especially hard to search — staff must watch
them to find relevant information.

**How this solves it:** Semantic search across all sources with direct links to the
exact moment in a video or section of a page.

## 3. Core Features

### 3.1 Ingestion Pipeline (Batch Job)

Separate CLI process that fetches, chunks, embeds, and stores all source material.

*User story:* As a system administrator, I want to run a single command to process all
YouTube videos and knowledge base URLs so the support chatbot has up-to-date content.

*Acceptance criteria:*
- Reads source URLs from a YAML config file (`config/sources.yaml`)
- YouTube: fetches timestamped transcripts via `youtube-transcript-api`
- Web: crawls 3 root URLs and all child pages via Firecrawl
- Chunks text into ~300-500 token segments preserving source metadata
- Video chunks: video title, video URL, start timestamp (seconds), chunk text
- Web chunks: page title, page URL, section heading (if available), chunk text
- Embeds chunks via configurable OpenRouter embedding model (default: `openai/text-embedding-3-small`)
- Stores embeddings + metadata in Weaviate
- Idempotent: re-running skips already-processed sources (keyed on URL hash)
- Triggered via `make ingest` or `uv run python -m multimodal_rag.ingest`

*Technical considerations:*
- Chunking strategy for video: group consecutive transcript segments until ~300-500 tokens, preserve start timestamp of first segment in each chunk
- Chunking strategy for web: split by section headers, fall back to token-based splitting
- Store source type (`video` | `web`) as metadata for citation formatting
- Log progress: number of videos processed, pages crawled, chunks stored

*Priority:* Must-have (v1)

### 3.2 Query Pipeline (RAG)

LangChain-based retrieval and generation pipeline.

*User story:* As a support staff member, I want to ask a question in natural language
and receive an accurate answer with links to the source material.

*Acceptance criteria:*
- Embeds user question via same embedding model as ingestion
- Performs similarity search against Weaviate, retrieves top-k chunks (default k=5)
- Passes retrieved chunks + user question to LLM via OpenRouter
- LLM generates a synthesized answer with inline citations
- Each citation includes: source title, clickable URL (with `&t=Ns` for videos), relevance score
- Maintains conversation history within a session for follow-up questions

*Technical considerations:*
- Use LangChain `BaseChatModel` and `Embeddings` interfaces for all model access — never call provider SDKs directly
- LangChain model client factory: configure provider (OpenRouter, Ollama) via env vars, swap without code changes
- Prompt template must instruct LLM to cite sources using retrieved chunk metadata
- Relevance score: cosine similarity from Weaviate, passed through to UI

*Priority:* Must-have (v1)

### 3.3 Gradio Chat Interface

Support-focused chat UI.

*User story:* As a support staff member, I want a simple chat interface where I can ask
questions and see answers with clickable source links.

*Acceptance criteria:*
- Chat interface with message history
- Answers rendered as markdown with clickable citation links
- Video citations formatted as: `[Video Title @ MM:SS](https://youtube.com/watch?v=ID&t=Ns)`
- Web citations formatted as: `[Page Title](https://url)`
- Relevance scores displayed per citation (e.g. percentage or bar)
- Source type indicator (video icon vs page icon) per citation
- Model selector dropdown (OpenRouter models)
- Clear conversation button

*Technical considerations:*
- `gr.ChatInterface` or `gr.Blocks` with `gr.Chatbot` component
- Markdown rendering handles clickable links natively in Gradio
- No FastAPI layer in v1 — Gradio standalone

*Priority:* Must-have (v1)

### 3.4 Source Configuration

YAML-based source management.

*User story:* As a system administrator, I want to add or remove video URLs and
knowledge base URLs without changing code.

*Acceptance criteria:*
- `config/sources.yaml` defines all sources
- YouTube section: list of video URLs
- Web section: list of root URLs (all child pages crawled automatically)
- Changes take effect on next `make ingest` run

*Example:*
```yaml
youtube:
  - url: https://www.youtube.com/watch?v=KHo5xEaPyAI
    title: "HydroSym Tutorials: 4 minutes Quickstart"
  - url: https://www.youtube.com/watch?v=ANOTHER_ID
    title: "HydroSym Advanced Features"

knowledge_bases:
  - url: https://docs.example.com
    name: "HydroSym Documentation"
  - url: https://support.example.com
    name: "Paro Support Portal"
  - url: https://wiki.example.com
    name: "HydroSym Wiki"
```

*Priority:* Must-have (v1)

## 4. Technical Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python >=3.12 | Standard, ecosystem support |
| Package manager | uv | Project convention |
| RAG framework | LangChain | Mature RAG tooling, provider-agnostic model abstraction |
| Model abstraction | LangChain `BaseChatModel` / `Embeddings` | Swap providers (OpenRouter, Ollama, etc.) without code changes |
| Embeddings | Configurable (default: OpenRouter openai/text-embedding-3-small, local: Ollama nomic-embed-text) | LangChain interface enables provider diversity |
| Vector store | Weaviate | Team experience, Docker for local, Cloud for HF Spaces |
| LLM gateway | Configurable (default: OpenRouter, local: Ollama) | LangChain abstraction — never lock into a single provider |
| Transcript extraction | youtube-transcript-api | Timestamped segments, no compute needed |
| Web crawling | Firecrawl | Handles full-site crawling from root URL |
| Data validation | Pydantic | Type-safe models, config via BaseSettings |
| UI | Gradio | Fast prototyping, HF Spaces deployment path |
| Containerization | Docker (Weaviate) | Local vector store instance |

## 5. Conceptual Data Model

### Weaviate Collections

| Collection | Field | Type | Notes |
|------------|-------|------|-------|
| SupportChunk | chunk_id | UUID | Primary key |
| SupportChunk | text | text | Chunk content |
| SupportChunk | source_type | string | `video` or `web` |
| SupportChunk | source_url | string | Full URL to source |
| SupportChunk | source_title | string | Video/page title |
| SupportChunk | timestamp_seconds | int | Start time (video only, nullable) |
| SupportChunk | section_heading | string | Section header (web only, nullable) |
| SupportChunk | url_hash | string | SHA256 of source URL, for idempotency |
| SupportChunk | ingested_at | datetime | Processing timestamp |

### Pydantic Models

| Model | Purpose |
|-------|---------|
| `SourceConfig` | Parsed `sources.yaml` structure |
| `TranscriptChunk` | Video transcript segment with timestamp metadata |
| `WebChunk` | Web page segment with URL and section metadata |
| `SearchResult` | Retrieved chunk + relevance score |
| `CitedAnswer` | LLM response with structured citations |
| `AppSettings` | BaseSettings for API keys, model config (LLM + embedding model), Weaviate connection |

## 6. UI/UX Design Principles

- **Simplicity first** — support staff need answers fast, not a complex interface
- **Single-page chat** — no navigation, no tabs in v1
- **Citations are first-class** — visually distinct from answer text, always clickable
- **Video vs web distinction** — clear visual indicator for source type
- **Confidence visible** — relevance scores help staff judge answer quality

### Key Screen: Chat Interface

- Top: model selector dropdown + clear conversation button
- Center: scrollable chat history with markdown rendering
- Bottom: text input with send button
- Citations rendered inline in assistant messages as clickable markdown links
- Relevance scores shown as percentage badges next to each citation

## 7. Security Considerations

- **API keys** — OpenRouter and OpenAI keys stored in `.env`, never committed
- **Weaviate** — local Docker instance, no authentication needed for v1
- **Source content** — all sources are already public (YouTube, published knowledge bases)
- **No user auth in v1** — internal tool, network-level access control assumed
- **HF Spaces deployment** — secrets managed via HF Spaces secrets, not environment files

## 8. Development Phases

### Phase 1: MVP (v1)

**Epic 1: Ingestion Pipeline** (completed)

| Feature | Branch | Description |
|---------|--------|-------------|
| CORE-001 | Data models | Pydantic models for chunks, sources, config, query results |
| CORE-002 | YouTube ingest | Transcript fetching + timestamp-preserving chunking |
| CORE-003 | Web ingest | Firecrawl crawling + header-based markdown chunking |
| CORE-004 | Weaviate store | Collection management, OpenRouter embedding, vector storage + search |
| CORE-005 | Ingest CLI | Orchestrator reading sources.yaml, runnable via `make ingest` |

**Epic 2: Query Pipeline + Gradio UI**

| Feature | Branch | Description |
|---------|--------|-------------|
| QUERY-001 | Retrieval chain | Embed question → Weaviate top-k search → format context |
| QUERY-002 | Cited answer generation | Prompt template + LLM call producing CitedAnswer with structured citations |
| QUERY-003 | Gradio chat interface | Chat UI with markdown citations, relevance scores, model selector, clear button |
| QUERY-004 | LangChain model client | Replace direct openai SDK with LangChain BaseChatModel/Embeddings, support OpenRouter + Ollama |

**Epic 3: Per-source Ingestion Pipeline** (completed)

| Feature | Branch | Description |
|---------|--------|-------------|
| INGEST-001 | Per-source ingest loop | Process each YouTube video and each KB page individually — scrape, chunk, embed, store per unit |
| INGEST-002 | Error isolation | Try/except per video and per page — failures logged and skipped, pipeline continues |
| INGEST-003 | Embedding safety | Lower max words 800→400, retry with 200 on context-length errors, warning logs |

**Completion criteria:** Support staff can ask a question and receive a cited answer
linking to specific video timestamps and knowledge base pages.

### Phase 2: Enhancement

- Conversation memory / follow-up question handling
- Source freshness detection (re-ingest changed content)
- Analytics: most asked questions, most cited sources
- Feedback mechanism (thumbs up/down on answers)
- HF Spaces deployment with Weaviate Cloud

### Phase 3: Expansion

- Customer-facing self-service interface
- Additional source types (PDF manuals, release notes)
- Multi-language support (Dutch/English)
- FastAPI layer for API access by other systems

## 9. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| YouTube transcript quality (auto-generated) | Medium | Medium | Verified: test video has clean captions. Flag low-quality transcripts during ingestion |
| Weaviate Docker overhead for local dev | Low | Low | Single container, minimal resources for dozens of videos |
| OpenRouter rate limits | Medium | Low | Implement retry logic with exponential backoff via tenacity |
| Chunking too coarse or too fine | High | Medium | Start with 300-500 tokens, tune based on retrieval quality |
| Citation hallucination (LLM invents sources) | High | Medium | Strict prompt template: only cite from provided chunks. Validate citations against retrieved metadata |

## 10. Future Expansion

- **PDF ingestion** — HydroSym manuals and release notes
- **Multi-language** — Dutch support content
- **Customer portal** — self-service version with simplified UI
- **Incremental ingestion** — watch for new videos/pages automatically
- **Evaluation framework** — measure retrieval quality and answer accuracy
- **FastAPI backend** — expose RAG as API for integration with existing support systems
