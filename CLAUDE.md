# CLAUDE.md

## Project Overview

**multimodal-rag** — RAG-powered support knowledge base for Paro Software. Indexes
YouTube tutorial transcripts and web knowledge bases into Weaviate, enabling support
staff to ask natural language questions and receive cited answers with clickable video
timestamps and source page links.

**Tech stack:** Python 3.12–3.14, LangChain Core with OpenRouter/OpenAI/Gemini/Ollama integrations, Gradio, Weaviate, youtube-transcript-api, Firecrawl, yt-dlp, Gemini (transcription for videos without manual captions)

## Development Commands

```bash
make install      # Install dependencies (uv sync)
make dev          # Configure development environment
make docker-up    # Start Weaviate (Docker)
make docker-down  # Stop Weaviate
make ingest       # Run maintainer ingestion pipeline (YouTube + web → Weaviate)
make snapshot-export # Export the fixed SupportChunk collection outside Git
make run          # Start Gradio chat interface
make test         # Run test suite with coverage
make quality      # Code quality checks (ruff, mypy)
make quality-fix  # Auto-fix linting issues
make pre-commit   # Full pre-commit validation (quality + tests)
make clean        # Clean build artifacts
make git-status   # Show git overview
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│ Ingestion Pipeline (batch, CLI)                 │
│                                                 │
│ YouTube URLs ──→ youtube-transcript-api         │
│                  │  (manual captions exist)     │
│                  ├─ none, or automatic only     │
│                  │  ──→ yt-dlp (download audio) │
│                  │     ──→ Gemini transcription │
│                  ──→ chunk (30 s windows)       │
│                     ──→ embed (LangChain)        │
│                        ──→ Weaviate             │
│                                                 │
│ Web URLs ──→ Firecrawl (crawl child pages)      │
│              ──→ chunk (with source URLs)        │
│                 ──→ embed (LangChain)            │
│                    ──→ Weaviate                  │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Query Pipeline (Gradio UI)                      │
│                                                 │
│ User question ──→ embed (LangChain)             │
│                   ──→ Weaviate similarity search │
│                      ──→ top-k chunks + metadata │
│                         ──→ LLM (LangChain)     │
│                            ──→ cited answer     │
└─────────────────────────────────────────────────┘
```

## Project Structure

```
src/multimodal_rag/
├── models/            # Pydantic models + factories
│   ├── config.py      #   AppSettings (BaseSettings, env-based)
│   ├── llm.py         #   LangChain model factories (chat + embeddings)
│   ├── sources.py     #   YouTubeSource, KnowledgeBaseSource, SourceConfig
│   ├── chunks.py      #   TranscriptChunk, WebChunk, SupportChunk
│   └── query.py       #   SearchResult, Citation, CitedAnswer
├── ingest/            # Ingestion pipeline
│   ├── transcripts.py #   Transcript source: manual captions, else Gemini
│   ├── videos.py      #   Per-video ingest (make ingest, make sync-videos)
│   ├── youtube.py     #   Video id and cookie helpers; legacy captions chunking
│   ├── voxtral.py     #   Voxtral STT: demoted 2026-11-23, no longer called
│   ├── web.py         #   Firecrawl crawling + markdown splitting
│   └── __main__.py    #   CLI orchestrator (make ingest)
├── store/             # Vector store layer
│   ├── embeddings.py  #   Provider-specific embeddings (batched)
│   ├── snapshot.py    #   Fixed collection export + manifest
│   └── weaviate.py    #   Local/cloud collection management + search
├── query/             # Query pipeline
│   ├── retriever.py   #   Embed query → Weaviate search → SearchResults
│   └── generator.py   #   LLM cited answer generation
└── app.py             # Gradio chat interface (make run)
```

## Code Quality Standards

- **Linting:** ruff (E, W, F, I rules)
- **Type checking:** mypy (strict — disallow_untyped_defs)
- **Testing:** pytest with coverage
- **Line length:** 88

## Git Workflow

- **Branches:** `main` ← `feature/*`
- **All changes:** create a feature branch directly from main. Never commit directly to main.
- **Merge flow:** keep all work for an epic on its feature branch; merge feature → main
  (`--no-ff`) only after the epic's acceptance criteria and release checks are complete.
- **Commit format:** `type(scope): description` (conventional commits)
- **Types:** feat, fix, docs, style, refactor, test, chore
- **Quality gate:** run `make quality` and `make test` before code commits. Documentation-only
  commits may use the lighter check agreed for that change. All tests must pass before merging.

## Key Configuration

All configurable via environment variables (`.env`):

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLM_PROVIDER` | LLM backend (`openrouter`, `openai`, `gemini`, `ollama`, or `nvidia`) | `openrouter` |
| `EMBEDDING_PROVIDER` | Embedding backend (`openrouter` or `ollama`) | `openrouter` |
| `OPENROUTER_API_KEY` | OpenRouter API access | — |
| `OPENAI_API_KEY` | OpenAI API access | — |
| `GEMINI_API_KEY` | Gemini API access (chat, and video transcription when a video has no manual captions) | — |
| `NVIDIA_API_KEY` | NVIDIA NIM API access | — |
| `OPENROUTER_BASE_URL` | OpenRouter endpoint | `https://openrouter.ai/api/v1` |
| `OLLAMA_BASE_URL` | Ollama endpoint | `http://localhost:11434` |
| `LLM_MODEL` | Chat model | Provider registry default |
| `EMBEDDING_MODEL` | Embedding model | `nomic-embed-text` |
| `WEAVIATE_URL` | Weaviate instance | `http://localhost:8080` |
| `WEAVIATE_MODE` | Weaviate deployment (`local` or `cloud`) | `local` |
| `WEAVIATE_API_KEY` | Backward-compatible hosted Weaviate key | — |
| `WEAVIATE_ADMIN_API_KEY` | Hosted bootstrap/maintenance key | — |
| `WEAVIATE_VIEWER_API_KEY` | Hosted runtime/read-only key | — |
| `FIRECRAWL_API_KEY` | Firecrawl API access | — |
| `MISTRAL_API_KEY` | Legacy Voxtral key: demoted, not used by ingest, safe to leave unset | — |
| `VISION_MODEL` | Vision LLM for frame/screenshot description (empty = disabled) | `""` |
| `GRADIO_SHARE` | Enable a public Gradio share link | `false` |

The fixed `SupportChunk` collection uses Ollama `nomic-embed-text` vectors with a
768-dimensional space. Local Docker and hosted Weaviate are interchangeable deployment
targets, but they must contain a compatible collection. Snapshot artifacts and API keys stay
outside Git; only deliberately selected documentation or query results are tracked.
