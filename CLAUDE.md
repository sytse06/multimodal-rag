# CLAUDE.md

## Project Overview

**multimodal-rag** — RAG-powered support knowledge base for Paro Software. Indexes
YouTube tutorial transcripts and web knowledge bases into Weaviate, enabling support
staff to ask natural language questions and receive cited answers with clickable video
timestamps and source page links.

**Tech stack:** Python, LangChain, Gradio, Weaviate, OpenRouter, youtube-transcript-api, Firecrawl

## Development Commands

```bash
make install      # Install dependencies (uv sync)
make dev          # Configure development environment
make docker-up    # Start Weaviate (Docker)
make docker-down  # Stop Weaviate
make ingest       # Run ingestion pipeline (YouTube + web → Weaviate)
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
│ YouTube URLs ──→ youtube-transcript-api          │
│                  ──→ chunk (with timestamps)     │
│                     ──→ embed (OpenRouter)       │
│                        ──→ Weaviate             │
│                                                 │
│ Web URLs ──→ Firecrawl (crawl child pages)      │
│              ──→ chunk (with source URLs)        │
│                 ──→ embed (OpenRouter)           │
│                    ──→ Weaviate                  │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Query Pipeline (Gradio UI)                      │
│                                                 │
│ User question ──→ embed (OpenRouter)            │
│                   ──→ Weaviate similarity search │
│                      ──→ top-k chunks + metadata │
│                         ──→ LLM (OpenRouter)    │
│                            ──→ cited answer     │
└─────────────────────────────────────────────────┘
```

## Project Structure

```
src/multimodal_rag/
├── models/            # Pydantic models
│   ├── config.py      #   AppSettings (BaseSettings, env-based)
│   ├── sources.py     #   YouTubeSource, KnowledgeBaseSource, SourceConfig
│   ├── chunks.py      #   TranscriptChunk, WebChunk, SupportChunk
│   └── query.py       #   SearchResult, Citation, CitedAnswer
├── ingest/            # Ingestion pipeline
│   ├── youtube.py     #   Transcript fetching + chunking
│   ├── web.py         #   Firecrawl crawling + markdown splitting
│   └── __main__.py    #   CLI orchestrator (make ingest)
├── store/             # Vector store layer
│   ├── embeddings.py  #   OpenRouter embedding (batched)
│   └── weaviate.py    #   Weaviate collection management + search
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

- **Branches:** `main` ← `develop` ← `feature/*`
- **Code changes:** always on a feature branch off develop. Never commit code directly to develop or main.
- **Docs-only changes:** commit directly on develop, then merge to main.
- **Merge flow:** feature → develop (--no-ff) → main (--no-ff)
- **Commit format:** `type(scope): description` (conventional commits)
- **Types:** feat, fix, docs, style, refactor, test, chore
- **Quality gate:** run `make quality` and `make test` before every commit. All tests must pass.

## Key Configuration

All configurable via environment variables (`.env`):

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API access | — |
| `OPENROUTER_BASE_URL` | OpenRouter endpoint | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | Chat model | `openai/gpt-4o-mini` |
| `EMBEDDING_MODEL` | Embedding model | `openai/text-embedding-3-small` |
| `WEAVIATE_URL` | Weaviate instance | `http://localhost:8080` |
| `FIRECRAWL_API_KEY` | Firecrawl API access | — |
