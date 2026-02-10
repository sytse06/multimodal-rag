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
multimodal-rag/
├── src/multimodal_rag/
│   ├── __init__.py
│   ├── __main__.py          # Entry point for python -m
│   ├── app.py               # Gradio chat interface
│   └── ingest.py            # Ingestion pipeline
├── tests/
│   └── test_placeholder.py
├── config/
│   ├── development.env      # Dev environment variables
│   ├── .env.example         # Template (committed)
│   └── sources.yaml         # YouTube URLs + knowledge base URLs
├── compose.yaml             # Weaviate Docker service
├── pyproject.toml
├── Makefile
├── PRD.md
└── CLAUDE.md
```

## Code Quality Standards

- **Linting:** ruff (E, W, F, I rules)
- **Type checking:** mypy (strict — disallow_untyped_defs)
- **Testing:** pytest with coverage
- **Line length:** 88

## Git Workflow

- **Branch strategy:** main → develop → feature/*
- **Commit format:** `type(scope): description` (conventional commits)
- **Types:** feat, fix, docs, style, refactor, test, chore

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
