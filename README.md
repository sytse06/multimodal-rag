# Multimodal RAG — Support Knowledge Base

RAG-powered support tool for Paro Software. Indexes YouTube tutorial transcripts and web knowledge bases into Weaviate, enabling support staff to ask natural language questions and receive cited answers with clickable video timestamps and source page links.

## How It Works

```
YouTube videos ──→ transcript extraction ──→ chunking ──→ embedding ──→ Weaviate
Web KBs        ──→ Firecrawl crawling    ──→ chunking ──→ embedding ──→ Weaviate

User question ──→ embedding ──→ Weaviate search ──→ top-k chunks ──→ LLM ──→ cited answer
```

Answers include:
- Clickable YouTube timestamp links (`Video Title @ 02:15`)
- Clickable knowledge base page links with section headings
- Relevance scores per source

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker (for Weaviate)
- API keys: [OpenRouter](https://openrouter.ai/), [Firecrawl](https://firecrawl.dev/)

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Configure environment
cp config/.env.example config/development.env
# Edit config/development.env with your API keys
make dev

# 3. Start Weaviate
make docker-up

# 4. Configure sources
# Edit config/sources.yaml with your YouTube URLs and knowledge base URLs

# 5. Run ingestion
make ingest

# 6. Launch chat interface
make run
```

## Configuration

All settings are managed via environment variables (`.env`):

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | LLM + embedding API access | — |
| `OPENROUTER_BASE_URL` | OpenRouter endpoint | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | Chat model | `openai/gpt-4o-mini` |
| `EMBEDDING_MODEL` | Embedding model | `openai/text-embedding-3-small` |
| `WEAVIATE_URL` | Weaviate instance | `http://localhost:8080` |
| `FIRECRAWL_API_KEY` | Web crawling API access | — |
| `CHUNK_SIZE` | Target tokens per chunk | `400` |
| `TOP_K` | Retrieved chunks per query | `5` |

Sources are configured in `config/sources.yaml`:

```yaml
youtube:
  - url: https://www.youtube.com/watch?v=KHo5xEaPyAI
    name: "HydroSym Tutorials: 4 minutes Quickstart"

knowledge_bases:
  - url: https://docs.example.com
    name: "Product Documentation"
```

## Project Structure

```
src/multimodal_rag/
├── models/          # Pydantic models (chunks, config, sources, query)
├── ingest/          # Ingestion pipeline (YouTube, web, CLI orchestrator)
│   ├── youtube.py   # Transcript fetching + chunking
│   ├── web.py       # Firecrawl crawling + markdown splitting
│   └── __main__.py  # CLI entrypoint (make ingest)
├── store/           # Vector store layer
│   ├── embeddings.py  # OpenRouter embedding (batched)
│   └── weaviate.py    # Weaviate collection management + search
├── query/           # Query pipeline
│   ├── retriever.py   # Embed query → Weaviate search → SearchResults
│   └── generator.py   # LLM cited answer generation
└── app.py           # Gradio chat interface (make run)
```

## Development

```bash
make test          # Run test suite
make quality       # Ruff lint + mypy type checking
make quality-fix   # Auto-fix lint issues
make pre-commit    # Full quality gate (lint + type check + tests)
```

### Git Workflow

- **Branches:** `main` → `develop` → `feature/*`
- **Commits:** conventional format — `feat(scope): description`

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Package manager | uv |
| RAG framework | LangChain |
| Vector store | Weaviate |
| Embeddings + LLM | OpenRouter (OpenAI-compatible) |
| Transcripts | youtube-transcript-api |
| Web crawling | Firecrawl |
| Data validation | Pydantic + BaseSettings |
| UI | Gradio |

## Documentation

- [PRD](docs/PRD.md) — product requirements
- [Epics](docs/epics.md) — development epics and feature breakdown
