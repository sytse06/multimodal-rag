# Multimodal RAG — Support Knowledge Base

A RAG pipeline that turns YouTube tutorials and web documentation into a
searchable, cited support tool. Ask a question in plain English; get an answer
with clickable video timestamp links and source page references.

Handles caption-disabled YouTube videos automatically via Mistral Voxtral audio
transcription — no manual intervention needed.

## How It Works

```
YouTube URLs  ──→ captions available? ──yes──→ youtube-transcript-api ──→ chunks
                                      └─no───→ yt-dlp + Voxtral Mini  ──→ chunks
                                                                            │
Web KB URLs   ──→ Firecrawl crawl ──→ markdown split ──────────────────→ chunks
                                                                            │
                                      embed ──→ Weaviate ←─────────────────┘
                                                  │
User question ──→ embed ──→ similarity search ──→ top-k chunks ──→ LLM ──→ cited answer
```

Answers include clickable `Video Title @ 02:15` timestamp links and KB page links
with section headings.

## Prerequisites

- Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker
- [OpenRouter](https://openrouter.ai/) API key (LLM + embeddings)
- [Firecrawl](https://firecrawl.dev/) API key (web crawling)
- [Mistral](https://console.mistral.ai/) API key (optional — Voxtral fallback for caption-disabled videos)

## Quick Start

```bash
make install      # install dependencies
make docker-up    # start Weaviate
# edit .env with your API keys, config/sources.yaml with your sources
make ingest       # index YouTube + web sources
make run          # launch Gradio chat interface
```

## Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | LLM + embedding access | — |
| `LLM_PROVIDER` | Backend (`openrouter` / `ollama`) | `openrouter` |
| `LLM_MODEL` | Chat model | `google/gemini-3-flash-preview` |
| `EMBEDDING_MODEL` | Embedding model | `nomic-embed-text` |
| `WEAVIATE_URL` | Weaviate instance | `http://localhost:8080` |
| `FIRECRAWL_API_KEY` | Web crawling | — |
| `MISTRAL_API_KEY` | Voxtral audio transcription fallback | — |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Package manager | uv |
| RAG framework | LangChain |
| Vector store | Weaviate |
| LLM + embeddings | OpenRouter / Ollama |
| Transcripts | youtube-transcript-api + Mistral Voxtral |
| Audio download | yt-dlp |
| Web crawling | Firecrawl |
| Data validation | Pydantic + BaseSettings |
| UI | Gradio |

## Development

```bash
make test         # run test suite with coverage
make quality      # ruff + mypy
make pre-commit   # full quality gate
```

## Documentation

- [PRD](docs/PRD.md) — product requirements
- [Epics](docs/epics.md) — feature breakdown
