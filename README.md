# Multimodal RAG — Support Knowledge Base

A RAG pipeline that turns YouTube tutorials and web documentation into a
searchable, cited support tool. Ask a question in plain English; get an answer
with clickable video timestamp links and source page references.

All YouTube videos are transcribed via Mistral Voxtral audio transcription and
optionally combined with vision LLM frame descriptions — producing fused chunks
that capture both what was said and what was shown on screen.

## How It Works

```
YouTube URLs  ──→ yt-dlp download ──→ Voxtral Mini (audio) ──→ transcript segments ──┐
                                  └──→ ffmpeg keyframes ──→ Vision LLM (recommended) ──→ fused chunks per 30s window
                                                                                        │
Web KB URLs   ──→ Firecrawl crawl ──→ markdown split ──────────────────────────→ chunks
                                                                                        │
                                        embed ──→ Weaviate ←────────────────────────────┘
                                                    │
User question ──→ embed ──→ similarity search ──→ top-k chunks ──→ LLM ──→ cited answer
```

Each video chunk combines `[Transcript] what was said` with `[Visual] what was shown`
in a single 30-second window, keeping related audio and visual context together.
Answers include clickable `Video Title @ 02:15` timestamp links and KB page links
with section headings.

## Prerequisites

- Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker
- [OpenRouter](https://openrouter.ai/) API key (LLM + embeddings)
- [Firecrawl](https://firecrawl.dev/) API key (web crawling)
- [Mistral](https://console.mistral.ai/) API key (required for video transcription via Voxtral)

## Quick Start

```bash
make install      # install dependencies
make docker-up    # start Weaviate
# edit .env with your API keys, config/sources.yaml with your sources
make ingest       # index YouTube + web sources
make purge-video  # remove all video chunks (re-ingest after pipeline changes)
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
| `MISTRAL_API_KEY` | Voxtral audio transcription (required for video ingest) | — |
| `VISION_MODEL` | Vision LLM for frame descriptions (e.g. `openai/gpt-4o-mini`) | disabled |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Package manager | uv |
| RAG framework | LangChain |
| Vector store | Weaviate |
| LLM + embeddings | OpenRouter / Ollama |
| Transcription | Mistral Voxtral Mini (all videos) |
| Visual grounding | Vision LLM via OpenRouter (optional) |
| Video download | yt-dlp + ffmpeg |
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
