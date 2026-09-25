# Multimodal RAG — Support Knowledge Base

An inference application for Paro support staff. Ask a question in plain English and
receive a streamed answer grounded in retrieved support content, with clickable links to
knowledge-base pages and video timestamps. The Gradio interface also supports reviewing
an answer, inspecting its sources, editing an article draft, and saving the article as
Markdown.

The application uses a fixed `SupportChunk` collection in Weaviate. Collection
ingestion is a separate maintainer workflow; colleagues normally only need the
inference path.

## How the application works

```text
Question
  → query embedding
  → Weaviate similarity search
  → ranked support chunks
  → selected chat provider/model
  → streamed cited answer
```

Chat providers are selected explicitly in the UI. Supported providers are OpenRouter,
OpenAI, Gemini, NVIDIA NIM, and Ollama. Chat-provider selection does not change the
embedding provider or the vectors already stored in Weaviate.

## Prerequisites

- Python 3.12–3.14
- [uv](https://docs.astral.sh/uv/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) for macOS,
  Windows, or Linux
- An API key for the chat provider you intend to use: [OpenRouter](https://openrouter.ai/),
  [OpenAI](https://platform.openai.com/), or [Gemini](https://ai.google.dev/). Ollama
  runs locally without an API key.
- [Ollama](https://ollama.com/) with the `nomic-embed-text` model for the current local
  collection

Video/web ingestion additionally requires Firecrawl, Mistral Voxtral, YouTube access,
and `ffmpeg`; those dependencies are not needed for ordinary inference.

### Install Docker

Install Docker Desktop from the [official Docker downloads page](https://www.docker.com/products/docker-desktop/)
for your operating system. Verify that both Docker and Compose are available:

```bash
docker --version
docker compose version
```

Start Docker Desktop before running `make docker-up`.

### Install the fixed embedding model

The current collection was built with Ollama `nomic-embed-text`. Install Ollama for
[macOS, Windows, or Linux](https://ollama.com/download), then start its local service
and download the model before running the application:

```bash
ollama serve              # Linux; Ollama Desktop provides the service on macOS/Windows
ollama pull nomic-embed-text
```

Follow the official Ollama installation instructions for your operating system if the
`ollama` command is not available in your terminal.

The Ollama embedding provider/model and vector dimensions are part of the collection
contract. Do not replace them with another embedding model without rebuilding the
collection.

## Inference quick start

### 1. Install dependencies

```bash
make install
```

This runs `uv sync` and creates or updates the project environment.

### 2. Configure the application

```bash
cp .env.example .env
```

Edit `.env` and set the credentials for the providers you will use. At minimum, set:

- `OPENROUTER_API_KEY` or the key for your selected chat provider;
- `EMBEDDING_PROVIDER=ollama` and `EMBEDDING_MODEL=nomic-embed-text`;
- `WEAVIATE_URL` for the Weaviate instance containing `SupportChunk`.

`GRADIO_SHARE=false` keeps the interface local. Do not commit `.env` or place real
secrets in configuration files.

`make dev` is available for maintainers who intentionally want to replace `.env` with
the checked-in development template. It overwrites the existing `.env`; use it only
when that is deliberate.

### 3. Start local Weaviate

```bash
make docker-up
```

The local instance is available at `http://localhost:8080` by default. It must already
contain the compatible `SupportChunk` collection before inference can return results.

The fixed collection artifact is shared separately through an approved Google Drive
link. Download the snapshot directory outside the repository and keep the JSONL file,
manifest, and checksum out of Git. Restore it with:

```bash
SNAPSHOT_DIR=/path/to/multimodal-rag-weaviate-snapshot make snapshot-restore
```

The restore command validates the checksum, schema, and object count. Do not use an
unverified collection or change the embedding model. The hosted cluster is restored
once by a maintainer; colleagues use its viewer key and never run the cloud restore.

### Hosted Weaviate (optional)

Local Docker remains the default. To use the shared hosted collection, set these values
in `.env` instead of starting Docker:

```dotenv
WEAVIATE_MODE=cloud
WEAVIATE_URL=https://<cluster>.weaviate.cloud
WEAVIATE_VIEWER_API_KEY=<read-only-key>
WEAVIATE_TENANT=
WEAVIATE_VECTOR_DIMENSION=768
```

Use an HTTPS cluster URL and a viewer/read-only key. Keep admin keys restricted to the
one-time bootstrap procedure. The application validates collection existence, schema,
tenant access, stored vector dimension, and query embedding dimension before inference.

### 4. Start Gradio

```bash
make run
```

Open the displayed local URL. Select a provider and model, enter a question, and submit
it. The answer streams into the chat. Use **Review & save as article** to open the
four-step editorial workflow:

1. Review answer
2. Inspect sources
3. Edit draft
4. Save Markdown article

If a provider is temporarily busy or times out, the UI reports a retry/provider-switch
message instead of incorrectly blaming local configuration.

Stop Weaviate when finished:

```bash
make docker-down
```

## Configuration

`.env.example` documents the full configuration surface. Important settings include:

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | Active chat provider | `openrouter` |
| `LLM_MODEL` | Active chat model | `google/gemini-3-flash-preview` |
| `EMBEDDING_PROVIDER` | Fixed collection embedding provider | `ollama` |
| `EMBEDDING_MODEL` | Fixed collection embedding model | `nomic-embed-text` |
| `OPENROUTER_API_KEY` | OpenRouter credential | empty |
| `OPENAI_API_KEY` | OpenAI credential | empty |
| `GEMINI_API_KEY` | Gemini credential | empty |
| `NVIDIA_API_KEY` | NVIDIA NIM credential | empty |
| `OLLAMA_BASE_URL` | Local Ollama endpoint | `http://localhost:11434` |
| `WEAVIATE_URL` | Weaviate endpoint | `http://localhost:8080` |
| `WEAVIATE_MODE` | `local` Docker or `cloud` hosted deployment | `local` |
| `WEAVIATE_VIEWER_API_KEY` | Read-only hosted credential | empty |
| `WEAVIATE_ADMIN_API_KEY` | Maintainer-only bootstrap credential | empty |
| `WEAVIATE_TENANT` | Optional Weaviate tenant | empty |
| `WEAVIATE_VECTOR_DIMENSION` | Expected fixed collection dimension | `768` |
| `GRADIO_SHARE` | Enable a public Gradio share link | `false` |

Provider endpoints, timeouts, retries, context limits, and ingestion settings are also
available in `.env.example`. Provider and model names are explicit; a model name never
implicitly selects a provider.

## Bring your own chat provider

Each colleague may provide their own credential for one or more of the supported chat
providers. Only configured providers appear in the Gradio provider selector.

| Provider | Credential/endpoint | Model value example |
|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL` | `openai/gpt-5.4-mini` |
| OpenAI | `OPENAI_API_KEY` and `OPENAI_BASE_URL` | `gpt-4o-mini` |
| Gemini | `GEMINI_API_KEY` and `GEMINI_BASE_URL` | `gemini-3.6-flash` |
| NVIDIA NIM | `NVIDIA_API_KEY` and `NVIDIA_BASE_URL` | `nvidia/nemotron-3-ultra-550b-a55b` |
| Ollama | `OLLAMA_BASE_URL`; no API key | `nemotron-3.5-lightning:30b-mlx` |

To use a different model from the same provider, set `LLM_MODEL` to that provider's
model identifier and restart the application. For Ollama, download the model first
with `ollama pull <model-name>`. The model must support the application's chat and
streaming path.

This is provider configuration, not unrestricted provider extensibility. Adding a new
provider requires a LangChain integration, typed configuration, factory branch, registry
entry, and contract tests. Chat-provider changes never change the fixed embedding
provider/model or the stored Weaviate vectors.

## Maintainer: ingestion workflow

Ingestion is not required for normal use. Maintainers who need to build or refresh the
local collection can use:

```bash
cp .env.example .env
# configure Firecrawl, Mistral, YouTube, embedding, and Weaviate settings
edit config/sources.yaml
make ingest
```

The ingestion pipeline handles YouTube transcripts with Voxtral fallback, optional visual
grounding, web crawling, chunking, embedding, and Weaviate writes. Keep source credentials
and generated logs outside version control.

Destructive maintenance commands require confirmation:

```bash
make purge-source URL=https://example.com/page
make purge-video
make purge-web
make purge
```

Do not run these commands as part of ordinary onboarding.

## Development checks

```bash
make test         # pytest suite with coverage, excluding integration tests
make quality      # ruff and mypy
make pre-commit   # quality checks followed by tests
```

Integration tests require external services and are run separately:

```bash
make test-integration
```

## Documentation

- [Product requirements](docs/PRD.md)
- [Epics and acceptance criteria](docs/epics.md)
- [Query and inference pipeline](docs/pipeline.md)
- [Contributor and development workflow](CLAUDE.md)

## Current deployment boundary

The application supports local single-node Docker and a hosted Weaviate cluster for the
fixed `SupportChunk` collection. Both targets use the same `nomic-embed-text` Ollama
embeddings and 768-dimensional vectors. Ingestion, snapshot creation, cloud bootstrap,
and collection versioning remain maintainer concerns and are outside colleague onboarding.
