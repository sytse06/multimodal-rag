# Debugging Guide: Ingestion Pipeline

**Date:** 2026-02-16
**Project:** multimodal-rag
**Pipeline:** `src/multimodal_rag/ingest/`

This document chronicles every major issue encountered during the development and operation of the ingestion pipeline, which fetches YouTube transcripts and web knowledge base content and indexes them into Weaviate for RAG queries.

## Table of Contents

1. [Environment Issues](#environment-issues)
2. [External Service Issues](#external-service-issues)
3. [Dependency Issues](#dependency-issues)
4. [Quick Reference](#quick-reference)

---

## Environment Issues

### Issue 1: Shell Environment Conflicts (EMBEDDING_MODEL, FIRECRAWL_API_KEY)

**Severity:** HIGH — Silent config override causing runtime failures
**Component:** `src/multimodal_rag/models/config.py` (Pydantic BaseSettings)

#### Symptom

`EMBEDDING_MODEL` was set to `ollama/nomic-embed-text` in `.env`, but Ollama expects bare model names like `nomic-embed-text`. Even after fixing `.env` to the correct value, the wrong value persisted and caused embedding failures.

Similarly, `FIRECRAWL_API_KEY` in the shell environment had an old value without the `fc-` prefix, causing HTTP 401 Unauthorized errors from Firecrawl even though `.env` had the correct value.

#### Root Cause Chain

1. `.env` originally had `EMBEDDING_MODEL=ollama/nomic-embed-text` (wrong — Ollama needs bare name) and `FIRECRAWL_API_KEY` without the `fc-` prefix (wrong — Firecrawl requires it)
2. The oh-my-zsh `dotenv` plugin auto-exports `.env` vars into the shell environment on `cd` into project directory
3. Pydantic `BaseSettings` priority order (highest wins):
   - Shell environment variable
   - `.env` file
   - Code defaults in `AppSettings`
4. Even after fixing `.env` to correct values, the shell environment variables still had the old values
5. Additionally, a `load_project` function in `.zshrc` uses `set -a; source .env; set +a` which also exports vars into shell

**Result:** The shell environment variables overrode the corrected `.env` file values.

#### How Pydantic BaseSettings Works

```python
class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    embedding_model: str = "nomic-embed-text"  # ← Code default (lowest priority)
```

Priority resolution:
1. **Shell env var** (`export EMBEDDING_MODEL=...`) — HIGHEST
2. **`.env` file** (`EMBEDDING_MODEL=...`)
3. **Code default** (`embedding_model: str = "..."`) — LOWEST

#### Fix

Three-part fix:

1. **Remove oh-my-zsh dotenv plugin** from `.zshrc`:
   ```bash
   # Before
   plugins=(git dotenv other-plugins)

   # After
   plugins=(git other-plugins)
   ```

2. **Fix `.env` to correct value**:
   ```bash
   # For Ollama (bare name only)
   EMBEDDING_MODEL=nomic-embed-text

   # For OpenRouter (provider/model format)
   EMBEDDING_MODEL=openai/text-embedding-3-small
   ```

3. **Clear stale shell env vars**:
   ```bash
   unset EMBEDDING_MODEL
   unset FIRECRAWL_API_KEY
   # OR restart terminal
   ```

#### Verification

```bash
# Check if shell env vars exist
echo $EMBEDDING_MODEL
echo $FIRECRAWL_API_KEY

# If set, unset them
unset EMBEDDING_MODEL
unset FIRECRAWL_API_KEY

# Verify .env file
grep EMBEDDING_MODEL .env
grep FIRECRAWL_API_KEY .env

# Run ingestion and check logs for model name
uv run python -m multimodal_rag.ingest
```

#### Lesson

**Pydantic BaseSettings + oh-my-zsh dotenv plugin + manual .env sourcing = triple-loading nightmare.**

Shell environment variables **always** win over `.env` file values. If you change `.env`, you must also clear any shell env vars.

#### Configuration Reference: Model Naming Conventions

| Provider | Format | Example | Wrong Example |
|----------|--------|---------|---------------|
| OpenRouter | `provider/model` | `openai/text-embedding-3-small` | `text-embedding-3-small` |
| Ollama | Bare name only | `nomic-embed-text` | `ollama/nomic-embed-text` |

---

### Issue 2: Python Alias Breaking venv Activation

**Severity:** MEDIUM — Blocks venv activation entirely
**Component:** Shell config (`.zshrc`)

#### Symptom

```bash
$ source .venv/bin/activate
/Users/sytsevanderschaaf/Documents/Dev/Projects/multimodal-rag/.venv/bin/activate:31: parse error near `-m'
```

#### Root Cause

A `python` alias in `.zshrc` contained an `if` statement:

```bash
# Problematic alias
alias python='if [[ -n "$VIRTUAL_ENV" ]]; then "$VIRTUAL_ENV/bin/python"; else /usr/bin/python3; fi'
```

When the activate script defined `pydoc()` using `python -m pydoc`, the alias expanded the `if` block inside the function definition, causing a syntax error:

```bash
# What the activate script tried to do
pydoc () {
    python -m pydoc "$@"
}

# What actually got expanded
pydoc () {
    if [[ -n "$VIRTUAL_ENV" ]]; then "$VIRTUAL_ENV/bin/python"; else /usr/bin/python3; fi -m pydoc "$@"
}
# ↑ Syntax error: '-m' appears after fi statement
```

#### Fix

Change the `python` alias to a function in `.zshrc`:

```bash
# Correct approach — use function instead of alias
python() {
    if [[ -n "$VIRTUAL_ENV" ]]; then
        "$VIRTUAL_ENV/bin/python" "$@"
    else
        /usr/bin/python3 "$@"
    fi
}
```

Functions can contain control flow and accept arguments properly. Aliases are simple text replacements and fail when expanded inside other code.

#### Verification

```bash
source .venv/bin/activate
# Should succeed without parse errors

which python
# Output: /Users/.../multimodal-rag/.venv/bin/python

deactivate
```

#### Lesson

**Never use aliases with control flow (if/while/case). Use shell functions instead.**

Aliases are text substitution only and break when expanded in unexpected contexts.

---

## External Service Issues

### Issue 3: YouTube Transcript Language Fallback

**Severity:** LOW — Some videos fail to ingest
**Component:** `src/multimodal_rag/ingest/youtube.py`

#### Symptom

Some YouTube videos failed with:

```
youtube_transcript_api.exceptions.NoTranscriptFound: No transcripts were found for video ID: [...]
```

Videos **did** have English subtitles, but they were tagged as `en-GB` (British English) rather than `en`.

#### Root Cause

`youtube-transcript-api` `fetch()` was called with only `languages=["en"]`:

```python
# Original code
transcript = ytt_api.fetch(video_id, languages=["en"])
```

The library requires **exact language code matches**. It does not fall back from `en` to `en-GB` automatically.

#### Fix

Add regional English variants to the language list:

```python
# File: src/multimodal_rag/ingest/youtube.py, line 29
transcript = ytt_api.fetch(video_id, languages=["en", "en-GB", "en-US"])
```

The library tries each language in order and returns the first match.

#### Verification

Test with a video known to have `en-GB` subtitles:

```python
from youtube_transcript_api import YouTubeTranscriptApi
ytt_api = YouTubeTranscriptApi()

# This will fail if video only has en-GB
# transcript = ytt_api.fetch("VIDEO_ID", languages=["en"])

# This will succeed
transcript = ytt_api.fetch("VIDEO_ID", languages=["en", "en-GB", "en-US"])
```

#### Lesson

**YouTube transcript language codes are specific regional variants. Always include `en`, `en-GB`, and `en-US` for English content.**

The library does not perform fuzzy language matching.

---

### Issue 4: Firecrawl SDK v2 Breaking Changes

**Severity:** HIGH — Complete API incompatibility
**Component:** `src/multimodal_rag/ingest/web.py`, `tests/test_web_ingest.py`

#### Symptom

```python
AttributeError: 'FirecrawlApp' object has no attribute 'crawl_url'
```

#### Root Cause

Firecrawl Python SDK v2 renamed methods and changed the entire API from dict-based to typed object-based.

| Aspect | v1 API | v2 API |
|--------|--------|--------|
| Method name | `crawl_url()` | `crawl()` |
| Parameters | `params={"limit": ..., "scrapeOptions": {...}}` | `limit=..., scrape_options=ScrapeOptions(...)` |
| Return type | `list[dict]` | `CrawlJob` with `.data: list[Document]` |
| Content access | `result[0]["content"]` | `result.data[0].markdown` |
| Metadata access | `result[0]["url"]`, `result[0]["title"]` | `result.data[0].metadata.source_url`, `.metadata.title` |

#### Fix

**Updated code in `src/multimodal_rag/ingest/web.py`:**

```python
from firecrawl import FirecrawlApp
from firecrawl.v2.types import ScrapeOptions

def crawl_knowledge_base(
    root_url: str,
    api_key: str,
    limit: int = 100,
) -> list[dict[str, str]]:
    app = FirecrawlApp(api_key=api_key)

    # v2 API call
    result = app.crawl(
        root_url,
        limit=limit,
        max_concurrency=1,  # Added for rate limiting
        scrape_options=ScrapeOptions(
            formats=["markdown"],
            only_main_content=True,
        ),
    )

    pages: list[dict[str, str]] = []
    for doc in result.data:  # ← .data returns list of Document objects
        markdown = doc.markdown or ""
        meta = doc.metadata
        url = meta.source_url if meta and meta.source_url else root_url
        title = meta.title if meta else ""
        if markdown.strip():
            pages.append({"url": url, "title": title or "", "content": markdown})

    return pages
```

**Updated tests in `tests/test_web_ingest.py`:**

```python
from unittest.mock import MagicMock

def _make_doc(url: str, title: str, markdown: str) -> MagicMock:
    """Create a mock Firecrawl Document."""
    doc = MagicMock()
    doc.markdown = markdown
    doc.metadata = MagicMock()
    doc.metadata.source_url = url
    doc.metadata.title = title
    return doc

def _make_crawl_result(pages: list[dict[str, str]]) -> MagicMock:
    """Create a mock CrawlJob result."""
    result = MagicMock()
    result.data = [
        _make_doc(p["url"], p.get("title", ""), p["content"]) for p in pages
    ]
    return result

@patch("multimodal_rag.ingest.web.FirecrawlApp")
def test_returns_pages_with_content(mock_fc_cls: MagicMock) -> None:
    mock_app = MagicMock()
    mock_fc_cls.return_value = mock_app
    mock_app.crawl.return_value = _make_crawl_result([
        {"url": "https://example.com/a", "title": "A", "content": "# Hello"},
    ])
    pages = crawl_knowledge_base("https://example.com", "fake-key", limit=10)
    assert len(pages) == 1
```

#### Verification

```bash
# Check installed SDK version
uv pip show firecrawl

# Run tests to verify mocking works
uv run pytest tests/test_web_ingest.py -v

# Run actual ingestion (requires valid API key)
uv run python -m multimodal_rag.ingest
```

#### Lesson

**Major version bumps in external SDKs require comprehensive API review. Update both production code and all test mocks.**

The v2 API is more type-safe but requires explicit typed parameter objects instead of dicts.

---

### Issue 5: YouTube IP Rate Limiting (HTTP 429)

**Severity:** HIGH — Blocks ingestion of multiple videos
**Component:** `src/multimodal_rag/ingest/__main__.py`

#### Symptom

```python
youtube_transcript_api.exceptions.IpBlocked: HTTP 429: Too Many Requests
```

Occurred when ingesting 25+ videos in rapid succession during bulk ingestion.

#### Root Cause

YouTube's transcript API has aggressive rate limiting. Too many requests from the same IP address in a short time window results in a temporary IP ban.

#### Fix

Add rate limiting between video fetches:

```python
# File: src/multimodal_rag/ingest/__main__.py, lines 44-54

for i, yt in enumerate(sources.youtube):
    if i > 0:
        time.sleep(2)  # ← Rate limit: 2 seconds between videos
    logger.info("Processing YouTube: %s", yt.name)
    tc = fetch_transcript_chunks(
        video_url=str(yt.url),
        source_name=yt.name,
        target_tokens=settings.chunk_size,
    )
    all_chunks.extend(SupportChunk.from_transcript_chunk(c) for c in tc)
```

**Rate limits:**
- 2 seconds between videos works reliably for 25+ videos
- If still hitting 429s, increase to 3-5 seconds

#### Recovery from IP Ban

**There is no code workaround.** If you get IP banned:

1. **Wait 2-6 hours** for the ban to lift
2. Resume ingestion with rate limiting in place
3. Consider ingesting in smaller batches (5-10 videos at a time)

**Do NOT:**
- Retry immediately — this extends the ban
- Use proxies/VPNs without permission — violates YouTube ToS
- Remove the rate limiting

#### Verification

```bash
# Check ingestion logs for timing
uv run python -m multimodal_rag.ingest 2>&1 | grep "Processing YouTube"

# Output should show ~2 second gaps:
# 2026-02-16 10:00:00 ... Processing YouTube: Video 1
# 2026-02-16 10:00:02 ... Processing YouTube: Video 2
# 2026-02-16 10:00:04 ... Processing YouTube: Video 3
```

#### Lesson

**External APIs with generous free tiers still have rate limits. Always add throttling for bulk operations.**

IP bans cannot be bypassed programmatically — prevention is the only strategy.

---

### Issue 6: Firecrawl Concurrency Limit (HTTP 401)

**Severity:** MEDIUM — Blocks parallel crawling
**Component:** `src/multimodal_rag/ingest/__main__.py`, `src/multimodal_rag/ingest/web.py`

#### Symptom

```
firecrawl.exceptions.UnauthorizedError: HTTP 401
Error message: "Concurrency limit exceeded for your plan"
```

Occurred when crawling multiple knowledge bases simultaneously.

#### Root Cause

Firecrawl free tier has concurrency limits (typically 1 concurrent crawl). Running multiple `crawl()` calls in parallel exceeded this limit.

The Firecrawl API returns HTTP 401 (not 429) for concurrency errors, which is misleading — it looks like an auth error but it's actually a rate limit.

#### Fix

Two-part fix:

**1. Limit per-crawl concurrency:**

```python
# File: src/multimodal_rag/ingest/web.py, lines 20-29

result = app.crawl(
    root_url,
    limit=limit,
    max_concurrency=1,  # ← Force sequential page crawling
    scrape_options=ScrapeOptions(
        formats=["markdown"],
        only_main_content=True,
    ),
)
```

**2. Add delay between knowledge base crawls:**

```python
# File: src/multimodal_rag/ingest/__main__.py, lines 56-67

for i, kb in enumerate(sources.kb_sources):
    if i > 0:
        time.sleep(5)  # ← 5 seconds between KB crawls
    logger.info("Processing KB: %s", kb.name)
    wc = fetch_web_chunks(
        root_url=str(kb.url),
        source_name=kb.name,
        api_key=settings.firecrawl_api_key,
        target_tokens=settings.chunk_size,
    )
    all_chunks.extend(SupportChunk.from_web_chunk(c) for c in wc)
```

#### Rate Limits by Plan

| Plan | Concurrent Crawls | Pages/Month |
|------|------------------|-------------|
| Free | 1 | 500 |
| Starter | 2 | 5,000 |
| Standard | 5 | 50,000 |

#### Verification

```bash
# Check API plan limits at https://firecrawl.dev/dashboard

# Watch logs for sequential processing
uv run python -m multimodal_rag.ingest 2>&1 | grep "Processing KB"

# Output should show ~5 second gaps:
# 2026-02-16 10:00:00 ... Processing KB: Knowledge Base 1
# 2026-02-16 10:00:05 ... Processing KB: Knowledge Base 2
```

#### Lesson

**HTTP 401 is not always an auth error. Check API docs for rate limit and concurrency constraints.**

Sequential processing with delays is the only solution on free/low-tier plans.

---

### Issue 7: Firecrawl API Key Invalid (HTTP 401)

**Severity:** HIGH — Blocks all web ingestion
**Component:** `.env`, `src/multimodal_rag/models/config.py`

#### Symptom

```
firecrawl.exceptions.UnauthorizedError: Invalid token
HTTP 401
```

All Firecrawl `crawl()` requests fail immediately with authentication error.

#### Root Cause

Firecrawl API key is expired, revoked, or incorrectly copied in `.env` file.

#### Fix

1. **Regenerate API key** at https://firecrawl.dev/dashboard/api-keys
2. **Update `.env`:**
   ```bash
   FIRECRAWL_API_KEY=fc-your-new-key-here
   ```
3. **Clear any shell env var:**
   ```bash
   unset FIRECRAWL_API_KEY
   # OR restart terminal
   ```
4. **Verify config loading:**
   ```python
   from multimodal_rag.models.config import AppSettings
   settings = AppSettings()
   print(settings.firecrawl_api_key)  # Should show new key
   ```

#### Verification

```bash
# Test API key with minimal crawl
uv run python -c "
from firecrawl import FirecrawlApp
app = FirecrawlApp(api_key='fc-your-key')
result = app.scrape('https://example.com')
print('API key valid')
"
```

#### Distinguishing Auth Errors from Concurrency Errors

| Error Message | Cause |
|--------------|-------|
| `"Invalid token"` | Bad/expired API key |
| `"Concurrency limit exceeded"` | Rate limiting (see Issue 6) |
| HTTP 401 with no message | Check dashboard for API key status |

#### Lesson

**Status:** UNRESOLVED in original debugging session — user needed to regenerate API key.

**Always test API keys in isolation before debugging complex pipelines.**

---

## Dependency Issues

### Issue 8: Weaviate Client + Python 3.12.12 Typing Recursion

**Severity:** HIGH — Blocks all imports
**Component:** `weaviate-client` 4.19.2, Python 3.12.12

#### Symptom

```python
KeyboardInterrupt

# Stack trace shows infinite recursion:
File "lib/python3.12/typing.py", line ..., in _is_dunder
  ...
File "site-packages/weaviate/collections/aggregations/hybrid/executor.py", line ...
  [repeating hundreds of times]
```

The import of `weaviate` client hangs indefinitely, consuming CPU. Interrupting with Ctrl+C shows a `KeyboardInterrupt` in Python's `typing` module during recursive type resolution.

#### Root Cause

Python 3.12.12 has a **regression** in the `typing` module that causes infinite recursion in `_is_dunder()` when `weaviate-client` 4.19.2 constructs generic classes.

This is NOT a transient issue — it's a reproducible bug in Python 3.12.12.

#### Fix

**Downgrade Python to 3.12.11:**

```bash
# Pin Python version in project
uv python pin 3.12.11

# Remove existing venv
rm -rf .venv

# Recreate venv with pinned version
uv sync
```

The `.python-version` file now locks this project to Python 3.12.11 until the 3.12.12 typing regression is fixed upstream.

#### Verification

```bash
# Check Python version
python --version
# Should show: Python 3.12.11

# Check .python-version file
cat .python-version
# Should show: 3.12.11

# Test import in isolation
python -c "import weaviate; print('Import successful')"

# If successful, test full ingestion
uv run python -m multimodal_rag.ingest
```

#### Environment Details

```
weaviate-client: 4.19.2
Python: 3.12.11 (pinned via .python-version)
OS: macOS (Darwin 24.6.0)
pydantic: 2.x
```

#### Lesson

**Python 3.12.12 has a typing regression that breaks weaviate-client. Pin Python to 3.12.11 via `uv python pin`.**

The `.python-version` file ensures all team members use the same Python version.

---

### Issue 9: Ollama Embedding Context Length Overflow

**Severity:** HIGH — Blocks all web ingestion when using Ollama
**Component:** `src/multimodal_rag/store/embeddings.py`, Ollama `/api/embed` endpoint

#### Symptom

```
ollama._types.ResponseError: the input length exceeds the context length (status code: 400)
```

Occurs during `store.add_chunks()` — **after** web crawling succeeds, during the embedding step.

#### Root Cause

Ollama's `/api/embed` endpoint sums all input token lengths across a batch. The original batch size of 100 texts easily exceeded `nomic-embed-text`'s 8192-token context window. Even batch size 10 could overflow with longer web chunks.

**Key insight:** Ollama's context window is shared across ALL texts in a single API call, not per-text. A batch of 10 chunks averaging 1000 tokens each = 10,000 tokens → exceeds 8192 limit.

#### Fix (Two-Part)

**Part 1: Code fix in `src/multimodal_rag/store/embeddings.py`**

```python
# Constants at module level
_BATCH_SIZE = 5  # ← Reduced from 100
_MAX_WORDS = 800  # ← New: truncate texts before embedding

def _truncate(text: str) -> str:
    """Truncate text to _MAX_WORDS words to prevent context overflow."""
    words = text.split()
    if len(words) <= _MAX_WORDS:
        return text
    return " ".join(words[:_MAX_WORDS])

async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, truncating each to prevent overflow."""
    truncated = [_truncate(t) for t in texts]
    # ... rest of embedding logic
```

**Worst case calculation:**
- 5 texts × 800 words × 1.3 tokens/word = 5,200 tokens
- Safely under 8192-token limit

**Part 2: Ollama model fix — increase context window**

Create a custom Ollama model with doubled context window:

```bash
# Create Modelfile
printf 'FROM nomic-embed-text\nPARAMETER num_ctx 16384\n' > /tmp/Modelfile

# Create custom model
ollama create nomic-embed-text-16k -f /tmp/Modelfile

# Verify
ollama show nomic-embed-text-16k
```

Update `.env` and `src/multimodal_rag/models/config.py`:

```python
# In config.py, update default
embedding_model: str = "nomic-embed-text-16k"  # ← Changed from nomic-embed-text
```

```bash
# In .env
EMBEDDING_MODEL=nomic-embed-text-16k
```

**IMPORTANT:** After changing the embedding model name, you **must** purge Weaviate and re-ingest:

```bash
# Stop Weaviate
make docker-down

# Clear Weaviate data
rm -rf weaviate-data/

# Start fresh Weaviate
make docker-up

# Re-ingest all sources
make ingest
```

The model name is part of the embedding identity — switching models requires rebuilding the vector store.

#### Verification

```bash
# Check Ollama model exists
ollama list | grep nomic-embed-text-16k

# Check configuration
uv run python -c "
from multimodal_rag.models.config import AppSettings
s = AppSettings()
print(f'Embedding model: {s.embedding_model}')
"

# Test embedding a large batch
uv run python -c "
from multimodal_rag.store.embeddings import embed_texts
import asyncio

# Simulate worst case: 5 × 800 words
texts = [' '.join(['word'] * 800) for _ in range(5)]

async def test():
    embeddings = await embed_texts(texts)
    print(f'Embedded {len(embeddings)} texts successfully')

asyncio.run(test())
"

# Full ingestion test
make ingest
```

#### Why Not Just Increase Batch Size?

With the custom 16k context model, you could theoretically use larger batches. However:

- Keeping batch size small (5) + truncation provides defense in depth
- Some pages may have extremely long chunks even after splitting
- Ollama's token counting is approximate — safer to leave headroom
- Performance difference between batch size 5 and 50 is negligible for ingestion

#### Lesson

**When using Ollama for embeddings, the context window is shared across ALL texts in a single API call. Keep batches small and texts bounded.**

Creating a custom Ollama model with a larger `num_ctx` is cheap insurance and allows for comfortable headroom.

#### Related Configuration

| Parameter | Location | Value | Purpose |
|-----------|----------|-------|---------|
| `_BATCH_SIZE` | `store/embeddings.py` | 5 | Texts per API call |
| `_MAX_WORDS` | `store/embeddings.py` | 800 | Words per text (truncation) |
| `num_ctx` | Ollama model | 16384 | Context window (tokens) |

---

### Issue 10: Duplicate Chunks from Non-Deterministic UUIDs

**Severity:** MEDIUM — Bloats vector store with duplicates on every re-ingest
**Component:** `src/multimodal_rag/models/chunks.py`

#### Symptom

Every time `make ingest` runs, Weaviate's chunk count increases even though the source content hasn't changed. Re-ingesting the same 3 YouTube videos and 2 knowledge bases creates new chunks instead of updating existing ones.

#### Root Cause

`SupportChunk.chunk_id` used `uuid4()` (random UUID):

```python
# Original code (WRONG)
class SupportChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    # ...
```

Every instantiation of `SupportChunk` generates a **new** random UUID, even for identical content. Weaviate treats these as different objects and inserts duplicates.

#### Fix

Use **deterministic UUIDs** based on content:

```python
# File: src/multimodal_rag/models/chunks.py
from uuid import uuid5, NAMESPACE_URL

class SupportChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))  # Default for safety
    # ...

    @staticmethod
    def _make_deterministic_id(source_url: str, text: str) -> str:
        """Generate deterministic UUID from source URL and text content."""
        composite_key = f"{source_url}|{text}"
        return str(uuid5(NAMESPACE_URL, composite_key))

    @classmethod
    def from_transcript_chunk(cls, tc: "TranscriptChunk") -> "SupportChunk":
        """Convert TranscriptChunk to SupportChunk with deterministic ID."""
        return cls(
            chunk_id=cls._make_deterministic_id(tc.video_url, tc.text),
            # ... rest of fields
        )

    @classmethod
    def from_web_chunk(cls, wc: "WebChunk") -> "SupportChunk":
        """Convert WebChunk to SupportChunk with deterministic ID."""
        return cls(
            chunk_id=cls._make_deterministic_id(wc.page_url, wc.text),
            # ... rest of fields
        )
```

**How `uuid5()` works:**
- Takes a namespace (we use `NAMESPACE_URL`) and a string
- Produces the **same UUID** for the same input every time
- Different inputs produce different UUIDs (via SHA-1 hash)

**Result:**
- Same content → same UUID → Weaviate upserts (updates existing) instead of inserting duplicate
- Changed content → different UUID → new chunk inserted
- Sources can stay in `config/sources.yaml` permanently — re-ingesting is a no-op for unchanged content

#### Verification

```python
# Test deterministic ID generation
from multimodal_rag.models.chunks import SupportChunk

# Create two chunks with identical content
chunk1 = SupportChunk._make_deterministic_id("https://example.com", "Hello world")
chunk2 = SupportChunk._make_deterministic_id("https://example.com", "Hello world")

assert chunk1 == chunk2  # Same content = same ID

# Different content = different ID
chunk3 = SupportChunk._make_deterministic_id("https://example.com", "Different text")
assert chunk1 != chunk3
```

```bash
# Test idempotent ingestion
make docker-up

# First ingest
make ingest
# Note the chunk count from logs

# Second ingest (no changes to sources)
make ingest
# Chunk count should be identical (upserts, not inserts)

# Verify in Weaviate
curl http://localhost:8080/v1/objects?class=SupportChunk | jq '.objects | length'
# Should show same count after both ingests
```

#### Benefits

1. **Idempotent ingestion:** Run `make ingest` anytime without creating duplicates
2. **Persistent source configuration:** Keep all sources in `sources.yaml` permanently
3. **Incremental updates:** Only new/changed content creates new chunks
4. **Reproducible builds:** Same sources → same UUIDs → same vector store state

#### Lesson

**Use deterministic UUIDs (`uuid5`) for content-based entities in vector stores. Random UUIDs (`uuid4`) cause duplicate insertions on re-ingestion.**

The composite key `source_url|text` ensures uniqueness while remaining deterministic.

---

## Quick Reference

### Diagnostics Checklist

Run these checks before filing an issue or debugging further:

```bash
# 1. Check environment variable conflicts
echo $EMBEDDING_MODEL
echo $FIRECRAWL_API_KEY
# If any are set, unset them: unset VAR_NAME

# 2. Verify .env file
cat .env | grep -E "(EMBEDDING_MODEL|FIRECRAWL_API_KEY|LLM_PROVIDER)"

# 3. Test config loading
uv run python -c "
from multimodal_rag.models.config import AppSettings
s = AppSettings()
print(f'Provider: {s.embedding_provider}')
print(f'Model: {s.embedding_model}')
print(f'Firecrawl: {s.firecrawl_api_key[:10]}...')
"

# 4. Check Weaviate status
curl http://localhost:8080/v1/.well-known/ready

# 5. Verify venv activation
which python
# Should show: .../multimodal-rag/.venv/bin/python

# 6. Run ingestion with debug logging
LOG_LEVEL=DEBUG uv run python -m multimodal_rag.ingest
```

### Configuration Reference

#### Environment Variables Priority (Pydantic BaseSettings)

1. **Shell env var** (highest) — `export VAR=value`
2. **`.env` file** — `VAR=value`
3. **Code default** (lowest) — `var: str = "default"`

#### Model Naming by Provider

| Provider | Variable | Format | Example |
|----------|----------|--------|---------|
| OpenRouter | `EMBEDDING_MODEL` | `provider/model` | `openai/text-embedding-3-small` |
| Ollama | `EMBEDDING_MODEL` | bare name | `nomic-embed-text-16k` |
| OpenRouter | `LLM_MODEL` | `provider/model` | `google/gemini-3-flash-preview` |
| Ollama | `LLM_MODEL` | bare name | `llama3` |

#### Ollama Embedding Parameters

| Parameter | Location | Value | Purpose |
|-----------|----------|-------|---------|
| Batch size | `store/embeddings.py` `_BATCH_SIZE` | 5 | Texts per API call |
| Text truncation | `store/embeddings.py` `_MAX_WORDS` | 800 | Words per text before embedding |
| Context window | Custom Ollama model `num_ctx` | 16384 | Total tokens per batch (5 × ~1000 tokens) |

#### Rate Limiting Parameters

| Service | Parameter | Location | Recommended Value |
|---------|-----------|----------|------------------|
| YouTube transcripts | `time.sleep()` | `ingest/__main__.py` line 47 | 2 seconds |
| Firecrawl crawls | `time.sleep()` | `ingest/__main__.py` line 59 | 5 seconds |
| Firecrawl concurrency | `max_concurrency` | `ingest/web.py` line 24 | 1 |

### Common Error Messages

| Error | Likely Cause | See Issue |
|-------|--------------|-----------|
| `AttributeError: 'FirecrawlApp' object has no attribute 'crawl_url'` | Using v1 API with v2 SDK | Issue 4 |
| `youtube_transcript_api.exceptions.NoTranscriptFound` | Video has regional English (en-GB) | Issue 3 |
| `youtube_transcript_api.exceptions.IpBlocked` | Rate limiting / IP ban | Issue 5 |
| `UnauthorizedError: Invalid token` (Firecrawl) | Bad API key | Issue 7 |
| `UnauthorizedError: Concurrency limit exceeded` | Free tier limit | Issue 6 |
| `KeyboardInterrupt` during import | Python 3.12.12 typing regression | Issue 8 |
| `ollama._types.ResponseError: input length exceeds context length` | Ollama batch context overflow | Issue 9 |
| Embedding model name wrong | Shell env var override | Issue 1 |
| `parse error near '-m'` during venv activate | Python alias with control flow | Issue 2 |
| Duplicate chunks on re-ingest | Random UUIDs (uuid4) | Issue 10 |

### Testing After Changes

```bash
# 1. Quality checks
make quality

# 2. Run unit tests
make test

# 3. Test YouTube ingestion (single video)
uv run python -c "
from multimodal_rag.ingest.youtube import fetch_transcript_chunks
chunks = fetch_transcript_chunks(
    'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    'Test Video',
    target_tokens=400
)
print(f'Fetched {len(chunks)} chunks')
"

# 4. Test Firecrawl (requires valid API key)
uv run python -c "
from multimodal_rag.ingest.web import fetch_web_chunks
from multimodal_rag.models.config import AppSettings
s = AppSettings()
chunks = fetch_web_chunks(
    'https://example.com',
    'Test KB',
    s.firecrawl_api_key,
    target_tokens=400
)
print(f'Fetched {len(chunks)} chunks')
"

# 5. Full ingestion pipeline
make ingest
```

### File Locations

| File | Purpose |
|------|---------|
| `src/multimodal_rag/models/config.py` | Pydantic settings (reads `.env`) |
| `src/multimodal_rag/ingest/__main__.py` | CLI orchestrator with rate limiting |
| `src/multimodal_rag/ingest/youtube.py` | YouTube transcript fetching |
| `src/multimodal_rag/ingest/web.py` | Firecrawl web crawling |
| `config/sources.yaml` | Source URLs (gitignored) |
| `.env` | Environment config (gitignored) |
| `tests/test_web_ingest.py` | Web ingestion tests with v2 API mocks |

### Shell Configuration Issues

Check these files if experiencing environment variable problems:

```bash
# Check for dotenv plugin
grep "plugins=" ~/.zshrc | grep dotenv

# Check for manual .env sourcing
grep "source.*\.env" ~/.zshrc

# Check python aliases
grep "alias python" ~/.zshrc
grep "^python()" ~/.zshrc
```

---

## Document Maintenance

**Last updated:** 2026-02-16
**Author:** Sytse van der Schaaf

To regenerate this document after fixing new issues:
1. Document the symptom, root cause, and fix in this file
2. Add to the [Quick Reference](#quick-reference) tables
3. Update the git history with the fix commit SHA

To report a new issue:
1. Run the [Diagnostics Checklist](#diagnostics-checklist)
2. Include full error messages and stack traces
3. Note the environment (Python version, OS, package versions)
4. Document resolution steps taken
