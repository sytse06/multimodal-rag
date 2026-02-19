# Bug Report: Ingestion Pipeline — Silent Failures and Missing Log Persistence

**Date:** 2026-02-18
**Project:** multimodal-rag
**Pipeline:** `src/multimodal_rag/ingest/`
**Trigger:** Run reported `INFO Ingestion complete: 402 added, 2 failed, 402 total in store`

---

## Background

A full ingestion run completed with 2 reported failures. Investigation of the pipeline code identified three bugs that collectively mean: (1) the "402 added" count may be overstated due to silent Weaviate batch rejections, (2) the "2 failed" count understates actual YouTube failures because transcript errors are swallowed before they can reach the failure counter, and (3) the details of which sources failed and why are permanently lost because the run's stdout was not captured. The 2 counted failures must have originated in the web pipeline — either a full KB crawl raising an exception or individual page processing failing across the 3 configured KB sources (`hydrosym.paro.nl`, `hydroman.paro.nl`, `hydrocam.paro.nl`) — but there is no way to determine which without re-running.

These bugs need to be fixed before the next production run.

---

## Bug 1 — Added counter incremented before Weaviate batch flushes

**File:** `src/multimodal_rag/store/weaviate.py`, `add_chunks()` method
**Lines:** 83–102

### What

In `add_chunks()`, the `added` counter is incremented inside the `with collection.batch.dynamic() as batch:` context manager, once per `batch.add_object()` call:

```python
# weaviate.py, lines 83–102
with collection.batch.dynamic() as batch:
    for chunk, vector in zip(chunks, vectors):
        ...
        batch.add_object(
            properties=props,
            vector=vector,
            uuid=chunk.chunk_id,
        )
        added += 1  # ← line 100: counted here, BEFORE the batch is flushed
```

The batch is not sent to Weaviate until the `with` block exits. If Weaviate rejects any objects (schema mismatch, invalid vector dimension, connection drop during flush, etc.), the rejections are available on `batch.failed_objects` after the context manager — but that attribute is never read. The function returns `added` based on call count, not actual successful insertions.

### Why it matters

The reported "402 added" figure may be higher than the real number of objects in Weaviate. The count is meaningless as a reliability signal. `store.count()` is called at the end of `run()` and does reflect the true Weaviate state, but per-source added counts in the logs are wrong whenever Weaviate rejects objects silently.

### Fix needed

After the `with` block closes, check `batch.failed_objects`. Subtract the count from `added` and emit a `WARNING` log per failed object (including the object UUID and the error message from Weaviate). Return the corrected count.

```python
# Sketch of the fix
with collection.batch.dynamic() as batch:
    for chunk, vector in zip(chunks, vectors):
        ...
        batch.add_object(...)
        added += 1

# After flush:
if batch.failed_objects:
    for failed in batch.failed_objects:
        logger.warning(
            "Weaviate rejected object %s: %s",
            failed.original_uuid,
            failed.error,
        )
    added -= len(batch.failed_objects)
```

---

## Bug 2 — YouTube failure counter never triggered when transcript is unavailable

**Files:** `src/multimodal_rag/ingest/youtube.py` (lines 111–120), `src/multimodal_rag/ingest/__main__.py` (lines 73–83)

### What

`fetch_transcript_chunks()` catches all transcript fetch failures internally and returns an empty list on any error path:

```python
# youtube.py, lines 111–120
video_id = extract_video_id(video_url)
if not video_id:
    logger.error("Could not extract video ID from URL: %s", video_url)
    return []          # ← failure swallowed, returns []

try:
    segments = fetch_transcript(video_id)
except Exception:
    logger.exception("Failed to fetch transcript for %s", video_url)
    return []          # ← failure swallowed, returns []
```

Back in `__main__.py`, the caller's `except Exception` block — the one that increments `total_failed` — can only be triggered if `SupportChunk.from_transcript_chunk(c)` raises, which is a Pydantic validation failure on a correctly typed model and is effectively unreachable under normal conditions:

```python
# __main__.py, lines 73–83
try:
    tc = fetch_transcript_chunks(...)     # never raises; returns [] on failure
    chunks = [SupportChunk.from_transcript_chunk(c) for c in tc]
    total_added += _ingest_chunks(store, chunks, label)
except Exception:
    logger.exception("[%s] Failed, skipping", label)
    total_failed += 1                     # ← unreachable for YouTube failures
```

A YouTube video with disabled captions, a private video, or a video with no English transcript produces 0 chunks, logs a WARNING, and contributes 0 to `total_added` — but `total_failed` is never incremented.

### Why it matters

The failure counter gives a false sense of which sources are healthy. A channel with 10 videos, all with captions disabled, reports 0 failures while silently contributing nothing to the knowledge base. There is no way to distinguish "this video has no chunks because the transcript was empty" from "this video has no chunks because it was never configured" by looking at the summary log line.

### Fix needed

Two acceptable approaches:

**Option A — Raise from `fetch_transcript_chunks` on transcript unavailability.** Remove the internal `except` that returns `[]` and let the exception propagate. The caller's `except Exception` block in `__main__.py` then catches it and increments `total_failed` correctly. The `extract_video_id` failure (invalid URL) should also raise rather than return `[]`.

**Option B — Detect the empty-return case in `__main__.py`.** After calling `fetch_transcript_chunks`, check whether both the returned list is empty and no exception was raised, and increment `total_failed` explicitly. This requires inspecting internal state that is currently invisible to the caller and is the weaker approach.

Option A is cleaner. It restores the contract that `fetch_transcript_chunks` either returns usable chunks or raises.

---

## Bug 3 — No persistent log file; failure details lost after each run

**File:** `src/multimodal_rag/ingest/__main__.py`, `run()` function
**Lines:** 50–53

### What

`run()` configures logging with `basicConfig`, which writes to stdout only:

```python
# __main__.py, lines 50–53
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
```

There is no file handler. Once the terminal session ends or the output scrolls past, all `WARNING` and `ERROR` log lines — including the exception tracebacks for whichever sources failed — are gone.

### Why it matters

This is why the 2 failures from the most recent run cannot be investigated without re-running. The specific KB source that failed (crawl error vs. page processing error), the HTTP status code, the exception type, and the URL are all in the logs that no longer exist. Every run is effectively a write-once, read-never event unless the operator manually redirects stdout.

### Fix needed

Add a `FileHandler` to the logging setup that writes to `logs/ingest.log` (appending, not overwriting, so history accumulates). The `logs/` directory should be created if it does not exist. Either hard-code the path or add a `--log-file` CLI argument if configurability is preferred later.

```python
# Sketch of the fix
import sys
from pathlib import Path

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "ingest.log"

file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(file_handler.formatter)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    handlers=[stream_handler, file_handler],
)
```

Add `logs/` to `.gitignore`.

---

## Next Steps

Fix in this order:

1. **Bug 3 — Add log file persistence** (`__main__.py`). This must be done first. Without it, re-running to reproduce the 2 failures produces new logs that will also be lost if not captured. Fix this, then re-run ingestion, then read `logs/ingest.log` to identify which KB sources failed and why.

2. **Bug 1 — Check `batch.failed_objects` after flush** (`weaviate.py`). Once log persistence is in place, any Weaviate batch rejections will be visible in the log file. Fix the counter and the warning logging so the reported "added" count reflects reality.

3. **Bug 2 — Surface YouTube transcript failures to the failure counter** (`youtube.py`, `__main__.py`). Least urgent since YouTube ingestion appeared to work (402 chunks produced), but the counter is structurally broken and will mislead on future runs with unavailable transcripts.

All three fixes are small and self-contained. None require changes to the data model, Weaviate schema, or test fixtures.

---

**Last updated:** 2026-02-18
