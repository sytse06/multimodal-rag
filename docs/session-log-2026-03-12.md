# Session Log — multimodal-rag
**Date:** 2026-03-12
**Period:** 2026-03-10 to 2026-03-12
**Branch flow:** `feature/youtube-cookies-ingest-fixes` → `develop` → `main`, `feature/transcribe-mode-purge-source` → `develop` → `main`, `feature/ingest-report` → `develop` → `main`
**Commits (session):** 3 feature commits + 6 merge commits

---

## Summary

This session was driven by two runs of `make ingest` that exposed multiple blocking failures in the YouTube and video frame extraction pipelines. The root causes were a YouTube IP block, a stale yt-dlp version that could not solve YouTube's n-challenge and therefore had no usable video-only format, a wrong OpenRouter vision model ID, and silent screen recording videos that were being described visually instead of having their on-screen text transcribed. All four were resolved. Two operational tools were also added: timestamped log files per ingest run (`tee` in `make ingest`) and `make ingest-report` to grep the latest log for failures. A per-source Weaviate purge (`make purge-source URL=...`) rounds out the session. The vector store grew from 408 chunks to 774 by end of session.

---

## Git Activity

```
3f408fb 2026-03-12  Merge develop into main
6b30df6 2026-03-12  Merge feature/ingest-report into develop
6bd036a 2026-03-12  chore: add make ingest-report for failure summary from latest log
2d8d210 2026-03-10  Merge develop into main
eecad8b 2026-03-10  Merge feature/transcribe-mode-purge-source into develop
a30e185 2026-03-10  feat(ingest): transcribe mode for silent screen recordings + per-source purge
87bbf71 2026-03-10  Merge develop into main
5db61d5 2026-03-10  Merge feature/youtube-cookies-ingest-fixes into develop
ef38427 2026-03-10  fix(ingest): bypass YouTube IP blocks and n-challenge format errors
```

---

## Problems Diagnosed and Fixed

### Fix 1 — OpenRouter 404 on vision model (`google/gemini-flash-1.5`)

**Component:** `src/multimodal_rag/models/config.py`, `.env`

OpenRouter returned HTTP 404 when the vision LLM was invoked. The configured model ID was `google/gemini-flash-1.5`. OpenRouter's naming convention is `provider/model-version` with the version suffix appended to the model name, making the correct ID `google/gemini-1.5-flash`. One character transposition in the ordering of the version number caused every vision frame description call to fail.

**Resolution:** Updated the model ID in `.env` (and any development config) to `google/gemini-1.5-flash`. No code change required — `VISION_MODEL` is read directly from the environment by `AppSettings`.

---

### Fix 2 — YouTube IP block on transcript fetching (`IpBlocked`)

**Component:** `src/multimodal_rag/models/config.py`, `src/multimodal_rag/ingest/youtube.py`

`youtube-transcript-api` raised `IpBlocked` (HTTP 429) for all videos. The 2-second inter-request sleep added in a prior session was insufficient to prevent an IP ban during a full re-ingest of ~40 videos. The library provides no built-in mechanism for cookie-based authentication to bypass rate limiting.

**Resolution:** Three-part fix.

1. Added `youtube_cookies_file: str = ""` to `AppSettings` in `config.py`. When non-empty, the setting points to a Netscape-format cookies file exported from a browser session that is authenticated to YouTube.

2. Added `_build_http_client()` to `youtube.py`. This function loads the cookies file via `http.cookiejar.MozillaCookieJar`, injects the cookies into a `requests.Session`, and returns the session. If `cookies_file` is empty it returns `None`.

3. Threaded `cookies_file` through `fetch_transcript()` and `fetch_transcript_chunks()`. `YouTubeTranscriptApi` is now instantiated with `http_client=session` when a cookies file is present, causing all transcript requests to carry the authenticated session cookies.

The cookies file path is set via `YOUTUBE_COOKIES_FILE` in `.env`. An empty value keeps prior behaviour (no authentication, rate-limited access only).

**Relevant code — `_build_http_client` in `youtube.py`:**
```python
def _build_http_client(cookies_file: str) -> requests.Session | None:
    if not cookies_file:
        return None
    jar = http.cookiejar.MozillaCookieJar(cookies_file)
    jar.load(ignore_discard=True, ignore_expires=True)
    session = requests.Session()
    session.cookies.update(jar)
    return session
```

---

### Fix 3 — yt-dlp format errors on video download (`DownloadError`)

**Component:** `src/multimodal_rag/ingest/video_frames.py`, `pyproject.toml`, `uv.lock`

`fetch_frame_chunks` failed with "Requested format is not available" during the `download_video` step. Two compounding problems caused this.

**Problem A — Stale yt-dlp version.** yt-dlp 2026.02.04 (the pinned version) was unable to solve YouTube's n-challenge, which is required to decrypt the signature of video-only streams. Without a valid solution, YouTube served only the muxed format 18 (360p), which is a combined audio+video container. The prior format selector `bestvideo[ext=mp4]/bestvideo` matched only video-only streams and found nothing when they were locked behind the unsolved n-challenge.

**Problem B — Storyboard stream pollution.** YouTube serves storyboard preview streams (MIME type `video/webm; codecs="vp9"` with `vcodec=images`) that yt-dlp enumerates as available video formats. These are sequences of thumbnail tiles, not real video. The original format selector could accidentally resolve to a storyboard stream, which then failed to yield usable frames during ffmpeg extraction.

**Resolution:** Two changes.

1. Updated `yt-dlp` version constraint in `pyproject.toml` to `>=2026.3.1`. yt-dlp 2026.03.03 includes an updated n-challenge solver that handles the current YouTube format without requiring the deno runtime.

2. Updated the format selector in `download_video` to `"bestvideo[vcodec!=images]/best[vcodec!=images]/best"`. This preference order is: best video-only stream that is not a storyboard, then best muxed stream that is not a storyboard, then any best stream as a last resort. The muxed fallback ensures a download succeeds even when video-only streams remain unavailable.

**Relevant code — `ydl_opts` in `video_frames.py`:**
```python
ydl_opts: dict = {
    "format": "bestvideo[vcodec!=images]/best[vcodec!=images]/best",
    "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
    "quiet": True,
    "no_warnings": True,
}
```

`cookies_file` is also passed to `ydl_opts["cookiefile"]` when set, so video downloads use the same authenticated session as transcript fetching.

---

### Fix 4 — Silent screen recording videos producing no useful content

**Component:** `src/multimodal_rag/ingest/video_frames.py`, `src/multimodal_rag/ingest/__main__.py`

Videos marked `skip_voxtral: true` in `sources.yaml` are screen recordings with no audio. They were already skipping the Voxtral transcription path because there is no spoken content to transcribe. When the vision LLM was enabled, these videos fell through to `fetch_frame_chunks`, which called `describe_frame` with the generic visual description prompt ("Describe what is shown in this video frame in detail..."). The resulting chunks described UI chrome, colour schemes, and mouse positions — not the on-screen text and step instructions that make screen recordings useful for a support knowledge base.

A second problem was frame interval: 30 seconds per frame is appropriate for narrated tutorial videos where each segment covers a meaningful topic. Screen recordings typically show step-by-step UI interactions, where 30 seconds can skip over multiple distinct steps.

**Resolution:**

1. Added `transcribe_mode: bool = False` parameter to both `describe_frame` and `fetch_frame_chunks`.

2. Added `_PROMPT_TRANSCRIBE` constant in `video_frames.py`. When `transcribe_mode=True`, `describe_frame` uses this prompt instead of `_PROMPT_DESCRIBE`:
   ```
   "Transcribe all text visible in this video frame exactly as written.
   Include any on-screen captions, labels, step descriptions, callouts,
   or text overlays.
   If no text is visible, respond with 'No text visible.'"
   ```

3. In `__main__.py`, wired the flag through from `sources.yaml`:
   ```python
   frame_chunks = fetch_frame_chunks(
       str(yt.url),
       yt.name,
       vision_llm,
       interval_seconds=5 if yt.skip_voxtral else 30,
       cookies_file=settings.youtube_cookies_file,
       transcribe_mode=yt.skip_voxtral,
   )
   ```
   When `skip_voxtral=True`: frame interval is 5 seconds (instead of 30), and transcribe mode is active. When `skip_voxtral=False`: standard 30-second interval and visual description prompt.

---

### Recurring issue identified — 7 HydroMan tutorial videos unavailable

These video IDs consistently fail with "This video is not available" regardless of cookies:

`9vlY7cY3om0`, `u62q7SaznFI`, `Cd23zL0SHM4`, `A2a_9cv6CrY`, `8SzYjz20O_M`, `U5b3_e9U280`, `Pvg8F7xPm_o`

The error is not an IP block or format issue — it is a privacy/availability error. These videos are likely private or unlisted. They cannot be ingested without credentials from the YouTube creator's account. Resolution options: obtain creator-level cookies, or remove the entries from `config/sources.yaml`.

---

## New Features Added

### Feature 1 — Timestamped log files for `make ingest`

**Component:** `Makefile`, `.gitignore`

Before this session, every `make ingest` run wrote only to stdout. Once the terminal scrolled or the session closed, all failure details were permanently lost. (This was documented as Bug 3 in `docs/ingestion-bug-report.md`, filed 2026-02-18.)

`make ingest` now pipes all output through `tee` to a timestamped file:
```makefile
@uv run python -m multimodal_rag.ingest 2>&1 | tee logs/ingest-$(date +%Y%m%d-%H%M%S).log
```

The `logs/` directory is created automatically by `mkdir -p logs` before the pipeline starts. `logs/` is added to `.gitignore` so log files are never committed. Each run produces a new file, e.g. `logs/ingest-20260310-121354.log`, preserving history across runs.

---

### Feature 2 — `make ingest-report`

**Component:** `Makefile`

A fast post-ingest diagnostic that greps the most recent log file for signals of interest: `ERROR`, `failed, skipping`, `No chunks produced`, `IP blocked`, `not available`, and the final `Ingestion complete` summary line. Output is grouped into two sections: failures/warnings and summary.

Usage:
```bash
make ingest-report
```

The target errors with a clear message if `logs/` is empty. It picks the latest log by modification time (`ls -t logs/ingest-*.log | head -1`), so it always reports on the most recent run without needing a path argument.

This closes the investigation gap that made the two failures from the 2026-02-18 run impossible to diagnose after the fact.

---

### Feature 3 — `make purge-source URL=...`

**Component:** `Makefile`, `src/multimodal_rag/store/weaviate.py`

Previously the only way to remove content from the vector store was `make purge`, which deletes the entire Weaviate collection. Re-ingesting a single source that produced incorrect chunks (wrong format selector, old model, bad cookies) required a full purge and full re-ingest of all sources.

`WeaviateStore.delete_by_source()` was added to `weaviate.py`:
```python
def delete_by_source(self, source_url: str) -> int:
    collection = self._client.collections.get(COLLECTION_NAME)
    result = collection.data.delete_many(
        where=Filter.by_property("source_url").equal(source_url)
    )
    deleted = result.successful if result else 0
    logger.info("Deleted %d chunks for source: %s", deleted, source_url)
    return deleted
```

The `make purge-source` Makefile target wraps this method with a URL argument, a confirmation prompt, and error handling for missing `URL`:
```bash
make purge-source URL=https://www.youtube.com/watch?v=VIDEOID
```

This enables surgical re-ingestion: purge one source, re-run `make ingest`, without touching the rest of the vector store.

---

### Feature 4 — `make test-integration` + integration test isolation

**Component:** `Makefile`, `pyproject.toml`

The `test_hydrosym_quickstart` integration test in `tests/test_ingest_cli.py` hits YouTube's transcript API over the network. After the IP block, this test failed on every `make test` run in CI and locally. The test is correct — it validates end-to-end ingestion from a real video — but it should not be part of the default test gate.

`make test` now passes `-m "not integration"` to pytest:
```makefile
@uv run pytest tests/ -v --cov=src/multimodal_rag -m "not integration"
```

A separate `make test-integration` target runs only the integration tests:
```makefile
@uv run pytest tests/ -v -m "integration"
```

The `integration` marker is declared in `pyproject.toml` under `[tool.pytest.ini_options]`:
```toml
markers = ["integration: tests requiring network access"]
```

---

## Ingestion Progress

| Run | Chunks Added | Sources Failed | Total in Store |
|-----|-------------|----------------|----------------|
| Before session | — | many | 408 |
| Mid-session (after cookies fix, first re-run) | 244 | 6 | 637 |
| End of session (after format selector + yt-dlp bump) | 263 | 7 | 774 |

The 7 persistent failures are the unavailable HydroMan videos identified above. All other configured sources ingested successfully by end of session.

---

## Files Changed

| File | Change | Commit |
|------|--------|--------|
| `src/multimodal_rag/models/config.py` | Added `youtube_cookies_file: str = ""` | `ef38427` |
| `src/multimodal_rag/ingest/youtube.py` | Added `_build_http_client()`, threaded `cookies_file` through `fetch_transcript` and `fetch_transcript_chunks` | `ef38427` |
| `src/multimodal_rag/ingest/video_frames.py` | Added `transcribe_mode` parameter and `_PROMPT_TRANSCRIBE`, updated format selector to skip storyboard streams, added `cookies_file` to `download_video` | `ef38427`, `a30e185` |
| `src/multimodal_rag/ingest/__main__.py` | Wired `cookies_file`, `transcribe_mode`, `interval_seconds=5` for `skip_voxtral` sources | `ef38427`, `a30e185` |
| `src/multimodal_rag/store/weaviate.py` | Added `delete_by_source()` | `a30e185` |
| `Makefile` | Added `make test-integration`, `make ingest-report`, `make purge-source`; added `tee` log output to `make ingest` | `ef38427`, `a30e185`, `6bd036a` |
| `pyproject.toml` | Added `types-requests` dev dependency, bumped `yt-dlp` to `>=2026.3.1` | `ef38427` |
| `uv.lock` | Updated for yt-dlp and types-requests | `ef38427` |
| `.gitignore` | Added `logs/` | `ef38427` |

**Session diff stats (feature commits only):** +149 lines / -35 lines across 10 files

---

## Current State

### Test Suite

```
142 passed, 2 deselected (integration tests) in 4.18s
```

All unit tests pass. Integration tests are excluded from `make test` and can be run separately with `make test-integration` when network access and a valid cookies file are available.

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_app.py` | 5 | Pass |
| `test_generator.py` | 9 | Pass |
| `test_ingest_cli.py` | 9 | Pass (integration tests excluded) |
| `test_llm.py` | 4 | Pass |
| `test_models.py` | 32 | Pass |
| `test_retriever.py` | 11 | Pass |
| `test_store.py` | 8 | Pass |
| `test_video_frames.py` | 10 | Pass |
| `test_voxtral_ingest.py` | 13 | Pass |
| `test_web_images.py` | 13 | Pass |
| `test_web_ingest.py` | 14 | Pass |
| `test_youtube_ingest.py` | 13 | Pass |
| `test_placeholder.py` | 1 | Pass |

### Configuration Reference (updated this session)

| Variable | Purpose | Default |
|----------|---------|---------|
| `YOUTUBE_COOKIES_FILE` | Path to Netscape-format cookies file for authenticated YouTube requests | `""` (disabled) |
| `VISION_MODEL` | Vision LLM model ID. Must use correct OpenRouter format: `google/gemini-1.5-flash` | `""` (disabled) |

---

## Open Items

| Item | Priority | Notes |
|------|----------|-------|
| 7 unavailable HydroMan videos | Medium | Private/unlisted. Either obtain creator cookies or remove from `sources.yaml`. |
| `batch.failed_objects` not checked in `WeaviateStore.add_chunks` | Low | Documented in `docs/ingestion-bug-report.md` as Bug 1 (filed 2026-02-18). The `total_added` counter may overcount if Weaviate rejects objects silently during batch flush. |
| YouTube failure counter not triggered on transcript unavailability | Low | Documented in `docs/ingestion-bug-report.md` as Bug 2 (filed 2026-02-18). `fetch_transcript_chunks` returns `[]` on failure; the caller's `total_failed` counter is unreachable for transcript errors. |

---

## Related Documents

- `docs/debugging-ingestion.md` — Persistent log of all ingestion pipeline issues. Issues 5 (IP blocks) and related yt-dlp issues from this session should be appended there.
- `docs/ingestion-bug-report.md` — Bug 3 (no log file persistence) is now closed by the `tee` change in `make ingest`. Bugs 1 and 2 remain open.

---

**Last updated:** 2026-03-12
