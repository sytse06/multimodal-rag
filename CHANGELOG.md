# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com);
versions follow `docs/version-split.md` (the tag, `pyproject.toml` and the release commit
must match). Stories are described in `docs/epics.md`; this file says **when and where each
one reached users**.

Two kinds of change reach colleagues, and they are recorded separately:

| Kind | Reaches users by | Recorded as |
|---|---|---|
| Code (app, query, generator) | `make deploy` | A version and a Cloud Run revision |
| Content (chunks in hosted Weaviate) | `make promote` | A content promotion, with its backup name |

Most of Epic 11 is tooling or content: it changes what a maintainer can do, not what the app
does. Only a deploy changes the app.

## [Unreleased]

Everything since `v0.2.0` (2026-09-30). Nothing here is tagged: the work was deployed as
Cloud Run revisions 00015 to 00025 without a version bump (see "Deployed revisions" below).
The next release, **v0.3.0**, will tag this together with whatever is deployed next, following
`docs/version-split.md` (bump `pyproject.toml` on `main`, annotated tag, push both).

### Added

- **Query:** answers reproduce full procedures, and the context budget is larger (`d8b6ec5`).
- **Observability, OBS-001 to OBS-003 (Epic 11, Phase 1):** shared logging with JSON output and
  correlation IDs; one structured metrics record per generation; secrets redacted and the
  user's query text no longer logged.
- **Hosted sync, SYNC-000 to SYNC-004, SYNC-007, SYNC-008 (Epic 11, Phase 2):**
  - additive schema properties and the `document_id` / `content_hash` / `source_lastmod`
    fields on every chunk (SYNC-000, SYNC-001)
  - the source registry in the state bucket with `make sources-pull` / `sources-push`
  - `make sync-preflight`, the retrieval gate and canary questions (`make sync-gate`)
  - the sitemap inventory, page ledger and hashed sync plan (SYNC-002), applying a reviewed
    plan to local (SYNC-003), and removals that a person must name (SYNC-004)
  - `make promote` with a canary write, backups before and after, verification and rollback,
    plus a rehearsal copy of hosted (SYNC-007)
  - the pinned embedding runtime (Ollama 0.40.14 on port 11435) and a check that stored
    vectors match a fresh embedding
  - the maintainer runbook, `docs/maintainer-runbook.md` (SYNC-008)
- **Video lane, SYNC-011:** `make sync-videos` ingests only videos with no stored chunks; the
  transcript is the manual captions, else a Gemini transcription (Voxtral is demoted).
- **Video frame descriptions, SYNC-012 part 1:** a plain-text description of at most 100
  words, and two chunks per 30-second window (`[Transcript]` and `[Visual]`); an opt-in
  Gemini vision path (`VISION_MODEL=gemini:<model>`).
- `make weaviate-compat` tests our code against a given Weaviate version.

### Changed

- Gemini and OpenAI model defaults updated; temperature skipped for `gpt-6` (`c4109c5`).
- Cloud Run image pins Ollama to 0.40.14 and downloads it from GitHub releases (`393c9bb`).
- Local Weaviate aligned with the hosted 1.39.13 and the client floor raised (`8a2c232`).
- The registry in the bucket is the source of truth; `config/sources.yaml` is an editing
  buffer (decided 2026-11-24, documented in the runbook and SYNC-009).
- The app warns at startup when the default model is not in the registry (`a9783ad`).
- Writes to the hosted collection are refused unless a promotion passes the flag (`1cbdecc`).

### Fixed

- Gemini token counts: usage is summed across streamed chunks and all-zero chunks are
  ignored (`3f40d15`, `f81ddc0`); OpenAI streamed usage is requested explicitly (`3e164c8`).
- google-genai function-calling notices dropped from the logs (`d971abc`).
- Real environment variables override `.env` for flat settings (`f82c90c`); `VISION_MODEL`
  is read from `.env` (`9d7056f`).
- A promotion treats an unset property as absent when verifying (`721d920`).
- yt-dlp upgraded to 2026.10.26 (`7dfb203`).

### Planned

- Epic 12, answer and source quality calibration (QUAL-001 to QUAL-007): not started.
- SYNC-013, SYNC-009 and the OPS tickets: see `docs/epics.md`.

## Deployed revisions since v0.2.0

Cloud Run service `multimodal-rag`, region `europe-west4`. **The commit is inferred:** it is the
last non-merge commit on `main` (documentation commits included) before the revision was created, because `make deploy` builds
from the working tree (`--source .`) and the exact tree is not recorded. Treat it as the
nearest commit, not proof. "Image" says whether the container image changed from the revision
before it; the same image means a configuration-only revision.

| Revision | Created (UTC) | Nearest commit | Image | What it carried |
|---|---|---|---|---|
| 00012 | 2026-09-30 08:58 | `cfa58b1`, before the `v0.2.0` bump `5a3f733` | n/a | The `v0.2.0` deploy (Epic 10). It was created about 30 minutes before the tagged commit, so it may not be exactly the tagged tree |
| 00013, 00014 | 2026-11-03, 11-05 08:31 | none | same as 00012 | Configuration only |
| 00015 | 2026-11-05 09:20 | `c4109c5` | new | Model defaults |
| 00016 | 2026-11-05 09:34 | `393c9bb` | new | Ollama pin, trimmed source upload |
| 00017 | 2026-11-05 10:56 | `d8b6ec5` | new | Full procedures in answers, larger context budget |
| 00018 | 2026-11-09 10:06 | `7c0eb5a` | new | OBS-001, OBS-002 (logging, metrics) |
| 00019 | 2026-11-09 10:41 | `5769ae5` (last app change `1b5a82a`) | new | OBS-003 (redaction, no query text), usage fix |
| 00020 | 2026-11-09 11:09 | `d971abc` | new | Usage summed across chunks, function-calling log noise removed |
| 00021 | 2026-11-13 09:01 | `3e164c8` | new | OpenAI streamed usage; Phase 1 verified against live traffic |
| 00022, 00023 | 2026-11-13 09:40 | none | same as 00021 | Configuration only |
| 00024 | 2026-11-13 13:28 | `81a4140` | new | SYNC-000 (additive schema properties), cookies from Secret Manager |
| 00025 | 2026-11-17 13:56 | `45ff43c` | new | SYNC-001 (stored identity), hosted-write guard, startup model warning. **Current** |

No query-side commit has landed since 00025, so the live app and `main` agree on how answers
are produced. Everything after `45ff43c` is ingest and tooling (SYNC-011, SYNC-012 part 1, the
runbook) plus documentation.

The deployed `LLM_TEMPERATURE` is `1` on Cloud Run; local defaults to `0.3` (see QUAL-002).

## Content promotions

Hosted Weaviate content, newest first. Restore only from a backup named here or later: every
older backup holds the March vectors (see `docs/maintainer-runbook.md`, section 0).

| Date | Change | Hosted chunks after | Backup after |
|---|---|---|---|
| 2026-11-24 | First new video `yWuBQik8KPs` (SYNC-011/012): 8 chunks, 4 windows of narration plus description. Canary wrote 1, then 7 | 951 | `snapshots/20261124T111609Z-promote-after` |
| 2026-11-17 | SYNC-001: stored identity properties added to every chunk | 943 | `snapshots/20261117T134901Z-promote-after` |
| 2026-11-17 | SYNC-007: hosted made equal to local after SYNC-003/004 (pages replaced, article 437 removed); every vector re-embedded on the pinned runtime | 943 | `snapshots/20261117T125431Z-promote-after` |
| 2026-03 | Initial ingest of the YouTube transcripts and knowledge bases (922 objects at the 2026-11-13 backup) | 922 | Older backups hold the March vectors; do not restore from them |

## [0.2.0] - 2026-09-30

- Inference app deployed to Google Cloud Run (Epic 10, DATA-007).
- Branch and release versioning documented (`docs/version-split.md`).

## [0.1.2] - 2026-09-25

- Hosted storage and runtime validation (DATA-004, DATA-005); restore of fixed Weaviate
  snapshots (DATA-003); local and hosted Weaviate deployment documented (DATA-006); hosted
  embedding deployment defined (DATA-007).

## [0.1.1] - 2026-09-24

- First tagged release. Weaviate cloud configuration and snapshot export (Epic 9) landed
  around it; earlier work (Epics 1 to 8) is in the history before it.
