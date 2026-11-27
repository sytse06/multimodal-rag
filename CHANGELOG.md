# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/); the routine is `docs/version-split.md`
and the `/release` skill (`.claude/skills/release.md`).

Stories are described in `docs/epics.md`; this file says when each one shipped. Two kinds of
change reach colleagues, and the release entries below cover the first. Both are also listed
in ledgers at the end of this file:

| Kind | Reaches users by | Recorded as |
|---|---|---|
| Code (app, query, generator) | `make deploy` | A release entry, plus a row in "Cloud Run revisions" |
| Content (chunks in hosted Weaviate) | `make promote` | A row in "Content promotions" |

---

## [0.3.0] — unreleased

*Draft written at the backfill (2026-11-24). `/release 0.3.0` finalises it: date, test count,
and whatever else is deployed first.*

v0.3.0 is the first release since v0.2.0 (2026-09-30) and records nine Cloud Run revisions with new code (and four configuration-only ones) that were deployed from 2026-11-05 to 2026-11-17 without a version bump. For the support colleague it makes answers reproduce the full procedure from the sources, and it makes the live service observable: one structured record per generation with token counts, secrets redacted and the user's question never logged (OBS-001 to OBS-003). For the maintainer it adds the whole hosted-sync toolchain of Epic 11: a source registry in the state bucket, a preflight, a retrieval gate, a page ledger, plan and apply, removals a person must name, and a promotion from local to hosted with a canary write, backups and rollback (SYNC-000 to SYNC-004, SYNC-007, SYNC-008). Hosted chunks now carry a stored identity (SYNC-001). The video lane (`make sync-videos`) and the new frame descriptions (SYNC-011, SYNC-012 part 1) are tooling and are not part of any deployed revision yet. All N tests pass.

### Added

- **OBS-001, OBS-002, OBS-003** — shared logging with JSON output and correlation IDs; one structured metrics record per generation; secrets redacted and the question text no longer logged.
- **SYNC-000, SYNC-001** — additive schema properties and `document_id`, `content_hash`, `source_lastmod` and video links on every chunk; writes to the hosted collection are refused unless a promotion passes the flag.
- **SYNC-002 to SYNC-004** — the sitemap inventory, page ledger and hashed sync plan; applying a reviewed plan to local; removals only when a person names them.
- **SYNC-007** — `make promote` with a canary write, verification, backups before and after, and rollback; a rehearsal copy of hosted.
- **SYNC-008** — `make sync-preflight`, the retrieval gate and canary questions (`make sync-gate`), the pinned embedding runtime check, and the maintainer runbook (`docs/maintainer-runbook.md`).
- **SYNC-013** — `make sync-split-videos` cuts the 213 old hybrid video chunks down to their narration under the same ids (the descriptions are left out); applied to local 2026-11-27, promotion pending.
- **SYNC-011** — `make sync-videos` ingests only videos with no stored chunks; the transcript comes from manual captions, else Gemini (Voxtral is demoted).
- **Voxtral removed** — `voxtral.py`, the `mistralai` dependency, the `MISTRAL_API_KEY` setting and the Voxtral branch of `fetch_transcript_chunks` are deleted; a stray `MISTRAL_API_KEY` in `.env` is ignored. The registry flag `skip_voxtral` is now `silent` (a silent recording); the old name is still accepted, so registries in the bucket need no change.
- **SYNC-012 (part 1)** — a frame description of at most 100 words and two chunks per 30-second window (`[Transcript]`, `[Visual]`); an opt-in Gemini vision path (`VISION_MODEL=gemini:<model>`).
- `make weaviate-compat` tests our code against a given Weaviate version.
- **OPS-007** — a `hosted cluster` line in the preflight (awake, or paused with how to wake it) and the Cloud Scheduler job `weaviate-keepalive`.
- `make sources-diff`, and a preflight registry line that compares content, so unpushed local edits no longer read as "matches the bucket".
- The source registry in the state bucket: `make sources-pull` and `sources-push`.
- `CHANGELOG.md`, the `/release` skill and `make release`.

### Fixed

- Gemini token counts: usage is summed across streamed chunks and all-zero chunks are ignored (`3f40d15`, `f81ddc0`); OpenAI streamed usage is requested explicitly (`3e164c8`).
- google-genai function-calling notices dropped from the logs (`d971abc`).
- Real environment variables override `.env` for flat settings (`f82c90c`); `VISION_MODEL` is read from `.env` (`9d7056f`).
- A promotion treats an unset property as absent when verifying (`721d920`).
- yt-dlp upgraded to 2026.10.26 (`7dfb203`).

### Changed

- Answers reproduce full procedures and the context budget is larger (`d8b6ec5`).
- Gemini and OpenAI model defaults updated; temperature skipped for `gpt-6` (`c4109c5`).
- The Cloud Run image pins Ollama to 0.40.14 and downloads it from GitHub releases (`393c9bb`).
- Local Weaviate aligned with the hosted 1.39.13 and the client floor raised (`8a2c232`); on 2026-11-27 local and the rehearsal copy moved to 1.40.5, the version hosted had moved to.
- The registry in the bucket is the source of truth; `config/sources.yaml` is an editing buffer (decided 2026-11-24).
- The app warns at startup when the default model is not in the registry (`a9783ad`).
- A refused registry pull points to `make sources-diff` and `make sources-push` instead of `FORCE=1`.

## [0.2.0] — 2026-09-30

v0.2.0 deploys the inference app to Google Cloud Run (Epic 10, DATA-007) and documents branch and release versioning.

## [0.1.2] — 2026-09-25

v0.1.2 completes hosted storage and runtime validation (DATA-004, DATA-005), restores fixed Weaviate snapshots (DATA-003), documents local and hosted Weaviate deployment (DATA-006) and defines the hosted embedding deployment (DATA-007).

## [0.1.1] — 2026-09-24

v0.1.1 is the first tagged release. Weaviate cloud configuration and snapshot export (Epic 9) landed around it; earlier work (Epics 1 to 8) is in the history before it.

---

## Cloud Run revisions

Cloud Run service `multimodal-rag`, region `europe-west4`. Revisions 00013 to 00025 were deployed without a release, so **their commit is inferred:** it is the
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
| 2026-11-24 | Web content sync of the HelpScout updates (SYNC-002 to SYNC-007): six changed HydroSym pages, 5 chunks added, 59 replaced (21 text changes, 38 `source_lastmod` only), 2 deleted; the canary wrote 1 first | 954 | `snapshots/20261124T125642Z-promote-after` |
| 2026-11-24 | First new video `yWuBQik8KPs` (SYNC-011/012): 8 chunks, 4 windows of narration plus description. Canary wrote 1, then 7 | 951 | `snapshots/20261124T111609Z-promote-after` |
| 2026-11-17 | SYNC-001: stored identity properties added to every chunk | 943 | `snapshots/20261117T134901Z-promote-after` |
| 2026-11-17 | SYNC-007: hosted made equal to local after SYNC-003/004 (pages replaced, article 437 removed); every vector re-embedded on the pinned runtime | 943 | `snapshots/20261117T125431Z-promote-after` |
| 2026-03 | Initial ingest of the YouTube transcripts and knowledge bases (922 objects at the 2026-11-13 backup) | 922 | Older backups hold the March vectors; do not restore from them |
