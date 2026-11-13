# Content Update Workflow

Status: **proposal**. Written 2026-11-13. Facts marked "verified" were checked that day;
decisions are listed in section 8 and need confirmation before the stories are built.

## 1. Why this note exists

Epic 11 described the content-sync mechanism (hashes, diffs, UUIDs) but not who runs it,
what they need beforehand, how the local Docker collection relates to the hosted one, or
what an update run looks like from start to finish. This note supplies that operating
model. The mechanism details stay in `docs/chunk-parent-design.md`; the evidence for the
screenshot problems is in `docs/screenshot-description-pilot.md`.

## 2. Personas

| Persona | Who today | Goal | Touches | Needs from this system |
|---|---|---|---|---|
| **Support colleague** | Paro support staff | Correct, current, cited answers | The hosted Cloud Run app (read-only viewer key). Never runs ingestion. | No outage while content is updated. Open: a way to see how current the content is. |
| **Maintainer** (data steward) | The project owner, from a laptop | Keep the collection in step with the sources and decide what goes live | `config/sources.yaml`, local Docker Weaviate, Firecrawl, Mistral, a vision model, the hosted admin key | A preview of every change, cost and quota control, tests before going live, a rollback |
| **Operator** (developer) | Same person today | Deploy, monitor, rotate keys | Cloud Run, Secret Manager, Cloud Logging, schema compatibility | Deploy ordering rules, the observability from Epic 11 Phase 1 |
| **Content owner** | Paro documentation and video authors | Publish and correct documentation | The doc sites and YouTube | Not a system user. Their change rate sizes the workload. |

The maintainer and operator are the same person today. They are kept separate because they
use different keys and fail in different ways: the maintainer can corrupt data, the
operator can break the running service.

## 3. Starting point: the frozen collection

All verified on 2026-11-13.

- **Hosted `SupportChunk`:** 922 objects (708 web, 214 video), ingested 2026-03-13 to
  2026-03-16, Weaviate 1.39.13, 8 properties, no hashes of any kind that detect change.
- **Local Docker `SupportChunk`:** identical to hosted. The same 922 UUIDs, identical text
  on all 922, the same schema and dates. Weaviate 1.39.13 (pinned in `compose.yaml`, aligned with the hosted cluster on 2026-11-13; it was 1.28.4 before). The
  hosted collection was restored from a snapshot of the local one.
- **Both are frozen.** The only paths that change either are a full re-ingest, a purge
  target and a snapshot restore. The deployed app is read-only by design (Epics 9 and 10).
- **The source list is not reproducible from tracked files.** `config/sources.yaml` is
  gitignored and currently lists only the 25 videos. The three knowledge bases
  (`hydrosym`, `hydroman`, `hydrocam` on `paro.nl/article/`) exist only in
  `config/sources_original.yaml`, also gitignored.
- **The vision model is ambiguous.** `.env` sets `VISION_MODEL` four times (the last value
  wins), `create_vision_llm` always goes through OpenRouter, and the model that wrote the
  stored descriptions is recorded nowhere.
- **Promotion tooling is first-time only.** `make snapshot-bootstrap-cloud` has no replace
  option. Updating hosted by snapshot means running the restore by hand with `--replace`,
  which deletes the hosted collection and recreates it, so the app has no collection until
  the restore finishes.
- **One `.env` drives both targets** through `WEAVIATE_MODE` and `WEAVIATE_URL`. A wrong
  value makes `make ingest` write to production.

### Change rate of the sources

From the sites' sitemaps against the March ingest: of 160 stored pages, 146 appear in the
sitemaps; 11 were modified since the ingest, 135 were not; 14 stored pages are gone from the
sitemaps and 45 sitemap pages are not stored. That is roughly 11 edits, 45 new pages and 14
removals over eight months. Content changes are small and infrequent.

## 4. Prerequisites for an update run

| Area | Requirement | Status on the maintainer machine | Gap |
|---|---|---|---|
| Tools | Docker, Ollama with `nomic-embed-text` (768-dimensional), ffmpeg, yt-dlp, uv and Python 3.12 to 3.14 | All present | None |
| Keys | `FIRECRAWL_API_KEY` (crawling, uses credits) | Set | Credit balance not checked |
| Keys | `MISTRAL_API_KEY` (Voxtral transcription of videos) | Set | |
| Keys | Vision model access. Today OpenRouter only. The new descriptions use Gemini directly. | OpenRouter key set | `create_vision_llm` has no Gemini path (SYNC-001) |
| Quota | Gemini quota and spend cap for about 380 image descriptions | The pilot hit a 429 after 8 calls | Needs a known quota before a full run |
| Keys | `WEAVIATE_ADMIN_API_KEY` for hosted writes | Set | |
| Access | YouTube cookies for yt-dlp (IP blocking) | Secret `YOUTUBE_COOKIES` in Secret Manager (version 1, 2026-11-13), fetched at run time | The uploaded cookies are from March: of 8 cookies, 5 expired and 3 are session-only. Export fresh cookies before the next run. |
| Config | A source list for all three knowledge bases and the videos, kept in one private versioned bucket (section 10) | Missing, and held on one Mac only (sections 3 and 10) | SYNC-008 |
| Config | One `VISION_MODEL` value | Four duplicate lines | SYNC-008 |
| Config | Separate local and hosted settings so a run cannot write to the wrong target | One shared `.env` | SYNC-008 |
| Versions | Local Weaviate equal to the hosted version | 1.39.13 on both (aligned 2026-11-13) | **Done.** Fresh volume, restored from the bucket backup: 922 identical objects |
| Data | Local equal to hosted before an update starts | Verified identical | Needs a repeatable check (SYNC-008) |
| Schema | Deployed app tolerates extra properties | Merged, **not yet deployed** | SYNC-000 deploy |
| Network | The doc sites, their sitemaps and YouTube reachable | Yes | |

## 5. The update workflow

Two stages with different mechanisms. The local Docker collection is a disposable staging
copy; the hosted collection is the only production copy.

```text
Content owners edit pages and videos
        |
Stage A: refresh the LOCAL collection      Maintainer, laptop, local Docker
   preflight -> fetch -> clean -> hash -> diff against local -> plan -> apply to local
        gates: plan reviewed, quality limits met, golden questions pass
        |
Stage B: promote to HOSTED                 Maintainer, admin key
   snapshot hosted -> diff local against hosted -> plan -> apply -> verify
        |
Colleagues use the hosted app (viewer key). No redeploy and no outage.
```

### Stage A: refresh local

| Step | Command (proposed) | Persona | Decision or output |
|---|---|---|---|
| 1. Preflight | `make sync-preflight` | Maintainer | Tools, keys, quota, version drift and the local-equals-hosted check report pass or fail. No changes made. |
| 2. Plan | `make sync-plan` | Maintainer | Per-source counts of added, changed, unchanged and removed pages, images and videos, plus the expected number of embedding and vision calls. Nothing written. |
| 3. Review | | Maintainer | Approve the plan, restrict it to some sources, or stop. |
| 4. Apply | `make sync` | Maintainer | Writes to the **local** collection only and refuses a hosted URL. Insert and update first, delete last. |
| 5. Validate | `make sync-check` | Maintainer | Schema compatible, no oversize or repetitive chunks, counts match the plan, golden questions give acceptable answers. |

### Stage B: promote to hosted

| Step | Command (proposed) | Persona | Decision or output |
|---|---|---|---|
| 1. Back up | `make snapshot-export` of the hosted collection | Maintainer | The rollback point, stored outside Git. |
| 2. Plan | `make promote-plan` | Maintainer | A diff of local against hosted by UUID and stored `content_hash`: added, changed, removed. Nothing written. |
| 3. Apply | `make promote` | Maintainer | Admin key. Insert and update first, delete last, no collection delete, abort above a deletion threshold. |
| 4. Verify | built in | Maintainer | The UUID set and text of local and hosted are identical, the count matches, one known query retrieves correctly. |
| 5. Smoke test | | Operator, colleague view | Ask known questions on Cloud Run and read the inference logs. |

**Rollback:** restore the Stage B snapshot with `--replace`. That deletes and recreates the
hosted collection, so it causes an outage and is for failures only. A promotion that
fails part-way is safe to repeat, because the diff is recomputed.

**Invariant:** after every promotion the local and hosted collections are identical.
Stage A and Stage B both depend on it, so it is checked in preflight and after promotion.

## 6. Where hashing fits, and where it does not

- **Stage A needs it.** Page hashes, chunk `content_hash`, image hashes with the model and
  prompt version, and video fingerprints decide what to re-crawl, re-describe and
  re-embed. They avoid repeating expensive work (design note D7).
- **Stage B mostly does not.** Local and hosted share a schema and UUIDs, so promotion is a
  collection-to-collection diff. It uses UUIDs and the stored `content_hash`, not source
  hashes.
- **The frozen collection has no hashes.** The first Stage A run is a baseline run. It
  compares re-chunked text with the stored text, uses the sitemap `lastmod` as the
  independent change signal (design note D8), re-describes every screenshot anyway, and
  writes `content_hash` for all chunks. After the first promotion both collections carry
  hashes.
- **What hashing buys here.** With about 11 edits in eight months, text is not the
  expensive part. The cost is the roughly 380 image descriptions and the 25 videos
  (Voxtral and the vision model). Hashing earns its place by not redoing those on every
  run, not by speeding up text sync.

## 7. Alternatives considered

| Option | How | For | Against |
|---|---|---|---|
| 1. Full refresh | Rebuild local from all sources, export a snapshot, restore to hosted with `--replace` | Simplest. Uses existing tooling. Needs no hashes. | Re-describes all images and reprocesses all videos every time. The hosted app has no collection during the restore. |
| 2. Direct incremental sync | Sources straight to hosted (the earlier epic wording) | No local step | No staging, no test, no cheap rollback. Production receives crawler variance and the admin key is used on every run. |
| 3. Two-stage (recommended) | Incremental into local, then promote the diff to hosted | Staging and tests before production. Cheap rollback. Hosted stays up. | Two stages to build and an invariant to maintain. |

Recommendation: option 3. Option 1 remains the fallback if hashing proves unreliable, and
it is the bootstrap path for a brand new hosted cluster.

## 8. Assumptions to confirm

1. One person is maintainer and operator today; the roles are conceptual.
2. The local Docker collection is a disposable staging copy and hosted is the only
   production copy.
3. Promotion writes to hosted with the admin key while the app stays up (no replace).
4. Updates are run by hand when needed. Scheduled runs are out of scope (candidate Epic 12).
5. The freshness colleagues expect is acceptable at days to weeks, not minutes.
6. The source registry and the run state share one private, versioned bucket (section 10),
   and the public repository holds only an example file.
7. New videos and knowledge bases are added by editing that file. It lists roots only:
   the pages under a knowledge base come from its sitemap, not from this file.
8. The first run is gated by the sitemap `lastmod`: only the 11 modified pages are crawled,
   alongside the 45 new pages and the 14 removal candidates.

## 9. Gaps to close, mapped to stories

| Gap | Story |
|---|---|
| Deployed app rejects extra properties | SYNC-000 (done, deployed 2026-11-13, revision 00024) |
| Identity, provenance and hashes; vision path limited to OpenRouter | SYNC-001 |
| Plan against the stored collection | SYNC-002 |
| Apply to the local collection only | SYNC-003 |
| Removed pages and sources | SYNC-004 |
| Snapshot, verification, credentials | SYNC-005 |
| Tests and documentation | SYNC-006 |
| Promote local to hosted, equality check, rollback | SYNC-007 |
| Preflight, tracked sources, one vision model, target separation, golden-question gate | SYNC-008 |

## 10. Where sources, secrets and state live

The repository is **public**, so it holds code, tests, documentation and example files
only. Nothing below goes into it.

Keep two classes of data apart, because credentials need different protection from
everything else:

| Class | Examples | Home | Copies |
|---|---|---|---|
| **Registry and state** | The source registry (the list of knowledge bases and videos, scope, crawl rules, owner); per-run reports; sitemap captures; the manifest of what each run ingested (URLs, `lastmod`, hashes); snapshots of the collection | **One private Cloud Storage bucket** with object versioning, soft delete and uniform bucket-level access | The bucket, with versions; a local cache of the registry |
| **Credentials** | Firecrawl, Mistral, OpenRouter, Gemini and Weaviate admin keys; the YouTube cookie files (a logged-in session is a credential) | **Secret Manager**, which already holds the three runtime secrets. The local `.env` is a gitignored cache, never the only copy. | Secret Manager plus a password manager for recovery |

### One bucket for registry and state

The registry and the run state are both private, non-secret, and need history and an
off-machine copy, so they share a home. A private git repository is the wrong shared home:
a snapshot is about 18 MB of vectors and every new version would be added to the history
for good. A bucket handles both the small, hand-edited registry and the large, append-only
artifacts.

Layout, with a policy per prefix:

| Prefix | Content | Behaviour |
|---|---|---|
| `registry/` | `sources.yaml` and its schema | Small, hand-edited, versioned. History comes from object versions. |
| `runs/<timestamp>/` | The plan, the apply report, the lock manifest | Append-only. Each manifest records the registry **generation** it ran against, so a run can be reproduced exactly. |
| `captures/` | Sitemap captures per run | Append-only. They are the evidence for the `lastmod` gate. |
| `snapshots/` | Collection snapshots (hosted before each promotion) | Large. A lifecycle rule keeps the last N versions. |

Working with the registry:

- The maintainer edits a local copy, which is a gitignored cache. `make sources-pull` and
  `make sources-push` move it; the push validates the file against its schema and refuses
  if the remote changed since the pull (a generation check).
- Preflight compares the local cache with the bucket and reports a difference.
- Cloud Run's service account gets **no access** to this bucket. The app needs the viewer
  key only.
- Use a dedicated bucket. The existing `run-sources-<project>-<region>` bucket is created
  and used by `gcloud run deploy --source`, so it is not a place for these files.

What is lost against git, and how it is covered:

| Lost | Mitigation |
|---|---|
| Review and diff before a change | Object versions plus the generation in every run manifest. A `make sources-diff` against the previous version if needed. |
| Branch and pull-request workflow | Not needed while one person maintains it. If more maintainers join, add a review step or move the registry back to a private repository. |

Principles:

- Keep credentials apart from everything else.
- Version everything that changes, and keep one copy off the machine.
- Least privilege: the maintainer holds the admin key and the bucket, Cloud Run holds the
  viewer key only.
- Reproducible: every run records the registry generation and what it ingested.
- The public repository contains `sources.example.yaml` with placeholders only.

### Today (verified 2026-11-13)

- `config/sources.yaml` and `config/sources_original.yaml` are gitignored, so they are
  safe from the public repository, but they exist **only on one Mac**.
- The YouTube cookie files in `config/` were protected only by the broad `*.txt` ignore
  rule. **Fixed:** explicit rules now ignore `config/*cookies*` and `*.cookies`, and the
  configured file is in Secret Manager as `YOUTUBE_COOKIES` (version 1, verified by
  SHA-256). The three local files are all from March 2026; the one `.env` points at has no
  valid persistent cookie left. They stay on disk until you confirm they can be deleted.
- The tracked `Makefile` hardcoded the GCP project ID as the default for `make deploy`.
  **Fixed:** `GCP_PROJECT` comes from the environment or `.env` and has no default; the
  target stops with a clear message when it is unset.
- The collection snapshot (18.5 MB, taken 2026-09-25 from Weaviate 1.28.4; a current copy from 1.39.13 is in the state bucket) is on the Mac
  (`../multimodal-rag-weaviate-snapshot`) and shared through an approved Drive link
  (README).
- The project's only bucket is `run-sources-<project>-europe-west4`, created by
  `gcloud run deploy --source` for build uploads. No bucket exists for the registry or the
  run state.
- The runtime secrets for the hosted app are already in Secret Manager.

### What "production ready" means for the pipeline

Reproducible sources, managed secrets, persisted run state, idempotent runs (the hashes),
observable runs (the Phase 1 logs), a rollback (snapshot and restore), and access control
by role. Today the pipeline has the idempotent IDs and the logs; the rest is the work in
SYNC-005, SYNC-007 and SYNC-008.

### Decisions (2026-11-13)

1. **One bucket** for the registry and the run state, with the prefixes above, separate
   from the deploy bucket. **Done:** `gs://multimodal-rag-state-gen-lang-client-0552956404`,
   private, public access blocked, versioning on (`STATE_BUCKET` in `.env`).
2. **Region:** `us-central1`, chosen over `europe-west1`. Outside europe-west4 (the Cloud
   Run region), so one regional problem does not take out both. The data therefore leaves
   the EU; the chunks come from public docs and videos. A bucket's location cannot be
   changed later.
3. **Access:** the admin and operator only. Cloud Run's service account has no access.
4. **Snapshot retention:** keep the last 2 versions. **Done** as a bucket-wide lifecycle
   rule (current version plus one older); per-object, not per-snapshot, so it is not yet
   scoped to `snapshots/`.
5. **Cookies:** moved to Secret Manager. **Done.** The secret `YOUTUBE_COOKIES` exists, and
   ingestion fetches it into a private temporary file for the run and deletes it
   afterwards (`YOUTUBE_COOKIES_SECRET` and `GCP_PROJECT` in `.env`). A secret wins over a
   local file. To rotate: `gcloud secrets versions add YOUTUBE_COOKIES --data-file=<file>`.
6. **Deploy variable:** the Makefile no longer hardcodes the project. **Done.**

**Backup:** `make snapshot-backup [TARGET=cloud]` exports the collection to a temporary
directory and uploads it to `snapshots/<UTC timestamp>-<target>/`, data first and
`manifest.json` last, verifying each object's MD5. A prefix without a manifest is an
aborted upload. First backup taken 2026-11-13: 922 chunks, 768 dimensions, Weaviate
1.39.13, SHA-256 `261ee24a61a6...`.

## 11. Reproducibility

The local-versus-hosted comparison reads both collections (anonymous local access and the
hosted viewer key) and compares UUID sets and text. The sitemap counts come from
`sitemap.xml` on the two doc sites. The scripts were exploratory and are not in the
repository.
