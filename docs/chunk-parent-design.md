# Chunk Parent and Video Link Design

Status: **proposal**. Nothing in this note is implemented, and no schema change has been
made to the hosted collection. It records the findings and the design considerations from
the Epic 11 investigation so the SYNC-001 work can start from them.

Measurements were taken on 2026-11-09 against the hosted `SupportChunk` collection with
the read-only viewer key. Related evidence: `docs/screenshot-description-pilot.md`.

## 1. Where the code stands today

The application works with chunks only. There is one flat Weaviate collection,
`SupportChunk`, and no document or parent object, no second collection and no
cross-reference. A search of `src/` for `parent`, `document_id` and `doc_id` finds nothing
relevant.

The implicit parent is the `source_url` string repeated on every chunk. It is what
`delete_by_source(source_url)` and `make purge-source URL=...` filter on.

### Chunk properties (hosted collection, 922 objects)

`text`, `source_type`, `source_url`, `source_name`, `timestamp_seconds`,
`section_heading`, `url_hash`, `ingested_at`. The UUID is the Weaviate object ID, not a
property. All vectors are 768-dimensional.

### How a chunk UUID is built

Every UUID is `uuid5(NAMESPACE_URL, <key string>)`, a name-based hash of a key string
(`SupportChunk` in `models/chunks.py`). It is never a hash of the content.

| Chunk type | Key string | Objects in hosted collection |
|---|---|---|
| Web text (normal path) | `source_url \| chunk_index` | 327 |
| Web text without `chunk_index` (fallback) | `source_url \| text` | rarely used |
| Video, fused audio and visual window | `source_url \| fused \| start_seconds` | 214 |
| Video transcript-only or frame-only | `...\|transcript\|...`, `...\|frame\|...` | none left |
| Web screenshot description | `image_url` | 381 |

The key tells you which slot a chunk occupies, not what it says, and nothing stored says
whether the text in a slot has changed.

### Where hashing is used

| Where | What is hashed | Purpose |
|---|---|---|
| `models/chunks.py` | `sha256(source_url)` stored as `url_hash` | page fingerprint |
| `store/snapshot.py`, `store/restore.py` | SHA-256 of the snapshot file | integrity checksum |
| `logging_config.py` | SHA-256 of the query text, 8 characters | `query_hash` in logs |
| `uuid5` (standard library) | SHA-1 over namespace and key | every chunk ID |

`url_hash` is written on every insert and defined in the schema, but nothing in the code
reads it. It hashes the URL, not the content, so it identifies a page and cannot detect a
change. There is no content hash anywhere. I found no recorded reason; the likely one is
that the pipeline was built for one-time ingestion with deterministic IDs making re-runs
idempotent, and change detection only became a requirement with Epic 11.

## 2. Problems driving the change

1. **Screenshot chunks have no provenance.** 381 of 922 chunks (41%) are keyed on
   `uuid5(image_url)` while `image_url` is not stored. Nothing links a chunk to its image,
   so a sync cannot tell which chunk belongs to which image, and a replaced screenshot
   (new CDN URL, new UUID) leaves its old description behind. See the pilot note.
2. **No page-level change detection.** An edit to a page cannot be classified as added,
   changed or removed, and an unchanged page cannot be skipped. Image description is the
   expensive step: the pilot hit Gemini quota after 8 calls.
3. **No cascade or orphan handling.** All chunks of a page can only be addressed by
   filtering on the `source_url` string.
4. **Noise chunks.** 41 web chunks on 21 pages are YouTube player boilerplate (see 3).

## 3. Findings on videos

Question examined: can screenshot descriptions be attributed to a video directly?

### A page-to-video link exists for most videos

- 25 videos are ingested. 20 are embedded on at least one ingested page; 5 are not (the
  Explainer videos on Electrical Diagrams, Danfoss PVG, Multiple sheets and Dimensions,
  and "Creating Symbols and Components").
- 27 of 160 web pages embed an ingested video. 13 videos appear on 1 page, 6 on 2 pages and
  1 on 3 pages. Only 1 page embeds more than one video.
- The crawled text contains the player's "Watch on [youtube.com/watch?v=...]" link, so the
  video ID can be read directly. Matching on the "<title> - YouTube" text gives the same
  result. Pages whose embed is missing from the chunk text are not counted, so these
  numbers are a lower bound.
- Of the 43 pages that have screenshot chunks, only 6 embed an ingested video, covering
  42 of about 380 screenshot chunks (11%). For the other 89% there is no video on the page.

### It is a relation, not a parent

A screenshot has exactly one page. A page has zero or one video (26 of 27 pages with a
video), and a video can sit on up to 3 pages. A video cannot be the parent of a screenshot.
The correct shape is an optional edge: `Page -> embeds -> Video`.

### Similarity cannot attribute a screenshot to a video moment

Tested on the 42 screenshot chunks whose page embeds a video:

| Measure | Result |
|---|---|
| Best similarity to a window of the page's embedded video | 0.757 mean |
| Best similarity to any one other video | 0.778 mean (higher) |
| Screenshots where the embedded video beats all 24 other videos | 7 of 42 |
| Best versus second-best window within the video | 0.012 mean margin; above 0.03 for 3 of 42 |

The embedded video cannot be told apart from unrelated ones, and within a video no window
stands out. Any attribution must come from the page's embed, not from vectors. All 214
video chunks do already carry per-window `[Visual]` text, so video content is already
searchable; there is nothing to gain by tying web screenshots to it.

### The embeds also leave noise in the index

41 web chunks on 21 pages (6% of web chunks, median 507 characters) are YouTube player
boilerplate: "You're signed out. Videos you watch may be added to the TV's watch history...",
"An error occurred while retrieving sharing information" and the "Watch on" link. These
are the duplicate-text groups found earlier. They add retrieval noise and carry the only
trace of the video link.

## 4. Design considerations

Each item is marked **leaning** (the direction the investigation points to) or **open**.
None is decided.

### D1. How a chunk points to its parent

Options: a Weaviate cross-reference property; a plain `document_id` property; or no
parent field and continue using `source_url`.

**Leaning: plain `document_id` property** on each chunk, equal to
`uuid5(NAMESPACE_URL, source_url)`. It is deterministic, can be computed without a lookup,
and queries need no cross-reference resolution, so the read path stays unchanged. The app
joins to a parent record only when it needs one.

### D2. Where the parent record lives

Options: a new `Document` collection in Weaviate, or a local sync-state manifest per
source (outside Weaviate).

| | Document collection | Local manifest |
|---|---|---|
| Self-describing hosted data | yes | no |
| A second maintainer sees the state | yes | only if the file is shared |
| Fixed-collection contract (Epics 9 and 10) | breaks the one-collection assumption | untouched |
| Snapshot, restore, `validate_compatibility` | all need updating | untouched |
| Effort | higher | lower |

**Leaning: stage it.** Stage 1 adds only chunk properties, no new collection. Build a
`Document` collection only if sync needs page-level hashes and video edges to live in the
hosted data. **Open** until SYNC-002 shows how much state the diff needs.

### D3. Chunk key: position or content

- Position keys (`url|index`) keep an edited chunk in the same slot, so an update is an
  in-place replace. They break when a section is inserted, because every later chunk shifts
  to a new text.
- Content keys (`url|content-hash`) survive an insert, but every edit becomes a delete plus
  an add.

**Leaning: keep the existing position keys** for web text (327 are already stored this
way) and detect change with a per-chunk `content_hash`. When a page's chunk count or
positions shift, replace the whole page's text chunks by `document_id` instead of diffing
slot by slot. **Open**, to be confirmed in SYNC-001 with the duplicate-UUID test (D6).

### D4. Meaning of the video link

An edge, not a parent (see section 3). Store the embedded video IDs as page metadata
(`embedded_video_ids`), read from the "Watch on" link at crawl time. Use it for "there is
a video on this page" in an answer, not for chunk-level attribution. This applies to 27
pages and is an add-on, not a prerequisite for the screenshot fix.

### D5. Strip player boilerplate before chunking

Remove the YouTube player text from the crawled markdown, keep the video ID as metadata
(D4), and do not index the remainder. Removes 41 noise chunks and the duplicate-text
groups. **Leaning.** Needs a pattern that is tested against pages without embeds so
legitimate text is not removed.

### D6. How writes are applied

- Take a snapshot (`make snapshot-export`) before any cloud write, as SYNC-005 requires.
- Replace objects by explicit UUID, with a re-embedded vector computed from
  `"<source_name>: <text>"`, the same input `add_chunks` uses.
- Do not rely on `batch.add_object` overwriting an existing UUID. That behaviour is
  untested and is the first SYNC-001 check (it needs the admin key).

### D7. Hash choices

- Per page: `content_hash`, a SHA-256 of the crawled markdown after boilerplate removal.
- Per chunk: `content_hash`, a SHA-256 of the stored text.
- Per image (optional): a hash of the image bytes, so a changed image can be told from an
  unchanged one even when the URL stays the same.
- Store full hex digests for these. The 8-character form is only for log correlation.

## 5. Proposed model

### Chunk (existing collection, additive properties)

| Property | Type | Purpose |
|---|---|---|
| `document_id` | text | `uuid5(NAMESPACE_URL, source_url)`; parent key |
| `image_url` | text | set on screenshot chunks; null otherwise |
| `content_hash` | text | SHA-256 of the stored text |

Existing chunks get null until backfilled. Adding a property needs no reindex.

### Page record (only if D2 stage 2 is taken)

`document_id`, `source_url`, `source_type` (`web` or `video`), `source_name`, `title`,
`content_hash`, `crawled_at`, `chunk_count`, `embedded_video_ids`. A video is also a
document (its `source_url` already groups its chunks), so `embedded_video_ids` points at
other documents.

## 6. Compatibility constraints

These decide the order of work and are the main risk.

1. **`validate_compatibility` requires an exact property match.** `EXPECTED_PROPERTIES` in
   `store/weaviate.py` lists the eight current properties and the check is
   `actual != EXPECTED_PROPERTIES`. If `document_id`, `image_url` or `content_hash` is
   added to the hosted collection **before** the deployed app is updated, the running app
   fails its startup validation with "schema is incompatible" and the container exits.
   Weaviate Cloud problems already produce a 503 that way (see the deployment notes).
   **Order:** first change the check to require the expected properties without rejecting
   extra ones, deploy it, and only then add properties to the hosted schema.
2. **Snapshot and restore are tied to one collection.** `store/snapshot.py` exports
   `COLLECTION_NAME` only and stores the schema in the manifest. Older snapshots lack the
   new properties. A `Document` collection would need to be exported and restored too.
3. **Keys.** Property additions and writes need the admin key. Cloud Run keeps the
   read-only viewer key.
4. **Fixed-collection contract (Epics 9 and 10).** Additive chunk properties keep the one
   collection, the 768-dimensional embeddings and the query path unchanged. A new
   collection would reopen that contract.

## 7. Staged plan

1. Relax `validate_compatibility` to tolerate extra properties, test it, deploy.
2. Add `document_id`, `image_url` and `content_hash` to the hosted schema (additive).
3. Backfill `document_id` from `source_url` for all 922 chunks and `image_url` by
   re-deriving `uuid5(image_url)` from the live pages (the pilot matched 3 of 4, 19 of 19
   and 7 of 7 on three pages; full coverage is unmeasured). Snapshot first.
4. Verify duplicate-UUID behaviour of `batch.add_object` (SYNC-001).
5. Re-describe screenshots and remove the three runaway chunks, per the pilot note.
6. Strip player boilerplate and record `embedded_video_ids`.
7. Decide on the `Document` collection (D2) from what SYNC-002 needs.

## 8. Open questions

- Is a page-level record needed in the hosted data, or is a local sync manifest enough?
- Position keys or content keys for web text after an insert (D3)?
- What to do with stored screenshot chunks that no live image matches (the 873-character
  chunk on `article/449`): keep or delete?
- Should the video edge also be exposed in the UI (a "watch the video" link), and does it
  belong in this epic?

## 9. Reproducibility

The analysis scripts were exploratory and are not in the repository. They need the hosted
cluster URL, the viewer key, local Ollama with `nomic-embed-text` for the retrieval tests,
and numpy for the similarity test.
