# Chunk Parent and Video Link Design

Status: **proposal**. Nothing in this note is implemented, and no schema change has been
made to the hosted collection. It records the findings and the design considerations from
the Epic 11 investigation so the SYNC-000 and SYNC-001 work can start from them.

Measurements were taken on 2026-11-09 against the hosted `SupportChunk` collection with
the read-only viewer key. Related evidence: `docs/screenshot-description-pilot.md`.
Revised on 2026-11-13 after review against the code (schema checks, hashing per layer,
baseline for the existing data, update loop) and again after the SYNC-000 spike, which
corrected the assumption about duplicate UUIDs (see section 6).

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

The 381 screenshot chunks are the web chunks whose UUID cannot be recomputed from stored
data. 380 of them open with "This screenshot..." or "This image..." and one differs, so the
figures 381 and "about 380" elsewhere in this note refer to the same set.

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
slot by slot. The SYNC-000 spike confirmed that a same-slot edit overwrites cleanly (section 6). **Open** for position shifts, to be confirmed by the baseline check (D8).

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
- An overwrite is a **full replace** (verified, section 6): every write must carry every
  property, or the omitted ones become null. Use `data.update` for metadata-only changes.

### D7. What to hash, per layer

Principle: **decide on hashes of inputs, never on regenerated outputs.** LLM-written text
(screenshot descriptions, transcripts, frame descriptions) differs on every call, so
comparing outputs would flag everything as changed on every run. Generated text is an
artefact: stored once, keyed on the hash of its inputs, and replaced only when those
inputs change.

| Layer | What is hashed | Reprocess when |
|---|---|---|
| Web text | SHA-256 of the cleaned page markdown (per page) and of each chunk's stored text (`content_hash`) | the page hash changes; then diff chunks |
| Screenshot | SHA-256 of the image bytes (`image_hash`), plus `description_model` and `prompt_version` | image bytes, model or prompt version change |
| Video | SHA-256 fingerprint of video ID, duration, title and caption-text hash (`source_fingerprint`), plus `pipeline_version` (transcription model, frame interval, vision model and prompt) | fingerprint or pipeline version changes |

- Remove boilerplate such as the YouTube player text before hashing (D5), or the hash
  would change with the player UI.
- Temperature 0 and a seed reduce LLM variation but do not remove it. The caching rule
  makes variation irrelevant for sync.
- Embeddings are recomputed only when a chunk's stored text changes (`content_hash`).
- Store full hex digests. The 8-character form is for log correlation only.
- Screenshot hashing costs nothing extra: all screenshot chunks are rewritten in the
  re-description anyway, and that is the only chance to record the image bytes, because
  the old images are gone.
- Videos rarely change at the same URL, so skip them by default and reprocess only on a
  changed fingerprint or an explicit flag. Transcription and frame description are the
  costliest steps.

### D8. Baseline for the existing data

The hosted collection has no content hash and does not store the source page, so a new
crawl is compared with the output of the March pipeline. A difference can mean the page
changed, the pipeline changed or the crawl varied.

Evidence that separates these causes:

- The doc sites publish a sitemap with `lastmod` per URL. Of the 160 stored web pages, 146
  appear in the sitemaps and all 146 have a `lastmod`. 11 were modified after the March
  ingest (the latest `ingested_at` is 2026-03-16) and 135 were not. 14 stored pages are
  not in the sitemaps (removed or renamed) and 45 sitemap pages are not stored.
- For the 135 unchanged pages, comparing a re-chunk made with current code against the
  stored text measures pure pipeline drift. On modified pages a real edit should differ in
  a few localised chunks, while drift differs everywhere.
- Stored text chunks are position-keyed and contiguous from 0 on all 160 pages, so a
  chunk-by-chunk text comparison needs no hash. Over-detection costs a re-embed of a few
  hundred chunks on local Ollama and does not affect correctness.
- Not recoverable: screenshots (the old image bytes are gone and `image_url` is not
  stored, so re-describe all) and videos (no source state, so take a new fingerprint and
  compare current durations with the stored windows).
- Caveats: `lastmod` is the CMS's claim, and whether it changes when only an image is
  replaced has not been checked. The re-chunk comparison has not been run.

## 5. Proposed model

### Chunk (existing collection, additive properties)

| Property | Type | Applies to | Purpose |
|---|---|---|---|
| `document_id` | text | all | `uuid5(NAMESPACE_URL, source_url)`; parent key |
| `content_hash` | text | all | SHA-256 of the stored text |
| `image_url` | text | screenshots | source image; null otherwise |
| `image_hash` | text | screenshots | SHA-256 of the image bytes |
| `description_model` | text | screenshots | vision model that wrote the text |
| `prompt_version` | text | screenshots | version of the description prompt |
| `source_fingerprint` | text | video | video fingerprint (D7) |
| `pipeline_version` | text | video | transcription and vision settings |

Existing chunks read null until backfilled. Adding a property to a populated collection
needs no reindex (verified on small scratch collections, section 6). Whether many optional
properties or one nested object property is better is open (section 9).

### Page record (only if D2 stage 2 is taken)

`document_id`, `source_url`, `source_type` (`web` or `video`), `source_name`, `title`,
`content_hash`, `crawled_at`, `chunk_count`, `embedded_video_ids`. A video is also a
document (its `source_url` already groups its chunks), so `embedded_video_ids` points at
other documents.

## 6. Compatibility constraints

These decide the order of work and are the main risk.

1. **The schema is checked for an exact property match.**
   - `WeaviateStore.validate_compatibility` compares the collection's properties with
     `EXPECTED_PROPERTIES` (the 8 current ones) and raises on any difference. The app
     calls it at startup (`app.py`). If a property is added to the hosted collection
     **before** the deployed app is updated, the running app fails its startup check and
     the container exits (503), as Weaviate Cloud outages already do.
   - `ensure_collection` creates the same fixed 8-property schema, so fresh local
     collections would lack any new property.
   - `tests/test_store.py` builds its fixture from `EXPECTED_PROPERTIES`.
   - `restore_snapshot` builds the schema from the manifest and validates against it, so
     it is self-consistent: an old snapshot restores to the old schema, a new one to the
     new schema.
   - **Rule:** require the 8 original properties and allow extras. The new properties are
     optional until a feature needs them. **Order:** change the check, deploy it, verify
     startup, and only then add properties to the hosted schema.
2. **Snapshot and restore are tied to one collection.** `store/snapshot.py` exports
   `COLLECTION_NAME` only and stores the schema in the manifest. A `Document` collection
   would need to be exported and restored too.
3. **Keys.** Property additions and writes need the admin key. Cloud Run keeps the
   read-only viewer key.
4. **Fixed-collection contract (Epics 9 and 10).** Additive chunk properties keep the one
   collection, the 768-dimensional embeddings and the query path unchanged. A new
   collection would reopen that contract.

### SYNC-000 spike results (2026-11-13)

Run on throwaway collections in local Docker, on Weaviate 1.28.4 (the compose file) and
1.39.13 (the version the hosted cluster reports), with 5 objects per collection. 18 of 18
checks passed on both versions. The hosted collection itself was not touched.

| Question | Result |
|---|---|
| Does the old exact-match check fail once a property is added? | Yes. The new check passes on the original and the extended schema. |
| Can a property be added to a populated collection? | Yes. Object count and vectors are unchanged and old objects read null. |
| Is a reindex needed? | No for reads, vector search and filtering after a backfill. |
| Does a partial `data.update` keep the vector and other properties? | Yes. The vector was identical before and after, and a backfilled property was filterable. |
| What does `batch.add_object` (the `add_chunks` path) do on an existing UUID? | It **replaces the whole object**: text and vector are overwritten, with no error and no duplicate row. |
| Does that overwrite keep properties it does not supply? | **No.** A `document_id` and `image_url` set earlier became null after a batch write that omitted them. |
| `data.insert` on an existing UUID? | Rejected with a 422. |
| `data.replace` by UUID? | Replaces text, vector and the new property. |
| Does `ensure_collection` build the extended schema, and does a fresh collection pass the new check? | Yes. |

Consequences:

- Re-ingesting an edited chunk in the same slot does update it. The earlier worry that
  edits could be silently skipped was wrong for that case. Stale chunks remain possible
  when positions shift, when content is removed and for screenshot chunks.
- A plain re-ingest erases any property it does not write. `SupportChunk` and
  `add_chunks` must write `document_id`, `content_hash` and `image_url` **before** any
  backfill, or the next `make ingest` would wipe the backfilled values.
- Not covered: the hosted 922-object collection, a Cloud Run revision, and the old app
  code against an extended schema (the old rule was reproduced, not run).

## 7. Staged plan

1. **SYNC-000, schema readiness:** relax `validate_compatibility` (8 required, extras
   allowed), add the optional properties to `ensure_collection`, update the tests. Code
   merged to `main`. **Next: deploy to Cloud Run and verify startup.**
2. **Spike on local Docker:** done on 1.28.4 and 1.39.13 (section 6).
3. **Make ingestion write the new fields:** `SupportChunk` and `add_chunks` set
   `document_id`, `content_hash` and `image_url` on every insert. This must precede the
   backfill, because an overwrite erases omitted properties.
4. Snapshot, then add the properties to the hosted schema (additive). Only after step 1
   is deployed and verified.
5. Backfill `document_id` and `content_hash` with `data.update` (metadata-only), and
   `image_url` by re-deriving `uuid5(image_url)` from the live pages. The pilot matched
   3 of 4, 19 of 19 and 7 of 7 on three pages; full coverage is unmeasured.
6. **Baseline validation:** re-chunk the unchanged pages and compare (D8); compare video
   durations with the stored windows.
7. Re-describe screenshots and remove the three runaway chunks (pilot note), strip the
   player boilerplate and record `embedded_video_ids`.
8. Decide on the `Document` collection (D2) from what SYNC-002 needs.

## 8. Update loop (outline)

A maintainer-run, per-source diff-and-apply pass with the page as the unit of work
(`make sync-plan`, then `make sync`).

```text
start run (new correlation ID, admin key, target=cloud)
  validate_compatibility()                     # model and 768-d dimension must match
  for each source in config/sources.yaml:      # failures isolated per source
      crawl source -> pages (markdown), or video -> metadata and captions
      for each page:
          clean markdown (strip player text, keep the video id)
          page_hash = sha256(cleaned markdown)
          compare with the stored chunks for document_id
              unchanged            -> skip: no embedding, no image calls
              changed or new       -> chunk, hash each chunk, diff
          images: extract image_urls, diff against the stored image_url values
      pages stored but absent from the crawl -> candidates for removal
  build the plan: added / changed / removed per source
  sync-plan: print counts, write nothing, stop
  sync: snapshot -> insert/update -> delete stale last -> verify
```

| Situation | Action |
|---|---|
| Page hash equals stored hash | Skip entirely |
| Text chunks differ | Re-embed only added and changed chunks. If positions shifted, replace the page's text chunks as a set. |
| Image URL already stored, same bytes, model and prompt version | Keep the stored description |
| New image URL, changed bytes, or new model or prompt version | Describe, embed, insert with `image_url` and the input hashes |
| Stored `image_url` no longer on the page | Delete that chunk |
| Page gone from the crawl, or source dropped | Delete only after an explicit flag or confirmation |
| Source or page failed to crawl | Leave its stored chunks untouched |

Safety rails: snapshot before any cloud write, insert and update before delete, an abort
threshold on the share of the collection to be deleted, replace by explicit UUID writing
every property (an overwrite is a full replace) with a re-embedded vector when the text
changes, `data.update` for metadata-only changes, verify the object count and one known
retrieval query afterwards,
admin key used only for sync, and a report saved outside Git. Cloud Run needs no
redeploy because it reads the live collection with the viewer key.

## 9. Open questions

- Is a page-level record needed in the hosted data, or is a local sync manifest enough?
- Many optional chunk properties, or one nested object property for the per-layer
  provenance fields?
- Position keys or content keys for web text after an insert (D3)?
- Does the CMS `lastmod` change when only an image is replaced? If not, image changes
  need the image-bytes hash even for unchanged pages.
- What to do with stored screenshot chunks that no live image matches (the 873-character
  chunk on `article/449`): keep or delete?
- Should the video edge also be exposed in the UI (a "watch the video" link), and does it
  belong in this epic?

## 10. Reproducibility

The analysis scripts were exploratory and are not in the repository. They need the hosted
cluster URL, the viewer key, local Ollama with `nomic-embed-text` for the retrieval tests,
numpy for the similarity test, and plain HTTP access to the doc sites' `sitemap.xml` for
the baseline counts.
