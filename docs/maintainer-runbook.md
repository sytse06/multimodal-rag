# Maintainer runbook: updating the support knowledge base

How to bring changed documentation into the app people use, safely. It is a procedure, so
it says what to run and what to check. The reasons behind it are in
`docs/content-update-workflow.md`; the stories are in `docs/epics.md` (Epic 11).

An update has two stages. **Stage A** refreshes the copy on your Mac (local Docker).
**Stage B** copies that reviewed state to the hosted collection that colleagues search. Nothing
reaches hosted without passing through local first, and a person approves each step that
writes.

**Where the truth lives.** Content and configuration follow different directions.

| State | Source of truth | Direction |
|---|---|---|
| Content (the chunks) | Local Weaviate, until promoted | local → hosted, by `make promote` |
| Registry, page ledger, locks, run records, backups | The state bucket | You edit a local copy and push it; runs read and write the bucket |

The bucket leads for configuration and state, for four reasons. (1) It is not tied to one
machine: the registry once existed on one Mac only, and a laptop file cannot be read by a
second maintainer or a scheduled job. (2) It is versioned with soft delete, so a bad edit
is recoverable. (3) The ledger, the locks and every run record already live there, and a
plan is refused if the registry or the ledger changed after it was built, so the registry
must sit with the state it is checked against. (4) The serving app has no access to it, so
configuration and the published content stay separate. The local `config/sources.yaml` is
an editing buffer, nothing more. Full automation (SYNC-009) will read the bucket and no
laptop file.

## 0. Rules that are never bent

1. **Local first.** Hosted changes only through `make promote`. The store refuses any other
   write to it.
2. **A person approves every write.** You review the plan, you name any page to delete, you
   confirm the promotion. Exit code 10 always means "a person must read this".
3. **Embeddings come from the pinned Ollama on port 11435**, never from the Mac app. The
   version is pinned in `compose.yaml` to match production. A mix of runtimes makes new
   chunks outrank old ones, which is how a first apply once failed its gate.
4. **Restore only from the newest backup: the `promote-after` backup of 2026-11-17 or
   anything later.** Every earlier backup holds the March vectors, and restoring one brings
   the drift back.
5. **Do not use `make ingest` to update content.** It reprocesses every video and every page
   and, with a vision model set, calls it on every frame and image. To add a video use
   section 3b. Never use `make purge-video` to redo one video: it deletes every video chunk.
6. **Keep Docker's disk under 80%.** Above 90% Weaviate sets itself read-only and writes
   fail.

## 1. Who decides what

| Decision | Who | Where it is recorded |
|---|---|---|
| Which sources are in scope | Maintainer | The registry (`make sources-push`) |
| Whether a plan may be applied (and an exit code 10 accepted) | Maintainer | The run in `runs/<id>/` and the bucket |
| Which removed page is deleted | Maintainer, by name | `CONFIRM_REMOVALS` on the apply command |
| When hosted is promoted, and at what moment | Maintainer | The promotion's run directory |
| Keys, bucket, Cloud Run, the Weaviate plan | Operator (today the same person) | `docs/gcp-deployment.txt`, the epic |
| What the documentation says | Content owner | Not a system user |

## 2. Before every run

Do these in order. Stop at the first problem.

1. **Docker Desktop is running and its disk is under 80%.** `docker system df` shows what is
   reclaimable; `make docker-up` starts Weaviate and the pinned Ollama together. The first
   time only, `make ollama-up` also pulls the model.
2. **Registry:** the preflight (next step) says whether your local copy matches the bucket.
   You only need `make sources-pull` when it does not, or before you edit the registry
   (section 3b). It refuses to overwrite local edits. If it refuses, you have edits that are
   not pushed: push them, or compare with `config/sources.yaml.prev`. Do not reach for
   `FORCE=1`; it discards those edits.
3. **Run the readiness report:** `make sync-preflight` (`LOCAL_ONLY=1` skips the cloud
   checks). Read it before anything else:

   | Line | Good | Stop if |
   |---|---|---|
   | embedding runtime | Ollama at the pinned version, weights `970aa74c0a90` | It names another version: `make ollama-up`, then re-check |
   | local vectors | sampled vectors match a fresh embedding, lowest 1.0000 | Lowest under 0.99: `make sync-reembed` first |
   | registry | matches the bucket | The local copy is stale: `make sources-pull` |
   | latest backup | under 7 days old | Older: `make snapshot-backup TARGET=cloud` |
   | gemini key | accepted by the API | Rejected: a key that is set can still be invalid; fix before any video work |
   | local vs hosted | identical, or a difference you expect | A difference you cannot explain |
   | hosted vectors | match a fresh embedding | Drifted: a promotion is needed |
   | keys, tools, `.env` duplicates | OK | Any FAIL |

   A WARN is information. A FAIL means stop and fix it.
4. **Gemini quota and spend** cannot be queried; check AI Studio before any run that calls a
   vision model. The web lane does not call one.

## 3. Stage A: refresh local

1. **See what changed.** `make sync-inventory` compares each knowledge base's sitemap with
   the ledger and prints the pages that are modified, new or removed. "Nothing to do" is the
   usual answer; stop there. A host whose sitemap failed or looks too small is skipped, so a
   fetch problem never turns into deletions.
2. **Build the plan.** `make sync-plan` fetches only the listed pages (about one Firecrawl
   credit per page), chunks them with the same code `make ingest` uses, compares them with
   what is stored, and writes `runs/<id>/plan.json`. It also uploads the run to the bucket.
   It writes nothing to Weaviate. Read the summary:

   - **pages, failures:** failures should be 0. A page that failed to fetch is listed and
     makes the plan need review.
   - **positions write / delete / skip,** and how many pages really changed. `lastmod`
     over-reports: many "modified" pages differ only by the site template.
   - **template findings:** any line here means the site's header or footer changed and the
     normaliser may need a new rule. Do not apply until you understand it.
   - **removal candidates:** each shows what the page itself says: `gone 404` can be
     deleted, `still-served` and `unknown` cannot.
   - **policy:** `within` is exit 0. `needs_review` is exit 10 with the reasons listed: a
     large batch, a source missing too many pages, a changed template. Reading the reasons
     is the review.
3. **Apply it.** `make sync-apply RUN=runs/<id>` shows the summary again and asks before
   writing to **local only**. It takes a lock, proves each write matches the plan, takes a
   snapshot of local, writes, deletes, verifies, runs the retrieval gate, and updates the
   ledger only if every step held. Add `YES=1` to skip the question and `ACCEPT_REVIEW=1` if
   you have read an exit-10 plan and accept it. Status `applied` or `nothing-to-write` is
   success.
4. **Delete a removed page only by naming it.** `make sync-apply RUN=runs/<id>
   CONFIRM_REMOVALS='<url>'` deletes that page's chunks, screenshot chunks included. The page
   must be a candidate in this plan and its evidence must say it is gone, or moved to a page
   that is held. Pages of a source you dropped from the registry are confirmed with
   `CONFIRM_SOURCE='<host>'`. Without either, removals are listed as waiting and nothing is
   deleted.
5. **Check the result.** `make sync-inventory` should now show nothing modified, new or
   removed, and `make sync-preflight` should show local differing from hosted only by what
   you just changed.

Do not run another command that changes the ledger between building a plan and applying it.
A plan records the ledger and registry versions it was built from and is refused if either
moved. If that happens, build a new plan.

### 3a. When a story adds stored fields (a conversion)

Some stories add properties to every chunk (SYNC-001: `document_id`, `content_hash`,
`source_lastmod`, `embedded_video_ids`). That is neither a page sync nor a promotion, so it
has its own short stage. It changes metadata only: no text, no vector, no crawl, no
embedding.

1. **Run the checks in section 2.** The readiness report must show local vectors current.
2. **Dry run:** `make sync-convert DRY=1`. It shows how many chunks change, which
   properties will be added to the schema, how many pages get a `lastmod` (a page the site
   has modified since keeps what it already holds, or none) and how many link a video.
   Nothing is written.
3. **Convert:** `make sync-convert`. It asks, takes a lock, snapshots local to
   `runs/convert-<time>/snapshot-local`, adds the missing properties **with their declared
   types** (never by a write, because Weaviate would guess a type it cannot change on a
   populated collection), updates the fields, and then compares the object count and a
   checksum of every text and vector with what they were before. Any difference is printed
   with the rollback command and exits 1.
4. **Check it:** `make sync-preflight` still passes, and `make sync-convert DRY=1` now says
   `0 to update`. Re-running is always safe.
5. **Promote:** `make promote-plan` now shows every chunk as a replace (the properties
   differ, the vectors do not) and names the properties the target lacks. Read it; there
   should be no add, no delete and no vector refresh. A schema addition always needs review
   (`ACCEPT_REVIEW=1`) because **Weaviate cannot remove a property again**:
   `make promote-rollback` puts objects back but leaves the extra properties, which are
   harmless to the app (its startup check ignores unknown properties). The promotion
   creates the properties itself, with their declared types, after the backup and before the
   canary write. Rehearse it first (section 4, step 2): the rehearsal copy is seeded from a
   backup that still has the old schema, so it runs exactly this path.

**Rollback:** `make snapshot-restore SNAPSHOT_DIR=runs/convert-<time>/snapshot-local
REPLACE=1`. The conversion changes nothing the snapshot does not hold.

### 3b. Adding a video

> **Two things changed in SYNC-012 (part 1).** The 25 videos already stored have frame
> descriptions of a median 476 words, mostly an inventory of the window's menus and
> toolbars, cut at the embedder's 400-word limit. A new video now gets a plain-text
> description of at most 100 words, and each 30-second window is stored as **two chunks**:
> its narration (`[Transcript] ...`) and its description (`[Visual] ...`), with the same
> address and timestamp, so a question can match either without the other diluting it.
> `make ingest` writes the same format. Before you add a video, set `VISION_MODEL` in
> `.env` to `gemini:gemini-3.8-flash`: that calls Gemini directly with the
> settings the prompt was measured with (temperature 0, low thinking, a capped output). An
> OpenRouter id such as `google/gemini-3.8-flash` still works but runs at temperature 0.3
> with no cap, which nobody measured; the preflight's `vision model` line checks the
> name. A video that has chunks is never looked at again. `docs/epics.md` (SYNC-012) has
> the evidence and the open decisions. `make sync-videos DRY=1` is safe to run at any
> time.

A video is new when no stored chunk carries its `document_id`, which is a hash of the
canonical address, so `youtu.be/<id>` and `watch?v=<id>` are the same video. Only new
videos are ingested; nothing already stored is touched. Everything happens on local.

1. **Run the checks in section 2.** The report must show the embedding runtime pinned, the
   YouTube cookies not expired, the Gemini key accepted, and `VISION_MODEL` set. Without a vision model the lane
   refuses (`--allow-no-vision` overrides it): a video stored with no frame descriptions is
   never revisited, so it would stay different from the others for good.
2. **Add it to the registry:** `make sources-pull` first (the bucket is the source of truth,
   your file is a copy), edit `config/sources.yaml`, then `make sources-push`. Use `https://www.youtube.com/watch?v=<id>` without the `?si=...`
   share token (it would end up in every stored address and citation link), and the video's
   own title as the name. Set `skip_voxtral: true` only for a silent screen recording (it
   has no transcript at all; the name is historical).
3. **Dry run:** `make sync-videos DRY=1`. Read it:
   - how many registry videos are already stored and how many are new;
   - per new video its length, the path ingest will take, the vision calls (one frame per
     30 seconds on the normal path, one per 5 on a silent recording) and where the
     transcript comes from: the video's **manual captions** (free), or a **Gemini
     transcription** with its minutes of audio. Automatic YouTube captions are never used,
     because they are weakest on product names. This is what the run will spend;
   - a warning that `config/sources.yaml` has edits not pushed yet. The dry run goes on,
     so you can preview before pushing, but the real run **refuses** until you push;
   - a video whose length is "unknown" (YouTube's metadata could not be read): its cost
     is not in the total, which then says it is incomplete;
   - `Refused`: an address it cannot identify (`shorts/`, `live/`), two entries for one
     video, or a stored video with no `document_id` (`make sync-convert` first). Fix the
     registry and push it; nothing is ingested while anything is refused.
4. **Ingest:** `make sync-videos`. It asks, takes the lock, snapshots local, then ingests
   each new video **all or nothing**: if a frame cannot be described the whole video fails
   and none of its chunks are stored (a description that is empty, far over the cap or a
   repetition loop is retried once, then counts as a failed frame). Each window stores up
   to two chunks, narration and description, and a chunk the embedder would cut also fails
   the video. Afterwards it proves that every earlier object is
   unchanged and that every new object belongs to a new video, writes exactly which
   objects it added to `runs/videos-<time>/videos.json` (it writes the list of planned
   videos there **before** ingesting, so even a crashed run can be rolled back), and runs
   the retrieval gate. Exit 1 means a video failed or produced no chunks, the proof failed
   or the gate failed; it prints the video and the reason.
5. **Check it:** `make sync-videos DRY=1` now says there is nothing to ingest, and the
   preflight is green. Ask a question the video answers and look at the cited timestamp.
6. **Promote:** `make promote-plan` shows the new chunks as **adds**, about two per
   30-second window, and nothing else: no
   replace, no delete, no vector refresh, no schema change. If it shows anything else,
   stop and find out what changed. Then follow section 4.

**Rollback:** `make sync-videos ROLLBACK=runs/videos-<time>` deletes every object of the
videos that run ingested, which were new when it began, so it removes exactly what the run
added and nothing else. It changes local only: after a promotion, `make promote-plan` will
then show the matching deletes, and they need their own promotion. That is also how to redo one video that was stored wrongly: roll
the run back, fix the cause, and run again. The snapshot in the same run directory is the
backstop.

## 4. Stage B: promote to hosted

Choose a quiet moment. While the promotion runs, some hosted chunks hold new vectors and
some old, which pushes search results toward the new ones. It lasts a few minutes.

1. **Plan:** `make promote-plan`. It reads local and hosted and writes
   `runs/promote-hosted-<time>/plan.json`. It lists how many objects to add, replace, refresh
   the vector of, and delete, and holds a fingerprint of both sides. A normal promotion has
   almost no vector-only updates; more than 25 needs review, because it means local drifted.
2. **Rehearse when the code changed since the last promotion, or after a long gap.**
   `make rehearsal-up` starts a second Weaviate; download the newest hosted backup from
   `gs://<bucket>/snapshots/` and seed it with `make rehearsal-seed SNAPSHOT_DIR=<dir>`.
   `make promote-plan TARGET=rehearsal` should show the same plan as hosted. Run the
   promotion and the rollback against it with `TARGET=rehearsal`, check a new plan shows
   nothing to do, then `make rehearsal-down`.
3. **Canary first.** A plan is a snapshot of hosted at the moment it was made, with a
   fingerprint of both sides. **Every write to hosted (the canary, the promotion, a rollback)
   makes every earlier plan stale, and `make promote` refuses a stale plan.** So the
   sequence uses three plans, and each command takes the newest one:

   | Step | Command | Plan it takes | What it does to hosted |
   |---|---|---|---|
   | 1 | `make promote-plan` | (builds plan A) | nothing |
   | 2 | `make promote RUN=<plan A> CANARY_ONLY=1` | A | backs hosted up, writes **one** object, reads it back and compares it, stops |
   | 3 | `make promote-plan` | (builds plan B) | nothing. Plan A is stale now: the canary changed hosted |
   | 4 | `make promote RUN=<plan B>` | **B, never A** | backs up, writes the rest, verifies, gates, backs up again |
   | 5 | `make promote-plan` | (builds plan C) | nothing. It must say 0, 0, 0, 0 |

   Add `ACCEPT_REVIEW=1` only for a reviewed exit-10 plan. The newest plan is the directory
   the last `make promote-plan` printed (`kept: runs/promote-hosted-<time>`); `ls -t runs |
   grep promote-hosted | head -1` shows it too. Copy that name, not an earlier one from your
   terminal history. If you paste the wrong one, nothing is written: the command stops with
   `the hosted changed after the plan was made`. Build a plan again and use that.
4. **Promote.** `make promote RUN=<plan B>` asks, then: takes a backup, writes the rest
   in batches, deletes last, verifies **every** object, runs the gate, and takes a backup
   after. Status `promoted` is success; `gate-failed` means the write is done but a canary
   question ranks worse, so read the gate lines and decide whether to roll back.
5. **Confirm it.** `make promote-plan` must say 0 to add, replace, refresh and delete.
   `make sync-preflight` should show local identical to hosted and hosted vectors current.
   Then ask the three canary questions in the app (`config/canaries.yaml`).

## 5. Rolling back

**Local, after a gate failure.** The ledger was not updated. The apply prints the command:
`make snapshot-restore SNAPSHOT_DIR=runs/<id>/snapshot-local REPLACE=1`. Then rebuild the
plan.

**Hosted.** `make promote-rollback SNAPSHOT_DIR=runs/promote-hosted-<time>/snapshot-hosted-before
TARGET=hosted`. It writes every object of that snapshot back in place and deletes what the
snapshot lacks, with no outage, and reports any difference left. If the promotion stopped
part-way, hosted may be partly changed: either roll back like this, or build a new plan
(which is the remaining difference) and promote that.

**After a rollback**, hosted holds whatever the snapshot held. The last resort for a damaged
collection is a full restore, which deletes and recreates the collection and so causes an
outage: `uv run python -m multimodal_rag.store.restore --target cloud --snapshot-dir <dir>
--replace`. (`make snapshot-bootstrap-cloud` passes no `--replace` and refuses while the
collection exists.) It has not been run against hosted since the first bootstrap. Use the
newest backup only (rule 4).

**Not rehearsed:** rolling local back after a *successful* apply, which leaves the ledger
claiming pages local no longer has. The ledger object in the bucket is versioned, so an
earlier generation can be restored, but this has not been tried. Prefer building a new plan
that corrects forward.

## 6. When something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `the ledger changed after the plan was made` / `the registry changed...` | The plan is stale | Build a new plan |
| `the hosted changed after the plan was made` | Something wrote to hosted since the plan: usually your own canary, because you used the canary's plan again | Run `make promote-plan` and use the plan it prints; if you did not just write to hosted, find out what did |
| `plan hash does not match` | The plan file was edited | Build a new plan; never edit one |
| `Ollama X is not the pinned Y` | Wrong embedding runtime | `make ollama-up`; check `EMBEDDING_BASE_URL` |
| `stored vectors no longer match their text` | Index and runtime differ | `make sync-reembed` (try `DRY=1`), then re-run |
| `store is read-only due to: resource pressure` | Docker's disk is nearly full | Free space, below 80%; Weaviate recovers by itself |
| `another run holds the lock` | A run is active or died | Check nobody is running; then `BREAK_LOCK=1` |
| Plan exit 10, `the site template may have changed` | The header or footer wording moved | Read the template lines; the normaliser may need a rule |
| `is still served` / `no evidence that it is gone` | The sitemap omitted a page that exists | Do not delete it; the sitemap is the problem |
| `N% of <host>'s pages are missing from its sitemap` | Looks like a sitemap fault | Look at the sitemap before accepting |
| A gate line says FAIL after an apply | New content outranks a canary's source | Read the lines; roll back local, or fix the canary if the page really moved |
| `Failed verification: a text or a vector changed` after a conversion | Should be impossible: the conversion sends no text or vector | Restore the snapshot it printed and report it |
| Preflight warns `weaviate versions` | Hosted upgraded; local, the rehearsal copy and the client are pinned behind it | Follow section 7a before bumping anything; a rehearsal against a different version is not an exact copy |
| `Refused: ... identity would be a guess` | The registry address is not a watch, youtu.be or embed link | Register it as `https://www.youtube.com/watch?v=<id>` and push |
| `Refused: ... are the same video` | Two registry entries for one video | Remove one and push |
| `a video has no manual captions and GEMINI_API_KEY is unset` | Nothing can transcribe that video | Set the key; the preflight's `gemini key` line says whether it is accepted |
| `VISION_MODEL is not set` | The lane will not store a video without frame descriptions | Set it, or pass `ALLOW_NO_VISION=1` knowing it is permanent |
| A video fails with a vision error | A frame could not be described; nothing of that video was stored | Run again; check the Gemini quota first |
| Firecrawl fails or credits run out | The fetch stops | Failed pages are listed; wait or top up, then build a new plan |

## 7. Routine

- **Weekly, or when someone says docs changed:** `make sync-preflight`, then
  `make sync-inventory`. Usually there is nothing to do.
- **After any apply or promotion:** the preflight, and the three canary questions.
- **Backups:** `make snapshot-backup TARGET=cloud` when the newest is over a week old. The
  promotion takes its own before and after.
- **Docker:** keep its disk under 80%; `docker builder prune` is the safe lever.
- **Hosted is on the free plan:** it is paused for inactivity and deleted after 30 idle
  days. It paused after about six days without traffic on 2026-11-23 and the app returned
  503 until it was woken in the Weaviate Cloud console (OPS-007). The data survived. When
  the app or the preflight reports `503 no healthy upstream`, check the console first. A
  keep-alive is not built yet, so do not assume daily use.

### 7a. When the hosted Weaviate version changes

The free cluster upgrades without notice (1.39.13 to 1.40.5 when it was woken on
2026-11-23), while local Docker, the rehearsal copy and the app's `weaviate-client` are
pinned by us. The preflight line `weaviate versions` warns about it. The app kept working
through the drift, but a bump must be tested first:

1. Read the new minor version's release notes for changes to collection creation, vector
   or index defaults and deprecated endpoints. 1.40 changed the vector defaults; we are
   not affected because every collection we create sets `Configure.Vectors.self_provided`.
2. Check that our code works on the new version: `make weaviate-compat VERSION=<x.y.z>`.
   It starts a throwaway server on spare ports and runs `ensure_collection`, a restore of
   the newest hosted backup with the `hfresh` index, an equality check of every object,
   a nearest-neighbour search, `update_metadata`, `put_objects` batches and the app's
   startup check, then removes the server. It refuses the local and rehearsal ports and
   a target that already holds the collection. Run it again with `CLIENT=<x.y.z>` for
   the `weaviate-client` that Weaviate's release table pairs with that version (an
   isolated environment; the lockfile is untouched). `SNAPSHOT_DIR=` picks another backup.
3. Test the in-place upgrade on a **copy** of the rehearsal volume:
   `make weaviate-compat VERSION=<x.y.z> UPGRADE=1`. It briefly stops the rehearsal
   container, copies its volume, opens the copy with the new version, compares object
   count, texts and vectors, and restarts the rehearsal. Weaviate cannot downgrade a data
   directory, so never try this on the real volume first.
4. Only then take `make snapshot-export`, change both images in `compose.yaml`, restart,
   and confirm the preflight shows the versions equal.
5. Bump `weaviate-client`, run the tests, and deploy with `--no-traffic --tag`; check the
   revision became ready, ask one real question through the tag URL, then shift traffic.

State on 2026-11-23: steps 1 to 3 passed for 1.40.5 (every check with client 4.20.4 and
4.24.1; a copy of the data opened with identical contents). Steps 4 and 5 are not done.
`make weaviate-compat VERSION=1.28.4` fails at the restore, as it should: that version
predates the `hfresh` index.

## 8. Not covered yet

- **Changing a video that is already stored** (a new cut at the same address). Not built
  (SYNC-011, second half). A video with any stored chunk is never looked at again, so
  re-recording one needs the fingerprint work. Adding a new video is section 3b.
- **Screenshot descriptions** (SYNC-010) are never touched by the web lane.
- **Unattended runs** (SYNC-009) are not allowed. Removals will always need a person.
- **Operations:** spend caps, abuse limits, alerting and a dedicated service account (OPS-001
  to OPS-008).

## 9. Where things live

| What | Where |
|---|---|
| Source registry, page ledger, locks, backups, run records (the bucket leads; see the start of this runbook) | The private bucket `multimodal-rag-state-gen-lang-client-0552956404` (`registry/`, `ledger/`, `locks/`, `snapshots/`, `runs/`) |
| Fetched pages, plans, local snapshots | `runs/` in the repo (git-ignored) |
| Limits | `config/sync-policy.yaml` |
| Canary questions | `config/canaries.yaml` |
| Keys and the hosted address | `.env` and Secret Manager (never in Git) |

## 10. Command reference

| Command | Reads | Writes | Exit |
|---|---|---|---|
| `make sync-preflight` | Everything | Nothing | 1 on a FAIL |
| `make sync-inventory` | Sitemaps, ledger, local | Nothing (`INIT=1` creates the ledger once) | 1 on error |
| `make sync-plan` | Sitemaps, pages, local | `runs/`, the bucket | 10 needs review |
| `make sync-apply RUN=` | The plan | **Local**, ledger | 10 needs review, 1 failed or gate failed |
| `make sync-convert` | Local, sitemaps, ledger | **Local properties** (`DRY=1` writes nothing) | 1 on failed verification |
| `make weaviate-compat` | A throwaway Weaviate, the newest hosted backup | Nothing outside its own container (`UPGRADE=1` stops the rehearsal briefly) | 1 on a failed check |
| `make sync-videos` | Registry, local, YouTube metadata | **Local** (`DRY=1` writes nothing; `ROLLBACK=` deletes what a run added) | 1 on a refusal, a failed video or a failed gate |
| `make promote-plan` | Local, hosted | `runs/` | 10 needs review |
| `make promote RUN=` | The plan | **Hosted**, backups | 10 needs review, 1 failed |
| `make promote-rollback` | A snapshot | **Hosted** or rehearsal | 1 if any difference remains |
| `make sync-gate` | Both | Nothing | 1 on a FAIL |
| `make sync-reembed` | Local | **Local vectors** (`DRY=1` writes nothing) | 1 on a problem |
| `make snapshot-backup TARGET=cloud` | Hosted | The bucket | 1 on error |
