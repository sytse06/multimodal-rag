---
name: release
description: Full release routine for multimodal-rag — pre-flight checks, epic status update, CHANGELOG entry, version bump, tag and push, Cloud Run deploy, revision ledger row, GitHub release, and a post-deploy health check. Run whenever shipping a new version. Argument: version number (e.g. 0.3.0).
argument-hint: [version]
disable-model-invocation: true
---

# Release

Execute the routine in order. Do not skip steps. Do not ask for confirmation between steps unless a step fails or a step below says to ask. Report at the end.

A release is a **code** release: the app, query path or generator reaches colleagues through `make deploy`. A **content** change (`make promote`) is not a release; it gets a row in the CHANGELOG's "Content promotions" ledger and follows `docs/maintainer-runbook.md`. See Step 9.

---

## Inputs

Collect before executing anything:

1. **Version** — `$ARGUMENTS[0]` if given (e.g. `0.3.0`); otherwise read the current version from `pyproject.toml` and ask for the new one. Semver: minor for features, patch for fixes.
2. **Release title** — one line for the GitHub release.
3. **Stories shipped** — story IDs completed in this release (e.g. `QUAL-003, QUAL-004`), from `docs/epics.md`.
4. **Changelog summary** — one paragraph: what changed for the support colleague, then for the maintainer. Must end with the test count, "All N tests pass.", taken from Step 1.
5. **Changelog sections** — Added / Fixed / Changed bullets, derived from the story list and `git log <previous tag>..HEAD`. Cite exact story IDs. Ask only if genuinely ambiguous.

---

## Step 1 — Pre-flight

1. On `main`, synced: `git switch main && git fetch origin && git status -sb` must read `## main...origin/main` with no ahead or behind and no modified tracked files. Anything uncommitted blocks `make release`; find out now.
2. `make quality` — ruff and mypy clean.
3. `make test` — must pass. Capture the exact test count.
4. **Hosted Weaviate version** (`docs/gcp-deployment.txt`, "Before each deploy"): `make sync-preflight` and read the "weaviate versions" line. If hosted moved to a version the app's client was not tested on, run `make weaviate-compat VERSION=<hosted version>` before going on. This needs Docker running.
5. **Query-side release only:** if the release changes how answers are made, the calibration run (`QUAL-001`, once it exists) must show no question got worse. Say if it does not exist yet.

If any check fails, stop and report.

---

## Step 2 — Update epic status

For each shipped story, in `docs/epics.md`: update the story heading's status to `done, released vX.Y.Z (YYYY-MM-DD)` and tick or update its acceptance items that were actually verified. Update the Epic's own status line and the Summary table if the epic's state changed. Do not touch stories that are not in the shipped list.

**Commit and push this on `main` now, separately from the CHANGELOG:** `docs(epics): mark <IDs> shipped in vX.Y.Z`. `make release` refuses unless `CHANGELOG.md` is the only modified file.

---

## Step 3 — CHANGELOG entry

Read the top of `CHANGELOG.md`. If a draft `## [X.Y.Z] — unreleased` entry exists, finalise it; otherwise write a new one directly under the first `---` of the file, above the previous release. Draft it **in chat first and wait for approval** before writing it.

Format:

```
## [VERSION] — YYYY-MM-DD

SUMMARY_PARAGRAPH (ends with "All N tests pass.")

### Added
- **STORY-ID** — what

### Fixed
- ...

### Changed
- ...
```

Today's date. Omit empty sections. Cite story IDs exactly; details live in `docs/epics.md`. Write it to the file and **leave it uncommitted**: `make release` commits it.

---

## Step 4 — Version bump, commit, tag, push

```bash
make release VERSION=X.Y.Z
```

Do not hand-edit `pyproject.toml` or run `uv lock` yourself. `make release` refuses unless: on `main`; only `CHANGELOG.md` modified; `## [X.Y.Z]` present; tag not taken; nothing unpushed. It bumps `pyproject.toml`, runs `uv lock`, commits `chore: release vX.Y.Z`, creates the annotated tag and pushes both. If it refuses, read the exact reason it prints.

---

## Step 5 — Deploy

Confirm `git describe --tags --exact-match` prints `vX.Y.Z` and the tree is clean: `make deploy` builds from the working tree (`--source .`), so what is checked out is what ships. Then:

```bash
make deploy
```

It needs `GCP_PROJECT` (in `.env` or exported). It does not touch env vars or secrets (`docs/gcp-deployment.txt`, "Updating configuration after the initial deploy"). If it fails, stop and report; do not create the GitHub release.

---

## Step 6 — Record the revision

```bash
gcloud run revisions list --service=multimodal-rag --region=europe-west4 --project="$GCP_PROJECT" --limit=3
```

Add a row to the CHANGELOG's "Cloud Run revisions" ledger: revision name, created (UTC), commit (the exact tag commit, no longer inferred), image `new`, and what it carried. Commit and push as `docs(changelog): record Cloud Run revision <name> for vX.Y.Z`. This is a documentation commit after the tag; that is expected.

---

## Step 7 — GitHub release

```bash
gh release create vX.Y.Z --title "vX.Y.Z — RELEASE_TITLE" --notes "CHANGELOG_SUMMARY"
```

The notes are the summary paragraph only, not the whole entry.

---

## Step 8 — Verify the service

1. The new revision serves traffic: `gcloud run services describe multimodal-rag --region=europe-west4 --project="$GCP_PROJECT" --format='value(status.latestReadyRevisionName)'` equals the revision from Step 6.
2. No errors since the deploy: query Cloud Logging for `severity>=ERROR` on the service since the deploy time, and for `jsonPayload.event="inference" AND jsonPayload.status="error"` (the queries are in the README, "How to read logs").
3. Ask the three canary questions (`config/canaries.yaml`) in the app and check that each answers with sources and working links. If the release changed answers, also ask the question that motivated it.

Pass when the revision is serving, no new errors appear, and the canaries answer. If not, report **NEEDS ATTENTION** and the rollback: route traffic back to the previous revision (`gcloud run services update-traffic multimodal-rag --to-revisions=<previous>=100`).

---

## Step 9 — Content is separate

A release does not promote content. If hosted content changed since the last release, make sure each promotion has its row in "Content promotions" (date, change, hosted chunk count, `promote-after` backup). If this session also ran a promotion, add the row now and commit it with Step 6's commit.

---

## Done

Report concisely:

```
Released: vVERSION
Stories shipped: STORY-ID, STORY-ID
Cloud Run revision: <name>, serving: yes / NEEDS ATTENTION
Canaries: OK / NEEDS ATTENTION
GitHub release: <url from gh release create>
```
