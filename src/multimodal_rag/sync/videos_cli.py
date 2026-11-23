"""Ingest only registry videos that have no stored chunks yet (SYNC-011). Local only.

``DRY=1`` lists them with what ingesting them will cost and writes nothing. A real run
snapshots local, ingests each new video all-or-nothing, proves that nothing but the new
videos was added, records exactly which objects it added, and runs the retrieval gate.
``--rollback`` deletes exactly the objects a run added.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from multimodal_rag.checks import Status
from multimodal_rag.ingest import registry
from multimodal_rag.ingest.cookies import youtube_cookies
from multimodal_rag.ingest.transcripts import has_manual_captions
from multimodal_rag.ingest.videos import ingest_video
from multimodal_rag.models import identity
from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.llm import create_embeddings, create_vision_llm
from multimodal_rag.models.sources import YouTubeSource
from multimodal_rag.retrieval_gate import gate_stores, load_canaries, run_gate
from multimodal_rag.store.snapshot import export_snapshot
from multimodal_rag.store.weaviate import COLLECTION_NAME, WeaviateStore
from multimodal_rag.sync.embedding_check import apply_problems, inspect_embeddings
from multimodal_rag.sync.lock import LockHeld, RunLock
from multimodal_rag.sync.plan_cli import registry_state
from multimodal_rag.sync.videos import (
    Estimate,
    estimate,
    rollback_uuids,
    select_new,
    stored_video_documents,
    verify_added,
)

RUNS = Path("runs")


def duration_seconds(url: str, cookies_file: str) -> int | None:
    """The video's length from YouTube's metadata, without downloading it."""
    command = [sys.executable, "-m", "yt_dlp", "--skip-download", "--no-warnings"]
    command += ["--print", "duration"]
    if cookies_file:
        command += ["--cookies", cookies_file]
    try:
        out = subprocess.run(
            [*command, url], capture_output=True, text=True, timeout=90, check=True
        ).stdout.strip()
        return int(float(out))
    except (subprocess.SubprocessError, ValueError):
        return None


def summary(estimates: list[Estimate]) -> list[str]:
    lines = []
    for e in estimates:
        if e.seconds is None:
            lines.append(f"  {e.name}: length unknown, {e.path} path, cost unknown")
            continue
        length = f"{e.seconds // 60}:{e.seconds % 60:02d}"
        source = {
            "captions": "manual captions",
            "gemini": f"Gemini transcription ({e.gemini_minutes} min)",
            "none": "no transcript",
        }[e.transcript]
        lines.append(
            f"  {e.name}: {length}, {e.path} path, {e.vision_calls} vision calls,"
            f" {source}"
        )
    calls = sum(e.vision_calls for e in estimates)
    minutes = round(sum(e.gemini_minutes for e in estimates), 2)
    unknown = sum(1 for e in estimates if e.seconds is None)
    note = f" (incomplete: {unknown} with unknown length)" if unknown else ""
    lines.append(
        f"total: {calls} vision calls, {minutes} minutes of Gemini transcription{note}"
    )
    return lines


def rollback(run_dir: Path, settings: AppSettings, assume_yes: bool) -> None:
    record = json.loads((run_dir / "videos.json").read_text())
    documents = {v["document_id"] for v in record["planned"]}
    with WeaviateStore(
        weaviate_url=str(settings.weaviate_url),
        embeddings=create_embeddings(settings),
    ) as store:
        uuids = rollback_uuids(
            store.read_objects(), documents, record.get("new_uuids", [])
        )
        print(f"{len(uuids)} objects belong to the videos {run_dir} ingested.")
        if not uuids:
            print("Nothing to delete.")
            return
        if not assume_yes and input("Delete exactly these? [y/N] ") != "y":
            print("Nothing deleted.")
            return
        store.delete_chunks([UUID(u) for u in uuids])
        left = {u for u, _, _ in store.read_objects()} & set(uuids)
    print("rolled back" if not left else f"still present: {len(left)} objects")
    print("This changed local only; the next make promote-plan will show the deletes.")
    sys.exit(1 if left else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true", help="list and estimate only")
    parser.add_argument("--yes", action="store_true", help="do not ask first")
    parser.add_argument("--break-lock", action="store_true")
    parser.add_argument("--allow-no-vision", action="store_true")
    parser.add_argument("--rollback", type=Path, metavar="RUN_DIR")
    args = parser.parse_args()

    settings = AppSettings()
    if settings.weaviate_mode != "local":
        print("Error: the video lane writes to the local collection only.")
        sys.exit(1)
    if args.rollback:
        rollback(args.rollback, settings, args.yes)
        return

    _, stale = registry_state(settings)
    if stale:
        print("Error: the local registry is older than the bucket's; make sources-pull")
        sys.exit(1)
    remote = registry._remote(registry._url(settings), settings.gcp_project)
    unpushed = remote is None or remote[1] != registry._md5_hex(registry.SOURCES_PATH)
    if unpushed and not args.dry:
        print("Error: config/sources.yaml has edits that are not in the bucket;")
        print("push them first with make sources-push, so the registry stays true.")
        sys.exit(1)
    if unpushed:
        print("Warning: config/sources.yaml has edits that are not pushed yet.")
    videos: list[YouTubeSource] = registry.validate(registry.SOURCES_PATH).youtube
    has_vision = bool(settings.vision_model)
    if not has_vision and not args.allow_no_vision:
        print(
            "Error: VISION_MODEL is not set. A video stored without frame descriptions"
            " is never revisited; pass --allow-no-vision to accept that."
        )
        sys.exit(1)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / f"videos-{stamp}"
    embeddings = create_embeddings(settings)
    with (
        WeaviateStore(weaviate_url=str(settings.weaviate_url), embeddings=embeddings)
        as store,
        youtube_cookies(settings) as cookies,
    ):
        before = store.read_objects()
        stored, unsafe = stored_video_documents(before)
        selection = select_new(videos, stored)
        problems = unsafe + selection.refused
        if problems:
            print("Refused:\n" + "\n".join(f"  - {p}" for p in problems))
            sys.exit(1)
        print(f"{len(selection.stored)} registry videos already stored")
        if not selection.new:
            print("Nothing to ingest.")
            return
        estimates = [
            estimate(
                v,
                duration_seconds(str(v.url), cookies),
                manual_captions=has_manual_captions(str(v.url), cookies),
                has_vision=has_vision,
            )
            for v in selection.new
        ]
        gemini_key = settings.credentials.gemini_api_key.get_secret_value()
        if not gemini_key and any(e.transcript == "gemini" for e in estimates):
            print("Error: a video has no manual captions and GEMINI_API_KEY is unset.")
            sys.exit(1)
        print(f"{len(selection.new)} new videos:")
        print("\n".join(summary(estimates)))
        if args.dry:
            print("\nDry run: nothing was written.")
            return
        question = "\nIngest these into the LOCAL collection? [y/N] "
        if not args.yes and input(question) != "y":
            print("Nothing ingested.")
            return

        collection = store.client.collections.get(COLLECTION_NAME)
        embedding_problems = apply_problems(
            inspect_embeddings(settings, collection, embeddings)
        )
        if embedding_problems:
            print("Refused:\n" + "\n".join(f"  - {p}" for p in embedding_problems))
            sys.exit(1)

        lock = RunLock(settings.state_bucket, settings.gcp_project, f"videos-{stamp}")
        try:
            lock.acquire(break_lock=args.break_lock)
        except LockHeld as exc:
            print(f"Failed: {exc}")
            sys.exit(1)
        results: list[dict[str, object]] = []
        try:
            snapshot = run_dir / "snapshot-local"
            export_snapshot(
                store.client,
                snapshot,
                embedding_provider=settings.embedding_provider,
                embedding_model=settings.embedding_model,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            planned = [
                {
                    "name": v.name,
                    "url": str(v.url),
                    "document_id": identity.document_id(str(v.url)),
                }
                for v in selection.new
            ]
            (run_dir / "videos.json").write_text(
                json.dumps({"planned": planned, "snapshot": str(snapshot)}, indent=2)
            )
            vision = create_vision_llm(settings) if has_vision else None
            for i, video in enumerate(selection.new):
                if i:
                    time.sleep(2)
                entry: dict[str, object] = {
                    "name": video.name,
                    "url": str(video.url),
                    "document_id": identity.document_id(str(video.url)),
                }
                try:
                    ingested = ingest_video(
                        store, video, settings, vision, cookies, strict=True
                    )
                    if ingested.chunks == 0:
                        raise RuntimeError("no chunks were produced")
                    entry["chunks"] = ingested.chunks
                    entry["transcript_source"] = ingested.transcript_source
                    entry["transcript_model"] = ingested.transcript_model
                except Exception as exc:  # one video failing must not hide the rest
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                results.append(entry)
            after = store.read_objects()
        finally:
            lock.release()

    documents = {str(r["document_id"]) for r in results if not r.get("error")}
    found, added = verify_added(before, after, documents)
    (run_dir / "videos.json").write_text(
        json.dumps(
            {
                "planned": planned,
                "videos": results,
                "new_uuids": added,
                "snapshot": str(snapshot),
            },
            indent=2,
        )
    )
    for r in results:
        print(f"  {r['name']}: {r.get('chunks', r.get('error'))}")
    if found:
        print("\nFailed verification:\n" + "\n".join(f"  - {p}" for p in found))
        print(f"Roll back: make sync-videos ROLLBACK={run_dir}")
        sys.exit(1)

    with gate_stores(settings) as (hosted, local):
        checks = run_gate(load_canaries(), hosted, local, settings.top_k)
    for c in checks:
        print(f"  gate [{c.status.value}] {c.name}: {c.detail}")
    failed = [r for r in results if r.get("error")]
    print(f"\n{len(added)} objects added; record in {run_dir}/videos.json")
    print(f"Roll back exactly these: make sync-videos ROLLBACK={run_dir}")
    gate_failed = any(c.status is Status.FAIL for c in checks)
    sys.exit(1 if failed or gate_failed else 0)


if __name__ == "__main__":
    main()
