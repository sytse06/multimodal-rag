"""Does our code work on a given Weaviate version? Runs against a throwaway server.

Hosted Weaviate upgrades without notice, so before bumping local Docker or the client we
start a throwaway server of the new version on spare ports and run the code paths that
production uses: create the collection, restore a backup with the ``hfresh`` index, read
every object back, search, update metadata, write in batches and run the app's startup
check. ``--upgrade`` also opens a *copy* of an existing data volume with the new
version.

It never touches the real local or rehearsal servers: it refuses their ports, refuses a
target that already holds the collection, and removes its container and scratch volume.
"""

import argparse
import contextlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import weaviate
from langchain_core.embeddings import Embeddings

from multimodal_rag.checks import Check, Status
from multimodal_rag.preflight import format_report
from multimodal_rag.store.restore import restore_snapshot
from multimodal_rag.store.weaviate import (
    COLLECTION_NAME,
    EXPECTED_PROPERTIES,
    OPTIONAL_PROPERTIES,
    WeaviateStore,
)
from multimodal_rag.sync.convert import checksum
from multimodal_rag.sync.promote import differences, load_snapshot, to_objects

IMAGE = "cr.weaviate.io/semitechnologies/weaviate"
# Local Docker and the rehearsal copy: a compat run must never be pointed at these.
PROTECTED_PORTS = {8080, 8081, 50051, 50052}
ENVIRONMENT = {
    "QUERY_DEFAULTS_LIMIT": "25",
    "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
    "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
    "CLUSTER_HOSTNAME": "node1",
}
REHEARSAL_VOLUME = "multimodal-rag_weaviate_rehearsal_data"
VECTOR_SIZE = 768

Row = tuple[str, dict[str, Any], list[float]]


class Refused(Exception):
    """The run was refused before it could touch anything it should not."""


class FixedEmbeddings(Embeddings):
    """A constant vector: the checks exercise Weaviate, not an embedding model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * VECTOR_SIZE for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * VECTOR_SIZE


def newest_snapshot(runs: Path) -> Path:
    """The newest hosted backup a promotion left in ``runs/``."""
    found = sorted(runs.glob("promote-hosted-*/snapshot-hosted-after"))
    if not found:
        raise FileNotFoundError(
            f"no promote-hosted-*/snapshot-hosted-after under {runs}; "
            "pass --snapshot-dir (make snapshot-backup TARGET=cloud, then download it)"
        )
    return found[-1]


def container_command(
    version: str,
    name: str,
    http_port: int,
    grpc_port: int,
    volume: str | None = None,
) -> list[str]:
    command = ["docker", "run", "-d", "--name", name]
    command += ["-p", f"127.0.0.1:{http_port}:8080"]
    command += ["-p", f"127.0.0.1:{grpc_port}:50051"]
    if volume:
        command += ["-v", f"{volume}:/var/lib/weaviate"]
    for key, value in ENVIRONMENT.items():
        command += ["-e", f"{key}={value}"]
    return [*command, f"{IMAGE}:{version}"]


def _docker(args: list[str], *, check: bool = True) -> str:
    done = subprocess.run(["docker", *args], capture_output=True, text=True)
    if check and done.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:2])}: {done.stderr.strip()[:300]}")
    return done.stdout.strip()


def _ready(http_port: int, seconds: int = 90) -> bool:
    url = f"http://127.0.0.1:{http_port}/v1/.well-known/ready"
    for _ in range(seconds):
        with contextlib.suppress(urllib.error.URLError, OSError):
            if urllib.request.urlopen(url, timeout=2).status == 200:
                return True
        time.sleep(1)
    return False


@contextlib.contextmanager
def server(
    version: str, http_port: int, grpc_port: int, volume: str | None = None
) -> Iterator[str]:
    """A throwaway Weaviate of ``version``; yields the container name and removes it."""
    if {http_port, grpc_port} & PROTECTED_PORTS:
        raise Refused(f"ports {http_port}/{grpc_port} belong to local or the rehearsal")
    name = f"wv-compat-{http_port}"
    _docker(["rm", "-f", name], check=False)  # a leftover of a crashed run
    _docker(container_command(version, name, http_port, grpc_port, volume)[1:])
    try:
        if not _ready(http_port):
            raise RuntimeError(f"Weaviate {version} did not become ready")
        yield name
    finally:
        _docker(["rm", "-f", name], check=False)


def _step(name: str, run: Callable[[], str]) -> Check:
    try:
        return Check(name, Status.OK, run())
    except Exception as exc:  # a failed step is a result, not a crash
        return Check(name, Status.FAIL, f"{type(exc).__name__}: {str(exc)[:200]}")


def run_checks(store: WeaviateStore, snapshot_dir: Path) -> list[Check]:
    """The code paths production uses, on a fresh server. Refuses a used target."""
    client = store.client
    if client.collections.exists(COLLECTION_NAME):
        raise Refused(
            f"the target already holds {COLLECTION_NAME}; compat only runs against a "
            "fresh throwaway server"
        )
    checks: list[Check] = []
    wanted = {**EXPECTED_PROPERTIES, **OPTIONAL_PROPERTIES}

    def create() -> str:
        store.ensure_collection()
        config = client.collections.get(COLLECTION_NAME).config.get()
        types = {str(p.name): str(p.data_type.value) for p in config.properties}
        if types != wanted:
            raise ValueError(f"properties differ: {sorted(set(types) ^ set(wanted))}")
        if not config.vector_config or "default" not in config.vector_config:
            raise ValueError("no self-provided vector was configured")
        client.collections.delete(COLLECTION_NAME)
        return f"{len(types)} properties with their types, a self-provided vector"

    checks.append(_step("create the collection", create))

    snapshot = load_snapshot(snapshot_dir)
    restored = _step(
        "restore the backup with the hfresh index",
        lambda: (
            restore_snapshot(client, snapshot_dir, replace=True, vector_index="hfresh"),
            f"{len(snapshot)} objects",
        )[1],
    )
    checks.append(restored)
    if restored.status is not Status.OK:
        for name in ("objects equal the backup", "search", "update_metadata",
                     "put_objects", "startup check"):
            checks.append(Check(name, Status.FAIL, "not run: the restore failed"))
        return checks

    collection = client.collections.get(COLLECTION_NAME)
    rows: list[Row] = []

    def read_back() -> str:
        for obj in collection.iterator(include_vector=True):
            rows.append((str(obj.uuid), dict(obj.properties), _vector(obj)))
        problems = differences(snapshot, to_objects(rows))
        if problems:
            raise ValueError(f"{len(problems)} differ, first: {problems[0]}")
        return f"{len(rows)} objects, properties and vectors equal"

    checks.append(_step("objects equal the backup", read_back))
    first = rows[0] if rows else None

    def search() -> str:
        assert first is not None
        hit = collection.query.near_vector(near_vector=first[2], limit=3)
        if str(hit.objects[0].uuid) != first[0]:
            raise ValueError("an object was not its own nearest neighbour")
        return "an object is its own nearest neighbour"

    def update() -> str:
        assert first is not None
        store.update_metadata({first[0]: {"embedded_video_ids": ["compat"]}})
        after = collection.query.fetch_object_by_id(first[0], include_vector=True)
        if _vector(after) != first[2]:
            raise ValueError("the vector changed")
        return "the vector is untouched"

    def batches() -> str:
        sample = rows[:150]
        written = store.put_objects(sample[:100]) + store.put_objects(sample[100:])
        count = sum(1 for _ in collection.iterator())
        if written != len(sample) or count != len(rows):
            raise ValueError(f"wrote {written} of {len(sample)}, holds {count}")
        return f"{written} objects in batches, none lost or duplicated"

    def startup() -> str:
        store.validate_compatibility()
        return "the app's startup check passes"

    for name, run in (("search", search), ("update_metadata", update),
                      ("put_objects", batches), ("startup check", startup)):
        checks.append(_step(name, run))
    return checks


def _vector(obj: Any) -> list[float]:
    vector = obj.vector["default"] if isinstance(obj.vector, dict) else obj.vector
    return list(vector)


def _rows(http_port: int, grpc_port: int) -> list[Row]:
    client = weaviate.connect_to_local(port=http_port, grpc_port=grpc_port)
    try:
        out = []
        collection = client.collections.get(COLLECTION_NAME)
        for obj in collection.iterator(include_vector=True):
            out.append((str(obj.uuid), dict(obj.properties), _vector(obj)))
        return out
    finally:
        client.close()


def upgrade_checks(
    version: str,
    volume: str,
    source_http: int,
    source_grpc: int,
    http_port: int,
    grpc_port: int,
) -> list[Check]:
    """Open a COPY of an existing data volume with the new version and compare."""
    before = _rows(source_http, source_grpc)
    scratch = f"wv-compat-copy-{int(time.time())}"
    holders = _docker(["ps", "-q", "--filter", f"volume={volume}"]).split()
    stopped: list[str] = []
    try:
        for container in holders:  # a consistent copy needs the source quiet
            _docker(["stop", container])
            stopped.append(container)
        _docker(["volume", "create", scratch])
        _docker(
            ["run", "--rm", "--entrypoint", "sh", "-v", f"{volume}:/from:ro",
             "-v", f"{scratch}:/to", f"{IMAGE}:{version}", "-c", "cp -a /from/. /to/"]
        )
    finally:
        for container in stopped:
            _docker(["start", container], check=False)
    checks: list[Check] = []
    try:
        with server(version, http_port, grpc_port, volume=scratch) as name:
            after = _rows(http_port, grpc_port)

            def same_count() -> str:
                if len(after) != len(before):
                    raise ValueError(f"{len(before)} objects became {len(after)}")
                return f"{len(after)} objects"

            def same_content() -> str:
                if checksum(before) != checksum(after):
                    raise ValueError("a text or a vector changed")
                return "checksums equal"

            def quiet_log() -> str:
                words = ("fatal", "panic", "migrat")
                lines = _docker(["logs", name], check=False).splitlines()
                bad = [ln for ln in lines if any(w in ln.lower() for w in words)]
                if bad:
                    raise ValueError(bad[0][:160])
                return "none"

            checks.append(_step("the copy opens with the same objects", same_count))
            checks.append(_step("texts and vectors are identical", same_content))
            checks.append(_step("no migration, fatal or panic in the log", quiet_log))
    finally:
        _docker(["volume", "rm", "-f", scratch], check=False)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="e.g. 1.40.5")
    parser.add_argument("--snapshot-dir", type=Path, help="default: newest in runs/")
    parser.add_argument("--http-port", type=int, default=8082)
    parser.add_argument("--grpc-port", type=int, default=50053)
    parser.add_argument("--upgrade", action="store_true", help="also test a data copy")
    parser.add_argument("--volume", default=REHEARSAL_VOLUME)
    parser.add_argument("--source-http", type=int, default=8081)
    parser.add_argument("--source-grpc", type=int, default=50052)
    args = parser.parse_args()
    try:
        snapshot = args.snapshot_dir or newest_snapshot(Path("runs"))
        print(f"weaviate-client {package_version('weaviate-client')}, "
              f"server {args.version}, backup {snapshot}")
        with server(args.version, args.http_port, args.grpc_port):
            store = WeaviateStore(
                weaviate_url=f"http://localhost:{args.http_port}",
                embeddings=FixedEmbeddings(),
                grpc_port=args.grpc_port,
            )
            try:
                checks = run_checks(store, snapshot)
            finally:
                store.close()
        if args.upgrade:
            checks += upgrade_checks(
                args.version, args.volume, args.source_http, args.source_grpc,
                args.http_port, args.grpc_port,
            )
    except (Refused, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(format_report(checks))
    sys.exit(1 if any(c.status is Status.FAIL for c in checks) else 0)


if __name__ == "__main__":
    main()
