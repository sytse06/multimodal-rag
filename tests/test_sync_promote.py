"""Tests for promoting the local collection to a target, and rolling back."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from multimodal_rag.checks import Check, Status
from multimodal_rag.retrieval_gate import Canary
from multimodal_rag.sync.apply import ApplyRefused
from multimodal_rag.sync.policy import Policy
from multimodal_rag.sync.promote import (
    PromoteDeps,
    PromoteFailed,
    PromoteOptions,
    Row,
    build_plan,
    compute_hash,
    hash_is_valid,
    load_snapshot,
    normalise,
    object_digest,
    promote,
    rollback,
    to_objects,
)

FINGERPRINT = {"ollama_version": "0.40.14", "model_digest": "d" * 64}
# The fake collections have a handful of objects, so one delete is a big share.
RELAXED = Policy(max_deleted_chunk_share=0.5, max_vector_only_updates=500)
WHEN = datetime(2026, 3, 13, 9, 0, tzinfo=timezone.utc)


def row(n: int, text: str = "t", vector: list[float] | None = None) -> Row:
    return (
        f"00000000-0000-5000-8000-{n:012d}",
        {
            "text": f"{text}{n}",
            "source_type": "web",
            "source_url": f"https://x/{n}",
            "source_name": "KB",
            "timestamp_seconds": None,
            "section_heading": "",
            "url_hash": f"h{n}",
            "ingested_at": WHEN,
        },
        vector or [1.0, float(n), 0.5],
    )


class FakeSide:
    def __init__(self, rows: list[Row]) -> None:
        self.rows = {r[0]: r for r in rows}
        self.puts: list[list[str]] = []
        self.deletes: list[list[str]] = []
        self.events: list[str] = []
        self.drop_after: int | None = None  # silently drop writes after N calls
        self.alter_first = False

    def read_objects(self) -> list[Row]:
        return list(self.rows.values())

    def put_objects(self, objects: list[Row]) -> int:
        self.events.append("put")
        self.puts.append([o[0] for o in objects])
        if self.drop_after is not None and len(self.puts) > self.drop_after:
            return len(objects)  # claims success, writes nothing
        for uuid, props, vector in objects:
            if self.alter_first and len(self.puts) == 1:
                vector = [v + 1.0 for v in vector]
            self.rows[uuid] = (uuid, dict(props), list(vector))
        return len(objects)

    def delete_chunks(self, uuids: list[UUID]) -> int:
        self.events.append("delete")
        self.deletes.append([str(u) for u in uuids])
        return sum(self.rows.pop(str(u), None) is not None for u in uuids)


class FakeLock:
    def __init__(self) -> None:
        self.events: list[str] = []

    def acquire(self, *, break_lock: bool = False) -> None:
        self.events.append("acquire")

    def release(self) -> None:
        self.events.append("release")


def deps(local: FakeSide, target: FakeSide, **extra: Any) -> PromoteDeps:
    backups: list[str] = []

    def backup(label: str) -> str:
        target.events.append(f"backup-{label}")
        backups.append(label)
        return f"gs://b/snapshots/{label}"

    base = {
        "local": local,
        "target": target,
        "lock": FakeLock(),
        "backup": backup,
        "gate": lambda canaries: [Check("canary", Status.OK, "rank 1")],
        "embedding_check": lambda: (FINGERPRINT, []),
        "canaries": [Canary(question="q?", expect=["https://x/1"])],
        "policy": RELAXED,
    }
    base.update(extra)
    return PromoteDeps(**base)  # type: ignore[arg-type]


def plan_for(local: FakeSide, target: FakeSide, policy: Policy | None = None):
    return build_plan(
        to_objects(local.read_objects()),
        to_objects(target.read_objects()),
        target_name="hosted",
        policy=policy or RELAXED,
        embedding=FINGERPRINT,
    )


def sides() -> tuple[FakeSide, FakeSide]:
    """Local re-embedded and edited; the target still holds the old state."""
    old = [row(n, vector=[0.0, float(n), 0.1]) for n in range(1, 7)]
    target = FakeSide(old + [row(90)])
    local_rows = [
        row(1, vector=[1.0, 1.0, 0.5]),
        row(2, text="edited", vector=[1.0, 2.0, 0.5]),
        row(3), row(4), row(5), row(6),
        row(10), row(11),
    ]
    return FakeSide(local_rows), target


def test_the_plan_classifies_every_object() -> None:
    local, target = sides()
    target.rows[row(5)[0]] = row(5)  # identical on both sides

    plan = plan_for(local, target)

    assert plan.add == [row(10)[0], row(11)[0]]
    assert plan.delete == [row(90)[0]]
    assert plan.replace == [row(2)[0]]
    assert set(plan.vector) == {row(n)[0] for n in (1, 3, 4, 6)}
    assert plan.unchanged == 1


def test_dates_in_different_zones_compare_equal() -> None:
    summer = WHEN.astimezone(timezone(timedelta(hours=2)))

    assert normalise({"ingested_at": WHEN}) == normalise({"ingested_at": summer})


def test_the_digest_covers_properties_and_vectors_but_not_order() -> None:
    a = to_objects([row(1), row(2)])
    b = to_objects([row(2), row(1)])

    assert object_digest(a) == object_digest(b)
    assert object_digest(a) != object_digest(to_objects([row(1), row(2, text="x")]))
    moved = to_objects([row(1), row(2, vector=[9, 9, 9])])
    assert object_digest(a) != object_digest(moved)


def test_a_changed_plan_is_refused() -> None:
    local, target = sides()
    plan = plan_for(local, target)
    plan.vector.pop()

    assert not hash_is_valid(plan)
    with pytest.raises(ApplyRefused, match="hash"):
        promote(plan, deps(local, target), PromoteOptions())


def test_the_target_ends_up_equal_to_local() -> None:
    local, target = sides()

    result = promote(plan_for(local, target), deps(local, target), PromoteOptions())

    assert result.status == "promoted"
    assert to_objects(target.read_objects()) == to_objects(local.read_objects())
    # 5 vector refreshes, 1 replace and 2 adds are written; 1 object is deleted
    assert (result.written, result.deleted) == (8, 1)


def test_one_canary_goes_first_then_batches_then_the_deletes_last() -> None:
    local, target = sides()
    plan = plan_for(local, target)

    promote(plan, deps(local, target), PromoteOptions())

    events = target.events
    last_put = max(i for i, e in enumerate(events) if e == "put")
    assert len(target.puts[0]) == 1 and target.puts[0][0] == plan.writes()[0]
    assert events.index("backup-before") < events.index("put")
    assert last_put < events.index("delete") < events.index("backup-after")


def test_vector_refreshes_come_before_replaces_and_adds() -> None:
    local, target = sides()
    plan = plan_for(local, target)

    order = plan.writes()

    assert order[: len(plan.vector)] == plan.vector
    assert order[-len(plan.add) :] == plan.add


def test_a_canary_that_does_not_read_back_stops_everything() -> None:
    local, target = sides()
    target.alter_first = True
    before = dict(target.rows)

    with pytest.raises(PromoteFailed, match="canary"):
        promote(plan_for(local, target), deps(local, target), PromoteOptions())

    assert len(target.puts) == 1
    assert target.deletes == []
    assert {u for u in target.rows} == set(before)


def test_a_write_that_silently_did_not_land_is_caught_by_the_full_check() -> None:
    local, target = sides()
    target.drop_after = 1

    with pytest.raises(PromoteFailed, match="differ after the write"):
        promote(plan_for(local, target), deps(local, target), PromoteOptions())


def test_a_local_that_moved_after_the_plan_is_refused_before_any_write() -> None:
    local, target = sides()
    plan = plan_for(local, target)
    local.rows[row(77)[0]] = row(77)

    with pytest.raises(ApplyRefused, match="local changed"):
        promote(plan, deps(local, target), PromoteOptions())

    assert target.events == []


def test_a_target_that_moved_after_the_plan_is_refused_before_any_write() -> None:
    local, target = sides()
    plan = plan_for(local, target)
    target.rows[row(88)[0]] = row(88)
    lock = FakeLock()

    with pytest.raises(ApplyRefused, match="hosted changed"):
        promote(plan, deps(local, target, lock=lock), PromoteOptions())

    assert target.puts == [] and lock.events == ["acquire", "release"]


def test_many_vector_only_updates_need_review_unless_accepted() -> None:
    local, target = sides()
    strict = Policy(max_deleted_chunk_share=0.5, max_vector_only_updates=2)
    plan = plan_for(local, target, strict)
    assert plan.policy.status == "needs_review"

    with pytest.raises(ApplyRefused, match="only in their vector") as raised:
        promote(plan, deps(local, target, policy=strict), PromoteOptions())
    assert raised.value.needs_review and target.events == []

    result = promote(
        plan, deps(local, target, policy=strict), PromoteOptions(accept_review=True)
    )
    assert result.status == "promoted"


def test_too_large_a_deletion_needs_review() -> None:
    local = FakeSide([row(1)])
    target = FakeSide([row(n) for n in range(1, 11)])
    plan = plan_for(local, target, Policy())

    assert any("deleting" in r for r in plan.policy.reasons)


def test_a_backup_is_taken_before_and_after_and_can_be_skipped() -> None:
    local, target = sides()
    result = promote(plan_for(local, target), deps(local, target), PromoteOptions())
    assert (result.backup_before, result.backup_after) == (
        "gs://b/snapshots/before",
        "gs://b/snapshots/after",
    )

    local, target = sides()
    result = promote(
        plan_for(local, target), deps(local, target), PromoteOptions(take_backup=False)
    )
    assert result.backup_before is None and "backup-before" not in target.events


def test_canary_only_stops_after_one_verified_write() -> None:
    local, target = sides()

    result = promote(
        plan_for(local, target), deps(local, target), PromoteOptions(canary_only=True)
    )

    assert result.status == "canary-ok" and result.written == 1
    assert len(target.puts) == 1 and target.deletes == []


def test_a_failed_gate_is_reported_and_the_backup_after_is_still_taken() -> None:
    local, target = sides()
    failing = lambda c: [Check("q", Status.FAIL, "gone")]  # noqa: E731

    result = promote(
        plan_for(local, target), deps(local, target, gate=failing), PromoteOptions()
    )

    assert result.status == "gate-failed" and result.backup_after


def test_a_wrong_embedding_runtime_refuses_the_promotion() -> None:
    local, target = sides()
    plan = plan_for(local, target)
    older = lambda: ({"ollama_version": "0.17"}, [])  # noqa: E731

    with pytest.raises(ApplyRefused, match="runtime changed"):
        promote(
            plan,
            deps(local, target, embedding_check=older),
            PromoteOptions(),
        )
    drifted = lambda: (FINGERPRINT, ["stored vectors no longer match"])  # noqa: E731
    with pytest.raises(ApplyRefused, match="no longer match"):
        promote(plan, deps(local, target, embedding_check=drifted), PromoteOptions())


def test_identical_collections_have_nothing_to_do() -> None:
    local = FakeSide([row(1), row(2)])
    target = FakeSide([row(1), row(2)])

    result = promote(plan_for(local, target), deps(local, target), PromoteOptions())

    assert result.status == "nothing-to-do" and target.events == []


def test_the_lock_is_released_even_when_the_promotion_fails() -> None:
    local, target = sides()
    target.alter_first = True
    lock = FakeLock()

    with pytest.raises(PromoteFailed):
        promote(
            plan_for(local, target), deps(local, target, lock=lock), PromoteOptions()
        )

    assert lock.events == ["acquire", "release"]


def test_a_half_finished_run_is_finished_by_a_new_plan() -> None:
    local, target = sides()
    first = plan_for(local, target)
    promote(first, deps(local, target), PromoteOptions(canary_only=True))

    with pytest.raises(ApplyRefused, match="hosted changed"):
        promote(first, deps(local, target), PromoteOptions())

    second = plan_for(local, target)
    assert len(second.writes()) == len(first.writes()) - 1
    assert promote(second, deps(local, target), PromoteOptions()).status == "promoted"
    assert to_objects(target.read_objects()) == to_objects(local.read_objects())


def _write_snapshot(directory: Path, rows: list[Row]) -> None:
    directory.mkdir(parents=True)
    path = directory / "SupportChunk.jsonl"
    lines = [
        json.dumps(
            {
                "uuid": u,
                "properties": {
                    k: (v.isoformat() if isinstance(v, datetime) else v)
                    for k, v in p.items()
                },
                "vector": {"default": vec},
            }
        )
        for u, p, vec in rows
    ]
    path.write_text("\n".join(lines) + "\n")
    manifest = {
        "collection_name": "SupportChunk",
        "object_count": len(rows),
        "vector_dimension": 3,
        "embedding_provider": "ollama",
        "embedding_model": "nomic-embed-text",
        "weaviate_version": "1.39.13",
        "created_at": WHEN.isoformat(),
        "source_commit": "x",
        "snapshot_file": path.name,
        "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "collection_schema": {},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_rollback_puts_the_target_back_exactly_as_the_snapshot_had_it(
    tmp_path: Path,
) -> None:
    local, target = sides()
    _write_snapshot(tmp_path / "before", target.read_objects())
    promote(plan_for(local, target), deps(local, target), PromoteOptions())
    assert row(90)[0] not in target.rows and row(10)[0] in target.rows

    left = rollback(load_snapshot(tmp_path / "before"), target, FakeLock())

    assert left == []
    assert row(90)[0] in target.rows and row(10)[0] not in target.rows
    assert to_objects(target.read_objects()) == load_snapshot(tmp_path / "before")


def test_a_snapshot_that_does_not_match_its_checksum_is_refused(tmp_path: Path) -> None:
    _write_snapshot(tmp_path / "s", [row(1)])
    (tmp_path / "s" / "SupportChunk.jsonl").write_text("tampered\n")

    with pytest.raises(ValueError, match="checksum"):
        load_snapshot(tmp_path / "s")


def test_the_plan_hash_changes_with_its_content() -> None:
    local, target = sides()
    plan = plan_for(local, target)
    original = plan.plan_hash
    plan.unchanged += 1

    assert compute_hash(plan) != original
    assert uuid4() != uuid4()
