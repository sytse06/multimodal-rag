"""Promote the local collection to a target: copy it verbatim, vectors included.

Local is the reviewed state and the target (hosted, or a rehearsal copy) is brought to
equal it. Nothing is embedded: every object is read from local and written by explicit
UUID with its own vector, so both sides end up byte for byte the same. The plan names
what will change and is bound to the exact state of both sides by a digest, so a plan
cannot be applied to a collection that moved after it was reviewed.

The write order limits the time the target holds a mix of old and new vectors: one
canary object first (read back and compared), then every write in batches, then the
deletes last. A run that stops part-way leaves the target changed, so the plan no longer
matches; build a new plan, which is the remaining difference, and apply that.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, Field

from multimodal_rag.checks import Check, Status
from multimodal_rag.retrieval_gate import Canary
from multimodal_rag.store.snapshot import SnapshotManifest
from multimodal_rag.sync.apply import ApplyRefused, LockLike
from multimodal_rag.sync.policy import Policy, PolicyResult

PROMOTE_VERSION = 1
VECTOR_SAME = 0.999999  # the same vector read from two servers
BATCH = 100

Row = tuple[str, dict[str, Any], list[float]]


class PromoteFailed(RuntimeError):
    """The promotion started and a check failed; the target may be partly changed."""


class Side(Protocol):
    def read_objects(self) -> list[Row]: ...


class Target(Side, Protocol):
    def put_objects(self, objects: list[Row]) -> int: ...
    def missing_properties(self) -> list[str]: ...
    def extend_schema(self) -> list[str]: ...
    def delete_chunks(self, uuids: list[UUID]) -> int: ...


@dataclass(frozen=True)
class Obj:
    uuid: str
    properties: dict[str, Any]
    vector: list[float]


def normalise(properties: dict[str, Any]) -> dict[str, Any]:
    """Properties as comparable values: dates in UTC, so two servers agree."""
    out = {}
    for key, value in properties.items():
        if isinstance(value, datetime):
            value = value.astimezone(timezone.utc).isoformat()
        out[key] = value
    return out


def to_objects(rows: Iterable[Row]) -> dict[str, Obj]:
    return {u: Obj(u, normalise(p), v) for u, p, v in rows}


def object_digest(objects: dict[str, Obj]) -> str:
    """One hash of every object: id, properties and vector, in id order."""
    digest = hashlib.sha256()
    for uuid in sorted(objects):
        obj = objects[uuid]
        vector = ",".join(f"{x:.7g}" for x in obj.vector)
        props = json.dumps(obj.properties, sort_keys=True)
        digest.update(f"{uuid}|{props}|{vector}\n".encode())
    return digest.hexdigest()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


class PromotePlan(BaseModel):
    version: int = PROMOTE_VERSION
    target: str
    local_digest: str
    target_digest: str
    local_count: int
    target_count: int
    embedding: dict[str, str] = Field(default_factory=dict)
    add: list[str] = Field(default_factory=list)
    replace: list[str] = Field(default_factory=list)  # properties differ
    vector: list[str] = Field(default_factory=list)  # only the vector differs
    delete: list[str] = Field(default_factory=list)
    unchanged: int = 0
    missing_properties: list[str] = Field(default_factory=list)  # the target lacks them
    policy: PolicyResult = Field(default_factory=lambda: PolicyResult(status="within"))
    plan_hash: str = ""

    def writes(self) -> list[str]:
        """Everything written, vector refreshes first.

        They make the old content consistent with the runtime, which is a better state
        to stop in than any other.
        """
        return self.vector + self.replace + self.add


def canonical_json(plan: PromotePlan) -> str:
    body = plan.model_dump(mode="json", exclude={"plan_hash"})
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def compute_hash(plan: PromotePlan) -> str:
    return hashlib.sha256(canonical_json(plan).encode()).hexdigest()


def hash_is_valid(plan: PromotePlan) -> bool:
    return bool(plan.plan_hash) and plan.plan_hash == compute_hash(plan)


def build_plan(
    local: dict[str, Obj],
    target: dict[str, Obj],
    *,
    target_name: str,
    policy: Policy,
    embedding: dict[str, str] | None = None,
    missing_properties: list[str] | None = None,
) -> PromotePlan:
    plan = PromotePlan(
        target=target_name,
        local_digest=object_digest(local),
        target_digest=object_digest(target),
        local_count=len(local),
        target_count=len(target),
        embedding=embedding or {},
        add=sorted(local.keys() - target.keys()),
        delete=sorted(target.keys() - local.keys()),
        missing_properties=list(missing_properties or []),
    )
    for uuid in sorted(local.keys() & target.keys()):
        mine, theirs = local[uuid], target[uuid]
        if mine.properties != theirs.properties:
            plan.replace.append(uuid)
        elif cosine(mine.vector, theirs.vector) < VECTOR_SAME:
            plan.vector.append(uuid)
        else:
            plan.unchanged += 1
    plan.policy = evaluate(plan, policy)
    plan.plan_hash = compute_hash(plan)
    return plan


def evaluate(plan: PromotePlan, policy: Policy) -> PolicyResult:
    reasons = []
    share = len(plan.delete) / plan.target_count if plan.target_count else 0.0
    if share > policy.max_deleted_chunk_share:
        limit = policy.max_deleted_chunk_share
        reasons.append(f"deleting {share:.0%} of the target exceeds {limit:.0%}")
    if len(plan.vector) > policy.max_vector_only_updates:
        reasons.append(
            f"{len(plan.vector)} objects differ only in their vector (limit"
            f" {policy.max_vector_only_updates}): expected after a re-embed or a"
            " runtime change, not in a normal promotion"
        )
    if plan.missing_properties:
        reasons.append(
            f"adds {', '.join(plan.missing_properties)} to the {plan.target} schema;"
            " Weaviate cannot remove a property again"
        )
    return PolicyResult(status="needs_review" if reasons else "within", reasons=reasons)


@dataclass
class PromoteDeps:
    local: Side
    target: Target
    lock: LockLike
    backup: Callable[[str], str]  # label -> where the backup is
    gate: Callable[[list[Canary]], list[Check]]
    embedding_check: Callable[[], tuple[dict[str, str], list[str]]]
    canaries: list[Canary] = field(default_factory=list)
    policy: Policy = field(default_factory=Policy)


@dataclass
class PromoteOptions:
    accept_review: bool = False
    break_lock: bool = False
    take_backup: bool = True
    canary_only: bool = False
    run_gate: bool = True


@dataclass
class PromoteResult:
    status: str  # promoted, canary-ok, gate-failed or nothing-to-do
    written: int = 0
    deleted: int = 0
    schema_added: list[str] = field(default_factory=list)
    backup_before: str | None = None
    backup_after: str | None = None
    canary: str | None = None
    gate: list[Check] = field(default_factory=list)


def _row(obj: Obj) -> Row:
    return (obj.uuid, obj.properties, obj.vector)


def differences(expected: dict[str, Obj], found: dict[str, Obj]) -> list[str]:
    """How `found` differs from `expected`, every object and vector compared."""
    problems = []
    for uuid in sorted(expected.keys() - found.keys()):
        problems.append(f"{uuid}: missing")
    for uuid in sorted(found.keys() - expected.keys()):
        problems.append(f"{uuid}: should not exist")
    for uuid in sorted(expected.keys() & found.keys()):
        want, got = expected[uuid], found[uuid]
        if want.properties != got.properties:
            problems.append(f"{uuid}: properties differ")
        elif cosine(want.vector, got.vector) < VECTOR_SAME:
            problems.append(f"{uuid}: vector differs")
    return problems


def promote(
    plan: PromotePlan, deps: PromoteDeps, options: PromoteOptions
) -> PromoteResult:
    if plan.version != PROMOTE_VERSION or not hash_is_valid(plan):
        raise ApplyRefused(["the promotion plan does not match its hash or format"])
    fingerprint, embedding_problems = deps.embedding_check()
    reasons = list(embedding_problems)
    if plan.embedding and fingerprint and plan.embedding != fingerprint:
        reasons.append("the embedding runtime changed since the plan was made")
    if reasons:
        raise ApplyRefused(reasons)
    local = to_objects(deps.local.read_objects())
    if object_digest(local) != plan.local_digest:
        raise ApplyRefused(["local changed after the plan was made; build a new plan"])
    if (review := evaluate(plan, deps.policy).reasons) and not options.accept_review:
        raise ApplyRefused(review, needs_review=True)
    deps.lock.acquire(break_lock=options.break_lock)
    try:
        before = to_objects(deps.target.read_objects())
        if object_digest(before) != plan.target_digest:
            raise ApplyRefused(
                [f"the {plan.target} changed after the plan was made; build a new plan"]
            )
        result = PromoteResult(status="nothing-to-do")
        if not (plan.writes() or plan.delete):
            return result
        if options.take_backup:
            result.backup_before = deps.backup("before")
        if plan.writes():
            result.schema_added = deps.target.extend_schema()
        writes = [_row(local[u]) for u in plan.writes()]
        first, rest = writes[:1], writes[1:]
        if first:
            result.canary = first[0][0]
            deps.target.put_objects(first)
            canary = result.canary
            landed = {
                u: o for u, o in to_objects(deps.target.read_objects()).items()
                if u == canary
            }
            bad = differences({canary: local[canary]}, landed)
            if bad:
                raise PromoteFailed(f"the canary write did not read back: {bad}")
            result.written = 1
            if options.canary_only:
                result.status = "canary-ok"
                return result
        for start in range(0, len(rest), BATCH):
            result.written += deps.target.put_objects(rest[start : start + BATCH])
        if plan.delete:
            result.deleted = deps.target.delete_chunks([UUID(u) for u in plan.delete])
        problems = differences(local, to_objects(deps.target.read_objects()))
        if problems:
            raise PromoteFailed(
                f"{len(problems)} objects differ after the write: {problems[:5]}"
            )
        result.status = "promoted"
        if options.run_gate:
            result.gate = deps.gate(deps.canaries)
            if any(c.status is Status.FAIL for c in result.gate):
                result.status = "gate-failed"
        if options.take_backup:
            result.backup_after = deps.backup("after")
        return result
    finally:
        deps.lock.release()


def load_snapshot(directory: Path) -> dict[str, Obj]:
    """The objects of a snapshot written by `export_snapshot`, checksum verified."""
    manifest = SnapshotManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    path = directory / manifest.snapshot_file
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.snapshot_sha256:
        raise ValueError("the snapshot does not match its checksum")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        vector = record["vector"]
        vector = vector["default"] if isinstance(vector, dict) else vector
        rows.append((record["uuid"], record["properties"], vector))
    return to_objects(rows)


def rollback(
    snapshot: dict[str, Obj], target: Target, lock: LockLike
) -> list[str]:
    """Put the target back exactly as a snapshot had it, in place: no recreate.

    Every snapshot object is written back verbatim, then whatever the snapshot did not
    contain is deleted. Returns the remaining differences (empty when it matches).
    """
    lock.acquire()
    try:
        target.put_objects([_row(o) for o in snapshot.values()])
        extra = to_objects(target.read_objects()).keys() - snapshot.keys()
        if extra:
            target.delete_chunks([UUID(u) for u in sorted(extra)])
        return differences(snapshot, to_objects(target.read_objects()))
    finally:
        lock.release()
