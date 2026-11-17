"""Apply a reviewed plan to the local collection, verify it, gate it, update the ledger.

The sequence is: validate the plan against the current registry, ledger and code; take
the run lock; re-derive every chunk from the stored pages and prove it matches the plan;
check the collection still looks as the plan expects; snapshot; write every changed
position, then delete; verify; run the retrieval gate; update the ledger only if all of
that held. Nothing here fetches a page, and nothing touches a chunk the plan does not
list, so screenshot and video chunks are never changed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import UUID

from multimodal_rag.checks import Check, Status
from multimodal_rag.ingest.web import NORMALISER_VERSION, chunk_page
from multimodal_rag.models.chunks import SupportChunk, WebChunk
from multimodal_rag.retrieval_gate import Canary
from multimodal_rag.sync.diff import ADDED
from multimodal_rag.sync.ledger import Ledger, PageEntry, text_chunk_id
from multimodal_rag.sync.plan import (
    DELETE,
    SKIP,
    WRITE,
    PagePlan,
    Plan,
    sha256_text,
    staleness,
)
from multimodal_rag.sync.policy import PlanStats, Policy, evaluate

VECTOR_DIMENSION = 768


class ApplyRefused(RuntimeError):
    """The plan must not be applied; nothing was written."""

    def __init__(self, reasons: list[str], *, needs_review: bool = False) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons
        self.needs_review = needs_review


class ApplyFailed(RuntimeError):
    """The apply started and something went wrong; the ledger was not updated."""


class StoreLike(Protocol):
    def add_chunks(self, chunks: list[SupportChunk]) -> int: ...
    def delete_chunks(self, uuids: list[UUID]) -> int: ...
    def validate_compatibility(self, expected_vector_dimension: int = 768) -> None: ...


class LockLike(Protocol):
    def acquire(self, *, break_lock: bool = False) -> None: ...
    def release(self) -> None: ...


@dataclass(frozen=True)
class PageState:
    text: dict[int, str]  # position -> text, page-text chunks only
    frozen: int  # this page's chunks that are not page text


@dataclass(frozen=True)
class Counts:
    total: int
    video: int


@dataclass
class ApplyDeps:
    store: StoreLike
    page_state: Callable[[str], PageState]
    counts: Callable[[], Counts]
    registry_generation: Callable[[], str | None]
    read_ledger: Callable[[], tuple[Ledger, str]]
    write_ledger: Callable[[Ledger, str], str]  # (ledger, if_generation) -> generation
    lock: LockLike
    snapshot: Callable[[], str]
    gate: Callable[[list[Canary]], list[Check]]
    policy: Policy
    chunk_size: int
    fixed_canaries: list[Canary] = field(default_factory=list)
    # (fingerprint of the embedding runtime, reasons the index is not consistent)
    embedding_check: Callable[[], tuple[dict[str, str], list[str]]] = lambda: ({}, [])
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


@dataclass
class ApplyOptions:
    accept_review: bool = False
    break_lock: bool = False
    take_snapshot: bool = True
    run_gate: bool = True


@dataclass
class ApplyResult:
    status: str  # applied, nothing-to-write or gate-failed
    inserted: int = 0
    replaced: int = 0
    deleted: int = 0
    pages_already_applied: int = 0
    counts_before: Counts | None = None
    counts_after: Counts | None = None
    ledger_generation: str | None = None
    snapshot: str | None = None
    gate: list[Check] = field(default_factory=list)


@dataclass(frozen=True)
class PageWork:
    page: PagePlan
    chunks: dict[int, WebChunk]  # every fresh chunk by position


def _stats(plan: Plan, ledger: Ledger) -> PlanStats:
    return PlanStats(
        pages_to_fetch=plan.totals["pages"] + plan.totals["failures"],
        new_pages=plan.totals["new_pages"],
        content_changed_pages=plan.totals["content_changed_pages"],
        deleted_chunks=plan.totals["positions_to_delete"],
        text_chunks_total=sum(e.text_chunks for e in ledger.pages.values()),
        fetch_failures=plan.totals["failures"],
        template_needs_review=bool(plan.template.get("needs_review")),
        registry_stale=False,
    )


def validate(
    plan: Plan, deps: ApplyDeps, options: ApplyOptions
) -> tuple[Ledger, str]:
    ledger, generation = deps.read_ledger()
    reasons = staleness(plan, deps.registry_generation(), generation)
    if plan.normaliser_version != NORMALISER_VERSION:
        reasons.append("the normaliser changed since the plan was made")
    if plan.chunk_size != deps.chunk_size:
        reasons.append("the chunk size changed since the plan was made")
    fingerprint, embedding_problems = deps.embedding_check()
    reasons += embedding_problems
    if ledger.embedding and fingerprint and ledger.embedding != fingerprint:
        reasons.append(
            "the embedding runtime changed since the ledger was last verified;"
            " re-embed with the new runtime first"
        )
    if reasons:
        raise ApplyRefused(reasons)
    review = list(plan.policy.reasons)
    for reason in evaluate(deps.policy, _stats(plan, ledger)).reasons:
        if reason not in review:
            review.append(reason)
    if review and not options.accept_review:
        raise ApplyRefused(review, needs_review=True)
    return ledger, generation


def prepare(plan: Plan, run_dir: Path) -> list[PageWork]:
    """Re-derive every chunk from the stored pages and prove it matches the plan."""
    problems: list[str] = []
    works = []
    for page in plan.pages:
        path = run_dir / "pages" / page.page_file
        markdown = path.read_text(encoding="utf-8") if path.exists() else None
        if markdown is None or sha256_text(markdown) != page.page_sha256:
            problems.append(f"{page.url}: the stored page does not match its checksum")
            continue
        fresh = chunk_page(markdown, page.url, page.kb_name, plan.chunk_size)
        chunks = {c.chunk_index: c for c in fresh if c.chunk_index is not None}
        if len(chunks) != page.fresh_chunks:
            problems.append(f"{page.url}: re-chunking gives a different chunk count")
            continue
        for position in page.positions:
            if position.uuid != str(text_chunk_id(page.url, position.index)):
                problems.append(f"{page.url}: position {position.index} has a wrong id")
            elif position.action == WRITE and (
                position.index not in chunks
                or sha256_text(chunks[position.index].text) != position.text_sha256
            ):
                problems.append(f"{page.url}: position {position.index} text differs")
        works.append(PageWork(page, chunks))
    if problems:
        raise ApplyRefused(problems)
    return works


def local_states(
    works: list[PageWork], deps: ApplyDeps
) -> tuple[list[PageWork], dict[str, int]]:
    """Return the pages still to write, and each page's frozen-chunk count.

    A page whose stored text already equals the plan was applied by an earlier run that
    did not finish, and is skipped. Anything else that differs means something else
    wrote to the collection after the plan was made, and the apply refuses.
    """
    pending, frozen, problems = [], {}, []
    for work in works:
        page, state = work.page, deps.page_state(work.page.url)
        frozen[page.url] = state.frozen
        fresh = {i: c.text for i, c in work.chunks.items()}
        before = {p.index for p in page.positions if p.status != ADDED}
        unchanged_ok = all(
            state.text.get(p.index) == fresh.get(p.index)
            for p in page.positions
            if p.action == SKIP
        )
        if state.text == fresh:
            continue
        if set(state.text) == before and unchanged_ok:
            pending.append(work)
        else:
            problems.append(f"{page.url}: the collection changed since the plan")
    if problems:
        raise ApplyRefused(problems)
    return pending, frozen


def execute(works: list[PageWork], deps: ApplyDeps) -> tuple[int, int, int]:
    """Write every changed position first, then delete.

    Returns (inserts, replaces, deletes).
    """
    chunks, deletes, inserts = [], [], 0
    for work in works:
        for position in work.page.positions:
            if position.action == WRITE:
                support = SupportChunk.from_web_chunk(work.chunks[position.index])
                if str(support.chunk_id) != position.uuid:
                    raise ApplyFailed(f"{work.page.url}: chunk id differs from plan")
                chunks.append(support)
                inserts += position.status == ADDED
            elif position.action == DELETE:
                deletes.append(UUID(position.uuid))
    deps.store.add_chunks(chunks)
    deps.store.delete_chunks(deletes)
    return inserts, len(chunks) - inserts, len(deletes)


def verify(
    works: list[PageWork],
    frozen: dict[str, int],
    before: Counts,
    deps: ApplyDeps,
    expected_delta: int,
) -> list[str]:
    problems = []
    for work in works:
        state = deps.page_state(work.page.url)
        if state.text != {i: c.text for i, c in work.chunks.items()}:
            problems.append(f"{work.page.url}: stored text differs from the plan")
        if state.frozen != frozen[work.page.url]:
            problems.append(f"{work.page.url}: screenshot chunks changed")
    after = deps.counts()
    if after.total != before.total + expected_delta:
        problems.append(
            f"collection has {after.total} chunks, expected"
            f" {before.total + expected_delta}"
        )
    if after.video != before.video:
        problems.append("video chunks changed")
    return problems


def updated_ledger(
    plan: Plan, ledger: Ledger, now: datetime, embedding: dict[str, str] | None = None
) -> Ledger:
    new = ledger.model_copy(deep=True)
    if embedding:
        new.embedding = embedding
    for page in plan.pages:
        held = ledger.pages.get(page.url)
        new.pages[page.url] = PageEntry(
            url=page.url,
            host=page.host,
            text_chunks=page.fresh_chunks,
            frozen_chunks=held.frozen_chunks if held else 0,
            ingested_at=now,
            lastmod=datetime.fromisoformat(page.lastmod) if page.lastmod else None,
            fingerprint=page.fingerprint,
        )
    return new


def apply_plan(
    plan: Plan, run_dir: Path, deps: ApplyDeps, options: ApplyOptions
) -> ApplyResult:
    ledger, ledger_generation = validate(plan, deps, options)
    deps.lock.acquire(break_lock=options.break_lock)
    try:
        works = prepare(plan, run_dir)
        deps.store.validate_compatibility(VECTOR_DIMENSION)
        pending, frozen = local_states(works, deps)
        before = deps.counts()
        result = ApplyResult(
            status="nothing-to-write",
            pages_already_applied=len(works) - len(pending),
            counts_before=before,
        )
        if pending:
            if options.take_snapshot:
                result.snapshot = deps.snapshot()
            result.inserted, result.replaced, result.deleted = execute(pending, deps)
            result.status = "applied"
        problems = verify(
            works, frozen, before, deps, result.inserted - result.deleted
        )
        if problems:
            raise ApplyFailed("; ".join(problems))
        result.counts_after = deps.counts()
        if options.run_gate:
            canaries = deps.fixed_canaries + [Canary(**c) for c in plan.canaries]
            result.gate = deps.gate(canaries)
            if any(check.status is Status.FAIL for check in result.gate):
                result.status = "gate-failed"
                return result
        result.ledger_generation = deps.write_ledger(
            updated_ledger(plan, ledger, deps.now(), deps.embedding_check()[0]),
            ledger_generation,
        )
        return result
    finally:
        deps.lock.release()
