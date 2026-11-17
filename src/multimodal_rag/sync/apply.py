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
from typing import Any, Protocol
from uuid import UUID

from multimodal_rag.checks import Check, Status
from multimodal_rag.ingest.web import NORMALISER_VERSION, chunk_page
from multimodal_rag.models.chunks import SupportChunk, WebChunk
from multimodal_rag.retrieval_gate import Canary
from multimodal_rag.sync.classify import DROPPED
from multimodal_rag.sync.diff import ADDED
from multimodal_rag.sync.ledger import Ledger, PageEntry, text_chunk_id
from multimodal_rag.sync.plan import (
    DELETE,
    SKIP,
    WRITE,
    PagePlan,
    Plan,
    Removal,
    sha256_text,
    staleness,
)
from multimodal_rag.sync.policy import PlanStats, Policy, evaluate
from multimodal_rag.sync.probe import GONE, REDIRECT, STILL_SERVED

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
    def update_metadata(self, updates: dict[str, dict[str, Any]]) -> int: ...
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
    # Every chunk id the collection holds for a page, screenshots included.
    page_chunk_ids: Callable[[str], list[str]] = lambda url: []
    # Video document ids the stored chunks of a page already link.
    stored_videos: Callable[[str], list[str]] = lambda url: []
    # (fingerprint of the embedding runtime, reasons the index is not consistent)
    embedding_check: Callable[[], tuple[dict[str, str], list[str]]] = lambda: ({}, [])
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


@dataclass
class ApplyOptions:
    accept_review: bool = False
    break_lock: bool = False
    take_snapshot: bool = True
    run_gate: bool = True
    # Removals are deleted only when named here. A person sets these; an automated run
    # never does, so it can add and change pages but never delete one.
    confirm_removals: list[str] = field(default_factory=list)
    confirm_sources: list[str] = field(default_factory=list)


@dataclass
class ApplyResult:
    status: str  # applied, nothing-to-write or gate-failed
    inserted: int = 0
    replaced: int = 0
    deleted: int = 0
    removed_pages: int = 0
    removed_chunks: int = 0
    pending_removals: list[str] = field(default_factory=list)
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


def _stats(plan: Plan, ledger: Ledger, removed_text_chunks: int = 0) -> PlanStats:
    return PlanStats(
        pages_to_fetch=plan.totals["pages"] + plan.totals["failures"],
        new_pages=plan.totals["new_pages"],
        content_changed_pages=plan.totals["content_changed_pages"],
        deleted_chunks=plan.totals["positions_to_delete"] + removed_text_chunks,
        text_chunks_total=sum(e.text_chunks for e in ledger.pages.values()),
        fetch_failures=plan.totals["failures"],
        template_needs_review=bool(plan.template.get("needs_review")),
        registry_stale=False,
    )


def validate(
    plan: Plan, deps: ApplyDeps, options: ApplyOptions
) -> tuple[Ledger, str, list[Removal]]:
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
    removals = select_removals(plan, ledger, options)
    review = list(plan.policy.reasons)
    removed_text = sum(r.text_chunks for r in removals)
    for reason in evaluate(deps.policy, _stats(plan, ledger, removed_text)).reasons:
        if reason not in review:
            review.append(reason)
    if review and not options.accept_review:
        raise ApplyRefused(review, needs_review=True)
    return ledger, generation, removals


def _held(url: str | None, plan: Plan, ledger: Ledger) -> bool:
    return bool(url) and (url in ledger.pages or any(p.url == url for p in plan.pages))


def _removal_problem(removal: Removal, plan: Plan, ledger: Ledger) -> str | None:
    """Why a page may not be deleted by name, or None if its evidence supports it."""
    if removal.reason == DROPPED:
        return "belongs to a dropped source: confirm the source, not the page"
    if removal.evidence == STILL_SERVED:
        return f"is still served (HTTP {removal.http_status}); the sitemap omitted it"
    if removal.evidence == REDIRECT and not _held(removal.redirect_to, plan, ledger):
        return f"moved to {removal.redirect_to}, which is not held"
    if removal.evidence not in (GONE, REDIRECT):
        return "has no evidence that it is gone (not checked, or the check failed)"
    if removal.rename_of and not _held(removal.rename_of, plan, ledger):
        return f"looks renamed to {removal.rename_of}, which is not held"
    return None


def select_removals(
    plan: Plan, ledger: Ledger, options: ApplyOptions
) -> list[Removal]:
    """The removals this run may delete: named, and supported by their evidence."""
    by_url = {r.url: r for r in plan.removals}
    chosen: dict[str, Removal] = {}
    problems = []
    for url in options.confirm_removals:
        removal = by_url.get(url)
        if removal is None:
            problems.append(f"{url}: is not a removal candidate in this plan")
        elif (problem := _removal_problem(removal, plan, ledger)) is not None:
            problems.append(f"{url}: {problem}")
        else:
            chosen[url] = removal
    for host in options.confirm_sources:
        dropped = [r for r in plan.removals if r.host == host and r.reason == DROPPED]
        if not dropped:
            problems.append(f"{host}: no pages of a dropped source in this plan")
        chosen.update({r.url: r for r in dropped})
    if problems:
        raise ApplyRefused(problems)
    return list(chosen.values())


def removal_ids(removals: list[Removal], deps: ApplyDeps) -> dict[str, list[UUID]]:
    """What to delete for each confirmed page, after checking it is still the plan's.

    The chunks the collection holds must all be among those the plan listed. A page
    that already lost some of them (an unfinished run) just has fewer left to delete;
    one that gained any since the plan is refused.
    """
    out: dict[str, list[UUID]] = {}
    problems = []
    for removal in removals:
        if not removal.chunk_uuids:
            problems.append(f"{removal.url}: the plan lists no chunk ids to delete")
            continue
        current = set(deps.page_chunk_ids(removal.url))
        if not current <= set(removal.chunk_uuids):
            problems.append(f"{removal.url}: the page's chunks changed since the plan")
        else:
            out[removal.url] = sorted(UUID(c) for c in current)
    if problems:
        raise ApplyRefused(problems)
    return out


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
        lastmod = datetime.fromisoformat(page.lastmod) if page.lastmod else None
        fresh = chunk_page(
            markdown, page.url, page.kb_name, plan.chunk_size, source_lastmod=lastmod
        )
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


def execute(
    works: list[PageWork], deps: ApplyDeps, removal_uuids: list[UUID] | None = None
) -> tuple[int, int, int]:
    """Write every changed position first, then delete positions, then removed pages.

    Returns (inserts, replaces, position deletes).
    """
    chunks, deletes, inserts = [], [], 0
    refresh: dict[str, dict[str, Any]] = {}
    for work in works:
        url = work.page.url
        sample = next(iter(work.chunks.values()), None)
        # Fresh scrapes often drop video links the stored text still has, so keep both.
        fresh = set(sample.embedded_video_ids) if sample else set()
        videos = sorted(fresh | set(deps.stored_videos(url)))
        leaving: set[str] = set()
        for position in work.page.positions:
            if position.action == WRITE:
                chunk = work.chunks[position.index].model_copy(
                    update={"embedded_video_ids": videos}
                )
                support = SupportChunk.from_web_chunk(chunk)
                if str(support.chunk_id) != position.uuid:
                    raise ApplyFailed(f"{work.page.url}: chunk id differs from plan")
                chunks.append(support)
                leaving.add(position.uuid)
                inserts += position.status == ADDED
            elif position.action == DELETE:
                deletes.append(UUID(position.uuid))
                leaving.add(position.uuid)
        if sample is not None:
            # Every other chunk of the page, screenshots included, keeps its text but
            # now belongs to this version of the page.
            values: dict[str, Any] = {"embedded_video_ids": videos}
            if sample.source_lastmod is not None:
                values["source_lastmod"] = sample.source_lastmod
            for uuid in deps.page_chunk_ids(url):
                if uuid not in leaving:
                    refresh[uuid] = values
    deps.store.add_chunks(chunks)
    if refresh:
        deps.store.update_metadata(refresh)
    deps.store.delete_chunks(deletes)
    if removal_uuids:
        deps.store.delete_chunks(removal_uuids)
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
    plan: Plan,
    ledger: Ledger,
    now: datetime,
    embedding: dict[str, str] | None = None,
    removed: list[str] | None = None,
) -> Ledger:
    new = ledger.model_copy(deep=True)
    if embedding:
        new.embedding = embedding
    for url in removed or []:
        new.pages.pop(url, None)
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
    ledger, ledger_generation, removals = validate(plan, deps, options)
    deps.lock.acquire(break_lock=options.break_lock)
    try:
        works = prepare(plan, run_dir)
        deps.store.validate_compatibility(VECTOR_DIMENSION)
        pending, frozen = local_states(works, deps)
        to_remove = removal_ids(removals, deps)
        before = deps.counts()
        confirmed = {r.url for r in removals}
        result = ApplyResult(
            status="nothing-to-write",
            pages_already_applied=len(works) - len(pending),
            counts_before=before,
            pending_removals=[r.url for r in plan.removals if r.url not in confirmed],
        )
        doomed = [u for ids in to_remove.values() for u in ids]
        if pending or doomed:
            if options.take_snapshot:
                result.snapshot = deps.snapshot()
            result.inserted, result.replaced, result.deleted = execute(
                pending, deps, doomed
            )
            result.removed_pages = sum(1 for ids in to_remove.values() if ids)
            result.removed_chunks = len(doomed)
            result.status = "applied"
        problems = verify(
            works,
            frozen,
            before,
            deps,
            result.inserted - result.deleted - result.removed_chunks,
        )
        problems += [
            f"{url}: chunks of a removed page are still there"
            for url in to_remove
            if deps.page_chunk_ids(url)
        ]
        if problems:
            raise ApplyFailed("; ".join(problems))
        result.counts_after = deps.counts()
        if options.run_gate:
            canaries = deps.fixed_canaries + [Canary(**c) for c in plan.canaries]
            result.gate = deps.gate(canaries)
            if any(check.status is Status.FAIL for check in result.gate):
                result.status = "gate-failed"
                return result
        new_ledger = updated_ledger(
            plan,
            ledger,
            deps.now(),
            deps.embedding_check()[0],
            removed=list(to_remove),
        )
        if new_ledger == ledger:  # nothing to record: keep the generation as it is
            result.ledger_generation = ledger_generation
        else:
            result.ledger_generation = deps.write_ledger(new_ledger, ledger_generation)
        return result
    finally:
        deps.lock.release()
