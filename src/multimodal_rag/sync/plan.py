"""The sync plan: a reviewable, hashed description of exactly what an apply will do.

A plan is built from the inventory, the fetched pages and the stored chunks. It is
canonical JSON with a hash, and it records the registry and ledger generations it was
built from, so an apply can refuse a plan whose inputs have moved. It lists every
position as write, skip or delete, and the fetched pages it refers to are kept with
their checksums, so the apply fetches nothing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from multimodal_rag.ingest.template import TemplateReport
from multimodal_rag.ingest.web import (
    NORMALISER_VERSION,
    chunk_page,
    normalise_page,
    page_fingerprint,
)
from multimodal_rag.sync.classify import Item, Kind
from multimodal_rag.sync.diff import (
    IDENTICAL,
    REMOVED,
    TEMPLATE_ONLY,
    StoredChunk,
    diff_page,
)
from multimodal_rag.sync.fetch import FetchedPage, file_name
from multimodal_rag.sync.ledger import Ledger
from multimodal_rag.sync.policy import PlanStats, Policy, PolicyResult, evaluate

PLAN_VERSION = 2
WRITE, SKIP, DELETE = "write", "skip", "delete"


class PositionPlan(BaseModel):
    index: int
    status: str
    uuid: str
    action: str
    text_sha256: str | None = None  # of the chunk text an apply must write


class PagePlan(BaseModel):
    url: str
    host: str
    kb_name: str
    kind: str
    reason: str
    lastmod: str | None = None  # the sitemap version this plan brings the index to
    content_changed: bool
    title: str | None
    fingerprint: str
    stored_chunks: int
    fresh_chunks: int
    frozen_chunks: int
    page_file: str
    page_sha256: str
    positions: list[PositionPlan]


class Removal(BaseModel):
    url: str
    text_chunks: int
    frozen_chunks: int
    rename_of: str | None = None


class Failure(BaseModel):
    url: str
    error: str


class Plan(BaseModel):
    version: int = PLAN_VERSION
    run_id: str
    registry_generation: str | None
    ledger_generation: str
    normaliser_version: int = NORMALISER_VERSION
    chunk_size: int
    pages: list[PagePlan] = Field(default_factory=list)
    removals: list[Removal] = Field(default_factory=list)
    failures: list[Failure] = Field(default_factory=list)
    totals: dict[str, int] = Field(default_factory=dict)
    template: dict[str, Any] = Field(default_factory=dict)
    canaries: list[dict[str, Any]] = Field(default_factory=list)
    policy: PolicyResult = Field(default_factory=lambda: PolicyResult(status="within"))
    plan_hash: str = ""


def canonical_json(plan: Plan) -> str:
    """Stable text of the plan without its own hash: same plan, same bytes."""
    body = plan.model_dump(mode="json", exclude={"plan_hash"})
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_hash(plan: Plan) -> str:
    return hashlib.sha256(canonical_json(plan).encode()).hexdigest()


def hash_is_valid(plan: Plan) -> bool:
    return bool(plan.plan_hash) and plan.plan_hash == compute_hash(plan)


def staleness(
    plan: Plan, registry_generation: str | None, ledger_generation: str
) -> list[str]:
    """Reasons an apply must refuse this plan; empty if its inputs have not moved."""
    reasons = []
    if plan.version != PLAN_VERSION:
        reasons.append(f"plan format {plan.version} is not {PLAN_VERSION}")
    if not hash_is_valid(plan):
        reasons.append("the plan hash does not match its content")
    if plan.registry_generation != registry_generation:
        reasons.append("the registry changed after the plan was made")
    if plan.ledger_generation != ledger_generation:
        reasons.append("the ledger changed after the plan was made")
    return reasons


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class PlanInputs:
    run_id: str
    registry_generation: str | None
    registry_stale: bool
    ledger: Ledger
    ledger_generation: str
    items: list[Item]
    fetched: list[FetchedPage]
    kb_names: dict[str, str]
    chunk_size: int
    stored_loader: Callable[[str], dict[int, StoredChunk]]
    template: TemplateReport
    policy: Policy


def _action(status: str) -> str:
    if status == IDENTICAL:
        return SKIP
    return DELETE if status == REMOVED else WRITE


def build_plan(inputs: PlanInputs) -> Plan:
    kinds = {i.url: i for i in inputs.items}
    plan = Plan(
        run_id=inputs.run_id,
        registry_generation=inputs.registry_generation,
        ledger_generation=inputs.ledger_generation,
        chunk_size=inputs.chunk_size,
    )
    for page in inputs.fetched:
        if page.markdown is None:
            plan.failures.append(Failure(url=page.url, error=page.error or "unknown"))
            continue
        item = kinds.get(page.url)
        if item is None:
            raise ValueError(f"{page.url} was fetched but is not in the inventory")
        host = urlparse(page.url).netloc
        name = inputs.kb_names[host]
        normalised = normalise_page(page.markdown)
        fresh = chunk_page(page.markdown, page.url, name, inputs.chunk_size)
        diff = diff_page(
            page.url, inputs.stored_loader(page.url), fresh, title=normalised.title
        )
        entry = inputs.ledger.pages.get(page.url)
        fresh_text = {c.chunk_index: c.text for c in fresh}
        plan.pages.append(
            PagePlan(
                url=page.url,
                host=host,
                kb_name=name,
                kind=item.kind.value,
                reason=item.reason,
                lastmod=item.lastmod.isoformat() if item.lastmod else None,
                content_changed=item.kind is Kind.NEW or diff.content_changed,
                title=normalised.title,
                fingerprint=page_fingerprint(page.markdown),
                stored_chunks=diff.stored,
                fresh_chunks=diff.fresh,
                frozen_chunks=entry.frozen_chunks if entry else 0,
                page_file=file_name(page.url),
                page_sha256=sha256_text(page.markdown),
                positions=[
                    PositionPlan(
                        index=c.index,
                        status=c.status,
                        uuid=str(c.uuid),
                        action=_action(c.status),
                        text_sha256=(
                            sha256_text(fresh_text[c.index])
                            if _action(c.status) == WRITE
                            else None
                        ),
                    )
                    for c in diff.changes
                ],
            )
        )
    for item in inputs.items:
        if item.kind is Kind.REMOVED:
            entry = inputs.ledger.pages[item.url]
            plan.removals.append(
                Removal(
                    url=item.url,
                    text_chunks=entry.text_chunks,
                    frozen_chunks=entry.frozen_chunks,
                    rename_of=item.rename_of,
                )
            )
    _finish(plan, inputs)
    return plan


def _count(plan: Plan, action: str) -> int:
    return sum(1 for p in plan.pages for pos in p.positions if pos.action == action)


def _finish(plan: Plan, inputs: PlanInputs) -> None:
    text_total = sum(e.text_chunks for e in inputs.ledger.pages.values())
    plan.totals = {
        "pages": len(plan.pages),
        "new_pages": sum(1 for p in plan.pages if p.kind == Kind.NEW.value),
        "content_changed_pages": sum(1 for p in plan.pages if p.content_changed),
        "template_only_pages": sum(
            1
            for p in plan.pages
            if not p.content_changed
            and any(pos.status == TEMPLATE_ONLY for pos in p.positions)
        ),
        "positions_to_write": _count(plan, WRITE),
        "positions_to_delete": _count(plan, DELETE),
        "positions_skipped": _count(plan, SKIP),
        "failures": len(plan.failures),
        "removal_candidates": len(plan.removals),
        "frozen_chunks_untouched": sum(p.frozen_chunks for p in plan.pages),
    }
    plan.template = {
        "needs_review": inputs.template.needs_review,
        "lines": inputs.template.lines(),
    }
    plan.canaries = [
        {"question": p.title, "expect": [p.url]}
        for p in plan.pages
        if p.title and p.content_changed
    ]
    stats = PlanStats(
        pages_to_fetch=len(plan.pages) + len(plan.failures),
        new_pages=plan.totals["new_pages"],
        content_changed_pages=plan.totals["content_changed_pages"],
        deleted_chunks=plan.totals["positions_to_delete"],
        text_chunks_total=text_total,
        fetch_failures=len(plan.failures),
        template_needs_review=inputs.template.needs_review,
        registry_stale=inputs.registry_stale,
    )
    plan.policy = evaluate(inputs.policy, stats)
    plan.plan_hash = compute_hash(plan)
