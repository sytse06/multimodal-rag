"""Policy: the limits that decide whether a plan may proceed without review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

POLICY_PATH = Path("config/sync-policy.yaml")


class Policy(BaseModel):
    max_pages_to_fetch: int = Field(default=30, ge=0)
    max_new_pages: int = Field(default=30, ge=0)
    max_content_changed_pages: int = Field(default=30, ge=0)
    max_deleted_chunk_share: float = Field(default=0.10, ge=0, le=1)


def load_policy(path: Path = POLICY_PATH) -> Policy:
    if not path.exists():
        return Policy()
    return Policy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


@dataclass(frozen=True)
class PlanStats:
    pages_to_fetch: int
    new_pages: int
    content_changed_pages: int
    deleted_chunks: int
    text_chunks_total: int
    fetch_failures: int
    template_needs_review: bool
    registry_stale: bool


class PolicyResult(BaseModel):
    status: str  # "within" or "needs_review"
    reasons: list[str] = Field(default_factory=list)


def evaluate(policy: Policy, stats: PlanStats) -> PolicyResult:
    reasons = []
    if stats.pages_to_fetch > policy.max_pages_to_fetch:
        reasons.append(
            f"{stats.pages_to_fetch} pages to fetch exceeds {policy.max_pages_to_fetch}"
        )
    if stats.new_pages > policy.max_new_pages:
        reasons.append(f"{stats.new_pages} new pages exceeds {policy.max_new_pages}")
    if stats.content_changed_pages > policy.max_content_changed_pages:
        reasons.append(
            f"{stats.content_changed_pages} pages with changed content exceeds"
            f" {policy.max_content_changed_pages}"
        )
    total = stats.text_chunks_total
    share = stats.deleted_chunks / total if total else 0.0
    if share > policy.max_deleted_chunk_share:
        reasons.append(
            f"deleting {share:.0%} of the page-text chunks exceeds"
            f" {policy.max_deleted_chunk_share:.0%}"
        )
    if stats.fetch_failures:
        reasons.append(f"{stats.fetch_failures} pages failed to fetch")
    if stats.template_needs_review:
        reasons.append("the site template may have changed")
    if stats.registry_stale:
        reasons.append("the registry changed since the last pull")
    return PolicyResult(status="needs_review" if reasons else "within", reasons=reasons)
