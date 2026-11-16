"""The run manifest and the run files kept in the bucket.

Each run leaves ``plan.json``, ``manifest.json`` and the fetched pages under
``runs/<run id>/``: the audit trail, and the exact input an apply reuses.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from multimodal_rag.gcloud import run_gcloud
from multimodal_rag.sync.plan import Plan


class Manifest(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    git_commit: str
    plan_hash: str
    registry_generation: str | None
    ledger_generation: str
    normaliser_version: int
    policy_status: str
    totals: dict[str, int]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_manifest(plan: Plan) -> Manifest:
    return Manifest(
        run_id=plan.run_id,
        git_commit=git_commit(),
        plan_hash=plan.plan_hash,
        registry_generation=plan.registry_generation,
        ledger_generation=plan.ledger_generation,
        normaliser_version=plan.normaliser_version,
        policy_status=plan.policy.status,
        totals=plan.totals,
    )


def write_plan_files(run_dir: Path, plan: Plan, manifest: Manifest) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )


def upload_run(run_dir: Path, bucket: str, project: str) -> str:
    """Copy the run directory to ``runs/<run id>/``; return the destination."""
    destination = f"gs://{bucket}/runs/"
    run_gcloud(["storage", "cp", "-r", str(run_dir), destination], project)
    return f"{destination}{run_dir.name}/"
