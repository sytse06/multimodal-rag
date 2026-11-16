"""Tests for the sync plan: actions, hashing, staleness, policy and run files."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from multimodal_rag.ingest.template import TemplateReport, check_template
from multimodal_rag.ingest.web import chunk_page
from multimodal_rag.sync import manifest as manifest_module
from multimodal_rag.sync.classify import Item, Kind
from multimodal_rag.sync.diff import StoredChunk
from multimodal_rag.sync.fetch import FetchedPage
from multimodal_rag.sync.ledger import Ledger, PageEntry, text_chunk_id
from multimodal_rag.sync.manifest import (
    build_manifest,
    upload_run,
    write_plan_files,
)
from multimodal_rag.sync.plan import (
    Plan,
    PlanInputs,
    build_plan,
    canonical_json,
    compute_hash,
    hash_is_valid,
    staleness,
)
from multimodal_rag.sync.policy import PlanStats, Policy, evaluate, load_policy

HOST = "hydroman.paro.nl"
FOOTER = (
    "Did this answer your question?Thanks for the feedback\n\nYesNo\n\n"
    "Last updated on June 1, 2026\n\nNo results found"
)
MARCH = datetime(2026, 3, 13, tzinfo=timezone.utc)


def url(slug: str) -> str:
    return f"https://{HOST}/article/{slug}"


def md(title: str, body: str) -> str:
    byline = "HydroManUpdated 1 May 20252 min readPrint"
    return f"# {title}\n\n{byline}\n\n{body}\n\n{FOOTER}"


def stored_as_old_template(markdown: str, page_url: str) -> dict[int, StoredChunk]:
    """Chunks as March stored them: no title in chunk 0, footer on the last chunk."""
    chunks = chunk_page(markdown, page_url, "KB")
    title = markdown.split("\n", 1)[0][2:]
    out = {}
    for i, c in enumerate(chunks):
        text = c.text[len(title) + 2 :] if i == 0 else c.text
        if i == len(chunks) - 1:
            text += " Did this answer your question?Thanks YesNo No results found"
        out[i] = StoredChunk(text_chunk_id(page_url, i), text)
    return out


def entry(page_url: str, text: int = 2, frozen: int = 0) -> PageEntry:
    return PageEntry(
        url=page_url,
        host=HOST,
        text_chunks=text,
        frozen_chunks=frozen,
        ingested_at=MARCH,
    )


def inputs(
    fetched: list[FetchedPage],
    items: list[Item],
    stored: dict[str, dict[int, StoredChunk]],
    ledger: Ledger | None = None,
    template: TemplateReport | None = None,
    **overrides: object,
) -> PlanInputs:
    base = PlanInputs(
        run_id="20261116T000000Z",
        registry_generation="7",
        registry_stale=False,
        ledger=ledger or Ledger(),
        ledger_generation="42",
        items=items,
        fetched=fetched,
        kb_names={HOST: "KB"},
        chunk_size=400,
        stored_loader=lambda u: stored.get(u, {}),
        template=template or TemplateReport(0, 1.0, 1.0),
        policy=Policy(),
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


OLD = md("Cut outs", "## Goal:\n\nCreating cut outs in your manifold.")
EDITED = md("Cut outs", "## Goal:\n\nCreating cut outs in your manifold, edited.")


def test_a_template_only_page_is_written_but_not_counted_as_changed() -> None:
    page = url("1-cut")
    plan = build_plan(
        inputs(
            [FetchedPage(page, OLD)],
            [Item(page, Kind.MODIFIED, "lastmod 2026-06-01")],
            {page: stored_as_old_template(OLD, page)},
            Ledger(pages={page: entry(page, 1)}),
        )
    )

    page_plan = plan.pages[0]
    assert not page_plan.content_changed
    assert {p.action for p in page_plan.positions} == {"write"}
    assert plan.totals["template_only_pages"] == 1
    assert plan.totals["content_changed_pages"] == 0
    assert plan.canaries == []


def test_a_real_edit_is_flagged_and_gets_a_canary() -> None:
    page = url("1-cut")
    plan = build_plan(
        inputs(
            [FetchedPage(page, EDITED)],
            [Item(page, Kind.MODIFIED, "lastmod")],
            {page: stored_as_old_template(OLD, page)},
            Ledger(pages={page: entry(page, 1)}),
        )
    )

    assert plan.pages[0].content_changed
    assert plan.canaries == [{"question": "Cut outs", "expect": [page]}]


def test_a_new_page_is_all_writes() -> None:
    page = url("9-new")
    plan = build_plan(
        inputs([FetchedPage(page, OLD)], [Item(page, Kind.NEW, "not stored")], {})
    )

    assert plan.pages[0].content_changed and plan.pages[0].stored_chunks == 0
    assert {p.status for p in plan.pages[0].positions} == {"added"}
    assert plan.totals["new_pages"] == 1


def test_trailing_positions_that_disappear_are_deletes() -> None:
    page = url("1-cut")
    stored = stored_as_old_template(OLD, page)
    stored[5] = StoredChunk(text_chunk_id(page, 5), "an old tail chunk")
    plan = build_plan(
        inputs(
            [FetchedPage(page, OLD)],
            [Item(page, Kind.MODIFIED, "lastmod")],
            {page: stored},
            Ledger(pages={page: entry(page, 2)}),
        )
    )

    deletes = [p for p in plan.pages[0].positions if p.action == "delete"]
    assert [d.index for d in deletes] == [5]
    assert plan.totals["positions_to_delete"] == 1


def test_a_failed_fetch_is_listed_and_needs_review() -> None:
    page = url("1-cut")
    plan = build_plan(
        inputs(
            [FetchedPage(page, None, "RuntimeError: 429")],
            [Item(page, Kind.MODIFIED, "lastmod")],
            {},
        )
    )

    assert plan.failures[0].error == "RuntimeError: 429"
    assert plan.policy.status == "needs_review"
    assert any("failed to fetch" in r for r in plan.policy.reasons)


def test_a_removal_candidate_reports_its_orphaned_screenshots() -> None:
    gone = url("437-old")
    plan = build_plan(
        inputs(
            [],
            [Item(gone, Kind.REMOVED, "stored, not in the sitemap", url("525-new"))],
            {},
            Ledger(pages={gone: entry(gone, text=3, frozen=4)}),
        )
    )

    assert plan.removals[0].frozen_chunks == 4
    assert plan.removals[0].rename_of == url("525-new")
    assert plan.policy.status == "within"  # removals wait for a confirmation


def test_frozen_chunks_are_counted_but_never_planned() -> None:
    page = url("1-cut")
    plan = build_plan(
        inputs(
            [FetchedPage(page, OLD)],
            [Item(page, Kind.MODIFIED, "lastmod")],
            {page: stored_as_old_template(OLD, page)},
            Ledger(pages={page: entry(page, 1, frozen=6)}),
        )
    )

    assert plan.totals["frozen_chunks_untouched"] == 6
    assert all(len(p.uuid) == 36 for p in plan.pages[0].positions)
    assert len(plan.pages[0].positions) == 1


def test_a_fetched_page_missing_from_the_inventory_is_an_error() -> None:
    with pytest.raises(ValueError, match="not in the inventory"):
        build_plan(inputs([FetchedPage(url("x"), OLD)], [], {}))


def _plan() -> Plan:
    page = url("1-cut")
    return build_plan(
        inputs(
            [FetchedPage(page, EDITED)],
            [Item(page, Kind.MODIFIED, "lastmod")],
            {page: stored_as_old_template(OLD, page)},
            Ledger(pages={page: entry(page, 1)}),
        )
    )


def test_the_same_inputs_give_the_same_plan_and_hash() -> None:
    assert _plan().plan_hash == _plan().plan_hash
    assert len(_plan().plan_hash) == 64


def test_the_hash_survives_a_json_round_trip() -> None:
    plan = _plan()

    assert Plan.model_validate_json(plan.model_dump_json()).plan_hash == plan.plan_hash
    assert canonical_json(plan) == canonical_json(
        Plan.model_validate_json(plan.model_dump_json())
    )


def test_changing_the_plan_invalidates_its_hash() -> None:
    plan = _plan()
    assert hash_is_valid(plan)

    plan.pages[0].positions[0].action = "delete"

    assert not hash_is_valid(plan)
    assert compute_hash(plan) != plan.plan_hash


def test_an_apply_refuses_a_plan_whose_inputs_moved() -> None:
    plan = _plan()

    assert staleness(plan, "7", "42") == []
    assert "registry changed" in staleness(plan, "8", "42")[0]
    assert "ledger changed" in staleness(plan, "7", "43")[0]
    plan.totals["pages"] = 99
    assert "hash does not match" in staleness(plan, "7", "42")[0]


def test_template_trouble_makes_the_plan_need_review() -> None:
    page = url("1-cut")
    pages = {page: "# T\n\nbyline missing here\n\nbody text " * 3}
    plan = build_plan(
        inputs(
            [FetchedPage(page, OLD)],
            [Item(page, Kind.MODIFIED, "lastmod")],
            {},
            template=check_template(pages),
        )
    )

    assert plan.template["needs_review"]
    assert plan.policy.status == "needs_review"


STATS = PlanStats(
    pages_to_fetch=10,
    new_pages=5,
    content_changed_pages=3,
    deleted_chunks=2,
    text_chunks_total=327,
    fetch_failures=0,
    template_needs_review=False,
    registry_stale=False,
)


def test_policy_allows_a_plan_inside_every_limit() -> None:
    assert evaluate(Policy(), STATS).status == "within"


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"pages_to_fetch": 31}, "pages to fetch"),
        ({"new_pages": 31}, "new pages"),
        ({"content_changed_pages": 31}, "changed content"),
        ({"deleted_chunks": 40}, "deleting"),
        ({"fetch_failures": 1}, "failed to fetch"),
        ({"template_needs_review": True}, "template"),
        ({"registry_stale": True}, "registry"),
    ],
)
def test_policy_names_each_limit_that_is_exceeded(
    change: dict[str, object], fragment: str
) -> None:
    result = evaluate(Policy(), PlanStats(**{**STATS.__dict__, **change}))

    assert result.status == "needs_review"
    assert any(fragment in reason for reason in result.reasons)


def test_the_tracked_policy_file_loads_and_a_missing_one_gives_defaults(
    tmp_path: Path,
) -> None:
    assert load_policy(Path("config/sync-policy.yaml")).max_pages_to_fetch == 30
    assert load_policy(tmp_path / "absent.yaml") == Policy()


def test_run_files_hold_the_plan_and_a_manifest_that_names_its_hash(
    tmp_path: Path,
) -> None:
    plan = _plan()

    write_plan_files(tmp_path / "run", plan, build_manifest(plan))

    assert Plan.model_validate_json((tmp_path / "run" / "plan.json").read_text())
    saved = (tmp_path / "run" / "manifest.json").read_text()
    assert plan.plan_hash in saved and '"ledger_generation": "42"' in saved


def test_the_run_is_uploaded_under_runs_with_its_id(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake(args: list[str], project: str = "") -> str:
        calls.append(args)
        return ""

    run = tmp_path / "20261116T000000Z"
    run.mkdir()
    with patch.object(manifest_module, "run_gcloud", fake):
        where = upload_run(run, "bucket", "proj")

    assert calls[0][:3] == ["storage", "cp", "-r"]
    assert calls[0][-1] == "gs://bucket/runs/"
    assert where == "gs://bucket/runs/20261116T000000Z/"
