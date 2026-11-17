"""Tests for removed pages: the probe, dropped sources, the plan and the apply."""

import urllib.error
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from multimodal_rag.ingest.template import TemplateReport
from multimodal_rag.sync import probe as probe_module
from multimodal_rag.sync.apply import (
    ApplyFailed,
    ApplyOptions,
    ApplyRefused,
    apply_plan,
)
from multimodal_rag.sync.classify import DROPPED, Item, Kind, dropped_source_items
from multimodal_rag.sync.fetch import FetchedPage
from multimodal_rag.sync.ledger import Ledger, PageEntry, text_chunk_id
from multimodal_rag.sync.plan import PlanInputs, build_plan, compute_hash
from multimodal_rag.sync.policy import Policy
from multimodal_rag.sync.probe import (
    GONE,
    NOT_CHECKED,
    REDIRECT,
    STILL_SERVED,
    UNKNOWN,
    Probe,
    probe_url,
)
from tests.test_sync_apply import (
    CUT,
    CUT_NEW,
    HOST,
    LASTMOD,
    NEW,
    NEW_PAGE,
    World,
    _stored,
    entry,
)

OLD = f"https://{HOST}/article/437-old-page"
NEWER = f"https://{HOST}/article/525-new-home"
MARCH = datetime(2026, 3, 13, tzinfo=timezone.utc)


# ---------------------------------------------------------------- the probe


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _Opener:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def open(self, request: object, timeout: float = 0) -> _Response:
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return _Response(int(outcome))  # type: ignore[call-overload]


def _http_error(code: int, location: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if location:
        headers["Location"] = location
    return urllib.error.HTTPError("https://x", code, "msg", headers, None)


def _probe(*outcomes: object) -> Probe:
    opener = _Opener(list(outcomes))
    with patch.object(probe_module.urllib.request, "build_opener", lambda *a: opener):
        return probe_url("https://x/page", attempts=2)


@pytest.mark.parametrize("code", [404, 410])
def test_a_missing_page_is_gone(code: int) -> None:
    assert _probe(_http_error(code)) == Probe(GONE, code)


def test_a_page_that_is_still_served_is_not_a_removal() -> None:
    assert _probe(200) == Probe(STILL_SERVED, 200)


def test_a_redirect_reports_where_the_page_went() -> None:
    found = _probe(_http_error(301, "https://x/new-home"))

    assert (found.evidence, found.redirect_to) == (REDIRECT, "https://x/new-home")


def test_a_server_error_is_unknown_not_gone() -> None:
    assert _probe(_http_error(503), _http_error(503)).evidence == UNKNOWN


def test_a_network_failure_is_unknown_not_gone() -> None:
    found = _probe(urllib.error.URLError("timed out"), urllib.error.URLError("x"))

    assert found.evidence == UNKNOWN and "URLError" in (found.error or "")


def test_a_transient_failure_is_retried_once() -> None:
    assert _probe(_http_error(503), _http_error(404)).evidence == GONE


# ------------------------------------------------------------ dropped sources


def test_pages_of_a_source_the_registry_dropped_become_candidates() -> None:
    ledger = Ledger(
        pages={
            "https://hydrocam.paro.nl/article/1-a": PageEntry(
                url="https://hydrocam.paro.nl/article/1-a",
                host="hydrocam.paro.nl",
                text_chunks=1,
                frozen_chunks=0,
                ingested_at=MARCH,
            ),
            CUT: entry(CUT, 1, 0),
        }
    )

    items = dropped_source_items(ledger, {HOST})

    assert [(i.url, i.kind, i.reason) for i in items] == [
        ("https://hydrocam.paro.nl/article/1-a", Kind.REMOVED, DROPPED)
    ]


# ------------------------------------------------------------------ the plan


def _plan_for(
    world: World,
    *,
    evidence: str = GONE,
    status: int | None = 404,
    redirect_to: str | None = None,
    rename_of: str | None = None,
    reason: str = "stored, not in the sitemap",
    host: str = HOST,
    page_url: str = OLD,
    others: int = 40,
):
    """Seed a page that left the sitemap, then rebuild the plan around it."""
    index = world.index
    index.objects.update(
        {
            text_chunk_id(page_url, 0): (page_url, "web", "old text"),
            text_chunk_id(page_url, 1): (page_url, "web", "more old text"),
            uuid4(): (page_url, "web", "screenshot one"),
            uuid4(): (page_url, "web", "screenshot two"),
        }
    )
    ledger = world.ledger.ledger
    ledger.pages[page_url] = PageEntry(
        url=page_url, host=host, text_chunks=2, frozen_chunks=2, ingested_at=MARCH
    )
    for n in range(others):  # a realistic collection, so one page is a small share
        other = f"https://{host}/article/{n}-kept"
        ledger.pages[other] = entry(other, 5, 0)
    pages = [FetchedPage(CUT, CUT_NEW), FetchedPage(NEW, NEW_PAGE)]
    world.plan = build_plan(
        PlanInputs(
            run_id="run",
            registry_generation="7",
            registry_stale=False,
            ledger=ledger,
            ledger_generation="42",
            items=[
                Item(CUT, Kind.MODIFIED, "lastmod", lastmod=LASTMOD),
                Item(NEW, Kind.NEW, "not stored", lastmod=LASTMOD),
                Item(page_url, Kind.REMOVED, reason, rename_of),
            ],
            fetched=pages,
            kb_names={HOST: "KB"},
            chunk_size=400,
            stored_loader=lambda u: {
                i: _stored(u, t) for i, t in index.page_state(u).text.items()
            },
            template=TemplateReport(0, 1.0, 1.0),
            policy=Policy(),
            probe=lambda u: Probe(evidence, status, redirect_to),
            chunk_ids_loader=lambda u: [
                str(k) for k, v in index.objects.items() if v[0] == u
            ],
        )
    )
    return world.plan


def _apply(world: World, **options: object):
    deps = world.deps()
    deps.page_chunk_ids = lambda u: [
        str(k) for k, v in world.index.objects.items() if v[0] == u
    ]
    return apply_plan(world.plan, world.run_dir, deps, ApplyOptions(**options))  # type: ignore[arg-type]


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def test_the_plan_records_the_evidence_and_every_chunk_of_the_page(
    world: World,
) -> None:
    plan = _plan_for(world, rename_of=NEWER)

    removal = plan.removals[0]
    assert (removal.evidence, removal.http_status, removal.reason) == (
        GONE,
        404,
        "stored, not in the sitemap",
    )
    assert len(removal.chunk_uuids) == 4  # two text chunks and two screenshots
    assert removal.rename_of == NEWER and removal.host == HOST


def test_a_dropped_source_is_not_probed(world: World) -> None:
    calls: list[str] = []
    plan = _plan_for(world, reason=DROPPED)
    del calls

    assert plan.removals[0].evidence == "source-dropped"


def test_changing_the_evidence_invalidates_the_plan_hash(world: World) -> None:
    plan = _plan_for(world)
    plan.removals[0].evidence = GONE
    plan.plan_hash = compute_hash(plan)
    plan.removals[0].evidence = STILL_SERVED

    assert plan.plan_hash != compute_hash(plan)


def test_a_source_that_loses_a_big_share_of_its_pages_needs_review(
    world: World,
) -> None:
    plan = _plan_for(world, others=5)  # 1 of 6 pages missing

    assert plan.policy.status == "needs_review"
    assert any("missing from its sitemap" in r for r in plan.policy.reasons)


# ----------------------------------------------------------------- the apply


def test_nothing_is_deleted_unless_a_person_names_the_page(world: World) -> None:
    _plan_for(world)

    result = _apply(world)

    assert result.pending_removals == [OLD]
    assert result.removed_pages == 0
    assert len([v for v in world.index.objects.values() if v[0] == OLD]) == 4
    assert OLD in world.ledger.ledger.pages


def test_a_confirmed_page_loses_every_chunk_including_its_screenshots(
    world: World,
) -> None:
    _plan_for(world)
    videos = world.index.counts().video

    result = _apply(world, confirm_removals=[OLD])

    assert (result.removed_pages, result.removed_chunks) == (1, 4)
    assert not [v for v in world.index.objects.values() if v[0] == OLD]
    assert world.index.counts().video == videos
    assert OLD not in world.ledger.ledger.pages
    assert result.pending_removals == []
    assert world.index.page_state(CUT).frozen == 2  # other pages are untouched


def test_removals_are_deleted_after_the_writes(world: World) -> None:
    _plan_for(world)

    _apply(world, confirm_removals=[OLD])

    events = world.index.events
    assert events.index("add") < len(events) - 1 and events[-1] == "delete"


@pytest.mark.parametrize(
    ("evidence", "fragment"),
    [
        (STILL_SERVED, "still served"),
        (UNKNOWN, "no evidence"),
        (NOT_CHECKED, "no evidence"),
    ],
)
def test_a_page_the_site_does_not_confirm_as_gone_is_refused(
    world: World, evidence: str, fragment: str
) -> None:
    _plan_for(world, evidence=evidence, status=200)

    with pytest.raises(ApplyRefused, match=fragment):
        _apply(world, confirm_removals=[OLD])

    assert world.index.events == []


def test_naming_a_page_that_is_not_a_candidate_is_refused(world: World) -> None:
    _plan_for(world)

    with pytest.raises(ApplyRefused, match="not a removal candidate"):
        _apply(world, confirm_removals=[OLD + "-typo"])


def test_a_redirect_is_deletable_only_if_its_target_is_held(world: World) -> None:
    _plan_for(world, evidence=REDIRECT, status=301, redirect_to="https://x/elsewhere")
    with pytest.raises(ApplyRefused, match="not held"):
        _apply(world, confirm_removals=[OLD])

    world.ledger.ledger.pages[CUT] = entry(CUT, 1, 2)
    world.ledger.generation = "42"
    _plan_for(world, evidence=REDIRECT, status=301, redirect_to=CUT)

    assert _apply(world, confirm_removals=[OLD]).removed_pages == 1


def test_a_page_with_a_replacement_waits_until_the_replacement_is_held(
    world: World,
) -> None:
    _plan_for(world, rename_of=f"https://{HOST}/article/999-nowhere")
    with pytest.raises(ApplyRefused, match="not held"):
        _apply(world, confirm_removals=[OLD])

    _plan_for(world, rename_of=NEW)  # NEW is written by this very plan

    assert _apply(world, confirm_removals=[OLD]).removed_pages == 1


def test_a_dropped_source_is_confirmed_by_source_not_by_page(world: World) -> None:
    _plan_for(world, reason=DROPPED)

    with pytest.raises(ApplyRefused, match="confirm the source"):
        _apply(world, confirm_removals=[OLD])
    with pytest.raises(ApplyRefused, match="no pages of a dropped source"):
        _apply(world, confirm_sources=["other.paro.nl"])

    result = _apply(world, confirm_sources=[HOST])

    assert result.removed_pages == 1 and OLD not in world.ledger.ledger.pages


def test_a_page_that_gained_chunks_since_the_plan_is_refused(world: World) -> None:
    _plan_for(world)
    world.index.objects[uuid4()] = (OLD, "web", "added after the plan")

    with pytest.raises(ApplyRefused, match="chunks changed since the plan"):
        _apply(world, confirm_removals=[OLD])

    assert not {"add", "delete", "snapshot"} & set(world.index.events)


def test_a_page_that_already_lost_some_chunks_is_finished(world: World) -> None:
    _plan_for(world)
    world.index.objects.pop(text_chunk_id(OLD, 0))  # an unfinished earlier run

    result = _apply(world, confirm_removals=[OLD])

    assert result.removed_chunks == 3 and not [
        v for v in world.index.objects.values() if v[0] == OLD
    ]


def test_a_removal_that_is_too_large_needs_review_even_when_named(
    world: World,
) -> None:
    _plan_for(world, others=0)
    world.plan.policy.status = "within"
    world.plan.policy.reasons = []
    world.plan.plan_hash = compute_hash(world.plan)

    with pytest.raises(ApplyRefused, match="deleting") as raised:
        _apply(world, confirm_removals=[OLD])

    assert raised.value.needs_review


def test_a_failed_delete_leaves_the_ledger_alone(world: World) -> None:
    _plan_for(world)
    original = world.index.delete_chunks

    def silently_keep(uuids: list) -> int:  # type: ignore[type-arg]
        original([])
        return 0

    world.index.delete_chunks = silently_keep  # type: ignore[method-assign]

    with pytest.raises(ApplyFailed, match="still there|expected"):
        _apply(world, confirm_removals=[OLD])

    assert OLD in world.ledger.ledger.pages and world.ledger.writes == 0


def test_the_default_options_can_never_delete_a_page() -> None:
    options = ApplyOptions()

    assert options.confirm_removals == [] and options.confirm_sources == []


def test_a_run_that_changes_nothing_leaves_the_ledger_and_its_generation_alone(
    world: World,
) -> None:
    _plan_for(world)
    _apply(world)  # writes the plan's pages and records them
    writes = world.ledger.writes
    world.ledger.generation = "42"
    world.plan.ledger_generation = "42"
    world.plan.plan_hash = compute_hash(world.plan)

    result = _apply(world)

    assert result.status == "nothing-to-write"
    assert world.ledger.writes == writes and result.ledger_generation == "42"
