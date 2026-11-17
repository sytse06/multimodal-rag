"""Tests for applying a plan: validation, writes, verification and the ledger."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from multimodal_rag.checks import Check, Status
from multimodal_rag.ingest.template import TemplateReport
from multimodal_rag.ingest.web import chunk_page
from multimodal_rag.models.chunks import SupportChunk
from multimodal_rag.retrieval_gate import Canary
from multimodal_rag.sync.apply import (
    ApplyDeps,
    ApplyFailed,
    ApplyOptions,
    ApplyRefused,
    Counts,
    PageState,
    apply_plan,
)
from multimodal_rag.sync.classify import Item, Kind
from multimodal_rag.sync.fetch import FetchedPage, save_run
from multimodal_rag.sync.ledger import Ledger, PageEntry, text_chunk_id
from multimodal_rag.sync.lock import LockHeld
from multimodal_rag.sync.plan import PlanInputs, build_plan, compute_hash
from multimodal_rag.sync.policy import Policy

HOST = "hydroman.paro.nl"
CUT = f"https://{HOST}/article/1-cut"
NEW = f"https://{HOST}/article/2-new"
FOOTER = "Did this answer your question?Thanks\n\nYesNo\n\nNo results found"
MARCH = datetime(2026, 3, 13, tzinfo=timezone.utc)
LASTMOD = datetime(2026, 6, 17, 10, tzinfo=timezone.utc)


def md(title: str, body: str) -> str:
    byline = "HydroManUpdated 1 May 20252 min readPrint"
    return f"# {title}\n\n{byline}\n\n{body}\n\n{FOOTER}"


CUT_OLD = md("Cut outs", "## Goal:\n\nCreating cut outs in your manifold.")
CUT_NEW = md("Cut outs", "## Goal:\n\nCreating cut outs in your manifold, edited.")
NEW_PAGE = md("Brand new", "## Goal:\n\nSomething that did not exist.")


class FakeIndex:
    """A collection, a store and a state reader in one, so tests can look inside."""

    def __init__(self) -> None:
        self.objects: dict[UUID, tuple[str, str, str]] = {}
        self.events: list[str] = []
        self.reject = False
        self.drop_writes = False

    def seed_page(self, url: str, markdown: str, screenshots: int = 0) -> None:
        for i, chunk in enumerate(chunk_page(markdown, url, "KB")):
            old = chunk.text.split("\n\n", 1)[-1] + " Did this answer your question?"
            self.objects[text_chunk_id(url, i)] = (url, "web", old)
        for n in range(screenshots):
            self.objects[uuid4()] = (url, "web", f"screenshot description {n}")

    def add_chunks(self, chunks: list[SupportChunk]) -> int:
        self.events.append("add")
        if self.reject:
            raise RuntimeError("Weaviate rejected 1 of 3 chunks")
        for chunk in chunks if not self.drop_writes else chunks[:-1]:
            assert chunk.chunk_id is not None
            self.objects[chunk.chunk_id] = (chunk.source_url, "web", chunk.text)
        return len(chunks)

    def delete_chunks(self, uuids: list[UUID]) -> int:
        self.events.append("delete")
        return sum(self.objects.pop(u, None) is not None for u in uuids)

    def validate_compatibility(self, expected_vector_dimension: int = 768) -> None:
        self.events.append("validate")

    def page_state(self, url: str) -> PageState:
        mine = {u: v for u, v in self.objects.items() if v[0] == url}
        text = {
            i: mine[text_chunk_id(url, i)][2]
            for i in range(len(mine))
            if text_chunk_id(url, i) in mine
        }
        return PageState(text, len(mine) - len(text))

    def counts(self) -> Counts:
        return Counts(
            len(self.objects), sum(v[1] == "video" for v in self.objects.values())
        )


class FakeLock:
    def __init__(self, held_by: str | None = None) -> None:
        self.held_by = held_by
        self.events: list[str] = []

    def acquire(self, *, break_lock: bool = False) -> None:
        if self.held_by and not break_lock:
            raise LockHeld(f"another run holds the lock: {self.held_by}")
        self.events.append("acquire")

    def release(self) -> None:
        self.events.append("release")


class FakeLedgerStore:
    def __init__(self, ledger: Ledger, generation: str = "42") -> None:
        self.ledger, self.generation, self.writes = ledger, generation, 0

    def read(self) -> tuple[Ledger, str]:
        return self.ledger.model_copy(deep=True), self.generation

    def write(self, ledger: Ledger, if_generation: str) -> str:
        if if_generation != self.generation:
            raise RuntimeError("412 precondition failed")
        self.ledger, self.writes = ledger, self.writes + 1
        self.generation = str(int(self.generation) + 1)
        return self.generation


def entry(url: str, text: int, frozen: int) -> PageEntry:
    return PageEntry(
        url=url, host=HOST, text_chunks=text, frozen_chunks=frozen, ingested_at=MARCH
    )


class World:
    def __init__(self, tmp_path: Path) -> None:
        self.index = FakeIndex()
        self.index.seed_page(CUT, CUT_OLD, screenshots=2)
        self.index.objects[uuid4()] = ("https://youtu.be/abc", "video", "transcript")
        self.ledger = FakeLedgerStore(Ledger(pages={CUT: entry(CUT, 1, 2)}))
        self.lock = FakeLock()
        self.snapshots: list[str] = []
        self.gate_checks: list[Check] = [Check("canary", Status.OK, "rank 1")]
        self.gate_canaries: list[Canary] = []
        self.run_dir = tmp_path / "run"
        pages = [FetchedPage(CUT, CUT_NEW), FetchedPage(NEW, NEW_PAGE)]
        save_run(self.run_dir, pages)
        self.plan = build_plan(
            PlanInputs(
                run_id="run",
                registry_generation="7",
                registry_stale=False,
                ledger=self.ledger.ledger,
                ledger_generation="42",
                items=[
                    Item(CUT, Kind.MODIFIED, "lastmod", lastmod=LASTMOD),
                    Item(NEW, Kind.NEW, "not stored", lastmod=LASTMOD),
                ],
                fetched=pages,
                kb_names={HOST: "KB"},
                chunk_size=400,
                stored_loader=lambda u: {
                    i: _stored(u, t) for i, t in self.index.page_state(u).text.items()
                },
                template=TemplateReport(0, 1.0, 1.0),
                policy=Policy(),
            )
        )

    def deps(self) -> ApplyDeps:
        def gate(canaries: list[Canary]) -> list[Check]:
            self.gate_canaries = canaries
            return self.gate_checks

        def snapshot() -> str:
            self.index.events.append("snapshot")
            self.snapshots.append("runs/run/snapshot-local")
            return self.snapshots[-1]

        return ApplyDeps(
            store=self.index,
            page_state=self.index.page_state,
            counts=self.index.counts,
            registry_generation=lambda: "7",
            read_ledger=self.ledger.read,
            write_ledger=self.ledger.write,
            lock=self.lock,
            snapshot=snapshot,
            gate=gate,
            policy=Policy(),
            chunk_size=400,
            fixed_canaries=[Canary(question="fixed?", expect=["https://x"])],
            now=lambda: datetime(2026, 11, 17, tzinfo=timezone.utc),
        )

    def apply(self, **options: bool) -> object:
        return apply_plan(self.plan, self.run_dir, self.deps(), ApplyOptions(**options))


def _stored(url: str, text: str):  # type: ignore[no-untyped-def]
    from multimodal_rag.sync.diff import StoredChunk

    return StoredChunk(text_chunk_id(url, 0), text)


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def test_a_plan_is_applied_verified_gated_and_recorded(world: World) -> None:
    before = world.index.counts()

    result = world.apply()

    assert result.status == "applied"  # type: ignore[attr-defined]
    assert (result.inserted, result.replaced, result.deleted) == (  # type: ignore[attr-defined]
        len(chunk_page(NEW_PAGE, NEW, "KB")),
        1,
        0,
    )
    assert world.index.counts().total == before.total + result.inserted  # type: ignore[attr-defined]
    assert world.index.page_state(CUT).frozen == 2  # screenshots untouched
    assert world.index.counts().video == 1
    assert world.ledger.writes == 1 and result.ledger_generation == "43"  # type: ignore[attr-defined]


def test_the_ledger_records_what_is_now_held(world: World) -> None:
    world.apply()

    held = world.ledger.ledger.pages
    assert held[CUT].frozen_chunks == 2
    assert held[CUT].lastmod == LASTMOD
    assert held[CUT].fingerprint == world.plan.pages[0].fingerprint
    assert held[NEW].text_chunks == len(chunk_page(NEW_PAGE, NEW, "KB"))
    assert held[NEW].frozen_chunks == 0


def test_writes_come_before_deletes_and_the_snapshot_before_both(world: World) -> None:
    stale_tail = text_chunk_id(CUT, 1)
    world.index.objects[stale_tail] = (CUT, "web", "an old trailing chunk")
    other = f"https://{HOST}/article/9-other"
    world.ledger.ledger.pages[other] = entry(other, 300, 0)  # a realistic collection
    world.plan = _rebuilt_with_the_extra_position(world)

    world.apply()

    assert world.index.events == ["validate", "snapshot", "add", "delete"]
    assert stale_tail not in world.index.objects
    assert world.index.page_state(CUT).frozen == 2  # the page's screenshots survive


def _rebuilt_with_the_extra_position(world: World):  # type: ignore[no-untyped-def]
    pages = [FetchedPage(CUT, CUT_NEW), FetchedPage(NEW, NEW_PAGE)]
    return build_plan(
        PlanInputs(
            run_id="run",
            registry_generation="7",
            registry_stale=False,
            ledger=world.ledger.ledger,
            ledger_generation="42",
            items=[
                Item(CUT, Kind.MODIFIED, "lastmod", lastmod=LASTMOD),
                Item(NEW, Kind.NEW, "not stored", lastmod=LASTMOD),
            ],
            fetched=pages,
            kb_names={HOST: "KB"},
            chunk_size=400,
            stored_loader=lambda u: {
                i: _stored(u, t) for i, t in world.index.page_state(u).text.items()
            },
            template=TemplateReport(0, 1.0, 1.0),
            policy=Policy(),
        )
    )


def test_the_snapshot_can_be_skipped(world: World) -> None:
    world.apply(take_snapshot=False)

    assert "snapshot" not in world.index.events


def test_the_gate_gets_the_fixed_and_the_plans_canaries(world: World) -> None:
    world.apply()

    questions = [c.question for c in world.gate_canaries]
    assert questions[0] == "fixed?"
    assert "Cut outs" in questions and "Brand new" in questions


def test_a_plan_whose_ledger_moved_is_refused_before_anything_is_touched(
    world: World,
) -> None:
    world.ledger.generation = "43"

    with pytest.raises(ApplyRefused, match="ledger changed"):
        world.apply()

    assert world.index.events == [] and world.lock.events == []


def test_a_plan_whose_registry_moved_is_refused(world: World) -> None:
    deps = world.deps()
    deps.registry_generation = lambda: "8"

    with pytest.raises(ApplyRefused, match="registry changed"):
        apply_plan(world.plan, world.run_dir, deps, ApplyOptions())


def test_an_edited_plan_is_refused(world: World) -> None:
    world.plan.pages[0].positions[0].action = "delete"

    with pytest.raises(ApplyRefused, match="hash does not match"):
        world.apply()


def test_a_plan_that_needs_review_is_refused_unless_accepted(world: World) -> None:
    world.plan.policy.status = "needs_review"
    world.plan.policy.reasons = ["the site template may have changed"]
    world.plan.plan_hash = compute_hash(world.plan)

    with pytest.raises(ApplyRefused, match="template") as raised:
        world.apply()
    assert raised.value.needs_review
    assert world.index.events == []

    assert world.apply(accept_review=True).status == "applied"  # type: ignore[attr-defined]


def test_the_policy_is_checked_again_at_apply_time(world: World) -> None:
    deps = world.deps()
    deps.policy = Policy(max_new_pages=0)

    with pytest.raises(ApplyRefused, match="new pages") as raised:
        apply_plan(world.plan, world.run_dir, deps, ApplyOptions())
    assert raised.value.needs_review


def test_a_page_that_no_longer_matches_its_checksum_is_refused(world: World) -> None:
    saved = next((world.run_dir / "pages").glob("*1-cut*"))
    saved.write_text(saved.read_text() + "\n\nsneaked in")

    with pytest.raises(ApplyRefused, match="checksum"):
        world.apply()

    assert world.index.events == []


def test_a_changed_normaliser_or_chunk_size_is_refused(world: World) -> None:
    world.plan.normaliser_version = 0
    world.plan.plan_hash = compute_hash(world.plan)
    with pytest.raises(ApplyRefused, match="normaliser changed"):
        world.apply()

    world.plan.normaliser_version = 1
    world.plan.chunk_size = 999
    world.plan.plan_hash = compute_hash(world.plan)
    with pytest.raises(ApplyRefused, match="chunk size changed"):
        world.apply()


def test_a_collection_that_changed_since_the_plan_is_refused(world: World) -> None:
    world.index.objects[text_chunk_id(CUT, 1)] = (CUT, "web", "someone ingested")

    with pytest.raises(ApplyRefused, match="collection changed"):
        world.apply()

    assert "add" not in world.index.events


def test_a_rejected_batch_leaves_the_ledger_alone_and_releases_the_lock(
    world: World,
) -> None:
    world.index.reject = True

    with pytest.raises(RuntimeError, match="rejected"):
        world.apply()

    assert world.ledger.writes == 0
    assert world.lock.events == ["acquire", "release"]
    assert "delete" not in world.index.events


def test_a_write_that_silently_did_not_land_is_caught_by_verification(
    world: World,
) -> None:
    world.index.drop_writes = True

    with pytest.raises(ApplyFailed, match="differs from the plan|expected"):
        world.apply()

    assert world.ledger.writes == 0
    assert world.lock.events[-1] == "release"


def test_a_failed_gate_keeps_the_ledger_and_reports_the_snapshot(world: World) -> None:
    world.gate_checks = [Check("canary", Status.FAIL, "not in the top 10")]

    result = world.apply()

    assert result.status == "gate-failed"  # type: ignore[attr-defined]
    assert result.snapshot == "runs/run/snapshot-local"  # type: ignore[attr-defined]
    assert world.ledger.writes == 0


def test_rerunning_after_a_failed_gate_writes_nothing_and_finishes_the_ledger(
    world: World,
) -> None:
    world.gate_checks = [Check("canary", Status.FAIL, "x")]
    world.apply()
    world.gate_checks = [Check("canary", Status.OK, "fine")]
    world.index.events.clear()

    result = world.apply()

    assert result.status == "nothing-to-write"  # type: ignore[attr-defined]
    assert result.pages_already_applied == 2  # type: ignore[attr-defined]
    assert "add" not in world.index.events and "snapshot" not in world.index.events
    assert world.ledger.writes == 1


def test_a_held_lock_refuses_the_run_before_any_write(world: World) -> None:
    world.lock = FakeLock(held_by="other-run")

    with pytest.raises(LockHeld, match="other-run"):
        world.apply()

    assert "add" not in world.index.events


def test_breaking_a_stale_lock_is_explicit(world: World) -> None:
    world.lock = FakeLock(held_by="dead-run")

    assert world.apply(break_lock=True).status == "applied"  # type: ignore[attr-defined]


FINGERPRINT = {"ollama_version": "0.40.14", "model_digest": "d" * 64}


def test_an_index_that_drifted_from_its_text_refuses_the_apply(world: World) -> None:
    deps = world.deps()
    deps.embedding_check = lambda: (FINGERPRINT, ["stored vectors no longer match"])

    with pytest.raises(ApplyRefused, match="no longer match"):
        apply_plan(world.plan, world.run_dir, deps, ApplyOptions())

    assert world.index.events == [] and world.lock.events == []


def test_the_ledger_remembers_the_runtime_it_was_verified_against(world: World) -> None:
    deps = world.deps()
    deps.embedding_check = lambda: (FINGERPRINT, [])

    apply_plan(world.plan, world.run_dir, deps, ApplyOptions())

    assert world.ledger.ledger.embedding == FINGERPRINT


def test_a_runtime_that_changed_since_the_ledger_was_verified_refuses(
    world: World,
) -> None:
    world.ledger.ledger.embedding = {
        "ollama_version": "0.17.7",
        "model_digest": "d" * 64,
    }
    deps = world.deps()
    deps.embedding_check = lambda: (FINGERPRINT, [])

    with pytest.raises(ApplyRefused, match="runtime changed"):
        apply_plan(world.plan, world.run_dir, deps, ApplyOptions())


def test_a_ledger_with_no_recorded_runtime_accepts_the_first_one(world: World) -> None:
    assert world.ledger.ledger.embedding is None
    deps = world.deps()
    deps.embedding_check = lambda: (FINGERPRINT, [])

    result = apply_plan(world.plan, world.run_dir, deps, ApplyOptions())

    assert result.status == "applied"
