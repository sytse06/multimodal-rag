"""The maintainer runbook must only name commands, variables and files that exist."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (ROOT / "docs" / "maintainer-runbook.md").read_text()
MAKEFILE = (ROOT / "Makefile").read_text()


def _targets() -> set[str]:
    return set(re.findall(r"^([a-z][a-z0-9-]*):", MAKEFILE, re.M))


def test_every_make_target_in_the_runbook_exists() -> None:
    named = set(re.findall(r"\bmake ([a-z][a-z0-9-]*)", RUNBOOK))

    assert named, "the runbook names no make targets"
    assert named - _targets() == set()


def test_every_make_variable_in_the_runbook_is_read_by_the_makefile() -> None:
    variables = set(re.findall(r"\b([A-Z][A-Z_]{2,})=", RUNBOOK))

    missing = {v for v in variables if f"$({v})" not in MAKEFILE}
    assert missing == set()


def test_every_repo_path_the_runbook_points_to_exists() -> None:
    pattern = r"`((?:docs|config|src|tests)/[A-Za-z0-9_./-]+|compose\.yaml)`"
    paths = set(re.findall(pattern, RUNBOOK))

    assert paths
    assert {p for p in paths if not (ROOT / p).exists()} == set()


def test_the_runbook_keeps_its_hard_rules() -> None:
    for phrase in (
        "Local first",
        "pinned Ollama",
        "Restore only from the newest backup",
        "Do not use `make ingest`",
        "under 80%",
    ):
        assert phrase in RUNBOOK
