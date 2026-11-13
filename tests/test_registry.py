"""Tests for pulling and pushing the source registry."""

import base64
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from multimodal_rag.ingest import registry
from multimodal_rag.ingest.registry import pull, push, validate

GOOD = (
    "youtube:\n  - url: https://youtu.be/abc\n    name: A\n"
    "knowledge_bases:\n  - url: https://kb.example.com/\n    name: KB\n"
)


class FakeBucket:
    """Minimal stand-in for the gcloud calls the registry makes."""

    def __init__(self, content: str | None = None) -> None:
        self.content = content
        self.generation = 1
        self.calls: list[list[str]] = []

    def md5(self) -> str:
        assert self.content is not None
        return base64.b64encode(hashlib.md5(self.content.encode()).digest()).decode()

    def __call__(self, args: list[str], project: str = "") -> str:
        self.calls.append(args)
        if args[1] == "objects":
            if self.content is None:
                raise RuntimeError("gcloud storage objects failed: 404 No such object")
            meta = {"generation": str(self.generation), "md5_hash": self.md5()}
            return json.dumps(meta)
        src, dst = args[2], args[3]
        if src.startswith("gs://"):
            assert self.content is not None
            Path(dst).write_text(self.content)
            return ""
        match = args[4].split("=")[1]
        current = str(self.generation) if self.content is not None else "0"
        if match != current:
            raise RuntimeError("gcloud storage cp failed: 412 precondition")
        self.content = Path(src).read_text()
        self.generation += 1
        return ""


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.state_bucket = "bucket"
    settings.gcp_project = "proj"
    return settings


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "sources.yaml", tmp_path / ".generation"


def _with(bucket: FakeBucket):
    return patch.object(registry, "run_gcloud", bucket)


def test_validate_rejects_empty_and_duplicate_registries(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("youtube: []\n")
    with pytest.raises(ValueError, match="no sources"):
        validate(empty)

    dup = tmp_path / "dup.yaml"
    dup.write_text(
        "youtube:\n  - url: https://youtu.be/a\n    name: A\n"
        "  - url: https://youtu.be/a\n    name: B\n"
    )
    with pytest.raises(ValueError, match="Duplicate youtube"):
        validate(dup)


def test_first_push_requires_init(paths: tuple[Path, Path]) -> None:
    local, gen = paths
    local.write_text(GOOD)
    with _with(FakeBucket()), pytest.raises(RuntimeError, match="INIT=1"):
        push(_settings(), local, gen)


def test_init_push_creates_registry_and_records_generation(
    paths: tuple[Path, Path],
) -> None:
    local, gen = paths
    local.write_text(GOOD)
    bucket = FakeBucket()
    bucket.generation = 0
    with _with(bucket):
        message = push(_settings(), local, gen, init=True)

    assert bucket.content == GOOD
    assert "1 videos and 1 knowledge bases" in message
    assert gen.read_text().strip() == str(bucket.generation)


def test_push_after_pull_updates_the_registry(paths: tuple[Path, Path]) -> None:
    local, gen = paths
    bucket = FakeBucket(GOOD)
    with _with(bucket):
        pull(_settings(), local, gen)
        local.write_text(GOOD + "  - url: https://kb.example.com/two\n    name: KB2\n")
        push(_settings(), local, gen)

    assert "KB2" in (bucket.content or "")
    assert gen.read_text().strip() == str(bucket.generation)


def test_push_refuses_when_remote_changed_since_pull(
    paths: tuple[Path, Path],
) -> None:
    local, gen = paths
    bucket = FakeBucket(GOOD)
    with _with(bucket):
        pull(_settings(), local, gen)
        bucket.content = GOOD + "# someone else\n"
        bucket.generation += 1
        local.write_text(GOOD + "# mine\n")
        with pytest.raises(RuntimeError, match="changed since your last pull"):
            push(_settings(), local, gen)

    assert "someone else" in (bucket.content or "")


def test_push_without_a_pull_is_refused(paths: tuple[Path, Path]) -> None:
    local, gen = paths
    local.write_text(GOOD)
    with _with(FakeBucket(GOOD)), pytest.raises(RuntimeError, match="pull"):
        push(_settings(), local, gen)


def test_push_of_identical_file_uploads_nothing(paths: tuple[Path, Path]) -> None:
    local, gen = paths
    bucket = FakeBucket(GOOD)
    with _with(bucket):
        pull(_settings(), local, gen)
        before = bucket.generation
        message = push(_settings(), local, gen)

    assert "Nothing to push" in message
    assert bucket.generation == before


def test_push_validates_before_touching_the_bucket(paths: tuple[Path, Path]) -> None:
    local, gen = paths
    local.write_text("youtube: []\n")
    bucket = FakeBucket(GOOD)
    with _with(bucket), pytest.raises(ValueError, match="no sources"):
        push(_settings(), local, gen)

    assert bucket.calls == []


def test_pull_refuses_to_overwrite_local_edits_unless_forced(
    paths: tuple[Path, Path],
) -> None:
    local, gen = paths
    local.write_text(GOOD + "# unpushed\n")
    with _with(FakeBucket(GOOD)):
        with pytest.raises(RuntimeError, match="differs"):
            pull(_settings(), local, gen)
        assert "unpushed" in local.read_text()

        pull(_settings(), local, gen, force=True)

    assert local.read_text() == GOOD
    assert "unpushed" in local.with_name("sources.yaml.prev").read_text()


def test_pull_without_a_registry_explains_how_to_create_one(
    paths: tuple[Path, Path],
) -> None:
    local, gen = paths
    with _with(FakeBucket()), pytest.raises(RuntimeError, match="INIT=1"):
        pull(_settings(), local, gen)


def test_missing_bucket_setting_is_reported(paths: tuple[Path, Path]) -> None:
    local, gen = paths
    settings = _settings()
    settings.state_bucket = ""
    with pytest.raises(ValueError, match="STATE_BUCKET"):
        pull(settings, local, gen)
