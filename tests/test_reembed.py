"""Tests for the re-embed tool and the embedding fingerprint."""

import hashlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from multimodal_rag.store.fingerprint import embedding_fingerprint
from multimodal_rag.sync.reembed import (
    cosine,
    embedding_input,
    reembed,
    verify_current,
)


class FakeEmbeddings:
    """A stable, text-dependent 4-d vector, so a change of text changes the vector."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 + 0.1 for b in digest[:4]]


class FakeCollection:
    def __init__(self, objects: list[SimpleNamespace], *, drop_updates: bool = False):
        self.objects = {str(o.uuid): o for o in objects}
        self.drop_updates = drop_updates
        self.data = SimpleNamespace(update=self._update)

    def _update(self, uuid: object, vector: dict[str, list[float]]) -> None:
        if not self.drop_updates:
            self.objects[str(uuid)].vector = vector

    def iterator(self, include_vector: bool = False) -> list[SimpleNamespace]:
        return list(self.objects.values())


def chunk(text: str, source_type: str = "web", vector: list[float] | None = None):
    return SimpleNamespace(
        uuid=uuid4(),
        properties={
            "text": text,
            "source_name": "KB",
            "source_type": source_type,
            "source_url": "https://x",
        },
        vector={"default": vector or [1.0, 0.0, 0.0, 0.0]},
    )


def test_the_embedding_input_is_the_source_name_then_the_text() -> None:
    assert embedding_input(chunk("hello").properties) == "KB: hello"


def test_a_dry_run_measures_the_drift_and_writes_nothing() -> None:
    old = chunk("old text")
    collection = FakeCollection([old])
    before = list(old.vector["default"])

    report = reembed(collection, FakeEmbeddings(), write=False)

    assert report.total == 1 and report.written == 0
    assert old.vector["default"] == before
    assert report.by_kind["web"].summary()["already_current"] == 0


def test_a_write_replaces_only_the_vector() -> None:
    objects = [chunk("one"), chunk("two", source_type="video")]
    collection = FakeCollection(objects)
    properties = [dict(o.properties) for o in objects]
    embeddings = FakeEmbeddings()

    report = reembed(collection, embeddings, write=True)

    assert report.written == 2 and report.problems == []
    assert [o.properties for o in objects] == properties
    assert objects[0].vector["default"] == FakeEmbeddings._vector("KB: one")
    assert set(report.by_kind) == {"web", "video"}


def test_a_vector_that_is_already_current_is_counted_as_such() -> None:
    current = chunk("same", vector=FakeEmbeddings._vector("KB: same"))
    stale = chunk("other")

    report = reembed(FakeCollection([current, stale]), FakeEmbeddings(), write=False)

    assert report.by_kind["web"].already_current == 1


def test_an_update_that_does_not_land_is_reported_not_hidden() -> None:
    collection = FakeCollection([chunk("one")], drop_updates=True)

    report = reembed(collection, FakeEmbeddings(), write=True)

    assert any("do not read back" in p for p in report.problems)


def test_verify_current_returns_the_worst_match() -> None:
    good = chunk("a", vector=FakeEmbeddings._vector("KB: a"))
    bad = chunk("b", vector=[0.0, 0.0, 0.0, 1.0])

    worst, mean = verify_current(FakeCollection([good, bad]), FakeEmbeddings())

    assert worst < 0.999 and worst < mean < 1.0
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)


def test_the_fingerprint_reads_the_runtime_version_and_the_weights_digest() -> None:
    digest = "970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"

    def http(url: str, body: dict[str, str] | None) -> dict[str, object]:
        if url.endswith("/api/version"):
            return {"version": "0.40.14"}
        blob = f"/root/.ollama/models/blobs/sha256-{digest}"
        return {"modelfile": f"# Modelfile\nFROM {blob}\n"}

    found = embedding_fingerprint("http://127.0.0.1:11435/", "nomic-embed-text", http)

    assert found.ollama_version == "0.40.14"
    assert found.model_digest == digest
    assert found.as_dict()["input_format"] == "source_name: text"


def test_a_modelfile_without_a_weights_line_is_an_error() -> None:
    def http(url: str, body: dict[str, str] | None) -> dict[str, object]:
        return {"version": "0.40.14", "modelfile": "# nothing here"}

    with pytest.raises(RuntimeError, match="weights digest"):
        embedding_fingerprint("http://x", "nomic-embed-text", http)


def test_a_runtime_other_than_the_pin_or_the_weights_is_a_problem() -> None:
    from multimodal_rag.store.fingerprint import (
        EXPECTED_MODEL_DIGEST,
        EmbeddingFingerprint,
        runtime_problems,
    )

    good = EmbeddingFingerprint("0.40.14", EXPECTED_MODEL_DIGEST)

    assert runtime_problems(good, "0.40.14") == []
    assert "not the pinned" in runtime_problems(good, "0.40.20")[0]
    other = EmbeddingFingerprint("0.40.14", "0" * 64)
    assert "weights differ" in runtime_problems(other, "0.40.14")[0]
    assert runtime_problems(good, None) == []


def test_the_pinned_version_is_read_from_the_dockerfile(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from multimodal_rag.store.fingerprint import pinned_ollama_version

    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM x\nARG OLLAMA_VERSION=0.40.14\nRUN y\n")

    assert pinned_ollama_version(dockerfile) == "0.40.14"
    assert pinned_ollama_version(tmp_path / "absent") is None


@pytest.mark.parametrize(
    ("lowest", "status", "refuses"),
    [
        (1.0, "OK", False),
        (0.99995, "OK", False),
        (0.995, "WARN", True),
        (0.83, "FAIL", True),
    ],
)
def test_drift_levels_decide_the_status_and_whether_an_apply_refuses(
    lowest: float, status: str, refuses: bool
) -> None:
    from multimodal_rag.store.fingerprint import EmbeddingFingerprint
    from multimodal_rag.sync.embedding_check import (
        EmbeddingStatus,
        apply_problems,
        drift_status,
    )

    assert drift_status(lowest).value == status
    fingerprint = EmbeddingFingerprint("0.40.14", "d" * 64)
    found = EmbeddingStatus(fingerprint, [], lowest, lowest)
    assert bool(apply_problems(found)) is refuses


def test_a_wrong_runtime_also_blocks_an_apply_even_when_vectors_match() -> None:
    from multimodal_rag.store.fingerprint import EmbeddingFingerprint
    from multimodal_rag.sync.embedding_check import EmbeddingStatus, apply_problems

    problem = "Ollama 0.40.16 is not the pinned"
    status = EmbeddingStatus(
        EmbeddingFingerprint("0.40.16", "d" * 64), [problem], 1.0, 1.0
    )

    assert apply_problems(status) == [problem]
