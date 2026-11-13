"""Writes to the hosted collection must be opt-in; local writes are unaffected."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from multimodal_rag.env_check import duplicate_env_keys
from multimodal_rag.ingest import __main__ as ingest_main
from multimodal_rag.store.weaviate import WeaviateStore

HOSTED = "https://cluster.c0.weaviate.cloud"


def _hosted_store(**kwargs: object) -> WeaviateStore:
    with patch(
        "multimodal_rag.store.weaviate.weaviate.connect_to_weaviate_cloud",
        return_value=MagicMock(),
    ):
        return WeaviateStore(
            weaviate_url=HOSTED,
            embeddings=MagicMock(),
            weaviate_mode="cloud",
            weaviate_api_key="key",
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.ensure_collection(),
        lambda s: s.add_chunks([MagicMock()]),
        lambda s: s.delete_by_source("https://x"),
        lambda s: s.delete_by_source_type("web"),
        lambda s: s.delete_collection(),
    ],
)
def test_hosted_store_refuses_every_write(call: object) -> None:
    store = _hosted_store()

    with pytest.raises(PermissionError, match="hosted Weaviate"):
        call(store)  # type: ignore[operator]


def test_hosted_store_still_reads() -> None:
    store = _hosted_store()
    store._client.collections.get.return_value.aggregate.over_all.return_value = (
        MagicMock(total_count=7)
    )

    assert store.count() == 7


def test_hosted_write_is_possible_when_explicitly_allowed() -> None:
    store = _hosted_store(allow_hosted_write=True)
    store._client.collections.exists.return_value = False

    store.ensure_collection()

    store._client.collections.create.assert_called_once()


def test_local_store_writes_are_unaffected() -> None:
    with patch(
        "multimodal_rag.store.weaviate.weaviate.connect_to_local",
        return_value=MagicMock(),
    ):
        store = WeaviateStore(
            weaviate_url="http://localhost:8080", embeddings=MagicMock()
        )
    store._client.collections.exists.return_value = True

    store.delete_collection()

    store._client.collections.delete.assert_called_once()


def test_ingest_refuses_to_run_against_hosted_mode() -> None:
    settings = MagicMock()
    settings.weaviate_mode = "cloud"
    with (
        patch.object(ingest_main, "AppSettings", return_value=settings),
        patch.object(ingest_main, "configure_logging"),
        patch.object(ingest_main, "_run") as run,
        pytest.raises(SystemExit) as exit_info,
    ):
        ingest_main.run()

    assert exit_info.value.code == 1
    run.assert_not_called()


def test_duplicate_env_keys_reports_names_and_lines_not_values(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "VISION_MODEL=alpha\n"
        "SAME=1\n"
        "VISION_MODEL=beta\n"
        "SAME=1\n"
        "ONLY_ONCE=x\n"
    )

    result = duplicate_env_keys(env)

    assert result == {"VISION_MODEL": [2, 4]}
    assert "alpha" not in str(result)


def test_duplicate_env_keys_handles_a_missing_file(tmp_path: Path) -> None:
    assert duplicate_env_keys(tmp_path / "absent.env") == {}
