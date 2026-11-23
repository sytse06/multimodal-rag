import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from multimodal_rag.checks import Check, Status
from multimodal_rag.store import compat
from multimodal_rag.store.weaviate import (
    COLLECTION_NAME,
    EXPECTED_PROPERTIES,
    OPTIONAL_PROPERTIES,
)


def test_the_newest_hosted_backup_is_picked(tmp_path: Path) -> None:
    for stamp in ("20261117T125500Z", "20261117T134823Z", "20261101T090000Z"):
        (tmp_path / f"promote-hosted-{stamp}" / "snapshot-hosted-after").mkdir(
            parents=True
        )
    (tmp_path / "promote-hosted-20261201T000000Z" / "snapshot-hosted-before").mkdir(
        parents=True
    )

    newest = compat.newest_snapshot(tmp_path)

    assert newest.parent.name == "promote-hosted-20261117T134823Z"


def test_no_backup_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--snapshot-dir"):
        compat.newest_snapshot(tmp_path)


def test_the_container_binds_loopback_only_and_pins_the_version() -> None:
    command = compat.container_command("1.40.5", "wv-compat-8082", 8082, 50053)

    assert command[-1] == "cr.weaviate.io/semitechnologies/weaviate:1.40.5"
    assert "127.0.0.1:8082:8080" in command and "127.0.0.1:50053:50051" in command
    assert "-v" not in command


def test_an_upgrade_copy_is_mounted_as_a_volume() -> None:
    command = compat.container_command("1.40.5", "n", 8082, 50053, volume="scratch")

    assert command[command.index("-v") + 1] == "scratch:/var/lib/weaviate"


@pytest.mark.parametrize(
    "ports", [(8080, 50053), (8081, 50053), (8082, 50051), (8082, 50052)]
)
def test_the_real_local_and_rehearsal_ports_are_refused(ports: tuple[int, int]) -> None:
    with pytest.raises(compat.Refused), compat.server("1.40.5", *ports):
        pytest.fail("a protected port must be refused before docker is called")


def fake_store(*, exists: bool = False) -> Any:
    store = MagicMock()
    store.client.collections.exists.return_value = exists
    config = SimpleNamespace(
        properties=[
            SimpleNamespace(name=n, data_type=SimpleNamespace(value=t))
            for n, t in {**EXPECTED_PROPERTIES, **OPTIONAL_PROPERTIES}.items()
        ],
        vector_config={"default": object()},
    )
    store.client.collections.get.return_value.config.get.return_value = config
    return store


def test_a_target_that_already_holds_the_collection_is_refused() -> None:
    with pytest.raises(compat.Refused, match=COLLECTION_NAME):
        compat.run_checks(fake_store(exists=True), Path("snap"))


def test_a_failed_restore_fails_the_dependent_checks_instead_of_hiding_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError('unsupported vectorIndexType "hfresh"')

    monkeypatch.setattr(compat, "restore_snapshot", refuse)
    monkeypatch.setattr(compat, "load_snapshot", lambda path: {})

    checks = compat.run_checks(fake_store(), Path("snap"))

    by_name = {c.name: c for c in checks}
    assert by_name["create the collection"].status is Status.OK
    assert by_name["restore the backup with the hfresh index"].status is Status.FAIL
    assert "hfresh" in by_name["restore the backup with the hfresh index"].detail
    later = [c for c in checks if c.detail == "not run: the restore failed"]
    assert len(later) == 5 and all(c.status is Status.FAIL for c in later)


def test_a_wrong_property_type_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = fake_store()
    config = store.client.collections.get.return_value.config.get.return_value
    config.properties[0] = SimpleNamespace(
        name=config.properties[0].name, data_type=SimpleNamespace(value="int")
    )
    monkeypatch.setattr(compat, "load_snapshot", lambda path: {})
    monkeypatch.setattr(compat, "restore_snapshot", lambda *a, **k: None)

    first = compat.run_checks(store, Path("s"))[0]

    assert first.status is Status.FAIL and "properties differ" in first.detail


def run_main(monkeypatch: pytest.MonkeyPatch, checks: list[Check]) -> int:
    argv = ["compat", "--version", "1.40.5", "--snapshot-dir", "s"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(compat, "server", lambda *a, **k: contextlib.nullcontext("n"))
    monkeypatch.setattr(compat, "WeaviateStore", lambda **k: MagicMock())
    monkeypatch.setattr(compat, "run_checks", lambda store, snapshot: checks)
    with pytest.raises(SystemExit) as stop:
        compat.main()
    return int(stop.value.code or 0)


def test_the_exit_code_is_zero_when_every_check_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert run_main(monkeypatch, [Check("a", Status.OK, "")]) == 0


def test_the_exit_code_is_one_when_any_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = [Check("a", Status.OK, ""), Check("b", Status.FAIL, "")]

    assert run_main(monkeypatch, checks) == 1
