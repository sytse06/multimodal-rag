"""The embedding runtime is pinned in two places that must stay equal."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dockerfile_version() -> str:
    text = (ROOT / "Dockerfile").read_text()
    match = re.search(r"^ARG OLLAMA_VERSION=(\S+)", text, re.M)
    assert match, "the Dockerfile no longer pins OLLAMA_VERSION"
    return match.group(1)


def _compose_version() -> str:
    text = (ROOT / "compose.yaml").read_text()
    match = re.search(r"image:\s*ollama/ollama:(\S+)", text)
    assert match, "compose.yaml no longer pins an ollama/ollama image"
    return match.group(1)


def test_compose_and_dockerfile_pin_the_same_ollama_version() -> None:
    assert _compose_version() == _dockerfile_version()


def test_the_pin_is_a_version_not_a_moving_tag() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", _compose_version())


def test_the_embedding_runtime_is_only_reachable_from_this_machine() -> None:
    compose = (ROOT / "compose.yaml").read_text()

    assert '"127.0.0.1:11435:11434"' in compose
