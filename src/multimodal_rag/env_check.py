"""Detect settings defined more than once in a .env file."""

from __future__ import annotations

from pathlib import Path


def duplicate_env_keys(path: Path) -> dict[str, list[int]]:
    """Return keys defined more than once with different values, with line numbers.

    Only names and line numbers are reported, never values, because the file holds
    secrets. The last definition wins when the file is loaded, so an earlier line that
    looks active is silently ignored.
    """
    seen: dict[str, list[tuple[int, str]]] = {}
    if not path.exists():
        return {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, _, value = text.partition("=")
        seen.setdefault(key.strip(), []).append((number, value.strip()))
    return {
        key: [number for number, _ in entries]
        for key, entries in seen.items()
        if len({value for _, value in entries}) > 1
    }
