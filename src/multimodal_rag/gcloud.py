"""Thin wrapper around the gcloud CLI for the maintainer tools."""

from __future__ import annotations

import subprocess


def run_gcloud(args: list[str], project: str = "", timeout: int = 600) -> str:
    command = ["gcloud", *args]
    if project:
        command += ["--project", project]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gcloud CLI not found; install Google Cloud SDK") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"gcloud {' '.join(args[:3])} failed: {exc.stderr.strip()}"
        ) from exc
    return result.stdout.strip()
