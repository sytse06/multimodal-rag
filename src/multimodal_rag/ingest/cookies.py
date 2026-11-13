"""Resolve the YouTube cookies file for yt-dlp, optionally from Secret Manager."""

import logging
import os
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from multimodal_rag.models.config import AppSettings

logger = logging.getLogger(__name__)


def _read_secret(name: str, project: str) -> str:
    command = ["gcloud", "secrets", "versions", "access", "latest", "--secret", name]
    if project:
        command += ["--project", project]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=60
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "gcloud CLI not found: install the Google Cloud SDK, or set "
            "YOUTUBE_COOKIES_FILE and leave YOUTUBE_COOKIES_SECRET empty"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Could not read secret {name!r} (project {project or 'default'!r}): "
            f"{exc.stderr.strip()[:300]}"
        ) from exc
    return result.stdout


@contextmanager
def youtube_cookies(settings: AppSettings) -> Iterator[str]:
    """Yield the cookies file path; a Secret Manager secret wins over a local file.

    A secret is written to a private temporary file for the duration of the block and
    deleted afterwards, so no cookies stay on disk between runs.
    """
    secret = settings.youtube_cookies_secret
    if not secret:
        yield settings.youtube_cookies_file
        return

    content = _read_secret(secret, settings.gcp_project)
    fd, name = tempfile.mkstemp(prefix="yt-cookies-", suffix=".txt")  # mode 0600
    path = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        logger.info("Using YouTube cookies from Secret Manager secret %s", secret)
        yield str(path)
    finally:
        path.unlink(missing_ok=True)
