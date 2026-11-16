"""Read-only report on whether this machine is ready to run a content update.

Run it with ``make sync-preflight`` before ingesting or promoting. It changes nothing,
never prints a secret value, and exits non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import weaviate
from pydantic import SecretStr
from weaviate.auth import AuthApiKey

from multimodal_rag.env_check import duplicate_env_keys
from multimodal_rag.gcloud import run_gcloud
from multimodal_rag.ingest import registry
from multimodal_rag.ingest.cookies import youtube_cookies
from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.providers import is_registered
from multimodal_rag.store.weaviate import COLLECTION_NAME

COOKIE_WARN_DAYS = 14
BACKUP_WARN_DAYS = 7
LOGIN_COOKIES = ("LOGIN_INFO", "SID", "__Secure-1PSID")
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"


class Status(str, Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"
    MANUAL = "MANUAL"


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


def _has(value: SecretStr | str) -> bool:
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    return bool(raw)


def _get_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def check_tools(settings: AppSettings) -> list[Check]:
    checks = []
    for tool, why in (
        ("docker", "runs the local Weaviate"),
        ("ffmpeg", "extracts video frames and audio"),
        ("gcloud", "bucket, secrets and deploy"),
    ):
        found = shutil.which(tool)
        checks.append(
            Check(tool, Status.OK if found else Status.FAIL, found or f"missing: {why}")
        )
    has_ytdlp = importlib.util.find_spec("yt_dlp") is not None
    checks.append(
        Check("yt-dlp", Status.OK if has_ytdlp else Status.FAIL, "python package")
    )
    return checks


def check_ollama(settings: AppSettings) -> Check:
    if settings.embeddings.provider != "ollama":
        return Check("ollama", Status.OK, "not the embedding provider")
    wanted = settings.embedding_model
    try:
        tags = _get_json(f"{settings.ollama_base_url}/api/tags")
    except (urllib.error.URLError, OSError) as exc:
        return Check("ollama", Status.FAIL, f"unreachable at base URL: {exc}")
    names = [m.get("name", "") for m in tags.get("models", [])]
    if any(n == wanted or n.startswith(f"{wanted}:") for n in names):
        return Check("ollama", Status.OK, f"{wanted} is installed")
    return Check("ollama", Status.FAIL, f"{wanted} is not pulled (ollama pull)")


def check_keys(settings: AppSettings) -> list[Check]:
    required = (
        ("FIRECRAWL_API_KEY", settings.firecrawl_api_key, Status.FAIL, "web crawl"),
        ("MISTRAL_API_KEY", settings.mistral_api_key, Status.WARN, "Voxtral fallback"),
        (
            "OPENROUTER_API_KEY",
            settings.credentials.openrouter_api_key,
            Status.FAIL,
            "vision client",
        ),
        (
            "GEMINI_API_KEY",
            settings.credentials.gemini_api_key,
            Status.WARN,
            "default chat model",
        ),
        (
            "WEAVIATE_ADMIN_API_KEY",
            settings.weaviate_admin_api_key,
            Status.FAIL,
            "backup and promotion",
        ),
        (
            "WEAVIATE_VIEWER_API_KEY",
            settings.weaviate_viewer_api_key,
            Status.WARN,
            "read-only comparison",
        ),
        ("STATE_BUCKET", settings.state_bucket, Status.FAIL, "registry and backups"),
        ("GCP_PROJECT", settings.gcp_project, Status.FAIL, "gcloud calls"),
        (
            "WEAVIATE_HOSTED_URL",
            settings.weaviate_hosted_url,
            Status.FAIL,
            "hosted cluster",
        ),
    )
    return [
        Check(name, Status.OK, "set")
        if _has(value)
        else Check(name, missing, why)
        for name, value, missing, why in required
    ]


def _openrouter_vision_ids() -> set[str]:
    data = _get_json(OPENROUTER_MODELS)["data"]
    return {
        m["id"]
        for m in data
        if "image" in m.get("architecture", {}).get("input_modalities", [])
    }


def check_config(settings: AppSettings) -> list[Check]:
    checks = []
    if settings.weaviate_mode == "local":
        checks.append(Check("weaviate mode", Status.OK, "local"))
    else:
        checks.append(
            Check("weaviate mode", Status.FAIL, "ingestion refuses to run unless local")
        )
    duplicates = duplicate_env_keys(Path(".env"))
    if duplicates:
        detail = "; ".join(f"{k} on lines {v}" for k, v in duplicates.items())
        checks.append(Check(".env duplicates", Status.WARN, detail))
    else:
        checks.append(Check(".env duplicates", Status.OK, "none"))
    chat = f"{settings.chat.provider}:{settings.chat.model}"
    registered = is_registered(settings.chat.provider, settings.chat.model)
    checks.append(
        Check(
            "default chat model",
            Status.OK if registered else Status.WARN,
            chat if registered else f"{chat} is not in the registry",
        )
    )
    vision = settings.vision_model
    if not vision:
        checks.append(Check("vision model", Status.WARN, "not set; grounding is off"))
    else:
        try:
            known = vision in _openrouter_vision_ids()
        except (urllib.error.URLError, OSError, KeyError, ValueError):
            checks.append(Check("vision model", Status.WARN, f"{vision}: not verified"))
        else:
            checks.append(
                Check(
                    "vision model",
                    Status.OK if known else Status.FAIL,
                    vision if known else f"{vision} is not an OpenRouter image model",
                )
            )
    return checks


def check_cookies(settings: AppSettings) -> Check:
    if not settings.youtube_cookies_secret:
        return Check("youtube cookies", Status.WARN, "no secret configured")
    with youtube_cookies(settings) as path:
        rows = [
            line.split("\t")
            for line in Path(path).read_text().splitlines()
            if line.count("\t") >= 6
        ]
    now = time.time()
    expiries = [int(r[4]) for r in rows if r[5] in LOGIN_COOKIES and int(r[4]) > 0]
    if not expiries or max(expiries) < now:
        return Check("youtube cookies", Status.FAIL, "no valid login cookie; re-export")
    days = int((max(expiries) - now) // 86400)
    status = Status.OK if days >= COOKIE_WARN_DAYS else Status.WARN
    return Check("youtube cookies", status, f"login cookie valid for {days} days")


def check_registry(settings: AppSettings) -> Check:
    path = registry.SOURCES_PATH
    if not path.exists():
        return Check("registry", Status.FAIL, "no local copy; run make sources-pull")
    config = registry.validate(path)
    recorded = registry.GENERATION_PATH
    remote = registry._remote(registry._url(settings), settings.gcp_project)
    summary = (
        f"{len(config.youtube)} videos, {len(config.knowledge_bases)} knowledge bases"
    )
    if remote is None:
        return Check("registry", Status.FAIL, "not in the bucket; run the first push")
    if not recorded.exists() or recorded.read_text().strip() != remote[0]:
        return Check("registry", Status.WARN, f"{summary}; local copy is stale (pull)")
    return Check("registry", Status.OK, f"{summary}; matches the bucket")


def check_backup(settings: AppSettings) -> Check:
    out = run_gcloud(
        ["storage", "ls", f"gs://{settings.state_bucket}/snapshots/"],
        settings.gcp_project,
    )
    stamps = sorted(
        line.rstrip("/").rsplit("/", 1)[-1].split("-")[0]
        for line in out.splitlines()
        if line.strip()
    )
    if not stamps:
        return Check("latest backup", Status.FAIL, "no snapshot in the bucket")
    taken = datetime.strptime(stamps[-1], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - taken).days
    status = Status.OK if age <= BACKUP_WARN_DAYS else Status.WARN
    return Check("latest backup", status, f"{stamps[-1]} ({age} days old)")


def fingerprints(collection: Any) -> dict[str, str]:
    """Map each object UUID to a hash of what identifies its content."""
    result = {}
    for obj in collection.iterator():
        props = obj.properties
        text = f"{props.get('text', '')}\x00{props.get('source_url', '')}"
        result[str(obj.uuid)] = hashlib.sha256(text.encode()).hexdigest()
    return result


def compare_collections(local: dict[str, str], hosted: dict[str, str]) -> Check:
    only_local = len(local.keys() - hosted.keys())
    only_hosted = len(hosted.keys() - local.keys())
    differ = sum(1 for k in local.keys() & hosted.keys() if local[k] != hosted[k])
    if not (only_local or only_hosted or differ):
        return Check("local vs hosted", Status.OK, f"identical, {len(local)} chunks")
    return Check(
        "local vs hosted",
        Status.WARN,
        f"local {len(local)}, hosted {len(hosted)}: {only_local} only local,"
        f" {only_hosted} only hosted, {differ} changed",
    )


def _connect_hosted(settings: AppSettings) -> Any:
    key = settings.weaviate_viewer_api_key
    if not _has(key):
        key = settings.weaviate_admin_api_key
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.weaviate_hosted_url,
        auth_credentials=AuthApiKey(key.get_secret_value()),
    )


def check_weaviate(settings: AppSettings) -> list[Check]:
    url = urlparse(str(settings.weaviate_url))
    local = weaviate.connect_to_local(
        host=url.hostname or "localhost", port=url.port or 8080
    )
    hosted = _connect_hosted(settings)
    try:
        local_version = str(local.get_meta().get("version"))
        hosted_version = str(hosted.get_meta().get("version"))
        drift = local_version != hosted_version
        checks = [
            Check(
                "weaviate versions",
                Status.WARN if drift else Status.OK,
                f"local {local_version}, hosted {hosted_version}",
            ),
            compare_collections(
                fingerprints(local.collections.get(COLLECTION_NAME)),
                fingerprints(hosted.collections.get(COLLECTION_NAME)),
            ),
        ]
    finally:
        local.close()
        hosted.close()
    return checks


def _safe(name: str, run: Callable[[], Check | list[Check]]) -> list[Check]:
    try:
        result = run()
    except Exception as exc:  # a failed probe is a result, not a crash
        return [Check(name, Status.FAIL, f"{type(exc).__name__}: {exc}")]
    return [result] if isinstance(result, Check) else result


def run_preflight(settings: AppSettings, *, local_only: bool = False) -> list[Check]:
    steps: list[tuple[str, Callable[[], Check | list[Check]]]] = [
        ("tools", lambda: check_tools(settings)),
        ("ollama", lambda: check_ollama(settings)),
        ("keys", lambda: check_keys(settings)),
        ("config", lambda: check_config(settings)),
    ]
    if not local_only:
        steps += [
            ("youtube cookies", lambda: check_cookies(settings)),
            ("registry", lambda: check_registry(settings)),
            ("latest backup", lambda: check_backup(settings)),
            ("weaviate", lambda: check_weaviate(settings)),
        ]
    checks = [c for name, run in steps for c in _safe(name, run)]
    checks.append(
        Check(
            "gemini quota and spend cap",
            Status.MANUAL,
            "not queryable here; check AI Studio before a bulk run",
        )
    )
    return checks


def format_report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = [f"[{c.status.value:^6}] {c.name:<{width}}  {c.detail}" for c in checks]
    counts = {s: sum(1 for c in checks if c.status is s) for s in Status}
    lines.append(
        f"\n{counts[Status.OK]} ok, {counts[Status.WARN]} warnings,"
        f" {counts[Status.FAIL]} failed, {counts[Status.MANUAL]} manual"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true", help="skip cloud checks")
    args = parser.parse_args()
    checks = run_preflight(AppSettings(), local_only=args.local_only)
    print(format_report(checks))
    sys.exit(1 if any(c.status is Status.FAIL for c in checks) else 0)


if __name__ == "__main__":
    main()
