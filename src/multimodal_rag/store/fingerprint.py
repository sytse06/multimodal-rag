"""What produced the vectors: the embedding runtime version and the model's weights.

The same model gives different vectors on different Ollama versions (a casing change
alone moved capital-heavy chunks by up to 0.3 cosine), so a vector is only comparable
with a query embedded by the same runtime. This is the identity a run records.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

INPUT_FORMAT = "source_name: text"  # what add_chunks embeds, see WeaviateStore
# The nomic-embed-text weights every runtime so far has served (byte-identical on the
# registry, the Mac and production on 2026-11-17).
EXPECTED_MODEL_DIGEST = (
    "970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
)
DOCKERFILE = Path(__file__).resolve().parents[3] / "Dockerfile"


@dataclass(frozen=True)
class EmbeddingFingerprint:
    ollama_version: str
    model_digest: str  # the weights blob, as in the model's FROM line
    input_format: str = INPUT_FORMAT

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


Http = Callable[[str, dict[str, str] | None], dict[str, object]]


def _http(url: str, body: dict[str, str] | None = None) -> dict[str, object]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
        result: dict[str, object] = json.loads(response.read())
        return result


def embedding_fingerprint(
    base_url: str, model: str, http: Http = _http
) -> EmbeddingFingerprint:
    base = base_url.rstrip("/")
    version = str(http(f"{base}/api/version", None)["version"])
    modelfile = str(http(f"{base}/api/show", {"model": model})["modelfile"])
    match = re.search(r"^FROM\s+\S*sha256-([0-9a-f]{64})", modelfile, re.M)
    if not match:
        raise RuntimeError(f"cannot find the weights digest of {model} at {base}")
    return EmbeddingFingerprint(ollama_version=version, model_digest=match.group(1))


def pinned_ollama_version(dockerfile: Path = DOCKERFILE) -> str | None:
    """The Ollama version production builds with (the Dockerfile's ARG)."""
    if not dockerfile.exists():
        return None
    match = re.search(r"^ARG OLLAMA_VERSION=(\S+)", dockerfile.read_text(), re.M)
    return match.group(1) if match else None


def runtime_problems(
    found: EmbeddingFingerprint, pinned_version: str | None
) -> list[str]:
    """Ways the embedding runtime differs from what production queries with."""
    problems = []
    if pinned_version and found.ollama_version != pinned_version:
        problems.append(
            f"Ollama {found.ollama_version} is not the pinned {pinned_version}"
        )
    if found.model_digest != EXPECTED_MODEL_DIGEST:
        problems.append("the nomic-embed-text weights differ from the expected blob")
    return problems
