"""Stable identity and fingerprint values for stored chunks (SYNC-001)."""

import re
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

_YOUTUBE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:[^\s#]*&)?v=|embed/))([A-Za-z0-9_-]{11})",
    re.IGNORECASE,
)
_WATCH = "https://www.youtube.com/watch?v={}"


def canonical_url(url: str) -> str:
    """One URL per video: ``youtu.be/<id>`` and ``watch?v=<id>`` are the same video.

    Any other URL is returned unchanged. Only identity uses this; the stored
    ``source_url`` is never rewritten, because chunk UUIDs derive from it.
    """
    match = _YOUTUBE.search(url)
    return _WATCH.format(match.group(1)) if match else url


def is_video_url(url: str) -> bool:
    """Whether the URL is a YouTube shape that ``canonical_url`` can identify.

    ``shorts/``, ``live/`` and the like are not, so they would get an identity of their
    own and could never be matched with the same video registered another way.
    """
    return _YOUTUBE.search(url) is not None


def document_id(source_url: str) -> str:
    return str(uuid5(NAMESPACE_URL, canonical_url(source_url)))


def content_hash(text: str) -> str:
    """Full SHA-256 of the stored text; describes what is stored, not its source."""
    return sha256(text.encode()).hexdigest()


def embedded_video_ids(text: str) -> list[str]:
    """``document_id`` of every YouTube video linked in the text, sorted, unique."""
    return sorted({document_id(_WATCH.format(m)) for m in _YOUTUBE.findall(text)})
