"""Ask the site whether a page that left the sitemap is really gone.

A page missing from the sitemap is only a hint: the sitemap can drop pages for reasons
that have nothing to do with the page. The page's own address settles it. 404 or 410 is
gone, a redirect means it moved, and a 200 means it is still served, so the sitemap
simply omitted it and nothing may be deleted.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

GONE = "gone"
REDIRECT = "redirect"
STILL_SERVED = "still-served"
UNKNOWN = "unknown"
SOURCE_DROPPED = "source-dropped"  # the registry no longer lists the source
NOT_CHECKED = "not-checked"


@dataclass(frozen=True)
class Probe:
    evidence: str
    http_status: int | None = None
    redirect_to: str | None = None
    error: str | None = None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # report the 3xx instead of following it


def probe_url(url: str, timeout: float = 20.0, attempts: int = 2) -> Probe:
    """One GET without following redirects; a transient failure is tried once more."""
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(url, headers={"User-Agent": "multimodal-rag-sync"})
    last = "no attempt"
    for _ in range(attempts):
        try:
            with opener.open(request, timeout=timeout) as response:
                return Probe(STILL_SERVED, response.status)
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                return Probe(GONE, exc.code)
            if 300 <= exc.code < 400:
                target = exc.headers.get("Location")
                return Probe(REDIRECT, exc.code, redirect_to=target)
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
    return Probe(UNKNOWN, error=last)


ProbeFn = Callable[[str], Probe]
