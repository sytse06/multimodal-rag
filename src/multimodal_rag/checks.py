"""The result type shared by the preflight, the retrieval gate and the apply.

It lives in its own module on purpose: a module run with ``python -m`` is loaded twice
(as ``__main__`` and under its own name), and two copies of an enum never compare
equal with ``is``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
