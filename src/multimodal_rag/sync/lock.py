"""One sync run at a time: a create-only lock object in the state bucket."""

from __future__ import annotations

import json
import socket
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from multimodal_rag.gcloud import run_gcloud

LOCK_OBJECT = "locks/sync.lock"

Runner = Callable[..., str]


class LockHeld(RuntimeError):
    """Another run holds the lock."""


class RunLock:
    def __init__(
        self, bucket: str, project: str, run_id: str, runner: Runner = run_gcloud
    ) -> None:
        self._url = f"gs://{bucket}/{LOCK_OBJECT}"
        self._project = project
        self._run_id = run_id
        self._runner = runner
        self._held = False

    def acquire(self, *, break_lock: bool = False) -> None:
        if break_lock:
            self._runner(["storage", "rm", "--quiet", self._url], self._project)
        with tempfile.TemporaryDirectory(prefix="lock-") as tmp:
            body = Path(tmp) / "sync.lock"
            body.write_text(
                json.dumps(
                    {
                        "run_id": self._run_id,
                        "host": socket.gethostname(),
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            )
            try:
                self._runner(
                    ["storage", "cp", str(body), self._url, "--if-generation-match=0"],
                    self._project,
                )
            except RuntimeError as exc:
                if "412" not in str(exc) and "recondition" not in str(exc):
                    raise
                raise LockHeld(f"another run holds the lock: {self.holder()}") from exc
        self._held = True

    def holder(self) -> str:
        try:
            return self._runner(["storage", "cat", self._url], self._project)
        except RuntimeError:
            return "unknown holder"

    def release(self) -> None:
        if self._held:
            self._runner(["storage", "rm", "--quiet", self._url], self._project)
            self._held = False

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
