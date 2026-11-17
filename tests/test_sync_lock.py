"""Tests for the bucket run lock."""

import pytest

from multimodal_rag.sync.lock import LockHeld, RunLock


class FakeBucket:
    def __init__(self) -> None:
        self.exists = False
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], project: str = "") -> str:
        self.calls.append(args)
        if args[1] == "cp":
            if self.exists:
                raise RuntimeError("gcloud storage cp failed: 412 Precondition Failed")
            self.exists = True
            return ""
        if args[1] == "cat":
            return '{"run_id": "other-run"}'
        if args[1] == "rm":
            self.exists = False
        return ""


def test_the_lock_is_created_only_if_absent_and_released() -> None:
    bucket = FakeBucket()
    lock = RunLock("b", "p", "run-1", bucket)

    lock.acquire()
    assert bucket.exists
    assert "--if-generation-match=0" in bucket.calls[0]
    lock.release()

    assert not bucket.exists


def test_a_held_lock_refuses_and_names_the_holder() -> None:
    bucket = FakeBucket()
    bucket.exists = True

    with pytest.raises(LockHeld, match="other-run"):
        RunLock("b", "p", "run-2", bucket).acquire()


def test_breaking_the_lock_takes_it_over() -> None:
    bucket = FakeBucket()
    bucket.exists = True
    lock = RunLock("b", "p", "run-2", bucket)

    lock.acquire(break_lock=True)

    assert bucket.exists
    lock.release()


def test_releasing_a_lock_that_was_never_taken_removes_nothing() -> None:
    bucket = FakeBucket()
    bucket.exists = True

    RunLock("b", "p", "run-2", bucket).release()

    assert bucket.exists


def test_the_context_manager_releases_even_when_the_run_fails() -> None:
    bucket = FakeBucket()

    with pytest.raises(ValueError), RunLock("b", "p", "run-3", bucket):
        raise ValueError("boom")

    assert not bucket.exists


def test_an_unrelated_gcloud_error_is_not_mistaken_for_a_held_lock() -> None:
    def broken(args: list[str], project: str = "") -> str:
        raise RuntimeError("gcloud storage cp failed: 403 permission denied")

    with pytest.raises(RuntimeError, match="403"):
        RunLock("b", "p", "run-4", broken).acquire()
