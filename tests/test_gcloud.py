"""Tests for the gcloud CLI wrapper."""

import subprocess
from unittest.mock import patch

import pytest

from multimodal_rag import gcloud
from multimodal_rag.gcloud import run_gcloud


def test_command_includes_project_and_returns_stripped_output() -> None:
    done = subprocess.CompletedProcess([], 0, stdout=" value\n", stderr="")
    with patch.object(gcloud.subprocess, "run", return_value=done) as run:
        assert run_gcloud(["storage", "ls"], "proj") == "value"

    assert run.call_args.args[0] == ["gcloud", "storage", "ls", "--project", "proj"]


def test_missing_gcloud_is_reported() -> None:
    with patch.object(gcloud.subprocess, "run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="gcloud CLI not found"):
            run_gcloud(["storage", "cp"])


def test_failure_includes_stderr() -> None:
    err = subprocess.CalledProcessError(1, "gcloud", stderr="denied")
    with patch.object(gcloud.subprocess, "run", side_effect=err):
        with pytest.raises(RuntimeError, match="denied"):
            run_gcloud(["storage", "cp", "a", "b"], "proj")
