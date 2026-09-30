import os
import subprocess
from pathlib import Path

ENTRYPOINT = Path(__file__).parents[1] / "space-entrypoint.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_space_entrypoint_starts_ollama_and_launches_app(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    event_log = tmp_path / "events.log"
    app_log = tmp_path / "app.log"

    _write_executable(
        bin_dir / "ollama",
        '#!/bin/sh\nprintf "ollama:%s\\n" "$*" >> "$EVENT_LOG"\n',
    )
    _write_executable(bin_dir / "curl", "#!/bin/sh\nexit 0\n")
    _write_executable(
        bin_dir / "python",
        '#!/bin/sh\nprintf "python:%s|%s|%s\\n" "$*" '
        '"$EMBEDDING_PROVIDER" "$OLLAMA_BASE_URL" >> "$APP_LOG"\n',
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "EVENT_LOG": str(event_log),
            "APP_LOG": str(app_log),
            "WEAVIATE_MODE": "cloud",
            "WEAVIATE_URL": "https://example.weaviate.cloud",
            "WEAVIATE_VIEWER_API_KEY": "test-viewer-key",
        }
    )

    result = subprocess.run(
        ["bash", str(ENTRYPOINT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert event_log.read_text(encoding="utf-8").splitlines() == ["ollama:serve"]
    assert app_log.read_text(encoding="utf-8").strip() == (
        "python:app.py|ollama|http://127.0.0.1:11434"
    )


def test_space_entrypoint_rejects_non_cloud_weaviate(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update({"WEAVIATE_MODE": "local"})

    result = subprocess.run(
        ["bash", str(ENTRYPOINT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode != 0
    assert "requires WEAVIATE_MODE=cloud" in result.stderr
