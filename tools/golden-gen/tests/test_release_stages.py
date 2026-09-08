from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "tools/validate-release.sh"


def test_release_stage_rejects_traversal_and_symlink_before_creating_run(ticket_artifact_root):
    run = ticket_artifact_root
    traversal = str(run.parent / ".." / "outside")
    result = subprocess.run(["bash", str(SCRIPT), "env", traversal], capture_output=True)
    assert result.returncode == 2
    assert b"canonical" in result.stderr
    alias = run.parent / (run.name + "-alias")
    alias.symlink_to("/tmp")
    try:
        result = subprocess.run(
            ["bash", str(SCRIPT), "env", str(alias / run.name)], capture_output=True
        )
        assert result.returncode == 2
        assert b"canonical" in result.stderr
        assert not (Path("/tmp") / run.name).exists()
    finally:
        alias.unlink()


def test_observe_primary_process_failure_does_not_write_successor_marker(
    tmp_path, ticket_artifact_root
):
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    shutil.copyfile(SCRIPT, repo / "tools/validate-release.sh")
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "stage"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    run = ticket_artifact_root
    python = run / "env/.venv/bin/python"
    python.parent.mkdir(parents=True)
    log = tmp_path / "commands.log"
    python.write_text(f"""#!{sys.executable}
import pathlib, sys
command = sys.argv[3]
with pathlib.Path({str(log)!r}).open("a") as output:
    output.write(command + "\\n")
if command == "guard":
    raise SystemExit(7)
if command == "observe":
    pathlib.Path(sys.argv[sys.argv.index("--output") + 1]).write_text("{{}}")
    raise SystemExit(3)
""")
    python.chmod(0o755)
    executables = tmp_path / "bin"
    executables.mkdir()
    cargo = executables / "cargo"
    cargo.write_text("#!/bin/sh\nexit 0\n")
    cargo.chmod(0o755)
    result = subprocess.run(
        ["bash", str(repo / "tools/validate-release.sh"), "observe", str(run)],
        env={**os.environ, "PATH": f"{executables}:{os.environ['PATH']}"},
        capture_output=True,
    )
    assert result.returncode == 7, result.stderr.decode()
    assert log.read_text().splitlines() == ["verify-stage-marker", "guard"]
    assert not (run / "markers/observe.complete.json").exists()


@pytest.mark.parametrize("stage", ["authoritative", "publish"])
def test_legacy_release_stages_reject_before_loading_runtime(tmp_path, stage):
    result = subprocess.run(
        ["bash", str(SCRIPT), stage, str(tmp_path / "no-run")],
        env={**os.environ, "VLLM_OXIDE_ALLOW_GOLDEN_PUBLISH": "goldens-v0.2"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "layered-accuracy-v1" in result.stderr
    assert list(tmp_path.iterdir()) == []
