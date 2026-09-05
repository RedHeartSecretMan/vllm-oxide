from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from golden_gen.stages import verify_stage_marker, write_stage_marker

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_stage_marker_rejects_executable_renamed_into_evidence(tmp_path, ticket_artifact_root):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    def git(*args):
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                *args,
            ],
            check=True,
        )

    git("config", "diff.renames", "true")
    (repo / "runner.py").write_text("print('measured executable')\n")
    git("add", ".")
    git("commit", "-qm", "measurement")
    (ticket_artifact_root / "env").mkdir(parents=True)
    (ticket_artifact_root / "env/runtime.json").write_bytes(b"runtime")
    write_stage_marker(ticket_artifact_root, "env", repo)
    (repo / "docs/adr").mkdir(parents=True)
    (repo / "runner.py").rename(repo / "docs/adr/renamed-source.md")
    git("add", "-A")
    git("commit", "-qm", "hide source deletion as evidence rename")

    with pytest.raises(ValueError, match="executable inputs changed"):
        verify_stage_marker(ticket_artifact_root, "env", repo)


def test_stage_marker_binds_outputs_commit_tree_and_predecessor(ticket_artifact_root):
    run_root = ticket_artifact_root
    env_dir = run_root / "env"
    env_dir.mkdir(parents=True)
    runtime = env_dir / "runtime.json"
    runtime.write_bytes(b"runtime")

    env_marker = write_stage_marker(run_root, "env", REPO_ROOT)
    env_record = json.loads(env_marker.read_text())
    assert verify_stage_marker(run_root, "env", REPO_ROOT) == env_marker

    assert env_record["stage"] == "env"
    assert len(env_record["generator_commit"]) == 40
    assert len(env_record["generator_tree"]) == 40
    assert env_record["outputs"] == [
        {
            "path": "env/runtime.json",
            "size": 7,
            "sha256": hashlib.sha256(b"runtime").hexdigest(),
        }
    ]

    generate = run_root / "generate"
    generate.mkdir()
    (generate / "evidence.json").write_bytes(b"generation")
    generated_marker = write_stage_marker(run_root, "generate", REPO_ROOT)
    generated_record = json.loads(generated_marker.read_text())
    assert (
        generated_record["predecessor_marker_sha256"]
        == hashlib.sha256(env_marker.read_bytes()).hexdigest()
    )
    assert verify_stage_marker(run_root, "generate", REPO_ROOT) == generated_marker

    runtime.write_bytes(b"changed predecessor")
    with pytest.raises(ValueError, match="output identity changed"):
        verify_stage_marker(run_root, "generate", REPO_ROOT)
    runtime.write_bytes(b"runtime")

    unexpected = generate / "unexpected.json"
    unexpected.write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="output set changed"):
        verify_stage_marker(run_root, "generate", REPO_ROOT)
    unexpected.unlink()

    (generate / "evidence.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="output identity changed"):
        verify_stage_marker(run_root, "generate", REPO_ROOT)

    with pytest.raises(FileExistsError, match="already exists"):
        write_stage_marker(run_root, "generate", REPO_ROOT)
