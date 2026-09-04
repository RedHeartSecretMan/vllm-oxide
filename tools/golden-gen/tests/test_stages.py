from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from golden_gen.stages import write_stage_marker

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_stage_marker_binds_outputs_commit_tree_and_predecessor(ticket_artifact_root):
    run_root = ticket_artifact_root
    env_dir = run_root / "env"
    env_dir.mkdir(parents=True)
    runtime = env_dir / "runtime.json"
    runtime.write_bytes(b"runtime")

    env_marker = write_stage_marker(run_root, "env", REPO_ROOT)
    env_record = json.loads(env_marker.read_text())

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

    with pytest.raises(FileExistsError, match="already exists"):
        write_stage_marker(run_root, "generate", REPO_ROOT)
