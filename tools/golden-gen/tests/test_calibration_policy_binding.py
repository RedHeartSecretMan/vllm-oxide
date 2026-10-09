"""A policy hash pointer is not permission to change execution or supervision."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from golden_gen.layered_manifest import _require_calibration_definition_changes
from golden_gen.layered_release import POLICY_PATH
from golden_gen.supervision import POLICY_PATH as SUPERVISION_POLICY_PATH


@pytest.mark.parametrize(
    "mutation",
    [
        "valid",
        "ram",
        "new-field",
        "false-to-zero",
        "old-hash",
        "new-hash",
        "missing-hash",
        "non-object",
        "malformed",
        "deleted",
        "symlink",
        "mode",
        "execution",
        "rename",
    ],
)
def test_only_bound_numerical_policy_pointer_can_change(tmp_path: Path, mutation: str) -> None:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "CPU Test")
    git("config", "user.email", "cpu@example.invalid")
    git("config", "diff.renames", "true")
    numerical = tmp_path / POLICY_PATH
    numerical.parent.mkdir(parents=True)
    numerical.write_text('{"values":{"a_mean":0.002},"binding":"old"}\n')
    supervision = tmp_path / SUPERVISION_POLICY_PATH
    policy = dict(
        numerical_policy_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        ram_floor_bytes=16 * 1024**3,
        require_fresh_before=True,
        retained_owner_count=0,
        disabled=False,
    )
    if mutation == "old-hash":
        policy["numerical_policy_sha256"] = "0" * 64
    supervision.write_text(json.dumps(policy))
    if mutation == "rename":
        (tmp_path / "runtime.py").write_text("actual_execution = True\n")
    git("add", ".")
    git("commit", "-qm", "calibration")
    before = git("rev-parse", "HEAD")
    numerical.write_text('{"values":{"a_mean":0.002},"binding":"approved"}\n')
    policy["numerical_policy_sha256"] = hashlib.sha256(numerical.read_bytes()).hexdigest()
    if mutation == "ram":
        policy["ram_floor_bytes"] = 1
    elif mutation == "new-field":
        policy["waive_guard"] = True
    elif mutation == "false-to-zero":
        policy["disabled"] = 0
    elif mutation == "new-hash":
        policy["numerical_policy_sha256"] = "f" * 64
    elif mutation == "missing-hash":
        del policy["numerical_policy_sha256"]
    supervision.write_text(json.dumps(policy))
    if mutation == "non-object":
        supervision.write_text("[]")
    elif mutation == "malformed":
        supervision.write_text("{")
    elif mutation == "deleted":
        supervision.unlink()
    elif mutation == "symlink":
        supervision.unlink()
        supervision.symlink_to("other.json")
    elif mutation == "mode":
        supervision.chmod(0o755)
    elif mutation == "execution":
        (tmp_path / "runtime.py").write_text("changed_execution = True\n")
    if mutation == "rename":
        (tmp_path / "docs/adr").mkdir()
        git("mv", "runtime.py", "docs/adr/moved.md")
    git("add", ".")
    git("commit", "-qm", "policy binding")
    after = git("rev-parse", "HEAD")
    if mutation == "valid":
        _require_calibration_definition_changes(tmp_path, before, after)
    else:
        with pytest.raises(ValueError, match="execution source changed"):
            _require_calibration_definition_changes(tmp_path, before, after)
