import json
import subprocess
from pathlib import Path

import pytest

from golden_gen.layered_artifacts import source_identity
from golden_gen.release_cpu import PRESSURE, collect_cpu, gate_commands, validate_cpu


@pytest.mark.parametrize("failure", [None, "dependencies"])
def test_cpu_producer_runs_fixed_vectors_and_retains_failed_raw_logs(
    tmp_path, monkeypatch, failure
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo / "source").write_text("CPU fixture source\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=CPU",
            "-c",
            "user.email=cpu@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    actual_run = subprocess.run
    calls = []
    commands = gate_commands("/synthetic/python")
    outputs = dict.fromkeys(commands, "")
    for name in ("default-core", "workspace", "internal-golden"):
        outputs[name] = (
            f"test llm::{PRESSURE} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored\n"
        )
    outputs.update(
        python="1 passed in 0.01s",
        ruff="All checks passed!",
        mypy="Success: no issues found in 1 source file",
        dependencies="advisories ok, bans ok, licenses ok, sources ok",
    )

    def run(command, **kwargs):
        if command[0] == "git":
            return actual_run(command, **kwargs)
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, stdout="synthetic tool version\n")
        name = next(name for name, value in commands.items() if value == command)
        calls.append(name)
        assert kwargs["cwd"] == repo
        assert kwargs["env"]["CARGO_BUILD_JOBS"] == "1"
        return subprocess.CompletedProcess(
            command,
            1 if name == failure else 0,
            stdout=outputs[name].encode(),
            stderr=b"denied" if name == failure else b"",
        )

    monkeypatch.setattr(subprocess, "run", run)
    directory = tmp_path / "cpu"
    args = (
        repo,
        directory,
        Path("/synthetic/python"),
        Path("/synthetic/worker"),
        Path("/synthetic/binary"),
        tmp_path / "target",
    )
    if failure:
        with pytest.raises(ValueError, match="CPU gate failed"):
            collect_cpu(*args)
        assert not (directory / "evidence.json").exists()
        assert (directory / "dependencies.stderr.log").read_bytes() == b"denied"
        assert "python" not in calls
    else:
        evidence = collect_cpu(*args)
        assert calls == list(commands)
        assert len(validate_cpu(evidence, source_identity(repo))) == 19
        data = json.loads(evidence.read_text())
        data["gates"].pop()
        evidence.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="missing/duplicate"):
            validate_cpu(evidence, source_identity(repo))
