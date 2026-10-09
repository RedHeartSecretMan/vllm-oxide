import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("relative_paths", [False, True])
def test_fresh_calibration_uses_current_source_and_guard(
    ticket_artifact_root, monkeypatch, relative_paths
):
    import json
    import os

    from golden_gen import guard, layered_cli

    source = {"commit": "a" * 40, "tree": "b" * 40}
    monkeypatch.setattr(layered_cli, "source_identity", lambda repo: source)
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/unrelated-environment")
    model, binary = Path("/synthetic-model"), Path("/synthetic-binary")
    repo, run = REPO, ticket_artifact_root
    if relative_paths:
        monkeypatch.chdir(REPO.parent)
        repo, run, model, binary = (
            Path(os.path.relpath(path)) for path in (repo, run, model, binary)
        )
    calls = []

    def guarded(command, evidence, *, cwd, env):
        calls.append(command)
        assert cwd == REPO.resolve()
        assert env["PYTHONPATH"] == str(REPO / "tools/golden-gen/src")
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert env["UV_PROJECT_ENVIRONMENT"] == sys.prefix
        assert command[command.index("--repo-root") + 1] == str(REPO)
        assert command[command.index("--run-dir") + 1] == str(ticket_artifact_root)
        assert command[command.index("--model-dir") + 1] == "/synthetic-model"
        assert command[command.index("--candidate-binary") + 1] == "/synthetic-binary"
        evidence.write_text(json.dumps({"after": {"compute_processes": ""}, "child_pid": 123}))
        (evidence.parent / "worker.json").write_text(
            json.dumps({"source": source, "driver_pid": 123})
        )
        (evidence.parent / "capture.json").write_text("{}")

    monkeypatch.setattr(guard, "run_guarded", guarded)
    result = layered_cli.collect_group(
        repo,
        run,
        model,
        "calibration-length-1",
        "reference",
        "primary",
        binary,
        fresh_observation=True,
    )
    assert len(calls) == 1
    assert result["accepting"] is False
    receipt = json.loads((ticket_artifact_root / result["receipt"]["path"]).read_text())
    assert receipt["source"] == source
    assert not list(ticket_artifact_root.glob("*.complete.json"))


@pytest.mark.parametrize(
    ("group", "auxiliary"),
    [
        ("dev-canonical_01", False),
        ("acceptance-length-1", False),
        ("acceptance-behavior-repeated", True),
    ],
)
def test_fresh_observation_rejects_owners_outside_calibration(
    ticket_artifact_root, monkeypatch, group, auxiliary
):
    from golden_gen import layered_cli

    monkeypatch.setattr(layered_cli, "source_identity", lambda repo: {})
    with pytest.raises(ValueError, match="frozen inventory"):
        layered_cli.collect_group(
            REPO,
            ticket_artifact_root,
            Path("/synthetic-model"),
            group,
            "candidate",
            "primary",
            auxiliary=auxiliary,
            fresh_observation=True,
        )
    assert not ticket_artifact_root.exists()


@pytest.mark.parametrize("fresh", [False, True])
def test_fresh_observation_does_not_bypass_existing_supervision(ticket_artifact_root, fresh):
    from golden_gen.layered_cli import collect_group

    with pytest.raises(ValueError, match="measurement checkout"):
        collect_group(
            REPO,
            ticket_artifact_root,
            Path("/synthetic-model"),
            "calibration-length-1",
            "reference",
            "primary",
            measurement_repo=REPO if fresh else None,
            fresh_observation=fresh,
        )
    assert not ticket_artifact_root.exists()


@pytest.mark.parametrize("action", ["authoritative", "assemble-authoritative", "worker"])
def test_fresh_observation_flag_cannot_change_acceptance_or_worker_mode(tmp_path, action):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "golden_gen.layered_cli",
            action,
            "--repo-root",
            str(REPO),
            "--run-dir",
            str(tmp_path / "run"),
            "--fresh-observation",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "only valid for calibration collection" in result.stderr
    assert not (tmp_path / "run").exists()


def test_failed_candidate_subprocess_preserves_both_log_streams(capsys) -> None:
    from golden_gen.layered_cli import run_candidate_capture

    with pytest.raises(subprocess.CalledProcessError):
        run_candidate_capture(
            [
                sys.executable,
                "-c",
                "import sys; print('retained stdout'); "
                "print('retained stderr',file=sys.stderr); sys.exit(5)",
            ]
        )
    captured = capsys.readouterr()
    assert captured.out == "retained stdout\n"
    assert captured.err == "retained stderr\n"
