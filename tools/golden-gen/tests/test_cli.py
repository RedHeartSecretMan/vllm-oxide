from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import golden_gen.cli as cli


def _local_validation_tools(tmp_path):
    import os

    root = tmp_path / "repo"
    (root / "tools/golden-gen").mkdir(parents=True)
    script = root / "tools/validate-release.sh"
    script.write_bytes((Path(__file__).resolve().parents[2] / "validate-release.sh").read_bytes())
    model = tmp_path / "model with spaces"
    model.mkdir()
    temporary = tmp_path / "temporary outputs"
    temporary.mkdir()
    executables = tmp_path / "bin"
    executables.mkdir()
    log = tmp_path / "commands.jsonl"
    command = f"""#!{sys.executable}
import json, os, pathlib, sys
tool = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with pathlib.Path({str(log)!r}).open("a") as output:
    output.write(json.dumps(dict(tool=tool, args=args,
        environment=os.environ.get("UV_PROJECT_ENVIRONMENT"), cwd=os.getcwd())) + "\\n")
if tool == "gh":
    raise SystemExit(93)
if tool == "uv" and "generate" in args and os.environ.get("V010_TEST_FAILURE") == "generate":
    raise SystemExit(7)
if tool == "cargo" and args[0] == "run" and os.environ.get("V010_TEST_FAILURE") == "compare":
    raise SystemExit(9)
print("synthetic local tool output")
"""
    for name in ("uv", "cargo", "gh"):
        executable = executables / name
        executable.write_text(command)
        executable.chmod(0o700)
    env = {**os.environ, "PATH": str(executables) + os.pathsep + os.environ["PATH"]}
    env["TMPDIR"] = str(temporary)
    env["UV_PROJECT_ENVIRONMENT"] = "/unrelated-existing-environment"
    return script, model, temporary, log, env


def test_local_validation_uses_fresh_locked_gpu_runs_and_never_uploads(tmp_path):
    import json

    script, model, temporary, log, env = _local_validation_tools(tmp_path)
    for _ in range(2):
        result = subprocess.run(
            ["bash", str(script), str(model)], env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(record["tool"] == "gh" for record in records)
    generations = [r for r in records if r["tool"] == "uv" and "generate" in r["args"]]
    outputs = [Path(r["args"][r["args"].index("--output-dir") + 1]) for r in generations]
    assert len(outputs) == len(set(outputs)) == 2
    assert all(output.parent.parent == temporary for output in outputs)
    for record in records:
        if record["tool"] == "uv":
            assert "--locked" in record["args"]
            assert record["args"][record["args"].index("--extra") + 1] == "gpu"
            assert Path(record["environment"]).parent in {p.parent for p in outputs}
        else:
            assert record["args"][record["args"].index("--features") + 1] == "cuda"
    for output in outputs:
        assert (output.parent / "generation.log").is_file()
        assert (output.parent / "calibration.log").is_file()
        assert (output.parent / "build.log").is_file()
        assert (output.parent / "comparison.log").is_file()


@pytest.mark.parametrize("stage,status", [("generate", 7), ("compare", 9)])
def test_local_validation_preserves_stage_failure_without_a_success_message(
    tmp_path, stage, status
):
    import json

    script, model, _, log, env = _local_validation_tools(tmp_path)
    env["V010_TEST_FAILURE"] = stage
    result = subprocess.run(
        ["bash", str(script), str(model)], env=env, capture_output=True, text=True
    )
    assert result.returncode == status, result.stdout + result.stderr
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(record["tool"] == "gh" for record in records)
    if stage == "generate":
        assert not any(record["tool"] == "cargo" for record in records)
    assert "Local validation passed" not in result.stdout


@pytest.mark.parametrize("failing_oracle", ["transformers", "vllm"])
@pytest.mark.parametrize("existing_manifest", [False, True])
def test_oracle_failure_is_nonzero_and_never_publishes_a_partial_manifest(
    tmp_path, monkeypatch, capsys, failing_oracle, existing_manifest
):
    import json

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "regression.jsonl").write_text(
        "\n".join(
            json.dumps(dict(id=name, category="regression", prompt=name, description="CPU fixture"))
            for name in ("first", "second")
        )
        + "\n"
    )
    monkeypatch.setattr(cli, "_resolve_prompts_dir", lambda: prompts_dir)
    output = tmp_path / "output"
    arguments = ["generate", "--dry-run", "--output-dir", str(output)]
    manifest = output / "manifest.json"
    original_manifest = None
    if existing_manifest:
        assert cli.main(arguments) == 0
        original_manifest = manifest.read_bytes()
    generate = cli.FakeOracle.generate
    closed = []

    def fail_after_one_prompt(self, prompt):
        if self.name == failing_oracle and prompt.id == "second":
            raise RuntimeError("synthetic second-prompt failure")
        return generate(self, prompt)

    monkeypatch.setattr(cli.FakeOracle, "generate", fail_after_one_prompt)
    monkeypatch.setattr(cli.FakeOracle, "close", lambda self: closed.append(self.name))

    assert cli.main(arguments) != 0
    assert closed == ["transformers", "vllm"]
    if original_manifest is None:
        assert not manifest.exists()
    else:
        assert manifest.read_bytes() == original_manifest
    assert "synthetic second-prompt failure" in capsys.readouterr().err


@pytest.mark.parametrize("fault", ["construct", "close", "short-batch"])
def test_oracle_lifecycle_or_short_batch_failure_cannot_publish_success(
    tmp_path, monkeypatch, fault
):
    from golden_gen.schema import PromptSpec

    prompt = PromptSpec(
        id="batch",
        category="canonical",
        prompt="batch",
        description="CPU fixture",
        sub_prompts=["a", "b"],
    )
    monkeypatch.setattr(cli, "load_prompts", lambda _: [prompt])

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic lifecycle failure")

    if fault == "construct":
        monkeypatch.setattr(cli, "FakeOracle", fail)
    elif fault == "close":
        monkeypatch.setattr(cli.FakeOracle, "close", fail)
    else:
        monkeypatch.setattr(cli.FakeOracle, "generate", lambda *_: [])
    output = tmp_path / "output"
    assert cli.main(["generate", "--dry-run", "--output-dir", str(output)]) != 0
    assert not (output / "manifest.json").exists()


class TestCLI:
    """Test CLI via subprocess (--help, generate --dry-run)."""

    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "golden_gen", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        assert "Generate golden fixtures" in result.stdout
        assert "generate" in result.stdout
        assert "calibrate" in result.stdout

    def test_generate_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "golden_gen", "generate", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        assert "--dry-run" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--only-category" in result.stdout

    def test_calibrate_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "golden_gen", "calibrate", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        assert "--manifest-dir" in result.stdout

    def test_dry_run_produces_manifest(self, tmp_path):
        """generate --dry-run should produce a fake manifest + fixtures."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "golden_gen",
                "generate",
                "--dry-run",
                "--output-dir",
                str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        manifest_path = tmp_path / "output" / "manifest.json"
        assert manifest_path.exists(), f"Manifest not found at {manifest_path}"

        import json

        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["schema_version"] == 1
        assert len(manifest["fixtures"]) > 0
        assert manifest["model"]["id"] == "Qwen/Qwen3-0.6B"

    def test_dry_run_produces_safetensors(self, tmp_path):
        """generate --dry-run should produce fake .safetensors fixture files."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "golden_gen",
                "generate",
                "--dry-run",
                "--output-dir",
                str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0

        safetensors_files = list((tmp_path / "output").glob("*.safetensors"))
        # 4 canonical singles + 4 canonical_05 sub-prompts + 20 regression = 28 prompt-ids
        # 28 x 2 oracles = 56 fixtures
        assert len(safetensors_files) == 56, (
            f"Expected 56 .safetensors files, got {len(safetensors_files)}"
        )

        # Verify safetensors content
        from safetensors.numpy import load_file

        sample = load_file(str(safetensors_files[0]))
        assert "token_ids" in sample

    def test_dry_run_produces_output(self, tmp_path):
        """generate --dry-run should still produce output."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "golden_gen",
                "generate",
                "--dry-run",
                "--output-dir",
                str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0

    def test_only_category_flag(self, tmp_path):
        """--only-category canonical should only generate canonical fixtures."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "golden_gen",
                "generate",
                "--dry-run",
                "--only-category",
                "canonical",
                "--output-dir",
                str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0

        safetensors_files = list((tmp_path / "output").glob("*.safetensors"))
        # canonical_01-04 (4 singles) + canonical_05 (4 sub-prompts) = 8 prompt-ids
        # 8 x 2 oracles = 16 fixtures
        assert len(safetensors_files) == 16, (
            f"Expected 16 .safetensors files for canonical-only, got {len(safetensors_files)}"
        )

    def test_version_accessible(self):
        """Verify the package version is importable."""
        result = subprocess.run(
            [sys.executable, "-c", "from golden_gen import __version__; print(__version__)"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "0.1.0"
