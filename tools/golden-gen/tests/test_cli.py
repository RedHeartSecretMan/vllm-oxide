from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import golden_gen.cli as cli
from golden_gen.config import (
    COMPARISON_KERNEL_SCOPE,
    MODEL_CONFIG_SHA256,
    MODEL_WEIGHTS_SHA256,
    TOKENIZER_SHA256,
    VOCAB_SIZE,
)
from golden_gen.oracles.base import OracleResult


def write_minimal_prompt_corpora(root: Path) -> Path:
    prompts_dir = root / "prompts"
    prompts_dir.mkdir()
    canonical = [
        {
            "id": "canonical_01",
            "category": "canonical",
            "prompt": "single",
            "description": "single",
        },
        {
            "id": "canonical_02",
            "category": "canonical",
            "prompt": "batch",
            "description": "batch",
            "sub_prompts": ["a", "b"],
        },
    ]
    regression = [
        {
            "id": "regression_01",
            "category": "regression",
            "prompt": "regression",
            "description": "regression",
        }
    ]
    (prompts_dir / "canonical.jsonl").write_text(
        "\n".join(json.dumps(item) for item in canonical) + "\n"
    )
    (prompts_dir / "regression.jsonl").write_text(
        "\n".join(json.dumps(item) for item in regression) + "\n"
    )
    return prompts_dir


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
        assert "preflight" in result.stdout
        assert "generate-oracle" in result.stdout
        assert "verify-replay" in result.stdout
        assert "assemble" in result.stdout
        assert "calibrate-baseline" in result.stdout
        assert "bundle" in result.stdout

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

    def test_calibrate_baseline_help_has_no_threshold_override(self):
        result = subprocess.run(
            [sys.executable, "-m", "golden_gen", "calibrate-baseline", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        assert "--manifest-dir" in result.stdout
        assert "--l1-near-tie-max-abs-logit-gap" not in result.stdout
        assert "--l2-atol" not in result.stdout

    def test_bundle_command_uses_the_local_publisher_seam(self, tmp_path, monkeypatch):
        fixture_dir = tmp_path / "fixtures"
        release_dir = tmp_path / "release"
        observed = []

        def fake_publish(source, destination):
            observed.append((source, destination))
            return destination

        monkeypatch.setattr(cli, "publish_release_bundle", fake_publish)

        exit_code = cli.main(
            [
                "bundle",
                "--fixture-dir",
                str(fixture_dir),
                "--release-dir",
                str(release_dir),
            ]
        )

        assert exit_code == 0
        assert observed == [(fixture_dir, release_dir)]

    def test_release_script_exposes_only_independent_fail_closed_stages(self):
        script = (Path(__file__).resolve().parents[3] / "tools" / "validate-release.sh").read_text()

        for stage in (
            "env",
            "generate",
            "calibrate",
            "observe",
            "authoritative",
            "benchmark",
            "report",
            "bundle",
            "publish",
            "verify",
        ):
            assert f"{stage})" in script
        assert "uv sync --extra gpu --frozen --no-build --no-install-project" in script
        assert "CARGO_BUILD_JOBS=1" in script
        assert (
            "CARGO_TARGET_DIR=/home/wanghao/Projects/Codes/VibeCodings/vllm-oxide/target" in script
        )
        assert "--features cuda" in script
        assert "-m golden_gen bundle" in script
        assert '"$BUNDLE_DIR/manifest.json"' in script
        assert '"$BUNDLE_DIR/goldens-v0.2.tar.gz"' in script
        assert "goldens-v0.1" not in script
        assert "*.safetensors" not in script
        assert "tools/golden-gen/output" not in script

    def test_publish_stage_is_inert_without_separate_authority(self, tmp_path):
        script = Path(__file__).resolve().parents[3] / "tools" / "validate-release.sh"
        result = subprocess.run(
            ["bash", str(script), "publish", str(tmp_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "VLLM_OXIDE_ALLOW_GOLDEN_PUBLISH" in result.stderr

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
        assert manifest["schema_version"] == 4
        assert manifest["product_version"] == "v0.2.0"
        assert manifest["golden_version"] == "goldens-v0.2"
        archive_path = tmp_path / "output" / "goldens-v0.2.tar.gz"
        assert archive_path.is_file()
        assert manifest["archive"] == {
            "filename": "goldens-v0.2.tar.gz",
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        }
        assert manifest["tolerance_policy"] == {
            "version": "same-prefix-v1",
            "dtype": "bfloat16",
            "kernel": COMPARISON_KERNEL_SCOPE,
            "l1_near_tie_max_abs_logit_gap": 0.0,
            "l2_atol": 0.0,
            "rationale": "pending reviewed policy selection",
            "evidence": [],
        }
        assert len(manifest["fixtures"]) > 0
        assert manifest["model"]["id"] == "Qwen/Qwen3-0.6B"
        assert manifest["model"]["tokenizer_revision"] == manifest["model"]["revision"]
        assert manifest["model"]["config_sha256"] == MODEL_CONFIG_SHA256
        assert manifest["model"]["tokenizer_sha256"] == TOKENIZER_SHA256
        assert manifest["model"]["weights_sha256"] == MODEL_WEIGHTS_SHA256
        assert manifest["runtime"]["evidence_mode"] == "dry-run"
        assert manifest["kernel_paths"]["reference"].endswith("sdpa-math")

    def test_dry_run_manifest_declares_every_expected_fixture_contract(self, tmp_path):
        """Every discovered case/oracle pair is declared before release validation."""
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
        assert result.returncode == 0, result.stderr

        import json

        manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
        expected = manifest["expected_fixtures"]

        assert len(expected) == 56
        assert {entry["family"] for entry in expected} == {
            "canonical",
            "batch",
            "regression",
        }
        assert Counter(entry["family"] for entry in expected) == {
            "canonical": 8,
            "batch": 8,
            "regression": 40,
        }
        assert {entry["model_revision"] for entry in expected} == {
            "7e4ae267688d671ddfca3122e4528ee980cf3234"
        }
        assert {entry["dtype"] for entry in expected} == {"bfloat16"}
        assert {entry["oracle_role"] for entry in expected} == {"reference", "baseline"}
        assert {entry["required_comparison"] for entry in expected} == {
            "l1",
            "l1_l2",
            "calibration",
        }
        assert manifest["calibrated_fixtures"] == []

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
        """Generation reports exact lifecycle totals."""
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
        assert (
            "Lifecycle totals: expected=56 discovered=56 generated=56 compared=0 skipped=0 failed=0"
        ) in result.stdout

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

    def test_oracle_exception_returns_nonzero_without_publishing_partial_manifest(
        self, tmp_path, monkeypatch, capsys
    ):
        prompts_dir = write_minimal_prompt_corpora(tmp_path)
        output_dir = tmp_path / "output"

        def fail_generate(self, prompt):
            raise RuntimeError("synthetic oracle failure")

        monkeypatch.setattr(cli, "_resolve_prompts_dir", lambda: prompts_dir)
        monkeypatch.setattr(cli.FakeOracle, "generate", fail_generate)

        exit_code = cli.main(["generate", "--dry-run", "--output-dir", str(output_dir)])

        assert exit_code != 0
        assert not (output_dir / "manifest.json").exists()
        assert list(output_dir.glob("*.safetensors")) == []
        captured = capsys.readouterr()
        assert (
            "Lifecycle totals: expected=8 discovered=8 generated=0 compared=0 skipped=0 failed=8"
        ) in captured.err

    @pytest.mark.parametrize("failing_oracle", ["transformers", "vllm"])
    def test_each_oracle_exception_prevents_partial_manifest_publication(
        self, tmp_path, monkeypatch, capsys, failing_oracle
    ):
        prompts_dir = write_minimal_prompt_corpora(tmp_path)
        output_dir = tmp_path / "output"

        def selectively_fail(self, prompt):
            if self.name == failing_oracle:
                raise RuntimeError(f"synthetic {failing_oracle} failure")
            count = len(prompt.sub_prompts) if prompt.is_batch else 1
            if prompt.category == "canonical":
                result = OracleResult.for_canonical(
                    token_ids=np.array([1], dtype=np.int64),
                    logits_per_step=np.zeros((1, VOCAB_SIZE), dtype=np.float32),
                    n_prompt_tokens=1,
                )
            else:
                result = OracleResult.for_regression(
                    token_ids=np.array([1], dtype=np.int64),
                    top5_indices=np.zeros((1, 5), dtype=np.int64),
                    top5_logits=np.zeros((1, 5), dtype=np.float32),
                    n_prompt_tokens=1,
                )
            return [result for _ in range(count)]

        monkeypatch.setattr(cli, "_resolve_prompts_dir", lambda: prompts_dir)
        monkeypatch.setattr(cli.FakeOracle, "generate", selectively_fail)

        exit_code = cli.main(["generate", "--dry-run", "--output-dir", str(output_dir)])

        assert exit_code != 0
        assert not (output_dir / "manifest.json").exists()
        assert list(output_dir.glob("*.safetensors")) == []
        captured = capsys.readouterr()
        assert (
            "Lifecycle totals: expected=8 discovered=8 generated=4 compared=0 skipped=0 failed=4"
        ) in captured.err

    def test_short_batch_oracle_result_fails_without_publishing_manifest(
        self, tmp_path, monkeypatch
    ):
        prompts_dir = write_minimal_prompt_corpora(tmp_path)
        output_dir = tmp_path / "output"

        def short_batch(self, prompt):
            if prompt.category == "canonical":
                result = OracleResult.for_canonical(
                    token_ids=np.array([1], dtype=np.int64),
                    logits_per_step=np.zeros((1, VOCAB_SIZE), dtype=np.float32),
                    n_prompt_tokens=1,
                )
            else:
                result = OracleResult.for_regression(
                    token_ids=np.array([1], dtype=np.int64),
                    top5_indices=np.zeros((1, 5), dtype=np.int64),
                    top5_logits=np.zeros((1, 5), dtype=np.float32),
                    n_prompt_tokens=1,
                )
            return [result]

        monkeypatch.setattr(cli, "_resolve_prompts_dir", lambda: prompts_dir)
        monkeypatch.setattr(cli.FakeOracle, "generate", short_batch)

        exit_code = cli.main(["generate", "--dry-run", "--output-dir", str(output_dir)])

        assert exit_code != 0
        assert not (output_dir / "manifest.json").exists()
        assert list(output_dir.glob("*.safetensors")) == []

    def test_only_category_failure_totals_separate_skipped_from_failed(
        self, tmp_path, monkeypatch, capsys
    ):
        prompts_dir = write_minimal_prompt_corpora(tmp_path)
        output_dir = tmp_path / "output"

        def fail_reference_only(self, prompt):
            if self.name == "transformers":
                raise RuntimeError("synthetic reference failure")
            count = len(prompt.sub_prompts) if prompt.is_batch else 1
            result = OracleResult.for_canonical(
                token_ids=np.array([1], dtype=np.int64),
                logits_per_step=np.zeros((1, VOCAB_SIZE), dtype=np.float32),
                n_prompt_tokens=1,
            )
            return [result for _ in range(count)]

        monkeypatch.setattr(cli, "_resolve_prompts_dir", lambda: prompts_dir)
        monkeypatch.setattr(cli.FakeOracle, "generate", fail_reference_only)

        exit_code = cli.main(
            [
                "generate",
                "--dry-run",
                "--only-category",
                "canonical",
                "--output-dir",
                str(output_dir),
            ]
        )

        assert exit_code != 0
        captured = capsys.readouterr()
        assert (
            "Lifecycle totals: expected=8 discovered=8 generated=3 compared=0 skipped=2 failed=3"
        ) in captured.err
