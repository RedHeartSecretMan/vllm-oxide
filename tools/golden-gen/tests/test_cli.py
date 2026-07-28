from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
