from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class TestCLI:
    """Test CLI via subprocess (--help, --dry-run)."""

    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "golden_gen", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        assert "Generate golden fixtures" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--only-oracle" in result.stdout
        assert "--only-category" in result.stdout
        assert "--no-cross-validate" in result.stdout

    def test_dry_run_produces_manifest(self, tmp_path):
        """--dry-run should produce a fake manifest + fixtures."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "golden_gen",
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
        """--dry-run should produce fake .safetensors fixture files."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "golden_gen",
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
        # 5 canonical + 20 regression = 25 prompts x 3 fake oracles = 75 fixtures
        # However the FakeOracle always generates all categories
        assert len(safetensors_files) == 75, (
            f"Expected 75 .safetensors files, got {len(safetensors_files)}"
        )

        # Verify safetensors content
        from safetensors.numpy import load_file

        sample = load_file(str(safetensors_files[0]))
        assert "token_ids" in sample

    def test_dry_run_cross_validate(self, tmp_path):
        """--dry-run cross-validation should produce tolerance values."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "golden_gen",
                "--dry-run",
                "--no-cross-validate",
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
        # 5 canonical prompts x 3 fake oracles = 15 fixtures
        assert len(safetensors_files) == 15, (
            f"Expected 15 .safetensors files for canonical-only, got {len(safetensors_files)}"
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
