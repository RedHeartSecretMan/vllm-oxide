from datetime import UTC, datetime

from golden_gen.manifest import build_manifest, get_oracle_versions, read_manifest, write_manifest
from golden_gen.schema import FixtureMetadata, OracleVersions, ToleranceCalibration


class TestManifest:
    def test_build_manifest_minimal(self):
        tolerance = ToleranceCalibration(
            atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="2x max pairwise abs diff",
        )
        fixtures = [
            FixtureMetadata(
                prompt_id="canonical_01",
                category="canonical",
                oracle="transformers",
                num_tokens=64,
                logits_dtype="float32",
                logits_shape=(64, 151936),
                sha256="abc123",
                filename="canonical_01.transformers.safetensors",
            )
        ]
        manifest = build_manifest(
            fixtures=fixtures,
            tolerance=tolerance,
            generated_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        )
        assert manifest.schema_version == 1
        assert manifest.model.id == "Qwen/Qwen3-0.6B"
        assert manifest.model.arch == "Qwen3ForCausalLM"
        assert manifest.model.vocab_size == 151936
        assert manifest.generation.canonical_max_tokens == 64
        assert manifest.generation.regression_max_tokens == 32
        assert len(manifest.fixtures) == 1
        assert manifest.generation.temperature == 0.0

    def test_write_read_roundtrip(self, tmp_path):
        tolerance = ToleranceCalibration(
            atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        )
        fixtures = [
            FixtureMetadata(
                prompt_id="canonical_01",
                category="canonical",
                oracle="transformers",
                num_tokens=64,
                logits_dtype="float32",
                logits_shape=(64, 151936),
                sha256="abc123",
                filename="canonical_01.transformers.safetensors",
            )
        ]
        manifest = build_manifest(
            fixtures=fixtures,
            tolerance=tolerance,
            generated_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        )
        path = tmp_path / "manifest.json"
        write_manifest(manifest, path)
        restored = read_manifest(path)
        assert restored.model.id == manifest.model.id
        assert restored.fixtures[0].sha256 == "abc123"

    def test_get_oracle_versions(self):
        versions = get_oracle_versions()
        assert isinstance(versions, OracleVersions)
        assert isinstance(versions.transformers, str)
        assert isinstance(versions.vllm, str)
