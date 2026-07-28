import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from golden_gen.schema import (
    FixtureMetadata,
    GenerationConfig,
    Manifest,
    ModelInfo,
    OracleVersions,
    PromptSpec,
    ToleranceCalibration,
)


class TestPromptSpec:
    def test_valid_canonical(self):
        spec = PromptSpec(
            id="canonical_01",
            category="canonical",
            prompt="Hello world",
            description="test",
        )
        assert spec.id == "canonical_01"
        assert spec.chat_template is False
        assert spec.note is None

    def test_valid_regression(self):
        spec = PromptSpec(
            id="regression_01",
            category="regression",
            prompt="Some code",
            description="code test",
            note="edge case",
        )
        assert spec.note == "edge case"

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            PromptSpec(
                id="bad_01",
                category="invalid",  # type: ignore[arg-type]
                prompt="test",
                description="test",
            )

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            PromptSpec(
                id="test_01",
                category="canonical",
                prompt="test",
                description="test",
                extra_field="nope",  # type: ignore[call-arg]
            )

    def test_frozen(self):
        spec = PromptSpec(
            id="canonical_01",
            category="canonical",
            prompt="Hello",
            description="test",
        )
        with pytest.raises(ValidationError):
            spec.id = "changed"  # type: ignore[misc]

    def test_is_batch_false_by_default(self):
        spec = PromptSpec(
            id="canonical_01",
            category="canonical",
            prompt="Hello",
            description="test",
        )
        assert spec.is_batch is False

    def test_is_batch_true_with_multiple_sub_prompts(self):
        spec = PromptSpec(
            id="canonical_05",
            category="canonical",
            prompt="batch test",
            description="test",
            sub_prompts=["prompt a", "prompt b", "prompt c", "prompt d"],
        )
        assert spec.is_batch is True
        assert len(spec.sub_prompts) == 4

    def test_is_batch_false_with_single_sub_prompt(self):
        spec = PromptSpec(
            id="test",
            category="canonical",
            prompt="test",
            description="test",
            sub_prompts=["only one"],
        )
        assert spec.is_batch is False

    def test_sub_prompts_roundtrip_json(self):
        spec = PromptSpec(
            id="canonical_05",
            category="canonical",
            prompt="batch test",
            description="test",
            sub_prompts=["a", "b", "c", "d"],
        )
        data = json.loads(spec.model_dump_json())
        restored = PromptSpec.model_validate(data)
        assert restored.sub_prompts == ["a", "b", "c", "d"]
        assert restored.is_batch is True

    def test_roundtrip_json(self):
        spec = PromptSpec(
            id="canonical_01",
            category="canonical",
            prompt="Hello",
            description="test",
            chat_template=True,
            note="some note",
        )
        data = json.loads(spec.model_dump_json())
        restored = PromptSpec.model_validate(data)
        assert restored == spec


class TestOracleVersions:
    def test_valid(self):
        versions = OracleVersions(
            transformers="4.43.0",
            vllm="0.26.0",
        )
        assert versions.transformers == "4.43.0"

    def test_frozen(self):
        versions = OracleVersions(
            transformers="4.43.0",
            vllm="0.26.0",
        )
        with pytest.raises(ValidationError):
            versions.transformers = "5.0.0"  # type: ignore[misc]


class TestFixtureMetadata:
    def test_valid_canonical(self):
        meta = FixtureMetadata(
            prompt_id="canonical_01",
            category="canonical",
            oracle="transformers",
            num_tokens=64,
            logits_dtype="float32",
            logits_shape=(64, 151936),
            sha256="abc123",
            filename="canonical_01.transformers.safetensors",
        )
        assert meta.oracle == "transformers"

    def test_invalid_dtype(self):
        with pytest.raises(ValidationError):
            FixtureMetadata(
                prompt_id="canonical_01",
                category="canonical",
                oracle="transformers",
                num_tokens=64,
                logits_dtype="bfloat16",  # type: ignore[arg-type]
                logits_shape=(64, 151936),
                sha256="abc123",
                filename="test.safetensors",
            )


class TestToleranceCalibration:
    def test_valid(self):
        tol = ToleranceCalibration(
            atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="2x max pairwise abs diff",
        )
        assert tol.atol == 0.01
        assert tol.calibration_factor == 2.0
        assert tol.observed_max_abs_diff == 0.005


class TestManifest:
    def test_build_and_roundtrip(self, tmp_path):
        tolerance = ToleranceCalibration(
            atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="2x max pairwise abs diff",
        )
        fixture = FixtureMetadata(
            prompt_id="canonical_01",
            category="canonical",
            oracle="transformers",
            num_tokens=64,
            logits_dtype="float32",
            logits_shape=(64, 151936),
            sha256="abc123",
            filename="canonical_01.transformers.safetensors",
        )
        manifest = Manifest(
            schema_version=1,
            generated_at=datetime.now(UTC),
            model=ModelInfo(
                id="Qwen/Qwen3-0.6B",
                revision="abc123",
                arch="Qwen3ForCausalLM",
                dtype="bfloat16",
                vocab_size=151936,
            ),
            oracle_versions=OracleVersions(transformers="4.43.0", vllm="0.26.0"),
            generation=GenerationConfig(
                canonical_max_tokens=64,
                regression_max_tokens=32,
                temperature=0.0,
                attn_implementation="eager",
            ),
            tolerance=tolerance,
            fixtures=[fixture],
        )
        path = tmp_path / "manifest.json"
        manifest.to_json(path)
        restored = Manifest.from_json(path)
        assert restored.schema_version == 1
        assert len(restored.fixtures) == 1
        assert restored.fixtures[0].sha256 == "abc123"
        assert restored.tolerance.atol == 0.01
