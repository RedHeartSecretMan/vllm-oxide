import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from golden_gen.config import (
    COMPARISON_KERNEL_SCOPE,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    TOKENIZER_SHA256,
)
from golden_gen.schema import (
    ArchiveInfo,
    BaselineCalibration,
    ExpectedFixture,
    FixtureMetadata,
    GenerationConfig,
    Manifest,
    ModelInfo,
    OracleVersions,
    PromptSpec,
    TolerancePolicy,
)
from tests.support import pinned_kernel_paths, release_runtime


class TestTolerancePolicy:
    def test_same_prefix_policy_is_an_explicit_versioned_input(self):
        policy = TolerancePolicy(
            version="same-prefix-v1",
            dtype="bfloat16",
            kernel="sdpa",
            l1_near_tie_max_abs_logit_gap=0.02,
            l2_atol=0.01,
            rationale="Reviewed synthetic policy",
            evidence=["synthetic:test_schema"],
        )

        assert policy.version == "same-prefix-v1"
        assert policy.l1_near_tie_max_abs_logit_gap == 0.02
        assert policy.l2_atol == 0.01
        assert policy.rationale == "Reviewed synthetic policy"
        assert policy.evidence == ["synthetic:test_schema"]

    def test_unknown_policy_version_is_rejected(self):
        with pytest.raises(ValidationError, match="version"):
            TolerancePolicy(
                version="latest",  # type: ignore[arg-type]
                dtype="bfloat16",
                kernel="sdpa",
                l1_near_tie_max_abs_logit_gap=0.02,
                l2_atol=0.01,
                rationale="Reviewed synthetic policy",
                evidence=["synthetic:test_schema"],
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

    def test_single_sub_prompt_is_an_unsupported_batch_shape(self):
        with pytest.raises(ValidationError, match="sub_prompts must contain 2 to 26"):
            PromptSpec(
                id="test",
                category="canonical",
                prompt="test",
                description="test",
                sub_prompts=["only one"],
            )

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
            sha256="a" * 64,
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
                sha256="a" * 64,
                filename="test.safetensors",
            )

    def test_noncanonical_sha256_is_rejected(self):
        with pytest.raises(ValidationError, match="sha256"):
            FixtureMetadata(
                prompt_id="canonical_01",
                category="canonical",
                oracle="transformers",
                num_tokens=1,
                logits_dtype="float32",
                logits_shape=(1, 151936),
                sha256="ABC123",
                filename="canonical_01.transformers.safetensors",
            )


class TestBaselineCalibration:
    def test_valid(self):
        calibration = BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="2x max pairwise abs diff",
        )
        assert calibration.candidate_atol == 0.01
        assert calibration.calibration_factor == 2.0
        assert calibration.observed_max_abs_diff == 0.005


class TestManifest:
    def test_shared_manifest_v4_fixture_is_compatible(self):
        path = Path(__file__).parent / "fixtures" / "manifest-v4.json"

        manifest = Manifest.from_json(path)

        assert manifest.schema_version == 4
        assert manifest.archive.sha256 == "a" * 64

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("schema_version", 5),
            ("product_version", "v0.3.0"),
            ("golden_version", "goldens-v0.3"),
        ],
    )
    def test_asset_contract_upgrade_is_rejected(self, field, value):
        path = Path(__file__).parent / "fixtures" / "manifest-v4.json"
        data = json.loads(path.read_text())
        data[field] = value

        with pytest.raises(ValidationError, match=field):
            Manifest.model_validate(data)

    @pytest.mark.parametrize(
        "field",
        ["schema_version", "product_version", "golden_version", "archive"],
    )
    def test_missing_asset_contract_field_is_rejected(self, field):
        path = Path(__file__).parent / "fixtures" / "manifest-v4.json"
        data = json.loads(path.read_text())
        del data[field]

        with pytest.raises(ValidationError, match=field):
            Manifest.model_validate(data)

    def test_case_colliding_fixture_names_are_rejected(self):
        path = Path(__file__).parent / "fixtures" / "manifest-v4.json"
        data = json.loads(path.read_text())
        for fixture in list(data["expected_fixtures"]):
            duplicate = dict(fixture)
            duplicate["fixture_id"] = duplicate["fixture_id"].replace(
                "canonical_01", "CANONICAL_01"
            )
            duplicate["prompt_id"] = "CANONICAL_01"
            duplicate["filename"] = duplicate["filename"].replace("canonical_01", "CANONICAL_01")
            data["expected_fixtures"].append(duplicate)

        with pytest.raises(ValidationError, match="case-colliding"):
            Manifest.model_validate(data)

    @pytest.mark.parametrize(
        ("filename", "sha256"),
        [
            ("fixtures.tar.gz", "a" * 64),
            ("goldens-v0.2.tar.gz", "A" * 64),
            ("goldens-v0.2.tar.gz", "abc"),
        ],
    )
    def test_noncanonical_archive_identity_is_rejected(self, filename, sha256):
        with pytest.raises(ValidationError):
            ArchiveInfo(filename=filename, sha256=sha256)  # type: ignore[arg-type]

    def test_build_and_roundtrip(self, tmp_path):
        calibration = BaselineCalibration(
            candidate_atol=0.01,
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
            sha256="a" * 64,
            filename="canonical_01.transformers.safetensors",
        )
        expected = [
            ExpectedFixture(
                fixture_id="canonical_01.transformers",
                prompt_id="canonical_01",
                family="canonical",
                model_revision=MODEL_REVISION,
                dtype="bfloat16",
                oracle="transformers",
                oracle_role="reference",
                required_comparison="l1_l2",
                filename="canonical_01.transformers.safetensors",
            ),
            ExpectedFixture(
                fixture_id="canonical_01.vllm",
                prompt_id="canonical_01",
                family="canonical",
                model_revision=MODEL_REVISION,
                dtype="bfloat16",
                oracle="vllm",
                oracle_role="baseline",
                required_comparison="calibration",
                filename="canonical_01.vllm.safetensors",
            ),
        ]
        manifest = Manifest(
            schema_version=4,
            product_version="v0.2.0",
            golden_version="goldens-v0.2",
            archive=ArchiveInfo(filename="goldens-v0.2.tar.gz", sha256="a" * 64),
            generated_at=datetime.now(UTC),
            model=ModelInfo(
                id="Qwen/Qwen3-0.6B",
                revision=MODEL_REVISION,
                tokenizer_revision=MODEL_REVISION,
                config_sha256=MODEL_CONFIG_SHA256,
                tokenizer_sha256=TOKENIZER_SHA256,
                weights_sha256=MODEL_WEIGHTS_SHA256,
                arch="Qwen3ForCausalLM",
                dtype="bfloat16",
                vocab_size=151936,
            ),
            oracle_versions=OracleVersions(transformers="4.57.6", vllm="0.18.1"),
            runtime=release_runtime(),
            kernel_paths=pinned_kernel_paths(),
            generation=GenerationConfig(
                canonical_max_tokens=64,
                regression_max_tokens=32,
                temperature=0.0,
                attn_implementation="sdpa",
            ),
            tolerance_policy=TolerancePolicy(
                version="same-prefix-v1",
                dtype="bfloat16",
                kernel=COMPARISON_KERNEL_SCOPE,
                l1_near_tie_max_abs_logit_gap=0.02,
                l2_atol=0.01,
                rationale="Reviewed synthetic policy",
                evidence=["synthetic:test_schema"],
            ),
            baseline_calibration=calibration,
            expected_fixtures=expected,
            fixtures=[fixture],
        )
        path = tmp_path / "manifest.json"
        manifest.to_json(path)
        restored = Manifest.from_json(path)
        assert restored.schema_version == 4
        assert restored.product_version == "v0.2.0"
        assert restored.golden_version == "goldens-v0.2"
        assert restored.archive.filename == "goldens-v0.2.tar.gz"
        assert len(restored.fixtures) == 1
        assert restored.fixtures[0].sha256 == "a" * 64
        assert restored.baseline_calibration.candidate_atol == 0.01
        assert restored.tolerance_policy.version == "same-prefix-v1"
