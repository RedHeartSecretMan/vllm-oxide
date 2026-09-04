from datetime import UTC, datetime
from functools import partial

import pytest
from pydantic import ValidationError

from golden_gen.config import COMPARISON_KERNEL_SCOPE
from golden_gen.manifest import (
    build_manifest as _build_manifest,
)
from golden_gen.manifest import (
    get_oracle_versions,
    read_manifest,
    write_manifest,
)
from golden_gen.schema import (
    ArchiveInfo,
    BaselineCalibration,
    ExpectedFixture,
    FixtureMetadata,
    Manifest,
    OracleVersions,
    TolerancePolicy,
)
from tests.support import pinned_kernel_paths, release_runtime

MODEL_REVISION = "7e4ae267688d671ddfca3122e4528ee980cf3234"
build_manifest = partial(
    _build_manifest,
    archive=ArchiveInfo(filename="goldens-v0.2.tar.gz", sha256="a" * 64),
    runtime=release_runtime(),
    kernel_paths=pinned_kernel_paths(),
)


def same_prefix_policy() -> TolerancePolicy:
    return TolerancePolicy(
        version="same-prefix-v1",
        dtype="bfloat16",
        kernel=COMPARISON_KERNEL_SCOPE,
        l1_near_tie_max_abs_logit_gap=0.02,
        l2_atol=0.01,
        rationale="Reviewed synthetic policy",
        evidence=["synthetic:test_manifest"],
    )


def expected_pair() -> list[ExpectedFixture]:
    return [
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


class TestManifest:
    def test_empty_expected_fixture_set_is_rejected(self):
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        )

        with pytest.raises(ValidationError, match="expected_fixtures"):
            build_manifest(
                fixtures=[],
                expected_fixtures=[],
                tolerance_policy=same_prefix_policy(),
                baseline_calibration=tolerance,
            )

    def test_generated_fixture_identifier_must_match_an_expectation(self):
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        )
        unexpected = FixtureMetadata(
            prompt_id="canonical_99",
            category="canonical",
            oracle="transformers",
            num_tokens=1,
            logits_dtype="float32",
            logits_shape=(1, 151936),
            sha256="a" * 64,
            filename="canonical_99.transformers.safetensors",
        )

        with pytest.raises(ValidationError, match="unmatched generated fixture"):
            build_manifest(
                fixtures=[unexpected],
                expected_fixtures=expected_pair(),
                tolerance_policy=same_prefix_policy(),
                baseline_calibration=tolerance,
            )

    def test_duplicate_expected_fixture_identifier_is_rejected(self):
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        )
        expected = ExpectedFixture(
            fixture_id="canonical_01.transformers",
            prompt_id="canonical_01",
            family="canonical",
            model_revision="7e4ae267688d671ddfca3122e4528ee980cf3234",
            dtype="bfloat16",
            oracle="transformers",
            oracle_role="reference",
            required_comparison="l1_l2",
            filename="canonical_01.transformers.safetensors",
        )

        with pytest.raises(ValidationError, match="duplicate expected fixture identifier"):
            build_manifest(
                fixtures=[],
                expected_fixtures=[expected, expected.model_copy()],
                tolerance_policy=same_prefix_policy(),
                baseline_calibration=tolerance,
            )

    def test_unsupported_manifest_schema_version_is_rejected(self):
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        )
        manifest = build_manifest(
            fixtures=[],
            expected_fixtures=expected_pair(),
            tolerance_policy=same_prefix_policy(),
            baseline_calibration=tolerance,
        ).model_dump()
        manifest["schema_version"] = 999

        with pytest.raises(ValidationError, match="schema_version"):
            Manifest.model_validate(manifest)

    @pytest.mark.parametrize("invalid_scope", ["dtype", "kernel"])
    def test_reference_contract_must_remain_bf16_sdpa(self, invalid_scope):
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        )
        manifest = build_manifest(
            fixtures=[],
            expected_fixtures=expected_pair(),
            tolerance_policy=same_prefix_policy(),
            baseline_calibration=tolerance,
        ).model_dump()
        if invalid_scope == "dtype":
            manifest["model"]["dtype"] = "float32"
            manifest["tolerance_policy"]["dtype"] = "float32"
            for expected in manifest["expected_fixtures"]:
                expected["dtype"] = "float32"
        else:
            manifest["generation"]["attn_implementation"] = "eager"
            manifest["tolerance_policy"]["kernel"] = "eager"

        with pytest.raises(ValidationError, match="BF16 SDPA reference oracle"):
            Manifest.model_validate(manifest)

    def test_legacy_regression_skip_map_is_rejected(self):
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        )
        manifest = build_manifest(
            fixtures=[],
            expected_fixtures=expected_pair(),
            tolerance_policy=same_prefix_policy(),
            baseline_calibration=tolerance,
        ).model_dump()
        manifest["regression_skip_map"] = {"canonical_01": [0]}

        with pytest.raises(ValidationError, match="regression_skip_map"):
            Manifest.model_validate(manifest)

    def test_oracle_name_role_and_comparison_must_be_consistent(self):
        invalid = ExpectedFixture(
            fixture_id="canonical_01.transformers",
            prompt_id="canonical_01",
            family="canonical",
            model_revision="7e4ae267688d671ddfca3122e4528ee980cf3234",
            dtype="bfloat16",
            oracle="transformers",
            oracle_role="baseline",
            required_comparison="calibration",
            filename="canonical_01.transformers.safetensors",
        )
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        )

        with pytest.raises(ValidationError, match="oracle role contract"):
            build_manifest(
                fixtures=[],
                expected_fixtures=[invalid],
                tolerance_policy=same_prefix_policy(),
                baseline_calibration=tolerance,
            )

    def test_each_prompt_requires_reference_and_baseline_expectations(self):
        reference_only = ExpectedFixture(
            fixture_id="canonical_01.transformers",
            prompt_id="canonical_01",
            family="canonical",
            model_revision="7e4ae267688d671ddfca3122e4528ee980cf3234",
            dtype="bfloat16",
            oracle="transformers",
            oracle_role="reference",
            required_comparison="l1_l2",
            filename="canonical_01.transformers.safetensors",
        )
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        )

        with pytest.raises(ValidationError, match="reference and baseline"):
            build_manifest(
                fixtures=[],
                expected_fixtures=[reference_only],
                tolerance_policy=same_prefix_policy(),
                baseline_calibration=tolerance,
            )

    def test_expected_fixture_identity_must_match_manifest_model(self):
        expected = expected_pair()
        expected[1] = expected[1].model_copy(update={"model_revision": "moving-tag"})
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        )

        with pytest.raises(ValidationError, match="model revision or dtype"):
            build_manifest(
                fixtures=[],
                expected_fixtures=expected,
                tolerance_policy=same_prefix_policy(),
                baseline_calibration=tolerance,
            )

    def test_generated_fixture_family_must_match_expectation(self):
        fixture = FixtureMetadata(
            prompt_id="canonical_01",
            category="regression",
            oracle="transformers",
            num_tokens=1,
            logits_dtype="float32",
            logits_shape=(0, 0),
            sha256="a" * 64,
            filename="canonical_01.transformers.safetensors",
        )
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        )

        with pytest.raises(ValidationError, match="family does not match"):
            build_manifest(
                fixtures=[fixture],
                expected_fixtures=expected_pair(),
                tolerance_policy=same_prefix_policy(),
                baseline_calibration=tolerance,
            )

    def test_duplicate_generated_fixture_identifier_is_rejected(self):
        fixture = FixtureMetadata(
            prompt_id="canonical_01",
            category="canonical",
            oracle="transformers",
            num_tokens=1,
            logits_dtype="float32",
            logits_shape=(1, 151936),
            sha256="a" * 64,
            filename="canonical_01.transformers.safetensors",
        )
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        )

        with pytest.raises(ValidationError, match="duplicate generated fixture identifier"):
            build_manifest(
                fixtures=[fixture, fixture.model_copy()],
                expected_fixtures=expected_pair(),
                tolerance_policy=same_prefix_policy(),
                baseline_calibration=tolerance,
            )

    def test_expected_identifier_and_filename_are_canonical(self):
        expected = expected_pair()
        expected[0] = expected[0].model_copy(update={"fixture_id": "wrong"})
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        )

        with pytest.raises(ValidationError, match="non-canonical fixture identifier"):
            build_manifest(
                fixtures=[],
                expected_fixtures=expected,
                tolerance_policy=same_prefix_policy(),
                baseline_calibration=tolerance,
            )

    def test_build_manifest_minimal(self):
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
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
                sha256="a" * 64,
                filename="canonical_01.transformers.safetensors",
            )
        ]
        manifest = build_manifest(
            fixtures=fixtures,
            expected_fixtures=expected_pair(),
            tolerance_policy=same_prefix_policy(),
            baseline_calibration=tolerance,
            generated_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        )
        assert manifest.schema_version == 4
        assert manifest.product_version == "v0.2.0"
        assert manifest.golden_version == "goldens-v0.2"
        assert manifest.archive.filename == "goldens-v0.2.tar.gz"
        assert manifest.model.id == "Qwen/Qwen3-0.6B"
        assert manifest.model.arch == "Qwen3ForCausalLM"
        assert manifest.model.vocab_size == 151936
        assert manifest.generation.canonical_max_tokens == 64
        assert manifest.generation.regression_max_tokens == 32
        assert len(manifest.fixtures) == 1
        assert manifest.generation.temperature == 0.0
        assert manifest.tolerance_policy == same_prefix_policy()

    def test_write_read_roundtrip(self, tmp_path):
        tolerance = BaselineCalibration(
            candidate_atol=0.01,
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
                sha256="a" * 64,
                filename="canonical_01.transformers.safetensors",
            )
        ]
        manifest = build_manifest(
            fixtures=fixtures,
            expected_fixtures=expected_pair(),
            tolerance_policy=same_prefix_policy(),
            baseline_calibration=tolerance,
            generated_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        )
        path = tmp_path / "manifest.json"
        write_manifest(manifest, path)
        restored = read_manifest(path)
        assert restored.model.id == manifest.model.id
        assert restored.fixtures[0].sha256 == "a" * 64

    def test_get_oracle_versions(self):
        versions = get_oracle_versions()
        assert isinstance(versions, OracleVersions)
        assert isinstance(versions.transformers, str)
        assert isinstance(versions.vllm, str)
