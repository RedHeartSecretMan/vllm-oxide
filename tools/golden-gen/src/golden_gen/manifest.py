"""Build, write, and read manifest files."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from golden_gen.config import (
    ARCH,
    ATTN_IMPLEMENTATION,
    CANONICAL_MAX_TOKENS,
    GOLDEN_VERSION,
    MODEL_CONFIG_SHA256,
    MODEL_DTYPE,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    PRODUCT_VERSION,
    REGRESSION_MAX_TOKENS,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
    VOCAB_SIZE,
)
from golden_gen.schema import (
    ArchiveInfo,
    BaselineCalibration,
    DiscoveredFixture,
    ExpectedFixture,
    FixtureMetadata,
    GenerationConfig,
    KernelPaths,
    Manifest,
    ModelInfo,
    OracleName,
    OracleRole,
    OracleVersions,
    RequiredComparison,
    RuntimeInfo,
    TolerancePolicy,
)


def build_expected_fixtures(discovered: list[DiscoveredFixture]) -> list[ExpectedFixture]:
    """Declare every required reference and baseline artifact for discovered cases."""
    expected: list[ExpectedFixture] = []
    for case in discovered:
        reference_comparison: RequiredComparison = "l1" if case.family == "regression" else "l1_l2"
        oracle_contracts: tuple[tuple[OracleName, OracleRole, RequiredComparison], ...] = (
            ("transformers", "reference", reference_comparison),
            ("vllm", "baseline", "calibration"),
        )
        for oracle, role, comparison in oracle_contracts:
            fixture_id = f"{case.prompt_id}.{oracle}"
            expected.append(
                ExpectedFixture(
                    fixture_id=fixture_id,
                    prompt_id=case.prompt_id,
                    family=case.family,
                    model_revision=MODEL_REVISION,
                    dtype=MODEL_DTYPE,
                    oracle=oracle,
                    oracle_role=role,
                    required_comparison=comparison,
                    filename=f"{fixture_id}.safetensors",
                )
            )
    return expected


def get_oracle_versions() -> OracleVersions:
    """Capture installed oracle versions via importlib.metadata."""
    import importlib.metadata

    try:
        transformers_ver = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_ver = "unknown"

    try:
        vllm_ver = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_ver = "unknown"

    return OracleVersions(
        transformers=transformers_ver,
        vllm=vllm_ver,
    )


def build_manifest(
    fixtures: list[FixtureMetadata],
    baseline_calibration: BaselineCalibration,
    *,
    archive: ArchiveInfo,
    tolerance_policy: TolerancePolicy,
    expected_fixtures: list[ExpectedFixture],
    runtime: RuntimeInfo,
    kernel_paths: KernelPaths,
    generated_at: datetime | None = None,
) -> Manifest:
    """Build a Manifest from fixtures, baseline observations, and tolerance policy.

    Args:
        fixtures: List of FixtureMetadata for all generated fixtures.
        expected_fixtures: Independent contracts for all required fixtures.
        tolerance_policy: Versioned L1/L2 mathematical acceptance inputs.
        baseline_calibration: Baseline-oracle calibration observations.
        generated_at: Timestamp (defaults to now UTC).

    Returns:
        A fully populated Manifest.
    """
    if generated_at is None:
        generated_at = datetime.now(UTC)

    return Manifest(
        schema_version=4,
        product_version=PRODUCT_VERSION,
        golden_version=GOLDEN_VERSION,
        archive=archive,
        generated_at=generated_at,
        model=ModelInfo(
            id=MODEL_ID,
            revision=MODEL_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            config_sha256=MODEL_CONFIG_SHA256,
            tokenizer_sha256=TOKENIZER_SHA256,
            weights_sha256=MODEL_WEIGHTS_SHA256,
            arch=ARCH,
            dtype=MODEL_DTYPE,
            vocab_size=VOCAB_SIZE,
        ),
        oracle_versions=OracleVersions(
            transformers=runtime.transformers_version,
            vllm=runtime.vllm_version,
        ),
        runtime=runtime,
        kernel_paths=kernel_paths,
        generation=GenerationConfig(
            canonical_max_tokens=CANONICAL_MAX_TOKENS,
            regression_max_tokens=REGRESSION_MAX_TOKENS,
            temperature=0.0,
            attn_implementation=ATTN_IMPLEMENTATION,
        ),
        tolerance_policy=tolerance_policy,
        baseline_calibration=baseline_calibration,
        expected_fixtures=expected_fixtures,
        fixtures=fixtures,
    )


def write_manifest(manifest: Manifest, path: str | Path) -> None:
    """Write manifest to a JSON file."""
    manifest.to_json(path)


def read_manifest(path: str | Path) -> Manifest:
    """Read manifest from a JSON file."""
    return Manifest.from_json(path)
