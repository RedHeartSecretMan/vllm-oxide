"""Build, write, and read manifest files."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from golden_gen.config import (
    ARCH,
    ATTN_IMPLEMENTATION,
    CANONICAL_MAX_TOKENS,
    MODEL_DTYPE,
    MODEL_ID,
    MODEL_REVISION,
    NANO_VLLM_TEMPERATURE,
    REGRESSION_MAX_TOKENS,
    VOCAB_SIZE,
)
from golden_gen.schema import (
    FixtureMetadata,
    GenerationConfig,
    KnownDeviation,
    Manifest,
    ModelInfo,
    OracleVersions,
    ToleranceCalibration,
)


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

    try:
        nanovllm_ver = importlib.metadata.version("nanovllm")
    except importlib.metadata.PackageNotFoundError:
        nanovllm_ver = "unknown"

    return OracleVersions(
        transformers=transformers_ver,
        vllm=vllm_ver,
        nanovllm=nanovllm_ver,
    )


def build_manifest(
    fixtures: list[FixtureMetadata],
    tolerance: ToleranceCalibration,
    cross_validation: list[KnownDeviation],
    *,
    generated_at: datetime | None = None,
    suspect_prompt_ids: list[str] | None = None,
) -> Manifest:
    """Build a Manifest from fixture metadata and cross-validation results.

    Args:
        fixtures: List of FixtureMetadata for all generated fixtures.
        tolerance: Calibrated tolerance values.
        cross_validation: List of known oracle deviations.
        generated_at: Timestamp (defaults to now UTC).
        suspect_prompt_ids: Prompt IDs whose divergence exceeds threshold.

    Returns:
        A fully populated Manifest.
    """
    if generated_at is None:
        generated_at = datetime.now(UTC)

    return Manifest(
        schema_version=1,
        generated_at=generated_at,
        model=ModelInfo(
            id=MODEL_ID,
            revision=MODEL_REVISION,
            arch=ARCH,
            dtype=MODEL_DTYPE,
            vocab_size=VOCAB_SIZE,
        ),
        oracle_versions=get_oracle_versions(),
        generation=GenerationConfig(
            canonical_max_tokens=CANONICAL_MAX_TOKENS,
            regression_max_tokens=REGRESSION_MAX_TOKENS,
            temperature=NANO_VLLM_TEMPERATURE,
            attn_implementation=ATTN_IMPLEMENTATION,
        ),
        tolerance=tolerance,
        cross_validation=cross_validation,
        fixtures=fixtures,
        suspect_prompt_ids=suspect_prompt_ids or [],
    )


def write_manifest(manifest: Manifest, path: str | Path) -> None:
    """Write manifest to a JSON file."""
    manifest.to_json(path)


def read_manifest(path: str | Path) -> Manifest:
    """Read manifest from a JSON file."""
    return Manifest.from_json(path)
