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
    REGRESSION_MAX_TOKENS,
    VOCAB_SIZE,
)
from golden_gen.schema import (
    FixtureMetadata,
    GenerationConfig,
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

    return OracleVersions(
        transformers=transformers_ver,
        vllm=vllm_ver,
    )


def build_manifest(
    fixtures: list[FixtureMetadata],
    tolerance: ToleranceCalibration,
    *,
    generated_at: datetime | None = None,
    regression_skip_map: dict[str, list[int]] | None = None,
) -> Manifest:
    """Build a Manifest from fixture metadata and tolerance calibration.

    Args:
        fixtures: List of FixtureMetadata for all generated fixtures.
        tolerance: Calibrated tolerance values.
        generated_at: Timestamp (defaults to now UTC).
        regression_skip_map: Positions to skip per regression prompt.

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
            temperature=0.0,
            attn_implementation=ATTN_IMPLEMENTATION,
        ),
        tolerance=tolerance,
        fixtures=fixtures,
        regression_skip_map=regression_skip_map or {},
    )


def write_manifest(manifest: Manifest, path: str | Path) -> None:
    """Write manifest to a JSON file."""
    manifest.to_json(path)


def read_manifest(path: str | Path) -> Manifest:
    """Read manifest from a JSON file."""
    return Manifest.from_json(path)
