"""Golden fixture generator for vllm-oxide oracle triangle."""

from golden_gen.schema import (
    FixtureMetadata,
    KnownDeviation,
    Manifest,
    ManifestEntry,
    OracleName,
    OracleVersions,
    PromptCategory,
    PromptSpec,
    ToleranceCalibration,
)

__version__ = "0.1.0"
__all__ = [
    "FixtureMetadata",
    "KnownDeviation",
    "Manifest",
    "ManifestEntry",
    "OracleName",
    "OracleVersions",
    "PromptCategory",
    "PromptSpec",
    "ToleranceCalibration",
]
