"""Golden fixture generator for vllm-oxide oracle comparison."""

from golden_gen.schema import (
    FixtureMetadata,
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
    "Manifest",
    "ManifestEntry",
    "OracleName",
    "OracleVersions",
    "PromptCategory",
    "PromptSpec",
    "ToleranceCalibration",
]
