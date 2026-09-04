"""Golden fixture generator for vllm-oxide oracle comparison."""

from golden_gen.schema import (
    BaselineCalibration,
    FixtureMetadata,
    Manifest,
    ManifestEntry,
    OracleName,
    OracleVersions,
    PromptCategory,
    PromptSpec,
    TolerancePolicy,
)

__version__ = "0.1.0"
__all__ = [
    "BaselineCalibration",
    "FixtureMetadata",
    "Manifest",
    "ManifestEntry",
    "OracleName",
    "OracleVersions",
    "PromptCategory",
    "PromptSpec",
    "TolerancePolicy",
]
