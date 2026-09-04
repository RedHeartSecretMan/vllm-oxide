"""Pydantic v2 models for the golden fixture manifest schema."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PromptCategory = Literal["canonical", "regression"]
OracleName = Literal["transformers", "vllm", "fake"]
FixtureFamily = Literal["canonical", "batch", "regression"]
OracleRole = Literal["reference", "baseline"]
RequiredComparison = Literal["l1", "l1_l2", "calibration"]


class PromptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    category: PromptCategory
    prompt: str  # For batch prompts, this is a description; sub_prompts holds the actual prompts
    description: str
    chat_template: bool = False
    note: str | None = None
    sub_prompts: list[str] | None = None  # When set, this is a batch prompt

    @property
    def is_batch(self) -> bool:
        """Whether this prompt exercises the batch/continuous-batching path."""
        return self.sub_prompts is not None and len(self.sub_prompts) > 1

    @model_validator(mode="after")
    def validate_batch_shape(self) -> Self:
        if self.sub_prompts is not None and not 2 <= len(self.sub_prompts) <= 26:
            raise ValueError("sub_prompts must contain 2 to 26 prompts")
        if self.sub_prompts is not None and self.category != "canonical":
            raise ValueError("batch prompts must belong to the canonical corpus")
        return self


class DiscoveredFixture(BaseModel):
    """One concrete prompt case discovered from the prompt corpora."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str = Field(min_length=1)
    family: FixtureFamily
    prompt: str


class ExpectedFixture(BaseModel):
    """Manifest contract for one required oracle artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fixture_id: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    family: FixtureFamily
    model_revision: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    oracle: OracleName
    oracle_role: OracleRole
    required_comparison: RequiredComparison
    filename: str = Field(min_length=1)


class OracleVersions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transformers: str
    vllm: str


class FixtureMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str
    category: PromptCategory
    oracle: OracleName
    num_tokens: int = Field(gt=0)
    logits_dtype: Literal["float32"]
    logits_shape: tuple[int, int]
    sha256: str
    filename: str


class TolerancePolicy(BaseModel):
    """Versioned mathematical inputs for reference-oracle comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    version: Literal["same-prefix-v1"]
    dtype: str = Field(min_length=1)
    kernel: str = Field(min_length=1)
    l1_near_tie_max_abs_logit_gap: float = Field(ge=0.0)
    l2_atol: float = Field(ge=0.0)
    rationale: str = Field(min_length=1)
    evidence: list[str]


class BaselineCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    candidate_atol: float = Field(ge=0.0)
    observed_max_abs_diff: float = Field(ge=0.0)
    calibration_factor: float = Field(gt=0.0)
    method: str = Field(min_length=1)


class ModelInfo(BaseModel):
    """Provenance of the model used to generate goldens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    arch: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    vocab_size: int = Field(gt=0)


class GenerationConfig(BaseModel):
    """Parameters used during golden generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_max_tokens: int
    regression_max_tokens: int
    temperature: float
    attn_implementation: str


class ManifestEntry(BaseModel):
    """A single entry in a manifest: a (prompt_id, oracle) fixture file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str
    oracle: OracleName
    filename: str
    sha256: str
    num_tokens: int
    logits_shape: tuple[int, int]


class Manifest(BaseModel):
    """Top-level manifest describing all generated fixtures."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3] = 3
    generated_at: datetime
    model: ModelInfo
    oracle_versions: OracleVersions
    generation: GenerationConfig
    tolerance_policy: TolerancePolicy
    baseline_calibration: BaselineCalibration
    expected_fixtures: list[ExpectedFixture] = Field(min_length=1)
    fixtures: list[FixtureMetadata]
    calibrated_fixtures: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def generated_fixtures_match_expectations(self) -> Self:
        if (
            self.tolerance_policy.dtype != self.model.dtype
            or self.tolerance_policy.kernel != self.generation.attn_implementation
        ):
            raise ValueError("tolerance policy scope does not match model dtype and kernel")
        if not self.tolerance_policy.rationale.strip() or any(
            not item.strip() for item in self.tolerance_policy.evidence
        ):
            raise ValueError("tolerance policy rationale and evidence must be non-empty")
        if self.calibrated_fixtures and not self.tolerance_policy.evidence:
            raise ValueError("calibrated tolerance policy requires non-empty evidence")
        fixture_ids = [entry.fixture_id for entry in self.expected_fixtures]
        if len(fixture_ids) != len(set(fixture_ids)):
            raise ValueError("duplicate expected fixture identifier")
        for entry in self.expected_fixtures:
            if not entry.prompt_id.replace("_", "").replace("-", "").isalnum() or not (
                entry.prompt_id.isascii()
            ):
                raise ValueError(f"unsupported fixture identifier: {entry.prompt_id}")
            canonical_id = f"{entry.prompt_id}.{entry.oracle}"
            if entry.fixture_id != canonical_id or entry.filename != f"{canonical_id}.safetensors":
                raise ValueError(
                    f"non-canonical fixture identifier or filename: {entry.fixture_id}"
                )
            if entry.model_revision != self.model.revision or entry.dtype != self.model.dtype:
                raise ValueError(
                    f"fixture {entry.fixture_id} model revision or dtype does not match manifest"
                )
            reference_comparison = "l1" if entry.family == "regression" else "l1_l2"
            valid_reference = (
                entry.oracle == "transformers"
                and entry.oracle_role == "reference"
                and entry.required_comparison == reference_comparison
            )
            valid_baseline = (
                entry.oracle == "vllm"
                and entry.oracle_role == "baseline"
                and entry.required_comparison == "calibration"
            )
            if not (valid_reference or valid_baseline):
                raise ValueError(f"invalid oracle role contract: {entry.fixture_id}")
        roles_by_prompt: dict[str, set[str]] = {}
        families_by_prompt: dict[str, set[str]] = {}
        for entry in self.expected_fixtures:
            roles_by_prompt.setdefault(entry.prompt_id, set()).add(entry.oracle_role)
            families_by_prompt.setdefault(entry.prompt_id, set()).add(entry.family)
        for prompt_id, roles in roles_by_prompt.items():
            if roles != {"reference", "baseline"}:
                raise ValueError(
                    f"fixture {prompt_id} must declare reference and baseline oracle roles"
                )
            if len(families_by_prompt[prompt_id]) != 1:
                raise ValueError(f"fixture {prompt_id} declares inconsistent families")
        expected = {
            (entry.prompt_id, entry.oracle, entry.filename): entry
            for entry in self.expected_fixtures
        }
        generated_ids = [(fixture.prompt_id, fixture.oracle) for fixture in self.fixtures]
        if len(generated_ids) != len(set(generated_ids)):
            raise ValueError("duplicate generated fixture identifier")
        if len(self.calibrated_fixtures) != len(set(self.calibrated_fixtures)):
            raise ValueError("duplicate calibrated fixture identifier")
        expected_by_id = {entry.fixture_id: entry for entry in self.expected_fixtures}
        generated_by_id = {f"{fixture.prompt_id}.{fixture.oracle}" for fixture in self.fixtures}
        for fixture_id in self.calibrated_fixtures:
            calibrated = expected_by_id.get(fixture_id)
            if calibrated is None:
                raise ValueError(f"unmatched calibrated fixture: {fixture_id}")
            if calibrated.oracle_role != "baseline" or fixture_id not in generated_by_id:
                raise ValueError(f"invalid baseline calibration evidence: {fixture_id}")
        for fixture in self.fixtures:
            key = (fixture.prompt_id, fixture.oracle, fixture.filename)
            if key not in expected:
                raise ValueError(
                    "unmatched generated fixture: "
                    f"{fixture.prompt_id}.{fixture.oracle} ({fixture.filename})"
                )
            expected_category = (
                "regression" if expected[key].family == "regression" else "canonical"
            )
            if fixture.category != expected_category:
                raise ValueError(
                    f"generated fixture {fixture.prompt_id} family does not match expectation"
                )
        return self

    def to_json(self, path: str | Path) -> None:
        """Serialize to JSON file."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> Manifest:
        """Deserialize from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)
