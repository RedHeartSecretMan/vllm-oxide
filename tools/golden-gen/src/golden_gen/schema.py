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


class KernelPaths(BaseModel):
    """Exact kernel identities used by the three correctness engines."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: Literal["transformers-4.57.6/torch-2.10.0/sdpa-math"]
    baseline: Literal["vllm-0.18.1/flash-attn-v2/eager"]
    candidate: Literal[
        "vllm-oxide/candle-27f20fea993c81ea6d32ce44018f42b68466525e/"
        "flash-attn-varlen+paged-windowed"
    ]

    @property
    def comparison_scope(self) -> str:
        return f"{self.reference}::vs::{self.candidate}"


class WheelIdentity(BaseModel):
    """One installed registry wheel bound to its lockfile digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    filename: str = Field(pattern=r"^[^/\\]+\.whl$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeInfo(BaseModel):
    """Pinned software and live release-host identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_mode: Literal["release", "dry-run"]
    registry_install_mode: Literal["locked-wheels-only"]
    pythonhashseed: Literal["0"]
    cublas_workspace_config: Literal[":4096:8"]
    python_version: str = Field(pattern=r"^3\.12\.\d+$")
    torch_version: Literal["2.10.0"]
    torch_cuda_version: Literal["12.8"]
    transformers_version: Literal["4.57.6"]
    vllm_version: Literal["0.18.1"]
    xgrammar_version: Literal["0.2.3"]
    triton_version: Literal["3.6.0"]
    cuda_toolkit_version: Literal["13.2.51"]
    rustc_version: str = Field(min_length=1)
    nvidia_driver_version: str = Field(min_length=1)
    gpu_name: str = Field(min_length=1)
    compute_capability: Literal["8.9"]
    os_kernel: str = Field(min_length=1)
    generator_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wheels: list[WheelIdentity] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_locked_wheels_and_host(self) -> Self:
        expected = {
            "torch": self.torch_version,
            "transformers": self.transformers_version,
            "vllm": self.vllm_version,
            "xgrammar": self.xgrammar_version,
            "triton": self.triton_version,
        }
        observed: dict[str, str] = {}
        for wheel in self.wheels:
            normalized = wheel.name.lower().replace("_", "-")
            if normalized in observed:
                raise ValueError(f"duplicate resolved wheel identity: {normalized}")
            observed[normalized] = wheel.version
        if any(observed.get(name) != version for name, version in expected.items()):
            raise ValueError("runtime locked wheel set is incomplete or version-inconsistent")

        if self.evidence_mode == "release":
            live_values = (
                self.rustc_version,
                self.nvidia_driver_version,
                self.gpu_name,
                self.os_kernel,
            )
            forbidden = ("unknown", "fallback", "auto", "dry-run")
            if any(any(word in value.lower() for word in forbidden) for value in live_values):
                raise ValueError("release runtime contains an unknown or fallback host identity")
        return self


class FixtureMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str
    category: PromptCategory
    oracle: OracleName
    num_tokens: int = Field(gt=0)
    logits_dtype: Literal["float32"]
    logits_shape: tuple[int, int]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
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
    tokenizer_revision: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arch: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    vocab_size: int = Field(gt=0)

    @model_validator(mode="after")
    def tokenizer_uses_the_model_revision(self) -> Self:
        if self.tokenizer_revision != self.revision:
            raise ValueError("tokenizer_revision must equal the immutable model revision")
        return self


class GenerationConfig(BaseModel):
    """Parameters used during golden generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_max_tokens: int
    regression_max_tokens: int
    temperature: float
    attn_implementation: str


class ArchiveInfo(BaseModel):
    """Identity of the sole compressed fixture archive release asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: Literal["goldens-v0.2.tar.gz"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


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

    schema_version: Literal[4]
    product_version: Literal["v0.2.0"]
    golden_version: Literal["goldens-v0.2"]
    archive: ArchiveInfo
    generated_at: datetime
    model: ModelInfo
    oracle_versions: OracleVersions
    runtime: RuntimeInfo
    kernel_paths: KernelPaths
    generation: GenerationConfig
    tolerance_policy: TolerancePolicy
    baseline_calibration: BaselineCalibration
    expected_fixtures: list[ExpectedFixture] = Field(min_length=1)
    fixtures: list[FixtureMetadata]
    calibrated_fixtures: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def generated_fixtures_match_expectations(self) -> Self:
        if self.model.dtype != "bfloat16" or self.generation.attn_implementation != "sdpa":
            raise ValueError("golden manifest requires the Transformers BF16 SDPA reference oracle")
        from golden_gen.config import (
            ARCH,
            MODEL_CONFIG_SHA256,
            MODEL_DTYPE,
            MODEL_ID,
            MODEL_REVISION,
            MODEL_WEIGHTS_SHA256,
            TOKENIZER_REVISION,
            TOKENIZER_SHA256,
            VOCAB_SIZE,
        )

        expected_model = {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
            "config_sha256": MODEL_CONFIG_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "arch": ARCH,
            "dtype": MODEL_DTYPE,
            "vocab_size": VOCAB_SIZE,
        }
        if self.model.model_dump() != expected_model:
            raise ValueError("manifest model and tokenizer identity does not match ADR-0012")
        if (
            self.oracle_versions.transformers != self.runtime.transformers_version
            or self.oracle_versions.vllm != self.runtime.vllm_version
        ):
            raise ValueError("oracle_versions do not match the pinned runtime")
        if (
            self.tolerance_policy.dtype != self.model.dtype
            or self.tolerance_policy.kernel != self.kernel_paths.comparison_scope
        ):
            raise ValueError("tolerance policy scope does not match model dtype and kernel paths")
        if not self.tolerance_policy.rationale.strip() or any(
            not item.strip() for item in self.tolerance_policy.evidence
        ):
            raise ValueError("tolerance policy rationale and evidence must be non-empty")
        if self.calibrated_fixtures and not self.tolerance_policy.evidence:
            pending = (
                self.tolerance_policy.l1_near_tie_max_abs_logit_gap == 0.0
                and self.tolerance_policy.l2_atol == 0.0
                and "pending" in self.tolerance_policy.rationale.lower()
            )
            if not pending:
                raise ValueError("calibrated tolerance policy requires non-empty evidence")
        fixture_ids = [entry.fixture_id for entry in self.expected_fixtures]
        if len(fixture_ids) != len(set(fixture_ids)):
            raise ValueError("duplicate expected fixture identifier")
        portable_filenames: set[str] = set()
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
            portable_filename = entry.filename.lower()
            if portable_filename in portable_filenames:
                raise ValueError(f"case-colliding fixture filename: {entry.filename}")
            portable_filenames.add(portable_filename)
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
