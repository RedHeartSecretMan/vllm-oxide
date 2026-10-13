"""Pydantic v2 models for the golden fixture manifest schema."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

PromptCategory = Literal["canonical", "regression"]
OracleName = Literal["transformers", "vllm", "fake"]


def validate_fixture_component(value: str) -> str:
    """Keep identifiers portable and independent of filesystem path syntax."""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is None
    ):
        raise ValueError("fixture identifiers must be 1 to 128 ASCII letters, digits, '_' or '-'")
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if value.upper() in reserved:
        raise ValueError("fixture identifiers must not be reserved device names")
    return value


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
        validate_fixture_component(self.id)
        if self.sub_prompts is not None and not 2 <= len(self.sub_prompts) <= 26:
            raise ValueError("sub_prompts must contain 2 to 26 prompts")
        return self


class OracleVersions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transformers: str
    vllm: str


class FixtureMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str
    category: PromptCategory
    oracle: OracleName
    num_tokens: int
    logits_dtype: Literal["float32"]
    logits_shape: tuple[int, int]
    sha256: str
    filename: str


class ToleranceCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    atol: float
    observed_max_abs_diff: float
    calibration_factor: float
    method: str


class ModelInfo(BaseModel):
    """Provenance of the model used to generate goldens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    revision: str
    arch: str
    dtype: str
    vocab_size: int


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

    schema_version: int = 1
    generated_at: datetime
    model: ModelInfo
    oracle_versions: OracleVersions
    generation: GenerationConfig
    tolerance: ToleranceCalibration
    fixtures: list[FixtureMetadata]
    regression_skip_map: dict[str, list[int]] = {}

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
