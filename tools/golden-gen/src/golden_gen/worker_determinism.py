"""Process-local determinism and typed evidence from the actual baseline worker."""

from __future__ import annotations

import os
import random
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from golden_gen.environment import validate_deterministic_environment


class DeterminismState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: Literal[True]
    warn_only: Literal[False]
    cuda_initialized: bool


def seed_and_enable_determinism(torch: Any) -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=False)


def require_worker_determinism(torch: Any) -> DeterminismState:
    validate_deterministic_environment(os.environ)
    return DeterminismState(
        enabled=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cuda_initialized=torch.cuda.is_initialized(),
    )


def configure_worker_before_cuda(torch: Any) -> DeterminismState:
    validate_deterministic_environment(os.environ)
    if torch.cuda.is_initialized():
        raise RuntimeError("baseline worker CUDA initialized before deterministic setup")
    seed_and_enable_determinism(torch)
    state = require_worker_determinism(torch)
    if state.cuda_initialized:
        raise RuntimeError("baseline deterministic setup unexpectedly initialized CUDA")
    return state


class AttentionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    layer: str = Field(min_length=1)
    backend: Literal["FLASH_ATTN"]
    flash_attn_version: Literal[2]


class BaselineWorkerEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    phase: Literal["ready", "complete"]
    pid: int = Field(gt=0)
    worker_class: Literal["golden_gen.oracles.vllm_worker.DeterministicWorker"]
    before_cuda: DeterminismState
    current: DeterminismState
    attention: list[AttentionEvidence] = Field(min_length=1)
    expected_attention_layers: int = Field(gt=0)
    dtype: Literal["torch.bfloat16"]
    seed: Literal[0]
    tensor_parallel_size: Literal[1]
    enforce_eager: Literal[True]
    compilation_mode: Literal["NONE"]
    cudagraph_mode: Literal["NONE"]

    @model_validator(mode="after")
    def complete_worker_state(self) -> BaselineWorkerEvidence:
        if self.before_cuda.cuda_initialized or not self.current.cuda_initialized:
            raise ValueError("worker evidence does not prove setup before CUDA initialization")
        if len(self.attention) != self.expected_attention_layers or len(
            {entry.layer for entry in self.attention}
        ) != len(self.attention):
            raise ValueError("worker evidence does not cover every attention layer")
        return self
