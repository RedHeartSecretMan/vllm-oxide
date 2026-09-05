"""vLLM 0.18.1 supported worker_cls hook, loaded in the real spawn GPU worker."""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.v1.worker.gpu_worker import Worker

from golden_gen.worker_determinism import (
    BaselineWorkerEvidence,
    configure_worker_before_cuda,
    require_worker_determinism,
)


class DeterministicWorker(Worker):  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._release_before_cuda = configure_worker_before_cuda(torch)
        super().__init__(*args, **kwargs)

    def init_device(self) -> None:
        # Check, never silently repair a lost flag before CUDA initialization.
        require_worker_determinism(torch)
        super().init_device()
        require_worker_determinism(torch)

    def release_worker_evidence(self, phase: str) -> dict[str, Any]:
        layers = [
            {
                "layer": name,
                "backend": layer.attn_backend.get_name(),
                "flash_attn_version": layer.impl.vllm_flash_attn_version,
            }
            for name, layer in self.model_runner.model.named_modules()
            if isinstance(layer, Attention)
        ]
        config = self.vllm_config
        evidence = BaselineWorkerEvidence.model_validate(
            {
                "phase": phase,
                "pid": os.getpid(),
                "worker_class": f"{type(self).__module__}.{type(self).__qualname__}",
                "before_cuda": self._release_before_cuda,
                "current": require_worker_determinism(torch),
                "attention": layers,
                "expected_attention_layers": config.model_config.hf_config.num_hidden_layers,
                "dtype": str(config.model_config.dtype),
                "seed": config.model_config.seed,
                "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
                "enforce_eager": config.model_config.enforce_eager,
                "compilation_mode": config.compilation_config.mode.name,
                "cudagraph_mode": config.compilation_config.cudagraph_mode.name,
            }
        )
        return evidence.model_dump()
