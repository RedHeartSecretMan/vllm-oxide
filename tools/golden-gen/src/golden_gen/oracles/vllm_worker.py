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

    def load_model(self) -> None:
        super().load_model()
        from golden_gen.fixed_prefix_vllm import FixedPrefixProcessor

        self._fixed_events: list[dict[str, Any]] = []
        processors = [
            p
            for p in self.model_runner.input_batch.logitsprocs.non_argmax_invariant
            if isinstance(p, FixedPrefixProcessor)
        ]
        if not processors:
            return
        if len(processors) != 1:
            raise ValueError("fixed-prefix worker requires exactly one identity owner")

        def record_execution(_module: Any, _args: Any, kwargs: Any) -> None:
            runner = self.model_runner
            if not any(p.state for p in processors):
                return  # warmup/empty request state consumes no fixed replay step
            ids, positions = kwargs["input_ids"], kwargs["positions"]
            members = []
            execution = {}
            for index, request_id in enumerate(runner.input_batch.req_ids):
                start, end = map(int, runner.query_start_loc.np[index : index + 2])
                if end <= start:
                    continue
                tokens = ids[start:end].cpu().tolist()
                pos = positions[start:end].cpu().tolist()
                if pos != list(range(pos[0], pos[-1] + 1)):
                    raise ValueError("vLLM executed noncontiguous positions")
                history = runner.requests[request_id].output_token_ids
                row = dict(
                    request_id=processors[0].request_identity(index, request_id),
                    native_request_id=request_id,
                    input_token_ids=tokens,
                    positions=[pos[0], pos[-1] + 1],
                    cached_range=[0, pos[0]],
                    kv_length=int(runner.seq_lens.np[index]),
                    sampling_allowed=not bool(runner.discard_request_mask.np[index]),
                    completion_step=len(history),
                    phase="decode" if history and len(tokens) == 1 else "prefill",
                )
                execution[index] = row
                members.append(row)
            for processor in processors:
                processor.execution = execution
            self._fixed_events.append(
                dict(
                    plan_id=len(self._fixed_events),
                    token_budget=sum(len(m["input_token_ids"]) for m in members),
                    members=members,
                )
            )

        self._fixed_handle = self.model_runner.model.register_forward_pre_hook(
            record_execution, with_kwargs=True
        )

    def release_fixed_prefix_evidence(self) -> dict[str, Any]:
        from golden_gen.fixed_prefix_vllm import FixedPrefixProcessor

        rows = []
        bindings = []
        for processor in self.model_runner.input_batch.logitsprocs.non_argmax_invariant:
            if isinstance(processor, FixedPrefixProcessor):
                bindings.extend(processor.request_bindings())
                rows.extend(processor.rows)
                processor.rows = []
        events, self._fixed_events = self._fixed_events, []
        return dict(
            rows=rows,
            request_bindings=bindings,
            execution_events=events,
            allocated_cache_blocks=self.model_runner.kv_cache_config.num_blocks,
            cache_block_size=self.vllm_config.cache_config.block_size,
        )

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
