"""vLLM V1 oracle adapter.

Uses vLLM's LLM.generate() with logprobs_mode='raw_logits' to capture
full pre-sampling logits for canonical prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from golden_gen.config import (
    CANONICAL_MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    REGRESSION_MAX_TOKENS,
    TOP_K_REGRESSION,
    VOCAB_SIZE,
)
from golden_gen.environment import validate_release_model
from golden_gen.oracles.base import OracleResult
from golden_gen.schema import PromptSpec
from golden_gen.worker_determinism import BaselineWorkerEvidence, seed_and_enable_determinism


def baseline_engine_kwargs() -> dict[str, Any]:
    """Exact no-fallback vLLM baseline construction contract from ADR-0012."""
    return {
        "model": MODEL_ID,
        "worker_cls": "golden_gen.oracles.vllm_worker.DeterministicWorker",
        "tokenizer": MODEL_ID,
        "revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "seed": 0,
        "enforce_eager": True,
        "logprobs_mode": "raw_logits",
        "max_logprobs": -1,
        "gpu_memory_utilization": 0.55,
        "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2},
    }


def _configure_determinism(torch: Any) -> None:
    seed_and_enable_determinism(torch)


def _extract_full_logits(completion: Any, n: int, vocab_size: int) -> NDArray[np.float32]:
    """Extract full logits from a vLLM completion's logprobs.

    Args:
        completion: vLLM Completion object with ``logprobs`` attribute.
        n: Number of generated tokens (steps).
        vocab_size: Vocabulary size.

    Returns:
        Array of shape ``(n, vocab_size)`` in float32.
    """
    if completion.logprobs is None or len(completion.logprobs) != n:
        actual = 0 if completion.logprobs is None else len(completion.logprobs)
        raise RuntimeError(
            f"vLLM returned {actual} logprob rows for {n} generated tokens; "
            "aborting to prevent zero-padded golden logits"
        )
    logits = np.zeros((n, vocab_size), dtype=np.float32)
    for t, step_dict in enumerate(completion.logprobs):
        if len(step_dict) != vocab_size:
            raise RuntimeError(
                f"vLLM V1 raw_logits mode returned {len(step_dict)} entries "
                f"at step {t}, expected full vocab ({vocab_size}). Check that "
                f"logprobs=-1 and max_logprobs=-1 are both set. Sparse logits "
                f"would produce mostly-zero ground truth -- aborting to prevent "
                f"silent corruption."
            )
        for tok_id, logprob_obj in step_dict.items():
            logits[t, tok_id] = logprob_obj.logprob
    return logits


class VllmOracle:
    """Oracle using vLLM V1 engine.

    Uses logprobs_mode='raw_logits' and logprobs=-1 to capture
    full pre-sampling logits (not log-softmax) for canonical prompts.
    For regression, uses logprobs=5 for top-5 only.
    """

    name = "vllm"

    def __init__(self, model_dir: Path) -> None:
        source = str(validate_release_model(model_dir))
        import torch
        from vllm import LLM

        _configure_determinism(torch)
        contract = baseline_engine_kwargs()
        contract.update(model=source, tokenizer=source)
        self.llm = LLM(**contract)
        self._worker_ready = self._worker_evidence("ready")

    def _worker_evidence(self, phase: str) -> BaselineWorkerEvidence:
        records = self.llm.collective_rpc("release_worker_evidence", args=(phase,), timeout=30)
        if len(records) != 1:
            raise ValueError("baseline requires exactly one evidenced GPU worker")
        return BaselineWorkerEvidence.model_validate(records[0])

    def protocol_evidence(self) -> list[BaselineWorkerEvidence]:
        completed = self._worker_evidence("complete")
        if completed.pid != self._worker_ready.pid:
            raise ValueError("baseline GPU worker changed during generation")
        return [self._worker_ready, completed]

    def _generate_canonical(self, prompt: PromptSpec) -> OracleResult:
        from vllm import SamplingParams

        sp = SamplingParams(
            temperature=0,
            max_tokens=CANONICAL_MAX_TOKENS,
            logprobs=-1,
        )
        out = self.llm.generate([prompt.prompt], sp)[0]
        completion = out.outputs[0]
        token_ids = np.fromiter(completion.token_ids, dtype=np.int64)
        n_prompt_tokens = len(out.prompt_token_ids)

        n = len(token_ids)
        logits = _extract_full_logits(completion, n, VOCAB_SIZE)

        return OracleResult.for_canonical(
            token_ids=token_ids,
            logits_per_step=logits,
            n_prompt_tokens=n_prompt_tokens,
        )

    def _generate_regression(self, prompts: list[str], sp: Any) -> list[OracleResult]:

        results: list[OracleResult] = []
        out_list = self.llm.generate(prompts, sp)
        for out in out_list:
            completion = out.outputs[0]
            token_ids = np.fromiter(completion.token_ids, dtype=np.int64)
            n_prompt_tokens = len(out.prompt_token_ids)

            n = len(token_ids)
            top5_indices = np.zeros((n, TOP_K_REGRESSION), dtype=np.int64)
            top5_logits = np.zeros((n, TOP_K_REGRESSION), dtype=np.float32)
            if completion.logprobs is None or len(completion.logprobs) != n:
                actual = 0 if completion.logprobs is None else len(completion.logprobs)
                raise RuntimeError(f"vLLM returned {actual} logprob rows for {n} generated tokens")
            for t, step_dict in enumerate(completion.logprobs):
                if len(step_dict) < TOP_K_REGRESSION:
                    raise RuntimeError(
                        f"vLLM returned only {len(step_dict)} regression logits at step {t}; "
                        f"expected at least {TOP_K_REGRESSION}"
                    )
                topk = sorted(step_dict.items(), key=lambda x: x[1].logprob, reverse=True)[
                    :TOP_K_REGRESSION
                ]
                for k, (tok_id, logprob_obj) in enumerate(topk):
                    top5_indices[t, k] = tok_id
                    top5_logits[t, k] = logprob_obj.logprob

            results.append(
                OracleResult(
                    token_ids=token_ids,
                    logits_per_step=np.empty((0, 0), dtype=np.float32),
                    top5_indices=top5_indices,
                    top5_logits=top5_logits,
                    n_prompt_tokens=n_prompt_tokens,
                )
            )
        return results

    def _generate_canonical_batch(self, sub_prompts: list[str]) -> list[OracleResult]:
        from vllm import SamplingParams

        sp = SamplingParams(
            temperature=0,
            max_tokens=CANONICAL_MAX_TOKENS,
            logprobs=-1,
        )
        out_list = self.llm.generate(sub_prompts, sp)
        results: list[OracleResult] = []
        for out in out_list:
            completion = out.outputs[0]
            token_ids = np.fromiter(completion.token_ids, dtype=np.int64)
            n_prompt_tokens = len(out.prompt_token_ids)
            n = len(token_ids)
            logits = _extract_full_logits(completion, n, VOCAB_SIZE)
            results.append(
                OracleResult.for_canonical(
                    token_ids=token_ids,
                    logits_per_step=logits,
                    n_prompt_tokens=n_prompt_tokens,
                )
            )
        return results

    def generate(self, prompt: PromptSpec) -> list[OracleResult]:
        if prompt.is_batch:
            assert prompt.sub_prompts is not None  # is_batch guarantees this
            return self._generate_canonical_batch(prompt.sub_prompts)
        elif prompt.category == "canonical":
            return [self._generate_canonical(prompt)]
        else:
            from vllm import SamplingParams

            sp = SamplingParams(
                temperature=0,
                max_tokens=REGRESSION_MAX_TOKENS,
                logprobs=TOP_K_REGRESSION,
            )
            return self._generate_regression([prompt.prompt], sp)

    def close(self) -> None:
        del self.llm
        import gc

        gc.collect()
