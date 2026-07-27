"""vLLM V1 oracle adapter.

Uses vLLM's LLM.generate() with logprobs_mode='raw_logits' to capture
full pre-sampling logits for canonical prompts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from golden_gen.config import (
    CANONICAL_MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    REGRESSION_MAX_TOKENS,
    TOP_K_REGRESSION,
    VOCAB_SIZE,
)
from golden_gen.oracles.base import OracleResult
from golden_gen.schema import PromptSpec


class VllmV1Oracle:
    """Oracle using vLLM V1 engine.

    Uses logprobs_mode='raw_logits' and logprobs=-1 to capture
    full pre-sampling logits (not log-softmax) for canonical prompts.
    For regression, uses logprobs=5 for top-5 only.
    """

    name = "vllm_v1"

    def __init__(self) -> None:
        from vllm import LLM

        self.llm = LLM(
            model=MODEL_ID,
            revision=MODEL_REVISION,
            dtype="bfloat16",
            enforce_eager=True,
            logprobs_mode="raw_logits",
            max_logprobs=-1,
            gpu_memory_utilization=0.65,
        )

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
        logits = np.zeros((n, VOCAB_SIZE), dtype=np.float32)
        for t, step_dict in enumerate(completion.logprobs):
            if len(step_dict) != VOCAB_SIZE:
                raise RuntimeError(
                    f"vLLM V1 raw_logits mode returned {len(step_dict)} entries "
                    f"at step {t}, expected full vocab ({VOCAB_SIZE}). Check that "
                    f"logprobs=-1 and max_logprobs=-1 are both set. Sparse logits "
                    f"would produce mostly-zero ground truth -- aborting to prevent "
                    f"silent corruption."
                )
            for tok_id, logprob_obj in step_dict.items():
                logits[t, tok_id] = logprob_obj.logprob

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
            for t, step_dict in enumerate(completion.logprobs):
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
            logits = np.zeros((n, VOCAB_SIZE), dtype=np.float32)
            for t, step_dict in enumerate(completion.logprobs):
                if len(step_dict) != VOCAB_SIZE:
                    raise RuntimeError(
                        f"vLLM V1 raw_logits mode returned {len(step_dict)} entries "
                        f"at step {t}, expected full vocab ({VOCAB_SIZE}). Check that "
                        f"logprobs=-1 and max_logprobs=-1 are both set. Sparse logits "
                        f"would produce mostly-zero ground truth -- aborting to prevent "
                        f"silent corruption."
                    )
                for tok_id, logprob_obj in step_dict.items():
                    logits[t, tok_id] = logprob_obj.logprob
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
