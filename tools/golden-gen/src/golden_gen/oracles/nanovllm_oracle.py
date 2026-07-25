"""nano-vllm oracle adapter with monkey-patch for logit capture.

nano-vllm does not expose logits publicly and forbids temperature=0.
We monkey-patch ModelRunner.run to stash per-step FP32 logits on
each Sequence object, and use temperature=1e-9 (near-zero, passes
the >1e-10 assert).

Key details:
  - nano-vllm's config.py asserts os.path.isdir(self.model), so we
    must pass a LOCAL directory path, not a HuggingFace hub ID.
    We use huggingface_hub.snapshot_download to materialize the model
    (pinned to MODEL_REVISION) in HuggingFace's cache, then pass that
    local path to LLM().
  - After generation completes, sequences are deallocated. We track
    them during the patched run() by storing them on the model_runner
    instance (_golden_tracked_seqs). After generate() returns, we
    retrieve tracked sequences from model_runner and match them to
    outputs by order (nano-vllm generates outputs sorted by seq_id,
    and sequences are created in add_request() order).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from golden_gen.config import (
    CANONICAL_MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    NANO_VLLM_TEMPERATURE,
    REGRESSION_MAX_TOKENS,
    TOP_K_REGRESSION,
)
from golden_gen.oracles.base import OracleResult
from golden_gen.schema import PromptSpec


def _apply_patch() -> None:
    """Monkey-patch ModelRunner.run to stash per-step FP32 logits on sequences.

    The patched method tracks ALL sequences it sees on the model_runner
    instance via _golden_tracked_seqs, so callers can retrieve them
    after generation completes.
    """
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.sequence import Sequence

    def _patched_run(self: Any, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids: list[int] = []
        if self.rank == 0:
            token_ids = self.sampler(logits, temperatures).tolist()
            fp32_logits = logits.float().cpu().numpy()
            # Lazily init the tracker on the model_runner instance
            if not hasattr(self, "_golden_tracked_seqs"):
                self._golden_tracked_seqs = []
            for seq, vec in zip(seqs, fp32_logits, strict=True):
                if not hasattr(seq, "_golden_logits_list"):
                    seq._golden_logits_list = []
                    self._golden_tracked_seqs.append(seq)
                seq._golden_logits_list.append(vec)
        from nanovllm.utils.context import reset_context

        reset_context()
        return token_ids

    ModelRunner.run = _patched_run


class NanovllmOracle:
    """Oracle using nano-vllm engine with monkey-patched logit capture.

    Uses huggingface_hub.snapshot_download to materialize the model
    to a local path (required by nano-vllm's os.path.isdir assert).
    """

    name = "nanovllm"

    def __init__(self) -> None:
        from huggingface_hub import snapshot_download

        _apply_patch()
        from nanovllm import LLM, SamplingParams

        self._LLM = LLM
        self._SamplingParams = SamplingParams

        # nano-vllm requires a LOCAL DIRECTORY path, not a hub ID
        # (see nanovllm/config.py line 21: assert os.path.isdir(self.model))
        local_model_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
        )
        self.llm = LLM(
            model=local_model_path,
            enforce_eager=True,
            max_model_len=2048,
        )
        self.tokenizer = self.llm.tokenizer

    def _run(self, prompts: list[str], max_tokens: int) -> list[OracleResult]:
        sp = self._SamplingParams(
            temperature=NANO_VLLM_TEMPERATURE,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate(prompts, sp, use_tqdm=False)

        # Retrieve tracked sequences from the model_runner (populated by the patched run)
        tracked: list[Any] = getattr(self.llm.model_runner, "_golden_tracked_seqs", [])
        # Reset for the next call
        if hasattr(self.llm.model_runner, "_golden_tracked_seqs"):
            self.llm.model_runner._golden_tracked_seqs = []

        if len(tracked) != len(prompts):
            raise RuntimeError(
                f"nano-vllm tracked {len(tracked)} sequences but got "
                f"{len(prompts)} prompts; the monkey-patch is not capturing "
                "sequences correctly."
            )

        # Verify the order-matching assumption: outputs are sorted by seq_id
        # ascending (nano-vllm llm_engine.py:88), and sequences are created
        # in add_request order. Tracked sequences were added in first-seen
        # order during run(), which matches add_request order at TP=1.
        for i, (output, seq) in enumerate(zip(outputs, tracked, strict=True)):
            expected_tokens = output["token_ids"]
            actual_tokens = (
                list(seq.completion_token_ids) if hasattr(seq, "completion_token_ids") else []
            )
            if abs(len(expected_tokens) - len(actual_tokens)) > 1:
                raise RuntimeError(
                    f"nano-vllm order-mismatch at index {i}: output has "
                    f"{len(expected_tokens)} tokens but tracked seq has "
                    f"{len(actual_tokens)}. The outputs[i] <-> tracked[i] "
                    f"assumption may be wrong."
                )

        results: list[OracleResult] = []
        for prompt_text, output, seq in zip(prompts, outputs, tracked, strict=True):
            token_ids = np.array(output["token_ids"], dtype=np.int64)
            n_prompt_tokens = len(self.tokenizer.encode(prompt_text))

            logits_list = getattr(seq, "_golden_logits_list", None)
            if logits_list is None or not logits_list:
                raise RuntimeError(
                    f"No logits captured for prompt {prompt_text!r}; the monkey-patch did not fire."
                )

            logits_per_step = np.stack(logits_list, axis=0).astype(np.float32)

            if logits_per_step.shape[0] != len(token_ids):
                min_len = min(logits_per_step.shape[0], len(token_ids))
                logits_per_step = logits_per_step[:min_len]
                token_ids = token_ids[:min_len]

            topk_indices = np.argsort(-logits_per_step, axis=1)[:, :TOP_K_REGRESSION].astype(
                np.int64
            )
            topk_logits = np.take_along_axis(logits_per_step, topk_indices, axis=1)

            results.append(
                OracleResult(
                    token_ids=token_ids,
                    logits_per_step=logits_per_step,
                    top5_indices=topk_indices,
                    top5_logits=topk_logits,
                    n_prompt_tokens=n_prompt_tokens,
                )
            )
        return results

    def generate(self, prompt: PromptSpec) -> OracleResult:
        max_tokens = (
            CANONICAL_MAX_TOKENS if prompt.category == "canonical" else REGRESSION_MAX_TOKENS
        )
        return self._run([prompt.prompt], max_tokens)[0]

    def close(self) -> None:
        self.llm.exit()
        del self.llm
        import gc

        gc.collect()
