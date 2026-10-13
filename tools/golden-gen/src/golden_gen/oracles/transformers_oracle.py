"""HF Transformers oracle adapter."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from golden_gen.config import (
    ATTN_IMPLEMENTATION,
    CANONICAL_MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    REGRESSION_MAX_TOKENS,
    TOP_K_REGRESSION,
)
from golden_gen.environment import validate_release_model
from golden_gen.oracles.base import OracleResult
from golden_gen.schema import PromptSpec

if TYPE_CHECKING:
    pass


def reference_model_kwargs() -> dict[str, object]:
    """Serializable construction contract independently checked by preflight."""
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "torch_dtype": "bfloat16",
        "attn_implementation": ATTN_IMPLEMENTATION,
    }


def _configure_determinism(torch: object) -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)  # type: ignore[attr-defined]
    torch.cuda.manual_seed_all(0)  # type: ignore[attr-defined]
    torch.use_deterministic_algorithms(True, warn_only=False)  # type: ignore[attr-defined]


def _generated_token_logits(
    output: object,
    batch_index: int,
    prompt_width: int,
    eos_token_ids: int | list[int] | tuple[int, ...] | None,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Return real generated steps, including EOS and excluding subsequent padding."""
    steps = output.logits  # type: ignore[attr-defined]
    sequences = output.sequences  # type: ignore[attr-defined]
    if steps is None or len(steps) == 0:
        raise ValueError("Transformers returned no generation logits")
    token_ids = sequences[batch_index, prompt_width:].cpu().numpy().astype(np.int64)
    if token_ids.ndim != 1 or len(token_ids) != len(steps):
        raise ValueError("Transformers generated tokens and logits have different step counts")
    if eos_token_ids is None:
        eos_ids: list[int] = []
    elif isinstance(eos_token_ids, int):
        eos_ids = [eos_token_ids]
    else:
        eos_ids = list(eos_token_ids)
    eos_positions = np.flatnonzero(np.isin(token_ids, eos_ids))
    length = int(eos_positions[0]) + 1 if len(eos_positions) else len(token_ids)
    logits = np.stack(
        [step[batch_index].cpu().numpy().astype(np.float32) for step in steps[:length]],
        axis=0,
    )
    return token_ids[:length], logits


class TransformersOracle:
    """Oracle using HuggingFace Transformers.

    Uses AutoModelForCausalLM.generate() with output_logits=True to capture
    full pre-sampling logits for canonical prompts, and top-5 for regression.
    """

    name = "transformers"

    def __init__(self, model_dir: Path) -> None:
        source = str(validate_release_model(model_dir))
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _configure_determinism(torch)
        contract = reference_model_kwargs()
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                source,
                local_files_only=True,
                revision=contract["revision"],
                torch_dtype=torch.bfloat16,
                attn_implementation=contract["attn_implementation"],
            )
            .to("cuda")
            .eval()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            local_files_only=True,
            revision=contract["tokenizer_revision"],
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, prompt: PromptSpec) -> list[OracleResult]:
        if prompt.is_batch:
            return self._generate_batch(prompt)
        return [self._generate_single(prompt)]

    def _generate_single(self, prompt: PromptSpec) -> OracleResult:
        import torch

        max_tokens = (
            CANONICAL_MAX_TOKENS if prompt.category == "canonical" else REGRESSION_MAX_TOKENS
        )

        inputs = self.tokenizer(
            prompt.prompt,
            return_tensors="pt",
            add_special_tokens=not prompt.chat_template,
        )
        input_ids = inputs["input_ids"].to("cuda")
        n_prompt_tokens = input_ids.shape[1]

        with (
            torch.inference_mode(),
            torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]),
        ):
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_logits=True,
            )

        token_ids, logits = _generated_token_logits(
            out, 0, n_prompt_tokens, self.model.generation_config.eos_token_id
        )
        if prompt.category == "canonical":
            return OracleResult.for_canonical(
                token_ids=token_ids,
                logits_per_step=logits,
                n_prompt_tokens=n_prompt_tokens,
            )
        else:
            topk = torch.topk(torch.from_numpy(logits), k=TOP_K_REGRESSION, dim=-1)
            return OracleResult.for_regression(
                token_ids=token_ids,
                top5_indices=topk.indices.numpy().astype(np.int64),
                top5_logits=topk.values.numpy().astype(np.float32),
                n_prompt_tokens=n_prompt_tokens,
            )

    def _generate_batch(self, prompt: PromptSpec) -> list[OracleResult]:
        import torch

        assert prompt.sub_prompts is not None  # is_batch guarantees this
        sub_prompts = prompt.sub_prompts
        max_tokens = (
            CANONICAL_MAX_TOKENS if prompt.category == "canonical" else REGRESSION_MAX_TOKENS
        )

        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            sub_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=not prompt.chat_template,
        )
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")
        n_prompt_tokens_per_seq = attention_mask.sum(dim=1).tolist()
        max_prompt_len = input_ids.shape[1]

        with (
            torch.inference_mode(),
            torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]),
        ):
            out = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_logits=True,
            )

        # Different members can finish before the last batch generation step.
        results: list[OracleResult] = []
        for b in range(len(sub_prompts)):
            token_ids_b, logits_b = _generated_token_logits(
                out, b, max_prompt_len, self.model.generation_config.eos_token_id
            )
            n_prompt_tokens_b = int(n_prompt_tokens_per_seq[b])
            if prompt.category == "canonical":
                results.append(
                    OracleResult.for_canonical(
                        token_ids=token_ids_b,
                        logits_per_step=logits_b,
                        n_prompt_tokens=n_prompt_tokens_b,
                    )
                )
            else:
                topk = torch.topk(torch.from_numpy(logits_b), k=TOP_K_REGRESSION, dim=-1)
                results.append(
                    OracleResult.for_regression(
                        token_ids=token_ids_b,
                        top5_indices=topk.indices.numpy().astype(np.int64),
                        top5_logits=topk.values.numpy().astype(np.float32),
                        n_prompt_tokens=n_prompt_tokens_b,
                    )
                )
        return results

    def close(self) -> None:
        import torch

        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
