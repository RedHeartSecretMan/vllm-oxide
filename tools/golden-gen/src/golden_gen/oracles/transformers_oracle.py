"""HF Transformers oracle adapter."""

from __future__ import annotations

import numpy as np

from golden_gen.config import (
    ATTN_IMPLEMENTATION,
    MODEL_ID,
    MODEL_REVISION,
    TOP_K_REGRESSION,
    max_tokens_for_category,
)
from golden_gen.oracles.base import OracleResult
from golden_gen.schema import PromptSpec


class TransformersOracle:
    """Oracle using HuggingFace Transformers.

    Uses AutoModelForCausalLM.generate() with output_logits=True to capture
    full pre-sampling logits for canonical prompts, and top-5 for regression.
    """

    name = "transformers"

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            )
            .to("cuda")
            .eval()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, prompt: PromptSpec) -> OracleResult:
        import torch

        max_tokens = max_tokens_for_category(prompt.category)

        inputs = self.tokenizer(
            prompt.prompt,
            return_tensors="pt",
            add_special_tokens=not prompt.chat_template,
        )
        input_ids = inputs["input_ids"].to("cuda")
        n_prompt_tokens = input_ids.shape[1]

        with torch.no_grad():
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

        # out.logits is tuple of length n_generated, each [1, vocab_size] FP32
        token_ids = out.sequences[0, n_prompt_tokens:].cpu().numpy().astype(np.int64)

        if prompt.category == "canonical":
            logits_list = [logit[0].cpu().numpy().astype(np.float32) for logit in out.logits]
            logits = np.stack(logits_list, axis=0)  # [n, vocab_size]
            return OracleResult.for_canonical(
                token_ids=token_ids,
                logits_per_step=logits,
                n_prompt_tokens=n_prompt_tokens,
            )
        else:
            logits_list = [logit[0].cpu().numpy().astype(np.float32) for logit in out.logits]
            logits = np.stack(logits_list, axis=0)
            topk = torch.topk(torch.from_numpy(logits), k=TOP_K_REGRESSION, dim=-1)
            return OracleResult.for_regression(
                token_ids=token_ids,
                top5_indices=topk.indices.numpy().astype(np.int64),
                top5_logits=topk.values.numpy().astype(np.float32),
                n_prompt_tokens=n_prompt_tokens,
            )

    def close(self) -> None:
        import torch

        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
