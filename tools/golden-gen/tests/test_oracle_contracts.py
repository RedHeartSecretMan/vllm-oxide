from types import SimpleNamespace

import numpy as np
import pytest

from golden_gen.config import VOCAB_SIZE
from golden_gen.generate import run_all
from golden_gen.oracles.base import OracleResult
from golden_gen.oracles.transformers_oracle import reference_model_kwargs
from golden_gen.oracles.vllm_oracle import _extract_full_logits, baseline_engine_kwargs
from golden_gen.schema import PromptSpec


class MalformedCanonicalOracle:
    name = "fake"

    def generate(self, prompt: PromptSpec) -> list[OracleResult]:
        return [
            OracleResult.for_canonical(
                token_ids=np.array([1, 2], dtype=np.int64),
                logits_per_step=np.zeros((1, VOCAB_SIZE), dtype=np.float32),
                n_prompt_tokens=1,
            )
        ]

    def close(self) -> None:
        pass


def test_generation_rejects_oracle_result_with_unmatched_tensor_rows(tmp_path):
    prompt = PromptSpec(
        id="canonical_01",
        category="canonical",
        prompt="hello",
        description="test",
    )

    with pytest.raises(ValueError, match="invalid canonical oracle result"):
        run_all([MalformedCanonicalOracle()], [prompt], tmp_path)

    assert list(tmp_path.glob("*.safetensors")) == []


def test_vllm_short_logprobs_fail_instead_of_zero_padding():
    completion = SimpleNamespace(logprobs=[{0: SimpleNamespace(logprob=1.0)}])

    with pytest.raises(RuntimeError, match="1 logprob rows for 2 generated tokens"):
        _extract_full_logits(completion, n=2, vocab_size=1)


def test_vllm_full_count_does_not_allow_negative_vocabulary_identity_alias():
    completion = SimpleNamespace(
        logprobs=[{0: SimpleNamespace(logprob=1.0), -1: SimpleNamespace(logprob=2.0)}]
    )
    with pytest.raises(RuntimeError, match="token ID"):
        _extract_full_logits(completion, n=1, vocab_size=2)


def test_oracle_construction_pins_tokenizer_dtype_eager_and_non_fallback_kernels():
    reference = reference_model_kwargs()
    baseline = baseline_engine_kwargs()

    assert reference["revision"] == reference["tokenizer_revision"]
    assert reference["attn_implementation"] == "sdpa"
    assert reference["torch_dtype"] == "bfloat16"
    assert baseline["revision"] == baseline["tokenizer_revision"]
    assert baseline["tensor_parallel_size"] == 1
    assert baseline["dtype"] == "bfloat16"
    assert baseline["seed"] == 0
    assert baseline["enforce_eager"] is True
    assert baseline["gpu_memory_utilization"] == 0.55
    assert baseline["attention_config"] == {
        "backend": "FLASH_ATTN",
        "flash_attn_version": 2,
    }
