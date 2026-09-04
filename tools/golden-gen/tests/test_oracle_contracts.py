from types import SimpleNamespace

import numpy as np
import pytest

from golden_gen.config import VOCAB_SIZE
from golden_gen.generate import run_all
from golden_gen.oracles.base import OracleResult
from golden_gen.oracles.vllm_oracle import _extract_full_logits
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
