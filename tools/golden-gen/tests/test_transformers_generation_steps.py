import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from golden_gen.oracles.transformers_oracle import TransformersOracle, _generated_token_logits
from golden_gen.schema import PromptSpec


class Tensor:
    """CPU-only stand-in for the tiny tensor surface used by generation adapters."""

    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape = self.data.shape

    def __getitem__(self, key):
        return Tensor(self.data[key])

    def to(self, device):
        assert device == "cuda"
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def sum(self, dim):
        return Tensor(self.data.sum(axis=dim))

    def tolist(self):
        return self.data.tolist()


def output(tokens, steps):
    sequences = Tensor([[41, 42, *row] for row in tokens])
    logits = tuple(
        Tensor(np.arange(len(tokens) * 8).reshape(len(tokens), 8) + i * 100) for i in range(steps)
    )
    return SimpleNamespace(sequences=sequences, logits=logits)


@pytest.mark.parametrize("eos", [9, [9, 10]])
def test_generated_steps_keep_eos_and_drop_padding(eos):
    result = output([[5, 9, 0, 0], [6, 7, 8, 9]], 4)
    tokens, logits = _generated_token_logits(result, 0, 2, eos)
    np.testing.assert_array_equal(tokens, [5, 9])
    assert logits.shape == (2, 8)
    assert logits.dtype == np.float32
    other, other_logits = _generated_token_logits(result, 1, 2, eos)
    np.testing.assert_array_equal(other, [6, 7, 8, 9])
    assert other_logits.shape == (4, 8)


def test_pad_equal_to_eos_keeps_only_first_eos():
    result = output([[9, 9, 9]], 3)
    tokens, logits = _generated_token_logits(result, 0, 2, [9, 10])
    np.testing.assert_array_equal(tokens, [9])
    assert logits.shape == (1, 8)


def test_no_eos_keeps_actual_short_generation_count():
    result = output([[5, 6]], 2)
    tokens, logits = _generated_token_logits(result, 0, 2, None)
    np.testing.assert_array_equal(tokens, [5, 6])
    assert logits.shape == (2, 8)


@pytest.mark.parametrize("logits", [(), None])
def test_empty_generation_is_explicit_error(logits):
    result = SimpleNamespace(sequences=Tensor([[41, 42]]), logits=logits)
    with pytest.raises(ValueError, match="no generation logits"):
        _generated_token_logits(result, 0, 2, 9)


def test_token_logit_step_mismatch_is_rejected():
    with pytest.raises(ValueError, match="different step counts"):
        _generated_token_logits(output([[5, 9]], 1), 0, 2, 9)


class Tokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, prompt, **kwargs):
        rows = len(prompt) if isinstance(prompt, list) else 1
        return {"input_ids": Tensor([[41, 42]] * rows), "attention_mask": Tensor([[1, 1]] * rows)}


def stub_torch(monkeypatch):
    def topk(values, k, dim):
        indices = np.argsort(-values.data, axis=dim)[..., :k]
        selected = np.take_along_axis(values.data, indices, axis=dim)
        return SimpleNamespace(indices=Tensor(indices), values=Tensor(selected))

    torch = SimpleNamespace(
        no_grad=nullcontext,
        inference_mode=nullcontext,
        from_numpy=Tensor,
        topk=topk,
        nn=SimpleNamespace(
            attention=SimpleNamespace(
                sdpa_kernel=lambda _: nullcontext(), SDPBackend=SimpleNamespace(MATH="math")
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


def adapter(monkeypatch, generated):
    stub_torch(monkeypatch)
    oracle = TransformersOracle.__new__(TransformersOracle)
    oracle.tokenizer = Tokenizer()
    oracle.model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[9, 10]),
        generate=lambda *args, **kwargs: generated,
    )
    return oracle


def test_batch_adapter_uses_actual_steps_and_trims_each_member(monkeypatch):
    oracle = adapter(monkeypatch, output([[9, 0, 0], [5, 6, 10]], 3))
    prompt = PromptSpec(
        id="batch", category="canonical", prompt="batch", description="test", sub_prompts=["A", "B"]
    )
    results = oracle.generate(prompt)
    assert [r.token_ids.tolist() for r in results] == [[9], [5, 6, 10]]
    assert [r.logits_per_step.shape for r in results] == [(1, 8), (3, 8)]
    assert [r.n_prompt_tokens for r in results] == [2, 2]


@pytest.mark.parametrize("category", ["canonical", "regression"])
def test_single_adapter_uses_real_eos_steps_for_both_fixture_shapes(monkeypatch, category):
    oracle = adapter(monkeypatch, output([[5, 9]], 2))
    prompt = PromptSpec(id="case", category=category, prompt="A", description="test")
    result = oracle.generate(prompt)[0]
    assert result.token_ids.tolist() == [5, 9]
    if category == "canonical":
        assert result.logits_per_step.shape == (2, 8)
    else:
        assert result.top5_indices.shape == result.top5_logits.shape == (2, 5)
