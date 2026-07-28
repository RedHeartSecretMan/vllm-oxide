import numpy as np
import pytest

from golden_gen.config import VOCAB_SIZE
from golden_gen.oracles.fake import FakeOracle
from golden_gen.schema import PromptSpec


class TestFakeOracle:
    def setup_method(self):
        self.oracle = FakeOracle()

    def test_name(self):
        assert self.oracle.name == "fake"

    def test_generate_canonical_returns_full_logits(self):
        prompt = PromptSpec(
            id="canonical_01",
            category="canonical",
            prompt="Hello",
            description="test",
        )
        results = self.oracle.generate(prompt)
        assert len(results) == 1
        result = results[0]
        assert len(result.token_ids) > 0
        assert result.token_ids.dtype == np.int64
        assert result.logits_per_step.shape[1] == VOCAB_SIZE
        assert result.logits_per_step.dtype == np.float32
        assert result.top5_indices.shape == (0, 5)
        assert result.top5_logits.shape == (0, 5)
        assert result.n_prompt_tokens > 0

    def test_generate_regression_returns_top5(self):
        prompt = PromptSpec(
            id="regression_01",
            category="regression",
            prompt="Write Python",
            description="test",
        )
        results = self.oracle.generate(prompt)
        assert len(results) == 1
        result = results[0]
        assert len(result.token_ids) > 0
        assert result.logits_per_step.shape == (0, 0)
        assert result.top5_indices.shape[1] == 5
        assert result.top5_logits.shape[1] == 5
        assert result.n_prompt_tokens > 0

    def test_deterministic_same_instance(self):
        """Same prompt + same oracle name -> identical output."""
        prompt = PromptSpec(
            id="canonical_01",
            category="canonical",
            prompt="Hello",
            description="test",
        )
        r1 = self.oracle.generate(prompt)[0]
        r2 = self.oracle.generate(prompt)[0]
        np.testing.assert_array_equal(r1.token_ids, r2.token_ids)
        np.testing.assert_array_equal(r1.logits_per_step, r2.logits_per_step)

    def test_deterministic_different_instances_same_name(self):
        """Two FakeOracle instances with the same name produce identical output."""
        prompt = PromptSpec(
            id="canonical_01",
            category="canonical",
            prompt="Hello",
            description="test",
        )
        o1 = FakeOracle()
        o2 = FakeOracle()
        r1 = o1.generate(prompt)[0]
        r2 = o2.generate(prompt)[0]
        np.testing.assert_array_equal(r1.token_ids, r2.token_ids)

    def test_different_names_different_output(self):
        """Different oracle names produce DIFFERENT outputs."""
        prompt = PromptSpec(
            id="canonical_01",
            category="canonical",
            prompt="Hello",
            description="test",
        )
        o_t = FakeOracle()
        o_v = FakeOracle()
        o_t.name = "transformers"
        o_v.name = "vllm"
        r_t = o_t.generate(prompt)[0]
        r_v = o_v.generate(prompt)[0]

        # Different names -> different logits
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(r_t.logits_per_step, r_v.logits_per_step)

        # L2 distance should be non-zero (different oracle names produce different outputs)
        l2_tv = float(np.linalg.norm(r_t.logits_per_step - r_v.logits_per_step))
        assert l2_tv > 0, "Expected non-zero L2 for different oracle names"

    def test_different_prompts_different_output(self):
        p1 = PromptSpec(
            id="canonical_01",
            category="canonical",
            prompt="Hello",
            description="test",
        )
        p2 = PromptSpec(
            id="canonical_02",
            category="canonical",
            prompt="World",
            description="test",
        )
        r1 = self.oracle.generate(p1)[0]
        r2 = self.oracle.generate(p2)[0]
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(r1.token_ids, r2.token_ids)

    def test_regression_max_tokens_limited(self):
        prompt = PromptSpec(
            id="regression_01",
            category="regression",
            prompt="short",
            description="test",
        )
        result = self.oracle.generate(prompt)[0]
        assert len(result.token_ids) <= 32

    def test_close_does_not_raise(self):
        self.oracle.close()  # should not raise

    def test_different_oracle_names_produce_different_logits(self):
        """Verify two differently-named FakeOracles produce non-zero L2."""
        prompt = PromptSpec(
            id="canonical_01",
            category="canonical",
            prompt="Hello",
            description="test",
        )
        o1 = FakeOracle()
        o2 = FakeOracle()
        o1.name = "transformers"
        o2.name = "vllm"
        r1 = o1.generate(prompt)[0]
        r2 = o2.generate(prompt)[0]
        per_step_l2 = np.linalg.norm(r1.logits_per_step - r2.logits_per_step, axis=1)
        assert per_step_l2.max() > 0, "Expected non-zero L2 for different oracle names"
        assert per_step_l2.max() < 10.0, f"L2 too large: {per_step_l2.max()}"

    def test_batch_returns_n_results(self):
        batch_prompt = PromptSpec(
            id="canonical_05",
            category="canonical",
            prompt="batch test",
            description="test",
            sub_prompts=["A", "B", "C", "D"],
        )
        results = self.oracle.generate(batch_prompt)
        assert len(results) == 4
        for r in results:
            assert len(r.token_ids) > 0
            assert r.logits_per_step.shape[1] == VOCAB_SIZE
            assert r.n_prompt_tokens > 0

    def test_batch_results_differ(self):
        batch_prompt = PromptSpec(
            id="canonical_05",
            category="canonical",
            prompt="batch test",
            description="test",
            sub_prompts=["A", "B", "C", "D"],
        )
        results = self.oracle.generate(batch_prompt)
        for i in range(1, len(results)):
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(results[0].token_ids, results[i].token_ids)
