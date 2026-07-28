import numpy as np

from golden_gen.config import VOCAB_SIZE
from golden_gen.io import load_fixture, save_fixture


class TestSaveLoadFixture:
    def test_save_load_canonical(self, tmp_output_dir, fake_logits, fake_token_ids):
        path = tmp_output_dir / "canonical_01.transformers.safetensors"
        sha256 = save_fixture(
            path=path,
            token_ids=fake_token_ids,
            logits=fake_logits,
            top5_indices=None,
            top5_logits=None,
            n_prompt_tokens=10,
        )
        assert len(sha256) == 64  # SHA-256 hex
        assert path.exists()

        loaded = load_fixture(path)
        assert "token_ids" in loaded
        assert "logits" in loaded
        assert "n_prompt_tokens" in loaded
        assert "top5_indices" not in loaded
        assert "top5_logits" not in loaded

        np.testing.assert_array_equal(loaded["token_ids"], fake_token_ids)
        np.testing.assert_array_equal(loaded["logits"], fake_logits)
        assert loaded["n_prompt_tokens"].item() == 10

    def test_save_load_regression(self, tmp_output_dir):
        n_tokens = 8
        token_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        top5_indices = np.array([[i] * 5 for i in range(n_tokens)], dtype=np.int64)
        top5_logits = np.ones((n_tokens, 5), dtype=np.float32)

        path = tmp_output_dir / "regression_01.vllm.safetensors"
        sha256 = save_fixture(
            path=path,
            token_ids=token_ids,
            logits=None,
            top5_indices=top5_indices,
            top5_logits=top5_logits,
            n_prompt_tokens=20,
        )
        assert len(sha256) == 64

        loaded = load_fixture(path)
        assert "logits" not in loaded
        assert loaded["top5_indices"].shape == (n_tokens, 5)
        assert loaded["n_prompt_tokens"].item() == 20

    def test_sha256_deterministic(self, tmp_output_dir, fake_logits, fake_token_ids):
        path1 = tmp_output_dir / "test1.safetensors"
        path2 = tmp_output_dir / "test2.safetensors"

        sha1 = save_fixture(
            path=path1,
            token_ids=fake_token_ids,
            logits=fake_logits,
            top5_indices=None,
            top5_logits=None,
            n_prompt_tokens=10,
        )
        sha2 = save_fixture(
            path=path2,
            token_ids=fake_token_ids,
            logits=fake_logits,
            top5_indices=None,
            top5_logits=None,
            n_prompt_tokens=10,
        )
        assert sha1 == sha2

    def test_different_data_different_sha(self, tmp_output_dir, fake_token_ids):
        logits_a = np.zeros((8, VOCAB_SIZE), dtype=np.float32)
        logits_b = np.ones((8, VOCAB_SIZE), dtype=np.float32)

        sha_a = save_fixture(
            path=tmp_output_dir / "a.safetensors",
            token_ids=fake_token_ids,
            logits=logits_a,
            top5_indices=None,
            top5_logits=None,
            n_prompt_tokens=5,
        )
        sha_b = save_fixture(
            path=tmp_output_dir / "b.safetensors",
            token_ids=fake_token_ids,
            logits=logits_b,
            top5_indices=None,
            top5_logits=None,
            n_prompt_tokens=5,
        )
        assert sha_a != sha_b

    def test_type_enforcement(self, tmp_output_dir, fake_token_ids):
        logits = np.zeros((8, VOCAB_SIZE), dtype=np.float64)
        sha256 = save_fixture(
            path=tmp_output_dir / "float64_logits.safetensors",
            token_ids=fake_token_ids,
            logits=logits,
            top5_indices=None,
            top5_logits=None,
            n_prompt_tokens=5,
        )
        assert len(sha256) == 64
        loaded = load_fixture(tmp_output_dir / "float64_logits.safetensors")
        assert "logits" in loaded
        assert loaded["logits"].dtype == np.float32
