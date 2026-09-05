from __future__ import annotations

import numpy as np
import pytest

from golden_gen.io import save_fixture
from golden_gen.replay import verify_oracle_replay


def _write_fixture(directory, token: int) -> None:
    directory.mkdir(exist_ok=True)
    save_fixture(
        directory / "canonical_01.transformers.safetensors",
        token_ids=np.array([token], dtype=np.int64),
        logits=np.array([[0.0, 1.0]], dtype=np.float32),
        top5_indices=None,
        top5_logits=None,
        n_prompt_tokens=1,
    )


def test_replay_verifier_requires_exact_tensor_keys_shapes_dtypes_and_values(tmp_path):
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    _write_fixture(primary, 1)
    _write_fixture(replay, 1)

    evidence = verify_oracle_replay(
        primary,
        replay,
        ("canonical_01.transformers.safetensors",),
    )

    assert evidence.fixture_count == 1
    assert evidence.verified_filenames == ("canonical_01.transformers.safetensors",)

    _write_fixture(replay, 0)
    with pytest.raises(ValueError, match="bit-identical"):
        verify_oracle_replay(
            primary,
            replay,
            ("canonical_01.transformers.safetensors",),
        )


def test_replay_verifier_rejects_missing_unexpected_or_partial_files(tmp_path):
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    _write_fixture(primary, 1)
    _write_fixture(replay, 1)
    (replay / "partial.tmp").write_text("partial")

    with pytest.raises(ValueError, match="replay file set"):
        verify_oracle_replay(
            primary,
            replay,
            ("canonical_01.transformers.safetensors",),
        )


def test_replay_rejects_different_signed_zero_bits(tmp_path):
    from safetensors.numpy import save_file

    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    primary.mkdir()
    replay.mkdir()
    save_file({"logits": np.array([0.0], dtype=np.float32)}, primary / "zero.safetensors")
    save_file({"logits": np.array([-0.0], dtype=np.float32)}, replay / "zero.safetensors")
    with pytest.raises(ValueError, match="bit-identical"):
        verify_oracle_replay(primary, replay, ("zero.safetensors",))
