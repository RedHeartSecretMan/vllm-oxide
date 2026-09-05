"""Independent CPU checks for the nonaccepting localization evidence seam."""

import numpy as np
import pytest

from golden_gen.layer_diagnostic import decode_trace, require_prefix_equivalence


def test_localization_rejects_one_changed_bit_before_interpreting_layers() -> None:
    old = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    new = old.copy()
    new[0, 0] = -0.0
    tokens = np.array([151667, 198], dtype=np.int64)
    with pytest.raises(ValueError, match="instrumentation changed"):
        require_prefix_equivalence(new, tokens, old, tokens)


def test_localization_accepts_exactly_two_identical_rows_only() -> None:
    old = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    tokens = np.array([151667, 198, 3], dtype=np.int64)
    require_prefix_equivalence(old[:2], tokens[:2], old, tokens)
    with pytest.raises(ValueError):
        require_prefix_equivalence(old, tokens, old, tokens)


def test_trace_rejects_missing_checkpoint_before_comparing() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        decode_trace([])


def test_trace_decodes_literal_bf16_and_rejects_order_dtype_and_shape() -> None:
    from copy import deepcopy

    names = ["embedding", *(f"layer_{i}" for i in range(28)), "final_norm"]
    rows = [
        dict(
            kind="header",
            diagnostic_only=True,
            accepting=False,
            prompt_id="canonical_03",
            step=0,
            token_ids=[17],
            positions=[0],
        ),
        *(
            dict(
                kind="checkpoint",
                name=name,
                dtype="BF16",
                shape=[1, 1024],
                bf16_bits=[16256] * 1024,
            )
            for name in names
        ),
        dict(kind="trailer", complete=True, checkpoints=30),
    ]
    assert np.array_equal(decode_trace(rows)["layer_27"], np.ones((1, 1024), dtype=np.float32))
    for field, value in (("name", "layer_1"), ("dtype", "F32"), ("shape", [1024])):
        invalid = deepcopy(rows)
        invalid[1][field] = value
        with pytest.raises(ValueError):
            decode_trace(invalid)
