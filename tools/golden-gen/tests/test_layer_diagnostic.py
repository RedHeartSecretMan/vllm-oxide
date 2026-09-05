"""Independent CPU checks for the nonaccepting localization evidence seam."""

import numpy as np
import pytest

from golden_gen.layer_diagnostic import (
    attention_gate,
    decode_trace,
    layer0_cause,
    require_prefix_equivalence,
)


def test_localization_rejects_one_changed_bit_before_interpreting_layers() -> None:
    old = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    new = old.copy()
    new[0, 0] = -0.0
    tokens = np.array([151667, 198], dtype=np.int64)
    with pytest.raises(ValueError, match="instrumentation changed"):
        require_prefix_equivalence(new, tokens, old, tokens)


def test_rmsnorm_difference_prevents_qkv_access_and_attribution() -> None:
    embedding = np.ones((1, 2), dtype=np.float32)
    reference = {
        "embedding": embedding,
        "layer0_input_norm": np.array([[0.0, 1.0]], dtype=np.float32),
    }
    candidate = {
        "embedding": embedding,
        "layer0_input_norm": np.array([[-0.0, 1.0]], dtype=np.float32),
    }
    assert layer0_cause(reference, candidate) == {
        "first_path": "input_rmsnorm",
        "qkv_compared": False,
    }


def test_qk_norm_difference_prevents_rope_and_attention_attribution() -> None:
    names = [
        "embedding",
        "layer0_input_norm",
        "layer0_q",
        "layer0_k",
        "layer0_q_norm",
        "layer0_k_norm",
    ]
    reference = {name: np.ones((1, 2), dtype=np.float32) for name in names}
    candidate = {name: value.copy() for name, value in reference.items()}
    candidate["layer0_q_norm"][0, 0] = 2.0
    result = attention_gate(reference, candidate, {}, {}, True)
    assert result["allowed"] == ["layer0_q_norm", "layer0_k_norm"]
    assert result["blocked_after"] == "qk_norm"


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

    names = [
        "embedding",
        "layer0_input_norm",
        "layer0_q",
        "layer0_k",
        "layer0_v",
        "layer0_q_norm",
        "layer0_k_norm",
        "layer0_q_rope",
        "layer0_k_rope",
        "layer0_attention_context",
        *(f"layer_{i}" for i in range(28)),
        "final_norm",
    ]
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
                shape=[
                    1,
                    2048
                    if name
                    in ("layer0_q", "layer0_q_norm", "layer0_q_rope", "layer0_attention_context")
                    else 1024,
                ],
                bf16_bits=[16256]
                * (
                    2048
                    if name
                    in ("layer0_q", "layer0_q_norm", "layer0_q_rope", "layer0_attention_context")
                    else 1024
                ),
            )
            for name in names
        ),
        dict(kind="trailer", complete=True, checkpoints=39),
    ]
    assert np.array_equal(decode_trace(rows)["layer_27"], np.ones((1, 1024), dtype=np.float32))
    for field, value in (("name", "layer_1"), ("dtype", "F32"), ("shape", [1024])):
        invalid = deepcopy(rows)
        invalid[1][field] = value
        with pytest.raises(ValueError):
            decode_trace(invalid)


def test_qkv_cause_requires_equal_embedding_and_norm_inputs() -> None:
    names = ["embedding", "layer0_input_norm", "layer0_q", "layer0_k", "layer0_v"]
    reference = {name: np.ones((1, 2), dtype=np.float32) for name in names}
    candidate = {name: value.copy() for name, value in reference.items()}
    assert layer0_cause(reference, candidate)["first_path"] == "after_qkv"
    candidate["layer0_k"][0, 0] = 2.0
    assert layer0_cause(reference, candidate) == {
        "first_path": "qkv_projection",
        "qkv_compared": True,
        "different_projections": ["layer0_k"],
    }
    candidate["embedding"][0, 0] = 2.0
    with pytest.raises(ValueError, match="identical embedding"):
        layer0_cause(reference, candidate)


def test_actual_scale_precision_and_mask_difference_block_context() -> None:
    names = [
        "embedding",
        "layer0_input_norm",
        "layer0_q",
        "layer0_k",
        "layer0_v",
        "layer0_q_norm",
        "layer0_k_norm",
        "layer0_q_rope",
        "layer0_k_rope",
    ]
    values = {name: np.ones((1, 2), dtype=np.float32) for name in names}
    metadata = dict(
        q_length=1,
        k_length=1,
        q_heads=16,
        kv_heads=8,
        head_dim=128,
        head_mapping=[head // 2 for head in range(16)],
        scale_argument=0.08838834764831845,
        dropout_p=0.0,
        mask=dict(kind="visibility", shape=[1, 1], allowed=[True]),
    )
    other = {**metadata, "scale_argument": float(np.float32(metadata["scale_argument"]))}
    assert np.float32(metadata["scale_argument"]) == np.float32(other["scale_argument"])
    result = attention_gate(values, values, metadata, other, True)
    assert result["blocked_after"] == "attention_metadata"
    assert "scale_argument" in result["metadata_differences"]
    assert "layer0_attention_context" not in result["allowed"]
    other = {**metadata, "mask": dict(kind="visibility", shape=[1, 1], allowed=[False])}
    assert "mask" in attention_gate(values, values, metadata, other, True)["metadata_differences"]
    other = {**metadata, "head_mapping": list(range(16))}
    assert (
        "head_mapping"
        in attention_gate(values, values, metadata, other, True)["metadata_differences"]
    )
    assert (
        "layer0_attention_context"
        in attention_gate(values, values, metadata, metadata, True)["allowed"]
    )
    assert (
        attention_gate(values, values, metadata, metadata, False)["blocked_after"]
        == "attention_tensor_inputs"
    )
