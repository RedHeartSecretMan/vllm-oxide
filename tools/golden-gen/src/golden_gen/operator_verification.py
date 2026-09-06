"""Independent deterministic L0 rules, with no inherited model-logit tolerance."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from golden_gen.layered_release import OperatorProfile


def _bf16(value: NDArray[Any]) -> NDArray[np.float32]:
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(
        np.float32
    )


def _attention_inputs(n: int, qn: int) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    q = np.zeros((qn, 16, 128), dtype=np.float32)
    k = np.zeros((n, 8, 128), dtype=np.float32)
    q[:, :, 0] = 0.25 * (np.arange(qn)[:, None] + 1) * (np.arange(16)[None, :] % 3 + 1)
    k[:, :, 0] = 0.125 * (np.arange(n)[:, None] % 7 + 1) * (np.arange(8)[None, :] % 3 + 1)
    v = np.broadcast_to(
        np.arange(n)[:, None, None] % 11 / 8
        + np.arange(8)[None, :, None] / 16
        + np.arange(128)[None, None, :] % 13 / 64,
        (n, 8, 128),
    ).astype(np.float32)
    return _bf16(q), _bf16(k), _bf16(v)


def reference_rule(rule_id: str) -> dict[str, Any]:
    if rule_id == "materialized_halfway_sum_v1":
        return dict(
            operator="rmsnorm",
            input_dtype="bfloat16",
            input_shape=[1, 4],
            expected=np.ones((1, 4), dtype=np.float32),
            discrete=False,
        )
    if rule_id == "gate_2_up_0.515625_v1":
        return dict(
            operator="silu",
            input_dtype="bfloat16",
            input_shape=[1, 2],
            expected=np.array([[0.90625]], dtype=np.float32),
            discrete=False,
        )
    if rule_id == "nonconsecutive_positions_and_half_rotation_v1":
        values = np.broadcast_to(
            (np.arange(128)[None, None, :] % 7 - 3) / 4 + np.arange(2)[None, :, None] / 8,
            (3, 2, 128),
        ).astype(np.float32)
        frequencies = (
            np.asarray([0, 7, 255], dtype=np.float32)[:, None]
            / np.power(
                np.float32(1_000_000), np.arange(0, 128, 2, dtype=np.float32) / np.float32(128)
            )[None, :]
        )
        cosine, sine = (
            _bf16(np.cos(frequencies))[:, None, :],
            _bf16(np.sin(frequencies))[:, None, :],
        )
        left, right = values[..., :64], values[..., 64:]
        expected = np.concatenate(
            (
                _bf16(_bf16(left * cosine) - _bf16(right * sine)),
                _bf16(_bf16(right * cosine) + _bf16(left * sine)),
            ),
            axis=-1,
        )
        return dict(
            operator="rope",
            input_dtype="bfloat16",
            input_shape=[3, 2, 128],
            expected=expected,
            discrete=False,
        )
    if rule_id in ("asymmetric_qkv_causal_gqa_v1", "257_visible_tokens_noncontiguous_pages_v1"):
        prefill = rule_id == "asymmetric_qkv_causal_gqa_v1"
        n, qn = (5, 5) if prefill else (257, 1)
        q, k, v = _attention_inputs(n, qn)
        output = np.empty((qn, 16, 128), dtype=np.float64)
        scale = float(np.float32(1) / np.sqrt(np.float32(128)))
        for row in range(qn):
            visible = row + 1 if prefill else n
            for head in range(16):
                scores = np.asarray(
                    [
                        math.fsum(
                            (q[row, head].astype(np.float64) * key.astype(np.float64)).tolist()
                        )
                        * scale
                        for key in k[:visible, head // 2]
                    ]
                )
                weights = np.exp(scores - scores.max())
                weights /= math.fsum(weights.tolist())
                output[row, head] = [
                    math.fsum((weights * v[:visible, head // 2, dim]).tolist())
                    for dim in range(128)
                ]
        return dict(
            operator="attention",
            input_dtype="bfloat16",
            input_shape=[qn, 16, 128],
            expected=_bf16(output),
            discrete=False,
        )
    if rule_id == "noncontiguous_block_readback_v1":
        _, k, v = _attention_inputs(257, 1)
        return dict(
            operator="kv_cache",
            input_dtype="bfloat16",
            input_shape=[2, 256, 8, 128],
            expected=np.stack([k, v]),
            discrete=True,
        )
    if rule_id == "unique_and_multiway_max_with_filter_penalties_v1":
        return dict(
            operator="sampling",
            input_dtype="float32",
            input_shape=[3, 151936],
            expected=np.array([7, 12, 29], dtype=np.float32),
            discrete=True,
        )
    raise ValueError("operator input rule has no reviewed deterministic implementation")


def compare_operator(profile: OperatorProfile, payload: dict[str, Any]) -> dict[str, Any]:
    rule = reference_rule(profile.input_rule)
    expected = rule["expected"]
    if (
        payload.get("rule_id") != profile.input_rule
        or profile.operator != rule["operator"]
        or profile.dtype != rule["input_dtype"]
        or profile.shape != rule["input_shape"]
        or payload.get("input_shape") != profile.shape
        or payload.get("input_dtype") != profile.dtype
        or payload.get("output_shape") != list(expected.shape)
    ):
        raise ValueError("operator rule/dtype/shape binding mismatch")
    actual = np.asarray(payload["values"], dtype=np.float32)
    if actual.size != expected.size or not np.isfinite(actual).all():
        raise ValueError("invalid operator output values")
    actual = actual.reshape(expected.shape)
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bitwise = bool(np.array_equal(actual.view(np.uint32), expected.view(np.uint32)))
    return dict(
        profile_id=profile.profile_id,
        max_abs_error=float(error.max()),
        structure_passed=bitwise if rule["discrete"] else True,
        bitwise_equal=bitwise,
        output_shape=list(expected.shape),
        worst_coordinate=[int(x) for x in np.unravel_index(np.argmax(error), error.shape)],
        reference_algorithm="fp64-independent-rules-bf16-materialization-v1",
    )
