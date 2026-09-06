"""Isolated mathematical fault models, not modified production/GPU measurements."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from golden_gen.layered_accuracy import Budgets, compare_case
from golden_gen.layered_release import OperatorProfile
from golden_gen.operator_verification import (
    _attention_inputs,
    _bf16,
    compare_operator,
    reference_rule,
)


def fault_payload(profile: OperatorProfile, fault: str) -> dict[str, Any]:
    rule = reference_rule(profile.input_rule)
    values = None
    if profile.input_rule == "noncontiguous_block_readback_v1":
        if fault == "slot_swap":
            values = rule["expected"][[1, 0]].copy()
        elif fault == "stale_owner":
            values = np.zeros_like(rule["expected"])
    if profile.input_rule == "unique_and_multiway_max_with_filter_penalties_v1" and fault in (
        "wrong_token_id",
        "wrong_tie",
        "ignored_penalty",
    ):
        scores = [{7: 4.0, 12: 4.0}, {7: 4.0, 12: 3.0}, {29: 4.0, 30: 4.0, 31: 4.0}]
        if fault != "ignored_penalty":
            scores[1][7] -= 2
        selected = [
            max(row, key=lambda token: (row[token], token if fault == "wrong_tie" else -token))
            for row in scores
        ]
        # Confusing a sorted rank with vocabulary identity returns rank zero.
        values = np.array([0] * 3 if fault == "wrong_token_id" else selected, dtype=np.float32)
    if profile.input_rule == "materialized_halfway_sum_v1":
        if fault == "missing_bf16_residual_round":
            residual_sum = np.array([[1 + 1 / 256, 1 + 1 / 256, 1 + 1 / 256, 1]], dtype=np.float64)
            variance = math.fsum((residual_sum.flatten() ** 2).tolist()) / 4
            values = _bf16(residual_sum / math.sqrt(variance + 1e-6))
        elif fault == "wrong_epsilon":
            # Named replacement: epsilon=0.1 instead of the declared 1e-6.
            values = _bf16(np.ones((1, 4), dtype=np.float64) / math.sqrt(1 + 0.1))
    if profile.input_rule == "gate_2_up_0.515625_v1":
        gate = np.array([[2]], dtype=np.float32)
        up = np.float32(0.515625)
        if fault == "missing_intermediate_round":
            values = _bf16(gate / (1 + np.exp(-gate)) * up)
        elif fault == "bf16_opmath":
            values = _bf16(_bf16(gate / _bf16(1 + _bf16(np.exp(-gate)))) * up)
    if profile.input_rule == "nonconsecutive_positions_and_half_rotation_v1" and fault in (
        "position_shift",
        "wrong_half_rotation",
    ):
        inputs = np.broadcast_to(
            (np.arange(128)[None, None, :] % 7 - 3) / 4 + np.arange(2)[None, :, None] / 8,
            (3, 2, 128),
        ).astype(np.float32)
        positions = np.array([0, 7, 255], dtype=np.float32) + (
            1 if fault == "position_shift" else 0
        )
        angles = (
            positions[:, None]
            / np.power(
                np.float32(1_000_000), np.arange(0, 128, 2, dtype=np.float32) / np.float32(128)
            )[None, :]
        )
        cosine, sine = _bf16(np.cos(angles))[:, None, :], _bf16(np.sin(angles))[:, None, :]
        if fault == "wrong_half_rotation":
            sine = -sine
        left, right = inputs[..., :64], inputs[..., 64:]
        values = np.concatenate(
            (
                _bf16(_bf16(left * cosine) - _bf16(right * sine)),
                _bf16(_bf16(right * cosine) + _bf16(left * sine)),
            ),
            axis=-1,
        )
    if profile.operator == "attention":
        prefill = profile.input_rule == "asymmetric_qkv_causal_gqa_v1"
        allowed = (
            ("future_mask", "head_mapping", "scale")
            if prefill
            else ("missing_history", "wrong_slot")
        )
        if fault not in allowed:
            raise ValueError("unknown attention fault")
        n, qn = (5, 5) if prefill else (257, 1)
        q, k, v = _attention_inputs(n, qn)
        if fault == "wrong_slot":
            # Wrong plane address: read K storage as the V plane.
            v = k.copy()
        output = np.empty((qn, 16, 128), dtype=np.float64)
        scale = 1.0 if fault == "scale" else float(np.float32(1) / np.sqrt(np.float32(128)))
        for row in range(qn):
            visible = n if not prefill or fault == "future_mask" else row + 1
            start = visible - 1 if fault == "missing_history" else 0
            for head in range(16):
                kvhead = head % 8 if fault == "head_mapping" else head // 2
                attention_scores = np.array(
                    [
                        math.fsum(
                            (q[row, head].astype(np.float64) * key.astype(np.float64)).tolist()
                        )
                        * scale
                        for key in k[start:visible, kvhead]
                    ]
                )
                weights = np.exp(attention_scores - attention_scores.max())
                weights /= math.fsum(weights.tolist())
                output[row, head] = [
                    math.fsum((weights * v[start:visible, kvhead, dim]).tolist())
                    for dim in range(128)
                ]
        values = _bf16(output)
    if values is None:
        raise ValueError("operator fault model is not implemented for this frozen rule")
    return dict(
        mode="isolated_cpu_fault_model",
        fault_id=fault,
        rule_id=profile.input_rule,
        input_shape=rule["input_shape"],
        input_dtype=rule["input_dtype"],
        output_shape=list(values.shape),
        values=values.flatten().tolist(),
    )


def make_fault_models(profiles: list[OperatorProfile]) -> dict[str, Any]:
    return dict(
        protocol="layered-accuracy-v1",
        schema_version=1,
        mode="isolated_cpu_fault_models",
        operator_faults=[
            dict(profile_id=profile.profile_id, **fault_payload(profile, fault))
            for profile in profiles
            for fault in profile.required_faults
        ],
    )


def evaluate_fault_models(
    profiles: list[OperatorProfile], raw: dict[str, Any], limits: dict[str, float | None]
) -> dict[str, dict[str, bool | None]]:
    if (raw.get("protocol"), raw.get("schema_version"), raw.get("mode")) != (
        "layered-accuracy-v1",
        1,
        "isolated_cpu_fault_models",
    ):
        raise ValueError("fault evidence mode/version mismatch")
    expected = {(p.profile_id, f) for p in profiles for f in p.required_faults}
    records = raw.get("operator_faults", [])
    if (
        len(records) != len(expected)
        or {(r.get("profile_id"), r.get("fault_id")) for r in records} != expected
    ):
        raise ValueError("fault model inventory mismatch")
    result: dict[str, dict[str, bool | None]] = {}
    for profile in profiles:
        result[profile.profile_id] = {}
        for fault in profile.required_faults:
            payload = next(
                r
                for r in records
                if (r["profile_id"], r["fault_id"]) == (profile.profile_id, fault)
            )
            regenerated = dict(profile_id=profile.profile_id, **fault_payload(profile, fault))
            if payload != regenerated:
                raise ValueError("isolated fault payload differs from its frozen mathematical rule")
            measurement = compare_operator(profile, payload)
            limit = limits.get(profile.profile_id)
            if limit is not None and (not math.isfinite(limit) or limit < 0):
                raise ValueError("invalid fault detection budget")
            result[profile.profile_id][fault] = (
                None
                if limit is None
                else not measurement["structure_passed"] or measurement["max_abs_error"] > limit
            )
    return result


def evaluate_distribution_fault(
    budgets: Budgets | None, definition: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Frozen two-token wrong-distribution model; values are stimuli, never budgets."""
    if definition is None:
        definition = dict(
            vocab_size=2,
            steps=1,
            token_ids=[0, 1],
            reference_logits=[[10.0, 0.0]],
            candidate_logits=[[0.0, 10.0]],
            baseline_logits=[[10.0, 0.0]],
            input_dtype="float32",
            predicted_token_ids=[1],
            analysis_temperature=1,
            raw_greedy_temperature=0,
        )
    reference, wrong, baseline = (
        np.asarray(definition[key], dtype=np.float32)
        for key in ("reference_logits", "candidate_logits", "baseline_logits")
    )
    if (
        (
            definition.get("vocab_size"),
            definition.get("steps"),
            definition.get("token_ids"),
            definition.get("input_dtype"),
            definition.get("analysis_temperature"),
            definition.get("raw_greedy_temperature"),
        )
        != (2, 1, [0, 1], "float32", 1, 0)
        or reference.shape != (1, 2)
        or not np.array_equal(wrong, reference[:, ::-1])
        or not np.array_equal(baseline, reference)
        or not reference[0, 0] > reference[0, 1]
        or definition.get("predicted_token_ids") != [1]
    ):
        raise ValueError("wrong-distribution fault does not match its declared column swap")
    result = compare_case(reference, wrong, baseline, definition["predicted_token_ids"], budgets)
    result.update(mode="isolated_cpu_fault_model", fault_id="obvious_distribution_swap_v1")
    return result
