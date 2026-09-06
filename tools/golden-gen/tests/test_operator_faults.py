def test_isolated_silu_fault_models_expose_each_materialization_error() -> None:
    from golden_gen.layered_release import OperatorProfile
    from golden_gen.operator_faults import fault_payload
    from golden_gen.operator_verification import compare_operator

    profile = OperatorProfile(
        profile_id="silu",
        operator="silu",
        dtype="bfloat16",
        shape=[1, 2],
        input_rule="gate_2_up_0.515625_v1",
        required_faults=["bf16_opmath", "missing_intermediate_round"],
    )
    for fault in profile.required_faults:
        payload = fault_payload(profile, fault)
        assert payload["values"] == [0.91015625]
        assert compare_operator(profile, payload)["max_abs_error"] == 0.00390625
        assert payload["mode"] == "isolated_cpu_fault_model"


def test_rms_fault_models_keep_residual_rounding_and_epsilon_distinct() -> None:
    from golden_gen.layered_release import OperatorProfile
    from golden_gen.operator_faults import fault_payload

    profile = OperatorProfile(
        profile_id="rms",
        operator="rmsnorm",
        dtype="bfloat16",
        shape=[1, 4],
        input_rule="materialized_halfway_sum_v1",
        required_faults=["missing_bf16_residual_round", "wrong_epsilon"],
    )
    assert fault_payload(profile, "missing_bf16_residual_round")["values"] == [1, 1, 1, 0.99609375]
    assert fault_payload(profile, "wrong_epsilon")["values"] == [0.953125] * 4


def test_rope_fault_models_detect_position_and_rotation_sign_errors() -> None:
    from golden_gen.layered_release import OperatorProfile
    from golden_gen.operator_faults import fault_payload
    from golden_gen.operator_verification import compare_operator

    profile = OperatorProfile(
        profile_id="rope",
        operator="rope",
        dtype="bfloat16",
        shape=[3, 2, 128],
        input_rule="nonconsecutive_positions_and_half_rotation_v1",
        required_faults=["position_shift", "wrong_half_rotation"],
    )
    shifted = fault_payload(profile, "position_shift")
    flipped = fault_payload(profile, "wrong_half_rotation")
    assert shifted["values"][0] == 0.015625
    assert flipped["values"][0] == -0.75  # Position zero cannot expose a sign error alone.
    assert compare_operator(profile, flipped)["max_abs_error"] > 0.5


def test_attention_faults_exercise_mask_heads_scale_and_real_history() -> None:
    from golden_gen.layered_release import OperatorProfile
    from golden_gen.operator_faults import fault_payload
    from golden_gen.operator_verification import compare_operator

    prefill = OperatorProfile(
        profile_id="prefill",
        operator="attention",
        dtype="bfloat16",
        shape=[5, 16, 128],
        input_rule="asymmetric_qkv_causal_gqa_v1",
        required_faults=["future_mask", "head_mapping", "scale"],
    )
    assert fault_payload(prefill, "head_mapping")["values"][128] == 0.0625
    for fault in prefill.required_faults:
        assert compare_operator(prefill, fault_payload(prefill, fault))["max_abs_error"] > 0
    paged = OperatorProfile(
        profile_id="paged",
        operator="attention",
        dtype="bfloat16",
        shape=[1, 16, 128],
        input_rule="257_visible_tokens_noncontiguous_pages_v1",
        required_faults=["missing_history", "wrong_slot"],
    )
    assert fault_payload(paged, "missing_history")["values"][0] == 0.375
    assert compare_operator(paged, fault_payload(paged, "wrong_slot"))["max_abs_error"] > 0.5


def test_kv_fault_models_are_discrete_plane_and_owner_failures() -> None:
    from golden_gen.layered_release import OperatorProfile
    from golden_gen.operator_faults import fault_payload
    from golden_gen.operator_verification import compare_operator

    profile = OperatorProfile(
        profile_id="kv",
        operator="kv_cache",
        dtype="bfloat16",
        shape=[2, 256, 8, 128],
        input_rule="noncontiguous_block_readback_v1",
        required_faults=["slot_swap", "stale_owner"],
    )
    assert fault_payload(profile, "slot_swap")["values"][1] == 0.015625
    for fault in profile.required_faults:
        assert compare_operator(profile, fault_payload(profile, fault))["structure_passed"] is False


def test_sampler_faults_preserve_token_id_semantics_ties_and_penalty_boundary() -> None:
    from golden_gen.layered_release import OperatorProfile
    from golden_gen.operator_faults import fault_payload
    from golden_gen.operator_verification import compare_operator

    profile = OperatorProfile(
        profile_id="sampler",
        operator="sampling",
        dtype="float32",
        shape=[3, 151936],
        input_rule="unique_and_multiway_max_with_filter_penalties_v1",
        required_faults=["wrong_token_id", "wrong_tie", "ignored_penalty"],
    )
    assert fault_payload(profile, "ignored_penalty")["values"] == [7, 7, 29]
    assert fault_payload(profile, "wrong_tie")["values"] == [12, 12, 31]
    assert fault_payload(profile, "wrong_token_id")["values"] == [0, 0, 0]
    for fault in profile.required_faults:
        assert compare_operator(profile, fault_payload(profile, fault))["structure_passed"] is False


def test_fault_detection_budget_cannot_hide_an_important_operator_error() -> None:
    from golden_gen.layered_release import OperatorProfile
    from golden_gen.operator_faults import evaluate_fault_models, make_fault_models

    profile = OperatorProfile(
        profile_id="silu",
        operator="silu",
        dtype="bfloat16",
        shape=[1, 2],
        input_rule="gate_2_up_0.515625_v1",
        required_faults=["missing_intermediate_round"],
    )
    raw = make_fault_models([profile])
    assert (
        evaluate_fault_models([profile], raw, {"silu": 0})["silu"]["missing_intermediate_round"]
        is True
    )
    assert (
        evaluate_fault_models([profile], raw, {"silu": 0.00390625})["silu"][
            "missing_intermediate_round"
        ]
        is False
    )
    assert (
        evaluate_fault_models([profile], raw, {"silu": None})["silu"]["missing_intermediate_round"]
        is None
    )


def test_obvious_distribution_fault_records_all_four_actual_budget_responses() -> None:
    from golden_gen.layered_accuracy import Budgets
    from golden_gen.operator_faults import evaluate_distribution_fault

    measured = evaluate_distribution_fault(None)
    assert measured["verdict"] == "INVALID"
    assert measured["behavior_checks"]["g_peak"] == 10
    rejected = evaluate_distribution_fault(Budgets(0, 0, 0, 0))
    assert set(rejected["numerical_checks"]["conditions"].values()) == {False}
    assert rejected["behavior_checks"]["choice_loss_passed"] is False
    # Excessively permissive synthetic budgets demonstrate failed detection, not success.
    assert evaluate_distribution_fault(Budgets(20, 20, 20, 20))["verdict"] == "PASS"


def test_distribution_fault_strength_comes_from_its_frozen_input_definition() -> None:
    from golden_gen.operator_faults import evaluate_distribution_fault

    definition = dict(
        vocab_size=2,
        steps=1,
        token_ids=[0, 1],
        reference_logits=[[5.0, 0.0]],
        candidate_logits=[[0.0, 5.0]],
        baseline_logits=[[5.0, 0.0]],
        input_dtype="float32",
        predicted_token_ids=[1],
        analysis_temperature=1,
        raw_greedy_temperature=0,
    )
    assert evaluate_distribution_fault(None, definition)["behavior_checks"]["g_peak"] == 5
    definition["candidate_logits"] = [[5.0, 0.0]]
    import pytest

    with pytest.raises(ValueError):
        evaluate_distribution_fault(None, definition)
