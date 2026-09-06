import pytest


def test_operator_rule_has_independent_materialization_answer_and_exact_shape() -> None:
    from golden_gen.layered_release import OperatorProfile
    from golden_gen.operator_verification import compare_operator

    profile = OperatorProfile.model_validate(
        dict(
            profile_id="silu",
            operator="silu",
            dtype="bfloat16",
            shape=[1, 2],
            input_rule="gate_2_up_0.515625_v1",
            required_faults=["wrong-opmath"],
        )
    )
    actual = dict(
        rule_id=profile.input_rule,
        input_shape=[1, 2],
        output_shape=[1, 1],
        input_dtype="bfloat16",
        values=[0.90625],
    )
    assert compare_operator(profile, actual)["max_abs_error"] == 0
    wrong = {**actual, "values": [0.91015625]}
    assert compare_operator(profile, wrong)["max_abs_error"] == 0.00390625
    with pytest.raises(ValueError):
        compare_operator(profile, {**actual, "output_shape": [1, 2]})


def test_operator_capture_recomputes_error_and_cannot_turn_cpu_into_gpu_evidence() -> None:
    from golden_gen.layered_release import OperatorProfile
    from golden_gen.operator_verification import verify_operator_capture

    profile = OperatorProfile(
        profile_id="silu",
        operator="silu",
        dtype="bfloat16",
        shape=[1, 2],
        input_rule="gate_2_up_0.515625_v1",
        required_faults=["wrong-opmath"],
    )
    raw = dict(
        protocol="layered-accuracy-v1",
        schema_version=1,
        mode="operator_verification",
        device="cuda:0",
        complete=True,
        accepting=False,
        operator_checks=[
            dict(
                profile_id="silu",
                rule_id=profile.input_rule,
                input_shape=[1, 2],
                output_shape=[1, 1],
                input_dtype="bfloat16",
                values=[0.91015625],
                max_abs_error=0,
                structure_passed=True,
            )
        ],
    )
    result = verify_operator_capture([profile], raw, require_cuda=True)
    assert result[0]["max_abs_error"] == 0.00390625
    assert result[0]["fault_checks"] == {}  # Missing fault evidence is never inferred.
    with pytest.raises(ValueError, match="CUDA"):
        verify_operator_capture([profile], {**raw, "device": "cpu"}, require_cuda=True)


def test_declared_operator_input_is_not_overridden_by_a_matching_rule_id() -> None:
    from golden_gen.layered_release import OperatorProfile
    from golden_gen.operator_inputs import input_definition
    from golden_gen.operator_verification import compare_operator

    definition = input_definition("materialized_halfway_sum_v1")
    definition["epsilon"] = 0.1
    profile = OperatorProfile(
        profile_id="rms",
        operator="rmsnorm",
        dtype="bfloat16",
        shape=[1, 4],
        input_rule="materialized_halfway_sum_v1",
        input_definition=definition,
        required_faults=["wrong_epsilon"],
    )
    with pytest.raises(ValueError, match="normative input definition"):
        compare_operator(
            profile,
            dict(
                rule_id=profile.input_rule,
                input_shape=[1, 4],
                input_dtype="bfloat16",
                output_shape=[1, 4],
                values=[1, 1, 1, 1],
            ),
        )
