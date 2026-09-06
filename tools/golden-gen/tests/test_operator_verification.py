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
