"""Versioned producer/consumer seam, independent of oracle internals."""

import pytest


def test_execution_group_keeps_independent_numerical_case_ids() -> None:
    from golden_gen.fixed_prefix import ReplayPlan

    plan = ReplayPlan.model_validate(
        dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            execution_group_id="batch",
            call_id="batch-call",
            vocab_size=4,
            members=[
                dict(case_id="short-case", member_id="a", prompt=[1], continuation=[2]),
                dict(case_id="long-case", member_id="b", prompt=[1, 2], continuation=[3, 0]),
            ],
        )
    )
    assert [m.case_id for m in plan.members] == ["short-case", "long-case"]
    assert [len(m.continuation) for m in plan.members] == [1, 2]


def test_frozen_history_distinguishes_prediction_from_advance_and_rejects_missing_rows() -> None:
    from golden_gen.fixed_prefix import ReplayPlan, validate_capture

    plan = ReplayPlan.model_validate(
        dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            execution_group_id="toy-group",
            call_id="call1",
            vocab_size=4,
            members=[dict(case_id="toy", member_id="a", prompt=[1, 2], continuation=[3, 0])],
        )
    )
    rows = [
        dict(
            kind="prediction",
            case_id="toy",
            execution_group_id="toy-group",
            call_id="call1",
            member_id="a",
            request_id=7,
            step=t,
            history_sha256=plan.history_sha256("a", t),
            position=1 + t,
            effective_length=2 + t,
            phase="prefill" if t == 0 else "decode",
            row_shape=[4],
            predicted_token_id=1,
            advance_token_id=advance,
            logits=[0.0, 2.0, 1.0, 0.0],
        )
        for t, advance in enumerate([3, 0])
    ]
    capture = dict(
        protocol="layered-accuracy-v1",
        schema_version=1,
        mode="fixed_prefix",
        ignore_eos=True,
        execution_group_id="toy-group",
        call_id="call1",
        rows=rows,
        complete=True,
        execution_events=[
            dict(
                plan_id=1,
                token_budget=2,
                members=[
                    dict(
                        request_id=7,
                        completion_step=0,
                        sampling_allowed=True,
                        input_token_ids=[1, 2],
                        positions=[0, 2],
                        kv_length=2,
                        phase="prefill",
                    )
                ],
            ),
            dict(
                plan_id=2,
                token_budget=1,
                members=[
                    dict(
                        request_id=7,
                        completion_step=1,
                        sampling_allowed=True,
                        input_token_ids=[3],
                        positions=[2, 3],
                        kv_length=3,
                        phase="decode",
                    )
                ],
            ),
        ],
    )
    accepted = validate_capture(plan, capture)
    assert accepted["a"][1]["predicted_token_id"] == 1
    assert accepted["a"][1]["advance_token_id"] == 0
    with pytest.raises(ValueError):
        validate_capture(plan, {**capture, "execution_events": []})
    for field, bad in (("history_sha256", "0" * 64), ("advance_token_id", 2), ("position", 9)):
        invalid = {**capture, "rows": [rows[0], {**rows[1], field: bad}]}
        with pytest.raises(ValueError):
            validate_capture(plan, invalid)
    with pytest.raises(ValueError):
        validate_capture(plan, {**capture, "rows": rows[:1]})
    with pytest.raises(ValueError):
        validate_capture(plan, {**capture, "protocol": "same-prefix-v1"})

    from copy import deepcopy

    from golden_gen.fixed_prefix import require_collection_equivalence

    control = deepcopy(capture)
    control["mode"] = "collection_control"
    control["rows"][0]["advance_token_id"] = 1
    control["rows"][1]["advance_token_id"] = 1
    control_plan = plan.model_copy(
        update={"members": [plan.members[0].model_copy(update={"continuation": [1, 1]})]}
    )
    control["rows"][1]["history_sha256"] = control_plan.history_sha256("a", 1)
    control["execution_events"][1]["members"][0]["input_token_ids"] = [1]
    assert require_collection_equivalence(plan, capture, control) == {"a": 1}
    control["rows"][0]["logits"][0] = -0.0
    with pytest.raises(ValueError):
        require_collection_equivalence(plan, capture, control)
