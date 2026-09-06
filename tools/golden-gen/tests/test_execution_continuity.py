import copy

import pytest


def test_decode_cannot_reprefill_and_chunked_prefill_cannot_skip_a_kv_interval() -> None:
    from golden_gen.fixed_prefix import ReplayPlan, validate_capture
    from golden_gen.fixed_prefix_oracles import prediction_row

    plan = ReplayPlan.model_validate(
        dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            execution_group_id="g",
            call_id="c",
            vocab_size=4,
            members=[dict(case_id="a", member_id="a", prompt=[1, 2, 3, 0], continuation=[1, 2])],
        )
    )

    def event(index, step, start, end, phase, sample):
        return dict(
            plan_id=index,
            token_budget=end - start,
            members=[
                dict(
                    request_id=0,
                    completion_step=step,
                    input_token_ids=plan.history("a", step)[start:end],
                    positions=[start, end],
                    kv_length=end,
                    phase=phase,
                    sampling_allowed=sample,
                )
            ],
        )

    good = dict(
        protocol="layered-accuracy-v1",
        schema_version=1,
        mode="fixed_prefix",
        ignore_eos=True,
        execution_group_id="g",
        call_id="c",
        complete=True,
        rows=[
            prediction_row(plan, "a", 0, t, [1.0, 0.0, 0.0, 0.0], "prefill" if t == 0 else "decode")
            for t in range(2)
        ],
        execution_events=[event(0, 0, 0, 4, "prefill", True), event(1, 1, 4, 5, "decode", True)],
    )
    assert len(validate_capture(plan, good)["a"]) == 2
    fake_decode = copy.deepcopy(good)
    fake_decode["execution_events"][1] = event(1, 1, 0, 5, "decode", True)
    with pytest.raises(ValueError):
        validate_capture(plan, fake_decode)
    skipped = copy.deepcopy(good)
    skipped["execution_events"] = [
        event(0, 0, 0, 1, "prefill", False),
        event(1, 0, 2, 4, "prefill", True),
        event(2, 1, 4, 5, "decode", True),
    ]
    with pytest.raises(ValueError):
        validate_capture(plan, skipped)
    repeated = copy.deepcopy(good)
    repeated["rows"][1]["phase"] = "prefill"
    repeated["execution_events"][1] = event(1, 1, 0, 5, "prefill", True)
    assert len(validate_capture(plan, repeated)["a"]) == 2
