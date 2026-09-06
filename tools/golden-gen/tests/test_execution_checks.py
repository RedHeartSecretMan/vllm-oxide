import pytest


def test_partial_prefix_reuse_binds_real_cached_range_to_frozen_shared_history() -> None:
    from golden_gen.execution_checks import verify_prefix_reuse
    from golden_gen.layered_release import NumericalCase

    def plan(name: str, members: list[dict]) -> dict:
        return dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            execution_group_id=name,
            call_id=name,
            vocab_size=20,
            members=members,
        )

    case = NumericalCase.model_validate(
        dict(
            split="calibration",
            required_mechanisms={"candidate": ["prefix_hit"]},
            engine_options={},
            expected_cached_tokens={"candidate": {"full": 512, "partial": 256}},
            setup_calls=[
                plan(
                    "seed",
                    [dict(case_id="seed", member_id="seed", prompt=[7] * 513, continuation=[8])],
                )
            ],
            plan=plan(
                "target",
                [
                    dict(case_id="full", member_id="full", prompt=[7] * 513, continuation=[8]),
                    dict(
                        case_id="partial",
                        member_id="partial",
                        prompt=[7] * 256 + [9] * 257,
                        continuation=[8],
                    ),
                ],
            ),
        )
    )
    rows = {"full": [dict(request_id=1)], "partial": [dict(request_id=2)]}
    capture = {
        "execution_events": [
            {
                "members": [
                    dict(request_id=1, positions=[512, 513], cached_range=[0, 512]),
                    dict(request_id=2, positions=[256, 513], cached_range=[0, 256]),
                ]
            }
        ]
    }
    assert verify_prefix_reuse(case, "candidate", capture, rows) == {"full": 512, "partial": 256}
    capture["execution_events"][0]["members"][1]["cached_range"] = [0, 512]
    with pytest.raises(ValueError, match="cached prefix"):
        verify_prefix_reuse(case, "candidate", capture, rows)


def test_execution_behavior_requires_an_actual_recompute_not_a_case_label() -> None:
    from golden_gen.execution_checks import compare_execution_behavior
    from golden_gen.fixed_prefix_oracles import prediction_row
    from golden_gen.layered_release import BehaviorCase, NumericalCase

    group = NumericalCase.model_validate(
        dict(
            split="calibration",
            required_mechanisms={"candidate": ["prefill"]},
            engine_options={},
            plan=dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                execution_group_id="pressure",
                call_id="call",
                vocab_size=4,
                members=[dict(case_id="p", member_id="a", prompt=[1], continuation=[2, 3])],
            ),
        )
    )
    behavior = BehaviorCase.model_validate(
        dict(
            case_id="pressure-check",
            mode="fixed_prefix_execution",
            required_checks=["all_complete", "execution_history", "recompute_observed"],
            scenario={"execution_groups": ["pressure"]},
        )
    )
    capture = dict(
        protocol="layered-accuracy-v1",
        schema_version=1,
        mode="fixed_prefix",
        ignore_eos=True,
        complete=True,
        execution_group_id="pressure",
        call_id="call",
        rows=[
            prediction_row(group.plan, "a", 0, t, [0.0, 1.0, 0.0, 0.0], "prefill") for t in range(2)
        ],
        execution_events=[
            dict(
                plan_id=t,
                token_budget=t + 1,
                members=[
                    dict(
                        request_id=0,
                        completion_step=t,
                        sampling_allowed=True,
                        input_token_ids=[1, 2][: t + 1],
                        positions=[0, t + 1],
                        kv_length=t + 1,
                        phase="prefill",
                    )
                ],
            )
            for t in range(2)
        ],
    )
    assert all(
        compare_execution_behavior(behavior, {"pressure": (group, capture)})["checks"].values()
    )
    capture["rows"][1]["phase"] = "decode"
    capture["execution_events"][1] = dict(
        plan_id=1,
        token_budget=1,
        members=[
            dict(
                request_id=0,
                completion_step=1,
                sampling_allowed=True,
                input_token_ids=[2],
                positions=[1, 2],
                kv_length=2,
                phase="decode",
            )
        ],
    )
    assert (
        compare_execution_behavior(behavior, {"pressure": (group, capture)})["checks"][
            "recompute_observed"
        ]
        is False
    )


def test_unforced_public_control_links_returned_outputs_to_actual_execution_history() -> None:
    from golden_gen.execution_checks import compare_unforced_control_behavior
    from golden_gen.fixed_prefix_oracles import prediction_row
    from golden_gen.layered_release import BehaviorCase, NumericalCase

    group = NumericalCase.model_validate(
        dict(
            split="calibration",
            required_mechanisms={"candidate": ["recompute"]},
            engine_options={"max_model_len": 4096},
            plan=dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                execution_group_id="pressure",
                call_id="call",
                vocab_size=4,
                members=[dict(case_id="p", member_id="a", prompt=[1], continuation=[2, 3])],
            ),
        )
    )
    behavior = BehaviorCase.model_validate(
        dict(
            case_id="public-pressure",
            mode="unforced_control",
            required_checks=[
                "count",
                "order",
                "finished",
                "stop_policy",
                "execution_history",
                "recompute_observed",
            ],
            scenario={
                "execution_groups": ["pressure"],
                "calls": [
                    dict(
                        call_id="call",
                        prompts=[[1]],
                        params=[dict(max_tokens=2, ignore_eos=True)],
                        expected="success",
                    )
                ],
            },
        )
    )
    control = dict(
        protocol="layered-accuracy-v1",
        schema_version=1,
        mode="collection_control",
        ignore_eos=True,
        complete=True,
        execution_group_id="pressure",
        call_id="call",
        rows=[
            prediction_row(group.plan, "a", 0, t, [0.0, 1.0, 0.0, 0.0], "prefill", [1] * (t + 1))
            for t in range(2)
        ],
        execution_events=[
            dict(
                plan_id=t,
                token_budget=t + 1,
                members=[
                    dict(
                        request_id=0,
                        completion_step=t,
                        sampling_allowed=True,
                        input_token_ids=[1] * (t + 1),
                        positions=[0, t + 1],
                        kv_length=t + 1,
                        phase="prefill",
                    )
                ],
            )
            for t in range(2)
        ],
        public_call=dict(
            call_id="call",
            prompts=[[1]],
            params=[
                dict(
                    max_tokens=2,
                    ignore_eos=True,
                    temperature=0,
                    top_k=None,
                    top_p=None,
                    presence_penalty=0,
                    frequency_penalty=0,
                    repetition_penalty=0,
                )
            ],
            error=None,
            binding=dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                mode="behavior_binding",
                forcing_enabled=False,
                device="cpu",
                request_ids=[0],
                prompt_lengths=[1],
                eos_token_ids=[3],
                max_model_len=4096,
            ),
            outputs=[dict(request_id=0, token_ids=[1, 1], text="hello", finished=True)],
        ),
    )
    assert all(compare_unforced_control_behavior(behavior, group, [], control)["checks"].values())
    control["public_call"]["outputs"][0]["token_ids"] = [2, 3]
    with pytest.raises(ValueError, match="public outputs"):
        compare_unforced_control_behavior(behavior, group, [], control)
