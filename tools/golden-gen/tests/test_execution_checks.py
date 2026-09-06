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
