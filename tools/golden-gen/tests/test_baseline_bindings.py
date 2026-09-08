import copy

import pytest

from golden_gen.fixed_prefix import ReplayPlan, validate_baseline_bindings


def fixture():
    plan = ReplayPlan.model_validate(
        dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            execution_group_id="g",
            call_id="c",
            vocab_size=3,
            members=[
                dict(case_id=m, member_id=m, prompt=[1], continuation=[2]) for m in ("a", "b")
            ],
        )
    )
    bindings = [
        dict(
            request_id=i,
            native_request_id=f"0-opaque-{m}",
            external_request_id=f"external/{m}",
            case_id=m,
            member_id=m,
            call_id="c",
            execution_group_id="g",
        )
        for i, m in enumerate(("a", "b"))
    ]
    capture = dict(
        request_identity="vllm-owner-local-v1",
        request_bindings=bindings,
        rows=[
            dict(
                member_id=b["member_id"],
                request_id=b["request_id"],
                native_request_id=b["native_request_id"],
            )
            for b in bindings
        ],
        execution_events=[
            dict(
                members=[
                    dict(request_id=b["request_id"], native_request_id=b["native_request_id"])
                    for b in bindings
                ]
            )
        ],
    )
    return plan, capture


@pytest.mark.parametrize(
    "mutation", ["missing", "native-alias", "external-alias", "row-rebind", "event-rebind"]
)
def test_baseline_native_external_mapping_cannot_be_omitted_aliased_or_rebound(mutation):
    plan, capture = fixture()
    validate_baseline_bindings(plan, capture)
    bad = copy.deepcopy(capture)
    if mutation == "missing":
        del bad["request_bindings"]
    elif mutation == "native-alias":
        bad["request_bindings"][1]["native_request_id"] = bad["request_bindings"][0][
            "native_request_id"
        ]
    elif mutation == "external-alias":
        bad["request_bindings"][1]["external_request_id"] = bad["request_bindings"][0][
            "external_request_id"
        ]
    elif mutation == "row-rebind":
        bad["rows"][0]["native_request_id"] = "0-another-suffix"
    else:
        bad["execution_events"][0]["members"][0]["native_request_id"] = "0-another-suffix"
    with pytest.raises(ValueError, match="binding"):
        validate_baseline_bindings(plan, bad)
