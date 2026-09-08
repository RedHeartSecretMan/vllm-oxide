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


@pytest.mark.parametrize("identity", ["request_id", "native_request_id", "external_request_id"])
def test_baseline_owner_rejects_setup_target_aliases_but_fresh_owners_are_independent(identity):
    from golden_gen.fixed_prefix import validate_baseline_owner_bindings

    setup, first = fixture()
    target = setup.model_copy(update={"call_id": "target"})
    second = copy.deepcopy(first)
    for binding in second["request_bindings"]:
        binding["call_id"] = "target"
        binding["request_id"] += 2
        binding["native_request_id"] += "-new"
        binding["external_request_id"] += "-new"

    def rebind_rows(capture):
        for index, binding in enumerate(capture["request_bindings"]):
            for row in (capture["rows"][index], capture["execution_events"][0]["members"][index]):
                row["request_id"] = binding["request_id"]
                row["native_request_id"] = binding["native_request_id"]

    rebind_rows(second)
    calls = [(setup, first), (target, second)]
    validate_baseline_owner_bindings(iter(calls))
    validate_baseline_owner_bindings(iter(calls))  # Another fresh owner may use identical IDs.
    bad = copy.deepcopy(second)
    bad["request_bindings"][0][identity] = first["request_bindings"][0][identity]
    rebind_rows(bad)
    validate_baseline_bindings(target, bad)  # Internally valid; only owner scope reveals aliasing.
    with pytest.raises(ValueError, match="owner.*binding"):
        validate_baseline_owner_bindings(iter([(setup, first), (target, bad)]))
