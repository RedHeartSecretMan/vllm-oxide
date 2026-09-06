"""Frozen execution expectations, separate from numerical metrics and budgets."""

from __future__ import annotations

from typing import Any

from golden_gen.layered_release import BehaviorCase, NumericalCase


def verify_prefix_reuse(
    case: NumericalCase, engine: str, capture: dict[str, Any], rows: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    expected = next(
        (values for key, values in case.expected_cached_tokens.items() if key == engine), {}
    )
    if not expected:
        return {}
    members = {m.member_id: m for m in case.plan.members}
    if set(expected) != members.keys():
        raise ValueError("cached prefix expectations must name every target member")
    computed_histories = [
        m.prompt + m.continuation[:-1] for setup in case.setup_calls for m in setup.members
    ]
    actual = {}
    for member_id, target in expected.items():
        if type(target) is not int or target < 0 or target >= len(members[member_id].prompt):
            raise ValueError("invalid cached prefix expectation")
        if target and not any(
            members[member_id].prompt[:target] == history[:target] and len(history) >= target
            for history in computed_histories
        ):
            raise ValueError("cached prefix cannot come from the declared setup history")
        request = rows[member_id][0]["request_id"]
        first = next(
            (
                m
                for event in capture["execution_events"]
                for m in event["members"]
                if m["request_id"] == request
            ),
            None,
        )
        if (
            first is None
            or first.get("positions", [None])[0] != target
            or first.get("cached_range") != [0, target]
        ):
            raise ValueError(
                "actual cached prefix differs from frozen partial/full reuse expectation"
            )
        actual[member_id] = target
    return actual


def compare_execution_behavior(
    case: BehaviorCase, captures: dict[str, tuple[NumericalCase, dict[str, Any]]]
) -> dict[str, Any]:
    from golden_gen.fixed_prefix import validate_capture, validate_execution_events

    if case.mode != "fixed_prefix_execution" or set(case.scenario) != {"execution_groups"}:
        raise ValueError("execution behavior must explicitly bind fixed-prefix groups")
    group_ids = case.scenario["execution_groups"]
    if not isinstance(group_ids, list) or not group_ids or len(group_ids) != len(set(group_ids)):
        raise ValueError("invalid execution behavior group inventory")
    checks = dict.fromkeys(
        (
            "all_complete",
            "execution_history",
            "recompute_observed",
            "prefix_ranges",
            "waiting_admission",
        ),
        True,
    )
    for group_id in group_ids:
        if group_id not in captures:
            raise ValueError("execution behavior group is missing")
        group, capture = captures[group_id]
        if group.plan.execution_group_id != group_id:
            raise ValueError("execution behavior group identity mismatch")
        rows = validate_capture(group.plan, capture)
        mechanisms = validate_execution_events(group.plan, capture, rows)
        prefix = verify_prefix_reuse(group, "candidate", capture, rows)
        checks["recompute_observed"] &= "recompute" in mechanisms
        checks["waiting_admission"] &= "waiting_admission" in mechanisms
        checks["prefix_ranges"] &= (
            bool(prefix) and len(set(prefix.values())) > 1 and min(prefix.values()) > 0
        )
    if (
        len(set(case.required_checks)) != len(case.required_checks)
        or not set(case.required_checks) <= checks.keys()
    ):
        raise ValueError("unknown execution behavior check")
    missing = [key for key in case.required_checks if not checks[key]]
    return dict(
        protocol="layered-accuracy-v1",
        case_id=case.case_id,
        mode="fixed_prefix_execution",
        accepting=False,
        evidence_complete=not missing,
        missing_mechanisms=missing,
        verdict="INVALID" if missing else "PASS",
        checks={key: checks[key] for key in case.required_checks},
    )


def compare_unforced_control_behavior(
    case: BehaviorCase, group: NumericalCase, setups: list[dict[str, Any]], control: dict[str, Any]
) -> dict[str, Any]:
    from golden_gen.behavior_verification import BehaviorScenario, compare_behavior
    from golden_gen.fixed_prefix import validate_control_capture, validate_execution_events

    if (
        case.mode != "unforced_control"
        or set(case.scenario) != {"execution_groups", "calls"}
        or case.scenario["execution_groups"] != [group.plan.execution_group_id]
    ):
        raise ValueError(
            "unforced public behavior must bind one frozen execution group and its calls"
        )
    scenario = BehaviorScenario.model_validate(dict(calls=case.scenario["calls"]))
    plans = [*group.setup_calls, group.plan]
    captures = [*setups, control]
    if len(plans) != len(captures) or len(plans) != len(scenario.calls):
        raise ValueError("unforced public setup/target call inventory mismatch")
    actual_calls = []
    target_rows = {}
    for plan, raw, expected in zip(plans, captures, scenario.calls, strict=True):
        rows = validate_control_capture(plan, raw)
        actual = raw.get("public_call")
        if (
            not isinstance(actual, dict)
            or expected.call_id != plan.call_id
            or actual.get("call_id") != expected.call_id
        ):
            raise ValueError("actual LLM::generate public call result is missing")
        prompts = [m.prompt for m in plan.members]
        if (
            expected.prompts != prompts
            or actual.get("prompts") != prompts
            or actual.get("params") != [p.model_dump() for p in expected.params]
        ):
            raise ValueError("unforced public call inputs/parameters differ from frozen calls")
        if actual.get("binding", {}).get("max_model_len") != group.engine_options.get(
            "max_model_len", 4096
        ):
            raise ValueError("unforced public context configuration mismatch")
        for member in plan.members:
            member_rows = rows[member.member_id]
            output = next(
                (
                    o
                    for o in actual.get("outputs", [])
                    if o.get("request_id") == member_rows[0]["request_id"]
                ),
                None,
            )
            if output is None or output.get("token_ids") != [
                r["predicted_token_id"] for r in member_rows
            ]:
                raise ValueError(
                    "unforced public outputs differ from their actual sampled execution history"
                )
        actual_calls.append(actual)
        target_rows = rows
    public_keys = {
        "count",
        "order",
        "finished",
        "stop_policy",
        "rejected_before_admission",
        "contextual_error",
        "resolved_eos_stop",
    }
    public_case = case.model_copy(
        update={
            "mode": "free_generation",
            "scenario": {"calls": case.scenario["calls"]},
            "required_checks": [key for key in case.required_checks if key in public_keys],
        }
    )
    result = compare_behavior(
        public_case,
        dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            mode="free_generation",
            calls=actual_calls,
        ),
    )
    actual_plan = group.plan.model_copy(
        update={
            "members": [
                m.model_copy(
                    update={
                        "continuation": [r["predicted_token_id"] for r in target_rows[m.member_id]]
                    }
                )
                for m in group.plan.members
            ]
        }
    )
    mechanisms = validate_execution_events(actual_plan, control, target_rows)
    prefix = verify_prefix_reuse(group, "candidate", control, target_rows)
    execution = dict(
        all_complete=True,
        execution_history=True,
        declared_mechanisms=set(group.required_mechanisms.get("candidate", [])) <= mechanisms,
        recompute_observed="recompute" in mechanisms,
        waiting_admission="waiting_admission" in mechanisms,
        prefix_ranges=bool(prefix) and len(set(prefix.values())) > 1 and min(prefix.values()) > 0,
    )
    if not set(case.required_checks) <= public_keys | execution.keys():
        raise ValueError("unknown unforced public behavior check")
    result["checks"].update(
        {key: execution[key] for key in case.required_checks if key in execution}
    )
    missing = [key for key in case.required_checks if key in execution and not execution[key]]
    result["evidence_complete"] &= not missing
    result["missing_mechanisms"].extend(missing)
    result["verdict"] = (
        "INVALID"
        if not result["evidence_complete"]
        else "PASS"
        if all(result["checks"].values())
        else "FAIL"
    )
    result.update(mode="unforced_control", actual_call_count=len(actual_calls))
    return result
