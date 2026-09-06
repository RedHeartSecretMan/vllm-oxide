"""Frozen execution expectations, separate from numerical metrics and budgets."""

from __future__ import annotations

from typing import Any

from golden_gen.layered_release import NumericalCase


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
