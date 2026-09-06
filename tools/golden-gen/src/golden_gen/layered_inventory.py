"""Unique guarded owners and call counts, derived then checked against frozen policy."""

from __future__ import annotations

import re
from typing import Any

from golden_gen.layered_release import Registry


def owner_key(owner: dict[str, Any]) -> tuple[str, str, str, str]:
    kind = owner.get("kind")
    name = owner.get("execution_group_id" if kind == "execution_group" else "verification_id")
    if (
        kind not in ("execution_group", "standalone_behavior", "operator_suite")
        or not isinstance(name, str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None
        or name in (".", "..")
    ):
        raise ValueError("unsafe/unknown owner inventory identity")
    engine, variant = owner.get("engine"), owner.get("variant")
    if engine not in ("reference", "baseline", "candidate") or variant not in (
        "primary",
        "replay",
        "control",
        "control-replay",
    ):
        raise ValueError("unknown owner inventory engine/variant")
    return kind, name, engine, variant


def _check_owners(declared: Any, expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(declared, list) or len(declared) != len(expected):
        raise ValueError("frozen owner inventory is missing or incomplete")
    by_key = {owner_key(o): o for o in declared}
    if len(by_key) != len(declared) or set(by_key) != {owner_key(o) for o in expected}:
        raise ValueError("frozen owner inventory has duplicate/unexpected owners")
    for target in expected:
        actual = by_key[owner_key(target)]
        if any(
            actual.get(key) != value or type(actual.get(key)) is not type(value)
            for key, value in target.items()
        ):
            raise ValueError("frozen owner inventory row/call metadata differs from the cases")
    return declared


def frozen_owner_inventory(registry: Registry, *, authoritative: bool) -> list[dict[str, Any]]:
    from golden_gen.behavior_verification import BehaviorScenario

    groups_by_id = {g.plan.execution_group_id: g for g in registry.numerical_cases}
    if len(groups_by_id) != len(registry.numerical_cases):
        raise ValueError("duplicate execution group in owner inventory")
    public_groups = set()
    for case in registry.behavior_cases:
        if case.mode != "unforced_control":
            continue
        ids = case.scenario.get("execution_groups", [])
        if len(ids) != 1 or ids[0] not in groups_by_id or groups_by_id[ids[0]].split != case.split:
            raise ValueError("unforced behavior does not bind a same-split owner group")
        group = groups_by_id[ids[0]]
        plans = [*group.setup_calls, group.plan]
        calls = BehaviorScenario.model_validate(dict(calls=case.scenario.get("calls"))).calls
        if len(calls) != len(plans):
            raise ValueError("unforced public call inventory mismatch")
        for plan, call in zip(plans, calls, strict=True):
            if (
                call.expected != "success"
                or call.call_id != plan.call_id
                or call.prompts != [m.prompt for m in plan.members]
                or len(call.params) != len(plan.members)
            ):
                raise ValueError("unforced public inputs differ from their owner group")
            for member, params in zip(plan.members, call.params, strict=True):
                if (
                    params.max_tokens != len(member.continuation)
                    or not params.ignore_eos
                    or params.temperature != 0
                    or params.top_k is not None
                    or params.top_p is not None
                    or any(
                        (
                            params.presence_penalty,
                            params.frequency_penalty,
                            params.repetition_penalty,
                        )
                    )
                ):
                    raise ValueError(
                        "unforced public owner requires its frozen neutral greedy completion budget"
                    )
        public_groups.add(ids[0])
    inventory = []
    splits = ("development", "calibration", "acceptance") if authoritative else ("calibration",)
    for split in splits:
        groups = [g for g in registry.numerical_cases if g.split == split]
        public = [
            b for b in registry.behavior_cases if b.split == split and b.mode == "free_generation"
        ]
        if not groups and not public:
            continue
        expected: list[dict[str, Any]] = []
        shared = [g for g in groups if g.plan.execution_group_id in public_groups]
        for group in groups:
            for engine in ("reference", "baseline", "candidate"):
                variants = ["primary", "replay", "control"] + (
                    ["control-replay"]
                    if engine == "candidate" and group.plan.execution_group_id in public_groups
                    else []
                )
                for variant in variants:
                    expected.append(
                        dict(
                            kind="execution_group",
                            execution_group_id=group.plan.execution_group_id,
                            engine=engine,
                            variant=variant,
                            target_rows=sum(len(m.continuation) for m in group.plan.members),
                            setup_calls=len(group.setup_calls),
                            setup_rows=sum(
                                len(m.continuation) for p in group.setup_calls for m in p.members
                            ),
                        )
                    )
        group_owners = list(expected)
        for case in public:
            calls = BehaviorScenario.model_validate(case.scenario).calls
            for variant in ("primary", "replay"):
                expected.append(
                    dict(
                        kind="standalone_behavior",
                        verification_id=case.case_id,
                        engine="candidate",
                        variant=variant,
                        calls=len(calls),
                    )
                )
        counts = registry.expected_counts.get(split, {})
        declared = _check_owners(counts.get("owner_inventory"), expected)
        calculated = dict(
            groups=len(groups),
            members=sum(len(g.plan.members) for g in groups),
            target_rows_per_engine_variant=sum(
                len(m.continuation) for g in groups for m in g.plan.members
            ),
            numerical_gpu_owners=9 * len(groups),
            group_collection_gpu_owners=len(group_owners),
            public_behavior_owners=2 * (len(public) + len(shared)),
            standalone_public_behavior_owners=2 * len(public),
            numerical_public_owner_overlap=len(shared),
            unique_gpu_owners=len(expected),
            group_collection_calls=sum(1 + o["setup_calls"] for o in group_owners),
            target_plus_setup_rows_all_engines_variants=sum(
                o["target_rows"] + o["setup_rows"] for o in group_owners
            ),
            candidate_control_replay_target_rows=sum(
                o["target_rows"] for o in group_owners if o["variant"] == "control-replay"
            ),
            candidate_control_replay_setup_rows=sum(
                o["setup_rows"] for o in group_owners if o["variant"] == "control-replay"
            ),
        )
        if any(
            key in counts and (type(counts[key]) is not int or counts[key] != value)
            for key, value in calculated.items()
        ):
            raise ValueError("frozen owner inventory counts disagree with unique owners")
        inventory.extend(declared)
    operators = [
        dict(kind="operator_suite", verification_id="operators", engine="candidate", variant=v)
        for v in ("primary", "replay")
    ]
    inventory.extend(_check_owners(registry.auxiliary_operators.get("owner_inventory"), operators))
    total = registry.auxiliary_operators.get("total_unique_gpu_owners_including_groups_and_public")
    if authoritative and total is not None and (type(total) is not int or total != len(inventory)):
        raise ValueError("frozen total owner inventory count mismatch")
    return inventory
