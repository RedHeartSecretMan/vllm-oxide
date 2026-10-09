"""Renewing independent evidence must retain every previously frozen obligation."""

import hashlib
import json
from copy import deepcopy

import pytest

from golden_gen.layered_release import Registry
from tests.test_confirmation_registry import registry_data


def test_prior_confirmation_cases_splits_inputs_and_owners_are_unchanged() -> None:
    data = registry_data()
    prior = dict(
        numerical_cases=[
            g
            for g in data["numerical_cases"]
            if g["plan"]["execution_group_id"].startswith("confirmation-")
        ],
        behavior_cases=[
            c for c in data["behavior_cases"] if c["case_id"].startswith("confirmation-")
        ],
        literal_sources=data["literal_sources"]["confirmation"],
        member_prompt_literals=data["member_prompt_literals"]["confirmation"],
        case_sources={
            k: v for k, v in data["case_sources"].items() if k.startswith("confirmation-")
        },
        owner_inventory=[
            o
            for o in data["expected_counts"]["confirmation"]["owner_inventory"]
            if o.get("execution_group_id", o.get("verification_id", "")).startswith("confirmation-")
        ],
    )
    canonical = (json.dumps(prior, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    assert (
        hashlib.sha256(canonical).hexdigest()
        == "9041ac8ce5bd2a494b309211aea1cfec68e64d926eb52ab65a45ef7a96db05a5"
    )
    old = next(c for c in data["confirmation_cohorts"] if c["cohort_id"] == "confirmation-1")
    assert old["role"] == "regression"
    assert set(old["execution_groups"]) == {
        g["plan"]["execution_group_id"] for g in prior["numerical_cases"]
    }
    assert set(old["behavior_cases"]) == {c["case_id"] for c in prior["behavior_cases"]}


def test_fresh_cohort_cannot_reuse_an_earlier_confirmation_prediction() -> None:
    data = registry_data()
    old = next(
        g
        for g in data["numerical_cases"]
        if g["plan"]["execution_group_id"] == "confirmation-length-1"
    )
    fresh = next(
        g
        for g in data["numerical_cases"]
        if g["plan"]["execution_group_id"] == "confirmation2-length-1"
    )
    fresh["plan"]["members"][0]["prompt"] = deepcopy(old["plan"]["members"][0]["prompt"])
    with pytest.raises(ValueError, match="confirmation may reuse observed prediction history"):
        Registry.model_validate(data)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "none_independent",
        "two_independent",
        "duplicate_id",
        "missing_group",
        "duplicate_group",
        "unknown_group",
        "empty",
        "borrowed_group",
    ],
)
def test_incomplete_or_conflicting_cohort_declarations_are_invalid(mutation: str) -> None:
    data = registry_data()
    old, fresh = data["confirmation_cohorts"]
    if mutation == "missing":
        data.pop("confirmation_cohorts")
    elif mutation == "none_independent":
        fresh["role"] = "regression"
    elif mutation == "two_independent":
        old["role"] = "independent"
    elif mutation == "duplicate_id":
        fresh["cohort_id"] = old["cohort_id"]
    elif mutation == "missing_group":
        old["execution_groups"].pop()
    elif mutation == "duplicate_group":
        fresh["execution_groups"].append(old["execution_groups"][0])
    elif mutation == "unknown_group":
        fresh["execution_groups"].append("missing-confirmation-group")
    elif mutation == "empty":
        data["confirmation_cohorts"].append(
            dict(cohort_id="empty", role="regression", execution_groups=[], behavior_cases=[])
        )
    else:
        case = next(
            c
            for c in data["behavior_cases"]
            if c["case_id"] in fresh["behavior_cases"] and c["scenario"].get("execution_groups")
        )
        case["scenario"]["execution_groups"] = [old["execution_groups"][0]]
    with pytest.raises(ValueError, match="cohort"):
        Registry.model_validate(data)


def test_historical_confirmation_failure_cannot_be_waived_by_fresh_passes() -> None:
    from pathlib import Path

    from golden_gen.layered_release import BudgetPolicy, release_verdict

    registry = Registry.model_validate(registry_data())
    root = Path(__file__).resolve().parents[3]
    policy = BudgetPolicy.model_validate_json(
        (root / "docs/validation/layered-accuracy-budgets.json").read_text()
    )
    cases = [
        dict(
            protocol="layered-accuracy-v1",
            case_id=m.case_id,
            verdict="FAIL" if m.case_id == "confirmation-batch-medium" else "PASS",
        )
        for g in registry.numerical_cases
        for m in g.plan.members
    ]
    operators = [
        dict(
            profile_id=p.profile_id,
            max_abs_error=0.0,
            structure_passed=True,
            fault_checks={key: True for key in p.required_faults},
        )
        for p in registry.operator_profiles
    ]
    behaviors = [
        dict(
            case_id=c.case_id,
            evidence_complete=True,
            checks={key: True for key in c.required_checks},
        )
        for c in registry.behavior_cases
    ]
    result = release_verdict(registry, policy, cases, operators, behaviors)
    assert result["verdict"] == "FAIL" and result["accepting"] is False
