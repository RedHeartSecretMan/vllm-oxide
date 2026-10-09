"""Versioned layered release gates. No legacy verdict can satisfy these gates."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from golden_gen.fixed_prefix import ReplayPlan
from golden_gen.layered_accuracy import ALGORITHM, PROTOCOL, Budgets

REGISTRY_PATH = "docs/validation/layered-accuracy-cases.json"
POLICY_PATH = "docs/validation/layered-accuracy-budgets.json"


class NumericalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    split: Literal["development", "calibration", "acceptance", "confirmation"]
    plan: ReplayPlan
    required_mechanisms: dict[
        Literal["reference", "baseline", "candidate"],
        list[
            Literal[
                "prefill",
                "decode",
                "batch",
                "chunked_prefill",
                "prefix_hit",
                "recompute",
                "decode_cross_page",
                "waiting_admission",
            ]
        ],
    ] = Field(min_length=1)
    engine_options: dict[str, Any]
    setup_calls: list[ReplayPlan] = Field(default_factory=list)
    expected_cached_tokens: dict[Literal["reference", "baseline", "candidate"], dict[str, int]] = (
        Field(default_factory=dict)
    )


class OperatorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    profile_id: str = Field(min_length=1)
    operator: Literal["rmsnorm", "rope", "silu", "attention", "kv_cache", "sampling"]
    dtype: str = Field(min_length=1)
    shape: list[int] = Field(min_length=1)
    input_rule: str = Field(min_length=1)
    input_definition: dict[str, Any] = Field(default_factory=dict)
    required_faults: list[str] = Field(min_length=1)


class BehaviorCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str = Field(min_length=1)
    mode: Literal["free_generation", "fixed_prefix_execution", "unforced_control"] = (
        "free_generation"
    )
    split: Literal["development", "calibration", "acceptance", "confirmation"] = "calibration"
    required_checks: list[str] = Field(min_length=1)
    scenario: dict[str, Any]
    engine_options: dict[str, Any] = Field(default_factory=dict)


class ConfirmationCohort(BaseModel):
    """A frozen role for complete confirmation groups, without relabeling cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    cohort_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    role: Literal["regression", "independent"]
    execution_groups: list[str] = Field(default_factory=list)
    behavior_cases: list[str] = Field(default_factory=list)


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: Literal["layered-accuracy-v1"]
    schema_version: Literal[3]
    confirmation_cohorts: list[ConfirmationCohort] = Field(default_factory=list)
    numerical_cases: list[NumericalCase] = Field(min_length=1)
    operator_profiles: list[OperatorProfile] = Field(min_length=1)
    behavior_cases: list[BehaviorCase] = Field(min_length=1)
    case_sources: dict[str, Any] = Field(default_factory=dict)
    literal_sources: dict[str, Any] = Field(default_factory=dict)
    member_prompt_literals: dict[str, Any] = Field(default_factory=dict)
    fault_matrix: list[dict[str, Any]] = Field(default_factory=list)
    behavior_check_definitions: dict[str, str] = Field(default_factory=dict)
    engine_option_semantics: dict[str, Any] = Field(default_factory=dict)
    expected_counts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    protocol_rules: dict[str, str] = Field(default_factory=dict)
    auxiliary_operators: dict[str, Any] = Field(default_factory=dict)
    tokenizer_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    legacy_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        ids = [m.case_id for c in self.numerical_cases for m in c.plan.members]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate numerical case")
        if len({p.profile_id for p in self.operator_profiles}) != len(self.operator_profiles):
            raise ValueError("duplicate operator profile")
        if len({c.case_id for c in self.behavior_cases}) != len(self.behavior_cases):
            raise ValueError("duplicate behavior case")
        seen: dict[str, str] = {}
        for case in self.numerical_cases:
            for plan in [case.plan, *case.setup_calls]:
                for member in plan.members:
                    for step in range(len(member.continuation)):
                        key = plan.history_sha256(member.member_id, step)
                        if key in seen and case.split != seen[key]:
                            raise ValueError("prediction prefix overlaps another registry split")
                        seen[key] = case.split
            if case.split != "development" and any(
                m.case_id.startswith(("canonical_", "regression_")) for m in case.plan.members
            ):
                raise ValueError("old 28 cases belong to development only")
        _check_confirmation_independence(self)
        return self


def _independent_confirmation(registry: Registry) -> tuple[set[str], set[str]]:
    groups = {
        group.plan.execution_group_id
        for group in registry.numerical_cases
        if group.split == "confirmation"
    }
    behaviors = {case.case_id for case in registry.behavior_cases if case.split == "confirmation"}
    if not groups and not behaviors:
        if registry.confirmation_cohorts:
            raise ValueError("confirmation cohorts have no registered cases")
        return set(), set()
    cohorts = registry.confirmation_cohorts
    if (
        not cohorts
        or len({cohort.cohort_id for cohort in cohorts}) != len(cohorts)
        or sum(cohort.role == "independent" for cohort in cohorts) != 1
    ):
        raise ValueError("confirmation requires unique cohorts and exactly one independent cohort")
    group_owners: dict[str, str] = {}
    behavior_owners: dict[str, str] = {}
    for cohort in cohorts:
        if not cohort.execution_groups and not cohort.behavior_cases:
            raise ValueError("empty confirmation cohort")
        for names, owners in (
            (cohort.execution_groups, group_owners),
            (cohort.behavior_cases, behavior_owners),
        ):
            for name in names:
                if name in owners:
                    raise ValueError("duplicate confirmation cohort member")
                owners[name] = cohort.cohort_id
    if set(group_owners) != groups or set(behavior_owners) != behaviors:
        raise ValueError("confirmation cohort coverage differs from registered cases")
    for case in registry.behavior_cases:
        if case.split != "confirmation":
            continue
        linked = case.scenario.get("execution_groups", [])
        if not isinstance(linked, list) or any(
            not isinstance(group, str) or group_owners.get(group) != behavior_owners[case.case_id]
            for group in linked
        ):
            raise ValueError("confirmation behavior borrows another cohort's execution group")
    active = next(cohort for cohort in cohorts if cohort.role == "independent")
    return set(active.execution_groups), set(active.behavior_cases)


def _check_confirmation_independence(registry: Registry) -> None:
    """Reject possible reuse even when a prior unforced continuation is unknown.

    For compatible prompt prefixes, reachable prediction-length intervals must
    be disjoint. This includes every setup and successful public call, rather
    than only checking the fixed continuation rows or changing case identifiers.
    """
    fresh_groups, fresh_behaviors = _independent_confirmation(registry)
    if not fresh_groups and not fresh_behaviors:
        return
    from golden_gen.behavior_verification import BehaviorScenario

    ranges: dict[tuple[bool, tuple[int, ...], int], str] = {}
    for group in registry.numerical_cases:
        for plan in [*group.setup_calls, group.plan]:
            for member in plan.members:
                ranges[
                    (
                        group.plan.execution_group_id in fresh_groups,
                        tuple(member.prompt),
                        len(member.continuation),
                    )
                ] = member.case_id
    for case in registry.behavior_cases:
        if case.mode not in ("free_generation", "unforced_control"):
            continue
        scenario = BehaviorScenario.model_validate(dict(calls=case.scenario.get("calls")))
        for call in scenario.calls:
            if call.expected != "success":
                continue
            if len(call.prompts) != len(call.params):
                raise ValueError("successful public call has mismatched parameters")
            for public_prompt, params in zip(call.prompts, call.params, strict=True):
                if not public_prompt or params.max_tokens <= 0:
                    raise ValueError("successful public call has no prediction history")
                ranges[
                    (case.case_id in fresh_behaviors, tuple(public_prompt), params.max_tokens)
                ] = case.case_id
    if not any(fresh for fresh, _, _ in ranges):
        raise ValueError("independent confirmation has no prediction history")
    observed = [
        (prompt, steps, name) for (fresh, prompt, steps), name in ranges.items() if not fresh
    ]
    for (fresh, prompt, steps), name in ranges.items():
        if not fresh:
            continue
        for prior, prior_steps, prior_name in observed:
            overlap = max(len(prompt), len(prior)) <= min(
                len(prompt) + steps - 1, len(prior) + prior_steps - 1
            )
            width = min(len(prompt), len(prior))
            if overlap and prompt[:width] == prior[:width]:
                raise ValueError(
                    f"confirmation may reuse observed prediction history: {name} / {prior_name}"
                )


class BudgetPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    protocol: Literal["layered-accuracy-v1"]
    schema_version: Literal[1]
    algorithm: Literal["fp64-logsoftmax-fsum-underflow-recorded-p95-linear-v1"]
    values: dict[str, float] | None = None
    operator_budgets: dict[str, float | None] = Field(default_factory=dict)
    registry_sha256: str | None = None
    calibration_source: dict[str, str] | None = None
    calibration_evidence_sha256: str | None = None
    fault_evidence_sha256: str | None = None
    rationale: str | None = None

    @model_validator(mode="after")
    def valid_values(self) -> Self:
        if self.values is not None:
            if set(self.values) != {"a_mean", "a_peak", "delta_mean", "g_limit"}:
                raise ValueError("layered numerical budgets require exactly four named values")
            Budgets(**self.values)
        if any(
            value is not None and (not math.isfinite(value) or value < 0)
            for value in self.operator_budgets.values()
        ):
            raise ValueError("operator budgets must be finite/nonnegative or pending")
        return self


def definition_document(repo: Path, relative: str) -> tuple[dict[str, Any], str]:
    """Rehash the selected, tracked Definition input; a caller's approval flag is insufficient."""
    if relative not in (
        REGISTRY_PATH,
        POLICY_PATH,
        "docs/validation/layered-supervision-policy.json",
    ):
        raise ValueError("unsupported layered Definition document")
    index_bytes = (repo / ".dag/definition-index.json").read_bytes()
    tracked_index = subprocess.check_output(
        ["git", "-C", str(repo), "show", "HEAD:.dag/definition-index.json"]
    )
    if index_bytes != tracked_index:
        raise ValueError("Definition index differs from the accepted tracked checkpoint")
    index = json.loads(index_bytes)
    entry = next((x for x in index["inputs"] if x["path"] == relative), None)
    if entry is None:
        raise ValueError("layered Definition input is pending")
    data = (repo / relative).read_bytes()
    blob = (
        subprocess.check_output(
            ["git", "-C", str(repo), "hash-object", "--stdin"], input=data, text=False
        )
        .decode()
        .strip()
    )
    head_blob = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", f"HEAD:{relative}"], text=True
    ).strip()
    if blob != entry["blob_oid"] or head_blob != blob:
        raise ValueError("layered Definition input differs from approved tracked bytes")
    return json.loads(data), hashlib.sha256(data).hexdigest()


def release_verdict(
    registry: Registry | None,
    policy: BudgetPolicy | None,
    numerical: list[dict[str, Any]],
    operators: list[dict[str, Any]],
    behaviors: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = []
    if registry is None:
        reasons.append("registry_pending")
    if policy is None or policy.values is None:
        reasons.append("budgets_pending")
    failed = False
    if registry is not None and policy is not None:
        if not all(
            (
                policy.registry_sha256,
                policy.calibration_source,
                policy.calibration_evidence_sha256,
                policy.fault_evidence_sha256,
                policy.rationale,
            )
        ):
            reasons.append("budget_provenance_pending")
        expected_cases = {m.case_id for c in registry.numerical_cases for m in c.plan.members}
        expected_ops = {p.profile_id for p in registry.operator_profiles}
        expected_behaviors = {b.case_id for b in registry.behavior_cases}
        for records, key, expected in (
            (numerical, "case_id", expected_cases),
            (operators, "profile_id", expected_ops),
            (behaviors, "case_id", expected_behaviors),
        ):
            if len(records) != len(expected) or {r.get(key) for r in records} != expected:
                reasons.append(f"incomplete_or_unexpected_{key}_evidence")
        if any(policy.operator_budgets.get(profile) is None for profile in expected_ops):
            reasons.append("operator_budgets_pending")
        for case in numerical:
            if case.get("protocol") != PROTOCOL or case.get("verdict") not in (
                "PASS",
                "FAIL",
                "INVALID",
            ):
                reasons.append("invalid_numerical_evidence")
            elif case["verdict"] == "INVALID":
                reasons.append("invalid_numerical_case")
            elif case["verdict"] == "FAIL":
                failed = True
        for profile in registry.operator_profiles:
            evidence = next(
                (o for o in operators if o.get("profile_id") == profile.profile_id), None
            )
            if evidence is None:
                continue
            error = evidence.get("max_abs_error")
            if (
                not isinstance(error, (int, float))
                or isinstance(error, bool)
                or not math.isfinite(error)
                or error < 0
                or type(evidence.get("structure_passed")) is not bool
            ):
                reasons.append("invalid_operator_measurement")
                continue
            faults = evidence.get("fault_checks", {})
            if set(faults) != set(profile.required_faults) or any(
                type(v) is not bool for v in faults.values()
            ):
                reasons.append("incomplete_fault_detection_evidence")
            limit = policy.operator_budgets.get(profile.profile_id)
            failed |= (
                not evidence["structure_passed"]
                or not all(faults.values())
                or (limit is not None and error > limit)
            )
        for behavior_case in registry.behavior_cases:
            evidence = next(
                (b for b in behaviors if b.get("case_id") == behavior_case.case_id), None
            )
            if evidence is None:
                continue
            checks = evidence.get("checks", {})
            if evidence.get("evidence_complete") is False:
                reasons.append("incomplete_behavior_coverage")
            if set(checks) != set(behavior_case.required_checks) or any(
                type(v) is not bool for v in checks.values()
            ):
                reasons.append("incomplete_behavior_evidence")
            failed |= not all(checks.values())
    # This pure fold is deliberately non-authorizing. The manifest consumer
    # must also verify source/Definition binding, replays, guards and raw bytes.
    return dict(
        protocol=PROTOCOL,
        schema_version=1,
        algorithm=ALGORITHM,
        accepting=False,
        verdict="INVALID" if reasons else "FAIL" if failed else "PASS",
        reasons=reasons,
        operator_checks=operators,
        numerical_checks=numerical,
        behavior_checks=behaviors,
    )
