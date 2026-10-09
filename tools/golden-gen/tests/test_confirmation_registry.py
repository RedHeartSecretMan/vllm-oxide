"""The new holdout supplements every observed obligation and cannot reuse it."""

import hashlib
import json
from pathlib import Path

import pytest

from golden_gen.layered_inventory import frozen_owner_inventory
from golden_gen.layered_release import BudgetPolicy, Registry, release_verdict

ROOT = Path(__file__).resolve().parents[3]


def registry_data() -> dict:
    return json.loads((ROOT / "docs/validation/layered-accuracy-cases.json").read_text())


def test_confirmation_preserves_all_original_registry_content_and_adds_130_owners() -> None:
    data = registry_data()
    registry = Registry.model_validate(data)
    assert len(frozen_owner_inventory(registry, authoritative=True)) == 617
    assert len(frozen_owner_inventory(registry, authoritative=False)) == 132
    for key in ("numerical_cases", "behavior_cases"):
        data[key] = [case for case in data[key] if case["split"] != "confirmation"]
    for key in ("literal_sources", "member_prompt_literals", "expected_counts"):
        del data[key]["confirmation"]
    data["case_sources"] = {
        key: value
        for key, value in data["case_sources"].items()
        if not key.startswith("confirmation-")
    }
    del data["protocol_rules"]["independence_rule"]
    data["schema_version"] = 1
    data["auxiliary_operators"]["total_unique_gpu_owners_including_groups_and_public"] = 487
    # Canonical JSON digest of original registry 88f89084...; original bytes
    # remain frozen at 479bf82 (different key order and epsilon formatting).
    original = (json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    assert hashlib.sha256(original).hexdigest() == (
        "7604df75d2f5ba2fc61143053ebaea4a06be6b83e8ad1758b67d712211435de2"
    )


def test_confirmation_rejects_renamed_reuse_of_an_unforced_history() -> None:
    data = registry_data()
    old_eos = next(
        case for case in data["behavior_cases"] if case["case_id"] == "acceptance-behavior-eos"
    )
    prior = old_eos["scenario"]["calls"][0]
    assert prior["params"][0]["max_tokens"] > 1
    fresh = next(case for case in data["numerical_cases"] if case["split"] == "confirmation")
    # Not a frozen fixed-prefix row: this could be a prior unforced continuation.
    fresh["plan"]["members"][0]["prompt"] = prior["prompts"][0] + [1234]
    with pytest.raises(ValueError, match="confirmation may reuse observed prediction history"):
        Registry.model_validate(data)


def test_confirmation_public_call_cannot_reuse_an_observed_numerical_prompt() -> None:
    data = registry_data()
    prior = next(case for case in data["numerical_cases"] if case["split"] == "calibration")
    fresh = next(
        case
        for case in data["behavior_cases"]
        if case["case_id"] == "confirmation-behavior-repeated"
    )
    fresh["scenario"]["calls"][0]["prompts"][0] = prior["plan"]["members"][0]["prompt"]
    with pytest.raises(ValueError, match="confirmation may reuse observed prediction history"):
        Registry.model_validate(data)


def test_confirmation_owner_omission_is_not_an_optional_report() -> None:
    data = registry_data()
    data["expected_counts"]["confirmation"]["owner_inventory"].pop()
    with pytest.raises(ValueError, match="owner inventory"):
        frozen_owner_inventory(Registry.model_validate(data), authoritative=True)


def test_old_passing_cases_cannot_satisfy_confirmation_release_evidence() -> None:
    registry = Registry.model_validate(registry_data())
    policy = BudgetPolicy.model_validate_json(
        (ROOT / "docs/validation/layered-accuracy-budgets.json").read_text()
    )
    old_cases = [
        dict(protocol="layered-accuracy-v1", case_id=member.case_id, verdict="PASS")
        for group in registry.numerical_cases
        if group.split != "confirmation"
        for member in group.plan.members
    ]
    result = release_verdict(registry, policy, old_cases, [], [])
    assert result["accepting"] is False
    assert "incomplete_or_unexpected_case_id_evidence" in result["reasons"]


@pytest.mark.parametrize("auxiliary", [False, True])
def test_confirmation_stays_sealed_when_policy_targets_the_observed_registry(
    monkeypatch, auxiliary
) -> None:
    from golden_gen import layered_cli
    from golden_gen.layered_release import POLICY_PATH, REGISTRY_PATH

    data = registry_data()
    digest = hashlib.sha256((ROOT / REGISTRY_PATH).read_bytes()).hexdigest()
    policy = json.loads((ROOT / POLICY_PATH).read_text())
    # Explicitly simulate the previously approved binding, not a new approval.
    policy["registry_sha256"] = "88f89084fe7b0c14a877c4c72f275314ad2a8e593d6fbc3ac63a194b91fb98f4"
    monkeypatch.setattr(
        layered_cli,
        "definition_document",
        lambda _, path: (data, digest) if path == REGISTRY_PATH else (policy, "old-policy"),
    )
    with pytest.raises(ValueError, match="remain.*sealed"):
        if auxiliary:
            layered_cli._auxiliary_definition(ROOT, "confirmation-behavior-repeated")
        else:
            layered_cli._group(ROOT, "confirmation-length-1")


def test_confirmation_behavior_only_still_rejects_reused_history() -> None:
    data = registry_data()
    data["numerical_cases"] = [g for g in data["numerical_cases"] if g["split"] != "confirmation"]
    data["behavior_cases"] = [
        b
        for b in data["behavior_cases"]
        if b["split"] != "confirmation" or b["case_id"] == "confirmation-behavior-repeated"
    ]
    prior = next(g for g in data["numerical_cases"] if g["split"] == "calibration")
    fresh = next(b for b in data["behavior_cases"] if b["split"] == "confirmation")
    data["expected_counts"]["confirmation"] = dict(
        owner_inventory=[
            dict(
                kind="standalone_behavior",
                verification_id=fresh["case_id"],
                engine="candidate",
                variant=v,
                calls=len(fresh["scenario"]["calls"]),
            )
            for v in ("primary", "replay")
        ],
        unique_gpu_owners=2,
    )
    data["auxiliary_operators"]["total_unique_gpu_owners_including_groups_and_public"] = 489
    assert len(frozen_owner_inventory(Registry.model_validate(data), authoritative=True)) == 489
    fresh["scenario"]["calls"][0]["prompts"][0] = prior["plan"]["members"][0]["prompt"]
    with pytest.raises(ValueError, match="confirmation may reuse observed prediction history"):
        Registry.model_validate(data)
