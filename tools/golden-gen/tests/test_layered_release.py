"""New release evidence cannot borrow legacy verdicts or an unapproved policy."""

import json
import subprocess
from pathlib import Path

import pytest


def test_missing_registry_and_budget_binding_is_invalid_even_with_case_passes() -> None:
    from golden_gen.layered_release import release_verdict

    result = release_verdict(None, None, [], [], [])
    assert result["protocol"] == "layered-accuracy-v1"
    assert result["verdict"] == "INVALID"
    assert result["accepting"] is False
    assert "registry_pending" in result["reasons"]
    assert "budgets_pending" in result["reasons"]


def test_definition_binding_rejects_an_uncommitted_index_even_if_registry_blob_matches(
    tmp_path: Path,
) -> None:
    from golden_gen.layered_release import REGISTRY_PATH, definition_document

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "CPU Test")
    git("config", "user.email", "cpu@example.invalid")
    path = tmp_path / REGISTRY_PATH
    path.parent.mkdir(parents=True)
    path.write_text('{"protocol":"layered-accuracy-v1"}\n')
    (tmp_path / ".dag").mkdir()
    index = tmp_path / ".dag/definition-index.json"
    index.write_text(json.dumps({"inputs": []}))
    git("add", ".")
    git("commit", "-qm", "initial")
    index.write_text(
        json.dumps(
            {"inputs": [{"path": REGISTRY_PATH, "blob_oid": git("hash-object", REGISTRY_PATH)}]}
        )
    )
    with pytest.raises(ValueError):
        definition_document(tmp_path, REGISTRY_PATH)
    git("add", ".")
    git("commit", "-qm", "approved binding")
    assert definition_document(tmp_path, REGISTRY_PATH)[0]["protocol"] == "layered-accuracy-v1"
    path.write_text('{"protocol":"legacy"}\n')
    with pytest.raises(ValueError):
        definition_document(tmp_path, REGISTRY_PATH)


def test_one_member_failure_or_missing_operator_blocks_the_execution_group() -> None:
    from golden_gen.layered_release import BudgetPolicy, Registry, release_verdict

    registry = Registry.model_validate(
        dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            numerical_cases=[
                dict(
                    split="calibration",
                    required_mechanisms={"candidate":["batch"]},
                    engine_options={},
                    plan=dict(
                        protocol="layered-accuracy-v1",
                        schema_version=1,
                        execution_group_id="g",
                        call_id="c",
                        vocab_size=3,
                        members=[
                            dict(case_id=name, member_id=name, prompt=[1], continuation=[2])
                            for name in ("a", "b")
                        ],
                    ),
                )
            ],
            operator_profiles=[
                dict(
                    profile_id="rms",
                    operator="rmsnorm",
                    dtype="bfloat16",
                    shape=[1, 2],
                    input_rule="literal-v1",
                    required_faults=["bad-scale"],
                )
            ],
            behavior_cases=[dict(case_id="order", required_checks=["order"], scenario={})],
        )
    )
    policy = BudgetPolicy.model_validate(
        dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            algorithm="fp64-logsoftmax-fsum-underflow-recorded-p95-linear-v1",
            values=dict(a_mean=0, a_peak=0, delta_mean=0, g_limit=0),
            operator_budgets={"rms": 0},
            registry_sha256="a" * 64,
            calibration_source={"commit": "b" * 40, "tree": "c" * 40},
            calibration_evidence_sha256="d" * 64,
            fault_evidence_sha256="e" * 64,
            rationale="synthetic CPU test only",
        )
    )
    cases = [
        dict(protocol="layered-accuracy-v1", case_id=name, verdict=verdict)
        for name, verdict in (("a", "PASS"), ("b", "FAIL"))
    ]
    operators = [
        dict(
            profile_id="rms",
            max_abs_error=0.0,
            structure_passed=True,
            fault_checks={"bad-scale": True},
        )
    ]
    behaviors = [dict(case_id="order", checks={"order": True})]
    assert release_verdict(registry, policy, cases, operators, behaviors)["verdict"] == "FAIL"
    assert release_verdict(registry, policy, cases, [], behaviors)["verdict"] == "INVALID"
    cases[1]["verdict"] = "PASS"
    assert release_verdict(registry, policy, cases, operators, behaviors)["verdict"] == "PASS"
