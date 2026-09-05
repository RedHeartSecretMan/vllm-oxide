from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from golden_gen.report import (
    ReleaseReportInput,
    ToleranceEvidence,
    render_release_report,
    validate_report_sources,
)
from golden_gen.schema import Manifest
from tests.support import pinned_kernel_paths, release_runtime


def _benchmark() -> dict:
    repetition = {
        "prefill_tokens_per_second": 100.0,
        "decode_tokens_per_second": 20.0,
        "time_to_first_token_ns": [10_000_000],
        "inter_token_latency_ns": {
            "samples": [2_000_000, 3_000_000],
            "mean": 2_500_000.0,
            "p50": 2_500_000.0,
            "p95": 2_950_000.0,
        },
        "memory": {
            "baseline_mib": 2000,
            "peak_mib": 3250,
            "delta_mib": 1250,
            "polling_interval_ms": 50,
            "sample_count": 3,
        },
    }
    return {
        "canonical_04": {"repetitions": [repetition, repetition, repetition]},
        "canonical_05": {"repetitions": [repetition, repetition, repetition]},
    }


def _input() -> ReleaseReportInput:
    return ReleaseReportInput(
        observation_commit="1" * 40,
        observation_tree="2" * 40,
        policy_checkpoint_commit="3" * 40,
        policy_checkpoint_tree="4" * 40,
        measurement_commit="5" * 40,
        measurement_tree="6" * 40,
        runtime=release_runtime(),
        kernel_paths=pinned_kernel_paths(),
        lifecycle={
            "expected": 56,
            "discovered": 56,
            "generated": 56,
            "calibration_compared": 56,
            "reference_compared": 28,
            "missing": 0,
            "unexpected": 0,
            "skipped": 0,
            "failed": 0,
            "duplicate": 0,
            "stale": 0,
            "unmatched": 0,
        },
        tolerance={
            "l1_near_tie_max_abs_logit_gap": 0.0,
            "l2_atol": 0.015625,
            "observation_sha256": "7" * 64,
            "raw_evidence_sha256": "8" * 64,
            "rationale": "Reviewed candidate kernel rounding at same-prefix rows.",
        },
        benchmark=_benchmark(),
        manifest_sha256="9" * 64,
        archive_sha256="a" * 64,
        limitations=["Single GPU Qwen3 offline generation only."],
    )


def test_report_records_three_commit_roles_all_metrics_and_no_self_reference():
    report = render_release_report(_input())

    for heading in (
        "Observation commit",
        "Policy checkpoint",
        "Measurement commit",
        "Prefill throughput",
        "Decode throughput",
        "Time to first token",
        "Per-token latency",
        "Peak memory",
    ):
        assert heading in report
    assert "final candidate commit" not in report.lower()
    assert "recorded externally" in report
    assert '"samples": [' in report
    assert '"mean": 2500000.0' in report
    assert '"baseline_mib": 2000' in report
    assert '"sample_count": 3' in report
    assert "2.10.0" in report
    assert "Reviewed candidate kernel rounding at same-prefix rows." in report


def test_report_rejects_missing_metric_or_a_final_commit_self_reference():
    payload = _input().model_dump()
    del payload["benchmark"]["canonical_04"]["repetitions"][0]["memory"]
    with pytest.raises(ValidationError, match="benchmark"):
        ReleaseReportInput.model_validate(payload)

    payload = _input().model_dump()
    payload["final_commit"] = "f" * 40
    with pytest.raises(ValidationError, match="final_commit"):
        ReleaseReportInput.model_validate(payload)


def test_report_accepts_revised_ceiling_endpoints_but_not_values_above_them():
    payload = _input().tolerance.model_dump()
    payload.update(l1_near_tie_max_abs_logit_gap=0.125, l2_atol=1.0)
    evidence = ToleranceEvidence.model_validate(payload)
    assert evidence.l1_near_tie_max_abs_logit_gap == 0.125
    assert evidence.l2_atol == 1.0
    for field, value in (("l1_near_tie_max_abs_logit_gap", 0.125001), ("l2_atol", 1.000001)):
        with pytest.raises(ValidationError):
            ToleranceEvidence.model_validate({**payload, field: value})


def test_report_identities_bind_machine_evidence_and_real_checkpoint_ancestry(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def commit(message):
        git("add", ".")
        git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            message,
        )
        return git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")

    git("init", "-q")
    observation_commit, observation_tree = commit("observation source")
    observation = {
        "identity": {"measurement_commit": observation_commit, "measurement_tree": observation_tree}
    }
    relative = "docs/releases/goldens-v0.2-calibration-observation.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(observation))
    blob = git("hash-object", str(path))
    (tmp_path / ".dag").mkdir()
    (tmp_path / ".dag/definition-index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "inputs": [{"path": relative, "blob_oid": blob}],
            }
        )
    )
    policy_commit, policy_tree = commit("approved policy")
    measured_commit, measured_tree = commit("reviewed measurement source")
    manifest = Manifest.from_json(Path(__file__).parent / "fixtures/manifest-v4.json")
    observation_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = _input().model_copy(
        update={
            "observation_commit": observation_commit,
            "observation_tree": observation_tree,
            "policy_checkpoint_commit": policy_commit,
            "policy_checkpoint_tree": policy_tree,
            "measurement_commit": measured_commit,
            "measurement_tree": measured_tree,
            "tolerance": _input().tolerance.model_copy(
                update={
                    "observation_sha256": observation_sha,
                    "rationale": manifest.tolerance_policy.rationale,
                }
            ),
        }
    )
    manifest.tolerance_policy = manifest.tolerance_policy.model_copy(
        update={
            "evidence": [
                f"measurement-commit:{measured_commit}",
                f"measurement-tree:{measured_tree}",
                f"observation-measurement-commit:{observation_commit}",
                f"observation-measurement-tree:{observation_tree}",
                f"definition-observation-sha256:{observation_sha}",
            ]
        }
    )
    benchmark = {"measurement_commit": measured_commit, "measurement_tree": measured_tree}
    validate_report_sources(evidence, tmp_path, manifest, observation, benchmark)
    with pytest.raises(ValueError, match="differs from benchmark"):
        validate_report_sources(
            evidence.model_copy(update={"measurement_commit": "f" * 40}),
            tmp_path,
            manifest,
            observation,
            benchmark,
        )


def _forked_report(tmp_path, scenario="valid"):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def commit(message):
        git("add", ".")
        git("commit", "--allow-empty", "-qm", message)
        return git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    code = tmp_path / "runner.py"
    code.write_text("accepted source\n")
    base, _ = commit("accepted integration tip")
    git("checkout", "-qb", "observation")
    code.write_text("observed executable source\n")
    observed, observed_tree = commit("observation O")
    observation = {"identity": {"measurement_commit": observed, "measurement_tree": observed_tree}}
    git("checkout", "-qb", "policy", base)
    path = tmp_path / "docs/releases/goldens-v0.2-calibration-observation.json"
    path.parent.mkdir(parents=True)
    approved_bytes = json.dumps(observation).encode()
    path.write_bytes(approved_bytes)
    definition = tmp_path / "docs/adr/approved-policy.md"
    definition.parent.mkdir()
    definition.write_text("approved Definition bytes\n")
    index = tmp_path / ".dag/definition-index.json"
    index.parent.mkdir()
    selected = [
        {"path": entry.relative_to(tmp_path).as_posix(), "blob_oid": git("hash-object", str(entry))}
        for entry in (definition, path)
    ]
    if scenario == "wrong_p_index":
        selected[1]["blob_oid"] = "f" * 40
    index_bytes = json.dumps({"schema_version": 1, "inputs": selected}).encode()
    index.write_bytes(index_bytes)
    policy, policy_tree = commit("Definition-only checkpoint P")
    assert (
        subprocess.run(
            ["git", "-C", str(tmp_path), "merge-base", "--is-ancestor", observed, policy],
            check=False,
        ).returncode
        == 1
    )
    if scenario == "missing_o":
        measured, measured_tree = commit("measurement without O ancestor")
    elif scenario == "missing_p":
        git("checkout", "-q", "observation")
        path.parent.mkdir(parents=True, exist_ok=True)
        index.parent.mkdir(exist_ok=True)
        path.write_bytes(approved_bytes)
        definition.parent.mkdir(parents=True, exist_ok=True)
        definition.write_text("approved Definition bytes\n")
        index.write_bytes(index_bytes)
        measured, measured_tree = commit("copied policy bytes without P ancestor")
    else:
        git("checkout", "-q", "observation")
        git("merge", "--no-ff", "-qm", "measurement merge M", "policy")
        assert git("rev-parse", "HEAD^1") == observed
        assert git("rev-parse", "HEAD^2") == policy
        if scenario == "executable_change":
            code.write_text("different executable bytes after O\n")
        elif scenario == "renamed_executable":
            code.rename(definition.parent / "hidden-executable.md")
        elif scenario == "wrong_blob":
            path.write_text(json.dumps({**observation, "unreviewed": True}))
        elif scenario == "wrong_index":
            index.write_text(json.dumps({"schema_version": 1, "inputs": []}))
        elif scenario == "wrong_definition":
            definition.write_text("unreviewed Definition bytes\n")
        if scenario != "valid":
            commit("invalid measurement mutation")
        measured, measured_tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    manifest = Manifest.from_json(Path(__file__).parent / "fixtures/manifest-v4.json")
    observation_sha = hashlib.sha256(approved_bytes).hexdigest()
    if scenario == "wrong_p_sha":
        observation_sha = "e" * 64
    evidence = _input().model_copy(
        update={
            "observation_commit": observed,
            "observation_tree": observed_tree,
            "policy_checkpoint_commit": policy,
            "policy_checkpoint_tree": policy_tree,
            "measurement_commit": measured,
            "measurement_tree": measured_tree,
            "tolerance": _input().tolerance.model_copy(
                update={
                    "observation_sha256": observation_sha,
                    "rationale": manifest.tolerance_policy.rationale,
                }
            ),
        }
    )
    manifest.tolerance_policy = manifest.tolerance_policy.model_copy(
        update={
            "evidence": [
                f"measurement-commit:{measured}",
                f"measurement-tree:{measured_tree}",
                f"observation-measurement-commit:{observed}",
                f"observation-measurement-tree:{observed_tree}",
                f"definition-observation-sha256:{observation_sha}",
            ]
        }
    )
    benchmark = {"measurement_commit": measured, "measurement_tree": measured_tree}
    return evidence, manifest, observation, benchmark


def test_report_accepts_true_merge_of_observation_and_definition_only_checkpoint(tmp_path):
    evidence, manifest, observation, benchmark = _forked_report(tmp_path)
    validate_report_sources(evidence, tmp_path, manifest, observation, benchmark)


@pytest.mark.parametrize("scenario", ["missing_o", "missing_p"])
def test_report_requires_both_real_ancestors_not_just_copied_checkpoint_bytes(tmp_path, scenario):
    evidence, manifest, observation, benchmark = _forked_report(tmp_path, scenario)
    with pytest.raises(subprocess.CalledProcessError):
        validate_report_sources(evidence, tmp_path, manifest, observation, benchmark)


@pytest.mark.parametrize(
    "scenario,message",
    [
        ("executable_change", "executable"),
        ("renamed_executable", "executable"),
        ("wrong_blob", "checkpoint observation"),
        ("wrong_index", "Definition index"),
        ("wrong_definition", "Definition input"),
        ("wrong_p_sha", "different observation"),
        ("wrong_p_index", "not bound"),
    ],
)
def test_report_merge_cannot_conceal_changed_source_or_definition(tmp_path, scenario, message):
    evidence, manifest, observation, benchmark = _forked_report(tmp_path, scenario)
    with pytest.raises(ValueError, match=message):
        validate_report_sources(evidence, tmp_path, manifest, observation, benchmark)


def test_report_still_requires_three_distinct_commit_roles():
    payload = _input().model_dump()
    payload["policy_checkpoint_commit"] = payload["observation_commit"]
    with pytest.raises(ValidationError, match="commits must differ"):
        ReleaseReportInput.model_validate(payload)
