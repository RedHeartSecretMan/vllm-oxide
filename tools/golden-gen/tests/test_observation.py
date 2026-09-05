from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from golden_gen.config import COMPARISON_KERNEL_SCOPE
from golden_gen.observation import (
    CALIBRATION_IDS,
    HOLDOUT_IDS,
    ObservationIdentity,
    ObservationInput,
    approve_manifest_policy,
    canonical_observation_json,
    observation_exit_code,
    observe_calibration,
    observe_case,
    prepare_definition_observation,
    propose_thresholds,
)
from golden_gen.schema import Manifest, TolerancePolicy


def test_same_prefix_observation_includes_first_divergence_and_uses_fixed_ladders():
    reference_logits = np.array([[0.0, 1.0], [0.19, 0.20], [9.0, 9.0]], dtype=np.float32)
    candidate_logits = np.array([[0.0, 1.0001], [0.20, 0.21], [-9.0, -9.0]], dtype=np.float32)
    reference_tokens = np.array([1, 1, 0], dtype=np.int64)
    candidate_tokens = np.array([1, 0, 1], dtype=np.int64)

    observed = observe_case(
        "canonical_01",
        reference_logits,
        candidate_logits,
        reference_tokens,
        candidate_tokens,
    )
    proposal = propose_thresholds([observed])

    assert observed.compared_rows == 2
    assert observed.compared_elements == 4
    assert observed.first_divergence == 1
    assert observed.excluded_rows == 1
    assert observed.candidate_token_gap == pytest.approx(0.01)
    assert observed.maximum_abs_error == pytest.approx(0.01)
    assert proposal.l1_near_tie_max_abs_logit_gap == 2**-6
    assert proposal.l2_atol == 2**-6


def test_non_finite_or_shape_invalid_observation_fails_instead_of_sampling_it():
    logits = np.array([[0.0, np.nan]], dtype=np.float32)
    tokens = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError, match="non-finite"):
        observe_case("canonical_01", logits, logits, tokens, tokens)

    with pytest.raises(ValueError, match="shape"):
        observe_case("canonical_01", logits[:, :1], logits, tokens, tokens)


@pytest.mark.parametrize(
    "maximum,expected",
    [(0.0, 0.0), (0.25, 0.25), (0.250001, 0.5), (0.5, 0.5), (0.500001, 1.0), (1.0, 1.0)],
)
def test_revised_l2_ladder_selects_smallest_cover_without_multiplier(maximum, expected):
    observed = observe_case(
        "canonical_01",
        np.array([[0.0]], dtype=np.float32),
        np.array([[maximum]], dtype=np.float32),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
    )
    proposal = propose_thresholds([observed])
    assert proposal.l2_atol == expected
    assert proposal.l1_near_tie_max_abs_logit_gap == 0.0


@pytest.mark.parametrize("gap,expected", [(0.0625, 0.0625), (0.062501, 0.125), (0.125, 0.125)])
def test_revised_l1_ladder_covers_every_actual_divergence_gap(gap, expected):
    observed = observe_case(
        "canonical_01",
        np.array([[gap, 0.0]], dtype=np.float32),
        np.array([[0.0, gap]], dtype=np.float32),
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
    )
    assert propose_thresholds([observed]).l1_near_tie_max_abs_logit_gap == expected


def test_full_observation_reads_exact_candidate_subset_and_is_always_non_accepting():
    opened: list[str] = []

    def load_case(prompt_id: str) -> ObservationInput:
        opened.append(prompt_id)
        logits = np.array([[0.0, 1.0]], dtype=np.float32)
        tokens = np.array([1], dtype=np.int64)
        return ObservationInput(logits, logits.copy(), tokens, tokens.copy())

    record = observe_calibration(
        load_case,
        ObservationIdentity(
            measurement_commit="1" * 40,
            measurement_tree="2" * 40,
            manifest_sha256="3" * 64,
            candidate_binary_sha256="4" * 64,
            runtime_sha256="5" * 64,
            kernel_scope="reference::vs::candidate",
            raw_evidence_sha256="6" * 64,
        ),
    )

    assert opened == list(CALIBRATION_IDS)
    assert record.opened_fixture_ids == CALIBRATION_IDS
    assert record.sealed_holdout_ids == HOLDOUT_IDS
    assert record.accepting is False
    assert record.input_l1_threshold == 0.0
    assert record.input_l2_threshold == 0.0
    assert record.aggregate.compared_rows == 4
    assert record.aggregate.compared_elements == 8
    assert record.proposal.l1_near_tie_max_abs_logit_gap == 0.0
    assert record.proposal.l2_atol == 0.0
    assert canonical_observation_json(record) == canonical_observation_json(record)
    assert observation_exit_code(record) != 0


@pytest.mark.parametrize(
    ("maximum_abs_error", "candidate_token_gap", "message"),
    [
        (1.000_001, None, "L2 absolute error"),
        (0.0, 0.125_001, "L1 candidate gap"),
    ],
)
def test_threshold_proposal_never_exceeds_approved_ceilings(
    maximum_abs_error, candidate_token_gap, message
):
    observed = observe_case(
        "canonical_01",
        np.array([[0.0]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
    )
    observed = observed.__class__(
        **{
            **observed.__dict__,
            "maximum_abs_error": maximum_abs_error,
            "candidate_token_gap": candidate_token_gap,
        }
    )

    with pytest.raises(ValueError, match=message):
        propose_thresholds([observed])


def _pending_manifest(tmp_path: Path) -> Path:
    fixture = Path(__file__).parent / "fixtures" / "manifest-v4.json"
    manifest = Manifest.from_json(fixture)
    reference_template, baseline_template = manifest.expected_fixtures
    reference_fixture, baseline_fixture = manifest.fixtures
    cases = [
        *[(f"canonical_{index:02d}", "canonical") for index in range(1, 5)],
        *[(f"canonical_05{suffix}", "batch") for suffix in "abcd"],
        *[(f"regression_{index:02d}", "regression") for index in range(1, 21)],
    ]
    manifest.expected_fixtures = []
    manifest.fixtures = []
    for prompt_id, family in cases:
        category = "regression" if family == "regression" else "canonical"
        required = "l1" if family == "regression" else "l1_l2"
        manifest.expected_fixtures.extend(
            [
                reference_template.model_copy(
                    update={
                        "fixture_id": f"{prompt_id}.transformers",
                        "prompt_id": prompt_id,
                        "family": family,
                        "required_comparison": required,
                        "filename": f"{prompt_id}.transformers.safetensors",
                    }
                ),
                baseline_template.model_copy(
                    update={
                        "fixture_id": f"{prompt_id}.vllm",
                        "prompt_id": prompt_id,
                        "family": family,
                        "filename": f"{prompt_id}.vllm.safetensors",
                    }
                ),
            ]
        )
        manifest.fixtures.extend(
            [
                reference_fixture.model_copy(
                    update={
                        "prompt_id": prompt_id,
                        "category": category,
                        "filename": f"{prompt_id}.transformers.safetensors",
                    }
                ),
                baseline_fixture.model_copy(
                    update={
                        "prompt_id": prompt_id,
                        "category": category,
                        "filename": f"{prompt_id}.vllm.safetensors",
                    }
                ),
            ]
        )
    manifest.calibrated_fixtures = [f"{prompt_id}.vllm" for prompt_id, _family in cases]
    manifest.tolerance_policy = TolerancePolicy(
        version="same-prefix-v1",
        dtype="bfloat16",
        kernel=COMPARISON_KERNEL_SCOPE,
        l1_near_tie_max_abs_logit_gap=0.0,
        l2_atol=0.0,
        rationale="pending empirical approval",
        evidence=[],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest_path


def _commit(repo: Path, message: str, *paths: str) -> None:
    observation = repo / "docs/releases/goldens-v0.2-calibration-observation.json"
    if observation.exists():
        blob = subprocess.run(
            ["git", "-C", str(repo), "hash-object", str(observation)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        (repo / ".dag").mkdir(exist_ok=True)
        (repo / ".dag/definition-index.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "inputs": [
                        {"path": observation.relative_to(repo).as_posix(), "blob_oid": blob}
                    ],
                }
            )
        )
        paths = (*paths, ".dag/definition-index.json")
    if paths:
        subprocess.run(["git", "-C", str(repo), "add", "--", *paths], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            message,
        ],
        check=True,
    )


def _policy_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    manifest_path = _pending_manifest(tmp_path)
    manifest_bytes = manifest_path.read_bytes()

    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "src.py").write_text("print('measured executable')\n")
    _commit(repo, "measurement candidate", "src.py")
    measurement_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    measurement_tree = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def load_case(_prompt_id: str) -> ObservationInput:
        return ObservationInput(
            reference_logits=np.array([[0.0, 1.0], [0.19, 0.20]], dtype=np.float32),
            candidate_logits=np.array([[0.0, 1.01], [0.20, 0.21]], dtype=np.float32),
            reference_tokens=np.array([1, 1], dtype=np.int64),
            candidate_tokens=np.array([1, 0], dtype=np.int64),
        )

    raw_observation = tmp_path / "raw-observation.json"
    record = observe_calibration(
        load_case,
        ObservationIdentity(
            measurement_commit=measurement_commit,
            measurement_tree=measurement_tree,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            candidate_binary_sha256="4" * 64,
            runtime_sha256=hashlib.sha256(
                Manifest.from_json(manifest_path).runtime.model_dump_json().encode()
            ).hexdigest(),
            kernel_scope=COMPARISON_KERNEL_SCOPE,
            raw_evidence_sha256="6" * 64,
        ),
    )
    raw_observation.write_bytes(canonical_observation_json(record))
    observation_path = repo / "docs/releases/goldens-v0.2-calibration-observation.json"
    observation_path.parent.mkdir(parents=True)
    prepare_definition_observation(
        raw_observation,
        observation_path,
        ["BF16 candidate-kernel rounding at same-prefix rows."],
    )
    _commit(
        repo,
        "approve observation Definition",
        "docs/releases/goldens-v0.2-calibration-observation.json",
    )
    head_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head_tree = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return manifest_path, observation_path, repo, head_commit, head_tree


def test_policy_approval_recomputes_tracked_definition_observation(tmp_path):
    manifest_path, observation_path, repo, head_commit, head_tree = _policy_fixture(tmp_path)

    approved = approve_manifest_policy(manifest_path, observation_path, repo)

    assert approved.tolerance_policy.l1_near_tie_max_abs_logit_gap == 0.015625
    assert approved.tolerance_policy.l2_atol == 0.015625
    assert approved.tolerance_policy.rationale == (
        "BF16 candidate-kernel rounding at same-prefix rows."
    )
    assert f"measurement-commit:{head_commit}" in approved.tolerance_policy.evidence
    assert f"measurement-tree:{head_tree}" in approved.tolerance_policy.evidence


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("proposal", "smallest covering ladder"),
        ("aggregate", "cover every case sample"),
    ],
)
def test_policy_approval_rejects_tampered_statistics_and_proposal(tmp_path, field, message):
    manifest_path, observation_path, repo, _head_commit, _head_tree = _policy_fixture(tmp_path)
    observation = json.loads(observation_path.read_bytes())
    if field == "proposal":
        observation["proposal"]["l2_atol"] = 0.03125
    else:
        observation["aggregate"]["compared_elements"] += 1
    observation_path.write_text(json.dumps(observation, sort_keys=True, separators=(",", ":")))
    _commit(
        repo,
        "tampered Definition",
        "docs/releases/goldens-v0.2-calibration-observation.json",
    )

    with pytest.raises(ValueError, match=message):
        approve_manifest_policy(manifest_path, observation_path, repo)


@pytest.mark.parametrize("rename", [False, True])
def test_policy_approval_rejects_executable_change_after_measurement(tmp_path, rename):
    manifest_path, observation_path, repo, _head_commit, _head_tree = _policy_fixture(tmp_path)
    if rename:
        subprocess.run(
            ["git", "-C", str(repo), "config", "diff.renames", "true"], check=True
        )
        (repo / "docs/adr").mkdir(parents=True)
        (repo / "src.py").rename(repo / "docs/adr/renamed-source.md")
    else:
        (repo / "src.py").write_text("print('changed')\n")
    _commit(repo, "changed executable", ".")

    with pytest.raises(ValueError, match="executable or non-Definition bytes"):
        approve_manifest_policy(manifest_path, observation_path, repo)
