from pathlib import Path

from golden_gen.layered_artifacts import sha
from golden_gen.layered_release import definition_document


def test_supervision_policy_uses_selected_definition_binding():
    repo = Path(__file__).resolve().parents[3]
    relative = "docs/validation/layered-supervision-policy.json"
    policy, digest = definition_document(repo, relative)
    assert policy["policy_id"] == "bounded-telemetry-recovery-v1"
    assert digest == sha(repo / relative)
    from golden_gen.supervision import load_policy

    validated, validated_digest = load_policy(repo)
    assert validated == policy and validated_digest == digest


def test_equivalence_rejects_changed_measurement_blob(tmp_path):
    import subprocess

    import pytest

    from golden_gen.supervision import validate_equivalence

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "CPU test")
    (tmp_path / "kernel.rs").write_text("original\n")
    git("add", ".")
    git("commit", "-qm", "measurement")
    measured = dict(commit=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"))
    policy = dict(measurement_source=measured, unchanged_measurement_paths=["kernel.rs"])
    (tmp_path / "supervisor.py").write_text("identity only\n")
    git("add", ".")
    git("commit", "-qm", "supervisor")
    current = dict(commit=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"))
    validate_equivalence(tmp_path, policy, current)
    (tmp_path / "kernel.rs").write_text("changed\n")
    git("add", ".")
    git("commit", "-qm", "invalid changed model")
    changed = dict(commit=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"))
    with pytest.raises(ValueError, match="protected"):
        validate_equivalence(tmp_path, policy, changed)


def test_retained_ledger_requires_bound_guard_bytes(tmp_path):
    import json

    import pytest

    from golden_gen.supervision import validate_retained_ledger

    source = dict(commit="a" * 40, tree="b" * 40)
    directory = tmp_path / "case-candidate-primary"
    directory.mkdir()
    (directory / "capture.json").write_text("{}")
    sample = dict(available_ram_bytes=32 * 1024**3, compute_processes="", gpu_memory="0, 1, 2")
    guard = dict(
        schema_version=1,
        child_returncode=0,
        child_pid=123,
        failure=None,
        before=sample,
        after=sample,
        minimum_available_ram_bytes=32 * 1024**3,
        cleanup_failure=None,
        remaining_owned_pids=[],
        resource_samples=[sample, sample],
    )
    (directory / "guard.json").write_text(json.dumps(guard))

    def ref(path):
        return dict(path=str(path.relative_to(tmp_path)), sha256=sha(path))

    (directory / "receipt.json").write_text(
        json.dumps(dict(source=source, driver_pid=123, guard=ref(directory / "guard.json")))
    )
    entry = dict(
        index=0,
        owner_key=["execution_group", "case", "candidate", "primary"],
        capture=ref(directory / "capture.json"),
        receipt=ref(directory / "receipt.json"),
        guard=ref(directory / "guard.json"),
        files=[ref(path) for path in sorted(directory.iterdir())],
    )
    ledger = dict(
        protocol="layered-retained-owner-ledger-v1",
        schema_version=1,
        measurement_source=source,
        registry_sha256="c" * 64,
        policy_sha256="d" * 64,
        owner_count=1,
        owners=[entry],
    )
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger))
    policy = dict(
        measurement_source=source,
        registry_sha256="c" * 64,
        numerical_policy_sha256="d" * 64,
        retained_ledger_sha256=sha(path),
        retained_owner_count=1,
        retained_index_start=0,
        retained_index_end_inclusive=0,
    )
    inventory = [
        dict(
            kind="execution_group", execution_group_id="case", engine="candidate", variant="primary"
        )
    ]
    validate_retained_ledger(tmp_path, ref(path), policy, inventory)
    (directory / "guard.json").write_text(json.dumps(dict(guard, failure="timed out")))
    with pytest.raises(ValueError, match="checksum"):
        validate_retained_ledger(tmp_path, ref(path), policy, inventory)


def test_manifest_two_requires_separate_supervision_identity():
    import pytest

    from golden_gen.layered_manifest import LayeredManifest

    artifact = dict(path="evidence.json", sha256="a" * 64)
    owner = dict(
        execution_group_id="g",
        engine="candidate",
        **{
            name: artifact
            for name in (
                "primary",
                "replay",
                "control",
                "primary_receipt",
                "replay_receipt",
                "control_receipt",
            )
        },
    )
    data = dict(
        protocol="layered-accuracy-v1",
        schema_version=2,
        source=dict(commit="a" * 40, tree="b" * 40),
        registry_sha256="c" * 64,
        policy_sha256="d" * 64,
        purpose="authoritative",
        captures=[owner],
        evaluator_source=dict(commit="e" * 40, tree="f" * 40),
        supervision_source=dict(commit="e" * 40, tree="f" * 40),
        supervision_policy=artifact,
        retained_owner_ledger=artifact,
    )
    parsed = LayeredManifest.model_validate(data)
    assert parsed.source != parsed.evaluator_source
    del data["supervision_source"]
    with pytest.raises(ValueError):
        LayeredManifest.model_validate(data)


def test_recorded_worker_invocation_cannot_point_at_supervisor_code():
    import pytest

    from golden_gen.supervision import validate_invocation

    source = dict(commit="a" * 40, tree="b" * 40)
    root = "/immutable/measurement"
    invocation = dict(
        cwd=root,
        repo_root=root,
        pythonpath=root + "/tools/golden-gen/src",
        pythondontwritebytecode="1",
        python_executable="/python",
        candidate_binary=None,
    )
    command = [
        "/python",
        "-m",
        "golden_gen.layered_cli",
        "worker",
        "--repo-root",
        root,
        "--run-dir",
        "/evidence",
        "--model-dir",
        "/model",
        "--group",
        "g",
        "--engine",
        "baseline",
        "--variant",
        "primary",
    ]
    record = dict(
        role="measurement_owner",
        measurement_source=source,
        measurement_invocation=invocation,
        command=command,
        child_pid=12,
        owned_pgid=12,
    )
    receipt = dict(source=source, driver_pid=12, engine="baseline")
    validate_invocation(
        record,
        dict(measurement_source=source),
        ("execution_group", "g", "baseline", "primary"),
        receipt,
    )
    record["measurement_invocation"]["pythonpath"] = "/new-supervisor/tools/golden-gen/src"
    with pytest.raises(ValueError, match="invocation"):
        validate_invocation(
            record,
            dict(measurement_source=source),
            ("execution_group", "g", "baseline", "primary"),
            receipt,
        )


def test_new_worker_metadata_must_match_the_original_receipt(tmp_path):
    import json

    import pytest

    from golden_gen.layered_artifacts import worker_metadata

    receipt = dict(source=dict(commit="a" * 40, tree="b" * 40), driver_pid=12, guard={})
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="worker metadata"):
        worker_metadata(tmp_path, path, receipt)
    worker = tmp_path / "worker.json"
    worker.write_text(json.dumps({key: value for key, value in receipt.items() if key != "guard"}))
    assert worker_metadata(tmp_path, path, receipt) == worker
    worker.write_text("{}")
    with pytest.raises(ValueError, match="worker metadata"):
        worker_metadata(tmp_path, path, receipt)
