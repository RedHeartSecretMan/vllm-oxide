import pytest

from golden_gen.release_performance import summarize_telemetry


def test_raw_synchronized_steps_reproduce_all_latency_samples():
    steps = [
        dict(
            phase="prefill" if i == 0 else "decode",
            started_ns=i * 10,
            ended_ns=(i + 1) * 10,
            prefill_tokens=8 if i == 0 else 0,
            emissions=[dict(request_id=0, completion_step=i, sampled_at_ns=(i + 1) * 10)],
        )
        for i in range(64)
    ]
    result = summarize_telemetry(steps, [0], 8)
    assert result["prefill_tokens_per_second"] == 800_000_000
    assert result["decode_tokens_per_second"] == 100_000_000
    assert result["time_to_first_token_ns"] == [[0, 10]]
    assert result["inter_token_latency_ns"] == [[0, 10]] * 63
    steps[2]["emissions"][0]["completion_step"] = 1
    with pytest.raises(ValueError, match="non-contiguous"):
        summarize_telemetry(steps, [0], 8)


def test_one_request_cannot_emit_63_tokens_in_one_decode_step():
    steps = [
        dict(
            phase="prefill",
            started_ns=0,
            ended_ns=10,
            prefill_tokens=8,
            emissions=[dict(request_id=0, completion_step=0, sampled_at_ns=10)],
        ),
        dict(
            phase="decode",
            started_ns=10,
            ended_ns=20,
            prefill_tokens=0,
            emissions=[
                dict(request_id=0, completion_step=i, sampled_at_ns=20) for i in range(1, 64)
            ],
        ),
    ]
    with pytest.raises(ValueError, match="one token per request"):
        summarize_telemetry(steps, [0], 8)


@pytest.mark.parametrize("noncanonical", [False, True])
def test_supervised_performance_executes_only_the_immutable_measurement_checkout(
    tmp_path, monkeypatch, noncanonical
):
    """A fake producer verifies orchestration only; no GPU acceptance is manufactured."""
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import golden_gen.environment as environment
    import golden_gen.guard as guard
    import golden_gen.release_performance as performance
    import golden_gen.supervision as supervision

    repo, measured_repo = tmp_path / "supervisor", tmp_path / "measurement"
    repo.mkdir()
    measured_repo.mkdir()
    policy = repo / supervision.POLICY_PATH
    policy.parent.mkdir(parents=True)
    policy.write_text('{"synthetic_test":true}')
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    binary = tmp_path / "benchmark"
    binary.write_text("synthetic binary, never executed")
    canonical_binary = binary
    if noncanonical:
        (tmp_path / "indirect").mkdir()
        binary = tmp_path / "indirect" / ".." / "benchmark"
    measured = dict(commit="a" * 40, tree="b" * 40)
    supervisor = dict(commit="c" * 40, tree="d" * 40)
    sources = {repo: supervisor, measured_repo: measured}
    monkeypatch.setattr(performance, "source_identity", lambda path: sources[path])
    runtime = dict(generator_commit=measured["commit"], synthetic_test=True)
    monkeypatch.setattr(
        performance,
        "evaluate_manifest",
        lambda *a, **k: dict(
            verdict="PASS",
            accepting=True,
            source=measured,
            supervision_source=supervisor,
            runtime_profile=runtime,
        ),
    )
    monkeypatch.setattr(performance, "_common_runtime", lambda value: value)
    preflights = []

    def preflight(model, path):
        preflights.append(path)
        assert path == measured_repo
        return SimpleNamespace(model_dump=lambda: runtime)

    monkeypatch.setattr(environment, "collect_release_runtime", preflight)
    context = dict(
        role="benchmark_owner",
        supervision_source=supervisor,
        measurement_invocation=dict(candidate_binary=dict(path=str(canonical_binary))),
    )

    def bind(root, checkout, python, executable, *, binary_kind):
        assert (root, checkout, python, executable, binary_kind) == (
            repo,
            measured_repo,
            sys.executable,
            binary,
            "benchmark",
        )
        return context

    monkeypatch.setattr(supervision, "measurement_context", bind)
    calls = []

    def fake_owner(command, evidence, *, cwd, env, supervision, timeout_seconds):
        calls.append(command)
        assert command[0] == str(canonical_binary)
        assert cwd == measured_repo and supervision is context and timeout_seconds == 1800
        assert env["PYTHONPATH"] == str(measured_repo / "tools/golden-gen/src")
        assert env["UV_PROJECT_ENVIRONMENT"] == sys.prefix
        options = dict(zip(command[1::2], command[2::2], strict=True))
        assert options["--repo-root"] == str(measured_repo)
        assert options["--measurement-commit"] == measured["commit"]
        assert options["--prompts-dir"] == str(measured_repo / "tools/golden-gen/prompts")
        Path(options["--output"]).write_text("{}")
        evidence.write_text("{}")
        for workload in ("canonical_04", "canonical_05"):
            for n in (1, 2, 3):
                (evidence.parent / f"{workload}-repetition-{n}.telemetry.json").write_text("{}")

    monkeypatch.setattr(guard, "run_guarded", fake_owner)
    with pytest.raises(ValueError, match="immutable authoritative measurement checkout"):
        performance.collect_performance(
            repo, tmp_path / "wrong", tmp_path / "model", binary, manifest
        )
    assert not calls and not (tmp_path / "wrong").exists()
    output = performance.collect_performance(
        repo,
        tmp_path / "correct",
        tmp_path / "model",
        binary,
        manifest,
        measurement_repo=measured_repo,
    )
    value = json.loads(output.read_text())
    assert len(calls) == 1 and preflights == [measured_repo, measured_repo]
    assert value["source"] == measured and value["supervision_source"] == supervisor
    assert value["schema_version"] == 2


def test_benchmark_invocation_rejects_unapproved_binary_or_supervisor_code():
    import copy

    from golden_gen.supervision import validate_benchmark_invocation

    source = dict(commit="a" * 40, tree="b" * 40)
    binary = dict(sha256="c" * 64, build_source_id="d" * 40)
    policy = dict(measurement_source=source, measurement_benchmark_binary=binary)
    guard = dict(
        role="benchmark_owner",
        measurement_source=source,
        measurement_invocation=dict(
            repo_root="/measured",
            cwd="/measured",
            pythonpath="/measured/tools/golden-gen/src",
            pythondontwritebytecode="1",
            candidate_binary=dict(path="/binary", **binary),
        ),
        command=[
            "/binary",
            "--model-path",
            "/model",
            "--prompts-dir",
            "/measured/tools/golden-gen/prompts",
            "--output",
            "/output/benchmark.json",
            "--repo-root",
            "/measured",
            "--measurement-commit",
            source["commit"],
            "--measurement-tree",
            source["tree"],
        ],
    )
    validate_benchmark_invocation(
        guard, policy, source, binary["sha256"], binary["build_source_id"]
    )
    wrong = copy.deepcopy(guard)
    wrong["command"][wrong["command"].index("--repo-root") + 1] = "/supervisor"
    with pytest.raises(ValueError, match="differs from measured"):
        validate_benchmark_invocation(
            wrong, policy, source, binary["sha256"], binary["build_source_id"]
        )
    with pytest.raises(ValueError, match="binary"):
        validate_benchmark_invocation(
            guard, {}, source, binary["sha256"], binary["build_source_id"]
        )
    wrong = copy.deepcopy(guard)
    wrong["command"].extend(["--repo-root", "/measured"])
    with pytest.raises(ValueError, match="differs from measured"):
        validate_benchmark_invocation(
            wrong, policy, source, binary["sha256"], binary["build_source_id"]
        )


def test_supervised_performance_rejects_downgrade_and_missing_ram_evidence(tmp_path):
    import json

    from golden_gen.layered_artifacts import sha
    from golden_gen.layered_manifest import _common_runtime
    from golden_gen.layered_release import REGISTRY_PATH, Registry
    from golden_gen.release_performance import validate_performance
    from tests.supervised_evidence_fixture import supervised_release_inputs

    repo, run, measured, evaluator, entries = supervised_release_inputs(tmp_path)
    path = run / entries["performance"]
    wrapper = json.loads(path.read_text())
    registry = Registry.model_validate_json((repo / REGISTRY_PATH).read_text())
    runtime = _common_runtime(wrapper["runtime"])
    roles = dict(source=evaluator, policy_sha256=wrapper["supervision_policy"]["sha256"])
    validate_performance(path, measured, runtime, registry, "e" * 40, supervision=roles)
    original = path.read_bytes()
    legacy = dict(wrapper, schema_version=1)
    del legacy["supervision_source"], legacy["supervision_policy"]
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="cannot downgrade"):
        validate_performance(path, measured, runtime, registry, "e" * 40, supervision=roles)
    path.write_bytes(original)
    guard_path = path.parent / wrapper["guard"]["path"]
    guard = json.loads(guard_path.read_text())
    guard["fast_ram_samples"] = []
    guard_path.write_text(json.dumps(guard))
    wrapper["guard"]["sha256"] = sha(guard_path)
    path.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError, match="RAM sample count"):
        validate_performance(path, measured, runtime, registry, "e" * 40, supervision=roles)
