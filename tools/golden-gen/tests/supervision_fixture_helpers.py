"""Synthetic CPU timelines and bindings; these are never measurement artifacts."""

import json

from golden_gen.layered_artifacts import sha


def synthetic_guard(pid):
    """A short owner with complete before/after queries and active RAM/process samples."""
    ram = 32 * 1024**3
    samples = []
    events = []
    for index, (phase, start, end) in enumerate((("before", 0.0, 0.05), ("after", 0.17, 0.22))):
        samples.append(
            dict(
                started_seconds=start,
                completed_seconds=end,
                elapsed_seconds=end,
                available_ram_bytes=ram,
                disk_free_bytes=1000,
                gpu_memory="0, 1, 2",
                compute_processes="",
            )
        )
        events.append(
            dict(
                attempt=index,
                phase=phase,
                started_seconds=start,
                ended_seconds=end,
                outcome="fresh",
                snapshot_index=index,
                error=None,
                queries=[
                    dict(
                        kind=kind,
                        pid=200 + index * 2 + offset,
                        timeout_seconds=5,
                        started_seconds=start + 0.01 + offset * 0.02,
                        ended_seconds=start + 0.02 + offset * 0.02,
                        outcome="ok",
                        error=None,
                    )
                    for offset, kind in enumerate(("gpu_memory", "compute_processes"))
                ],
            )
        )
    return dict(
        schema_version=2,
        child_pid=pid,
        owned_pgid=pid,
        child_returncode=0,
        failure=None,
        cleanup_failure=None,
        telemetry_cleanup_failure=None,
        remaining_owned_pids=[],
        remaining_telemetry_pids=[],
        ram_poll_interval_ms=100,
        telemetry_interval_ms=1000,
        child_started_seconds=0.06,
        child_exit_observed_seconds=0.15,
        owner_cleanup_completed_seconds=0.16,
        elapsed_seconds=0.25,
        fast_ram_samples=[
            dict(elapsed_seconds=t, available_ram_bytes=ram) for t in (0.0, 0.1, 0.25)
        ],
        ram_sample_count=3,
        maximum_fast_poll_gap_seconds=0.25 - 0.1,
        owned_process_samples=[dict(elapsed_seconds=0.1, pids=[pid])],
        resource_samples=samples,
        telemetry_events=events,
        before=samples[0],
        after=samples[1],
        minimum_available_ram_bytes=ram,
        minimum_disk_free_bytes=1000,
        peak_gpu_used_mib=1,
        minimum_gpu_free_mib=2,
    )


def bind_synthetic_performance(run, entries, policy, measured, evaluator, policy_path):
    path = run / entries["performance"]
    wrapper = json.loads(path.read_text())
    root = path.parent
    local_policy = root / "supervision-policy.json"
    local_policy.write_bytes(policy_path.read_bytes())

    def ref(path):
        return dict(path=path.name, sha256=sha(path))

    raw = json.loads((root / wrapper["raw"]["path"]).read_text())
    guard = synthetic_guard(raw["producer_pid"])
    guard.update(
        role="benchmark_owner",
        measurement_source=measured,
        supervision_source=evaluator,
        supervision_policy_sha256=sha(local_policy),
        measurement_invocation=dict(
            cwd="/measurement",
            repo_root="/measurement",
            pythonpath="/measurement/tools/golden-gen/src",
            pythondontwritebytecode="1",
            python_executable="/python",
            candidate_binary=dict(
                path="/measurement/benchmark", **policy["measurement_benchmark_binary"]
            ),
        ),
        command=[
            "/measurement/benchmark",
            "--model-path",
            "/model",
            "--prompts-dir",
            "/measurement/tools/golden-gen/prompts",
            "--output",
            "/evidence/benchmark.json",
            "--repo-root",
            "/measurement",
            "--measurement-commit",
            measured["commit"],
            "--measurement-tree",
            measured["tree"],
        ],
    )
    guard_path = root / wrapper["guard"]["path"]
    guard_path.write_text(json.dumps(guard))
    wrapper.update(
        schema_version=2,
        guard=ref(guard_path),
        supervision_source=evaluator,
        supervision_policy=ref(local_policy),
        authoritative_manifest_sha256=sha(run / entries["authoritative_manifest"]),
    )
    path.write_text(json.dumps(wrapper))
