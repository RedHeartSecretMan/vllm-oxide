"""Synthetic R/W protocol graph, never real role approval or GPU evidence."""

import copy
import json

import pytest

from golden_gen.layered_artifacts import sha
from golden_gen.layered_manifest import _common_runtime
from golden_gen.layered_release import REGISTRY_PATH, Registry
from golden_gen.release_memory_method import CALL_IDS
from golden_gen.release_performance import validate_performance, validate_role_performance
from golden_gen.release_roles import RolePolicy, SourceRoles
from tests.supervised_evidence_fixture import supervised_release_inputs
from tests.test_release_memory_method import method_record


def role_performance(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo, run, x, y, entries = supervised_release_inputs(tmp_path)
    path = run / entries["performance"]
    old = json.loads(path.read_text())
    root = path.parent
    sources = SourceRoles(
        numerical_measurement=x,
        numerical_evaluator=y,
        performance_producer=dict(commit="c" * 40, tree="c" * 40),
        release_evaluator=dict(commit="d" * 40, tree="d" * 40),
    )
    policy = RolePolicy.model_validate(
        dict(
            protocol="layered-release-source-roles-v1",
            schema_version=1,
            status="approved",
            numerical=dict(
                measurement=x,
                evaluator=y,
                calibration=x,
                policy_sha256="1" * 64,
                registry_sha256="2" * 64,
                manifest_sha256=sha(run / entries["authoritative_manifest"]),
                result_sha256="3" * 64,
                marker_sha256="4" * 64,
                original_cpu_roles_sha256="5" * 64,
            ),
            performance_producer=sources.performance_producer.model_dump(),
            benchmark_binary=dict(sha256=old["binary_sha256"], build_source_id="e" * 40),
            producer_changes=[
                dict(
                    path="crates/vllm-oxide-test/src/benchmark/nvml_monitor.rs",
                    before=None,
                    after=dict(mode="100644", kind="blob", oid="f" * 40),
                )
            ],
            monitor_method="direct-nvml-v1",
            monitor_schema_version=1,
        )
    )
    scope = root / "role-policy.json"
    scope.write_text(policy.model_dump_json())

    def ref(p):
        return dict(path=p.relative_to(root).as_posix(), sha256=sha(p))

    rawpath = root / old["raw"]["path"]
    raw = json.loads(rawpath.read_text())
    raw.update(measurement_commit="c" * 40, measurement_tree="c" * 40)
    guardpath = root / old["guard"]["path"]
    guard = json.loads(guardpath.read_text())
    guard.update(
        measurement_source=sources.performance_producer.model_dump(),
        supervision_source=sources.release_evaluator.model_dump(),
        supervision_policy_sha256=sha(scope),
    )
    guard["command"][guard["command"].index("--measurement-commit") + 1] = "c" * 40
    guard["command"][guard["command"].index("--measurement-tree") + 1] = "c" * 40
    methods = []
    events = []
    for wnumber, w in enumerate(("canonical_04", "canonical_05")):
        base = wnumber * 500_000_000
        for i, phase in enumerate(
            (
                "discarded_warm_llm_new_entered",
                "discarded_warm_llm_new_returned",
                "discarded_warm_generate_completed",
                "discarded_warm_dropped",
            )
        ):
            events.append(dict(call_id=w, phase=phase, elapsed_ns=base + i * 1_000_000))
        for n in (1, 2, 3):
            call = f"{w}-repetition-{n}"
            offset = base + n * 100_000_000
            record = method_record()
            record.update(call_id=call, build_source_id="e" * 40, benchmark_pid=raw["producer_pid"])
            record["trace"]["library"]["driver_version"] = old["runtime"]["nvidia_driver_version"]
            for key in record["times"]:
                record["times"][key] += offset
            for key in (
                "baseline_observed_ns",
                "finish_requested_ns",
                "final_sample_enter_ns",
                "final_sample_return_ns",
                "final_sample_observed_ns",
                "worker_finished_ns",
                "shutdown_enter_ns",
                "shutdown_return_ns",
            ):
                record["trace"][key] += offset
            for sample in record["trace"]["raw_samples"]:
                for key in ("api_enter_ns", "api_return_ns", "observed_ns"):
                    sample[key] += offset
            p = root / f"{call}.memory-method/record.json"
            p.parent.mkdir()
            p.write_text(json.dumps(record))
            methods.append(ref(p))
            phases = (
                ("measured_llm_new_entered", 0),
                ("measured_llm_new_returned", 500_000),
                ("measured_generate_entered", 6_900_000),
                ("measured_generate_returned", 12_150_000),
                ("measured_generate_succeeded_after_internal_final_step_sync", 12_300_000),
                ("final_memory_sample_and_worker_cleanup_completed", 19_100_000),
            )
            events.extend(
                dict(call_id=call, phase=phase, elapsed_ns=offset + value)
                for phase, value in phases
            )
            raw["workloads"][w]["repetitions"][n - 1]["memory"] = dict(
                polling_interval_ms=6,
                sample_count=3,
                baseline_mib=2,
                peak_mib=3,
                delta_mib=1,
                samples=[
                    dict(elapsed_ms=0, used_mib=2),
                    dict(elapsed_ms=6, used_mib=3),
                    dict(elapsed_ms=12, used_mib=2),
                ],
                other_compute_processes=[],
            )
        raw["workloads"][w]["headline_peak_memory_mib"] = 3
    rawpath.write_text(json.dumps(raw))
    guardpath.write_text(json.dumps(guard))
    journal = root / "benchmark-stages.json"
    journal.write_text(
        json.dumps(
            dict(
                protocol="memory-method-stage-record-v1",
                schema_version=1,
                method="direct-nvml-v1",
                record_only=True,
                accepting=False,
                completed=True,
                clock="shared Rust Instant origin with every NVML monitor trace",
                events=events,
                build_source_id="e" * 40,
            )
        )
    )
    runtime = dict(old["runtime"], generator_commit="c" * 40)
    wrapper = dict(
        protocol="layered-accuracy-v1",
        schema_version=3,
        kind="recorded_role_gpu_performance",
        sources=sources.model_dump(),
        runtime=runtime,
        binary_sha256=old["binary_sha256"],
        build_source_id="e" * 40,
        authoritative_manifest_sha256=policy.numerical.manifest_sha256,
        source_role_policy=ref(scope),
        raw=ref(rawpath),
        guard=ref(guardpath),
        telemetry=old["telemetry"],
        memory_methods=methods,
        stage_journal=ref(journal),
    )
    path.write_text(json.dumps(wrapper))
    registry = Registry.model_validate_json((repo / REGISTRY_PATH).read_text())
    return path, sources, policy, _common_runtime(runtime), registry, sha(scope), wrapper


def test_role_performance_full_six_record_graph_and_legacy_refusal(tmp_path):
    path, roles, policy, runtime, registry, digest, wrapper = role_performance(tmp_path)
    raw, closure = validate_role_performance(path, roles, policy, runtime, registry, digest)
    assert raw["measurement_commit"] == roles.performance_producer.commit
    assert all(path.parent / f"{call}.memory-method/record.json" in closure for call in CALL_IDS)
    with pytest.raises(ValueError, match="source/runtime/provenance"):
        validate_performance(
            path, roles.numerical_measurement.model_dump(), runtime, registry, "e" * 40
        )
    broken = copy.deepcopy(wrapper)
    broken["memory_methods"][0]["path"] = "arbitrary-method.json"
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="six fixed"):
        validate_role_performance(path, roles, policy, runtime, registry, digest)
