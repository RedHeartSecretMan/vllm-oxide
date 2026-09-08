"""Synthetic source/receipt/log inputs for CPU tests, never runtime GPU evidence."""

import json

from golden_gen.layered_artifacts import sha
from golden_gen.release_cpu import PRESSURE, gate_commands
from golden_gen.release_performance import FIXED_OPTIONS, FIXED_SAMPLING
from tests.test_layered_manifest import (
    test_complete_synthetic_three_engine_io_produces_only_nonaccepting_observation,
)


def complete_release_inputs(tmp_path):
    repo, run, source, _ = (
        test_complete_synthetic_three_engine_io_produces_only_nonaccepting_observation(
            tmp_path, release_fixture=True
        )
    )

    def store(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=path.name, sha256=sha(path))

    cpu = run / "cpu"
    cpu.mkdir()
    logs = {
        "default-core": f"test llm::tests::{PRESSURE} ... ok\n"
        "test result: ok. 1 passed; 0 failed; 0 ignored\n",
        "workspace": "test result: ok. 1 passed; 0 failed; 0 ignored\n",
        "internal-golden": "test result: ok. 1 passed; 0 failed; 0 ignored\n",
        "fmt": "",
        "clippy": "",
        "dependencies": "advisories ok, bans ok, licenses ok, sources ok\n",
        "python": "1 passed in 0.01s\n",
        "ruff": "All checks passed!\n",
        "mypy": "Success: no issues found in 1 source file\n",
    }
    gates = []
    for name, command in gate_commands("/synthetic/python").items():
        record = dict(gate=name, command=command, returncode=0)
        for channel, text in (("stdout", logs[name]), ("stderr", "")):
            path = cpu / f"{name}.{channel}.log"
            path.write_text(text)
            record[channel] = dict(path=path.name, sha256=sha(path))
        gates.append(record)
    store(
        cpu / "evidence.json",
        dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            kind="recorded_cpu_gates",
            producer="golden_gen.release_cpu-v1",
            source=source,
            python="/synthetic/python",
            environment=dict(
                CARGO_BUILD_JOBS="1",
                CARGO_NET_OFFLINE="true",
                CARGO_TARGET_DIR="/synthetic/target",
                PYTHONPATH="/synthetic/source",
                GOLDEN_WORKER_TEST_PYTHON="/synthetic/worker",
                GOLDEN_TRANSPORT_TEST_BINARY="/synthetic/binary",
            ),
            toolchain=dict(
                rustc="synthetic rustc", cargo="synthetic cargo", python="synthetic python"
            ),
            gates=gates,
        ),
    )
    performance = run / "benchmark"
    performance.mkdir()
    workloads = {}
    refs = []
    for workload, count in (("canonical_04", 1), ("canonical_05", 4)):
        repetitions = []
        for repetition in (1, 2, 3):
            steps = [
                dict(
                    phase="prefill" if i == 0 else "decode",
                    started_ns=i * 10,
                    ended_ns=(i + 1) * 10,
                    prefill_tokens=count if i == 0 else 0,
                    emissions=[
                        dict(request_id=r, completion_step=i, sampled_at_ns=(i + 1) * 10)
                        for r in range(count)
                    ],
                )
                for i in range(64)
            ]
            telemetry = dict(
                prefill_tokens=count,
                prefill_duration_ns=10,
                prefill_tokens_per_second=count * 1e8,
                decode_tokens=count * 63,
                decode_duration_ns=630,
                decode_tokens_per_second=count * 1e8,
                time_to_first_token_ns=[[r, 10] for r in range(count)],
                inter_token_latency_ns=[[r, 10] for _ in range(63) for r in range(count)],
                steps=steps,
            )
            call = f"{workload}-repetition-{repetition}"
            ref = store(
                performance / (call + ".telemetry.json"),
                dict(
                    format="vllm-oxide-internal-benchmark-json-v2",
                    schema_version=2,
                    call_id=call,
                    request_ids=list(range(count)),
                    binding=dict(
                        prompt_token_ids=[[2]] * count,
                        sampling_params=[FIXED_SAMPLING] * count,
                        engine_options=FIXED_OPTIONS,
                        device="cuda:0",
                        fresh_request_ids=True,
                        cold_prefix_state=True,
                    ),
                    sampled_token_ids=[[0] * 64 for _ in range(count)],
                    outputs=[
                        dict(request_id=r, token_ids=[0] * 64, text="synthetic", finished=True)
                        for r in range(count)
                    ],
                    complete=True,
                    telemetry=telemetry,
                ),
            )
            refs.append(ref)
            repetitions.append(
                dict(
                    prefill_tokens_per_second=count * 1e8,
                    decode_tokens_per_second=count * 1e8,
                    time_to_first_token_ns=[10] * count,
                    inter_token_latency_ns=dict(
                        samples=[10] * (63 * count), mean=10, p50=10, p95=10
                    ),
                    memory=dict(
                        baseline_mib=100,
                        peak_mib=101,
                        delta_mib=1,
                        polling_interval_ms=50,
                        sample_count=2,
                        samples=[
                            dict(elapsed_ms=0, used_mib=100),
                            dict(elapsed_ms=50, used_mib=101),
                        ],
                        other_compute_processes=[],
                    ),
                    telemetry_artifact_sha256=ref["sha256"],
                )
            )
        workloads[workload] = dict(
            discarded_warm_outputs=[
                dict(request_id=r, token_ids=[0] * 64, text="synthetic", finished=True)
                for r in range(count)
            ],
            repetitions=repetitions,
            headline_prefill_tokens_per_second=count * 1e8,
            headline_decode_tokens_per_second=count * 1e8,
            headline_time_to_first_token_ns=10,
            headline_peak_memory_mib=101,
        )
    raw = store(
        performance / "benchmark.json",
        dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            measurement_commit=source["commit"],
            measurement_tree=source["tree"],
            build_source_id="e" * 40,
            cuda_feature_enabled=True,
            producer_pid=987,
            workloads=workloads,
        ),
    )
    guard = store(
        performance / "guard.json",
        dict(
            child_pid=987,
            child_returncode=0,
            failure=None,
            minimum_available_ram_bytes=20 * 1024**3,
            before=dict(compute_processes=""),
            after=dict(compute_processes=""),
        ),
    )
    runtime = json.loads((run / "auth-candidate-primary-receipt.json").read_text())["runtime"]
    store(
        performance / "evidence.json",
        dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            kind="recorded_gpu_performance",
            source=source,
            runtime=runtime,
            binary_sha256="a" * 64,
            raw=raw,
            guard=guard,
            telemetry=refs,
            authoritative_manifest_sha256=sha(run / "authoritative-manifest.json"),
        ),
    )
    return (
        repo,
        run,
        source,
        dict(
            authoritative_manifest="authoritative-manifest.json",
            authoritative_marker="layered-markers/authoritative.complete.json",
            performance="benchmark/evidence.json",
            cpu_gates="cpu/evidence.json",
        ),
    )
