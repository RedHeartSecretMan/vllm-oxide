"""Recompute ADR-0012 performance evidence from original synchronized samples."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from golden_gen import config
from golden_gen.layered_accuracy import PROTOCOL
from golden_gen.layered_artifacts import atomic_json, bound_file, sha, source_identity
from golden_gen.layered_manifest import Artifact, _common_runtime, evaluate_manifest
from golden_gen.layered_release import Registry
from golden_gen.release_roles import RolePolicy, SourceRoles, StrictRecord
from golden_gen.release_transport import Entrypoints

if TYPE_CHECKING:
    from golden_gen.release_roles import FreezeBinding


class RolePerformanceWrapper(StrictRecord):
    protocol: Literal["layered-accuracy-v1"]
    schema_version: Literal[3]
    kind: Literal["recorded_role_gpu_performance"]
    sources: SourceRoles
    runtime: dict[str, Any]
    binary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    build_source_id: str = Field(pattern=r"^[0-9a-f]{40}$")
    authoritative_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_role_policy: Artifact
    raw: Artifact
    guard: Artifact
    telemetry: list[Artifact]
    memory_methods: list[Artifact]
    stage_journal: Artifact


FIXED_OPTIONS = dict(
    max_num_batched_tokens=16384,
    max_num_seqs=512,
    max_model_len=4096,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
    dtype="BF16",
)
FIXED_SAMPLING = dict(
    temperature=0,
    top_k=None,
    top_p=None,
    max_tokens=64,
    ignore_eos=True,
    presence_penalty=0,
    frequency_penalty=0,
    repetition_penalty=0,
)


def _public_tokens(outputs: list[dict[str, Any]], count: int) -> list[list[int]]:
    if len(outputs) != count:
        raise ValueError("performance public output count mismatch")
    for i, output in enumerate(outputs):
        if (
            output.get("request_id") != i
            or output.get("finished") is not True
            or not isinstance(output.get("text"), str)
            or len(output.get("token_ids", [])) != 64
            or any(
                type(t) is not int or not 0 <= t < config.VOCAB_SIZE for t in output["token_ids"]
            )
        ):
            raise ValueError("performance public token/request identity mismatch")
    return [output["token_ids"] for output in outputs]


def _integer(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("performance sample must be a nonnegative integer")
    return value


def summarize_telemetry(
    steps: list[dict[str, Any]], requests: list[int], prompt_tokens: int
) -> dict[str, Any]:
    if not requests or requests != list(range(len(requests))):
        raise ValueError("performance request identity/order mismatch")
    histories: dict[int, list[int]] = {r: [] for r in requests}
    previous_end = prefill = prefill_ns = decode_ns = decode = 0
    inter = []
    for step in steps:
        requests_in_step = [e["request_id"] for e in step["emissions"]]
        if len(set(requests_in_step)) != len(requests_in_step):
            raise ValueError("performance permits only one token per request in a step")
        start, end, count = (
            _integer(step[k]) for k in ("started_ns", "ended_ns", "prefill_tokens")
        )
        if (
            start < previous_end
            or end <= start
            or step["phase"] not in ("prefill", "decode", "mixed")
        ):
            raise ValueError("invalid synchronized performance step")
        if count and step["phase"] == "decode":
            raise ValueError("decode step claims prefill tokens")
        previous_end = end
        if count:
            prefill += count
            prefill_ns += end - start
        if any(_integer(e["completion_step"]) > 0 for e in step["emissions"]):
            decode_ns += end - start
        for emission in step["emissions"]:
            request, completion, sampled = (
                _integer(emission[k]) for k in ("request_id", "completion_step", "sampled_at_ns")
            )
            if request not in histories or completion != len(histories[request]) or sampled != end:
                raise ValueError("non-contiguous or unsynchronized performance emission")
            if completion:
                inter.append([request, sampled - histories[request][-1]])
                decode += 1
            histories[request].append(sampled)
    if (
        any(len(h) != 64 for h in histories.values())
        or prefill != prompt_tokens
        or not prefill_ns
        or not decode_ns
    ):
        raise ValueError("performance lacks cold prompt work or exactly64 completion tokens")
    return dict(
        prefill_tokens=prefill,
        prefill_duration_ns=prefill_ns,
        prefill_tokens_per_second=prefill * 1e9 / prefill_ns,
        decode_tokens=decode,
        decode_duration_ns=decode_ns,
        decode_tokens_per_second=decode * 1e9 / decode_ns,
        time_to_first_token_ns=[[r, histories[r][0]] for r in requests],
        inter_token_latency_ns=inter,
        steps=steps,
    )


def _percentile(samples: list[int], q: float) -> float:
    values = sorted(samples)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def validate_performance(
    path: Path,
    source: dict[str, str],
    runtime: dict[str, Any],
    registry: Registry,
    build_source_id: str,
    *,
    supervision: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    wrapper = json.loads(path.read_text())
    if (
        wrapper.get("protocol") != PROTOCOL
        or wrapper.get("schema_version") not in (1, 2)
        or wrapper.get("kind") != "recorded_gpu_performance"
        or wrapper.get("source") != source
        or _common_runtime(wrapper["runtime"]) != runtime
        or wrapper["runtime"].get("generator_commit") != source["commit"]
    ):
        raise ValueError("performance source/runtime/provenance mismatch")
    root = path.parent
    raw_path, guard_path = (bound_file(root, wrapper[k]) for k in ("raw", "guard"))
    raw, guard = (json.loads(p.read_text()) for p in (raw_path, guard_path))
    if (
        raw.get("protocol") != PROTOCOL
        or raw.get("schema_version") != 1
        or raw.get("measurement_commit") != source["commit"]
        or raw.get("measurement_tree") != source["tree"]
        or raw.get("cuda_feature_enabled") is not True
        or raw.get("build_source_id") != build_source_id
        or type(raw.get("producer_pid")) is not int
        or raw["producer_pid"] <= 0
        or raw["producer_pid"] != guard.get("child_pid")
        or guard.get("child_returncode") != 0
        or guard.get("failure") is not None
        or guard.get("minimum_available_ram_bytes", 0) < 16 * 1024**3
        or guard.get("before", {}).get("compute_processes") != ""
        or guard.get("after", {}).get("compute_processes") != ""
    ):
        raise ValueError("performance lacks actual guarded CUDA/source evidence")
    if set(raw["workloads"]) != {"canonical_04", "canonical_05"}:
        raise ValueError("performance workload inventory differs")
    closure = [path, raw_path, guard_path]
    if wrapper["schema_version"] == 2:
        from golden_gen.guard_evidence import validate_guard_timeline
        from golden_gen.supervision import validate_benchmark_invocation

        if supervision is None:
            raise ValueError("performance supervision lacks authoritative bindings")
        policy_path = bound_file(root, wrapper["supervision_policy"])
        digest = sha(policy_path)
        if (
            wrapper.get("supervision_source") != supervision["source"]
            or digest != supervision["policy_sha256"]
            or guard.get("supervision_source") != supervision["source"]
            or guard.get("supervision_policy_sha256") != digest
        ):
            raise ValueError("performance supervision differs from authoritative evidence")
        validate_guard_timeline(guard)
        validate_benchmark_invocation(
            guard,
            json.loads(policy_path.read_text()),
            source,
            wrapper["binary_sha256"],
            build_source_id,
        )
        closure.append(policy_path)
    elif supervision is not None or any(
        k in wrapper for k in ("supervision_source", "supervision_policy")
    ):
        raise ValueError("supervised performance cannot downgrade to legacy evidence")
    closure.extend(_validate_samples(root, raw, wrapper["telemetry"], registry))
    return raw, closure


def _validate_samples(
    root: Path, raw: dict[str, Any], telemetry_refs: list[dict[str, str]], registry: Registry
) -> list[Path]:
    closure: list[Path] = []
    members = {m.case_id: m for group in registry.numerical_cases for m in group.plan.members}
    expected_names = [
        f"{w}-repetition-{r}.telemetry.json"
        for w in ("canonical_04", "canonical_05")
        for r in (1, 2, 3)
    ]
    if [r["path"] for r in telemetry_refs] != expected_names:
        raise ValueError("performance telemetry inventory differs")
    for workload, value in raw["workloads"].items():
        ids = (
            ["canonical_04"] if workload == "canonical_04" else [f"canonical_05{x}" for x in "abcd"]
        )
        prompt_tokens = sum(len(members[i].prompt) for i in ids)
        _public_tokens(value["discarded_warm_outputs"], len(ids))
        repetitions = value["repetitions"]
        if len(repetitions) != 3:
            raise ValueError("performance needs three measured repetitions")
        for number, repetition in enumerate(repetitions, 1):
            name = f"{workload}-repetition-{number}.telemetry.json"
            ref = next(r for r in telemetry_refs if r["path"] == name)
            telemetry_path = bound_file(root, ref)
            closure.append(telemetry_path)
            if repetition["telemetry_artifact_sha256"] != sha(telemetry_path):
                raise ValueError("performance summary lost its raw telemetry hash")
            document = json.loads(telemetry_path.read_text())
            if (
                document.get("format") != "vllm-oxide-internal-benchmark-json-v2"
                or document.get("schema_version") != 2
                or document.get("complete") is not True
                or document.get("call_id") != name.removesuffix(".telemetry.json")
                or document.get("request_ids") != list(range(len(ids)))
            ):
                raise ValueError("performance telemetry identity mismatch")
            binding = document.get("binding", {})
            if (
                binding.get("prompt_token_ids") != [members[i].prompt for i in ids]
                or binding.get("engine_options") != FIXED_OPTIONS
                or binding.get("sampling_params") != [FIXED_SAMPLING] * len(ids)
                or binding.get("device") != "cuda:0"
                or binding.get("fresh_request_ids") is not True
                or binding.get("cold_prefix_state") is not True
            ):
                raise ValueError(
                    "performance actual token/configuration/cold-state binding differs"
                )
            if _public_tokens(document["outputs"], len(ids)) != document.get("sampled_token_ids"):
                raise ValueError(
                    "performance public outputs differ from actual sampled token stream"
                )
            recomputed = summarize_telemetry(
                document["telemetry"]["steps"], document["request_ids"], prompt_tokens
            )
            if recomputed != document["telemetry"]:
                raise ValueError("performance derived values differ from raw timestamps")
            intervals = [n for _, n in recomputed["inter_token_latency_ns"]]
            expected = dict(
                samples=intervals,
                mean=sum(intervals) / len(intervals),
                p50=_percentile(intervals, 0.5),
                p95=_percentile(intervals, 0.95),
            )
            if (
                repetition["inter_token_latency_ns"] != expected
                or repetition["time_to_first_token_ns"]
                != [n for _, n in recomputed["time_to_first_token_ns"]]
                or any(
                    repetition[k] != recomputed[k]
                    for k in ("prefill_tokens_per_second", "decode_tokens_per_second")
                )
            ):
                raise ValueError("performance summary latency/rate differs")
            memory = repetition["memory"]
            samples = memory["samples"]
            interval = _integer(memory["polling_interval_ms"])
            if (
                not 1 <= interval <= 50
                or len(samples) < 2
                or memory["sample_count"] != len(samples)
                or memory["other_compute_processes"]
            ):
                raise ValueError("performance memory monitoring incomplete")
            times = [_integer(s["elapsed_ms"]) for s in samples]
            used = [_integer(s["used_mib"]) for s in samples]
            if (
                times[0] != 0
                or any(not 0 < b - a <= interval for a, b in zip(times, times[1:], strict=False))
                or times[-1] + interval < recomputed["steps"][-1]["ended_ns"] / 1e6
                or memory["baseline_mib"] != used[0]
                or memory["peak_mib"] != max(used)
                or memory["delta_mib"] != max(used) - used[0]
            ):
                raise ValueError("performance memory timing/peak differs")
        headlines = dict(
            headline_prefill_tokens_per_second=median(
                r["prefill_tokens_per_second"] for r in repetitions
            ),
            headline_decode_tokens_per_second=median(
                r["decode_tokens_per_second"] for r in repetitions
            ),
            headline_time_to_first_token_ns=median(
                n for r in repetitions for n in r["time_to_first_token_ns"]
            ),
            headline_peak_memory_mib=median(r["memory"]["peak_mib"] for r in repetitions),
        )
        if any(value[k] != v for k, v in headlines.items()):
            raise ValueError("performance headline differs from raw medians")
    return closure


def validate_role_performance(
    path: Path,
    roles: SourceRoles,
    policy: RolePolicy,
    runtime: dict[str, Any],
    registry: Registry,
    policy_sha256: str,
) -> tuple[dict[str, Any], list[Path]]:
    """New explicit R/W route; no old SUP or X capture fingerprint projection."""
    from golden_gen.guard_evidence import validate_guard_timeline
    from golden_gen.release_memory_method import CALL_IDS, validate_memory_method
    from golden_gen.supervision import validate_role_benchmark_invocation

    wrapper = RolePerformanceWrapper.model_validate_json(path.read_text())
    root = path.parent
    if (
        wrapper.sources != roles
        or wrapper.binary_sha256 != policy.benchmark_binary.sha256
        or wrapper.build_source_id != policy.benchmark_binary.build_source_id
        or wrapper.authoritative_manifest_sha256 != policy.numerical.manifest_sha256
        or _common_runtime(wrapper.runtime) != runtime
        or wrapper.runtime.get("generator_commit") != roles.performance_producer.commit
    ):
        raise ValueError("role performance actual sources/binary/runtime differ")
    scope = bound_file(root, wrapper.source_role_policy.model_dump())
    if sha(scope) != policy_sha256 or RolePolicy.model_validate_json(scope.read_text()) != policy:
        raise ValueError("role performance source-role Definition differs")
    raw_path, guard_path = (
        bound_file(root, artifact.model_dump()) for artifact in (wrapper.raw, wrapper.guard)
    )
    raw, guard = (json.loads(p.read_text()) for p in (raw_path, guard_path))
    if (
        raw.get("protocol") != PROTOCOL
        or type(raw.get("schema_version")) is not int
        or raw["schema_version"] != 1
        or raw.get("measurement_commit") != roles.performance_producer.commit
        or raw.get("measurement_tree") != roles.performance_producer.tree
        or raw.get("build_source_id") != policy.benchmark_binary.build_source_id
        or raw.get("cuda_feature_enabled") is not True
        or type(raw.get("producer_pid")) is not int
        or raw["producer_pid"] <= 0
        or raw["producer_pid"] != guard.get("child_pid")
        or set(raw.get("workloads", {})) != {"canonical_04", "canonical_05"}
    ):
        raise ValueError("role performance native actual producer differs")
    validate_guard_timeline(guard)
    validate_role_benchmark_invocation(guard, roles, policy.benchmark_binary, policy_sha256)
    closure = [
        path,
        scope,
        raw_path,
        guard_path,
        *_validate_samples(root, raw, [a.model_dump() for a in wrapper.telemetry], registry),
    ]
    expected = [f"{call}.memory-method/record.json" for call in CALL_IDS]
    if [artifact.path for artifact in wrapper.memory_methods] != expected:
        raise ValueError("role performance six fixed memory-method paths differ")
    records = []
    for call, artifact in zip(CALL_IDS, wrapper.memory_methods, strict=True):
        method_path = bound_file(root, artifact.model_dump())
        if any(
            (method_path.parent / name).exists()
            for name in ("worker-timeout.json", "preservation-error.json")
        ):
            raise ValueError("role performance contains failed method preservation/worker state")
        record = json.loads(method_path.read_text())
        recomputed = validate_memory_method(
            record,
            call,
            roles.performance_producer.model_dump(),
            policy.benchmark_binary.build_source_id,
            raw["producer_pid"],
        )
        workload, repetition = call.split("-repetition-")
        if recomputed != raw["workloads"][workload]["repetitions"][int(repetition) - 1]["memory"]:
            raise ValueError(
                "role performance legacy memory does not reproduce from R raw method bytes"
            )
        if record["trace"]["library"]["driver_version"] != wrapper.runtime["nvidia_driver_version"]:
            raise ValueError("R actual memory library driver differs from runtime")
        telemetry_ref = next(a for a in wrapper.telemetry if a.path == f"{call}.telemetry.json")
        telemetry = json.loads(bound_file(root, telemetry_ref.model_dump()).read_text())
        final_step_end = telemetry["telemetry"]["steps"][-1]["ended_ns"]
        times = record["times"]
        # llm.rs creates admission inside LLM::generate and records ended_ns
        # after step-end synchronization. Its relative final time must fit in
        # this call's measured duration; the two clocks have different origins.
        if times["generate_returned_ns"] - times["generate_invoked_ns"] < final_step_end:
            raise ValueError("R generate duration does not cover same-call synchronized telemetry")
        records.append(record)
        closure.append(method_path)
    journal_path = bound_file(root, wrapper.stage_journal.model_dump())
    if wrapper.stage_journal.path != "benchmark-stages.json":
        raise ValueError("role performance actual stage journal path differs")
    _validate_method_stages(
        json.loads(journal_path.read_text()), records, policy.benchmark_binary.build_source_id
    )
    closure.append(journal_path)
    return raw, closure


def _validate_method_stages(
    journal: dict[str, Any], records: list[dict[str, Any]], build_id: str
) -> None:
    from golden_gen.release_memory_method import CALL_IDS

    if (
        journal.get("protocol") != "memory-method-stage-record-v1"
        or type(journal.get("schema_version")) is not int
        or journal["schema_version"] != 1
        or journal.get("method") != "direct-nvml-v1"
        or journal.get("record_only") is not True
        or journal.get("accepting") is not False
        or journal.get("completed") is not True
        or journal.get("build_source_id") != build_id
        or journal.get("clock") != "shared Rust Instant origin with every NVML monitor trace"
    ):
        raise ValueError("R stage journal method/source/completion differs")
    expected: list[tuple[str, str]] = []
    for workload in ("canonical_04", "canonical_05"):
        expected.extend(
            (workload, phase)
            for phase in (
                "discarded_warm_llm_new_entered",
                "discarded_warm_llm_new_returned",
                "discarded_warm_generate_completed",
                "discarded_warm_dropped",
            )
        )
        for call in (call for call in CALL_IDS if call.startswith(workload)):
            expected.extend(
                (call, phase)
                for phase in (
                    "measured_llm_new_entered",
                    "measured_llm_new_returned",
                    "measured_generate_entered",
                    "measured_generate_returned",
                    "measured_generate_succeeded_after_internal_final_step_sync",
                    "final_memory_sample_and_worker_cleanup_completed",
                )
            )
    events = journal["events"]
    if [(e["call_id"], e["phase"]) for e in events] != expected:
        raise ValueError("R stage journal does not cover fixed discarded/measured order")
    timestamps = [_integer(e["elapsed_ns"]) for e in events]
    if any(b < a for a, b in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("R stage journal clock is nonmonotonic")
    for record in records:
        phases = {
            event["phase"]: event["elapsed_ns"]
            for event in events
            if event["call_id"] == record["call_id"]
        }
        times = record["times"]
        if not (
            phases["measured_llm_new_returned"]
            <= times["post_initialization_monitor_started_ns"]
            <= times["baseline_ready_ns"]
            <= phases["measured_generate_entered"]
            <= times["generate_invoked_ns"]
            <= times["generate_returned_ns"]
            <= phases["measured_generate_returned"]
            <= times["generate_succeeded_after_internal_final_step_sync_ns"]
            <= phases["measured_generate_succeeded_after_internal_final_step_sync"]
            <= times["finish_requested_ns"]
            <= times["worker_join_completed_ns"]
            <= phases["final_memory_sample_and_worker_cleanup_completed"]
        ):
            raise ValueError(
                "R actual measured window does not cover init/baseline/inference/final sync"
            )


def collect_performance(
    repo: Path,
    directory: Path,
    model: Path,
    binary: Path,
    authoritative_manifest: Path,
    *,
    measurement_repo: Path | None = None,
    freeze: FreezeBinding | None = None,
    evidence_root: Path | None = None,
    entrypoints: Entrypoints | None = None,
) -> Path:
    """GPU-owning stage, called only with separate stage authority; never during CPU tests."""
    from golden_gen.environment import collect_release_runtime
    from golden_gen.guard import run_guarded

    repo, directory, model, binary, authoritative_manifest = (
        path.absolute() for path in (repo, directory, model, binary, authoritative_manifest)
    )
    prepared = None
    if freeze is not None:
        from golden_gen.release_roles import prepare_performance_binding

        if evidence_root is None or entrypoints is None:
            raise ValueError("role performance requires the complete numerical/CPU entrypoints")
        if (
            evidence_root / entrypoints.authoritative_manifest
        ).resolve() != authoritative_manifest.resolve():
            raise ValueError("role performance predecessor path differs")
        prepared = prepare_performance_binding(repo, evidence_root, entrypoints, freeze=freeze)
        result = prepared["numerical"]
    else:
        result = evaluate_manifest(repo, authoritative_manifest, authoritative=True)
    if result.get("verdict") != "PASS" or result.get("accepting") is not True:
        raise ValueError("performance requires complete approved authoritative evidence")
    supervisor = source_identity(repo)
    execution_repo = (measurement_repo or repo).resolve()
    source = source_identity(execution_repo)
    if prepared is None and source != result["source"]:
        raise ValueError("performance requires the immutable authoritative measurement checkout")
    context = None
    if prepared is not None:
        from golden_gen.supervision import performance_role_context

        context = performance_role_context(
            repo, execution_repo, sys.executable, binary, freeze=freeze
        )
        binary = Path(context["measurement_invocation"]["candidate_binary"]["path"])
    elif result.get("supervision_source") is not None:
        from golden_gen.supervision import measurement_context

        context = measurement_context(
            repo, execution_repo, sys.executable, binary, binary_kind="benchmark"
        )
        if context["supervision_source"] != result["supervision_source"]:
            raise ValueError("benchmark supervisor differs from authoritative evidence")
        # Check the caller's original path first (including final symlinks),
        # then execute the exact canonical path retained in the approved binding.
        binary = Path(context["measurement_invocation"]["candidate_binary"]["path"])
    runtime = collect_release_runtime(model, execution_repo).model_dump()
    if _common_runtime(runtime) != result["runtime_profile"]:
        raise ValueError("performance host differs from authoritative host")
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(directory)
    directory.mkdir(mode=0o700)
    raw, guard = directory / "benchmark.json", directory / "guard.json"
    binary_hash = sha(binary)
    env = dict(
        os.environ,
        PYTHONHASHSEED="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONPATH=str(execution_repo / "tools/golden-gen/src"),
        PYTHONDONTWRITEBYTECODE="1",
        UV_PROJECT_ENVIRONMENT=sys.prefix,
    )
    run_guarded(
        [
            str(binary),
            "--model-path",
            str(model),
            "--prompts-dir",
            str(execution_repo / "tools/golden-gen/prompts"),
            "--output",
            str(raw),
            "--repo-root",
            str(execution_repo),
            "--measurement-commit",
            source["commit"],
            "--measurement-tree",
            source["tree"],
        ],
        guard,
        cwd=execution_repo,
        env=env,
        supervision=context,
        timeout_seconds=1800,
    )
    if (
        source_identity(repo) != supervisor
        or source_identity(execution_repo) != source
        or sha(binary) != binary_hash
        or _common_runtime(collect_release_runtime(model, execution_repo).model_dump())
        != result["runtime_profile"]
    ):
        raise ValueError("performance source/binary/runtime changed during measurement")

    def ref(p: Path) -> dict[str, str]:
        return dict(path=p.relative_to(directory).as_posix(), sha256=sha(p))

    roles: dict[str, Any] = {}
    if prepared is not None:
        from golden_gen.release_memory_method import CALL_IDS
        from golden_gen.release_roles import POLICY_PATH, _load_binding

        _load_binding(repo, freeze)
        policy_path = directory / "source-role-policy.json"
        policy_path.write_bytes((repo / POLICY_PATH).read_bytes())
        roles = dict(
            sources=prepared["roles"].model_dump(),
            source_role_policy=ref(policy_path),
            build_source_id=prepared["policy"].benchmark_binary.build_source_id,
            memory_methods=[
                ref(directory / f"{call}.memory-method/record.json") for call in CALL_IDS
            ],
            stage_journal=ref(directory / "benchmark-stages.json"),
        )
    elif context is not None:
        from golden_gen.supervision import POLICY_PATH

        policy_path = directory / "supervision-policy.json"
        policy_path.write_bytes((repo / POLICY_PATH).read_bytes())
        roles = dict(supervision_source=supervisor, supervision_policy=ref(policy_path))
    output = directory / "evidence.json"
    atomic_json(
        output,
        dict(
            protocol=PROTOCOL,
            schema_version=3 if prepared is not None else 2 if context is not None else 1,
            kind="recorded_role_gpu_performance"
            if prepared is not None
            else "recorded_gpu_performance",
            **({"source": source} if prepared is None else {}),
            runtime=runtime,
            binary_sha256=binary_hash,
            raw=ref(raw),
            guard=ref(guard),
            authoritative_manifest_sha256=sha(authoritative_manifest),
            **roles,
            telemetry=[
                ref(directory / f"{w}-repetition-{n}.telemetry.json")
                for w in ("canonical_04", "canonical_05")
                for n in (1, 2, 3)
            ],
        ),
    )
    if prepared is not None:
        validate_role_performance(
            output,
            prepared["roles"],
            prepared["policy"],
            result["runtime_profile"],
            prepared["registry"],
            prepared["freeze"].policy_sha256,
        )
    return output
