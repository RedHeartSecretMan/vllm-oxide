"""Recompute ADR-0012 performance evidence from original synchronized samples."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from golden_gen.layered_accuracy import PROTOCOL
from golden_gen.layered_artifacts import atomic_json, bound_file, sha, source_identity
from golden_gen.layered_manifest import _common_runtime, evaluate_manifest
from golden_gen.layered_release import Registry


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
) -> tuple[dict[str, Any], list[Path]]:
    wrapper = json.loads(path.read_text())
    if (
        wrapper.get("protocol") != PROTOCOL
        or wrapper.get("schema_version") != 1
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
    members = {m.case_id: m for group in registry.numerical_cases for m in group.plan.members}
    telemetry_refs = wrapper["telemetry"]
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
                document.get("format") != "vllm-oxide-internal-benchmark-json-v1"
                or document.get("schema_version") != 1
                or document.get("complete") is not True
                or document.get("call_id") != name.removesuffix(".telemetry.json")
                or document.get("request_ids") != list(range(len(ids)))
            ):
                raise ValueError("performance telemetry identity mismatch")
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
    return raw, closure


def collect_performance(
    repo: Path, directory: Path, model: Path, binary: Path, authoritative_manifest: Path
) -> Path:
    """GPU-owning stage, called only with separate stage authority; never during CPU tests."""
    from golden_gen.environment import collect_release_runtime
    from golden_gen.guard import run_guarded

    result = evaluate_manifest(repo, authoritative_manifest, authoritative=True)
    if result.get("verdict") != "PASS" or result.get("accepting") is not True:
        raise ValueError("performance requires complete approved authoritative evidence")
    source = source_identity(repo)
    runtime = collect_release_runtime(model, repo).model_dump()
    if _common_runtime(runtime) != result["runtime_profile"]:
        raise ValueError("performance host differs from authoritative host")
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(directory)
    directory.mkdir(mode=0o700)
    raw, guard = directory / "benchmark.json", directory / "guard.json"
    binary_hash = sha(binary)
    run_guarded(
        [
            str(binary),
            "--model-path",
            str(model),
            "--prompts-dir",
            str(repo / "tools/golden-gen/prompts"),
            "--output",
            str(raw),
            "--repo-root",
            str(repo),
            "--measurement-commit",
            source["commit"],
            "--measurement-tree",
            source["tree"],
        ],
        guard,
    )
    if (
        source_identity(repo) != source
        or sha(binary) != binary_hash
        or _common_runtime(collect_release_runtime(model, repo).model_dump())
        != result["runtime_profile"]
    ):
        raise ValueError("performance source/binary/runtime changed during measurement")

    def ref(p: Path) -> dict[str, str]:
        return dict(path=p.name, sha256=sha(p))

    output = directory / "evidence.json"
    atomic_json(
        output,
        dict(
            protocol=PROTOCOL,
            schema_version=1,
            kind="recorded_gpu_performance",
            source=source,
            runtime=runtime,
            binary_sha256=binary_hash,
            raw=ref(raw),
            guard=ref(guard),
            authoritative_manifest_sha256=sha(authoritative_manifest),
            telemetry=[
                ref(directory / f"{w}-repetition-{n}.telemetry.json")
                for w in ("canonical_04", "canonical_05")
                for n in (1, 2, 3)
            ],
        ),
    )
    return output
