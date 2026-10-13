# Benchmarks

[Documentation](../README.md) · [Getting started](../getting-started.md) · [Validation](../validation/README.md)

Measure the complete workload before drawing a conclusion about application speed. End-to-end generation includes model forward execution, scheduling, cache operations, sampling, and token transfers; a faster isolated kernel does not imply the same improvement in total inference throughput.

## v0.2.0 end-to-end baseline

The [release validation summary](../releases/goldens-v0.2.md) records the completed source-bound measurements on Qwen3-0.6B BF16, RTX 4080/SM 8.9, WSL2, driver 595.71, and CUDA Toolkit 13.2.51.

| Workload | Prefill token/s | Decode token/s | TTFT ms | Peak device memory MiB |
|---|---:|---:|---:|---:|
| One request, `canonical_04` | 13910.2653 | 29.7601 | 40.9622 | 10756 |
| Four requests together, `canonical_05` | 1715.8632 | 115.8207 | 36.7626 | 10692 |

Each workload discards one warmup run and retains three measured repetitions. The table reports medians; batch throughput is the total across all four requests. Each request generates 64 completion tokens with greedy selection and `ignore_eos=true`, using a cold request/prefix-cache state. Construction and built-in model warmup are excluded from the reported generation timings.

The fixed engine configuration uses `gpu_memory_utilization=0.50`, `max_model_len=4096`, `max_num_batched_tokens=16384`, `max_num_seqs=512`, BF16, and eager execution. Normal API/CLI defaults use a `0.9` memory fraction, so the reported memory peak should not be read as a bound for the default CLI configuration.

Prefill throughput counts newly executed, uncached prompt tokens over synchronized prefill step time. Decode throughput counts completion tokens after the first token over synchronized decode time. TTFT runs from request admission, after tokenization, to the first selected token. Device memory is sampled device-wide, including allocated cache storage; it is not the checkpoint's weight size or a minimum-memory requirement.

All six repetitions passed their sample/config/output and memory-record checks. The 5 ms memory-sampling target retained a strict 50 ms timing limit; the largest observed completion interval was 13.359862 ms. The [original protocol](../adr/0012-goldens-v0.2-calibration-and-performance-protocol.md) describes the metric definitions, and the final summary supplies the actual sources and hashes.

These numbers describe one fixed environment and workload. They are not a cross-hardware performance guarantee, a general-purpose model benchmark, or a comparison claiming parity with Python vLLM. GPU exclusivity was not asserted.

## Focused microbenchmarks

| Report | What was measured | What it does not establish |
|---|---|---|
| [Greedy sampling](greedy-sampling.md) | CUDA greedy argmax on prepared logits, baseline versus optimized batch execution | End-to-end generation speed, other sampling modes, or a latency SLA |
| [Greedy sampling raw rounds](greedy-sampling-2026-10-09.json) | Unrounded iteration counts, timing rounds, and environment | A current benchmark of the entire release |

The greedy report includes the baseline source, batch sizes, synchronization method, run order, and reproduction commands. Reproduce it at the documented source revision rather than assuming current `HEAD` is the original measured implementation.

## Run and report a measurement

Use the [validation tooling](../../tools/golden-gen/README.md) for release-style measurements and the commands in the focused report for its microbenchmark. GPU measurements run separately from ordinary CPU CI.

When reporting a new result, include the source commit, model revision and dtype, GPU/driver/toolkit, OS, engine and sampling settings, workload shape, timing boundaries, warmup/repetition counts, summary statistic, and memory method. Keep raw samples so another reader can inspect both successful measurements and failures. Use a new output directory and preserve earlier source/result identities.

For application tuning, start with the [engine configuration](../getting-started.md#engine-configuration). Default cache sizing reserves a large fraction of free GPU memory; a lower cache budget changes capacity and workload behavior, so record it alongside timing results.
