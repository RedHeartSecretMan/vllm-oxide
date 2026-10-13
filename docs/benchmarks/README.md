# Benchmarks — historical RC

[Documentation](../README.md) · [Validation](../validation/README.md) · [Tooling](../../tools/golden-gen/README.md)

RC1 did not complete a publishable performance result. No throughput, latency or memory figure from final v0.2.0 is attributed to this source. The benchmark executables and validation code exist for maintainer reproduction, not as a performance claim.

[ADR-0012](../adr/0012-goldens-v0.2-calibration-and-performance-protocol.md) defines two fixed workloads: one canonical request and a four-request batch, greedy decoding and 64 completion tokens, each with one discarded warmup and three measured repetitions. Collect per-repetition prefill/decode throughput, TTFT, token-to-token latency, peak device memory and original synchronized timestamps. Comparisons require the same checkpoint, settings, source and measurement boundaries.

The RC samples memory using its original `nvidia-smi` process-based implementation. Later direct-NVML and memory-method validators are not in this version. Do not replace observed sampling gaps or interrupted runs with a successful summary. Performance observations have no cross-hardware floor and do not imply vLLM parity.

Use the [schema5 performance stage](../../tools/golden-gen/README.md#schema5-release-stages) only with its required complete authoritative inputs. The historical `tools/validate-release.sh benchmark` path uses the earlier schema-v4 procedure; it cannot establish layered release acceptance. Raw measurement files remain local.
