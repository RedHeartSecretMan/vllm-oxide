# Research and prototypes — RC source

[Project](../README.md) · [Documentation](../docs/README.md) · [Architecture](../docs/architecture.md)

Research records experiments that informed the engine. It is separate from the supported API and production Cargo workspace. This historical RC contains one standalone experiment; later research documents on `main` are not part of its implementation.

| Material | Purpose | How to read it |
|---|---|---|
| [Candle/CUDA stack prototype](prototype-candle-stack/README.md) | Validate raw-pointer/stream interoperability and the paged FlashAttention entrypoint | Independent CUDA executable with synthetic tensors, not model accuracy evidence |
| [Architecture decisions](../docs/adr/README.md) | Preserve the accepted or proposed design at each checkpoint | Check original status; newer v0.2.0 decisions are outside this RC |
| [Layered validation](../docs/validation/README.md) | Record version-specific accuracy and behavior evidence | Preserve original sources and the RC's 18 numerical failures |

Use the [main-workspace getting-started guide](../docs/getting-started.md) to run Qwen3. The experiment's FP16 tensors, 32-token page size and small synthetic dimensions do not describe the shipped engine's BF16 Qwen3 profile or 256-token KV blocks.

## Reproduce and report research

Start with the prototype's pinned dependency commit and build requirements. Its own workspace and committed `Cargo.lock` let you run it directly from this checkout. CPU tests cover the complete-output validator. Keep build artifacts and logs local, and report each run's source, resolved dependencies, GPU/driver/Toolkit, stdout, and exit code. The earlier test-machine description refers to the historical experiment.

A new experiment should name its question, entrypoint, fixed inputs, observable success criteria and limits. Synthetic shape and complete finite-output checks do not establish model numerical equivalence. Promote a useful conclusion through a reviewed implementation and relevant engine tests, with any numerical measurements attached to their actual source identity.

Architecture references: [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), [vLLM](https://github.com/vllm-project/vllm) and [Candle](https://github.com/huggingface/candle). Community/documentation research on current `main` is [available separately](https://github.com/RedHeartSecretMan/vllm-oxide/blob/main/research/README.md).
