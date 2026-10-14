# Tools — v0.1.0

[Project](../README.md) · [Documentation](../docs/README.md) · [Development](../docs/development.md) · [Validation](../docs/validation/README.md)

Use these v0.1.0 tools to generate and inspect historical fixtures. The Rust engine has no runtime dependency on them. Their schema, coverage, and failure handling differ from the current v0.2.0 tools.

| Entry | Purpose | Requirements / output |
|---|---|---|
| [golden-gen](golden-gen/README.md) | Generate paired Transformers/vLLM fixtures and calibrate schema-1 tolerances | Python 3.12; CPU fake/test paths or GPU oracle extras; manifest and safetensors |
| [Rust comparison crate](../crates/vllm-oxide-test/README.md) | Inspect historical L1/L2 results against local fixtures | GPU for engine execution; human or JSON report |
| [`validate-release.sh`](validate-release.sh) | Local generation → calibration → Rust comparison | GPU, uv and Cargo; fresh temporary environment, fixtures and stage logs; no upload |

There is no `tools/scripts/` module, modern schema-5 layered acceptance, public GPU smoke helper or direct NVML benchmark collector in this version. Use the actual entry points above rather than current-version commands.

## Recommended historical inspection

1. Read [validation](../docs/validation/README.md) before interpreting PASS. The original record contains token differences and exclusions, and the original runner could pass an incomplete report. Maintained comparison now rejects missing declared evidence and empty reports.
2. Inspect original assets using local `--manifest` mode. The release includes a fixture archive; the historical downloader expects individual attached safetensors and cannot unpack the archive automatically.
3. For CPU development, use the generator's `dev` extra and synthetic `--dry-run` output in a separate temporary directory. Synthetic fixtures do not establish model accuracy.
4. For a new GPU experiment, explicitly install the `gpu` extra, use a fresh output directory, inspect complete fixture coverage and record actual environment/source. Do not mix generated data with original assets or reuse a pending/stale tolerance as accepted evidence.

## Historical release wrapper

The maintained `validate-release.sh` runs the historical schema-1 workflow locally:

```bash
./tools/validate-release.sh /path/to/Qwen3-0.6B
```

Each invocation uses `mktemp -d` under `${TMPDIR:-/tmp}` for a fresh `.venv`, fixture directory and environment/generation/calibration/build/comparison logs. It installs the locked `gpu` extra, builds and runs the comparison crate with `--features cuda`, and stops with the original stage status on failure. Evidence remains local after success or failure. The script accepts one model-path argument and never creates a tag, release or upload. The model path selects the Rust comparison checkpoint; Python oracles retain the pinned model/revision in their configuration.

The generator now returns nonzero when either oracle fails and publishes a manifest only when every requested oracle/prompt succeeds. The comparison runner separately rejects missing declared evidence and failed comparisons. These repairs do not change the historical calibration method, exclusions, original assets or recorded verdicts; they do not turn schema-1 tooling into current layered acceptance. The explicit local steps are also documented in [golden-gen](golden-gen/README.md) and the [comparison guide](../crates/vllm-oxide-test/README.md).

## Artifacts and cleanup

| Location | Content | Handling |
|---|---|---|
| `tools/golden-gen/output/` | Default generated manifest and fixtures | Ignored; keep local |
| Custom `--output-dir` | Same output at an explicit path | Not automatically covered by the default directory ignore; use a temporary/external directory |
| `tools/golden-gen/.venv/` | uv environment | Ignored; disposable after use |
| `${TMPDIR:-/tmp}/vllm-oxide-v010-validation.*` | Local validation environment, fixtures and stage logs | One fresh directory per invocation; retain needed evidence before explicit cleanup |
| `/tmp/vllm-oxide-goldens` | Rust downloader's default cache | Local; inspect before clearing |
| Root `target/` | Rust workspace builds | Ignored; `cargo clean` removes builds |
| `research/prototype-candle-stack/target/` | Independent prototype builds | Clean from the prototype package |

On successful generation, the generator merges fixture metadata when a manifest already exists and may reuse an existing positive tolerance. A failed run returns nonzero without publishing or replacing the manifest; a prior manifest cannot make the current invocation succeed. Completed fixture files from a failed invocation may remain for diagnosis. Calibration updates the manifest **in place**. Use a fresh directory for reproducibility; never point these write operations at preserved release evidence.

Before cleanup, identify exactly which directory belongs to the current run and retain any required logs/manifests/results. Remove only those selected generated artifacts. Do not delete original fixture assets, model caches or other runs merely to tidy the repository. Raw acceptance packages and model weights must stay outside git and should not be uploaded as part of this documentation update.
