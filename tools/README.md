# Tools — v0.1.0

[Project](../README.md) · [Documentation](../docs/README.md) · [Development](../docs/development.md) · [Validation](../docs/validation/README.md)

These tools belong to the historical v0.1.0 source. They support fixture generation and inspection; they are not runtime dependencies of the Rust engine. Their schema, coverage and failure handling differ from the current v0.2.0 tooling.

| Entry | Purpose | Requirements / output |
|---|---|---|
| [golden-gen](golden-gen/README.md) | Generate paired Transformers/vLLM fixtures and calibrate schema-1 tolerances | Python 3.12; CPU fake/test paths or GPU oracle extras; manifest and safetensors |
| [Rust comparison crate](../crates/vllm-oxide-test/README.md) | Inspect historical L1/L2 results against local fixtures | GPU for engine execution; human or JSON report |
| [`validate-release.sh`](validate-release.sh) | Historical generation → calibration → Rust comparison → GitHub upload wrapper | GPU, uv, Cargo and gh; writes a fixed temporary directory and can create a remote release |

There is no `tools/scripts/` module, modern schema-5 layered acceptance, public GPU smoke helper or direct NVML benchmark collector in this version. Use the actual entry points above rather than current-version commands.

## Recommended historical inspection

1. Read [validation](../docs/validation/README.md) before interpreting PASS. The original record contains token differences and exclusions, and the runner can pass an incomplete report.
2. Inspect original assets using local `--manifest` mode. The release includes a fixture archive; the historical downloader expects individual attached safetensors and cannot unpack the archive automatically.
3. For CPU development, use the generator's `dev` extra and synthetic `--dry-run` output in a separate temporary directory. Synthetic fixtures do not establish model accuracy.
4. For a new GPU experiment, explicitly install the `gpu` extra, use a fresh output directory, inspect complete fixture coverage and record actual environment/source. Do not mix generated data with original assets or reuse a pending/stale tolerance as accepted evidence.

## Historical release wrapper

`validate-release.sh` is preserved unchanged for source history. It generates into `/tmp/vllm-oxide-goldens-release`, calibrates that directory and eventually calls `gh release create`, attaching manifest and individual safetensors files. Its optional second argument chooses the **new upload tag**, not a source for downloading existing fixtures.

The script uses plain `uv sync` without installing the GPU extra and builds the comparison crate without its `cuda` feature. Those commands are incomplete for a fresh environment's real GPU run. In addition, the generator/runner's historical failure and coverage handling is not fail-closed. A script success therefore does not prove complete numerical acceptance.

Prefer the explicit version-correct local commands in [golden-gen](golden-gen/README.md) and the [comparison guide](../crates/vllm-oxide-test/README.md). Do not run the upload wrapper merely to read or validate an old release. The present documentation maintenance keeps evidence local and does not invoke its publishing step.

## Artifacts and cleanup

| Location | Content | Handling |
|---|---|---|
| `tools/golden-gen/output/` | Default generated manifest and fixtures | Ignored; keep local |
| Custom `--output-dir` | Same output at an explicit path | Not automatically covered by the default directory ignore; use a temporary/external directory |
| `tools/golden-gen/.venv/` | uv environment | Ignored; disposable after use |
| `/tmp/vllm-oxide-goldens-release` | Historical wrapper output | Fixed path; avoid mixing runs; preserve needed evidence before cleanup |
| `/tmp/vllm-oxide-goldens` | Rust downloader's default cache | Local; inspect before clearing |
| Root `target/` | Rust workspace builds | Ignored; `cargo clean` removes builds |
| `research/prototype-candle-stack/target/` | Independent prototype builds | Clean from the prototype package |

The generator merges fixture metadata when a manifest already exists and may reuse an existing positive tolerance. Calibration updates the manifest **in place**. Use a fresh directory for reproducibility; never point these write operations at preserved release evidence.

Before cleanup, identify exactly which directory belongs to the current run and retain any required logs/manifests/results. Remove only those selected generated artifacts. Do not delete original fixture assets, model caches or other runs merely to tidy the repository. Raw acceptance packages and model weights must stay outside git and should not be uploaded as part of this documentation update.
