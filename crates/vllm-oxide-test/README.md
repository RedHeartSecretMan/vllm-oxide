# vllm-oxide-test — historical comparison tool

[Documentation](../../docs/README.md) · [Validation record](../../docs/validation/README.md) · [Fixture generator](../../tools/golden-gen/README.md)

This v0.1.0 package compares Rust engine output with schema-1 fixtures. Engine execution requires a GPU; the package's unit tests run on CPU. It is a historical manual comparison tool, not the stricter current acceptance evaluator. **A zero exit status does not guarantee full fixture coverage.**

## Local fixture comparison

From the repository root, with local Qwen3-0.6B weights and the original fixture archive extracted:

```bash
export QWEN3_MODEL_DIR=/absolute/path/to/Qwen3-0.6B
export V010_FIXTURE_DIR=/absolute/path/to/extracted-fixtures
cargo run --locked --release -p vllm_oxide_test --features cuda -- \
    --model-path "$QWEN3_MODEL_DIR" \
    --manifest "$V010_FIXTURE_DIR/manifest.json" \
    --prompts-dir tools/golden-gen/prompts
```

The package declares `cuda = ["vllm_oxide/cuda"]`; enable it for real inference. It declares the binary name `vllm-oxide-test`, so a release build places it at `target/release/vllm-oxide-test`. `--model-path` takes a local directory, not a Hub repo ID. The engine requires SM 8.9+; recorded validation was RTX 4080/WSL2. The historical manifest declares Rust 1.75 without an independently established minimum-toolchain matrix.

The local manifest must be next to its fixture files; safetensors SHA-256 values are checked when fixtures load. Use the accompanying original manifest and pinned model revision, rather than recalibrating original evidence in place.

## CLI options

| Option | Behavior |
|---|---|
| `--model-path PATH` | Required local checkpoint (`config.json`, `tokenizer.json`, weights and shard index when needed) |
| `--manifest PATH` | Local manifest, with fixtures in its parent directory |
| `--release-tag TAG` | Alternative fixture source using GitHub assets |
| `--repo OWNER/REPO` | Download repository; default `RedHeartSecretMan/vllm-oxide` |
| `--cache-dir PATH` | Download cache; default `/tmp/vllm-oxide-goldens` |
| `--prompts-dir PATH` | Defaults to `tools/golden-gen/prompts` relative to working directory |
| `--epsilon VALUE` | Canonical L1 near-tie override; default `2 × atol` |
| `--debug` | Add L3 placeholder output |
| `--json` | Emit a JSON report |
| `--l1-only` | Omit L2 |
| `--l2-only` | Omit L1 |

Choose either `--manifest` or `--release-tag`; one is required by runtime validation. Do not combine `--l1-only` and `--l2-only`: the archived CLI does not reject that combination, and it can leave both comparison lists empty.

The historical downloader reads release metadata, expects `manifest.json` plus every fixture as a **separate asset**, verifies fixture SHA-256 and caches files. It does not unpack `.tar.gz`. v0.1.0 attaches `manifest.json` and `goldens-v0.1.tar.gz`, so its original assets should be downloaded/extracted for local `--manifest` mode. The older example tag `goldens-v0.1` in source comments is not a replacement for this release's actual asset layout.

## Comparison behavior and limits

| Layer | Implemented behavior | Limits |
|---|---|---|
| L1 canonical | Greedy token IDs/lengths, optional near-tie classification using generated logits | Exclusions are separate from exact matches; the helper recognizes expected top-1 only, despite the intended top-2 comment |
| L1 regression | Token IDs/lengths with manifest skip-map exclusions | CLI loads only canonical prompt IDs, so regression fixtures are normally skipped |
| L2 | Absolute-only comparison (`abs(actual − expected) <= atol`) on rows whose current token matches | Does not enforce a common full prefix or stop at first divergence; omitted rows/empty comparisons are not full accuracy evidence |
| L3 | Placeholder message when `--debug` is supplied | No activation capture/comparison; stub result does not enter overall verdict |

The runner iterates canonical fixtures from both oracle identities. `canonical_05a`–`canonical_05d` are expanded fixture IDs but the archived canonical prompt loader only keys the parent `canonical_05`, so those subfixtures may not find a prompt and are skipped. Missing prompts and several generation/conversion errors log a message and continue.

Report aggregation checks `all()` over collected L1/L2 results. Empty result lists therefore pass; L2 can also pass with no compared elements. Inspect actual fixture/result counts, skipped/missing IDs, token lengths, errors, exact matches, exclusions and compared elements before interpreting output. This tooling limitation is preserved, not silently repaired by the documentation refresh.

The original release recorded four of eight groups with full-output token differences, `atol=66.46875`, and six regression cases with skip positions. Preserve those findings. The [golden-generation ADR](../../docs/adr/0005-golden-generation-correctness-strategy.md) describes the intended strategy; the [validation guide](../../docs/validation/README.md) distinguishes it from this source's actual coverage and implementation.

## CPU tests and source layout

```bash
cargo test --locked -p vllm_oxide_test
cargo test --locked --workspace
```

These tests do not load models or establish GPU correctness. `src/main.rs` orchestrates comparisons; `types.rs`/`manifest.rs` parse schema and tensors; `download.rs` handles assets and hashes; `prompts.rs` loads canonical JSONL; `l1.rs`, `l2.rs` and `l3.rs` implement comparison layers; `report.rs` aggregates output. There is no modern performance sampler or layered schema-5 reader in this version.

Cache and fixture directories stay local. Preserve original manifests/logs before cleaning a scratch directory, and keep raw acceptance packages out of git and release uploads for this documentation maintenance.
