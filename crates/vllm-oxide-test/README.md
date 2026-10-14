# vllm-oxide-test — schema-1 comparison tool

[Documentation](../../docs/README.md) · [Validation record](../../docs/validation/README.md) · [Fixture generator](../../tools/golden-gen/README.md)

This v0.1.0 package compares Rust engine output with schema-1 fixtures. Engine
execution requires a GPU; the package's regression tests run on CPU. It retains
legacy L1/token and L2/logit meanings and is separate from the current layered
acceptance evaluator. The maintained runner rejects missing declared evidence,
comparison errors and empty reports. Original release measurements and their
failures keep their original source identities.

## Local fixture comparison

From the repository root, with local Qwen3-0.6B weights and the original fixture
archive extracted:

```bash
export QWEN3_MODEL_DIR=/absolute/path/to/Qwen3-0.6B
export V010_FIXTURE_DIR=/absolute/path/to/extracted-fixtures
cargo run --locked --release -p vllm_oxide_test --features cuda -- \
    --model-path "$QWEN3_MODEL_DIR" \
    --manifest "$V010_FIXTURE_DIR/manifest.json" \
    --prompts-dir tools/golden-gen/prompts
```

The package declares `cuda = ["vllm_oxide/cuda"]`; enable it for inference.
Its binary is `target/release/vllm-oxide-test`. `--model-path` accepts a complete
local directory, not a Hub repository ID. The engine requires SM 8.9+; original
validation used RTX 4080/WSL2. The maintained source pins Rust 1.94.0. The
original Rust 1.75 declaration was not validated and did not satisfy the lock.

The manifest must be next to its fixture files. The runner preflights every
declared fixture's checksum, tensor geometry, finite values, token vocabulary
and prompt identity before the first model initialization. Use the original
manifest and pinned model revision; never recalibrate retained evidence in place.

## CLI options

| Option | Behavior |
|---|---|
| `--model-path PATH` | Required complete local checkpoint |
| `--manifest PATH` | Local manifest, with fixtures in its parent directory |
| `--release-tag TAG` | Alternative fixture source using GitHub assets |
| `--repo OWNER/REPO` | Download repository; default `RedHeartSecretMan/vllm-oxide` |
| `--cache-dir PATH` | Download cache; default `/tmp/vllm-oxide-goldens` |
| `--prompts-dir PATH` | Canonical and regression JSONL directory; default `tools/golden-gen/prompts` |
| `--epsilon VALUE` | Finite non-negative canonical L1 near-tie override; default `2 × atol` |
| `--debug` | Rejected: L3 activation comparison is not implemented |
| `--json` | Emit the comparison report as JSON |
| `--l1-only` | Explicitly select token comparison only |
| `--l2-only` | Explicitly select full-logit comparison only; rejects regression fixtures lacking full logits |

Select either `--manifest` or `--release-tag`; a source is required. The two
partial-comparison flags conflict. A deliberate partial run establishes only
the selected layer. A nonempty manifest is also required; its declared inventory
defines this tool's scope and does not provide the modern protocol's separate
frozen-registry completeness guarantee. Fake-oracle fixtures cannot establish
numerical acceptance.

The historical downloader expects `manifest.json` and each safetensors file as
a separate attached asset. It does not unpack `.tar.gz`. The actual v0.1.0
Release retains `manifest.json` and `goldens-v0.1.tar.gz`, so use the extracted
archive and local `--manifest` mode for those assets. The old `goldens-v0.1` tag
in source comments is not a replacement for the actual attachment layout.

## Maintained comparison behavior

| Layer | Behavior | Scope |
|---|---|---|
| L1 canonical | Greedy token IDs and lengths, with separately reported legacy near-tie exclusions | The actual token must be the candidate argmax and the expected token must fall in its top two, including ties; the top-two gap must be strictly below epsilon |
| L1 regression | Token IDs and lengths with original manifest skip-map exclusions | Excluded positions remain distinct from exact matches |
| L2 | Absolute-only full-logit comparison through the first divergent token, including that row | Later rows are excluded even if their selected token happens to match again |
| L3 | No implemented activation comparison | The CLI refuses `--debug` rather than returning a placeholder PASS |

Both canonical and regression corpora are loaded. Batch specifications expand
to their concrete member IDs; each fixture is captured as a separate request
for this diagnostic comparison. It does not recreate the original oracle's
batched execution geometry or prove production batching equivalence. Both
oracle identities remain part of the legacy manifest and are compared according
to this schema-1 tool, without acquiring the current reference/baseline contract.

A missing prompt/file, invalid fixture, generation/conversion error or incomplete
declared comparison now fails the run. L2 requires nonempty consistent geometry
and a finite non-negative tolerance; non-finite compared values cannot pass.
The legacy JSON names `same_token_steps` and `diff_token_steps` now count compared
shared-prefix rows and later excluded rows respectively.

The original implementation could skip missing prompts/errors and pass an empty
report; its L2 filter tested current-token equality without enforcing the whole
prefix. Those behaviors motivated the maintenance fixes. The near-tie helper now
recognizes a reference runner-up while requiring the actual token to be the
candidate argmax; the original `epsilon = 2 × atol` default and six-case
regression skip map are unchanged. This is not the modern reference-choice-loss
gate. These repairs do not change the
original release's four of eight full-output token differences,
`atol=66.46875`, or six regression cases with skipped positions. No new model
accuracy or performance result is claimed by repairing the runner. See
[historical validation](../../docs/validation/README.md) and the
[source maintenance record](../../docs/version-doc-update.md).

## CPU tests and source layout

```bash
cargo test --locked -p vllm_oxide_test
cargo test --locked --workspace
```

CPU tests cover prompt expansion, declared-fixture preflight, empty-report
rejection, malformed/non-finite logits, causal-prefix boundaries, unsupported
L3 and initialization-error propagation. They load no model and establish no GPU
accuracy result. `src/main.rs` orchestrates comparisons; `types.rs` and
`manifest.rs` parse schema and tensors; `download.rs` handles assets and hashes;
`prompts.rs` loads both corpora; `l1.rs`, `l2.rs`, `l3.rs` and `report.rs` retain
the legacy layers/report structures. There is no modern performance sampler or
layered schema-5 consumer in this version.

Keep caches, fixtures and complete raw evidence local. Preserve original
manifests and results before cleaning a scratch directory.
