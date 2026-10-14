# Diagnostic scripts

[Tooling overview](../../README.md) · [golden-gen guide](../README.md) · [Validation guide](../../../docs/validation/README.md)

Use these scripts to investigate specific BF16 rounding and historical layer
differences. Reports carry `diagnostic_only: true` and `accepting: false`. They
provide diagnostics without creating fixtures, approving budgets, or establishing
a new model-accuracy result. Keep their
JSON, traces and model inputs outside Git. Use the
[layered workflow](../README.md#layered-accuracy-protocol-new-workflow) for current
acceptance and the [public GPU smoke](../../../docs/validation/public-gpu-smoke.md)
for production API/CLI checks.

## Environment and invocation

Run from the repository root with Python **3.12** and the locked golden-gen
environment. All scripts import `golden_gen`, so use the prepared package
interpreter rather than an unrelated system Python. The two operator probes
need Torch; the RMSNorm/RoPE probe also imports Transformers Qwen3 internals.
The layer probe needs NumPy, Safetensors, Torch, Transformers and the complete
pinned local checkpoint. The package's `gpu` extra installs these runtime
dependencies; installing it does not itself require executing a GPU operation.

After preparing dependencies as described in the [package guide](../README.md),
set the deterministic environment **before** the process starts:

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Every probe validates these exact values. Missing or different values cause a
failure before the diagnostic calculation. The tiny probes seed Python,
NumPy and Torch with 0 and enable deterministic algorithms with
`warn_only=False`. CUDA mode requires a usable Torch CUDA installation and
device. CPU mode opens no weights and requires no GPU. Use a fresh process for
each invocation and record the actual package versions, device and script hash
printed in its JSON.

## BF16 residual, RMSNorm and RoPE

[`bf16_reference_probe.py`](bf16_reference_probe.py) accepts one required flag:
`--device cpu` or `--device cuda`. It uses four-element BF16 residual inputs and
a two-element rotary example to compare the Transformers operation order with
FP32/late-rounding emulations. It constructs `Qwen3RMSNorm` locally and never
downloads or opens a model or fixture.

```bash
uv run --project tools/golden-gen --locked --extra gpu \
    python tools/golden-gen/scripts/bf16_reference_probe.py --device cpu
```

Use `--device cuda` to inspect the CUDA path in a prepared GPU environment. JSON
is written to **stdout**; there is no `--output` flag. Its fields include
`bf16_residual_sum`, `rmsnorm_reference`, `legacy_unrounded_sum_emulation`,
BF16 cosine/sine, `rope_reference`, and `legacy_fp32_rotation_emulation`.
These examples show rounding-order differences; the script does not compare
the Rust engine or assert a model-level tolerance.

## BF16 SiLU gate multiplication

[`silu_reference_probe.py`](silu_reference_probe.py) also requires
`--device cpu|cuda` and prints JSON to stdout. Fixed BF16 inputs are gate `2.0`
and up `0.515625`. The source asserts the reference SiLU result `1.7578125`,
its gated product `0.90625`, and the low-precision/late-cast emulations
`0.91015625`. These are the script's explicit expected values, not newly
measured release results.

```bash
uv run --project tools/golden-gen --locked --extra gpu \
    python tools/golden-gen/scripts/silu_reference_probe.py --device cpu
```

Unlike the residual/RoPE probe, this script imports no Transformers model
classes. A failed assertion exits nonzero and produces no successful report. It opens neither model weights nor fixtures.

## Fixed historical layer localization

[`canonical03_layer_probe.py`](canonical03_layer_probe.py) has `reference` and
`compare` modes. Both require `--repo-root` and `--output-dir`; the directory
is the diagnostic evidence root, not an arbitrary destination for a summary.
This tool is tied to immutable historical fixtures and captures checked by
[`layer_diagnostic.py`](../src/golden_gen/layer_diagnostic.py). It is **not** a
generic layer comparator for a newly selected prompt, manifest or checkpoint.
If those original local inputs are unavailable, the comparison cannot be
reconstructed from a GitHub product source archive alone.

| Mode and flags | Inputs, execution and outputs |
|---|---|
| `reference --model-path PATH --manifest PATH` | Fresh, non-existing `--output-dir` with an existing parent; clean `--repo-root`; the exact historical Transformers fixture and manifest. Loads the hash-checked Qwen3-0.6B checkpoint on CUDA, forces SDPA math, and writes `request.json`, `reference.json`, `torch/logits.npz`, `torch/step-N.jsonl` and `torch/attention-N.json`. |
| `compare --previous-dir PATH --manifest PATH` | Existing reference output plus separately captured matching `rust/` evidence, and the exact previous uninstrumented candidate capture. CPU-only artifact verification and localization; prints a JSON report to stdout. It does not collect Rust traces or run a model. |
| `--regression11` | Available in both modes. Switches to the original `regression_11` case: 12 predictions, its fixed 11-token advance history, and embedding/layer-zero checkpoints only. Comparison verifies collection equivalence; it explicitly performs no tensor attribution. |

The default `canonical_03` run retains two predictions: prefill and one decode
using advance token `151667`. It records embedding, first-layer norm/Q/K/V,
Q/K norm, RoPE and attention context, all 28 layer outputs, and final norm. Both
engines must reproduce the original raw rows before layer differences are
interpreted. The default comparison binds the prior candidate commit
`0e09fb5fb9096f700d675f66b8c0ff81e56c4f81` and its capture-index hash.
Regression-11's original reference contains tokens and top-5 logits only;
it cannot establish a full-logit reference comparison.

With the **original historical inputs already retained locally**, example
invocations are:

```bash
uv run --project tools/golden-gen --locked --extra gpu \
    python tools/golden-gen/scripts/canonical03_layer_probe.py reference \
    --repo-root "$PWD" \
    --output-dir /absolute/path/to/fresh-canonical03-diagnostic \
    --model-path /absolute/path/to/pinned-qwen3-checkpoint \
    --manifest /absolute/path/to/original-reference/manifest.json

# After the separately retained matching Rust capture and traces are present:
uv run --project tools/golden-gen --locked --extra gpu \
    python tools/golden-gen/scripts/canonical03_layer_probe.py compare \
    --repo-root "$PWD" \
    --output-dir /absolute/path/to/fresh-canonical03-diagnostic \
    --previous-dir /absolute/path/to/original-candidate-capture \
    --manifest /absolute/path/to/original-reference/manifest.json
```

For the regression case add `--regression11` to both invocations and supply its
own frozen manifest, previous candidate and separate diagnostic root. Never
substitute a current release manifest merely because its schema or prompt name
looks compatible. `reference` reads the matching tracked prompt JSONL from the
reviewed repository. The checkpoint must have the exact config, tokenizer and
single Safetensors weight hashes required by the oracle; loading uses local
files only.

The expected `rust/` side includes `capture-index.json`, the named candidate
JSONL, and matching `step-N.jsonl` / `attention-N.json` traces. The Python script
does not create them. The retained `vllm-oxide-observe` producer selects this
diagnostic mode through `VLLM_OXIDE_INTERNAL_LAYER_TRACE_DIR`, which must point
to the **existing reference root**, not its `rust/` child. Its output must be
the fresh `rust/` child of that same root. With a reviewed, previously built
matching producer, the explicit invocation is:

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
VLLM_OXIDE_INTERNAL_LAYER_TRACE_DIR=/absolute/path/to/fresh-canonical03-diagnostic \
    /absolute/path/to/built/vllm-oxide-observe \
    --model-path /absolute/path/to/pinned-qwen3-checkpoint \
    --manifest /absolute/path/to/original-reference/manifest.json \
    --prompts-dir "$PWD/tools/golden-gen/prompts" \
    --output-dir /absolute/path/to/fresh-canonical03-diagnostic/rust \
    --repo-root "$PWD" \
    --measurement-commit "$(git rev-parse HEAD)" \
    --measurement-tree "$(git rev-parse 'HEAD^{tree}')"
```

The producer validates the clean checkout, build/binary identity, model bytes
and manifest, and reads the selected case from `request.json`. A newer engine
need not reproduce the old raw rows; such a mismatch is a rejected diagnostic
comparison, not permission to replace the historical reference. Run this GPU
producer after the reference owner finishes, with the same external resource
supervision used for other GPU work. This retained comparison format is not
interchangeable with a new layered capture. Current bounded layer tracing is available through
[`vllm-oxide-fixed-prefix --layer-trace-dir`](../../../crates/vllm-oxide-test/README.md#current-capture-workflow),
but its output alone does not recreate the historical capture index or original
source identity required here.

## Interpreting failures and preserving evidence

- Deterministic-environment errors: start a new process with the two exact
  environment values; changing them after Torch imports is insufficient.
- Missing packages or Torch/Transformers API differences: use the locked
  interpreter; the operator probes use internal model classes and the layer
  probe installs temporary hooks on those classes in its own process.
- Manifest, fixture, source or prefix mismatch: retain the rejected attempt.
  Locate the original inputs; editing constants, hashes or rows changes the
  experiment rather than repairing its provenance.
- Existing layer output path: select a fresh diagnostic root. Reference output
  uses exclusive directory/file creation and does not overwrite an old run.
- GPU memory or runtime errors: close the failed owner, retain its output, and
  investigate the environment. These standalone probes do not provide the
  layered collector's subprocess resource guard or acceptance marker.

Documentation changes do not claim a fresh probe run or alter historical
results. Use the [released validation summary](../../../docs/releases/goldens-v0.2.md)
for the product's accepted accuracy and performance results.
