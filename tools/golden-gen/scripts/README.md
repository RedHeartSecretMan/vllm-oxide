# Fixed diagnostic scripts — RC source

[Tool index](../../README.md) · [golden-gen](../README.md) · [Research](../../../research/README.md)

These three scripts are diagnostic-only. They produce no release acceptance and never select or weaken numerical budgets. Read their exact fixed inputs before running them. Later matrix/GEMM probes on `main` are not included in this RC.

## Prepare the runtime

Run from the repository root after preparing the locked worker environment described in [golden-gen](../README.md#development-setup-and-cpu-checks). Set the package path and deterministic environment before Torch initialization:

```bash
export GOLDEN_WORKER_PYTHON=/absolute/path/to/golden-worker/bin/python
export PYTHONPATH="$PWD/tools/golden-gen/src"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The two tiny probes import Torch; the BF16 probe also imports Transformers Qwen3 helpers. They load neither model weights nor fixture files. CPU mode works with the worker dependencies installed and does not require a GPU. CUDA mode must be selected explicitly; it owns a GPU process and should run separately from other GPU work.

## BF16 residual, RMSNorm and RoPE

```bash
"$GOLDEN_WORKER_PYTHON" tools/golden-gen/scripts/bf16_reference_probe.py --device cpu
```

`--device` is required and accepts `cpu` or `cuda`. Inputs are four BF16 ones with residual `[1/256, 1/256, 1/256, 0]`, plus a fixed two-component rotary example. The stdout JSON records reference rounded residual/RMSNorm/RoPE values, historical alternative emulations, actual Torch/device/deterministic flags and the script SHA-256. It carries `diagnostic_only=true`, `accepting=false`, `model_weights_loaded=false` and an empty fixture list. The script checks deterministic startup; it does not compute full-model KL or assert reference/candidate equivalence.

## BF16 SiLU rounding

```bash
"$GOLDEN_WORKER_PYTHON" tools/golden-gen/scripts/silu_reference_probe.py --device cpu
```

This required `--device` also accepts `cpu` or `cuda`. Fixed inputs are BF16 gate `2.0` and up value `0.515625`. The script asserts reference SiLU **1.7578125**, reference gated result **0.90625**, and historical low-precision/late-cast alternatives **0.91015625**. Failed assertions or startup violations exit nonzero. Stdout JSON retains the same diagnostic/source/runtime fields as the BF16 probe. These scalar differences explain a rounding boundary; they are not a release PASS.

## Frozen `canonical_03` layer localization

[canonical03_layer_probe.py](canonical03_layer_probe.py) has positional `reference` and `compare` modes. Both require `--repo-root`, `--output-dir` and `--manifest`: reference mode needs a fresh output directory; comparison reads the existing reference directory. This is a private reproduction tied to the exact historical manifest/checkpoint, not an arbitrary prompt runner. A newly generated manifest with the same filename is insufficient.

| Option | Meaning |
|---|---|
| `--repo-root` | Clean RC source identity and original prompt registry |
| `--output-dir` | External reference directory; fresh for collection, existing for comparison |
| `--manifest` | Original frozen manifest and its declared unchanged fixture files |
| `--model-path` | Complete fixed Qwen3 checkpoint; required in `reference` mode |
| `--previous-dir` | Existing original diagnostic/candidate capture tree; required in `compare` mode |
| `--regression11` | Select the fixed original `regression_11` row-0-through-11 collection check, layer zero only |

A prepared canonical reference capture can be invoked as follows; paths must point to the retained original inputs and a new output:

```bash
"$GOLDEN_WORKER_PYTHON" tools/golden-gen/scripts/canonical03_layer_probe.py reference \
    --repo-root "$PWD" --output-dir /tmp/vllm-oxide-rc-canonical03-new \
    --model-path /absolute/path/to/Qwen3-0.6B \
    --manifest /absolute/path/to/original/manifest.json
```

Reference mode validates exact manifest/fixture hashes and original token history before loading Transformers on CUDA. It captures embedding, layer-zero normalization/projection/rotary/attention checkpoints, all 28 layer outputs and final normalization for the fixed two-step `canonical_03` history. Outputs include request/setup data, Torch JSONL/attention records, `logits.npz` and a `reference.json` with checksums and original identities; inspect the source's actual metadata rather than inferring acceptance from files being present.

After the corresponding original candidate capture is available, comparison runs on CPU:

```bash
"$GOLDEN_WORKER_PYTHON" tools/golden-gen/scripts/canonical03_layer_probe.py compare \
    --repo-root "$PWD" --output-dir /absolute/path/to/new/reference-capture \
    --previous-dir /absolute/path/to/original/candidate-capture \
    --manifest /absolute/path/to/original/manifest.json
```

The comparison checks capture provenance, shapes/dtypes and bit-equivalence of the fixed prefix before printing a diagnostic JSON summary. It does not manufacture candidate traces or new full-model acceptance. `--regression11` is narrower: the original reference fixture has token/top-5 data only, so the checker establishes collection equivalence rather than a full-logit reference L1 comparison. Preserve all original IDs and never use these opened diagnostic cases as a fresh holdout.

## Tests and interpretation

CPU coverage for the localization reader is in [test_layer_diagnostic.py](../tests/test_layer_diagnostic.py). Run it with the [complete test environment](../README.md#development-setup-and-cpu-checks). The prototype operator scripts have assertions or reports, not separate model acceptance tests.

A missing dependency, nondeterministic environment, wrong manifest hash, changed fixture or missing candidate trace is an input/reproduction failure, not a numerical PASS. Save stdout, errors and source/runtime identities locally. None of these commands publishes assets, changes the frozen registry or certifies the refreshed historical release archive.
