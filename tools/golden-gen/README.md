# golden-gen — Golden Fixture Generator for vllm-oxide

> **Dev-only — NOT in CI.** This harness runs manually on a GPU machine before each
> release to produce golden fixtures. It is NOT run in continuous integration and
> is NOT a dependency of the vllm-oxide Rust crates. See issue #14.

This harness generates **golden fixture files** for the vllm-oxide project by running
two independent oracle LLM engines — **HuggingFace Transformers** (reference in FP32)
and **vLLM** (BF16 baseline) — on a fixed set of prompts. Cross-validation between
the two oracles calibrates numerical tolerances so the Rust port can verify correctness
against known-good outputs without needing a GPU for every test.

## Prerequisites

- Linux with an NVIDIA GPU (sm_89+; a single A10 is sufficient)
- Python 3.11
- `uv` package manager (see [docs.astral.sh/uv](https://docs.astral.sh/uv/))
- ~6 GB free disk space for model weights and oracle caches

## Install

```bash
cd tools/golden-gen
uv sync --extra gpu
```

## Usage

### 1. Generate fixtures (GPU required)

```bash
cd tools/golden-gen
uv run python -m golden_gen generate
```

This loads both oracles (transformers + vLLM), discovers all canonical, batch,
and regression cases, and writes `output/manifest.json` +
`output/*.safetensors`. The five canonical prompt specifications flatten to
eight fixture cases because the batch corpus contains four sub-prompts; with
20 regression cases and two oracle artifacts per case, a complete run produces
56 fixture files. Expect ~5-15 minutes on an A10.

Options:

| Flag | Description |
|------|-------------|
| `--dry-run` | Use fake oracle (no GPU, no model download) — for smoke testing |
| `--output-dir PATH` | Output directory (default: `./output`) |
| `--only-category {canonical,regression}` | Exploratory partial generation; cannot satisfy the release gate |

### 2. Calibrate tolerance after generation

```bash
cd tools/golden-gen
uv run python -m golden_gen calibrate \
  --manifest-dir ./output \
  --comparison-policy-version same-prefix-v1 \
  --l1-near-tie-max-abs-logit-gap <reviewed-threshold> \
  --l2-atol <reviewed-threshold>
```

Validates every declared reference/baseline pair, computes same-prefix
calibration observations, and records the consumed baseline artifacts. The
version and both acceptance thresholds are required reviewed inputs; observed
baseline differences are never promoted into reference acceptance thresholds
automatically. A missing pair, empty comparison set, unmatched oracle length,
or unsupported tensor shape exits non-zero.

```bash
uv run python -m golden_gen --help   # full usage
```

### Dry-run (no GPU)

```bash
cd tools/golden-gen
uv run python -m golden_gen generate --dry-run
```

Produces a schema-valid but uncalibrated fake manifest and synthetic
`.safetensors` fixtures without loading any model or GPU. It exercises lifecycle
plumbing only and is never publishable release evidence.

### Testing (CPU-only)

```bash
cd tools/golden-gen
uv run pytest
```

All unit tests run on CPU and do not require a GPU.

## Manifest Schema

`manifest.json` is the provenance record for a set of golden fixtures.

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Manifest format version (currently 3) |
| `generated_at` | ISO 8601 | UTC timestamp of generation |
| `model.id` | str | HuggingFace model ID (`Qwen/Qwen3-0.6B`) |
| `model.revision` | str | Git revision (commit hash) of the model weights |
| `model.arch` | str | Model architecture (`Qwen3ForCausalLM`) |
| `model.dtype` | str | Model dtype (`bfloat16`) |
| `model.vocab_size` | int | Vocabulary size (151936) |
| `oracle_versions.transformers` | str | Installed transformers version |
| `oracle_versions.vllm` | str | Installed vllm version |
| `generation.canonical_max_tokens` | int | Max generated tokens for canonical prompts (64) |
| `generation.regression_max_tokens` | int | Max generated tokens for regression prompts (32) |
| `generation.temperature` | float | Sampling temperature (0.0) |
| `generation.attn_implementation` | str | Attention backend (`eager`) |
| `comparison_policy.version` | str | Supported comparison semantics (`same-prefix-v1`) |
| `comparison_policy.l1_near_tie_max_abs_logit_gap` | float | Maximum expected/actual candidate-logit gap for an explicit L1 near tie |
| `comparison_policy.l2_atol` | float | Absolute tolerance for same-prefix L2 logit comparison |
| `tolerance.atol` | float | Candidate tolerance produced by baseline calibration |
| `tolerance.observed_max_abs_diff` | float | Maximum observed absolute difference across oracle pair |
| `tolerance.calibration_factor` | float | Safety factor applied (2.0) |
| `tolerance.method` | str | Description of calibration method |
| `expected_fixtures` | list[object] | Independent contract for every required oracle artifact |
| `calibrated_fixtures` | list[str] | Baseline fixture IDs successfully consumed by calibration |
| `fixtures` | list[object] | List of `FixtureMetadata` records |

Each `expected_fixtures` entry declares the concrete `fixture_id`, `prompt_id`,
fixture `family` (`canonical`, `batch`, or `regression`), immutable
`model_revision`, model `dtype`, oracle name and role (`reference` or
`baseline`), required comparison (`l1`, `l1_l2`, or `calibration`), and exact
filename. Transformers is always the reference; vLLM is always baseline
calibration evidence and never a second pass/fail truth.

The generator prints exact `expected`, `discovered`, `generated`, `compared`,
`skipped`, and `failed` totals. Partial `--only-category` runs remain explicit
through non-zero skipped coverage. Any oracle exception exits non-zero and does
not publish a new manifest.

Each `FixtureMetadata` entry:

| Field | Type | Description |
|-------|------|-------------|
| `prompt_id` | str | Prompt identifier (e.g., `canonical_01`) |
| `category` | str | `canonical` or `regression` |
| `oracle` | str | Oracle name (`transformers` or `vllm`) |
| `num_tokens` | int | Number of generated tokens |
| `logits_dtype` | str | Always `float32` |
| `logits_shape` | [int, int] | Shape of the full logits tensor, or `[0, 0]` for regression |
| `sha256` | str | SHA-256 hex digest of the `.safetensors` file |
| `filename` | str | Fixture filename (`{prompt_id}.{oracle}.safetensors`) |

## Fixture File Format

Each `.safetensors` file contains the following tensors:

| Key | Dtype | Shape | Description |
|-----|-------|-------|-------------|
| `token_ids` | int64 | `[n]` | Generated token IDs |
| `n_prompt_tokens` | int64 | `[]` (scalar) | Number of prompt tokens |
| `logits` (canonical only) | float32 | `[n, 151936]` | Full pre-sampling logits per step |
| `top5_indices` (regression only) | int64 | `[n, 5]` | Top-5 token indices per step |
| `top5_logits` (regression only) | float32 | `[n, 5]` | Top-5 logit values per step |

## Known Oracle Deviations

### HF Transformers `output_logits=True`

The `output_logits` parameter in `transformers>=4.43` returns logits as FP32 regardless
of the model's native dtype. This is the canonical format we store.

### vLLM `logprobs_mode="raw_logits"`

Requires `vllm>=0.10.0` where V1 engine is the default. The `raw_logits` mode returns
the lm_head output directly (not log-softmax). When `logprobs=-1`, each step returns
the full vocabulary. The `logprob` field on the `Logprob` object actually holds the raw
logit value when in raw_logits mode.

### Prompt canonical_05 (batch)

The single PromptSpec with `\n---\n`-separated sub-prompts is treated differently by
each oracle:
- **vLLM**: Natively handles 4 separate prompts in a single continuous-batching step.
- **HF Transformers**: Processes all text as one concatenated sequence — the output is
  one long sequence covering all 4 segments.

Cross-validation on canonical_05 will show deviations between HF and vLLM. The tolerance
calibration accounts for this.

## Upload Release Asset

After generating goldens on a GPU machine, upload them as a GitHub Release (do NOT commit
to git mainline):

```bash
gh release create goldens-v0.1 \
  --repo RedHeartSecretMan/vllm-oxide \
  --title "Golden fixtures -- v0.1" \
  --notes "Generated by tools/golden-gen/. See manifest.json in this release for provenance." \
  output/manifest.json output/*.safetensors
```

The manifest.json contains SHA-256 hashes for every fixture file, so consumers can verify
integrity after download.

## File Layout

```
tools/golden-gen/
├── README.md
├── pyproject.toml
├── .gitignore
├── .python-version
├── prompts/
│   ├── canonical.jsonl       # 5 canonical prompts
│   └── regression.jsonl      # 20 regression prompts
├── src/
│   └── golden_gen/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py            # argparse entrypoint
│       ├── config.py         # Constants
│       ├── schema.py         # Pydantic v2 models
│       ├── prompts.py        # JSONL loader
│       ├── io.py             # safetensors save/load
│       ├── manifest.py       # manifest build/write/read
│       ├── generate.py       # oracle × prompt orchestration
│       ├── calibrate.py      # baseline calibration observations + comparison policy
│       └── oracles/
│           ├── __init__.py
│           ├── base.py       # Oracle protocol + result dataclass
│           ├── fake.py       # Deterministic fake (--dry-run)
│           ├── transformers_oracle.py
│           └── vllm_oracle.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_schema.py
    ├── test_prompts.py
    ├── test_io.py
    ├── test_manifest.py
    ├── test_calibrate.py
    ├── test_fake_oracle.py
    └── test_cli.py
```
