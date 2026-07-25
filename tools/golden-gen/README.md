# golden-gen — Golden Fixture Generator for vllm-oxide

> **Dev-only — NOT in CI.** This harness runs manually on a GPU machine before each
> release to produce golden fixtures. It is NOT run in continuous integration and
> is NOT a dependency of the vllm-oxide Rust crates. See issue #14.

This harness generates **golden fixture files** for the vllm-oxide project by running three independent oracle LLM engines (HuggingFace Transformers, nano-vllm, and vLLM V1) on a fixed set of prompts. The oracle triangle cross-validation calibrates numerical tolerances so the Rust port can verify correctness against known-good outputs without needing a GPU for every test.

## Prerequisites

- Linux with an NVIDIA GPU (sm_89+; a single A10 is sufficient)
- Python 3.11
- `uv` package manager (see [docs.astral.sh/uv](https://docs.astral.sh/uv/))
- ~6 GB free disk space for model weights and oracle caches

## Install

```bash
cd tools/golden-gen
uv sync
```

### nano-vllm dependency

nano-vllm is loaded from a local path. By default the harness expects it at `/tmp/opencode/nano-vllm`:

```bash
git clone https://github.com/RedHeartSecretMan/nano-vllm /tmp/opencode/nano-vllm
```

To override the path, edit `pyproject.toml` under `[tool.uv.sources]`:

```toml
[tool.uv.sources]
nano-vllm = { path = "/your/path/to/nano-vllm", editable = true }
```

## Usage

### Run for real (GPU required)

```bash
cd tools/golden-gen
uv run python -m golden_gen
```

This loads all three oracles, runs all 25 prompts, cross-validates the results, and writes `output/manifest.json` + `output/*.safetensors`. Expect ~5-15 minutes on an A10.

Options:

| Flag | Description |
|------|-------------|
| `--dry-run` | Use fake oracle (no GPU, no model download) — for CI / smoke testing |
| `--output-dir PATH` | Output directory (default: `./output`) |
| `--only-oracle NAME` | Run only one oracle (`transformers`, `nanovllm`, or `vllm_v1`); repeatable |
| `--only-category {canonical,regression}` | Run only one prompt category |
| `--no-cross-validate` | Skip pairwise L2 comparison and tolerance calibration |

```bash
uv run python -m golden_gen --help   # full usage
```

### Dry-run (no GPU)

```bash
cd tools/golden-gen
uv run python -m golden_gen --dry-run
```

Produces a valid fake manifest and synthetic `.safetensors` fixtures without loading any model or GPU.

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
| `schema_version` | int | Manifest format version (currently 1) |
| `generated_at` | ISO 8601 | UTC timestamp of generation |
| `model.id` | str | HuggingFace model ID (`Qwen/Qwen3-0.6B`) |
| `model.revision` | str | Git revision (commit hash) of the model weights |
| `model.arch` | str | Model architecture (`Qwen3ForCausalLM`) |
| `model.dtype` | str | Model dtype (`bfloat16`) |
| `model.vocab_size` | int | Vocabulary size (151936) |
| `oracle_versions.transformers` | str | Installed transformers version |
| `oracle_versions.vllm` | str | Installed vllm version |
| `oracle_versions.nanovllm` | str | Installed nanovllm version |
| `generation.canonical_max_tokens` | int | Max generated tokens for canonical prompts (64) |
| `generation.regression_max_tokens` | int | Max generated tokens for regression prompts (32) |
| `generation.temperature` | float | Sampling temperature (1e-9 for nano-vllm) |
| `generation.attn_implementation` | str | Attention backend (`eager`) |
| `tolerance.atol` | float | Absolute tolerance for logit comparison |
| `tolerance.rtol` | float | Relative tolerance for logit comparison |
| `tolerance.observed_max_l2` | float | Maximum observed L2 distance across all oracle pairs |
| `tolerance.calibration_factor` | float | Safety factor applied (2.0) |
| `tolerance.method` | str | Description of calibration method |
| `cross_validation` | list[object] | List of `KnownDeviation` records |
| `fixtures` | list[object] | List of `FixtureMetadata` records |

Each `FixtureMetadata` entry:

| Field | Type | Description |
|-------|------|-------------|
| `prompt_id` | str | Prompt identifier (e.g., `canonical_01`) |
| `category` | str | `canonical` or `regression` |
| `oracle` | str | Oracle name (`transformers`, `nanovllm`, `vllm_v1`) |
| `num_tokens` | int | Number of generated tokens |
| `logits_dtype` | str | Always `float32` |
| `logits_shape` | [int, int] | Shape of the full logits tensor, or `[0, 0]` for regression |
| `sha256` | str | SHA-256 hex digest of the `.safetensors` file |
| `filename` | str | Fixture filename (`{prompt_id}.{oracle}.safetensors`) |

Each `KnownDeviation` entry:

| Field | Type | Description |
|-------|------|-------------|
| `pair` | [str, str] | The two oracles compared |
| `prompt_id` | str | Prompt that showed deviation |
| `max_l2` | float | Max per-step L2 distance on shared prefix |
| `argmax_mismatches` | int | Count of positions where argmax tokens differ |
| `note` | str | Human-readable explanation |

## Fixture File Format

Each `.safetensors` file contains the following tensors:

| Key | Dtype | Shape | Description |
|-----|-------|-------|-------------|
| `token_ids` | int64 | `[n]` | Generated token IDs |
| `n_prompt_tokens` | int64 | `[]` (scalar) | Number of prompt tokens |
| `logits` (canonical only) | float32 | `[n, 151936]` | Full pre-sampling logits per step |
| `top5_indices` (regression only) | int64 | `[n, 5]` | Top-5 token indices per step |
| `top5_logits` (regression only) | float32 | `[n, 5]` | Top-5 logit values per step |

For canonical_05 (batch), the prompt string contains 4 sequences separated by `\n---\n`. At generation time each oracle may handle this differently (vLLM treats them as a single continuous batch, HF generates one response with all 4 segments concatenated). This is documented as a known deviation.

## Known Oracle Deviations

### HF Transformers `output_logits=True`

The `output_logits` parameter in `transformers>=4.43` returns logits as FP32 regardless of the model's native dtype. This is the canonical format we store.

### vLLM V1 `logprobs_mode="raw_logits"`

Requires `vllm>=0.10.0` where V1 engine is the default. The `raw_logits` mode returns the lm_head output directly (not log-softmax). When `logprobs=-1`, each step returns the full vocabulary. The `logprob` field on the `Logprob` object actually holds the raw logit value when in raw_logits mode.

### nano-vllm local path requirement + revision pinning

nano-vllm's `Config.__post_init__` asserts `os.path.isdir(self.model)`, so we must pass a **local directory path**, not a HuggingFace hub ID. The harness uses `huggingface_hub.snapshot_download` with the pinned `MODEL_REVISION` to materialize the model weights in HuggingFace's cache, then passes that local path to `LLM()`. This also ensures the exact weight revision is used (reproducibility).

### nano-vllm monkey-patch

nano-vllm does not expose logits through its public API. The harness monkey-patches `ModelRunner.run` to stash per-step logits (converted to FP32) on each `Sequence` object via `_golden_logits_list`. Sequences are tracked on the model_runner instance (`_golden_tracked_seqs`) so they can be retrieved after `generate()` completes (sequences are deallocated from the scheduler after generation). The tracked sequences are matched to outputs by creation order — nano-vllm creates sequences in `add_request()` order and returns outputs sorted by `seq_id`, so index `i` in tracked sequences corresponds to index `i` in outputs.

The patch is applied before model initialization.

nano-vllm also forbids `temperature=0` (asserts `temperature > 1e-10`). We use `temperature=1e-9`, which produces an effectively greedy near-one-hot softmax. The small residual temperature causes no measurable difference in argmax token selection for BF16 precision.

### Prompt canonical_05 (batch)

The single PromptSpec with `\n---\n`-separated sub-prompts is treated differently by each oracle:
- **vLLM V1**: Natively handles 4 separate prompts in a single continuous-batching step.
- **nano-vllm**: Similar batching handled at the scheduler level.
- **HF Transformers**: Processes all text as one concatenated sequence — the output is one long sequence covering all 4 segments.

Cross-validation on canonical_05 will show deviations between HF and the other two oracles. The tolerance calibration accounts for this.

## Upload Release Asset

After generating goldens on a GPU machine, upload them as a GitHub Release (do NOT commit to git mainline):

```bash
gh release create goldens-v0.1 \
  --repo RedHeartSecretMan/vllm-oxide \
  --title "Golden fixtures -- v0.1" \
  --notes "Generated by tools/golden-gen/. See manifest.json in this release for provenance." \
  output/manifest.json output/*.safetensors
```

The manifest.json contains SHA-256 hashes for every fixture file, so consumers can verify integrity after download.

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
│       ├── generate.py       # oracle x prompt orchestration
│       ├── cross_validate.py # pairwise L2 + tolerance calibration
│       └── oracles/
│           ├── __init__.py
│           ├── base.py       # Oracle protocol + result dataclass
│           ├── fake.py       # Deterministic fake (--dry-run)
│           ├── transformers_oracle.py
│           ├── nanovllm_oracle.py
│           └── vllm_v1_oracle.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_schema.py
    ├── test_prompts.py
    ├── test_io.py
    ├── test_manifest.py
    ├── test_cross_validate.py
    ├── test_fake_oracle.py
    └── test_cli.py
```
