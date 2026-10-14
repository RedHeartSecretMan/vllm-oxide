# golden-gen — v0.1.0 fixture tooling

[Tools](../README.md) · [Historical validation](../../docs/validation/README.md) · [Rust comparison](../../crates/vllm-oxide-test/README.md)

This developer tool generates paired fixtures from **Transformers BF16/SDPA** and **vLLM BF16**, then calibrates the historical schema-1 absolute logit tolerance. The Rust runtime does not depend on this Python package. Real oracle inference is a manual GPU operation; CPU tests and synthetic fixtures validate tooling only.

The guide describes the maintained v0.1.0 source. For new acceptance work, use the [v0.2.0 tooling](https://github.com/RedHeartSecretMan/vllm-oxide/blob/v0.2.0/tools/golden-gen/README.md). Generation fails when any requested oracle/prompt or oracle cleanup fails, and publishes a manifest only after successful completion. It can reuse existing output/tolerance after success; inspect actual coverage and logs before interpreting the historical calibration as numerical acceptance.

## Environment and installation

The manifest and `.python-version` require **Python 3.12** (`>=3.12,<3.13`). Install [uv](https://docs.astral.sh/uv/) and use the committed `uv.lock`. Real inference needs Linux (including WSL2), a CUDA-capable NVIDIA GPU, compatible driver and enough host/device memory and disk for both oracle environments, the pinned model and full-vocabulary logits. Historical project measurement used RTX 4080/SM 8.9; no A10, minimum-VRAM or duration guarantee is claimed.

Choose the needed extra, from the repository root:

```bash
cd tools/golden-gen
uv sync --locked --extra dev
uv run --locked python -m golden_gen --help
```

For real oracle generation in this same directory:

```bash
uv sync --locked --extra gpu --extra dev
```

The base package installs NumPy, safetensors, Pydantic and Hub support. `dev` adds pytest/Ruff/mypy; `gpu` adds Transformers, vLLM, Torch and xgrammar. The declared ranges are Transformers `>=4.56,<6`, vLLM `>=0.18,<0.27`, Torch `>=2.10,<2.11` and xgrammar `>=0.1.32,<0.2.4`; the lockfile selects exact packages. These ranges are not proof that every combination was GPU tested.

## CPU tests and synthetic smoke

With the `dev` environment:

```bash
uv run --locked --extra dev pytest
uv run --locked --extra dev ruff check .
uv run --locked --extra dev ruff format --check .
uv run --locked --extra dev mypy src
uv run --locked python -m golden_gen generate --dry-run --output-dir /tmp/vllm-oxide-v010-synthetic
```

The fake adapter uses deterministic synthetic data and no GPU/model download. Dry-run fixtures are saved under the oracle labels `transformers` and `vllm`; those labels do not make them measured oracle output. Keep this directory separate from real fixtures, and never publish synthetic data as release assets or use it to establish numerical correctness.

Unit tests cover schema validation, prompt loading/expansion, fixture I/O/hashes, calibration arithmetic, fake adapters and CLI orchestration. GPU extras need not be installed for these CPU tests.

## Real generation and calibration

Install the `gpu` extra first and select a fresh output directory. Run from `tools/golden-gen/`:

```bash
export V010_GOLDEN_OUTPUT=/tmp/vllm-oxide-v010-oracles
uv run --locked --extra gpu python -m golden_gen generate --output-dir "$V010_GOLDEN_OUTPUT"
uv run --locked python -m golden_gen calibrate --manifest-dir "$V010_GOLDEN_OUTPUT"
```

Generation loads vLLM first, then Transformers, closing each adapter after use. It writes fixtures and `manifest.json`; uncalibrated output initially has `atol=0` with a pending method. Calibration reads paired canonical full logits, computes `atol = 2 × max(abs(transformers − vllm))` across available pairs/shared output lengths, derives regression skip positions and overwrites the manifest. It does **not** compute rtol, KL budgets, paired excess, layer metrics or a performance verdict.

| Command / option | Behavior |
|---|---|
| `generate --dry-run` | Replace both adapters with fake data |
| `generate --output-dir PATH` | Manifest/fixture destination; default `./output` relative to the current directory |
| `generate --only-category canonical` | Generate only canonical prompt groups |
| `generate --only-category regression` | Generate only regression prompt groups |
| `calibrate --manifest-dir PATH` | Required directory containing manifest and fixtures; updates manifest in place |

There are no model/revision/seed/dtype/backend overrides on this CLI. They come from `src/golden_gen/config.py` and adapter source. `--manifest-dir` belongs to `calibrate`, while `--output-dir` belongs to `generate`.

On success, the generator merges existing manifest entries by `(oracle, prompt_id)` and may preserve an existing positive tolerance. Oracle construction, generation and cleanup failures return nonzero without writing a new or replacement manifest. Existing manifest bytes remain unchanged on such a failed run and cannot stand in for current completion; completed fixture files may remain for diagnosis. Calibration still skips missing pairs/full logits and can return zero tolerance if no usable pair exists. Use complete, fresh data and inspect coverage before drawing conclusions.

## Fixed model, prompts and oracle paths

| Setting | Historical value |
|---|---|
| Model | `Qwen/Qwen3-0.6B` |
| Revision | `7e4ae267688d671ddfca3122e4528ee980cf3234` |
| Architecture / vocabulary | `Qwen3ForCausalLM` / `151936` |
| Model weights/computation | BF16 for both oracles |
| Transport logits | FP32 |
| Generation | Greedy; canonical max 64 tokens, regression max 32 |
| Transformers attention | SDPA, `output_logits=True` |
| vLLM setup | Eager mode, `logprobs_mode="raw_logits"`, `max_logprobs=-1`, GPU memory utilization `0.65` |

Five canonical specifications expand to **eight** groups: four single prompts plus `canonical_05a`–`canonical_05d`. The batch specifies four independent `sub_prompts`. vLLM submits a batch; Transformers uses left padding with attention masks, preserving per-request outputs. It does not concatenate the four prompts into one sequence. The 20 regression specifications remain 20 groups. A complete two-oracle run therefore yields **56 fixtures**, with 16 canonical/full-logit files and 40 regression/top-5 files.

Canonical prompts store full pre-sampling logits. The vLLM adapter requests `logprobs=-1` and rejects an incomplete vocabulary; `Logprob.logprob` carries raw logits in this mode, not normalized log probabilities. Regression output stores the top five candidates. Both adapters preserve completion tokens separately from prompt length; prompts are fixed text and any chat formatting is already in the prompt data.

The calibration compares independently generated trajectories on shared lengths; it does not replay a teacher-forced common prefix. Oracle divergence can therefore enlarge the global tolerance. The original release's `66.46875` tolerance and six regression cases with skipped positions remain historical facts, not a recommended new numerical budget.

## Manifest and fixture format

`manifest.json` follows [schema.py](src/golden_gen/schema.py), **schema version 1**:

| Section | Fields and meaning |
|---|---|
| Top level | `schema_version`, `generated_at`, `fixtures`, `regression_skip_map` |
| `model` | `id`, `revision`, `arch`, `dtype`, `vocab_size` |
| `oracle_versions` | Installed Transformers/vLLM distribution versions; `unknown` if absent |
| `generation` | `canonical_max_tokens`, `regression_max_tokens`, `temperature`, `attn_implementation` |
| `tolerance` | `atol`, `observed_max_abs_diff`, `calibration_factor`, `method`; no `rtol` |
| Fixture metadata | `prompt_id`, `category`, `oracle`, `num_tokens`, `logits_dtype`, `logits_shape`, `sha256`, `filename` |

Fixtures are named `{prompt_id}.{oracle}.safetensors`; SHA-256 covers each file's final bytes. Canonical metadata shape is `[generated_tokens, 151936]`; regression metadata uses `[0, 0]` because full logits are absent.

| Tensor | Dtype | Shape | Coverage |
|---|---|---|---|
| `token_ids` | int64 | `[n]` | Completion IDs in all fixtures |
| `n_prompt_tokens` | int64 | Scalar | Unpadded prompt-token count |
| `logits` | float32 | `[n, 151936]` | Canonical groups |
| `top5_indices` | int64 | `[n, 5]` | Regression groups |
| `top5_logits` | float32 | `[n, 5]` | Regression groups |

This schema does not contain modern layer traces, native runtime/source fingerprints, direct NVML measurements, synchronized stage timestamps or complete-case acceptance markers. Pair it only with the historical [Rust reader](../../crates/vllm-oxide-test/README.md); do not advertise it as schema-5 acceptance.

## Original assets, outputs and cleanup

The [v0.1.0 release](../../docs/releases/v0.1.0.en.md) retains the original `manifest.json` and `goldens-v0.1.tar.gz`, with original hashes. The Rust downloader expects separately attached safetensors files, so for this release use the extracted archive and local `--manifest` mode. Inspect the tar listing and ensure extraction places fixture files next to the selected manifest; archive layouts should not be guessed.

Generated default `output/`, `.venv/`, bytecode and Python tool caches are ignored by this directory's `.gitignore`. An arbitrary custom output directory is not automatically ignored by name. Choose `/tmp` or another external location for experiments, never overwrite original retained evidence, and keep synthetic/real outputs separate.

After retaining any needed logs and results, remove only the explicit scratch directory for that run. Environments and caches can be recreated, but model caches and original release evidence may belong to other work. Keep raw evidence packages local. The maintained [local validation wrapper](../README.md#historical-release-wrapper) creates a fresh temporary GPU environment and fixture directory, retains stage logs, uses CUDA features for the Rust comparison and performs no remote publication.

## Source map

| Path | Responsibility |
|---|---|
| [`cli.py`](src/golden_gen/cli.py) | Argument parsing, generation orchestration and calibration updates |
| [`config.py`](src/golden_gen/config.py), [`prompts/`](prompts/) | Fixed model/runtime configuration and prompt specifications |
| [`generate.py`](src/golden_gen/generate.py) | Oracle/prompt traversal, batch expansion and fixture naming |
| [`schema.py`](src/golden_gen/schema.py), [`manifest.py`](src/golden_gen/manifest.py) | Pydantic records and provenance serialization |
| [`io.py`](src/golden_gen/io.py) | Safetensors encoding/loading and SHA-256 |
| [`calibrate.py`](src/golden_gen/calibrate.py) | Pairwise absolute tolerance and regression skip map |
| [`oracles/`](src/golden_gen/oracles/) | Fake, Transformers and vLLM adapters |
| [`tests/`](tests/) | CPU tooling tests |

## Fixture output and early completion

Prompt IDs use portable ASCII file identifiers. Inventory validation rejects
path components, duplicate or case-colliding IDs and collisions introduced by
batch expansion before invoking an oracle. Output directories and files cannot
be symbolic links; hard-linked and non-regular targets are rejected. Fixture generation targets Linux/WSL2 with atomic no-replace rename support.
A fresh fixture is published only after its complete contents have been written. Deterministic retries reuse identical bytes,
while different existing contents require a fresh output directory.

The Transformers oracle uses the actual generation step count. Each batch
member retains its first EOS step and excludes later padding, including when
padding uses the EOS token ID. An empty output or mismatched token/logit lengths
is an error. These corrections do not alter the historical fixture assets or
recorded validation results.
