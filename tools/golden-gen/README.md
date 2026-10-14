# golden-gen: validation tooling

Use this private Python package for `layered-accuracy-v1` validation. L0 checks
operators and state invariants, L1 compares model numerics, and L2 checks decoding
and public behavior. Hugging Face Transformers BF16 is the reference oracle;
[vLLM](https://github.com/vllm-project/vllm) BF16 supplies baseline observations.
Reference failures cannot be waived by the baseline, and budgets require
explicit approval.

The package also provides CPU adapter/transport tests, fixed-workload performance
validation, local evidence packaging, and the retained historical schema-v4
fixture tools. It is not a runtime dependency of the inference engine and is not
published to PyPI.

| Start here | Purpose |
|---|---|
| [Validation guide](../../docs/validation/README.md) | Current protocol, entrypoints and evidence boundaries |
| [Development guide](../../docs/development.md) | Repository setup and ordinary CPU CI |
| [v0.2.0 validation summary](../../docs/releases/goldens-v0.2.md) | Actual released precision, behavior and performance results |
| [Rust tools](../../crates/vllm-oxide-test/README.md) | Capture, benchmark and CPU transport executables |
| [Repository tools](../README.md) | Documentation checks and the relationship between current and historical tooling |
| [Diagnostic scripts](scripts/README.md) | Exact BF16 probe and fixed historical layer-localization commands, inputs and outputs |

Ordinary GitHub CI runs CPU tests, static checks and validation-tool adapters.
Actual model collection, CUDA public smoke and performance measurements run on a
local GPU. Linux and WSL2 are supported; a GitHub GPU runner is optional. A green
CPU suite or a synthetic fixture does not establish model accuracy. Complete raw
evidence remains local, outside Git, including captures, receipts, guards,
original logs and telemetry. Product source releases do not imply that a golden
asset bundle was published or that an old combined bundle passed full verification.

## Documentation revision and source provenance

The maintained v0.2.0 archive includes documentation improvements and focused
source fixes. Its Git history retains the original implementation. To reproduce
a historical validation stage, use its original measurement/evaluator checkout,
Definition bytes, binaries, calibration records, and retained inputs from the
[validation summary](../../docs/releases/goldens-v0.2.md). The maintenance commit
has a separate source identity. See the
[version-specific revision](../../docs/releases/version-doc-update.md) and
[shared maintenance record](../../docs/version-doc-update.md). No new GPU run or
bundled unpublished input is implied by the documentation refresh.

## Setup and CPU tests

Use Python **3.12**, the repository's `uv.lock`, and Rust **1.94+** (the checkout
pins 1.94.0). Run these commands from the repository root:

```bash
uv sync --project tools/golden-gen --locked --extra dev
uv run --project tools/golden-gen --locked --extra dev \
    python -m golden_gen.layered_cli --help
uv run --project tools/golden-gen --locked --extra dev \
    python -m golden_gen.release_cli --help
uv run --project tools/golden-gen --locked --extra dev \
    python -m golden_gen.public_gpu_smoke --help
cargo build --locked -p vllm_oxide_test --bin vllm-oxide-verify-bundle
```

Choose the module for the protocol you need. `golden-gen` / `python -m golden_gen`
opens the retained schema-v4 CLI. Help commands parse arguments without loading
a model. The two current CLIs select an action through an argument and shared
flags; they have no separate installed command for each action.
For example, `python -m golden_gen.layered_cli collect --help` and
`python -m golden_gen.release_cli verify --help` show their parser's complete
flag set; stage-specific requirements are checked when the selected action runs.

Complete adapter tests use two environments, matching CI: the development
interpreter runs pytest/Ruff/mypy; the locked worker interpreter imports
Torch/vLLM in CPU subprocesses. Installing worker wheels does not run GPU
inference. Choose two separate absolute environment paths:

```bash
export CPU_DEV_ENV=/absolute/path/to/golden-dev
export CPU_WORKER_ENV=/absolute/path/to/golden-worker
UV_PROJECT_ENVIRONMENT="$CPU_DEV_ENV" \
    uv sync --project tools/golden-gen --locked --extra dev
UV_PROJECT_ENVIRONMENT="$CPU_WORKER_ENV" \
    uv sync --project tools/golden-gen --locked --all-extras
cargo build --locked -p vllm_oxide_test --bin vllm-oxide-verify-bundle
GOLDEN_WORKER_TEST_PYTHON="$CPU_WORKER_ENV/bin/python" \
GOLDEN_TRANSPORT_TEST_BINARY="$PWD/target/debug/vllm-oxide-verify-bundle" \
PYTEST_ADDOPTS="" PYTHONNOUSERSITE=1 \
    "$CPU_DEV_ENV/bin/python" -m pytest tools/golden-gen/tests -q
"$CPU_DEV_ENV/bin/python" -m ruff check \
    tools/golden-gen/src tools/golden-gen/tests tools/golden-gen/scripts
"$CPU_DEV_ENV/bin/python" -m mypy --config-file tools/golden-gen/pyproject.toml \
    tools/golden-gen/src tools/golden-gen/scripts
```

If `CARGO_TARGET_DIR` is set, point `GOLDEN_TRANSPORT_TEST_BINARY` at that target
directory instead. Missing worker/transport tools, skipped cases and empty test
output cannot satisfy complete CPU evidence. `release_cli cpu` records the nine
fixed CPU gates and their actual command outputs; hand-written PASS summaries
are not substitutes.

## GPU preparation

Production inference supports a Linux CUDA host, including WSL2, with an NVIDIA
SM 8.9+ GPU, a compatible driver and the CUDA Toolkit. The source-bound oracle
and acceptance reproduction tools are stricter: `environment.py` and the
runtime schema require **exactly SM 8.9 and CUDA Toolkit 13.2.51**, alongside the
locked runtime/model identities. These checks preserve the measured profile;
they are not a new publishing rule or a general production-hardware minimum.
Allow the required host/device memory and disk for the declared corpus.

Prepare the locked GPU extras, dependencies and complete pinned checkpoint
before collection. The recorded oracle environment is Python 3.12, PyTorch
2.10.0 (CUDA 12.8), Transformers 4.57.6, vLLM 0.18.1, xgrammar 0.2.3 and Triton
3.6.0. Exact model/config/tokenizer/weight and wheel identities are checked;
package version labels alone do not establish equivalence.

Before starting the prepared measurement process, set `PYTHONHASHSEED=0`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and the required Hugging Face, Transformers,
uv and Cargo offline settings. Run GPU owners serially in fresh processes under
the existing resource guards, including the 16 GiB available-host-RAM floor.
The candidate binary must already be built from the reviewed checkout; layered
collection neither installs dependencies nor invokes Cargo. Use fresh evidence
paths outside the reviewed source and retain failed attempts unchanged. Storage
needs follow the frozen row inventory; do not reuse the historical corpus's
small disk estimate for dense full-logit evidence.

For production build, warmup, repeated `LLM::generate`, mixed input and CLI
coverage, use the separate [public GPU smoke guide](../../docs/validation/public-gpu-smoke.md).
Its result explicitly does not replace numerical acceptance.

For the model source, the recorded checkpoint is
[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) revision
`7e4ae267688d671ddfca3122e4528ee980cf3234`. Supply its complete local directory
through `--model-dir` for the Python CLIs (`--model-path` for Rust producers and
the layer probe). The directory must include `config.json`, `tokenizer.json`
and the single `model.safetensors`; alternative shards or replacement files
are rejected by the release oracle. Model acquisition is a preparation step,
not an implicit download during source-bound collection.

## Layered accuracy protocol (new workflow)

`python -m golden_gen.layered_cli --help` lists the current entrypoint. The
protocol and layer fields are `layered-accuracy-v1`, `operator_checks`,
`numerical_checks`, and `behavior_checks`. Registry and budget bytes come from
[`layered-accuracy-cases.json`](../../docs/validation/layered-accuracy-cases.json)
and [`layered-accuracy-budgets.json`](../../docs/validation/layered-accuracy-budgets.json),
bound to the clean source and tracked Definition index. Pending budgets are
`INVALID`, never defaults. Every member is a numerical case; execution groups
only batch them. The old 28 development cases are not a fresh holdout.

All stages take `--repo-root` and `--run-dir`. Select group/engine/variant values
from the frozen owner inventory; internal `worker` and `worker-aux` actions are
launched by the collector rather than used as manual entrypoints.

| Action | Inputs and result |
|---|---|
| `collect` | `--model-dir`, `--group`, `--engine` (`reference`, `baseline`, `candidate`) and `--variant` (`primary`, `replay`, `control`, `control-replay`). Candidate collection also uses `--candidate-binary`. Collects a fixed-history or declared unforced owner. |
| `collect-aux` | Same owner flags. Collects the candidate `operators` suite or a declared standalone public behavior case with primary/replay variants. |
| `assemble-observation` | `--manifest` names the output; assembles every declared calibration and auxiliary owner. Assembly itself is nonaccepting and creates no success marker. |
| `observe` | Reads `--manifest`, checks raw rows, replays, mechanisms and ownership. Only complete observations receive an observation marker; unknown budgets remain `INVALID`. |
| `faults` | Requires a complete calibration `--manifest`; `--output` is inside the dedicated run root. Generates CPU-only isolated fault evidence and rechecks original hashes. |
| `assemble-authoritative` | In addition to `--manifest`, requires `--calibration-evidence`, `--calibration-manifest`, `--calibration-marker` and `--fault-evidence` inside the run root. Requires approved budgets. |
| `authoritative` | Independently reproduces calibration/fault provenance and assesses fresh declared measurements against every approved gate. |

For a prepared, reviewed measurement checkout, one operator owner can be
collected with the actual auxiliary entrypoint below. The absolute placeholders
must refer to the matching prebuilt binary, pinned local model and a fresh
external evidence root; this single owner is not complete release evidence.

```bash
uv run --project tools/golden-gen --locked --extra gpu \
    python -m golden_gen.layered_cli collect-aux \
    --repo-root "$PWD" --measurement-repo "$PWD" \
    --run-dir /absolute/path/to/fresh-layered-run \
    --model-dir /absolute/path/to/pinned-qwen3-checkpoint \
    --group operators --engine candidate --variant primary \
    --candidate-binary /absolute/path/to/built/vllm-oxide-fixed-prefix
```

For numerical owners use `collect` and the exact declared execution-group ID;
for an operator replay use the separately declared `replay` variant. Do not
derive acceptance from a successful collection exit: it only records that
owner's raw evidence, and complete inventory assembly/evaluation are separate.

Keep calibration inputs and fresh authoritative owners in separate directories
under the overall run root. `--fresh-observation` is valid only for `collect`
and `collect-aux`: it collects the frozen calibration inventory after an
implementation change, stays nonaccepting, and cannot inherit a previous
backend's approval. Definition-only approval changes have their documented
bridge; executable changes require fresh calibration.

The `bounded-telemetry-fresh-v1` policy uses layered manifest schema 3:
collection takes `--measurement-repo`, assembly takes `--supervision-policy`,
and every owner retains a full schema-2 guard and original worker metadata.
It forbids a retained-owner ledger. Historical recovery manifest schema 2
still requires its exact original `--retained-owner-ledger`; do not mix these
profiles. See the [supervision contract](../../docs/validation/layered-supervision-contract.md)
and [fresh evidence ADR](../../docs/adr/0021-fresh-supervised-release-evidence.md).

Fixed replay records exactly T predictions for `prompt + continuation[:t]`,
with separate predicted and advance tokens; the final advance does not claim
another forward. Controls advance only their own raw greedy predictions.
Prefix/cache-pressure/waiting cases retain actual ordered and detokenized
`LLM::generate` outputs plus independent controls. Shared owner evidence is
counted once. Relabeling forced rows does not establish public behavior.

For vLLM, native and externally returned request IDs are opaque complete
strings. Baseline captures retain both in explicit `request_bindings` using
`vllm-owner-local-v1`; capture-local IDs remain stable across batch movement.
Rows, execution events and mappings must agree; aliases and rebinding fail.

Absolute mean/peak KL, paired mean KL excess over the baseline, and reference
choice loss are independent case gates. TV and top-5 overlap are diagnostics;
case summaries are not token-weighted. Missing coverage is `INVALID`, while a
measured numerical or stop/length policy violation is `FAIL`. Do not select new
holdout prompts after inspecting acceptance outcomes. Full definitions live in
the [layered contract](../../docs/validation/layered-accuracy-contract.md).

Markers rehash the complete declared captures, receipts, guards and setup graph.
Changed executable/source inputs, unknown budgets, stale or missing evidence,
empty coverage and failed comparisons reject acceptance. Old markers and raw
FAIL results are never relabeled. This CLI exposes no tag, upload or publication
action.

## Schema5 release stages

`python -m golden_gen.release_cli --help` lists `cpu`, `performance`, `bundle`,
`verify`, `report` and optional `publish`. All require `--repo-root` and a fresh
external `--marker`; failures write no success marker. These implemented paths
and their CPU tests are distinct from actual model acceptance. In particular,
[v0.2.0's product validation summary](../../docs/releases/goldens-v0.2.md) does
not claim that the historical combined bundle passed complete consumer acceptance.

| Stage | Required stage inputs and behavior |
|---|---|
| `cpu` | Fresh `--output`, development `--python`, prepared `--worker-python`, CPU `--rust-binary` and `--target-dir`. Executes default core, workspace, internal-golden, fmt, Clippy, cargo-deny, pytest, Ruff and mypy. |
| `performance` | Fresh `--output`, `--model-dir`, built `--benchmark-binary` and `--authoritative-manifest`. Revalidates authoritative provenance before guarded GPU measurement and records raw synchronized telemetry. |
| `bundle` | `--run-dir`, fresh `--output`, `--authoritative-manifest`, `--authoritative-marker`, `--performance-evidence`, `--cpu-evidence`. All four evidence entrypoints must be under the same run root. Recomputes complete evidence and writes the exact local two-file bundle. |
| `verify` | `--bundle-dir`, fresh `--cache-dir`, CPU `--rust-binary`. Runs Rust/Python readers and reconstructs/evaluates the full raw semantics before immutable installation. |
| `report` | Verification inputs plus fresh external `--output`. Renders the verified Markdown report; it is separate from the asset pair. |
| `publish` | Verification inputs plus `--review`, `--review-base` and independent `VLLM_OXIDE_ALLOW_GOLDEN_PUBLISH=goldens-v0.2` authority. Optional manual GitHub operation; not part of local acceptance. |

For an already prepared local schema5 bundle, the basic consumer command is:

```bash
uv run --project tools/golden-gen --locked --extra dev \
    python -m golden_gen.release_cli verify \
    --repo-root "$PWD" \
    --bundle-dir /absolute/path/to/local-evidence-bundle \
    --cache-dir /absolute/path/to/fresh-verified-cache \
    --rust-binary "$PWD/target/debug/vllm-oxide-verify-bundle" \
    --marker /absolute/path/to/fresh-verify-marker.json
```

Use the actual CPU reader path if `CARGO_TARGET_DIR` differs. A source-routed
bundle also needs the exact approved `--source-role-freeze` and
`--source-role-freeze-sha256`; omission is not permission to ignore its source
roles. The command reconstructs original evidence and checks its semantics;
it does not rerun GPU model inference or make an arbitrary bundle accepting.

The CPU producer retains actual command/environment/toolchain/raw logs and
requires the default-core 128-token mixed pressure regression to run. It rejects
skipped or empty comparisons. CPU gate evidence validates code/tooling behavior,
not GPU model accuracy.

Performance uses the ADR-0012 fixed workloads: one request (`canonical_04`) and
four requests (`canonical_05`), each with one discarded warmup and three measured
64-token greedy repetitions. Original synchronized timing and raw memory traces
are mandatory. Supervised collection binds `--measurement-repo` and the policy's
benchmark binary/build hash; the legacy wrapper is not interchangeable. The
current direct NVML method targets **5 ms** and enforces the unchanged strict
**50 ms** completion-gap and API-span ceilings; a faster sampling target never
excuses a late record. Performance figures are local evidence, not a universal
throughput floor or a vLLM parity claim.

Source-role routing additionally takes both `--source-role-freeze` and its
independently approved `--source-role-freeze-sha256`, and is limited to local
`performance`, `bundle`, `verify` and `report`. It preserves the actual distinct
numerical measurement/evaluator, performance producer and release evaluator
identities. It cannot relabel old measurements or permit arbitrary source edits.
The routed performance stage also requires `--run-dir`, `--authoritative-marker`,
`--cpu-evidence` and `--measurement-repo`. Its CPU envelope/schema and memory
records remain bound to those original sources; a caller-supplied PASS or an
unselected proposal supplies no authority.

## Golden asset bundle contract

The local schema5 bundle contains exactly `manifest.json` and
`goldens-v0.2.tar.gz`, preserving all raw evidence and source relationships. Keep
both files and the original evidence on durable local storage; temporary run
directories can disappear. No GitHub evidence upload or archive
splitting is required. See [ADR-0028](../../docs/adr/0028-local-release-evidence.md)
and the [asset specification](../../docs/validation/layered-release-assets.md).

Schema5 limits are 16 MiB per manifest, 50,000 artifacts, 8 GiB minus one byte
per artifact (USTAR field limit), 1 TiB total extracted bytes and 2 TiB compressed
archive bytes. These finite bounds apply to local packing and Rust/Python
installation. Disk capacity is checked before packing/extraction.

Artifact names are `artifact-NNNNNN.bin` in sorted ASCII logical-path order;
identical bytes at different logical paths remain separate records. The gzip
header has mtime 0 and no source filename. USTAR entries are regular root files
with mode `0644`, uid/gid 0, empty owner/group names and mtime 0. The manifest is
outside the archive. Consumers enforce canonical headers, complete inventory,
exact lengths/hashes, bounded streaming and immutable no-replace installation
under `<cache-root>/goldens-v0.2/<archive-sha256>`. The Rust
`--layered-transport` flag verifies transport only and reports `accepting=false`;
Python verification also evaluates raw semantics.

Optional publication requires an independent full candidate review, not a
synthetic test or an archive's own summary. The review JSON binds protocol
`layered-accuracy-v1`, schema 1, kind `full_candidate_review`, exact `base`,
`candidate: {commit, tree}` and SHA-256 of `git diff --binary BASE...CANDIDATE`.
Both `axes.standards` and `axes.spec` bind original `{path, sha256}` reports
relative to the review JSON and require `unresolved_findings: []`. Review and
publication authority stay outside the archive to avoid self-reference.

`publish` revalidates the bundle/report, refuses existing or partial remote
state, creates the golden tag/release, uploads the exact two assets and checks
downloaded bytes. Each asset must be **strictly smaller than 2 GiB** before any
remote write. This remote limit is not a local bundle limit. Failures are not
automatically overwritten or repaired; splitting archives or dropping evidence
is outside the contract. GitHub's automatic product source archives are not
these golden API assets.

## Historical schema-v4 reference

The retained `python -m golden_gen` CLI and `tools/validate-release.sh` implement
the earlier `same-prefix-v1` corpus, with L1=token, L2=logits, and an unimplemented
debug L3 entrypoint. The original corpus had 28 cases / 56 oracle assets. Its manifests,
tolerances, markers and verdicts keep their old meanings; they do not satisfy
the current L0/L1/L2 protocol. The shell's `authoritative` and `publish` actions
and the old direct publication preparation are disabled.

### Legacy same-prefix release workflow

Historical stages were `env`, `generate`, `calibrate`, `observe`,
`authoritative`, `benchmark`, `report`, `bundle`, `verify-local`, `publish` and
`verify`. Generation ran separate Transformers/vLLM processes twice, required
bit-identical replay and kept baseline calibration separate from correctness.
The observation opened only `canonical_01`, `canonical_02`, `canonical_03` and
`canonical_05a`, left four full-logit cases sealed and exited 3 after recording
its nonaccepting result. It did not approve thresholds or holdout access.
The legacy shell retains its original fixed run-root assumptions; it is not a
general setup or release command for new work.

Historical markers recursively bound predecessor output, runtime, clean
source and binary identity. Generation, calibration and policy approval used
separate immutable copies. The spawn worker's deterministic flags, PID,
FlashAttention-2 layers and eager/no-graph settings were retained in
`oracle-run.json`; missing worker metadata rejected the old run.

### Dry-run (no GPU)

From the repository root:

```bash
uv run --project tools/golden-gen --locked --extra dev python -m golden_gen \
    generate --dry-run --output-dir /absolute/path/to/fresh-golden-dry-run
```

This creates a schema-valid **uncalibrated synthetic** manifest and fixtures to
exercise historical plumbing. It cannot establish model correctness or produce
publishable release evidence. Plain `generate` is dry-run only; there is no
arbitrary-tolerance bypass. `tools/golden-gen/output/` is ignored historical
working state, must not be committed, and is rejected as release evidence.

The following field and tensor tables describe **schema-v4 only** and retain
that format's exact definitions. Schema5 uses opaque raw artifact records and
the layered protocol described above.

## Manifest schema

`manifest.json` is the provenance record for a set of golden fixtures.

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Manifest format version (exactly 4) |
| `product_version` | str | Compatible product release (`v0.2.0`) |
| `golden_version` | str | Golden release name and required release tag (`goldens-v0.2`) |
| `archive.filename` | str | Sole fixture archive asset (`goldens-v0.2.tar.gz`) |
| `archive.sha256` | str | SHA-256 digest of the complete compressed archive |
| `generated_at` | ISO 8601 | UTC timestamp of generation |
| `model.id` | str | HuggingFace model ID (`Qwen/Qwen3-0.6B`) |
| `model.revision` | str | Git revision (commit hash) of the model weights |
| `model.tokenizer_revision` | str | Immutable tokenizer revision (equal to model revision) |
| `model.config_sha256` | str | Pinned `config.json` identity |
| `model.tokenizer_sha256` | str | Pinned `tokenizer.json` identity |
| `model.weights_sha256` | str | Pinned `model.safetensors` identity |
| `model.arch` | str | Model architecture (`Qwen3ForCausalLM`) |
| `model.dtype` | str | Model dtype (`bfloat16`) |
| `model.vocab_size` | int | Vocabulary size (151936) |
| `oracle_versions.transformers` | str | Installed transformers version |
| `oracle_versions.vllm` | str | Installed vllm version |
| `runtime` | object | Exact Python/Rust/CUDA/driver/GPU/OS/lock/wheel identities and deterministic process environment |
| `kernel_paths.reference` | str | Transformers 4.57.6 / PyTorch 2.10 SDPA math path |
| `kernel_paths.baseline` | str | vLLM 0.18.1 eager FlashAttention-2 path |
| `kernel_paths.candidate` | str | Pinned Candle varlen + paged-windowed path |
| `generation.canonical_max_tokens` | int | Max generated tokens for canonical prompts (64) |
| `generation.regression_max_tokens` | int | Max generated tokens for regression prompts (32) |
| `generation.temperature` | float | Sampling temperature (0.0) |
| `generation.attn_implementation` | str | Reference oracle attention backend (`sdpa`) |
| `tolerance_policy.version` | str | Supported comparison semantics (`same-prefix-v1`) |
| `tolerance_policy.dtype` | str | Dtype scope, matched to the manifest model |
| `tolerance_policy.kernel` | str | Exact reference-versus-candidate comparison scope derived from `kernel_paths` |
| `tolerance_policy.l1_near_tie_max_abs_logit_gap` | float | Maximum expected/actual candidate-logit gap for an explicit L1 near tie |
| `tolerance_policy.l2_atol` | float | Absolute tolerance for same-prefix L2 logit comparison |
| `tolerance_policy.rationale` | str | Reviewed reason for the selected thresholds |
| `tolerance_policy.evidence` | list[str] | Evidence identifiers or URIs supporting the selection |
| `baseline_calibration.candidate_atol` | float | Non-authoritative candidate derived from baseline observations |
| `baseline_calibration.observed_max_abs_diff` | float | Maximum observed same-prefix absolute difference |
| `baseline_calibration.calibration_factor` | float | Factor used for the candidate observation (2.0) |
| `baseline_calibration.method` | str | Description of the observation method |
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

## Fixture file format

Each `.safetensors` file contains the following tensors:

| Key | Dtype | Shape | Description |
|-----|-------|-------|-------------|
| `token_ids` | int64 | `[n]` | Generated token IDs |
| `n_prompt_tokens` | int64 | `[]` (scalar) | Number of prompt tokens |
| `logits` (canonical only) | float32 | `[n, 151936]` | Full pre-sampling logits per step |
| `top5_indices` (regression only) | int64 | `[n, 5]` | Top-5 token indices per step |
| `top5_logits` (regression only) | float32 | `[n, 5]` | Top-5 logit values per step |

## Known oracle deviations

### HF Transformers `output_logits=True`

The `output_logits` parameter in `transformers>=4.43` returns logits as FP32 regardless
of the model's native dtype. This is the canonical format we store.

### vLLM `logprobs_mode="raw_logits"`

Requires `vllm>=0.10.0` where V1 engine is the default. The `raw_logits` mode returns
the lm_head output directly (not log-softmax). When `logprobs=-1`, each step returns
the full vocabulary. The `logprob` field on the `Logprob` object actually holds the raw
logit value when in raw_logits mode.

### Batch prompts

Batch prompt specifications are expanded into concrete fixture cases and run through
both oracle adapters. Any vLLM deviation is retained only as Baseline oracle
calibration evidence; the Transformers output remains the Reference oracle target.

## Source map

The three [diagnostic scripts](scripts/README.md) complement this map:
`bf16_reference_probe.py` checks residual/RMSNorm/RoPE operation order,
`silu_reference_probe.py` checks the fixed gated-SiLU rounding example, and
`canonical03_layer_probe.py` verifies and localizes exact historical captures.
Their own guide documents device flags, immutable inputs, stdout reports and
trace layout. They do not generate accepting fixtures or rerun the release gates.

| Module or directory | Responsibility |
|---|---|
| `layered_cli.py`, `layered_workflow.py`, `layered_manifest.py` | Current collection, inventory assembly and source-bound evaluation |
| `fixed_prefix.py`, `fixed_prefix_oracles.py`, `fixed_prefix_vllm.py` | Frozen histories, replay validation and oracle adapters |
| `operator_verification.py`, `behavior_verification.py` | L0 operators and L2 public behavior checks |
| `layered_accuracy.py`, `layered_faults.py` | Numerical metrics, budgets and isolated fault models |
| `release_cli.py`, `release_cpu.py`, `release_performance.py` | CPU evidence and fixed-workload performance stages |
| `release_adapter.py`, `release_transport.py`, `release_roles.py` | Local bundle, raw verification and explicit source-role bindings |
| `release_memory_method.py`, `guard.py`, `supervision.py` | Memory records and guarded execution |
| `public_gpu_smoke.py` | Production build and public API/CLI smoke |
| `cli.py`, `schema.py`, `oracles/` | Retained schema-v4 fixture CLI/schema/oracle tools |
| `prompts/`, `tests/` | Historical prompt inputs and CPU regression/fault tests |

## Common setup and evidence errors

| Symptom | Check and next action |
|---|---|
| `ModuleNotFoundError: golden_gen`, Torch or Transformers | Use `uv run --project tools/golden-gen` with the required locked extras, or the prepared absolute interpreter. The development environment alone does not include the model runtime. |
| Deterministic-environment rejection | Set `PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching a fresh process. Runtime imports must not precede setup. |
| Missing adapter or CPU transport evidence | Build the CPU reader and set `GOLDEN_WORKER_TEST_PYTHON` / `GOLDEN_TRANSPORT_TEST_BINARY` to the prepared tools; inspect the raw test output. A partial or skipped suite is not complete CPU evidence. |
| Model hash or alternative-weight rejection | Locate the complete pinned local checkpoint. Do not edit expected hashes or use a different revision under the same directory name. |
| Source, tree, worker or binary mismatch | Use the reviewed checkout and its matching previously built binary. Source identity includes tracked definitions; replacing only a PASS marker cannot repair the graph. |
| Existing output, marker or cache | Choose the fresh path required by that stage and retain the rejected run. The tools use no-replace writes; deleting earlier evidence to retry is not required. |
| Performance telemetry gap or API-span failure | Preserve all original timing and memory traces and investigate sampler/owner contention. The 50 ms limits stay unchanged; no averaging or omitted repetition repairs a failed record. |
| Bundle verification returns `INVALID` | Read the reported reasons and verify the raw inputs, source topology and finite transport limits. Rust transport-only success does not replace Python semantic evaluation. |

For CUDA builds, hardware visibility, memory sizing and application errors, use
the [engine troubleshooting guide](../../docs/getting-started.md#common-errors).
For collection-policy and source-role questions, use the linked validation
contracts rather than substituting flags from the legacy CLI or shell wrapper.

## Fixture output and early completion

Prompt IDs use portable ASCII file identifiers. Inventory validation rejects
path components, duplicate or case-colliding IDs and collisions introduced by
batch expansion before invoking an oracle. Output directories and files cannot
be symbolic links; hard-linked and non-regular targets are rejected. Fixture generation targets Linux/WSL2 with atomic no-replace rename support.
The generator publishes a fixture only after writing its complete contents.
A deterministic retry can reuse identical bytes; if existing contents differ,
choose a fresh output directory.

The Transformers oracle uses the actual generation step count. Each batch
member retains its first EOS step and excludes later padding, including when
padding uses the EOS token ID. An empty output or mismatched token/logit lengths
is an error. These corrections do not alter the historical fixture assets or
recorded validation results.
