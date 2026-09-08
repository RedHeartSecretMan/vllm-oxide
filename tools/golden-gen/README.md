# golden-gen — Golden Fixture Generator for vllm-oxide

> **Dev-only — NOT in CI.** This harness runs manually on a GPU machine before each
> release to produce golden fixtures. It is NOT run in continuous integration and
> is NOT a dependency of the vllm-oxide Rust crates. See issue #14.

This harness generates **golden fixture files** for the vllm-oxide project with
**HuggingFace Transformers** BF16 SDPA as the authoritative Reference oracle and
**vLLM** BF16 as the Baseline oracle. Transformers supplies expected correctness;
vLLM supplies same-prefix calibration observations and can never override a
reference failure or select acceptance thresholds automatically.

## Layered accuracy protocol (new workflow)

`python -m golden_gen.layered_cli --help` is the versioned
`layered-accuracy-v1` entrypoint. L0 means operator verification, L1 model
numerical verification, and L2 decoding/public behavior verification. The
schema fields are `operator_checks`, `numerical_checks`, and `behavior_checks`.
The sections below labeled **legacy** retain their old L1=token/L2=logits meaning;
their manifests, markers, tolerances and release actions cannot satisfy this protocol.

The new reader requires clean source identity and tracked Definition/index binding
for `docs/validation/layered-accuracy-cases.json` and
`docs/validation/layered-accuracy-budgets.json`. Pending budgets are `INVALID`,
not defaults. The old 28 cases are development inputs, not a fresh holdout.
Every member is a separate numerical case; an execution group only batches them.

This is a CPU-prepared measurement workflow, **not release acceptance**. No GPU
command should run before its separate stage authorization. Use the prepared,
locked runtime with `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and HF,
Transformers, uv and Cargo offline settings established before process startup.
All GPU owners are serial fresh processes under the 16 GiB available-RAM guard.
The candidate binary must already be built from the reviewed source; collection
never invokes Cargo or installs dependencies.

The frozen unique-owner inventory supplies the `--group`, `--engine` and
`--variant` values. The supported stages are:

- `collect`: one fixed-history owner (`primary`/`replay`) or unforced owner
  (`control`/candidate-only `control-replay`), with `--model-dir` and, for the
  candidate, `--candidate-binary`.
- `collect-aux`: one candidate operator suite (`--group operators`) or a
  standalone public behavior case, with `primary` or `replay`.
- `assemble-observation`: hash and assemble every declared calibration and
  auxiliary owner. `--manifest` names the output manifest. Assembly alone is
  nonaccepting and creates no success marker.
- `observe`: validate raw data, replays, actual mechanisms and ownership against
  that manifest. Only complete observations receive an observation marker;
  incomplete/invalid evidence exits 2. Unknown numeric budgets remain `INVALID`
  even when observation collection is complete.
- `faults`: CPU-only isolated fault models based on a complete calibration
  `--manifest`, written to `--output`. Normal input files are hard-linked read-only
  into private temporary mirrors; mutations are written to fresh filenames, never
  into shared inodes. Original artifact hashes are rechecked after simulation.
- `assemble-authoritative`: available only after budget approval; additionally
  requires `--calibration-evidence`, `--calibration-manifest`,
  `--calibration-marker`, and `--fault-evidence` inside the run root.
- `authoritative`: independently reproduce approved calibration/fault evidence
  and assess fresh measurements. Every case's absolute mean/peak KL, paired mean
  KL excess over vLLM, and reference choice loss are independent gates. TV and
  top-5 overlap are diagnostics; the summary is case-equal, never token-weighted.

All stages also take `--repo-root` and `--run-dir`. Keep original calibration
artifacts in a subdirectory of the overall run root; authoritative owners use new
directories, so the original source/marker is not relabeled or overwritten.
Markers rehash the full declared capture/receipt/guard/setup dependency graph.
Definition-only approval changes may bridge calibration to fresh measurements;
execution-source changes require fresh calibration. Old markers are never inputs.

Fixed replay records exactly T predictions for prompt+continuation[:t], with a
separate predicted and advance token. Recording the final advance does not claim
another forward. Controls advance only their own raw greedy predictions. Public
prefix/cache-pressure/waiting cases also retain the actual ordered/detokenized
`LLM::generate` return values and use a separate control replay. Their owners are
shared with collection-equivalence evidence and counted once, not twice. Merely
relabeling forced rows never establishes public behavior coverage.

vLLM request IDs are opaque: its internal ID can differ from the externally
returned ID. Baseline captures declare `request_identity=vllm-owner-local-v1`
and retain both complete strings in an explicit per-member `request_bindings`
table. Capture-local integer IDs are assigned when native requests are first
observed and remain stable across batch moves and setup calls; they are not
parsed from a string prefix or recomputed from a current batch slot. Rows and
execution events must agree with the mapping, and aliases/rebinding are rejected.

No early resolved EOS means missing required coverage (`INVALID`), not a diagnosis
of model inaccuracy or an EOS-state-machine bug. Emitting EOS and violating the
stop/length policy is a behavior `FAIL`. Do not select replacement holdout prompts
after seeing acceptance outcomes.

The new CLI deliberately exposes no tag, release, upload or publication action.
The old shell `authoritative`/`publish` stages and direct publication preparation
stay disabled. The schema5 release adapter below consumes complete layered evidence.
Numerical acceptance is not publication authority: performance evidence, release
report, bundle/clean-consumer verification, final review and explicit publication
authorization remain separate requirements. CPU tests do not establish GPU
numerical accuracy. Dense raw inventories need substantially more disk space than
the legacy corpus; budget storage from the frozen row inventory before collection.

## Schema5 release stages

`python -m golden_gen.release_cli --help` exposes separately invocable `cpu`,
`performance`, `bundle`, `verify`, `report` and `publish` stages. All require
`--repo-root` and a fresh external `--marker`. Failure writes no success marker.
These are CPU-tested implementation paths, not evidence of GPU acceptance.

- `cpu`: `--output` is a fresh log directory; supply the development `--python`,
  prepared `--worker-python`, CPU `--rust-binary` (the verify-bundle executable),
  and shared `--target-dir`. The producer executes nine fixed gates: default
  core, workspace, internal-golden, fmt, Clippy, cargo-deny, pytest, Ruff and
  mypy. Actual commands, environment/toolchain, status and raw logs are retained;
  no supplied PASS summary can replace execution. The default core output must
  show the fixed 128-token pressure regression actually running.
- `performance`: after separate GPU stage authorization, use `--output` for a
  fresh directory, `--model-dir`, `--benchmark-binary` and
  `--authoritative-manifest`. This revalidates authoritative evidence before
  owning a GPU and records current runtime, guarded execution and original
  synchronized telemetry for the ADR-0012 fixed workloads. Never invoke it
  during CPU-only preparation.
- `bundle`: supply `--run-dir`, fresh `--output`, `--authoritative-manifest`,
  `--authoritative-marker`, `--performance-evidence` and `--cpu-evidence`.
  All four evidence entrypoints must be beneath the same run root. The adapter
  snapshots every relevant source's selected Definition bytes without changing
  their original identities, recomputes the evidence and emits the two assets.
- `verify`: use `--bundle-dir`, `--cache-dir` and CPU `--rust-binary`. It runs
  Rust and Python transport readers on the actual two files, reconstructs the
  closure in a private fresh cache, and re-evaluates raw semantics and performance
  before exposing the verified immutable install. The Rust flag
  `--layered-transport` alone explicitly returns `accepting=false`.
- `report`: the same verification arguments plus a fresh external `--output`
  produce Markdown. Commit those exact bytes at `docs/releases/goldens-v0.2.md`
  in the evidence-only candidate; do not add the report to the two asset files.
- `publish`: the verification arguments plus `--review`, `--review-base` and
  independent `VLLM_OXIDE_ALLOW_GOLDEN_PUBLISH=goldens-v0.2` authority are required.
  The review JSON uses protocol `layered-accuracy-v1`, schema1,
  kind `full_candidate_review`, exact `base`, `candidate: {commit,tree}` and
  SHA256 of the complete `git diff --binary BASE...CANDIDATE`. `axes.standards`
  and `axes.spec` each bind an original report `{path,sha256}` relative to the
  review JSON and require `unresolved_findings: []`. The adapter recomputes the
  bundle and report, refuses existing/partial remote state, creates one direct
  tag and release, uploads exactly two assets and validates downloaded bytes.
  Failed remote operations are never automatically overwritten or repaired.

Release schema5 bounds are 16 MiB per manifest, 50,000 artifacts, 8 GiB minus
one byte per artifact (the USTAR field limit), and 1 TiB total extracted bytes.
Artifact filenames are `artifact-NNNNNN.bin` in sorted ASCII logical-path order;
identical bytes at different logical paths remain distinct records. Consumers
enforce canonical gzip/USTAR headers, exact lengths/hashes, bounded streaming,
complete inventories and immutable no-replace installation. Disk capacity is
checked before packing/extraction. GitHub's per-asset limit, verified on
2026-09-08, is strictly less than 2 GiB; equal or larger assets require a capacity
decision, never silent splitting, evidence omission or encoding changes.

The committed registry/budgets remain authoritative. Pending budgets or any
missing source, owner, replay, raw telemetry, CPU gate or original evidence
reject the new release path. Final candidate review and publication authorization
stay outside the archive to avoid self-reference. Tests use synthetic evidence
and injected transports; they never establish Qwen GPU correctness or publication
permission.

## Legacy workflow prerequisites (historical)

- Linux with one NVIDIA sm_89 GPU and no unrelated CUDA compute process
- At least 32 GiB host RAM; every GPU-owning stage stops below 16 GiB available
- Python 3.12, PyTorch 2.10.0 (CUDA 12.8), Transformers 4.57.6,
  vLLM 0.18.1, xgrammar 0.2.3, and Triton 3.6.0 from `uv.lock`
- CUDA toolkit 13.2.51 for the Rust/Candle candidate
- `uv` package manager (see [docs.astral.sh/uv](https://docs.astral.sh/uv/))
- About 12 GiB free disk on a fresh host; the current release keeps all machine
  evidence below `/tmp/vllm-oxide-dag-v0.2.0/t45-artifacts`

## Legacy same-prefix release workflow

`tools/validate-release.sh` exposes separately invocable, content-marked stages:

- `env` installs registry dependencies from locked wheels only, then validates
  exact model/tokenizer hashes and runtime identities. It retains actual installer
  logs and matches every live registry distribution's wheel tags/build to the lock.
- `generate` runs Transformers and vLLM in separate fresh processes twice,
  verifies bit-identical replay, and assembles exactly 28 cases / 56 assets.
- `calibrate` records baseline evidence without selecting acceptance thresholds.
- `observe` opens only `canonical_01`, `canonical_02`, `canonical_03`, and
  `canonical_05a`; it keeps the other four full-logit cases sealed and always
  exits non-zero (`3`) after writing the normalized observation.
- `authoritative` is unavailable until that observation is added through a
  reviewed Definition checkpoint. It then runs and replays all 28 candidate cases.
- `benchmark`, `report`, and `bundle` produce the fixed performance evidence,
  evidence-only report, and exact two-asset bundle.
- `verify-local` exercises the production Rust downloader against that exact pair
  in a new cache before any publication operation is available.
- `publish` and `verify` are separate. `publish` is inert unless the user supplies
  the explicit publication guard and frozen candidate identity.

Start each release attempt with a fresh run root:

```bash
./tools/validate-release.sh env \
  /tmp/vllm-oxide-dag-v0.2.0/t45-artifacts/<run-id> \
  /path/to/Qwen3-0.6B
```

Run later stages one at a time with the same root. Never use
`tools/golden-gen/output/`: it is ignored legacy working state and is rejected as
release evidence. Each successful stage creates a JSON marker binding the
generator commit/tree, predecessor marker, and output checksums. A failed stage
creates no successor marker.

Generation, baseline calibration, and policy approval keep separate immutable
fixture/manifest copies. Marker checks recursively rehash predecessor outputs and
reject changed executable inputs. GPU owners run under a process-lifetime RAM
watchdog, with disk/RAM/GPU evidence before and after execution. Rust GPU runners
also check their compiler-produced source fingerprint against the clean reviewed
worktree. Reports retain complete runtime identities and every benchmark repetition.

The vLLM baseline uses a pinned `worker_cls` subclass because WSL starts a fresh
EngineCore process. That worker enables and verifies deterministic algorithms
before CUDA initialization. Its ready/complete RPC records retain actual flags,
PID, every attention layer's FlashAttention-2 backend, and eager/no-graph settings
in `oracle-run.json`; missing or inconsistent worker evidence fails generation.
The spawn regression is CPU-only and can use the prepared release interpreter:
`GOLDEN_WORKER_TEST_PYTHON=<run-root>/env/.venv/bin/python pytest tests/test_worker_determinism.py`.

```bash
uv run python -m golden_gen --help
```

### Dry-run (no GPU)

```bash
cd tools/golden-gen
uv run python -m golden_gen generate --dry-run --output-dir /tmp/golden-dry-run
```

Produces a schema-valid but uncalibrated fake manifest and synthetic
`.safetensors` fixtures without loading any model or GPU. It exercises lifecycle
plumbing only and is never publishable release evidence.

### Testing (CPU-only)

```bash
cd tools/golden-gen
uv run pytest
```

All unit tests run on CPU and do not require a GPU. The plain `generate` command
is dry-run only; there is no direct real-generation or arbitrary-tolerance bypass.

## Manifest Schema

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

### Batch prompts

Batch prompt specifications are expanded into concrete fixture cases and run through
both oracle adapters. Any vLLM deviation is retained only as Baseline oracle
calibration evidence; the Transformers output remains the Reference oracle target.

## Golden asset bundle contract

The GitHub Release asset set is exactly the local bundle's two files. GitHub's
automatic source archives are not API release assets for this contract, and
individual `.safetensors` files must not be uploaded. Publication is available
only through the separately authorized `publish` stage after the reviewed
candidate and report are frozen:

```bash
VLLM_OXIDE_ALLOW_GOLDEN_PUBLISH=goldens-v0.2 \
VLLM_OXIDE_GOLDEN_CANDIDATE=<reviewed-commit> \
./tools/validate-release.sh publish \
  /tmp/vllm-oxide-dag-v0.2.0/t45-artifacts/<run-id>
```

The archive is gzip-compressed USTAR. It contains every and only the declared
fixture filename at its root, sorted by ASCII filename. Every entry is a regular
file with mode `0644`, uid/gid `0`, empty owner/group names, and mtime `0`; the
gzip header has mtime `0` and no source filename. The manifest is never inside
the archive. A clean consumer downloads the exact two assets, verifies schema,
versions, archive and fixture checksums, and atomically installs them under
`<cache-root>/goldens-v0.2/<archive-sha256>`.

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
│       ├── assets.py         # deterministic archive + exact-two bundle publisher
│       ├── generate.py       # oracle × prompt orchestration
│       ├── calibrate.py      # baseline observations + explicit tolerance policy
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
