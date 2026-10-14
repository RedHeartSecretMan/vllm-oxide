# Release source maintenance

[Documentation](README.md) · [Release history](releases/README.md)

All three published versions received documentation updates and source fixes
on 2026-10-14. Each maintained tag points to a reviewed descendant of its
original implementation.

| Version | Original implementation | Maintained source |
|---|---|---|
| `v0.1.0` | [`5203749`](https://github.com/RedHeartSecretMan/vllm-oxide/commit/52037493b38c2d01c4c07a98d2d8f22ca68f2509) | [`v0.1.0`](https://github.com/RedHeartSecretMan/vllm-oxide/tree/v0.1.0) |
| `v0.2.0.rc.1` | [`21cd848`](https://github.com/RedHeartSecretMan/vllm-oxide/commit/21cd848fb1256dd908a29d4f3b08596fe1dd39fc) | [`v0.2.0.rc.1`](https://github.com/RedHeartSecretMan/vllm-oxide/tree/v0.2.0.rc.1) |
| `v0.2.0` | [`289cb65`](https://github.com/RedHeartSecretMan/vllm-oxide/commit/289cb65b475bb472b5c31db2615c3a5d4cbba0db) | [`v0.2.0`](https://github.com/RedHeartSecretMan/vllm-oxide/tree/v0.2.0) |

## Scope

The maintenance corrects extreme sampling inputs, malformed model configuration,
fixture output paths, unaligned safetensors reads and early EOS in the Python
batch oracle. Unsupported eager-mode options are rejected where required by the
version's API contract. It also reviews the research CUDA prototype, makes its
workspace independent, shares the CLI's production model parser with its tests,
clarifies source-only Cargo distribution and documents CUDA safety boundaries.
Each change is adapted to that version's actual API and tool schema; older
versions do not acquire the final release's scheduler or public API.

## Validation identity

Original accuracy, behavior and performance reports continue to describe their
original implementation and measurement commits. In particular, historical
v0.1.0 token differences and the RC's 18/66 numerical failures retain their
verdicts. The v0.2.0 747-item acceptance and performance baseline retain their
original numerical measurement/evaluation/producer identities. These records
are not renamed as measurements of the maintenance commit.

Maintenance regressions and relevant CUDA checks are recorded separately. Full
raw evidence remains local. Original Release IDs, dates, statuses and attached
assets are preserved; generated source archives contain the reviewed fixes.
Existing clones can retain an older local tag; use the full commit identity from
the Release page when comparing source archives or reproducing a measurement.
Original implementations remain reachable through their descendant histories.

This was an explicitly authorized maintenance update to the three existing
source snapshots. Subsequent behavior releases continue to use new version
numbers. Backups stay local; the repository keeps its three official tags
without extra backup tags or permanent work branches.

## Previous maintenance checks

The earlier focused 2026-10-14 maintenance was checked separately from the original full-model
acceptance records. Complete Rust workspace tests passed for all four source
checkouts, including internal release-tooling tests where that version has the
feature. Complete Python CPU suites had no failures or skipped cases:

| Source checkout | Python tests | Actual CUDA sampler tests |
|---|---:|---:|
| Maintained v0.1.0 | 124 | 1 legacy CUDA-tensor/host-adapter regression |
| Maintained RC | 354 | 7, including 4 new numeric-boundary cases |
| Maintained v0.2.0 | 460 | 9, including 4 new numeric-boundary cases |
| Main maintenance candidate | 460 | 9, including 4 new numeric-boundary cases |

The v0.2.0 and main sampler suites retain one explicitly ignored manual
microbenchmark; every required new boundary test ran. The research prototype's
identical four-version implementation passed both actual GPU phases: fill sum
43,008 and attention shape `[16, 4, 64]` with all 4,096 output values finite.
These are regression and interoperability checks, not a new 747-item numerical
measurement or a new performance baseline. The maintained snapshots pin Rust
1.94.0; early unverified Rust 1.75 declarations are corrected without changing
the existing engine dependency locks or Candle revision.

## Follow-up whole-project review — 2026-10-14

The follow-up review covered tracked source, configuration, and documentation
in all four snapshots, plus the two accuracy research drafts on `main`. Shared
files were reviewed once, then checked for version-specific differences. The
regression results below cover the tested inputs and hardware; they do not
cover every possible prompt or parameter combination.

The additional source fixes cover:

- Reject out-of-vocabulary prompt token IDs before request admission, unsupported
  Qwen3 computation modes, and malformed normalization/model settings; respect
  declared attention bias and Q/K normalization.
- Avoid directory-symlink cycles during cached Hub discovery, ignore directories
  named like tensor shards, report directory errors and fetch all validated
  indexed shards for a cold v0.1.0 Hub cache.
- Handle degenerate sampling inputs and validate CLI temperature/top-k/top-p/token
  limits before loading a model. Fixture publication and manifest errors now
  return failures while preserving retained output.
- Make the v0.1.0 comparator reject empty or missing declared evidence, propagate
  errors, load expanded canonical and regression prompts, and compare the full
  shared causal prefix through its first divergent row. Its L1 helper now
  recognizes a reference runner-up while requiring the actual candidate argmax;
  the original `2 × atol` default and regression skip map remain unchanged.
  All four versions refuse empty/inconsistent L2 data and invalid tolerances,
  and non-finite compared values cannot pass. Requesting the unimplemented L3 comparison returns an explicit
  error. Original measured failures remain
  separately identifiable.
- Audit the CUDA dependency graph as metadata with all features selected; the
  audit does not compile CUDA. The maintained RC CI explicitly tests Rust 1.94.0
  and stable with default and internal-golden suites, plus native cargo-deny.
  The v0.1.0 source now also contains the project's Apache-2.0 `LICENSE` file.
  RC/final/main CPU validation tools build from source archives without `.git`,
  with an explicitly unavailable receipt; authoritative stages still need a
  reviewed Git checkout and matching source-bound binaries.

Historical fixture assets, original model/policy/registry bytes, numerical
budgets and all original PASS/FAIL sources retain their identities. This review
adds no 747-item model-accuracy rerun or new performance baseline. Research
prototype and source regression results are reported only for the paths that
actually ran.

### Follow-up verification status

The actual CUDA sampler regressions completed independently of the historical
full-model acceptance:

| Source checkout | Executed CUDA sampler tests | Ignored tests |
|---|---:|---:|
| Maintained v0.1.0 | 1 | 0 |
| Maintained RC | 9 | 0 |
| Maintained v0.2.0 | 11 | 1 manual microbenchmark |
| Main review candidate | 11 | 1 manual microbenchmark |

The new required CUDA boundary cases executed; only the explicitly manual
microbenchmark stays ignored. CLI parser regressions also ran successfully in
all four checkouts. Complete default Rust workspace tests and Clippy passed
for all four sources; internal-golden tests also passed where the feature exists.
All four source archives built the CPU workspace without Git metadata. Ruff,
mypy and local documentation links passed, as did the main public Rust API docs.

| Source checkout | Complete Python CPU tests | Failures / errors / skips |
|---|---:|---:|
| Maintained v0.1.0 | 147 | 0 / 0 / 0 |
| Maintained RC | 368 | 0 / 0 / 0 |
| Maintained v0.2.0 | 473 | 0 / 0 / 0 |
| Main review candidate | 473 | 0 / 0 / 0 |

The legacy v0.1.0 generator now returns failure without publishing an incomplete
manifest when an oracle fails. Its validation wrapper uses fresh local fixtures,
a locked GPU environment and the CUDA build feature, retaining local logs without
automatic Release publication. These are source-maintenance regression results,
not a new full-model numerical verdict or performance measurement.
