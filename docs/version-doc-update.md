# Release source maintenance

[Documentation](README.md) · [Release history](releases/README.md)

The source archives include the version-specific documentation refresh and the
2026-10-14 maintenance fixes requested for all three published versions. Each
maintained tag selects a reviewed descendant of its original implementation.

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

Ordinary behavior releases continue to use a new version. This existing-tag
maintenance follows the user's explicit request to repair the three public
source snapshots together. No extra backup tags or permanent work branches are
published.

## Verified maintenance checks

The 2026-10-14 maintenance was checked separately from the original full-model
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
