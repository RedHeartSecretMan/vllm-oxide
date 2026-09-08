# Transport complete layered evidence in release schema 5

The layered protocol needs original captures, stage dependencies, calibration and fault evidence, performance records and CPU checks, which ADR-0010's fixtures-only schema 4 cannot express. We adopt the [layered release asset contract](../validation/layered-release-assets.md): one cross-language release manifest at schema 5 and one deterministic archive, preserving every original evidence byte and validating the reconstructed closure. Execution evidence manifests retain their existing schema 1; they are archived evidence, not additional release assets or wrappers around schema 4.

This explicitly supersedes ADR-0010's schema4-only and fixtures-only rules and ADR-0012's schema-v4-specific field layout for new `layered-accuracy-v1` releases. Required model, source, runtime, kernel, policy and evidence identities are preserved through the new contract, not discarded with the old fields. Schema 4 remains historical-only and cannot pass a layered release gate. Retaining schema 4 or uploading a third manifest/report asset was rejected because neither satisfies the approved complete-evidence, two-asset contract.

Ticket #45 owns migration of the Python producer and Rust consumer, semantic release adapter, report and publication tests. #43 retains its accepted historical transport delivery; #46 and #47 consume the new gates. No Ticket is accepted or reopened by this decision, and the historical tracker snapshot and dependency edges remain unchanged. This approval permits implementation and CPU verification, not budget approval, GPU execution, holdout access or tag/release/asset writes.

**Status:** accepted design; implementation and release evidence remain required.
