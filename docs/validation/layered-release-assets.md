# Layered golden release asset contract

This is the normative transport decision adopted by [ADR-0016](../adr/0016-layered-golden-release-assets.md). It does not change the [layered accuracy contract](layered-accuracy-contract.md), frozen cases, pending budgets, model scope or independent publication authority. It replaces incompatible schema4 transport rules, not their safeguards.

## 1. One release manifest and two assets

New releases use a single cross-language **Golden release manifest**, implemented consistently by the Python producer and Rust consumer:

- `schema_version=5`, `protocol=layered-accuracy-v1`;
- `product_version=v0.2.0`, `golden_version=goldens-v0.2`;
- exactly the uploaded assets `manifest.json` and `goldens-v0.2.tar.gz`;
- release tag equal to `golden_version`; GitHub-generated source downloads are not uploaded assets;
- schema 4 is available only through explicit historical reading and cannot satisfy new authoritative, release or publication validation.

The existing schema1 execution evidence manifests retain their meaning and original bytes. They are inventory members, not a second standalone release manifest and not a schema4 wrapper. The committed Markdown report is not an asset.

The release manifest binds the authoritative measurement commit/tree, archive filename/SHA-256, registry and approved-policy SHA-256, frozen Definition index identity, entrypoints and complete artifact inventory. Entrypoints identify the authoritative execution manifest and completion marker, raw performance evidence and mandatory CPU evidence. Counts are derived from the registry and validated evidence, distinguishing cases, groups, prediction/setup rows, owner identities and overlapping uses; summary counts or caller-supplied PASS flags are never substitutes for validation.

Model/weights/config/tokenizer identity, exact runtime, binary and kernel paths, metric algorithm and budget provenance remain mandatory. The release adapter resolves them from the original, hashed evidence and rejects conflicting copies. A transport-valid archive alone is not numerically validated. Unknown protocol/schema, unsupported versions, pending budgets, incomplete evidence or untrusted source prevent release PASS.

## 2. Byte-preserving artifact mapping

Every `artifacts[]` record declares a unique `logical_path`, a unique root `filename`, original `sha256` and integer `size_bytes`. Logical paths are canonical relative POSIX paths beneath one evidence root. Reject absolute paths, empty or dot segments, parent traversal, backslashes, NUL, reserved Git paths, normalization aliases, duplicate paths and file/directory prefix conflicts. Paths and references must not escape the reconstructed root or depend on symlinks.

Assign canonical flat ASCII filenames deterministically from the sorted logical-path inventory. The precise filename spelling is an implementation detail frozen and cross-language tested before measurement. Archive names cannot be nested paths. Each logical file remains a distinct record and archive member even if its bytes equal another file: no silent owner or reference-topology deduplication. Do not rewrite source identities, raw JSON references or original evidence bytes to fit the archive.

The inventory is the complete transitive closure of the declared entrypoints and their protocol dependencies: original authoritative manifest/result/marker; calibration observation and fault evidence supporting approval; captures, controls, independent replays, setup files, public outputs, receipts and resource guards; performance samples and telemetry; CPU gate output; and frozen Definition indexes/documents for the source identities they attest. Declared extra files without an authorized evidence role, missing dependencies or cycles that violate stage provenance are invalid. Model weights, runtime installations, arbitrary workspace files and executable code to run from the archive are not release evidence.

Keep the release manifest outside the archive to avoid digest self-reference. Final candidate/review records and the final report that names both asset hashes are also outside this inventory. Evidence may record several sources: a calibration snapshot must match its own observation source, and an authoritative snapshot its own measurement source. Do not replace all provenance with the final source just because it is a descendant.

## 3. Deterministic bounded archive

The archive contains exactly the declared flat filenames. Preserve ADR-0010 normalization: USTAR regular-file entries sorted by ASCII filename, mode `0644`, uid/gid `0`, empty owner/group names, mtime `0`; gzip mtime `0` and no source filename. Production and test consumers reject links, directories, devices, FIFOs, duplicate or undeclared entries, missing files, noncanonical names/metadata, unsupported extensions, malformed or truncated headers, invalid checksums, size mismatch and unexpected nonzero trailing tar content. Verify the compressed archive digest and each original file digest and length.

Parsing must be bounded before trusting manifest-declared sizes. Freeze finite manifest-size, artifact-count, per-artifact and total extracted-size limits in the reviewed implementation, use checked arithmetic, stream decompression and hashing, and enforce the declared exact sizes within those limits. Reject bombs, overflow and truncation without allocating the complete archive or corpus in RAM. Preflight disk capacity and the destination's current upload limits before real bundling/publication. A capacity failure does not authorize splitting assets, dropping evidence, changing encoding or weakening validation; report it for an explicit decision.

## 4. Clean consumption and immutable installation

A fresh consumer validates the two actual asset files in a new staging area, reconstructs the declared logical tree only after safe-path checks, and independently re-evaluates original authoritative evidence. Recheck full closure, marker dependencies, source/model/runtime identities, owner inventory, histories, raw numerical comparisons, operator/behavior checks and replays. Reproduce performance formulas from raw samples and check mandatory CPU evidence. Do not trust supplied result files or marker existence as proof of their claimed verdicts.

The Python and Rust transport/schema paths must agree on accepted bytes and refusal cases. Clean release verification composes transport verification with the existing layered semantic evaluator; it need not implement the mathematical evaluator twice. The evaluator is trusted reviewed source, never imported or executed from archive payloads. Do not download, install or build a runtime from the archive.

Verification receives the reviewed Git repository for object, ancestry and Definition checks. Each archived Definition index and selected document must equal the Git objects at its respective measured source. Use a clean checkout of the authoritative measurement commit for source-strict evaluation. A final tagged descendant is valid only when its post-measurement diff is the report and explicitly reviewed non-executable evidence; changed executable, schema, workflow, case or policy bytes invalidate the measurements. Historical sources are checked under their original provenance, not relabeled as current measurements.

Only verified content is exposed through a no-replace atomic rename to `<cache-root>/goldens-v0.2/<archive-sha256>`. Existing immutable content is reused only after complete revalidation; conflicts are never overwritten. Races must not replace another install. On failure, remove only this operation's unpublished staging directory, leaving prior installs unchanged. The API must distinguish transport installation from complete release acceptance.

## 5. Performance and CPU evidence

Preserve [ADR-0012](../adr/0012-goldens-v0.2-calibration-and-performance-protocol.md)'s performance contract: only after approved budgets and successful authoritative validation, measure canonical04 and canonical05 workloads, greedy64, one discarded warm run and three measured repetitions, fixed options and cold state. Validate original synchronized timing events, request/token identities, raw memory samples, baseline/peak/delta, polling completeness, source/runtime identities and resource guards. Recompute throughput, first-token latency, inter-token samples and medians by the existing formulas; no new hardware SLA is introduced. Synthetic CPU examples cannot become runtime GPU performance evidence.

CPU evidence binds the measurement source, actual fixed command/argument vectors, features, environment/toolchain, exit codes and hashed raw stdout/stderr. The reviewed producer executes the fixed project-required gates rather than accepting a hand-written `passed=true` document. Require the default CPU core and workspace suites, supported internal-golden checks, formatting/lint/dependency checks and applicable Python gates. Include the [128-token, three-block public pressure regression and its controls](t45-mixed-phase-pressure-repair.md); a successful 768-token scenario cannot replace it. Freeze the concrete gate inventory and command schema in the reviewed measurement tooling, reject missing/duplicate/mismatched commands and failed output, and keep synthetic test provenance separate from real producer attestations.

Offline consumption validates the recorded CPU execution and its source binding; it does not claim cryptographic proof that arbitrary host logs are truthful. Publication relies on the trusted local producer and fresh source-bound review, with no caller option to replace mandatory gates with supplied summaries.

## 6. Report, candidate and separate publication stage

Render `docs/releases/goldens-v0.2.md` only from revalidated evidence. Include all required cases/checks and expected counts, individual verdicts/metrics, model/runtime/kernel/source identities, approved budgets with provenance, evidence hashes, performance samples/medians and limitations, plus the SHA-256 of both release assets. The report does not contain its own final commit or hash. Commit it as the evidence-only descendant of the measurement source.

Before publication, recheck the exact clean final candidate, matching committed report, validated assets, fresh full-range Standards/Spec review and clean-consumer result. Review evidence binds the fixed Base, candidate, tree and complete diff and preserves both review axes. Final candidate/review/permission bindings remain external to the archive to avoid candidate/report/archive self-reference.

An explicit independent user authorization is still mandatory for tag/release/asset writes. The publication transport is injectable for CPU tests; test transports never contact GitHub. The authorized real sequence requires an absent tag, non-forced tag creation targeting the frozen final candidate, one matching release, exactly two asset uploads and readback of tag/metadata/names/downloaded bytes/hashes/content. Existing, partial, failed or conflicting remote state is refused, not overwritten, deleted or automatically repaired. No remote Ticket branch or product `v0.2.0` release is created by #45; product publication remains #47.

Legacy authoritative/publication refusal stays in place. CPU round-trip, tampering, malformed archives, immutable-cache races, source/report mismatch, missing authority and fake-transport readback tests are required implementation gates, not new permission to publish. This checkpoint accepts no Ticket, approves no budget and authorizes no GPU execution or new holdout output access.
