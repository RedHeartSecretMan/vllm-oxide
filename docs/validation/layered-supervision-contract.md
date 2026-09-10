# Bounded resource supervision and truthful measurement provenance

This contract is adopted by [ADR-0019](../adr/0019-bounded-resource-supervision.md). The [supervision policy](layered-supervision-policy.json) supplies the exact configuration and compatibility-ledger identity. It does not modify the [accuracy budgets](layered-accuracy-budgets.json), case registry, model, sampling rules or numerical algorithms. All numerical cases retain the same four approved conditions.

## 1. Separate real sources, not relabeled evidence

| Role | Identity and execution requirement |
| --- | --- |
| Measurement | The policy's fixed `8f21331` commit/tree; actual workers run from a clean immutable checkout of it, including their working directory, `--repo-root` and `PYTHONPATH`. The Rust binary must match the policy's original hash/build-source identity. |
| Supervision | A separately frozen, clean, reviewed commit/tree containing the new resource supervisor and evidence-producing adapters. This is the actual running implementation, not a caller's arbitrary source declaration. |
| Evaluation | A separately recorded clean, reviewed commit/tree running the trusted schema/identity adapter and numerical evaluator; for this policy it must equal the supervision source. |
| Calibration approval predecessor | The original `6b7ae7d` observation/manifest/marker/fault closure already bound by the numerical policy. Its source, bytes and verdict remain unchanged. |

The supervision/evaluation source must descend from the measurement source and preserve every path in `unchanged_measurement_paths` by Git object identity (directories include their complete tracked subtrees; deletion or a changed file mode also fails). Model/config/tokenizer/weight, binary, wheel/kernel/runtime and numerical-policy checks remain mandatory. Only the described supervision and evidence-identity routing may change, with associated tests/documentation and a complete candidate review. The path check is not permission for arbitrary changes elsewhere or for altering the evaluator's mathematical calls. The existing numerical and operator comparison implementations remain unchanged.

Before and after each worker execution, verify the immutable measurement checkout's exact commit/tree and clean tracked bytes. Launch a fresh worker process using the recorded measurement invocation, not a new worker with an old `source` field. Record and verify its actual working directory, `--repo-root`, `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, executable/binary and original worker metadata. Do not inject new supervisor modules into the measurement worker or overwrite its source. The collector may construct a receipt from verified original worker metadata; it must reject, not relabel, a mismatch.

The original calibration provenance continues to validate `6b7ae7d -> 8f21331`, including original hashes, marker, complete observation reproduction and fault detection. Do not broaden its executable-path allowance to treat new supervisor code as old measured code. Changing measurement implementation, inputs or runtime requires the existing new-measurement/approval process; this exception cannot validate a different model implementation with the old binary.

## 2. Resource policy

Fast checks read host RAM and track owned processes independently at a target 100 ms interval. A blocked NVML query must not block these checks. RAM below 16 GiB remains fatal before, during or after an owner. Missing/malformed fast-resource data, a failed worker, observed foreign CUDA compute, resource violations and owner/probe cleanup failure remain fatal; no successful telemetry recovery erases them. Existing disk/capacity checks remain in force.

Slow telemetry obtains one complete GPU-memory plus compute-process snapshot at a time, normally no more frequently than once per second. Each real query retains a five-second deadline. Only `subprocess.TimeoutExpired` permits recovery: discard the entire incomplete snapshot and retry one complete fresh snapshot immediately. At most one timeout may be recovered across the whole owner, including preflight, active monitoring and final checks. A second timeout or any non-timeout probe error is fatal.

The recovery window is at most 15 seconds starting from the first attempt of the affected complete snapshot, not 15 extra seconds after its timeout. The supervisor independently enforces that window and terminates its owned telemetry processes on expiry; a hung probe must not make the supervisor unbounded. Ordinary snapshots remain bounded by their two five-second query deadlines. Before/after snapshots and the immediate timeout retry are exceptions to the one-second scheduling interval, not to single-flight or deadline rules.

A successful owner requires a fresh complete snapshot before launch and another started after owner cleanup. Worker exit zero does not waive the final snapshot. Pending active telemetry must be resolved or failed/cleaned without overlapping a final attempt. The final check does not reset the owner's recovery allowance. If fresh resource/empty-compute evidence cannot be obtained, the stage remains INVALID, including when inference itself exited zero.

Never fabricate zero usage, reuse an old sample as fresh, accept a partial snapshot, suppress an error record, retry a failed model computation as a telemetry retry, or run independent competing probe loops. Fast polling is a scheduling target, not an OS hard-real-time guarantee: retain actual sample times and gaps, and do not claim that every interval was exactly 100 ms.

## 3. Guard evidence schema 2

Keep the existing command, child/PGID, exit, cleanup, before/after resource, minima/peaks, duration and failure fields, with their existing meanings. A new guard sets `schema_version=2` and additionally binds:

```text
measurement_source: {commit, tree}
supervision_source: {commit, tree}
supervision_policy_sha256: <original policy JSON SHA-256>
measurement_invocation: <actual worker cwd/repo-root/PYTHONPATH/executable and binary bindings>
telemetry_interval_ms: 1000
child_started_seconds: <finite relative monotonic time, or null if never started>
child_exit_observed_seconds: <finite relative monotonic time, or null if not observed>
owner_cleanup_completed_seconds: <finite relative monotonic time, or null on incomplete cleanup>
fast_ram_samples: [{elapsed_seconds, available_ram_bytes}]
maximum_fast_poll_gap_seconds: <actual maximum gap>
remaining_telemetry_pids: []
telemetry_cleanup_failure: null
telemetry_events: [{attempt, phase, started_seconds, ended_seconds,
                    outcome, snapshot_index, error, queries}]
```

Times use one monotonic origin established before the initial probe. `phase` is `before`, `active` or `after`; `outcome` is `fresh`, `timeout` or `fatal`; attempts are consecutive zero-based integers. Each `queries` entry records the actual query kind (`gpu_memory` or `compute_processes`), PID, configured timeout, start/end times, outcome (`ok`, `timeout`, `fatal`) and error. Missing operations must not be invented. Successful paths require all lifecycle anchors and consistent order; failed paths retain actual partial evidence and cannot pass.

`resource_samples` contains only complete successful fresh snapshots, adding `started_seconds` and `completed_seconds` to the existing resource fields and elapsed time. A fresh event's `snapshot_index` references exactly its complete sample; timeout/fatal events use null and preserve their errors. `before` and `after` must match the corresponding fresh samples. `ram_sample_count` counts independent fast samples; the recorded RAM minimum includes all of them and all complete snapshots. Record actual maximum sampling gap; no new numerical-accuracy threshold is inferred from it.

Consumers must reconstruct sample/event/query order, single-flight, recovery count/window, timeout configuration, lifecycle order, extrema, fast-sample count and required initial/final evidence. They also verify actual source/invocation/policy identity, worker exit zero, valid resource data, no observed foreign compute and empty owned/telemetry cleanup. Do not trust `failure=null` or a supplied PASS flag alone. Scheduling/cleanup latency must be recorded truthfully; a late result is not a pre-deadline fresh result. A deadline or cleanup failure remains invalid rather than being hidden by rounding timestamps.

## 4. Exact old-guard compatibility

The policy binds the original bytes of one `layered-retained-owner-ledger-v1`, schema 1 ledger by SHA-256. It declares the exact measurement source, registry hash, numerical-policy hash and 445 unique owner keys at indices 0 through 444 of the unchanged frozen 487-owner inventory. Each entry binds its capture, receipt, guard and full dependency file hashes, original execution identity and retention provenance. The ledger is evidence in the final artifact closure, not a mutable run receipt or an arbitrary caller allowlist.

Only an exact entry match permits a retained schema 1 guard in the new authoritative inventory. Revalidate all bound bytes, owner identity/history, original measurement/binary/model/runtime and every applicable old strict guard condition: zero worker status, no failure, valid samples and RAM floor, empty preflight/post-cleanup compute, successful cleanup and no owned remainder. A match by index, source or filename alone is insufficient. All required setup and worker dependencies remain included. Copies preserve original relative paths and bytes; copying is not a new GPU execution.

The remaining indices 445 through 486 require new normal executions with schema 2 guards. Neither prior failed attempt nor any diagnostic capture may be promoted into a successful entry, even if its logits exist or its worker exited zero. Keep those original failures and diagnostic records separate and unchanged. The original 132-owner calibration approval predecessor follows its existing schema 1 provenance path; it is not part of this 445-entry compatibility ledger.

## 5. Execution manifest, result and marker identities

New supervised authoritative execution manifests use `schema_version=2`, preserve all existing fields, and add these required fields:

```text
evaluator_source: {commit, tree}
supervision_source: {commit, tree}
supervision_policy: {path, sha256}
retained_owner_ledger: {path, sha256}
```

`source` remains the real measurement source, as do capture/receipt source and worker runtime `generator_commit`. Both new source fields must equal the actual clean, frozen evaluator/supervisor implementation, checked against Git and the trusted invocation. The policy must be a selected Definition input in that source, with exact bytes/hash. Ledger identity, owner set, coverage and hashes must match the policy, not merely the manifest's claim. References retain existing confined regular-file path/hash rules.

Receipt schema 1 and its original shape remain valid: the receipt's guard reference binds the new schema 2 guard for new owners, or the exact authorized old guard for retained owners. Reject missing supervision identities, mismatched sources/policy, unknown schemas, arbitrary schema 1 guards and incomplete/duplicated owners. The full inventory remains 487; all input, numerical, behavioral, replay and approval-predecessor checks still apply.

The new authoritative result records `source` as measurement plus the evaluator/supervision identities and policy/ledger references. Its new completion marker uses schema 2, binds `source` to the actual evaluator source, explicitly names `measurement_source` and `supervision_source`, and hashes its complete manifest/result/policy/ledger and predecessor dependencies under the existing atomic marker rules. This marker does not pretend that the new evaluator executed as `8f21331`. Only genuine full PASS creates a success marker; FAIL/INVALID do not. Original observation and CPU markers keep their original versions and identities.

## 6. Transport, release source and gates

The outer Golden release manifest stays at schema 5 with the same two-asset contract. Its measurement identity remains measurement identity; supervised execution manifests, guards, policy, retained ledger and role-specific Definition/CPU evidence become mandatory transitive dependencies. Do not silently deduplicate owners, rewrite raw references or import executable payloads from the archive. The trusted installed evaluator verifies the new supervision/measurement closure; archive-provided code is never executed.

Clean consumption uses the frozen reviewed evaluator source and verifies the separate immutable measurement source and original predecessor identities. Old `8f21331` CPU gates attest measurement; the new complete CPU gates/review attest supervisor/evaluator. Both are required and neither may be relabeled. The final evidence-only report/release candidate must descend from the frozen evaluator source, with only the already permitted report/non-executable evidence changes afterward. The narrow pre-evaluation supervision/identity-adapter exception described here replaces the old single-source assumption, not the post-evaluation immutability rule.

Before remaining GPU work, require deterministic CPU RED-to-GREEN coverage through the real supervision interface: one timeout then recovery with a zero-exit child; blocked telemetry with prompt RAM-floor termination; persistent/second timeout; non-timeout or malformed telemetry; worker failure despite recovered telemetry; foreign compute; missing final fresh sample; deadline and owner/telemetry cleanup failures. Identity/transport tests reject altered ledgers, unlisted old guards, failed/diagnostic substitution, false measurement invocation, changed protected measurement code, wrong evaluator/policy/marker identity and incomplete closure. Full project CPU gates and fresh complete candidate review precede GPU resumption. These tests establish supervisor behavior, not a causal fix for NVML or the driver.

After guarded preflight, continue only the remaining 42 normal owners, preserving the 445 exact retained owners and all failures. Revalidate full closure and run authoritative comparison after collection, with RAM/process protection during offline work too. Actual numerical failures remain failures; no new thresholds, case selection or backend changes are authorized. Performance and publication remain separate later stages.
