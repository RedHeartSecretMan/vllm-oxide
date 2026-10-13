# Architecture decision records

[Documentation index](../README.md) · [Current architecture](../architecture.md) · [Release history](../releases/README.md)

These records explain decisions at their original checkpoints. They preserve
their own scope, status and evidence identities. A proposed record is not an
implemented capability; an accepted historical decision may be superseded by a
later contract. The current usage and release guides describe supported behavior.

## Read by topic

- Engine and API: ADR-0004, ADR-0006 through ADR-0009, ADR-0011 and ADR-0020.
- Current layered numerical rules and frozen budgets: ADR-0015 and ADR-0018.
- Evidence transport and source supervision: ADR-0016, ADR-0019, ADR-0021, ADR-0022 and ADR-0028.
- Precision and independent confirmation checkpoints: ADR-0023 through ADR-0029.
- Early foundations and historical comparator/calibration: ADR-0001 through ADR-0005, ADR-0010, ADR-0012 through ADR-0014 and ADR-0017.

Use the [validation overview](../validation/README.md) for the current protocol
and [v0.2.0 summary](../releases/goldens-v0.2.md) for actual published results.
The original ADR-0029 remains a proposal record; the release summary documents
the source arrangement actually used. Historical pending or failed checkpoints
are not the current release status.

## Complete index

| Record | Original status |
|---|---|
| [0001 — Parametric parallel layers](0001-parametric-parallel-layers.md) | Recorded checkpoint |
| [0002 — Weight loader seam](0002-weight-loader-seam.md) | Recorded checkpoint |
| [0003 — Model registry and RoPE](0003-model-registry-and-rope.md) | Recorded checkpoint |
| [0004 — Engine architecture and dependency DAG](0004-engine-architecture-and-dependency-dag.md) | Recorded checkpoint |
| [0005 — Golden generation correctness strategy](0005-golden-generation-correctness-strategy.md) | Recorded checkpoint |
| [0006 — v0.2.0 correctness-first release scope](0006-v0.2-correctness-first-release-scope.md) | Recorded checkpoint |
| [0007 — Resolved model identity construction seam](0007-resolved-model-identity-seam.md) | Recorded checkpoint |
| [0008 — Public request identity and output order](0008-request-identity-and-output-order.md) | Recorded checkpoint |
| [0009 — Validate SamplingParams before atomic request admission](0009-sampling-params-validation-and-diagnostics.md) | Recorded checkpoint |
| [0010 — Publish one manifest and one deterministic golden archive](0010-golden-release-asset-contract.md) | Recorded checkpoint |
| [0011 — Contract v0.2.0 around LLM::generate](0011-public-generation-contract.md) | Recorded checkpoint |
| [0012 — Bind goldens-v0.2 calibration and performance evidence](0012-goldens-v0.2-calibration-and-performance-protocol.md) | Recorded checkpoint |
| [0013 — Bound recalibration for the fixed BF16 kernel comparison](0013-bf16-cross-kernel-calibration-ceilings.md) | Recorded checkpoint |
| [0014 — Approve the measured goldens-v0.2 tolerance policy](0014-goldens-v0.2-empirical-tolerance-policy.md) | Recorded checkpoint |
| [0015 — Version operator, logit and token acceptance separately](0015-layered-accuracy-validation.md) | Recorded checkpoint |
| [0016 — Transport complete layered evidence in release schema 5](0016-layered-golden-release-assets.md) | Recorded checkpoint |
| [0017 — Account for reserved blocks in the pressure baseline](0017-pressure-baseline-usable-capacity.md) | Recorded checkpoint |
| [0018 — Freeze the calibrated layered accuracy budgets](0018-freeze-layered-accuracy-budgets.md) | Recorded checkpoint |
| [0019 — Recover bounded telemetry faults without relabeling measurements](0019-bounded-resource-supervision.md) | Recorded checkpoint |
| [0020 — Bounded attention with reference-aligned arithmetic](0020-bounded-reference-arithmetic.md) | Recorded checkpoint |
| [0021 — Rebuild release evidence without retained owners](0021-fresh-supervised-release-evidence.md) | Recorded checkpoint |
| [0022 — Bind performance to the measured implementation and actual monitoring interval](0022-bind-performance-measurement-and-monitoring.md) | Recorded checkpoint |
| [0023 — Propose an explicit FP32 GEMM reduction reference](0023-proposed-explicit-gemm-reduction.md) | superseded by ADR-0024 |
| [0024 — Approve explicit FP32 GEMM reduction and fresh release evidence](0024-approve-explicit-gemm-reduction.md) | accepted |
| [0025 — Preserve observed regressions and add an unobserved confirmation split](0025-proposed-independent-confirmation.md) | proposed |
| [0026 — Approve the independent confirmation inventory and calibration bindings](0026-approve-independent-confirmation.md) | accepted |
| [0027 — Retain observed confirmation and freeze a new independent cohort](0027-proposed-confirmation-renewal.md) | accepted |
| [0028 — Retain complete release evidence locally](0028-local-release-evidence.md) | accepted |
| [0029 — Preserve numerical X/Y and declare a separate monitored performance R/W](0029-proposed-release-source-roles.md) | proposed |
