# Architecture decision records — RC source

[Documentation](../README.md) · [Architecture](../architecture.md) · [RC results](../releases/v0.2.0-rc.1.en.md)

These records explain the design at RC1 and retain their original status, source, and evidence. Later v0.2.0 precision and monitoring repairs have their own records. Accepted numerical budgets are unchanged by documentation maintenance.

## Read by topic

| Topic | Records |
|---|---|
| Engine, public API, request identity and validation | ADR-0004, ADR-0006 through ADR-0009, ADR-0011 |
| Initial Candle/model/loader stack | ADR-0001 through ADR-0003 |
| Earlier comparator, calibration and asset contracts | ADR-0005, ADR-0010, ADR-0012 through ADR-0014 |
| Layered operator/numerical/behavior acceptance | ADR-0015 and ADR-0018 |
| Layered archive, pressure and bounded supervision | ADR-0016, ADR-0017, ADR-0019 |

A design contract or CPU-tested implementation is not proof that its real-model gate passed. The original RC numerical result remains 18 failures out of 66 cases; final-release results use later sources. Original publication references are historical machinery and do not override the current local-only raw-evidence policy.

## Complete index

| Record | Original status |
|---|---|
| [Parametric parallel layers](0001-parametric-parallel-layers.md) | accepted |
| [Weight loader seam](0002-weight-loader-seam.md) | accepted |
| [Model registry and RoPE](0003-model-registry-and-rope.md) | accepted |
| [Engine architecture and dependency DAG](0004-engine-architecture-and-dependency-dag.md) | accepted |
| [Golden generation correctness strategy](0005-golden-generation-correctness-strategy.md) | accepted |
| [v0.2.0 correctness-first release scope](0006-v0.2-correctness-first-release-scope.md) | accepted |
| [Resolved model identity construction seam](0007-resolved-model-identity-seam.md) | accepted |
| [Public request identity and output order](0008-request-identity-and-output-order.md) | accepted |
| [Validate SamplingParams before atomic request admission](0009-sampling-params-validation-and-diagnostics.md) | accepted |
| [Publish one manifest and one deterministic golden archive](0010-golden-release-asset-contract.md) | accepted |
| [Contract v0.2.0 around LLM::generate](0011-public-generation-contract.md) | accepted |
| [Bind goldens-v0.2 calibration and performance evidence](0012-goldens-v0.2-calibration-and-performance-protocol.md) | accepted protocol |
| [Bound recalibration for the fixed BF16 kernel comparison](0013-bf16-cross-kernel-calibration-ceilings.md) | accepted calibration-ceiling revision |
| [Approve the measured goldens-v0.2 tolerance policy](0014-goldens-v0.2-empirical-tolerance-policy.md) | approved empirical numerical policy |
| [Version operator, logit and token acceptance separately](0015-layered-accuracy-validation.md) | accepted design |
| [Transport complete layered evidence in release schema 5](0016-layered-golden-release-assets.md) | accepted design |
| [Account for reserved blocks in the pressure baseline](0017-pressure-baseline-usable-capacity.md) | accepted case-configuration decision |
| [Freeze the calibrated layered accuracy budgets](0018-freeze-layered-accuracy-budgets.md) | accepted numerical-budget decision |
| [Recover bounded telemetry faults without relabeling measurements](0019-bounded-resource-supervision.md) | accepted supervision optimization and narrowly scoped evidence-reuse decision under the user's request |
