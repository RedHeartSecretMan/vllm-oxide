---
status: proposed
---

# Preserve observed regressions and add an unobserved confirmation split

The `880e22c` acceptance run failed on `acceptance-waiting-long`, and that case informed the subsequent projection-layout repair. Repeating the original 487-owner inventory is required regression evidence, but cannot restore its unseen status. The layered contract forbids reusing observed failed holdouts, while ADR-0024 preserves every existing obligation. We therefore propose an additional, independently frozen `confirmation` split, requiring both the original inventory and the new split to pass.

This is a proposed acceptance-list checkpoint, not another change to the user-approved FP32 intermediate reduction reference or numerical budgets. No confirmation model output has been obtained. The current budget policy intentionally continues to target the previous registry, so both numerical and public-behavior confirmation collection remain sealed. This proposal does not accept a Ticket, permit opening confirmation, or authorize publication.

## Inputs and gates

Registry schema 2 retains the original development, calibration and acceptance case objects, token streams, setup calls, options, mechanism requirements, fault definitions and owner lists. The original registry remains at commit `479bf82`, raw SHA-256 `88f89084fe7b0c14a877c4c72f275314ad2a8e593d6fbc3ac63a194b91fb98f4`; a CPU regression verifies the complete canonical content of that snapshot remains present.

The added split contains 13 execution groups, 19 numerical cases and 75 prediction rows per engine/variant, plus 11 behavior checks. It adds 130 unique guarded owners, including the required unforced controls and independent replays. The full inventory becomes 617 owners; calibration remains 132 owners including the shared L0 suite. Existing complete-inventory assembly, numerical/behavior evaluation, marker closure and clean-consumer checks require the new owners and cases as well as the old ones. An extra narrative report cannot satisfy this gate.

New literals concern a railway workshop and use the same pinned tokenizer and CPU token-cycle rules. Page boundaries, context limits, chunk remainders, mixed batches, waiting admission, prefix reuse, pressure, repeated calls, invalid-input recovery and EOS behavior retain their existing geometry and checks. Model outputs are not consulted when preparing these inputs. The validator rejects potential prediction-history reuse across the confirmation boundary, including setup calls and successful unforced calls: compatible prompt prefixes with overlapping reachable prediction lengths are rejected even when an old unforced continuation is unknown.

## Consequences

The registry hash changes, so the old calibration cannot be relabeled or used under the new registry. Freeze the new measurement source, complete its CPU gates and fresh calibration/fault evidence, then review and approve the new list and evidence bindings before opening confirmation. All four model and seven operator budget values, model identities, resource protections and the reference arithmetic profile remain unchanged. A confirmation failure remains a failure; do not replace prompts after observing outputs or weaken a limit.

The alternative of renaming the observed acceptance split or only rerunning its 487 owners would violate independence. Adding the new cases as another required split reuses the existing fail-closed evaluator and evidence transport rather than introducing a separate optional acceptance report.
