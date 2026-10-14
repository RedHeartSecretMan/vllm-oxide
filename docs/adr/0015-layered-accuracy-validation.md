# Version operator, logit and token acceptance separately

[ADR index](README.md) · [RC validation](../validation/README.md)

> Historical RC checkpoint. Original status, evidence identities and budgets are preserved; later final-release results do not apply to this source.

The approved accuracy design assigns L0 to operators, L1 to model logits and distributions, and L2 to final token choices and engine behavior. The [layered accuracy contract](../validation/layered-accuracy-contract.md) specifies these checks, using a fixed Transformers BF16 SDPA MATH reference and an additional paired comparison with vLLM. No extra backend-isolation experiment is required.

This is a design-and-implementation approval, not empirical approval of any operator or model budget. All unset budgets remain pending, and new authoritative validation must fail closed until the corpus, budget provenance and independent evidence have been frozen. Historical failures, observations and reports retain their original identities and verdicts.

**Supersedes for future release validation:** the legacy L1/token and L2/logit numbering, candidate-only near-tie rule, regression token-only evidence shape, fixed 56-fixture/split counts and empirical thresholds in ADR-0005/0006/0012/0013/0014, only to the extent explicitly replaced by the linked contract. Existing model scope, public API, causal-history correctness, integrity, resource protection and publication authority constraints remain unchanged. MATH is a finite-precision reference implementation, not exact real arithmetic.

**Ticket ownership:** #45 owns the new release comparator, evidence and integration work; #46 and #47 consume the resulting versioned gates. Tickets #29–#44 retain their original delivery obligations and their outputs remain inputs, rather than being relabeled as proof of the new release gates. The normalized historical GitHub snapshot and native dependencies remain unchanged; its earlier layer labels are interpreted in their original protocol. This revision changes release acceptance meaning explicitly without inventing new Tickets, dependency edges, or completion evidence.

**Status:** accepted design; exact case registry and empirical budgets require subsequent reviewed Definition Checkpoints. No Ticket is accepted by this decision.
