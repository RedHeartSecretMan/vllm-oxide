# Validate SamplingParams before atomic request admission

Ticket #36 completes a public input contract whose accepted v0.1 sampling pipeline did not define validation or diagnostic boundaries. `LLM::generate` validates the complete `SamplingParams` batch before tokenizing or admitting any request; a sampling-parameter validation failure leaves scheduler and cache state unchanged. Atomicity after prompt tokenization begins is not added to the #36 contract.

The accepted validation contract is exact:

- `temperature` is not NaN and is greater than or equal to zero; zero selects greedy sampling. Positive infinity remains valid and produces the existing uniform pre-filter distribution.
- `top_k` is `None` or `Some(k)` with `k >= 1`. It has no vocabulary upper bound; `k` at least as large as the vocabulary is a supported no-op.
- `top_p` is `None` or finite in `(0, 1]`.
- `presence_penalty` and `frequency_penalty` are finite in `[-2, 2]`.
- `repetition_penalty` is finite and greater than or equal to zero. The project keeps Issue #17's accepted `0 = no-op` convention; positive values retain the existing sampler meaning.
- `max_tokens` is at least one and counts completion tokens only. Resolved model metadata supplies EOS; `ignore_eos` bypasses EOS stopping but never the `max_tokens` limit.

Every combination of individually valid fields in the current public surface is supported. Penalties apply before selection; `temperature == 0` or `top_k == Some(1)` selects the accepted greedy path, so later top-k and top-p filtering is inert rather than rejected. The penalty and top-p bounds are informed by vLLM validation, but this is deliberately not a representation-compatibility promise: Issue #17 keeps optional `top_k`, a zero-valued repetition no-op, unbounded non-negative temperature, and the sampler's explicit positive-infinity corner case. These differences preserve the accepted project pipeline instead of silently importing newer vLLM limits.

One immutable complete `SamplingParams` value follows each accepted request through planning, sampling, completion, and failure. Validation errors identify the input batch position, field, rejected value, and reason; internal completion tracing and runtime error context identify the stable `request_id` and originating parameters. `RequestOutput` remains the four-field contract accepted by ADR-0008, and diagnostics do not expose `Sequence` or add a public diagnostics type.

**Status**: accepted
