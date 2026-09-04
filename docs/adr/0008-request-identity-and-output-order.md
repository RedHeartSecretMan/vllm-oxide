# Public request identity and output order

The v0.1 `RequestOutput.seq_id` field exposed scheduler identity and was also
treated as a call-local batch index. After one `LLM::generate` call, monotonic
sequence identifiers no longer started at zero, causing later outputs to be
dropped and replaced by placeholders. Issue #33 requires stable public identity
and input-order output assembly without coupling either to scheduler internals.

**Decision**: each accepted prompt receives a stable `request_id` independent
of its internal sequence identity. `RequestOutput` replaces `seq_id` with
`request_id`; no compatibility `seq_id` field remains because it would keep
the internal identity leak in the supported API.

`LLM::generate` records a call-local `request_id → input position` mapping,
rejects missing, duplicate, or unknown completed requests, and returns exactly
one output per accepted prompt in input order. Batch position is the output
vector index, so no redundant public `batch_position` field is added.

This contract preserves internal scheduling and completion reordering while
making output association deterministic. Ticket #44 retains `RequestOutput`
in the final public surface and hides the remaining scheduler implementation
types.

**Status**: accepted
