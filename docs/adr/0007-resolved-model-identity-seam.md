# Resolved model identity construction seam

Model construction previously followed the same `Source` independently for
configuration, tokenizer, and weights, so a moving Hub revision could produce
one `LLM` from inconsistent artifacts. Issue #30 requires one immutable
identity, requested dtype, and special-token contract to feed every construction
consumer without overlapping the cache-capacity and warmup work owned by
Issue #35.

**Decision**: the Weight loader owns source resolution and returns one immutable
`ResolvedModel`. A Hub branch or tag is resolved to an exact commit before any
artifact fetch; a local source is bound to one canonical root. Configuration,
tokenizer, deduplicated weight paths, dtype, and special-token metadata are
thereafter read only through that value.

The registry remains a pure architecture lookup with one `ModelEntry` and one
`inventory::submit!` per architecture. Its factory consumes
`&ResolvedModel`; model implementations neither resolve `Source` nor choose a
different dtype. `LLM::new` is the composition root that resolves once, selects
the factory, constructs the model, and injects resolved EOS metadata into the
scheduler.

`ModelEntry.factory` is already semver-visible before v0.2.0 public API
contraction, which makes the identity types reachable for the interim factory
signature. This is transitional visibility, not a supported end-user API:
Ticket #44 removes registry, loader, model, scheduler, cache, and attention
construction types from the supported public surface after their behavior
stabilizes.

**Status**: accepted
