# Engine architecture and dependency DAG

The internal module structure follows a strict acyclic dependency DAG and a
public API surface rule.

## Dependency DAG

```
llm  (composition root — owns everything)
 ├── engine        (Scheduler, BlockPool, KVCacheManager, EngineCore, Sequence)
 ├── attention     (flash-attn kernels, AttnMetadata, PagedKVCache)
 ├── models        (CausalLM impls — depends on layers + attention + loader)
 ├── sampler       (SamplingParams, Sampler)
 ├── config        (Source, dtype resolution, HF_HUB_OFFLINE shim)
 ├── loader        (weight loading — depends on config)
 ├── layers        (Linear, RMSNorm, RoPE, activations, parallel — leaf)
 └── causal_lm     (CausalLM trait — neutral contract; depends on candle_core)
```

**Rules**:
- `layers/`, `attention/`, `loader/`, `sampler/`, `causal_lm.rs` are leaves: no
  internal deps.
- `models/` depends on `layers/` + `attention/` + `loader/` and implements
  the `CausalLM` trait from the neutral `causal_lm.rs` contract.
- `engine/` does NOT depend on `models/` — it receives `Box<dyn CausalLM>` from
  the composition root and depends on the neutral `causal_lm.rs` contract instead.
- `llm.rs` is the ONLY composition root that wires everything.
- The potential `engine ↔ attention` cycle is broken: engine holds
  `Arc<Mutex<PagedKVCache>>` and builds `AttnMetadata` from its own scheduler
  state; `attention/` never imports `engine/`.

## R4: Public API surface

`lib.rs` is the ONLY module that issues top-level `pub use`. Internal modules
default to `pub(crate)` or stricter. Downstream callers never reach below the
re-exports curated in `lib.rs`.

## EngineCore: collapsed ModelRunner

**Decision**: v0.1 has no `model_runner` sub-module. `EngineCore::step()`
performs the full loop in one method: scheduler → tensor prep →
`model.forward()` → sampler → KV update. This collapses V1/nano-vllm's
`ModelRunner` separation.

**R5 split trigger**: Extract a separate `ModelRunner` when `step()` exceeds
~300 LOC or CUDA graph capture lands (v0.2).

## Micro-decisions

- **M1**: `SamplingParams` lives in `sampler.rs` (not `engine/`) so it can be
  re-exported at the crate root without pulling in the engine.
- **M2**: `Sequence` (carrying its own `request_id`) is the V1 data-model leaf —
  the scheduler's working set. The former 1:1 `SequenceGroup` wrapper was
  absorbed because it added delegation without depth; n>1 sampling (v0.2) will
  reintroduce grouping deliberately if/when the capability exists.
- **M3**: `KvCacheManager` is a deliberate information-hiding adapter
  (structural seam, not computational module). Its 6 delegations + 1
  bridge method are the intended final shape. Thinness is the design,
  not debt.

**Consequences**:
- New modules must fit the DAG; cyclic imports are a design error, not a
  refactor opportunity.
- TP wiring in v0.2 adds to `layers/parallel.rs` trait impls and
  `attention/` NCCL hooks — no engine-level restructuring expected.
- `engine/` receives its model as a trait object, keeping it decoupled from
  architecture-specific code.

**Status**: accepted
