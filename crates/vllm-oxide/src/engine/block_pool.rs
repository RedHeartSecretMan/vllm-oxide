//! Stub — physical `Block`s + free-list + prefix-cache hashtable (V1 parity).
//!
//! `can_allocate` / `allocate` / `can_append` / `may_append` / `deallocate`
//! / `hash_blocks` mirror nano-vllm's xxhash-chained prefix-cache algorithm.
//! CoW semantics for shared prefix blocks. This is where prefix caching
//! lives. Lands in T2.

#![allow(dead_code)]
