//! Stub — FFI to `candle-flash-attn`.
//!
//! Two entry points: `flash_attn_varlen` (prefill, unpaged) and
//! `flash_attn_varlen_paged_windowed` (decode, paged; PR #3655, prototype
//! validated in T10). Model code calls these directly — no trait. Lands in T4.

#![allow(dead_code)]
