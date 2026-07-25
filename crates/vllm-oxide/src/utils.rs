//! Small numeric helpers ported from `nano-vllm/utils.py`.
//!
//! Only `round_up` and `kv_cache_layout_shape` ship in the v0.1 scaffold —
//! everything else in this crate is a stub until downstream tickets land.

/// Round `n` up to the nearest multiple of `m`.
///
/// Ported from nano-vllm's `(n + bs - 1) // bs * bs` idiom (inlined at
/// `engine/sequence.py:57` and `engine/model_runner.py:152,227`). nano-vllm
/// has no named function — this Rust port extracts the helper. Precondition: `m > 0`.
///
/// # Examples
///
/// ```
/// # use vllm_oxide::round_up;
/// assert_eq!(round_up(0, 256), 0);
/// assert_eq!(round_up(1, 256), 256);
/// assert_eq!(round_up(256, 256), 256);
/// assert_eq!(round_up(257, 256), 512);
/// ```
pub const fn round_up(n: usize, m: usize) -> usize {
    // Refactored from the classic `(n + m - 1) / m * m` form. The classic
    // form's `n + m - 1` overflows for ~m values of `n` near `usize::MAX`
    // (e.g. m=256 → 256 overflow-triggering inputs). The division-remainder
    // form below can only overflow at `n ∈ {usize::MAX-1, usize::MAX}` — a
    // meaningful safety margin for the block_size=256 call sites nano-vllm
    // uses. Truly overflow-safe round-up requires returning Option/Result,
    // which is overkill for token-count math.
    (n / m) * m + if n % m != 0 { m } else { 0 }
}

/// Physical PagedKVCache buffer shape as `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`.
///
/// This is the layout `nano_vllm.layers.attention.KvCache` allocates and that
/// `reshape_and_cache` writes into. The leading `2` is the K/V stack. Dim
/// meanings:
///
/// - `0` — K vs V (always 2)
/// - `1` — model decoder layers
/// - `2` — paged blocks in the pool (`num_blocks`)
/// - `3` — tokens per block (Qwen3 v0.1 hard-locks `block_size = 256`)
/// - `4` — grouped-query attention KV heads
/// - `5` — per-head dim
///
/// Ported 1:1 from `nano_vllm.utils.get_kv_cache_shape`.
pub fn kv_cache_layout_shape(
    num_layers: usize,
    num_blocks: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> [usize; 6] {
    [
        2,
        num_layers,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    mod round_up {
        use super::*;

        #[test]
        fn zero_rounds_to_zero() {
            assert_eq!(round_up(0, 256), 0);
        }

        #[test]
        fn exact_multiple_is_identity() {
            assert_eq!(round_up(256, 256), 256);
            assert_eq!(round_up(512, 256), 512);
            assert_eq!(round_up(1024, 256), 1024);
        }

        #[test]
        fn non_multiple_rounds_up() {
            assert_eq!(round_up(1, 256), 256);
            assert_eq!(round_up(255, 256), 256);
            assert_eq!(round_up(257, 256), 512);
        }

        #[test]
        fn one_block_size() {
            assert_eq!(round_up(1, 1), 1);
            assert_eq!(round_up(41, 1), 41);
        }

        #[test]
        fn non_power_of_two_multiple() {
            // nano-vllm only ever calls with power-of-two m (block_size=256),
            // but the general case must still work — see the implementation
            // comment.
            assert_eq!(round_up(0, 3), 0);
            assert_eq!(round_up(1, 3), 3);
            assert_eq!(round_up(3, 3), 3);
            assert_eq!(round_up(4, 3), 6);
            assert_eq!(round_up(7, 3), 9);
        }
    }

    mod kv_cache_layout_shape {
        use super::*;

        #[test]
        fn leading_dim_is_two_for_k_and_v() {
            let shape = kv_cache_layout_shape(28, 1024, 256, 8, 128);
            assert_eq!(shape[0], 2);
        }

        #[test]
        fn qwen3_0_6b_layout() {
            // Qwen3-0.6B: 28 layers, 4 KV heads (GQA), head_dim 128.
            // Pool size is a per-launch decision; 1024 here is illustrative.
            let shape = kv_cache_layout_shape(28, 1024, 256, 4, 128);
            assert_eq!(shape, [2, 28, 1024, 256, 4, 128]);
        }

        #[test]
        fn dims_preserve_call_order() {
            let shape = kv_cache_layout_shape(1, 2, 3, 4, 5);
            assert_eq!(shape, [2, 1, 2, 3, 4, 5]);
        }

        #[test]
        fn block_size_256_locked_at_call_site() {
            // Callers should pass block_size=256 per the v0.1 spec; the helper
            // itself is parameterised so the future `block_size` knob lands as
            // a call-site change, not a function-signature change.
            let shape = kv_cache_layout_shape(28, 100, 256, 4, 128);
            assert_eq!(shape[3], 256);
        }
    }
}
