//! Numeric helpers used by the internal paged-cache implementation.

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
