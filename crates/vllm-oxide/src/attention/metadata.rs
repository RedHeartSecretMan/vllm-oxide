//! Per-step attention metadata — the scheduler→attention boundary.
//!
//! Carries `cu_seqlens_q` / `cu_seqlens_k` in the **flash-attn cumulative
//! convention** (prefix sums), NOT `context_lens` (vLLM's per-sequence lengths).
//! The conversion lives in `build_decode_metadata` (T10 finding).

#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub struct AttnMetadata {
    pub is_prefill: bool,
    pub cu_seqlens_q: Vec<u32>,
    pub cu_seqlens_k: Vec<u32>,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub slot_mapping: Vec<i64>,
    pub block_table: Vec<Vec<i32>>,
}

pub fn build_decode_metadata(
    context_lens: &[u32],
    block_table: &[Vec<i32>],
    slot_mapping: &[i64],
) -> AttnMetadata {
    let batch = context_lens.len();
    let mut cu_seqlens_q = Vec::with_capacity(batch + 1);
    let mut cu_seqlens_k = Vec::with_capacity(batch + 1);
    cu_seqlens_q.push(0);
    cu_seqlens_k.push(0);
    let mut max_seqlen_k: usize = 0;
    for (i, &ctx) in context_lens.iter().enumerate() {
        cu_seqlens_q.push(cu_seqlens_q[i] + 1);
        cu_seqlens_k.push(cu_seqlens_k[i] + ctx);
        if ctx as usize > max_seqlen_k {
            max_seqlen_k = ctx as usize;
        }
    }
    AttnMetadata {
        is_prefill: false,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q: 1,
        max_seqlen_k,
        slot_mapping: slot_mapping.to_vec(),
        block_table: block_table.to_vec(),
    }
}

pub fn build_prefill_metadata(
    scheduled_tokens: &[u32],
    kv_lengths: &[u32],
    slot_mapping: &[i64],
) -> AttnMetadata {
    let batch = scheduled_tokens.len();
    let mut cu_seqlens_q = Vec::with_capacity(batch + 1);
    let mut cu_seqlens_k = Vec::with_capacity(batch + 1);
    cu_seqlens_q.push(0);
    cu_seqlens_k.push(0);
    let mut max_seqlen_q: usize = 0;
    let mut max_seqlen_k: usize = 0;
    for (i, (&sq, &sk)) in scheduled_tokens.iter().zip(kv_lengths).enumerate() {
        cu_seqlens_q.push(cu_seqlens_q[i] + sq);
        cu_seqlens_k.push(cu_seqlens_k[i] + sk);
        if sq as usize > max_seqlen_q {
            max_seqlen_q = sq as usize;
        }
        if sk as usize > max_seqlen_k {
            max_seqlen_k = sk as usize;
        }
    }
    AttnMetadata {
        is_prefill: true,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping: slot_mapping.to_vec(),
        block_table: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod build_decode_metadata {
        use super::*;

        #[test]
        fn single_sequence_cu_seqlens() {
            let meta = build_decode_metadata(&[10u32], &[vec![0, 1]], &[255i64]);
            assert_eq!(meta.cu_seqlens_q, vec![0, 1]);
            assert_eq!(meta.cu_seqlens_k, vec![0, 10]);
        }

        #[test]
        fn batch_cu_seqlens_are_cumulative() {
            let meta = build_decode_metadata(
                &[5u32, 20, 3],
                &[vec![0], vec![1, 2], vec![3]],
                &[100i64, 200, 300],
            );
            assert_eq!(meta.cu_seqlens_q, vec![0, 1, 2, 3]);
            assert_eq!(meta.cu_seqlens_k, vec![0, 5, 25, 28]);
        }

        #[test]
        fn max_seqlen_q_is_one() {
            let meta = build_decode_metadata(
                &[100u32, 200, 300],
                &[vec![0], vec![1], vec![2]],
                &[0i64; 3],
            );
            assert_eq!(meta.max_seqlen_q, 1);
        }

        #[test]
        fn max_seqlen_k_is_max_context() {
            let meta =
                build_decode_metadata(&[5u32, 47, 3], &[vec![0], vec![1], vec![2]], &[0i64; 3]);
            assert_eq!(meta.max_seqlen_k, 47);
        }

        #[test]
        fn is_prefill_false() {
            let meta = build_decode_metadata(&[1u32], &[vec![0]], &[0i64]);
            assert!(!meta.is_prefill);
        }

        #[test]
        fn slot_mapping_passes_through() {
            let slots = vec![10i64, 20, 30, -1];
            let meta = build_decode_metadata(
                &[1u32, 1, 1, 1],
                &[vec![0], vec![1], vec![2], vec![3]],
                &slots,
            );
            assert_eq!(meta.slot_mapping, slots);
        }

        #[test]
        fn block_table_passes_through() {
            let blocks = vec![vec![0, 1], vec![2], vec![3, 4, 5]];
            let meta = build_decode_metadata(&[1u32, 1, 1], &blocks, &[0i64; 3]);
            assert_eq!(meta.block_table, blocks);
        }

        #[test]
        fn empty_batch() {
            let meta = build_decode_metadata(&[], &[], &[]);
            assert_eq!(meta.cu_seqlens_q, vec![0]);
            assert_eq!(meta.cu_seqlens_k, vec![0]);
            assert_eq!(meta.max_seqlen_k, 0);
        }

        #[test]
        fn cu_seqlens_lens_are_batch_plus_one() {
            let meta = build_decode_metadata(
                &[1u32, 2, 3, 4, 5],
                &(0..5).map(|i| vec![i]).collect::<Vec<_>>(),
                &[0i64; 5],
            );
            assert_eq!(meta.cu_seqlens_q.len(), 6);
            assert_eq!(meta.cu_seqlens_k.len(), 6);
        }
    }

    mod build_prefill_metadata {
        use super::*;

        #[test]
        fn single_sequence() {
            let meta = build_prefill_metadata(&[8u32], &[8u32], &[0i64]);
            assert_eq!(meta.cu_seqlens_q, vec![0, 8]);
            assert_eq!(meta.cu_seqlens_k, vec![0, 8]);
        }

        #[test]
        fn batch_cumulative() {
            let meta = build_prefill_metadata(&[4u32, 8, 3], &[4u32, 16, 3], &[0i64; 3]);
            assert_eq!(meta.cu_seqlens_q, vec![0, 4, 12, 15]);
            assert_eq!(meta.cu_seqlens_k, vec![0, 4, 20, 23]);
        }

        #[test]
        fn is_prefill_true() {
            let meta = build_prefill_metadata(&[1u32], &[1u32], &[0i64]);
            assert!(meta.is_prefill);
        }

        #[test]
        fn max_seqlens() {
            let meta = build_prefill_metadata(&[4u32, 12, 3], &[4u32, 20, 3], &[0i64; 3]);
            assert_eq!(meta.max_seqlen_q, 12);
            assert_eq!(meta.max_seqlen_k, 20);
        }

        #[test]
        fn block_table_empty() {
            let meta = build_prefill_metadata(&[1u32], &[1u32], &[0i64]);
            assert!(meta.block_table.is_empty());
        }

        #[test]
        fn slot_mapping_passes_through() {
            let slots = vec![10i64, 20, 30];
            let meta = build_prefill_metadata(&[1u32, 1, 1], &[1u32, 1, 1], &slots);
            assert_eq!(meta.slot_mapping, slots);
        }

        #[test]
        fn empty_batch() {
            let meta = build_prefill_metadata(&[], &[], &[]);
            assert_eq!(meta.cu_seqlens_q, vec![0]);
            assert_eq!(meta.cu_seqlens_k, vec![0]);
        }
    }
}
