//! Per-step attention metadata — the scheduler→attention boundary.
//!
//! Carries `cu_seqlens_q` / `cu_seqlens_k` in the **flash-attn cumulative
//! convention** (prefix sums), NOT `context_lens` (vLLM's per-sequence lengths).
//! The conversion lives in `build_decode_metadata` (T10 finding).

#![allow(dead_code)]

use candle_core::{Device, Result, Tensor};

use super::math_layout::{MathLimits, QueryWork};
use super::AttentionEpoch;

#[derive(Debug, Clone, PartialEq)]
pub struct AttnMetadata {
    pub is_prefill: bool,
    pub cu_seqlens_q: Vec<u32>,
    pub cu_seqlens_k: Vec<u32>,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub slot_mapping: Vec<i64>,
    /// Logical-to-physical KV blocks. Populated for decode and continued
    /// prefill; initial prefill remains unpaged (prefix-hit reuse is #39).
    pub block_table: Vec<Vec<i32>>,
}

impl AttnMetadata {
    pub(crate) fn uses_paged_kv(&self) -> bool {
        !self.block_table.is_empty()
    }
}

/// Device-ready form of one logical [`AttnMetadata`] value.
///
/// Construction performs every host-to-device metadata upload for an engine
/// step. Transformer layers only borrow these tensors; they never reconstruct
/// them from the logical vectors.
#[derive(Debug)]
pub(crate) struct PreparedAttention {
    epoch: AttentionEpoch,
    logical: AttnMetadata,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    slot_mapping: Tensor,
    block_table: Option<Tensor>,
    rectangular: bool,
    real_rows: Option<Tensor>,
    packed_work: Vec<QueryWork>,
    cache_shape: Option<[usize; 4]>,
}

impl PreparedAttention {
    pub(crate) fn prepare(
        epoch: AttentionEpoch,
        logical: AttnMetadata,
        device: &Device,
    ) -> Result<Self> {
        Self::prepare_with_limits(epoch, logical, device, None)
    }

    pub(crate) fn prepare_with_limits(
        epoch: AttentionEpoch,
        logical: AttnMetadata,
        device: &Device,
        limits: Option<MathLimits>,
    ) -> Result<Self> {
        validate_logical_metadata(&logical)?;

        let (rectangular, packed_work) = match limits {
            Some(limits) => limits.plan(&logical)?,
            None => (
                false,
                logical
                    .cu_seqlens_q
                    .windows(2)
                    .map(|pair| QueryWork::standard((pair[1] - pair[0]) as usize))
                    .collect(),
            ),
        };
        let batch = logical.cu_seqlens_q.len() - 1;
        let real_rows = if rectangular && batch * logical.max_seqlen_q != logical.slot_mapping.len()
        {
            let mut indices = Vec::with_capacity(logical.slot_mapping.len());
            for (member, pair) in logical.cu_seqlens_q.windows(2).enumerate() {
                let count = (pair[1] - pair[0]) as usize;
                for row in 0..count {
                    indices.push(
                        u32::try_from(
                            member * logical.max_seqlen_q + logical.max_seqlen_q - count + row,
                        )
                        .map_err(candle_core::Error::wrap)?,
                    );
                }
            }
            let count = indices.len();
            Some(Tensor::from_vec(indices, count, device)?)
        } else {
            None
        };

        let cu_seqlens_q = Tensor::from_vec(
            logical.cu_seqlens_q.clone(),
            logical.cu_seqlens_q.len(),
            device,
        )?;
        let cu_seqlens_k = Tensor::from_vec(
            logical.cu_seqlens_k.clone(),
            logical.cu_seqlens_k.len(),
            device,
        )?;
        let slot_mapping = Tensor::from_vec(
            logical.slot_mapping.clone(),
            logical.slot_mapping.len(),
            device,
        )?;
        let block_table = prepare_block_table(&logical.block_table, device)?;

        Ok(Self {
            epoch,
            logical,
            cu_seqlens_q,
            cu_seqlens_k,
            slot_mapping,
            block_table,
            rectangular,
            real_rows,
            packed_work,
            cache_shape: limits.map(|l| [l.num_blocks, l.block_size, l.kv_heads, l.head_dim]),
        })
    }

    pub(crate) fn validate_cache(&self, shape: &[usize]) -> Result<()> {
        if shape.len() != 4 || shape.contains(&0) {
            candle_core::bail!("invalid paged attention cache shape")
        }
        if let Some(expected) = self.cache_shape {
            if shape != expected {
                candle_core::bail!("cache differs from prepared attention geometry")
            }
            return Ok(());
        }
        for (member, blocks) in self.logical.block_table.iter().enumerate() {
            let length = (self.logical.cu_seqlens_k[member + 1] - self.logical.cu_seqlens_k[member])
                as usize;
            if blocks.len() < length.div_ceil(shape[1])
                || blocks
                    .iter()
                    .any(|&b| usize::try_from(b).map_or(true, |id| id >= shape[0]))
            {
                candle_core::bail!("paged attention block table exceeds cache bounds")
            }
        }
        Ok(())
    }

    pub(crate) fn epoch(&self) -> AttentionEpoch {
        self.epoch
    }

    pub(crate) fn rectangular(&self) -> bool {
        self.rectangular
    }

    pub(crate) fn query_work(&self, member: usize) -> Result<QueryWork> {
        self.packed_work
            .get(member)
            .copied()
            .ok_or_else(|| candle_core::Error::msg("no packed attention work for member"))
    }

    pub(crate) fn pad_queries(&self, input: &Tensor) -> Result<Tensor> {
        if self.real_rows.is_none() {
            return Ok(input.clone());
        }
        if input.dim(0)? != self.logical.slot_mapping.len() {
            candle_core::bail!("query tensor does not match prepared token rows")
        }
        let mut parts = Vec::new();
        for pair in self.logical.cu_seqlens_q.windows(2) {
            let count = (pair[1] - pair[0]) as usize;
            parts.push(input.narrow(0, pair[0] as usize, count)?.pad_with_zeros(
                0,
                self.logical.max_seqlen_q - count,
                0,
            )?);
        }
        Tensor::cat(&parts, 0)
    }

    pub(crate) fn unpad_queries(&self, input: &Tensor) -> Result<Tensor> {
        match &self.real_rows {
            Some(indices) => input.index_select(indices, 0),
            None => Ok(input.clone()),
        }
    }

    pub(crate) fn logical(&self) -> &AttnMetadata {
        &self.logical
    }

    pub(crate) fn cu_seqlens_q(&self) -> &Tensor {
        &self.cu_seqlens_q
    }

    pub(crate) fn cu_seqlens_k(&self) -> &Tensor {
        &self.cu_seqlens_k
    }

    pub(crate) fn slot_mapping(&self) -> &Tensor {
        &self.slot_mapping
    }

    pub(crate) fn block_table(&self) -> Result<&Tensor> {
        self.block_table.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "prepared attention metadata has no paged block table".to_string(),
            )
        })
    }

    pub(crate) fn device_tensor_uploads(&self) -> usize {
        3 + usize::from(self.block_table.is_some()) + usize::from(self.real_rows.is_some())
    }
}

fn validate_logical_metadata(metadata: &AttnMetadata) -> Result<()> {
    if metadata.cu_seqlens_q.is_empty() || metadata.cu_seqlens_k.is_empty() {
        candle_core::bail!("attention cumulative lengths must start with zero")
    }
    if metadata.cu_seqlens_q.len() != metadata.cu_seqlens_k.len() {
        candle_core::bail!(
            "attention query/key cumulative length vectors must have the same batch shape"
        )
    }
    if metadata.cu_seqlens_q[0] != 0 || metadata.cu_seqlens_k[0] != 0 {
        candle_core::bail!("attention cumulative lengths must start with zero")
    }
    if !metadata
        .cu_seqlens_q
        .windows(2)
        .all(|pair| pair[0] <= pair[1])
        || !metadata
            .cu_seqlens_k
            .windows(2)
            .all(|pair| pair[0] <= pair[1])
    {
        candle_core::bail!("attention cumulative lengths must be monotonic")
    }
    let query_tokens = metadata.cu_seqlens_q.last().copied().unwrap_or(0) as usize;
    if metadata
        .cu_seqlens_q
        .windows(2)
        .any(|pair| (pair[1] - pair[0]) as usize > metadata.max_seqlen_q)
        || metadata
            .cu_seqlens_k
            .windows(2)
            .any(|pair| (pair[1] - pair[0]) as usize > metadata.max_seqlen_k)
    {
        candle_core::bail!("attention maximum lengths are smaller than a member length")
    }
    if metadata.slot_mapping.len() != query_tokens {
        candle_core::bail!(
            "attention slot mapping has {} entries for {query_tokens} query tokens",
            metadata.slot_mapping.len()
        )
    }
    if metadata
        .cu_seqlens_q
        .windows(2)
        .zip(metadata.cu_seqlens_k.windows(2))
        .any(|(queries, keys)| queries[1] - queries[0] > keys[1] - keys[0])
    {
        candle_core::bail!("causal self-attention queries exceed their KV lengths")
    }
    let batch_size = metadata.cu_seqlens_q.len() - 1;
    if !metadata.block_table.is_empty() && metadata.block_table.len() != batch_size {
        candle_core::bail!(
            "attention block table has {} rows for batch size {batch_size}",
            metadata.block_table.len()
        )
    }
    Ok(())
}

fn prepare_block_table(block_table: &[Vec<i32>], device: &Device) -> Result<Option<Tensor>> {
    if block_table.is_empty() {
        return Ok(None);
    }
    let max_blocks = block_table.iter().map(Vec::len).max().unwrap_or(0);
    if max_blocks == 0 {
        candle_core::bail!("paged attention block table rows must be non-empty")
    }
    let batch = block_table.len();
    let mut flat = Vec::with_capacity(batch * max_blocks);
    for row in block_table {
        if row.iter().any(|&block| block < 0) {
            candle_core::bail!("attention block IDs must be nonnegative")
        }
        flat.extend(
            row.iter()
                .map(|&id| u32::try_from(id))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(candle_core::Error::wrap)?,
        );
        flat.resize(flat.len() + max_blocks - row.len(), 0);
    }
    Ok(Some(Tensor::from_vec(flat, (batch, max_blocks), device)?))
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

pub(crate) fn build_continued_prefill_metadata(
    scheduled_tokens: &[u32],
    kv_lengths: &[u32],
    block_table: &[Vec<i32>],
    slot_mapping: &[i64],
) -> AttnMetadata {
    let mut metadata = build_prefill_metadata(scheduled_tokens, kv_lengths, slot_mapping);
    metadata.block_table = block_table.to_vec();
    metadata
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
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

    mod prepared_metadata {
        use super::*;

        #[test]
        fn initial_prefill_preserves_lengths_and_uses_three_device_tensors() {
            let prepared = PreparedAttention::prepare(
                AttentionEpoch::StepPlan(1),
                build_prefill_metadata(&[2, 3], &[2, 3], &[10, 11, 12, 13, 14]),
                &Device::Cpu,
            )
            .unwrap();

            assert_eq!(prepared.cu_seqlens_q().to_vec1::<u32>().unwrap(), [0, 2, 5]);
            assert_eq!(prepared.cu_seqlens_k().to_vec1::<u32>().unwrap(), [0, 2, 5]);
            assert_eq!(
                prepared.slot_mapping().to_vec1::<i64>().unwrap(),
                [10, 11, 12, 13, 14]
            );
            assert!(prepared.block_table().is_err());
            assert_eq!(prepared.device_tensor_uploads(), 3);
        }

        #[test]
        fn decode_pads_ragged_block_tables_without_changing_lengths() {
            let prepared = PreparedAttention::prepare(
                AttentionEpoch::StepPlan(2),
                build_decode_metadata(&[3, 513], &[vec![7], vec![8, 9, 10]], &[100, 200]),
                &Device::Cpu,
            )
            .unwrap();

            assert_eq!(prepared.cu_seqlens_q().to_vec1::<u32>().unwrap(), [0, 1, 2]);
            assert_eq!(
                prepared.cu_seqlens_k().to_vec1::<u32>().unwrap(),
                [0, 3, 516]
            );
            assert_eq!(
                prepared.block_table().unwrap().to_vec2::<u32>().unwrap(),
                [vec![7, 0, 0], vec![8, 9, 10]]
            );
            assert_eq!(prepared.device_tensor_uploads(), 4);
        }

        #[test]
        fn chunked_continuation_keeps_new_query_and_full_kv_horizons() {
            let prepared = PreparedAttention::prepare(
                AttentionEpoch::StepPlan(3),
                build_continued_prefill_metadata(&[2], &[4], &[vec![3]], &[20, 21]),
                &Device::Cpu,
            )
            .unwrap();

            assert_eq!(prepared.cu_seqlens_q().to_vec1::<u32>().unwrap(), [0, 2]);
            assert_eq!(prepared.cu_seqlens_k().to_vec1::<u32>().unwrap(), [0, 4]);
            assert_eq!(prepared.slot_mapping().to_vec1::<i64>().unwrap(), [20, 21]);
            assert_eq!(prepared.block_table().unwrap().dims(), [1, 1]);
        }

        #[test]
        fn mixed_prefix_hit_and_miss_keep_independent_causal_horizons() {
            let prepared = PreparedAttention::prepare(
                AttentionEpoch::StepPlan(4),
                build_continued_prefill_metadata(
                    &[1, 3],
                    &[513, 3],
                    &[vec![4, 5, 6], vec![7]],
                    &[30, 40, 41, 42],
                ),
                &Device::Cpu,
            )
            .unwrap();

            assert_eq!(prepared.cu_seqlens_q().to_vec1::<u32>().unwrap(), [0, 1, 4]);
            assert_eq!(
                prepared.cu_seqlens_k().to_vec1::<u32>().unwrap(),
                [0, 513, 516]
            );
            assert_eq!(
                prepared.block_table().unwrap().to_vec2::<u32>().unwrap(),
                [vec![4, 5, 6], vec![7, 0, 0]]
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod layout_tests {
    use super::*;
    use candle_core::DType;
    #[test]
    fn padded_queries_roundtrip_and_preserve_only_real_cache_slots() {
        let limits = MathLimits {
            query_heads: 4,
            kv_heads: 2,
            head_dim: 8,
            block_size: 4,
            num_blocks: 2,
            max_query_tokens: 8,
            workspace_bytes: 1 << 20,
        };
        let metadata = build_prefill_metadata(&[1, 3], &[1, 3], &[0, 4, 5, 6]);
        let prepared = PreparedAttention::prepare_with_limits(
            AttentionEpoch::StepPlan(1),
            metadata,
            &Device::Cpu,
            Some(limits),
        )
        .unwrap();
        let input = Tensor::new(&[10u32, 20, 21, 22], &Device::Cpu).unwrap();
        let padded = prepared.pad_queries(&input).unwrap();
        assert_eq!(padded.to_vec1::<u32>().unwrap(), [0, 0, 10, 20, 21, 22]);
        assert_eq!(
            prepared
                .unpad_queries(&padded)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap(),
            [10, 20, 21, 22]
        );
        assert_eq!(
            prepared.slot_mapping().to_vec1::<i64>().unwrap(),
            [0, 4, 5, 6]
        );
        assert_eq!(prepared.device_tensor_uploads(), 4);
        assert!(prepared.validate_cache(&[2, 4, 2, 8]).is_ok());
        assert!(prepared.validate_cache(&[1, 4, 2, 8]).is_err());
        assert_eq!(prepared.slot_mapping().dtype(), DType::I64);
    }
}
