//! Select an attention layout before allocating per-step device metadata.
//!
//! Rectangular computation preserves batched reference arithmetic when it fits
//! the execution budget. Packed computation bounds large or uneven workloads;
//! its fallback processes individual query heads rather than expanding every
//! GQA key/value head at once.

use candle_core::Result;

use super::metadata::AttnMetadata;

#[derive(Debug, Clone, Copy)]
pub(crate) struct MathLimits {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub num_blocks: usize,
    pub max_query_tokens: usize,
    pub workspace_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QueryWork {
    pub query_chunk: usize,
    pub split_heads: bool,
}

impl QueryWork {
    pub(crate) fn standard(queries: usize) -> Self {
        Self {
            query_chunk: if queries <= 1024 { queries.max(1) } else { 128 },
            split_heads: false,
        }
    }
}

fn product(values: &[usize]) -> Option<usize> {
    values
        .iter()
        .try_fold(1usize, |value, &factor| value.checked_mul(factor))
}

fn sum(values: &[Option<usize>]) -> Option<usize> {
    values
        .iter()
        .try_fold(0usize, |total, value| total.checked_add((*value)?))
}

impl MathLimits {
    fn full_bytes(&self, batch: usize, queries: usize, keys: usize, chunk: usize) -> Option<usize> {
        // Conservative live-buffer allowance: packing/casts/output assembly,
        // expanded GQA K/V, then scores, probabilities and masking workspace.
        sum(&[
            product(&[batch, queries, self.query_heads, self.head_dim, 32]),
            product(&[batch, keys, self.query_heads, self.head_dim, 16]),
            product(&[batch, self.query_heads, chunk, keys, 24]),
        ])
    }

    fn split_bytes(&self, queries: usize, keys: usize, chunk: usize) -> Option<usize> {
        sum(&[
            product(&[queries, self.query_heads, self.head_dim, 8]),
            product(&[keys, self.kv_heads, self.head_dim, 4]),
            product(&[keys, self.head_dim, 16]),
            product(&[queries, self.head_dim, 16]),
            product(&[chunk, keys, 24]),
        ])
    }

    pub(crate) fn plan(&self, metadata: &AttnMetadata) -> Result<(bool, Vec<QueryWork>)> {
        if self.query_heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || self.block_size == 0
            || self.num_blocks == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.max_query_tokens == 0
        {
            candle_core::bail!("invalid attention compute geometry or token budget")
        }
        let capacity = self
            .num_blocks
            .checked_mul(self.block_size)
            .ok_or_else(|| candle_core::Error::msg("KV slot capacity overflows"))?;
        for &slot in &metadata.slot_mapping {
            if slot >= 0 && usize::try_from(slot).map_err(candle_core::Error::wrap)? >= capacity {
                candle_core::bail!("attention slot mapping exceeds the allocated KV cache")
            }
        }
        for (member, blocks) in metadata.block_table.iter().enumerate() {
            let keys = (metadata.cu_seqlens_k[member + 1] - metadata.cu_seqlens_k[member]) as usize;
            if blocks.len() < keys.div_ceil(self.block_size)
                || blocks
                    .iter()
                    .any(|&block| usize::try_from(block).map_or(true, |id| id >= self.num_blocks))
            {
                candle_core::bail!(
                    "attention block table exceeds or does not cover the allocated KV cache"
                )
            }
        }
        let batch = metadata.cu_seqlens_q.len() - 1;
        let queries = metadata.max_seqlen_q;
        let keys = metadata.max_seqlen_k;
        let rectangular = batch > 1
            && batch
                .checked_mul(queries)
                .is_some_and(|rows| rows <= self.max_query_tokens)
            && self
                .full_bytes(batch, queries, keys, queries)
                .is_some_and(|bytes| bytes <= self.workspace_bytes);
        if rectangular {
            return Ok((true, Vec::new()));
        }

        // Completed members remain live until the final packed concatenation.
        // Reserve both the retained outputs and the assembled batch first.
        let output_bytes = product(&[
            metadata.slot_mapping.len(),
            self.query_heads,
            self.head_dim,
            8,
        ])
        .ok_or_else(|| candle_core::Error::msg("attention output shape overflows"))?;
        let member_workspace = self
            .workspace_bytes
            .checked_sub(output_bytes)
            .ok_or_else(|| {
                candle_core::Error::msg("packed attention outputs exceed workspace budget")
            })?;
        let mut work = Vec::with_capacity(batch);
        for index in 0..batch {
            let queries =
                (metadata.cu_seqlens_q[index + 1] - metadata.cu_seqlens_q[index]) as usize;
            let keys = (metadata.cu_seqlens_k[index + 1] - metadata.cu_seqlens_k[index]) as usize;
            let standard = QueryWork::standard(queries);
            if self
                .full_bytes(1, queries, keys, standard.query_chunk)
                .is_some_and(|bytes| bytes <= member_workspace)
            {
                work.push(standard);
                continue;
            }
            let mut chunk = queries.clamp(1, 128);
            while chunk > 1
                && self
                    .split_bytes(queries, keys, chunk)
                    .is_none_or(|bytes| bytes > member_workspace)
            {
                chunk = chunk.div_ceil(2);
            }
            if self
                .split_bytes(queries, keys, chunk)
                .is_none_or(|bytes| bytes > member_workspace)
            {
                candle_core::bail!("attention workspace exceeds {} bytes; lower gpu_memory_utilization or reduce the context/batch budget", self.workspace_bytes)
            }
            work.push(QueryWork {
                query_chunk: chunk,
                split_heads: true,
            });
        }
        Ok((false, work))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::attention::build_prefill_metadata;

    fn limits() -> MathLimits {
        MathLimits {
            query_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            block_size: 256,
            num_blocks: 64,
            max_query_tokens: 2048,
            workspace_bytes: 512 * 1024 * 1024,
        }
    }
    #[test]
    fn rectangular_padding_requires_both_token_and_workspace_capacity() {
        let m = build_prefill_metadata(&[9, 31], &[9, 31], &vec![0; 40]);
        assert!(limits().plan(&m).unwrap().0);
        assert!(
            !MathLimits {
                max_query_tokens: 40,
                ..limits()
            }
            .plan(&m)
            .unwrap()
            .0
        );
        let bytes = limits().full_bytes(1, 31, 31, 31).unwrap() + 40 * 16 * 128 * 8;
        let (rect, packed) = MathLimits {
            workspace_bytes: bytes,
            ..limits()
        }
        .plan(&m)
        .unwrap();
        assert!(!rect);
        assert_eq!(packed.len(), 2);
        assert!(packed.iter().all(|work| !work.split_heads));
    }
    #[test]
    fn large_context_splits_heads_then_reduces_query_chunk_or_fails() {
        let m = build_prefill_metadata(&[1024], &[1024], &vec![0; 1024]);
        let bytes = limits().split_bytes(1024, 1024, 16).unwrap() + 1024 * 16 * 128 * 8;
        let (_, work) = MathLimits {
            workspace_bytes: bytes,
            ..limits()
        }
        .plan(&m)
        .unwrap();
        assert!(work[0].split_heads);
        assert_eq!(work[0].query_chunk, 16);
        assert!(MathLimits {
            workspace_bytes: 1,
            ..limits()
        }
        .plan(&m)
        .is_err());
    }
    #[test]
    fn invalid_physical_cache_addresses_are_rejected_before_upload() {
        let mut m = build_prefill_metadata(&[1], &[1], &[64 * 256]);
        assert!(limits().plan(&m).is_err());
        m.slot_mapping[0] = -1;
        assert!(limits().plan(&m).is_ok());
        m.block_table = vec![vec![64]];
        assert!(limits().plan(&m).is_err());
    }
    #[test]
    fn packed_output_accumulation_is_reserved_for_the_whole_batch() {
        let m = build_prefill_metadata(&vec![1; 1024], &vec![1; 1024], &vec![0; 1024]);
        let result = MathLimits {
            max_query_tokens: 1,
            workspace_bytes: 100 * 1024,
            ..limits()
        }
        .plan(&m);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("outputs exceed workspace"));
    }
}
