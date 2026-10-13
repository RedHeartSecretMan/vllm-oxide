//! Bounded FP32 causal attention over BF16/F16 projections and paged KV.
//! PreparedAttention owns layout selection and device metadata for the step.
#![cfg(feature = "cuda")]

use super::{
    math_ops::{self, Mask},
    PreparedAttention,
};
use candle_core::{DType, Result, Tensor};

pub(crate) const PREFILL_CAUSAL: bool = true;
pub(crate) const PAGED_WINDOW_LEFT: Option<usize> = None;
pub(crate) const PAGED_WINDOW_RIGHT: Option<usize> = Some(0);

fn validate(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<()> {
    let (_, heads, dim) = q.dims3()?;
    let kv_heads = k.dim(k.rank() - 2)?;
    if heads == 0
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
        || dim == 0
        || k.dim(k.rank() - 1)? != dim
        || k.dims() != v.dims()
        || q.dtype() != k.dtype()
        || q.dtype() != v.dtype()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || !scale.is_finite()
        || scale <= 0.0
    {
        candle_core::bail!("inconsistent attention projection/cache geometry")
    }
    Ok(())
}

/// Inputs are rectangular [batch, tokens, heads, width]; output is flattened
/// [batch * queries, query_heads, width]. The single-member path uses this
/// same arithmetic, including the separately materialized Q/K scaling.
#[allow(clippy::too_many_arguments)]
fn dense(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prepared: &PreparedAttention,
    scale: f32,
    member_offset: usize,
    chunk: usize,
) -> Result<Tensor> {
    let (batch, queries, heads, dim) = q.dims4()?;
    let (_, keys, kv_heads, _) = k.dims4()?;
    let expand = |input: &Tensor| -> Result<Tensor> {
        input
            .unsqueeze(3)?
            .broadcast_as((batch, keys, kv_heads, heads / kv_heads, dim))?
            .contiguous()?
            .reshape((batch, keys, heads, dim))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch * heads, keys, dim))?
            .to_dtype(DType::F32)
    };
    let query = q
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch * heads, queries, dim))?
        .to_dtype(DType::F32)?;
    let key = expand(k)?;
    let value = expand(v)?;
    let root_scale = f64::from(scale).sqrt();
    let query = (query * root_scale)?;
    let key = (key * root_scale)?.transpose(1, 2)?;
    let mut outputs = Vec::new();
    for start in (0..queries).step_by(chunk) {
        let count = (queries - start).min(chunk);
        let input = query.narrow(1, start, count)?;
        let rows = super::math_layout::score_rows(queries, keys, count);
        let scores = if rows > count {
            let zeros = Tensor::zeros((batch * heads, rows - count, dim), DType::F32, q.device())?;
            Tensor::cat(&[&input, &zeros], 1)?
                .matmul(&key)?
                .narrow(1, 0, count)?
                .contiguous()?
        } else {
            input.matmul(&key)?
        };
        let probabilities = math_ops::softmax(
            &scores,
            prepared,
            Mask {
                queries,
                keys,
                heads,
                query_start: start,
                chunk_rows: count,
                member_offset,
            },
        )?;
        outputs.push(probabilities.matmul(&value)?);
    }
    Tensor::cat(&outputs, 1)?
        .reshape((batch, heads, queries, dim))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch * queries, heads, dim))?
        .to_dtype(q.dtype())
}

fn packed_member(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prepared: &PreparedAttention,
    scale: f32,
    member: usize,
) -> Result<Tensor> {
    let work = prepared.query_work(member)?;
    if !work.split_heads {
        return dense(
            &q.unsqueeze(0)?,
            &k.unsqueeze(0)?,
            &v.unsqueeze(0)?,
            prepared,
            scale,
            member,
            work.query_chunk,
        );
    }
    let (_, heads, _) = q.dims3()?;
    let kv_heads = k.dim(1)?;
    let mut outputs = Vec::with_capacity(heads);
    for head in 0..heads {
        let kv_head = head / (heads / kv_heads);
        outputs.push(dense(
            &q.narrow(1, head, 1)?.unsqueeze(0)?,
            &k.narrow(1, kv_head, 1)?.unsqueeze(0)?,
            &v.narrow(1, kv_head, 1)?.unsqueeze(0)?,
            prepared,
            scale,
            member,
            work.query_chunk,
        )?);
    }
    Tensor::cat(&outputs, 1)
}

pub fn prefill_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prepared: &PreparedAttention,
    scale: f32,
) -> Result<Tensor> {
    validate(q, k, v, scale)?;
    let logical = prepared.logical();
    if logical.cu_seqlens_q != logical.cu_seqlens_k {
        candle_core::bail!("unpaged prefill requires complete query/key sequences")
    }
    let batch = logical.cu_seqlens_q.len() - 1;
    if prepared.rectangular() {
        let (_, heads, dim) = q.dims3()?;
        let kv_heads = k.dim(1)?;
        return dense(
            &q.reshape((batch, logical.max_seqlen_q, heads, dim))?,
            &k.reshape((batch, logical.max_seqlen_k, kv_heads, dim))?,
            &v.reshape((batch, logical.max_seqlen_k, kv_heads, dim))?,
            prepared,
            scale,
            0,
            logical.max_seqlen_q,
        );
    }
    let mut outputs = Vec::with_capacity(batch);
    for member in 0..batch {
        let start = logical.cu_seqlens_q[member] as usize;
        let count = (logical.cu_seqlens_q[member + 1] - logical.cu_seqlens_q[member]) as usize;
        outputs.push(packed_member(
            &q.narrow(0, start, count)?,
            &k.narrow(0, start, count)?,
            &v.narrow(0, start, count)?,
            prepared,
            scale,
            member,
        )?);
    }
    Tensor::cat(&outputs, 0)
}

pub fn paged_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prepared: &PreparedAttention,
    scale: f32,
    block_size: usize,
) -> Result<Tensor> {
    validate(q, k, v, scale)?;
    prepared.validate_cache(k.dims())?;
    if k.dim(1)? != block_size {
        candle_core::bail!("attention page size differs from cache")
    }
    let logical = prepared.logical();
    let batch = logical.cu_seqlens_q.len() - 1;
    if prepared.rectangular() {
        let (_, heads, dim) = q.dims3()?;
        return dense(
            &q.reshape((batch, logical.max_seqlen_q, heads, dim))?,
            &math_ops::gather_cache(k, prepared)?,
            &math_ops::gather_cache(v, prepared)?,
            prepared,
            scale,
            0,
            logical.max_seqlen_q,
        );
    }
    let mut outputs = Vec::with_capacity(batch);
    for member in 0..batch {
        let start = logical.cu_seqlens_q[member] as usize;
        let queries = (logical.cu_seqlens_q[member + 1] - logical.cu_seqlens_q[member]) as usize;
        outputs.push(packed_member(
            &q.narrow(0, start, queries)?,
            &math_ops::gather_member(k, prepared, member)?,
            &math_ops::gather_member(v, prepared, member)?,
            prepared,
            scale,
            member,
        )?);
    }
    Tensor::cat(&outputs, 0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::attention::{
        build_continued_prefill_metadata, math_layout::MathLimits, AttentionEpoch,
    };
    use candle_core::Device;

    #[test]
    #[ignore = "requires a guarded CUDA owner"]
    fn cuda_rectangular_paged_matches_causal_means_and_packed_fallback() {
        let device = Device::new_cuda(0).unwrap();
        // Two members: Q=[1,2], K=[2,4], with reordered physical blocks.
        let metadata =
            build_continued_prefill_metadata(&[1, 2], &[2, 4], &[vec![2], vec![1, 0]], &[5, 0, 1]);
        let limits = MathLimits {
            query_heads: 4,
            kv_heads: 2,
            head_dim: 8,
            block_size: 2,
            num_blocks: 3,
            max_query_tokens: 4,
            workspace_bytes: 1 << 20,
        };
        let rectangular = PreparedAttention::prepare_with_limits(
            AttentionEpoch::StepPlan(1),
            metadata.clone(),
            &device,
            Some(limits),
        )
        .unwrap();
        let packed = PreparedAttention::prepare_with_limits(
            AttentionEpoch::StepPlan(2),
            metadata,
            &device,
            Some(MathLimits {
                max_query_tokens: 3,
                workspace_bytes: 16 * 1024,
                ..limits
            }),
        )
        .unwrap();
        assert!(rectangular.rectangular());
        assert!(!packed.rectangular());
        assert!(packed.query_work(1).unwrap().split_heads);
        let q = Tensor::zeros((3, 4, 8), DType::BF16, &device).unwrap();
        let k = Tensor::zeros((3, 2, 2, 8), DType::BF16, &device).unwrap();
        let data: Vec<f32> = [6f32, 8., 2., 4., 10., 14.]
            .into_iter()
            .flat_map(|v| std::iter::repeat_n(v, 16))
            .collect();
        let v = Tensor::from_vec(data, (3, 2, 2, 8), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let rect = paged_attn(
            &rectangular.pad_queries(&q).unwrap(),
            &k,
            &v,
            &rectangular,
            0.5,
            2,
        )
        .unwrap();
        let rect = rectangular.unpad_queries(&rect).unwrap();
        let ragged = paged_attn(&q, &k, &v, &packed, 0.5, 2).unwrap();
        let expected: Vec<f32> = [12f32, 4., 5.]
            .into_iter()
            .flat_map(|v| std::iter::repeat_n(v, 32))
            .collect();
        assert_eq!(
            rect.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected
        );
        assert_eq!(
            ragged
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod long_tests {
    use super::*;
    use crate::attention::{
        build_continued_prefill_metadata, math_layout::MathLimits, AttentionEpoch,
    };
    use candle_core::Device;

    #[test]
    #[ignore = "requires a guarded CUDA owner"]
    fn cuda_long_keys_zero_padded_queries_and_chunked_attention() {
        let device = Device::new_cuda(0).unwrap();
        let metadata = build_continued_prefill_metadata(
            &[1, 2],
            &[1025, 1026],
            &[vec![0, 1, 2, 3, 4], vec![0, 1, 2, 3, 4]],
            &[1024, 1024, 1025],
        );
        let limits = MathLimits {
            query_heads: 2,
            kv_heads: 1,
            head_dim: 8,
            block_size: 256,
            num_blocks: 5,
            max_query_tokens: 4,
            workspace_bytes: 1 << 22,
        };
        let prepared = PreparedAttention::prepare_with_limits(
            AttentionEpoch::StepPlan(3),
            metadata,
            &device,
            Some(limits),
        )
        .unwrap();
        let q = Tensor::zeros((4, 2, 8), DType::BF16, &device).unwrap();
        let k = Tensor::zeros((5, 256, 1, 8), DType::BF16, &device).unwrap();
        let v = Tensor::ones((5, 256, 1, 8), DType::BF16, &device).unwrap();
        let output = paged_attn(&q, &k, &v, &prepared, 0.5, 256).unwrap();
        let values = output
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(&values[..16], &[0f32; 16]);
        assert_eq!(&values[16..], &[1f32; 48]);
        // Force a query tile boundary with exactly representable uniform PV.
        let q = q.reshape((2, 2, 2, 8)).unwrap();
        let k = math_ops::gather_cache(&k, &prepared).unwrap();
        let v = math_ops::gather_cache(&v, &prepared).unwrap();
        let tiled = dense(&q, &k, &v, &prepared, 0.5, 0, 1).unwrap();
        assert_eq!(
            tiled
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            values
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod causal_layout_regressions {
    use super::*;
    use crate::attention::math_layout::MathLimits;
    use crate::attention::metadata::build_continued_prefill_metadata;
    use crate::attention::AttentionEpoch;
    use candle_core::Device;
    use half::bf16;

    #[test]
    #[ignore = "requires a guarded CUDA owner"]
    fn cuda_causal_suffix_query_matches_full_prefill_rounding() {
        const QUERIES: usize = 3;
        const KEYS: usize = 257;
        const HEADS: usize = 16;
        const KV_HEADS: usize = 8;
        const DIM: usize = 128;
        let decode = |bytes: &[u8], count| {
            assert_eq!(bytes.len(), 2 * count);
            bytes
                .chunks_exact(2)
                .map(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])))
                .collect::<Vec<_>>()
        };
        let q = decode(
            include_bytes!("../llm/test_data/query-suffix-q.bf16"),
            QUERIES * DIM,
        );
        let k = decode(
            include_bytes!("../llm/test_data/query-suffix-k.bf16"),
            KEYS * DIM,
        );
        let v = decode(
            include_bytes!("../llm/test_data/query-suffix-v.bf16"),
            KEYS * DIM,
        );
        let mut queries = vec![bf16::ZERO; QUERIES * HEADS * DIM];
        let mut keys = vec![bf16::ZERO; 2 * 256 * KV_HEADS * DIM];
        let mut values = keys.clone();
        for row in 0..QUERIES {
            queries[(row * HEADS + 4) * DIM..(row * HEADS + 5) * DIM]
                .copy_from_slice(&q[row * DIM..(row + 1) * DIM]);
        }
        for row in 0..KEYS {
            keys[(row * KV_HEADS + 2) * DIM..(row * KV_HEADS + 3) * DIM]
                .copy_from_slice(&k[row * DIM..(row + 1) * DIM]);
            values[(row * KV_HEADS + 2) * DIM..(row * KV_HEADS + 3) * DIM]
                .copy_from_slice(&v[row * DIM..(row + 1) * DIM]);
        }
        let device = Device::new_cuda(0).unwrap();
        let query = Tensor::from_vec(queries, (QUERIES, HEADS, DIM), &device).unwrap();
        let key = Tensor::from_vec(keys, (2, 256, KV_HEADS, DIM), &device).unwrap();
        let value = Tensor::from_vec(values, (2, 256, KV_HEADS, DIM), &device).unwrap();
        let metadata =
            build_continued_prefill_metadata(&[3], &[257], &[vec![0, 1]], &[254, 255, 256]);
        let prepared = PreparedAttention::prepare_with_limits(
            AttentionEpoch::StepPlan(1),
            metadata,
            &device,
            Some(MathLimits {
                query_heads: HEADS,
                kv_heads: KV_HEADS,
                head_dim: DIM,
                block_size: 256,
                num_blocks: 2,
                max_query_tokens: 128,
                workspace_bytes: 64 * 1024 * 1024,
            }),
        )
        .unwrap();
        let output = paged_attn(&query, &key, &value, &prepared, 0.088_388_346, 256).unwrap();
        assert_eq!(output.dims(), [QUERIES, HEADS, DIM]);
        let bits = output.flatten_all().unwrap().to_vec1::<bf16>().unwrap();
        // Captured full-prefill result. The unpadded three-query path is 0xbcd5.
        assert_eq!(bits[4 * DIM + 52].to_bits(), 0xbcd4);
    }
}
