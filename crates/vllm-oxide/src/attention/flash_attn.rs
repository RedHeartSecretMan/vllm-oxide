//! EXPERIMENT ONLY: dense FP64 attention to isolate the accuracy failure.
//! This branch is not a production or acceptance candidate.
#![cfg(feature = "cuda")]
#![allow(dead_code)]
use super::PreparedAttention;
use candle_core::{DType, Result, Tensor};
pub(crate) const PREFILL_CAUSAL: bool = true;
pub(crate) const PAGED_WINDOW_LEFT: Option<usize> = None;
pub(crate) const PAGED_WINDOW_RIGHT: Option<usize> = Some(0);

fn dense(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let (queries, heads, dim) = q.dims3()?;
    let (keys, kv_heads, _) = k.dims3()?;
    if queries > 4096 || keys > 4096 || keys < queries || heads % kv_heads != 0 {
        candle_core::bail!("experimental dense attention exceeds bounded diagnostic scope");
    }
    let expand = |input: &Tensor| -> Result<Tensor> {
        input
            .unsqueeze(2)?
            .broadcast_as((keys, kv_heads, heads / kv_heads, dim))?
            .contiguous()?
            .reshape((keys, heads, dim))?
            .transpose(0, 1)?
            .contiguous()?
            .to_dtype(DType::F64)
    };
    let q64 = q.to_dtype(DType::F64)?.transpose(0, 1)?.contiguous()?;
    let k64 = expand(k)?;
    let v64 = expand(v)?;
    let root_scale = f64::from(scale).sqrt();
    let scaled_q = (q64 * root_scale)?;
    let scaled_k = (k64 * root_scale)?.transpose(1, 2)?;
    let mut output_chunks = Vec::new();
    for start in (0..queries).step_by(128) {
        let count = (queries - start).min(128);
        let scores = scaled_q.narrow(1, start, count)?.matmul(&scaled_k)?;
        let mask = (start..start + count)
            .flat_map(|row| {
                (0..keys).map(move |col| {
                    if col <= keys - queries + row {
                        0.0_f64
                    } else {
                        f64::NEG_INFINITY
                    }
                })
            })
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(mask, (count, keys), q.device())?;
        let probabilities = candle_nn::ops::softmax(&scores.broadcast_add(&mask)?, 2)?;
        output_chunks.push(probabilities.matmul(&v64)?);
    }
    Tensor::cat(&output_chunks, 1)?
        .transpose(0, 1)?
        .contiguous()?
        .to_dtype(q.dtype())
}

pub fn prefill_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prepared: &PreparedAttention,
    scale: f32,
) -> Result<Tensor> {
    let logical = prepared.logical();
    let mut outputs = Vec::new();
    for index in 0..logical.cu_seqlens_q.len() - 1 {
        let qstart = logical.cu_seqlens_q[index] as usize;
        let qlen = (logical.cu_seqlens_q[index + 1] - logical.cu_seqlens_q[index]) as usize;
        let kstart = logical.cu_seqlens_k[index] as usize;
        let klen = (logical.cu_seqlens_k[index + 1] - logical.cu_seqlens_k[index]) as usize;
        outputs.push(dense(
            &q.narrow(0, qstart, qlen)?,
            &k.narrow(0, kstart, klen)?,
            &v.narrow(0, kstart, klen)?,
            scale,
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
    _block_size: usize,
) -> Result<Tensor> {
    let logical = prepared.logical();
    let mut outputs = Vec::new();
    for index in 0..logical.cu_seqlens_q.len() - 1 {
        let qstart = logical.cu_seqlens_q[index] as usize;
        let qlen = (logical.cu_seqlens_q[index + 1] - logical.cu_seqlens_q[index]) as usize;
        let klen = (logical.cu_seqlens_k[index + 1] - logical.cu_seqlens_k[index]) as usize;
        let blocks = logical.block_table[index]
            .iter()
            .map(|&id| u32::try_from(id).map_err(candle_core::Error::wrap))
            .collect::<Result<Vec<_>>>()?;
        let count = blocks.len();
        let blocks = Tensor::from_vec(blocks, count, q.device())?;
        let keys = k
            .index_select(&blocks, 0)?
            .flatten(0, 1)?
            .narrow(0, 0, klen)?;
        let values = v
            .index_select(&blocks, 0)?
            .flatten(0, 1)?
            .narrow(0, 0, klen)?;
        outputs.push(dense(&q.narrow(0, qstart, qlen)?, &keys, &values, scale)?);
    }
    Tensor::cat(&outputs, 0)
}
