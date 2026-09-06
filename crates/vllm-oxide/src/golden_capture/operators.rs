//! Small deterministic L0 probes. No public API or alternate production backend.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

use crate::layers::{activation::silu_and_mul, rmsnorm::RMSNorm, rope::RotaryEmbedding};
use crate::sampler::{selected_token_ids_to_host, Sampler};
use crate::SamplingParams;

const PLAN: &str = "VLLM_OXIDE_INTERNAL_OPERATOR_PLAN";
const OUTPUT: &str = "VLLM_OXIDE_INTERNAL_OPERATOR_OUTPUT";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    profile_id: String,
    rule_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    protocol: String,
    schema_version: u32,
    profiles: Vec<Profile>,
}

pub(crate) fn requested() -> bool {
    std::env::var_os(PLAN).is_some() || std::env::var_os(OUTPUT).is_some()
}

pub(crate) fn run_from_env(device: &Device) -> Result<()> {
    let request: Request = serde_json::from_slice(&std::fs::read(
        std::env::var_os(PLAN).context("operator plan missing")?,
    )?)?;
    if request.protocol != "layered-accuracy-v1"
        || request.schema_version != 1
        || request.profiles.is_empty()
        || request.profiles.iter().any(|p| p.profile_id.is_empty())
        || request
            .profiles
            .iter()
            .map(|p| &p.profile_id)
            .collect::<HashSet<_>>()
            .len()
            != request.profiles.len()
    {
        bail!("invalid operator request");
    }
    let destination = PathBuf::from(std::env::var_os(OUTPUT).context("operator output missing")?);
    let parent = super::validate_private_temp_dir(
        destination
            .parent()
            .context("operator output parent missing")?,
    )?;
    let name = destination
        .file_name()
        .context("operator output name missing")?;
    super::validate_destination_name(name)?;
    let destination = parent.join(name);
    if destination.exists() || destination.is_symlink() {
        bail!("operator output must be fresh");
    }
    let mut records = Vec::new();
    for profile in request.profiles {
        let (output, shape, dtype) = run_rule(&profile.rule_id, device)?;
        let values = output
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if values.iter().any(|v| !v.is_finite()) {
            bail!("nonfinite operator output");
        }
        records.push(
            serde_json::json!({"profile_id":profile.profile_id,"rule_id":profile.rule_id,
            "input_shape":shape,"input_dtype":dtype,"output_shape":output.dims(),"values":values}),
        );
    }
    let value = serde_json::json!({"protocol":"layered-accuracy-v1","schema_version":1,"mode":"operator_verification",
        "device":if device.is_cuda(){"cuda:0"}else{"cpu"},"operator_checks":records,"accepting":false,"complete":true});
    let stage = destination.with_extension("partial");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&stage)?;
    serde_json::to_writer(&mut file, &value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    std::fs::hard_link(&stage, &destination)?;
    std::fs::remove_file(stage)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn run_rule(rule: &str, device: &Device) -> Result<(Tensor, Vec<usize>, &'static str)> {
    match rule {
        "materialized_halfway_sum_v1" => {
            let x = Tensor::ones((1, 4), DType::BF16, device)?;
            let residual =
                Tensor::new(&[[1.0_f32 / 256.0, 1.0 / 256.0, 1.0 / 256.0, 0.0]], device)?
                    .to_dtype(DType::BF16)?;
            let norm = RMSNorm::new(Tensor::ones(4, DType::BF16, device)?, 1e-6);
            Ok((norm.forward(&x, Some(&residual))?.0, vec![1, 4], "bfloat16"))
        }
        "gate_2_up_0.515625_v1" => {
            let x = Tensor::new(&[[2.0_f32, 0.515_625]], device)?.to_dtype(DType::BF16)?;
            Ok((silu_and_mul(&x)?, vec![1, 2], "bfloat16"))
        }
        "nonconsecutive_positions_and_half_rotation_v1" => {
            let mut values = Vec::new();
            for _ in 0..3 {
                for head in [0.0_f32, 1.0] {
                    for dimension in 0..128_u16 {
                        values.push((f32::from(dimension % 7) - 3.0) / 4.0 + head / 8.0);
                    }
                }
            }
            let x = Tensor::from_vec(values, (3, 2, 128), device)?.to_dtype(DType::BF16)?;
            let rope = RotaryEmbedding::new(128, 128, 256, 1_000_000.0, device)?;
            Ok((
                rope.forward(&Tensor::new(&[0_u32, 7, 255], device)?, &x, &x)?
                    .0,
                vec![3, 2, 128],
                "bfloat16",
            ))
        }
        "unique_and_multiway_max_with_filter_penalties_v1" => {
            let vocabulary = 151_936;
            let mut values = vec![-100.0_f32; 3 * vocabulary];
            values[7] = 4.0;
            values[12] = 4.0;
            values[vocabulary + 7] = 4.0;
            values[vocabulary + 12] = 3.0;
            for id in [29, 30, 31] {
                values[2 * vocabulary + id] = 4.0;
            }
            let params = [
                SamplingParams::default(),
                SamplingParams {
                    frequency_penalty: 2.0,
                    ..SamplingParams::default()
                },
                SamplingParams {
                    temperature: 1.0,
                    top_k: Some(1),
                    ..SamplingParams::default()
                },
            ];
            let sampled = Sampler::new_with_seed(0).forward(
                &Tensor::from_vec(values, (3, vocabulary), device)?,
                &params,
                &[vec![], vec![7], vec![]],
            )?;
            let ids = selected_token_ids_to_host(&sampled)?;
            Ok((
                Tensor::new(ids.as_slice(), device)?,
                vec![3, vocabulary],
                "float32",
            ))
        }
        "asymmetric_qkv_causal_gqa_v1"
        | "257_visible_tokens_noncontiguous_pages_v1"
        | "noncontiguous_block_readback_v1" => gpu_rule(rule, device),
        _ => bail!("unsupported operator input rule: {rule}"),
    }
}

#[cfg(not(feature = "cuda"))]
fn gpu_rule(_rule: &str, _device: &Device) -> Result<(Tensor, Vec<usize>, &'static str)> {
    bail!("attention/KV operator verification requires the CUDA build")
}

#[cfg(feature = "cuda")]
fn gpu_rule(rule: &str, device: &Device) -> Result<(Tensor, Vec<usize>, &'static str)> {
    use crate::attention::metadata::PreparedAttention;
    use crate::attention::{AttentionEpoch, AttnMetadata, PagedKVCache};
    use candle_core::IndexOp;
    if !device.is_cuda() {
        bail!("GPU operator profile received a non-CUDA device");
    }
    let prefill = rule == "asymmetric_qkv_causal_gqa_v1";
    let (n, qn) = if prefill {
        (5_usize, 5_usize)
    } else {
        (257, 1)
    };
    let mut q = vec![0.0_f32; qn * 16 * 128];
    let mut k = vec![0.0_f32; n * 8 * 128];
    let mut v = vec![0.0_f32; n * 8 * 128];
    for row in 0..qn {
        for head in 0..16 {
            q[(row * 16 + head) * 128] =
                0.25 * f32::from(u16::try_from(row + 1)?) * f32::from(u16::try_from(head % 3 + 1)?);
        }
    }
    for row in 0..n {
        for head in 0..8 {
            k[(row * 8 + head) * 128] = 0.125
                * f32::from(u16::try_from(row % 7 + 1)?)
                * f32::from(u16::try_from(head % 3 + 1)?);
            for d in 0..128 {
                v[(row * 8 + head) * 128 + d] = f32::from(u16::try_from(row % 11)?) / 8.0
                    + f32::from(u16::try_from(head)?) / 16.0
                    + f32::from(u16::try_from(d % 13)?) / 64.0;
            }
        }
    }
    let q = Tensor::from_vec(q, (qn, 16, 128), device)?.to_dtype(DType::BF16)?;
    let k = Tensor::from_vec(k, (n, 8, 128), device)?.to_dtype(DType::BF16)?;
    let v = Tensor::from_vec(v, (n, 8, 128), device)?.to_dtype(DType::BF16)?;
    let scale = 1.0_f32 / 128.0_f32.sqrt();
    if prefill {
        let metadata = AttnMetadata {
            is_prefill: true,
            cu_seqlens_q: vec![0, 5],
            cu_seqlens_k: vec![0, 5],
            max_seqlen_q: 5,
            max_seqlen_k: 5,
            slot_mapping: (0..5).collect(),
            block_table: vec![],
        };
        let prepared = PreparedAttention::prepare(AttentionEpoch::WarmupPrefill, metadata, device)?;
        return Ok((
            crate::attention::flash_attn::prefill_attn(&q, &k, &v, &prepared, scale)?,
            vec![5, 16, 128],
            "bfloat16",
        ));
    }
    let cache = PagedKVCache::new(1, 2, 256, 8, 128, DType::BF16, device)?;
    let slots = (256_i64..512).chain(std::iter::once(0)).collect::<Vec<_>>();
    cache.reshape_and_cache(0, &k, &v, &Tensor::new(slots.as_slice(), device)?)?;
    let kc = cache.k_cache(0)?;
    let vc = cache.v_cache(0)?;
    if rule == "noncontiguous_block_readback_v1" {
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for &slot in &slots {
            let slot = usize::try_from(slot)?;
            keys.push(kc.i((slot / 256, slot % 256, .., ..))?);
            values.push(vc.i((slot / 256, slot % 256, .., ..))?);
        }
        let keys = Tensor::stack(&keys, 0)?;
        let values = Tensor::stack(&values, 0)?;
        return Ok((
            Tensor::stack(&[keys, values], 0)?,
            vec![2, 256, 8, 128],
            "bfloat16",
        ));
    }
    let metadata = AttnMetadata {
        is_prefill: false,
        cu_seqlens_q: vec![0, 1],
        cu_seqlens_k: vec![0, 257],
        max_seqlen_q: 1,
        max_seqlen_k: 257,
        slot_mapping: vec![0],
        block_table: vec![vec![1, 0]],
    };
    let prepared = PreparedAttention::prepare(AttentionEpoch::WarmupDecode, metadata, device)?;
    Ok((
        crate::attention::flash_attn::paged_attn(&q, &kc, &vc, &prepared, scale, 256)?,
        vec![1, 16, 128],
        "bfloat16",
    ))
}
