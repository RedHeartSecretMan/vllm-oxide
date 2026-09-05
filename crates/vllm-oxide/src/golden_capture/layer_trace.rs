//! Fixed-case private diagnostics, never an L3 fixture producer.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor};
use serde::Deserialize;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub(crate) const TRACE_ENV: &str = "VLLM_OXIDE_INTERNAL_LAYER_TRACE_DIR";

const REGRESSION11_PREFIX: [u32; 11] = [7281, 11, 1128, 374, 279, 6010, 504, 16629, 362, 311, 279];
// Original regression_11 input at the pinned tokenizer revision.
const REGRESSION11_PROMPT: [u32; 54] = [
    2679, 264, 5426, 10901, 16629, 362, 20971, 518, 220, 21, 15, 8756, 817, 6460, 323, 2441, 5426,
    10901, 16629, 425, 20971, 518, 220, 24, 15, 8756, 817, 6460, 8841, 1817, 1008, 11, 323, 279,
    17337, 525, 220, 18, 15, 15, 8756, 10747, 11, 1246, 1293, 686, 432, 1896, 369, 279, 27688, 311,
    3367, 30,
];

#[derive(Deserialize)]
#[serde(tag = "prompt_id", deny_unknown_fields)]
enum Request {
    #[serde(rename = "canonical_03")]
    Canonical03 {
        token_ids: Vec<u32>,
        decode_token: u32,
    },
    #[serde(rename = "regression_11")]
    Regression11 {
        token_ids: Vec<u32>,
        decode_tokens: Vec<u32>,
    },
}

impl Request {
    fn parse(bytes: &str) -> Result<Self> {
        let request: Self = serde_json::from_str(bytes)?;
        let valid = match &request {
            Self::Canonical03 {
                token_ids,
                decode_token,
            } => *decode_token == 151_667 && !token_ids.is_empty() && token_ids.len() <= 1024,
            Self::Regression11 {
                token_ids,
                decode_tokens,
            } => {
                token_ids.as_slice() == REGRESSION11_PROMPT
                    && decode_tokens.as_slice() == REGRESSION11_PREFIX
            }
        };
        if !valid {
            bail!("layer trace request differs from the fixed case/prefix boundary");
        }
        Ok(request)
    }

    fn prompt_id(&self) -> &'static str {
        match self {
            Self::Canonical03 { .. } => "canonical_03",
            Self::Regression11 { .. } => "regression_11",
        }
    }

    fn token_ids(&self) -> &[u32] {
        match self {
            Self::Canonical03 { token_ids, .. } | Self::Regression11 { token_ids, .. } => token_ids,
        }
    }

    fn decode_tokens(&self) -> &[u32] {
        match self {
            Self::Canonical03 { decode_token, .. } => std::slice::from_ref(decode_token),
            Self::Regression11 { decode_tokens, .. } => decode_tokens,
        }
    }
}

pub(crate) struct LayerTrace {
    directory: PathBuf,
    request: Request,
    next_step: Mutex<usize>,
}

impl LayerTrace {
    pub(crate) fn from_env() -> Result<Option<Self>> {
        let Some(root) = std::env::var_os(TRACE_ENV) else {
            return Ok(None);
        };
        let root = PathBuf::from(root);
        let request = Request::parse(&std::fs::read_to_string(root.join("request.json"))?)?;
        let directory = super::validate_private_temp_dir(&root.join("rust"))?;
        Ok(Some(Self::new(directory, request)))
    }

    fn new(directory: PathBuf, request: Request) -> Self {
        Self {
            directory,
            request,
            next_step: Mutex::new(0),
        }
    }

    pub(crate) fn begin(&self, tokens: &Tensor, positions: &Tensor) -> Result<Option<StepTrace>> {
        self.begin_capture(std::env::var_os(super::CALL_ID_ENV), tokens, positions)
    }

    fn begin_capture(
        &self,
        call_id: Option<std::ffi::OsString>,
        tokens: &Tensor,
        positions: &Tensor,
    ) -> Result<Option<StepTrace>> {
        // LLM::new performs real warmup before the consumer installs the
        // existing golden capture call. Do not inspect tensors or advance
        // the fixed diagnostic until that request boundary is active.
        if call_id.is_none() {
            return Ok(None);
        }
        self.begin_values(
            &tokens.to_dtype(DType::U32)?.to_vec1::<u32>()?,
            &positions.to_dtype(DType::U32)?.to_vec1::<u32>()?,
        )
        .map(Some)
    }

    fn begin_values(&self, tokens: &[u32], positions: &[u32]) -> Result<StepTrace> {
        let mut step = self
            .next_step
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned layer trace"))?;
        let length = u32::try_from(self.request.token_ids().len())?;
        let valid = if *step == 0 {
            tokens == self.request.token_ids() && positions == (0..length).collect::<Vec<_>>()
        } else {
            self.request
                .decode_tokens()
                .get(*step - 1)
                .is_some_and(|token| {
                    tokens == [*token]
                        && positions == [length + u32::try_from(*step - 1).unwrap_or(u32::MAX)]
                })
        };
        if !valid {
            bail!("layer trace input tokens/positions exceed the fixed diagnostic request");
        }
        let trace = StepTrace::create_for(
            &self.directory,
            *step,
            tokens,
            positions,
            self.request.prompt_id(),
        )?;
        *step += 1;
        Ok(trace)
    }
}

pub(crate) struct StepTrace {
    output: File,
    #[cfg(any(feature = "cuda", test))]
    directory: PathBuf,
    #[cfg(any(feature = "cuda", test))]
    step: usize,
    attention_written: bool,
    next_checkpoint: usize,
    shape: Option<Vec<usize>>,
    layer0_only: bool,
}

impl StepTrace {
    #[cfg(test)]
    fn create(directory: &Path, step: usize, tokens: &[u32], positions: &[u32]) -> Result<Self> {
        Self::create_for(directory, step, tokens, positions, "canonical_03")
    }

    fn create_for(
        directory: &Path,
        step: usize,
        tokens: &[u32],
        positions: &[u32],
        prompt_id: &str,
    ) -> Result<Self> {
        let output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(directory.join(format!("step-{step}.jsonl")))?;
        let mut trace = Self {
            output,
            #[cfg(any(feature = "cuda", test))]
            directory: directory.to_path_buf(),
            #[cfg(any(feature = "cuda", test))]
            step,
            attention_written: false,
            next_checkpoint: 0,
            shape: None,
            layer0_only: prompt_id == "regression_11",
        };
        trace.write(&serde_json::json!({"kind":"header", "diagnostic_only":true,
            "accepting":false, "prompt_id":prompt_id, "step":step,
            "token_ids":tokens, "positions":positions}))?;
        Ok(trace)
    }

    pub(crate) fn record(&mut self, name: &str, value: &Tensor) -> Result<()> {
        if self.layer0_only && self.next_checkpoint >= 11 {
            bail!("regression_11 trace stops after layer zero");
        }
        if value.dtype() != DType::BF16 {
            bail!("layer trace requires BF16");
        }
        let expected = match self.next_checkpoint {
            0 => "embedding".to_string(),
            1 => "layer0_input_norm".to_string(),
            2 => "layer0_q".to_string(),
            3 => "layer0_k".to_string(),
            4 => "layer0_v".to_string(),
            5 => "layer0_q_norm".to_string(),
            6 => "layer0_k_norm".to_string(),
            7 => "layer0_q_rope".to_string(),
            8 => "layer0_k_rope".to_string(),
            9 => "layer0_attention_context".to_string(),
            10..=37 => format!("layer_{}", self.next_checkpoint - 10),
            38 => "final_norm".to_string(),
            _ => bail!("layer trace has extra checkpoints"),
        };
        if name != expected
            || value.rank() != 2
            || self.shape.as_deref().is_some_and(|shape| {
                let width = if matches!(
                    name,
                    "layer0_q" | "layer0_q_norm" | "layer0_q_rope" | "layer0_attention_context"
                ) {
                    shape[1] * 2
                } else {
                    shape[1]
                };
                value.dims() != [shape[0], width]
            })
        {
            bail!("layer trace checkpoint order or shape mismatch");
        }
        let values = value.contiguous()?.flatten_all()?.to_vec1::<half::bf16>()?;
        if values.iter().any(|v| !v.is_finite()) {
            bail!("layer trace contains non-finite values");
        }
        let bits: Vec<u16> = values.iter().map(|v| v.to_bits()).collect();
        self.write(&serde_json::json!({"kind":"checkpoint", "name":name,
            "dtype":"BF16", "shape":value.dims(), "bf16_bits":bits}))?;
        if self.shape.is_none() {
            self.shape = Some(value.dims().to_vec());
        }
        self.next_checkpoint += 1;
        Ok(())
    }

    pub(crate) fn layer0_only(&self) -> bool {
        self.layer0_only
    }

    pub(crate) fn finish(mut self) -> Result<()> {
        let checkpoints = if self.layer0_only { 11 } else { 39 };
        if self.next_checkpoint != checkpoints || !self.attention_written {
            bail!("incomplete layer trace");
        }
        self.write(
            &serde_json::json!({"kind":"trailer", "complete":true, "checkpoints":checkpoints}),
        )?;
        self.output.sync_all()?;
        Ok(())
    }

    fn write(&mut self, value: &serde_json::Value) -> Result<()> {
        serde_json::to_writer(&mut self.output, value)?;
        self.output.write_all(b"\n")?;
        self.output.flush()?;
        Ok(())
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn record_attention(&mut self, call: AttentionCall) -> Result<()> {
        if self.attention_written {
            bail!("duplicate attention call evidence");
        }
        let value = call.evidence(self.step)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(self.directory.join(format!("attention-{}.json", self.step)))?;
        serde_json::to_writer(&mut file, &value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        self.attention_written = true;
        Ok(())
    }
}

/// Values actually passed by the fixed FA wrappers, not inferred from dtype.
#[cfg(any(feature = "cuda", test))]
pub(crate) struct AttentionCall {
    pub cu_q: Vec<u32>,
    pub cu_k: Vec<u32>,
    pub max_q: usize,
    pub max_k: usize,
    pub heads: (usize, usize, usize),
    pub scale: f32,
    pub causal: Option<bool>,
    pub window: (Option<usize>, Option<usize>),
}

#[cfg(any(feature = "cuda", test))]
impl AttentionCall {
    fn evidence(self, step: usize) -> Result<serde_json::Value> {
        if self.cu_q.len() != 2
            || self.cu_k.len() != 2
            || self.cu_q[0] != 0
            || self.cu_k[0] != 0
            || usize::try_from(self.cu_q[1])? != self.max_q
            || usize::try_from(self.cu_k[1])? != self.max_k
            || self.max_q == 0
            || self.max_k < self.max_q
            || self.max_k > 1025
            || self.heads.1 == 0
            || self.heads.0 % self.heads.1 != 0
        {
            bail!("attention call is outside the single-request diagnostic");
        }
        let mut allowed = Vec::with_capacity(self.max_q * self.max_k);
        let head_mapping: Vec<usize> = (0..self.heads.0)
            .map(|head| head / (self.heads.0 / self.heads.1))
            .collect();
        for row in 0..self.max_q {
            let center = if self.causal.is_some() {
                row
            } else {
                row + self.max_k - self.max_q
            };
            for column in 0..self.max_k {
                allowed.push(
                    (!self.causal.unwrap_or(false) || column <= row)
                        && self.window.0.map_or(true, |left| column + left >= center)
                        && self.window.1.map_or(true, |right| column <= center + right),
                );
            }
        }
        Ok(
            serde_json::json!({"diagnostic_only":true,"accepting":false,"step":step,
            "raw":{"backend":if self.causal.is_some(){"flash_attn_varlen"}else{"flash_attn_varlen_paged_windowed"},
                "cu_seqlens_q":self.cu_q,"cu_seqlens_k":self.cu_k,"max_seqlen_q":self.max_q,"max_seqlen_k":self.max_k,
                "causal_argument":self.causal,"window_left":self.window.0,"window_right":self.window.1,
                "scale_f32_bits":self.scale.to_bits()},
            "common":{"q_length":self.max_q,"k_length":self.max_k,"q_heads":self.heads.0,"kv_heads":self.heads.1,
                "head_dim":self.heads.2,"head_mapping":head_mapping,"scale_argument":f64::from(self.scale),"dropout_p":0.0,
                "mask":{"kind":"visibility","shape":[self.max_q,self.max_k],"allowed":allowed}}}),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn regression11_rejects_a_different_prompt_of_the_same_length() {
        let request = serde_json::json!({"prompt_id":"regression_11",
            "token_ids":vec![17_u32;54],
            "decode_tokens":[7281,11,1128,374,279,6010,504,16629,362,311,279]});
        assert!(Request::parse(&request.to_string()).is_err());
    }

    #[test]
    fn regression11_trace_finishes_after_layer_zero_without_later_layers() {
        let directory = tempfile::tempdir().unwrap();
        let mut trace =
            StepTrace::create_for(directory.path(), 11, &[279], &[64], "regression_11").unwrap();
        let hidden = Tensor::zeros((1, 1024), DType::BF16, &Device::Cpu).unwrap();
        let query = Tensor::zeros((1, 2048), DType::BF16, &Device::Cpu).unwrap();
        for name in [
            "embedding",
            "layer0_input_norm",
            "layer0_q",
            "layer0_k",
            "layer0_v",
            "layer0_q_norm",
            "layer0_k_norm",
            "layer0_q_rope",
            "layer0_k_rope",
            "layer0_attention_context",
            "layer_0",
        ] {
            let value = if matches!(
                name,
                "layer0_q" | "layer0_q_norm" | "layer0_q_rope" | "layer0_attention_context"
            ) {
                &query
            } else {
                &hidden
            };
            trace.record(name, value).unwrap();
        }
        trace
            .record_attention(AttentionCall {
                cu_q: vec![0, 1],
                cu_k: vec![0, 65],
                max_q: 1,
                max_k: 65,
                heads: (16, 8, 128),
                scale: 0.088_388_346,
                causal: None,
                window: (None, Some(0)),
            })
            .unwrap();
        trace.finish().unwrap();
        let bytes = std::fs::read_to_string(directory.path().join("step-11.jsonl")).unwrap();
        assert_eq!(bytes.lines().count(), 13);
        assert!(!bytes.contains("final_norm"));
    }

    #[test]
    fn regression11_capture_keeps_warmup_out_and_stops_at_first_divergence() {
        let directory = tempfile::tempdir().unwrap();
        let tokens = REGRESSION11_PROMPT.to_vec();
        let prefix = vec![
            7281_u32, 11, 1128, 374, 279, 6010, 504, 16629, 362, 311, 279,
        ];
        let request = Request::parse(
            &serde_json::json!({"prompt_id":"regression_11", "token_ids":tokens,
                "decode_tokens":prefix})
            .to_string(),
        )
        .unwrap();
        let trace = LayerTrace::new(directory.path().to_path_buf(), request);
        let warmup = Tensor::new(&[0_u32], &Device::Cpu).unwrap();
        assert!(trace
            .begin_capture(None, &warmup, &warmup)
            .unwrap()
            .is_none());
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        trace
            .begin_values(&tokens, &(0..54).collect::<Vec<_>>())
            .unwrap();
        for (offset, token) in prefix.iter().enumerate() {
            let position = u32::try_from(54 + offset).unwrap();
            assert!(trace.begin_values(&[u32::MAX], &[position]).is_err());
            assert!(trace.begin_values(&[*token], &[position + 1]).is_err());
            trace.begin_values(&[*token], &[position]).unwrap();
        }
        assert!(trace.begin_values(&[1], &[65]).is_err());
        let header = std::fs::read_to_string(directory.path().join("step-11.jsonl")).unwrap();
        let value: serde_json::Value = serde_json::from_str(header.trim()).unwrap();
        assert_eq!(value["prompt_id"], "regression_11");
        assert_eq!(value["positions"], serde_json::json!([64]));
        assert_eq!(value["accepting"], false);
    }

    #[test]
    fn initialization_warmup_does_not_write_or_consume_a_capture_step() {
        let directory = tempfile::tempdir().unwrap();
        let request = Request::parse(
            r#"{"prompt_id":"canonical_03","token_ids":[17,18],"decode_token":151667}"#,
        )
        .unwrap();
        let trace = LayerTrace::new(directory.path().to_path_buf(), request);
        let warmup = Tensor::new(&[0_u32, 0], &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0_u32, 1], &Device::Cpu).unwrap();
        assert!(trace
            .begin_capture(None, &warmup, &positions)
            .unwrap()
            .is_none());
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        let actual = Tensor::new(&[17_u32, 18], &Device::Cpu).unwrap();
        assert!(trace
            .begin_capture(Some("capture".into()), &actual, &positions)
            .unwrap()
            .is_some());
        assert!(directory.path().join("step-0.jsonl").exists());
        trace.begin_values(&[151_667], &[2]).unwrap();
    }

    #[test]
    fn trace_records_bf16_bits_shape_and_materialized_residual_without_mutation() {
        let directory = tempfile::tempdir().unwrap();
        let hidden = Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let residual = Tensor::new(&[[1.0f32 / 256.0, 0.0]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let mut trace = StepTrace::create(directory.path(), 0, &[17], &[0]).unwrap();
        trace
            .record("embedding", &(&hidden + &residual).unwrap())
            .unwrap();
        let bytes = std::fs::read_to_string(directory.path().join("step-0.jsonl")).unwrap();
        let row: serde_json::Value = serde_json::from_str(bytes.lines().nth(1).unwrap()).unwrap();
        assert_eq!(row["bf16_bits"], serde_json::json!([16256, 16384]));
        assert_eq!(row["shape"], serde_json::json!([1, 2]));
        assert_eq!(
            hidden
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![1.0, 2.0]]
        );
    }

    #[test]
    fn session_rejects_wrong_case_prefix_position_and_third_forward() {
        let directory = tempfile::tempdir().unwrap();
        assert!(Request::parse(
            r#"{"prompt_id":"canonical_04","token_ids":[17],"decode_token":151667}"#
        )
        .is_err());
        let request = Request::parse(
            r#"{"prompt_id":"canonical_03","token_ids":[17,18],"decode_token":151667}"#,
        )
        .unwrap();
        let trace = LayerTrace::new(directory.path().to_path_buf(), request);
        assert!(trace.begin_values(&[19, 18], &[0, 1]).is_err());
        assert!(trace.begin_values(&[17, 18], &[1, 2]).is_err());
        trace.begin_values(&[17, 18], &[0, 1]).unwrap();
        assert!(trace.begin_values(&[151_668], &[2]).is_err());
        trace.begin_values(&[151_667], &[2]).unwrap();
        assert!(trace.begin_values(&[198], &[3]).is_err());
    }

    #[test]
    fn trace_requires_ordered_complete_checkpoints_and_never_overwrites() {
        let directory = tempfile::tempdir().unwrap();
        let value = Tensor::zeros((2, 2), DType::BF16, &Device::Cpu).unwrap();
        let mut trace = StepTrace::create(directory.path(), 0, &[17, 18], &[0, 1]).unwrap();
        assert!(StepTrace::create(directory.path(), 0, &[17, 18], &[0, 1]).is_err());
        assert!(trace.record("layer_0", &value).is_err());
        trace.record("embedding", &value).unwrap();
        trace.record("layer0_input_norm", &value).unwrap();
        let qkv = Tensor::new(
            &[
                [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            ],
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        trace
            .record("layer0_q", &qkv.narrow(1, 0, 4).unwrap())
            .unwrap();
        trace
            .record("layer0_k", &qkv.narrow(1, 4, 2).unwrap())
            .unwrap();
        trace
            .record("layer0_v", &qkv.narrow(1, 6, 2).unwrap())
            .unwrap();
        trace
            .record("layer0_q_norm", &qkv.narrow(1, 0, 4).unwrap())
            .unwrap();
        trace.record("layer0_k_norm", &value).unwrap();
        trace
            .record("layer0_q_rope", &qkv.narrow(1, 0, 4).unwrap())
            .unwrap();
        trace.record("layer0_k_rope", &value).unwrap();
        trace
            .record_attention(AttentionCall {
                cu_q: vec![0, 2],
                cu_k: vec![0, 2],
                max_q: 2,
                max_k: 2,
                heads: (16, 8, 128),
                scale: 0.088_388_346,
                causal: Some(true),
                window: (None, None),
            })
            .unwrap();
        trace
            .record("layer0_attention_context", &qkv.narrow(1, 0, 4).unwrap())
            .unwrap();
        for layer in 0..28 {
            trace.record(&format!("layer_{layer}"), &value).unwrap();
        }
        trace.record("final_norm", &value).unwrap();
        trace.finish().unwrap();
        let bytes = std::fs::read_to_string(directory.path().join("step-0.jsonl")).unwrap();
        assert_eq!(bytes.lines().count(), 41);
        let q: serde_json::Value = serde_json::from_str(bytes.lines().nth(3).unwrap()).unwrap();
        assert_eq!(
            q["bf16_bits"],
            serde_json::json!([0, 16256, 16384, 16448, 16640, 16656, 16672, 16688])
        );
        assert_eq!(q["shape"], serde_json::json!([2, 4]));
        let call: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(directory.path().join("attention-0.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            call["common"]["mask"]["allowed"],
            serde_json::json!([true, false, true, true])
        );
        assert_eq!(
            call["common"]["scale_argument"].as_f64().unwrap(),
            f64::from(0.088_388_346_f32)
        );
        assert!(StepTrace::create(directory.path(), 1, &[151_667], &[1])
            .unwrap()
            .finish()
            .is_err());
    }
}
