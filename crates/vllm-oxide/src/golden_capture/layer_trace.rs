//! Temporary-scope canonical_03 diagnostic, never an L3 fixture producer.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor};
use serde::Deserialize;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub(crate) const TRACE_ENV: &str = "VLLM_OXIDE_INTERNAL_LAYER_TRACE_DIR";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    prompt_id: String,
    token_ids: Vec<u32>,
    decode_token: u32,
}

impl Request {
    fn parse(bytes: &str) -> Result<Self> {
        let request: Self = serde_json::from_str(bytes)?;
        if request.prompt_id != "canonical_03"
            || request.decode_token != 151_667
            || request.token_ids.is_empty()
            || request.token_ids.len() > 1024
        {
            bail!("layer trace only permits canonical_03 prefill and first decode");
        }
        Ok(request)
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

    pub(crate) fn begin(&self, tokens: &Tensor, positions: &Tensor) -> Result<StepTrace> {
        self.begin_values(
            &tokens.to_dtype(DType::U32)?.to_vec1::<u32>()?,
            &positions.to_dtype(DType::U32)?.to_vec1::<u32>()?,
        )
    }

    fn begin_values(&self, tokens: &[u32], positions: &[u32]) -> Result<StepTrace> {
        let mut step = self
            .next_step
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned layer trace"))?;
        let length = u32::try_from(self.request.token_ids.len())?;
        let valid = match *step {
            0 => tokens == self.request.token_ids && positions == (0..length).collect::<Vec<_>>(),
            1 => tokens == [self.request.decode_token] && positions == [length],
            _ => false,
        };
        if !valid {
            bail!("layer trace input tokens/positions exceed the exact two-step request");
        }
        let trace = StepTrace::create(&self.directory, *step, tokens, positions)?;
        *step += 1;
        Ok(trace)
    }
}

pub(crate) struct StepTrace {
    output: File,
    next_checkpoint: usize,
    shape: Option<Vec<usize>>,
}

impl StepTrace {
    fn create(directory: &Path, step: usize, tokens: &[u32], positions: &[u32]) -> Result<Self> {
        let output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(directory.join(format!("step-{step}.jsonl")))?;
        let mut trace = Self {
            output,
            next_checkpoint: 0,
            shape: None,
        };
        trace.write(&serde_json::json!({"kind":"header", "diagnostic_only":true,
            "accepting":false, "prompt_id":"canonical_03", "step":step,
            "token_ids":tokens, "positions":positions}))?;
        Ok(trace)
    }

    pub(crate) fn record(&mut self, name: &str, value: &Tensor) -> Result<()> {
        if value.dtype() != DType::BF16 {
            bail!("layer trace requires BF16");
        }
        let expected = match self.next_checkpoint {
            0 => "embedding".to_string(),
            1..=28 => format!("layer_{}", self.next_checkpoint - 1),
            29 => "final_norm".to_string(),
            _ => bail!("layer trace has extra checkpoints"),
        };
        if name != expected
            || value.rank() != 2
            || self
                .shape
                .as_deref()
                .is_some_and(|shape| shape != value.dims())
        {
            bail!("layer trace checkpoint order or shape mismatch");
        }
        let values = value.flatten_all()?.to_vec1::<half::bf16>()?;
        if values.iter().any(|v| !v.is_finite()) {
            bail!("layer trace contains non-finite values");
        }
        let bits: Vec<u16> = values.iter().map(|v| v.to_bits()).collect();
        self.write(&serde_json::json!({"kind":"checkpoint", "name":name,
            "dtype":"BF16", "shape":value.dims(), "bf16_bits":bits}))?;
        self.shape = Some(value.dims().to_vec());
        self.next_checkpoint += 1;
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<()> {
        if self.next_checkpoint != 30 {
            bail!("incomplete layer trace");
        }
        self.write(&serde_json::json!({"kind":"trailer", "complete":true, "checkpoints":30}))?;
        self.output.sync_all()?;
        Ok(())
    }

    fn write(&mut self, value: &serde_json::Value) -> Result<()> {
        serde_json::to_writer(&mut self.output, value)?;
        self.output.write_all(b"\n")?;
        self.output.flush()?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

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
        let value = Tensor::zeros((1, 2), DType::BF16, &Device::Cpu).unwrap();
        let mut trace = StepTrace::create(directory.path(), 0, &[17], &[0]).unwrap();
        assert!(StepTrace::create(directory.path(), 0, &[17], &[0]).is_err());
        assert!(trace.record("layer_0", &value).is_err());
        trace.record("embedding", &value).unwrap();
        for layer in 0..28 {
            trace.record(&format!("layer_{layer}"), &value).unwrap();
        }
        trace.record("final_norm", &value).unwrap();
        trace.finish().unwrap();
        let bytes = std::fs::read_to_string(directory.path().join("step-0.jsonl")).unwrap();
        assert_eq!(bytes.lines().count(), 32);
        assert!(StepTrace::create(directory.path(), 1, &[151_667], &[1])
            .unwrap()
            .finish()
            .is_err());
    }
}
