#![allow(clippy::unwrap_used, clippy::expect_used)]

//! T19 acceptance demo — load a Qwen3 checkpoint via the registry, run a
//! forward pass, and print the hidden-state + logits shapes.
//!
//! Usage:
//!   cargo run --example forward_qwen3 --features cuda -- /path/to/Qwen3-0.6B
//!   cargo run --example forward_qwen3 --features cuda -- hub:Qwen/Qwen3-0.6B
//!
//! What you should see: the resolved architecture, the loaded model's vocab
//! size + device, the hidden-states shape from `forward(dummy_input_ids,
//! positions)`, and the logits shape from `compute_logits(hidden)`.

use std::path::PathBuf;
use std::process::ExitCode;

use candle_core::{DType, Device, Tensor};
use vllm_oxide::{Source, build_model};

fn parse_arg(arg: &str) -> Source {
    if let Some(rest) = arg.strip_prefix("hub:") {
        if let Some((repo, rev)) = rest.split_once('@') {
            Source::Hub { repo: repo.to_string(), revision: Some(rev.to_string()) }
        } else {
            Source::Hub { repo: rest.to_string(), revision: None }
        }
    } else {
        Source::Local(PathBuf::from(arg))
    }
}

fn main() -> ExitCode {
    let arg = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "hub:Qwen/Qwen3-0.6B".to_string());
    let source = parse_arg(&arg);

    eprintln!("[demo] source = {arg}");

    let device = match Device::cuda_if_available(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[demo] CUDA init failed: {e}");
            eprintln!("[demo] attention requires CUDA — falling back to CPU for construction only");
            Device::Cpu
        }
    };
    eprintln!("[demo] device = {device:?}");

    let max_model_len = 4096;
    let mut built = match build_model(source, &device, max_model_len) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[demo] build_model failed: {e:#}");
            return ExitCode::from(1);
        }
    };

    eprintln!("[demo] vocab_size  = {}", built.model.vocab_size());
    eprintln!("[demo] device       = {:?}", built.model.device());

    let input_ids = Tensor::zeros((4,), DType::U32, &device).unwrap();
    let positions = Tensor::arange(0u32, 4u32, &device).unwrap();

    eprintln!("[demo] forward(dummy_input_ids=[0,0,0,0], positions=[0,1,2,3])...");
    match built.model.forward(&input_ids, &positions) {
        Ok(hidden) => {
            eprintln!("[demo] hidden_states shape = {:?}", hidden.shape());
            match built.model.compute_logits(&hidden) {
                Ok(logits) => eprintln!("[demo] logits shape     = {:?}", logits.shape()),
                Err(e) => eprintln!("[demo] compute_logits failed: {e}"),
            }
        }
        Err(e) => {
            eprintln!("[demo] forward failed: {e}");
            eprintln!("[demo] (expected if running without --features cuda)");
            return ExitCode::from(2);
        }
    }

    ExitCode::SUCCESS
}
