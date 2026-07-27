use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use vllm_oxide::{LLM, Prompt, SamplingParams, Source};

/// Thin CLI over vllm-oxide's `LLM::generate`.
#[derive(Parser)]
#[command(
    name = "vllm-oxide",
    about = "Single-GPU, Qwen3-focused offline LLM inference engine",
    version
)]
struct Cli {
    /// Model: local checkpoint path or HuggingFace repo id (e.g. "Qwen/Qwen3-0.6B").
    /// Existing directories resolve to local checkpoints; everything else resolves to HuggingFace Hub.
    #[arg(short, long)]
    model: String,

    /// Prompt text. Reads from stdin when not provided.
    prompt: Option<String>,

    /// Sampling temperature. 0 = greedy (default).
    #[arg(long, default_value = "0")]
    temperature: f32,

    /// Top-k sampling: keep only the k highest-logit tokens.
    #[arg(long)]
    top_k: Option<usize>,

    /// Top-p (nucleus) sampling: keep smallest token set with cumulative prob ≥ p.
    #[arg(long)]
    top_p: Option<f32>,

    /// Maximum tokens to generate. Defaults to 16.
    #[arg(long, default_value = "16")]
    max_tokens: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let prompt_text = match cli.prompt {
        Some(text) => text,
        None => {
            use std::io::Read;
            let mut buffer = String::new();
            std::io::stdin()
                .read_to_string(&mut buffer)
                .context("reading prompt from stdin")?;
            buffer
        }
    };

    let model_path = PathBuf::from(&cli.model);
    let source = if model_path.is_dir() {
        Source::Local(model_path)
    } else {
        Source::Hub {
            repo: cli.model,
            revision: None,
        }
    };

    let mut llm = LLM::new(source, vllm_oxide::EngineOptions::default())
        .context("failed to initialise LLM")?;

    let sampling_params = SamplingParams {
        temperature: cli.temperature,
        top_k: cli.top_k,
        top_p: cli.top_p,
        max_tokens: cli.max_tokens,
        ..SamplingParams::default()
    };

    let outputs = llm.generate(
        &[Prompt::Text(prompt_text)],
        &[sampling_params],
    ).context("LLM::generate failed")?;

    if let Some(output) = outputs.first() {
        print!("{}", output.text);
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn model_local_directory() {
        let tmp = std::env::temp_dir();
        let path_str = tmp.to_str().unwrap();
        let source = parse_model_arg(path_str);
        match source {
            Source::Local(p) => assert_eq!(p, tmp),
            other => panic!("expected Source::Local, got {other:?}"),
        }
    }

    #[test]
    fn model_hub_repo() {
        let source = parse_model_arg("Qwen/Qwen3-0.6B");
        match source {
            Source::Hub { repo, revision } => {
                assert_eq!(repo, "Qwen/Qwen3-0.6B");
                assert!(revision.is_none());
            }
            other => panic!("expected Source::Hub, got {other:?}"),
        }
    }

    fn parse_model_arg(raw: &str) -> Source {
        let path = PathBuf::from(raw);
        if path.is_dir() {
            Source::Local(path)
        } else {
            Source::Hub {
                repo: raw.to_string(),
                revision: None,
            }
        }
    }
}
