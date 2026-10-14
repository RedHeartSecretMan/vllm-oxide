use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use vllm_oxide::{Prompt, SamplingParams, Source, LLM};

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
    #[arg(long, default_value = "0", value_parser = parse_temperature)]
    temperature: f32,

    /// Top-k sampling: keep only the k highest-logit tokens.
    #[arg(long, value_parser = parse_positive_count)]
    top_k: Option<usize>,

    /// Top-p (nucleus) sampling: keep smallest token set with cumulative prob ≥ p.
    #[arg(long, value_parser = parse_top_p)]
    top_p: Option<f32>,

    /// Maximum tokens to generate. Defaults to 16.
    #[arg(long, default_value = "16", value_parser = parse_positive_count)]
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

    let source = parse_model_arg(&cli.model);

    let mut llm = LLM::new(source, vllm_oxide::EngineOptions::default())
        .context("failed to initialise LLM")?;

    let sampling_params = SamplingParams {
        temperature: cli.temperature,
        top_k: cli.top_k,
        top_p: cli.top_p,
        max_tokens: cli.max_tokens,
        ..SamplingParams::default()
    };

    let outputs = llm
        .generate(&[Prompt::Text(prompt_text)], &[sampling_params])
        .context("LLM::generate failed")?;

    if let Some(output) = outputs.first() {
        print!("{}", output.text);
    }

    Ok(())
}

// Reject malformed CLI values during argv parsing, before stdin, model I/O or CUDA setup.
fn parse_temperature(raw: &str) -> std::result::Result<f32, String> {
    let value = raw.parse::<f32>().map_err(|error| error.to_string())?;
    if value.is_nan() || value < 0.0 {
        return Err("temperature must be nonnegative and not NaN".to_string());
    }
    Ok(value)
}

fn parse_top_p(raw: &str) -> std::result::Result<f32, String> {
    let value = raw.parse::<f32>().map_err(|error| error.to_string())?;
    if !(value.is_finite() && 0.0 < value && value <= 1.0) {
        return Err("top_p must be finite and in (0, 1]".to_string());
    }
    Ok(value)
}

fn parse_positive_count(raw: &str) -> std::result::Result<usize, String> {
    let value = raw.parse::<usize>().map_err(|error| error.to_string())?;
    if value == 0 {
        return Err("token count must be at least 1".to_string());
    }
    Ok(value)
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn invalid_sampling_values_fail_at_argument_parsing() {
        for argument in [
            "--temperature=-1",
            "--temperature=NaN",
            "--temperature=-inf",
            "--top-k=0",
            "--top-p=0",
            "--top-p=1.1",
            "--top-p=NaN",
            "--top-p=inf",
            "--max-tokens=0",
        ] {
            let error = Cli::try_parse_from(["vllm-oxide", "--model", "missing-model", argument])
                .err()
                .unwrap();
            assert_eq!(
                error.kind(),
                clap::error::ErrorKind::ValueValidation,
                "unexpected parser result for {argument}: {error}"
            );
        }
    }

    #[test]
    fn supported_sampling_boundaries_parse_without_a_model() {
        for arguments in [
            vec![
                "--temperature=0",
                "--top-p=1",
                "--top-k=1",
                "--max-tokens=1",
            ],
            vec!["--temperature=inf"],
            vec!["--temperature=1e-45", "--top-p=1e-45"],
        ] {
            let mut command = vec!["vllm-oxide", "--model", "missing-model"];
            command.extend(arguments);
            Cli::try_parse_from(command).unwrap();
        }
    }

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
}
