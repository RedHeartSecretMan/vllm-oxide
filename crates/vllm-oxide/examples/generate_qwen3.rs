//! Offline Qwen3 generation through the supported v0.2.0 interface.
//!
//! Usage:
//! `cargo run --release --example generate_qwen3 --features cuda -- hub:Qwen/Qwen3-0.6B`

use std::path::PathBuf;
use std::process::ExitCode;

use vllm_oxide::{EngineOptions, Prompt, SamplingParams, Source, LLM};

fn parse_source(argument: &str) -> Source {
    if let Some(hub) = argument.strip_prefix("hub:") {
        let (repo, revision) = hub
            .split_once('@')
            .map_or((hub, None), |(repo, revision)| (repo, Some(revision)));
        Source::Hub {
            repo: repo.to_string(),
            revision: revision.map(str::to_string),
        }
    } else {
        Source::Local(PathBuf::from(argument))
    }
}

fn main() -> ExitCode {
    let argument = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "hub:Qwen/Qwen3-0.6B".to_string());
    let mut llm = match LLM::new(parse_source(&argument), EngineOptions::default()) {
        Ok(llm) => llm,
        Err(error) => {
            eprintln!("failed to initialize Qwen3: {error:#}");
            return ExitCode::FAILURE;
        }
    };
    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: 16,
        ..SamplingParams::default()
    };
    match llm.generate(
        &[Prompt::Text("The meaning of life is".to_string())],
        &[params],
    ) {
        Ok(outputs) => {
            if let Some(output) = outputs.first() {
                println!("{}", output.text);
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("generation failed: {error:#}");
            ExitCode::FAILURE
        }
    }
}
