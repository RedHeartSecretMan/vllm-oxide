//! Explicit real-GPU smoke; missing prerequisites are errors, never skips.
#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)]

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Context, Result};
use vllm_oxide::{EngineOptions, Prompt, RequestOutput, SamplingParams, Source, LLM};

#[test]
fn cuda_public_generation_contract() -> Result<()> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let commit = std::env::var("VLLM_SMOKE_COMMIT").context("VLLM_SMOKE_COMMIT is required")?;
    let tree = std::env::var("VLLM_SMOKE_TREE").context("VLLM_SMOKE_TREE is required")?;
    vllm_oxide_test::measurement::validate_deterministic_environment()?;
    vllm_oxide_test::measurement::validate_measurement_identity(&repo, &commit, &tree)?;
    vllm_oxide_test::measurement::validate_running_binary(&repo)?;
    let model =
        PathBuf::from(std::env::var_os("QWEN3_MODEL_DIR").context("QWEN3_MODEL_DIR is required")?);
    vllm_oxide_test::measurement::validate_release_model(&model)?;
    let mut llm = LLM::new(
        Source::Local(model),
        EngineOptions {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            max_model_len: 512,
            gpu_memory_utilization: 0.5,
            ..EngineOptions::default()
        },
    )?;
    // Construction performs the real prefill/decode warmup before this call.
    let params = SamplingParams {
        max_tokens: 2,
        ignore_eos: true,
        ..SamplingParams::default()
    };
    let prompt = || Prompt::Text("The capital of France is".into());
    let baseline = llm.generate(&[prompt()], std::slice::from_ref(&params))?;
    assert_eq!(baseline.len(), 1);
    assert_eq!(baseline[0].token_ids.len(), 2);
    assert!(baseline[0].finished);
    let mut ids = HashSet::from([baseline[0].request_id]);
    for _ in 0..100 {
        let output = llm.generate(&[prompt()], std::slice::from_ref(&params))?;
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].token_ids, baseline[0].token_ids);
        assert_eq!(output[0].text, baseline[0].text);
        assert!(output[0].finished);
        assert!(
            ids.insert(output[0].request_id),
            "request identity was reused"
        );
    }

    let prompts = [
        prompt(),
        Prompt::Text("The capital of Japan is".into()),
        // Frozen canonical_01 tokens at the pinned Qwen3 tokenizer revision.
        Prompt::TokenIds(vec![9707, 11, 62951, 8273, 4086, 44, 13]),
    ];
    let parameters = [
        params.clone(),
        SamplingParams {
            max_tokens: 3,
            ..params.clone()
        },
        SamplingParams {
            max_tokens: 1,
            ..params
        },
    ];
    let mut individual = Vec::<RequestOutput>::new();
    for (prompt, params) in prompts.iter().zip(&parameters) {
        let output = llm.generate(std::slice::from_ref(prompt), std::slice::from_ref(params))?;
        assert_eq!(output.len(), 1);
        assert!(ids.insert(output[0].request_id));
        individual.extend(output);
    }
    let batch = llm.generate(&prompts, &parameters)?;
    assert_eq!(batch.len(), 3);
    for ((actual, expected), params) in batch.iter().zip(&individual).zip(&parameters) {
        assert_eq!(
            actual.token_ids, expected.token_ids,
            "mixed batch changed a request"
        );
        assert_eq!(actual.text, expected.text);
        assert_eq!(actual.token_ids.len(), params.max_tokens);
        assert!(actual.finished);
        assert!(ids.insert(actual.request_id));
    }
    assert!(batch
        .windows(2)
        .all(|pair| pair[0].request_id < pair[1].request_id));
    println!(
        "PUBLIC_GPU_SMOKE {}",
        serde_json::json!({"source":{"commit":commit,"tree":tree},
            "warmup":true,"repeated_calls":100,"mixed_requests":3,
            "unique_request_ids":ids.len(),"passed":true})
    );
    Ok(())
}
