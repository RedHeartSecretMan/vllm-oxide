//! External-crate view of the supported v0.2.0 generation interface.

use vllm_oxide::{EngineOptions, Prompt, RequestOutput, SamplingParams, Source, LLM};

/// This function is compiled but deliberately not executed by CPU CI. It
/// proves that an external consumer can construct the composition root and
/// invoke generation using only the six supported root types.
#[allow(dead_code)]
fn external_consumer_can_construct_and_generate(
) -> std::result::Result<Vec<RequestOutput>, anyhow::Error> {
    let mut llm = LLM::new(
        Source::Hub {
            repo: "Qwen/Qwen3-0.6B".to_string(),
            revision: None,
        },
        EngineOptions::default(),
    )?;
    llm.generate(
        &[Prompt::Text("The meaning of life is".to_string())],
        &[SamplingParams::default()],
    )
}

#[test]
fn default_engine_options_match_the_public_contract() {
    let options = EngineOptions::default();

    assert_eq!(options.max_num_batched_tokens, 16_384);
    assert_eq!(options.max_num_seqs, 512);
    assert_eq!(options.max_model_len, 4_096);
    assert!((options.gpu_memory_utilization - 0.9).abs() < f32::EPSILON);
    assert!(options.enforce_eager);
    assert!(options.dtype.is_none());
}

#[test]
fn default_sampling_is_greedy_and_bounded() {
    let params = SamplingParams::default();

    assert_eq!(params.temperature, 0.0);
    assert_eq!(params.top_k, None);
    assert_eq!(params.top_p, None);
    assert_eq!(params.max_tokens, 16);
    assert!(!params.ignore_eos);
    assert_eq!(params.presence_penalty, 0.0);
    assert_eq!(params.frequency_penalty, 0.0);
    assert_eq!(params.repetition_penalty, 0.0);
}

#[test]
fn prompt_and_source_variants_carry_public_inputs() {
    let prompt = Prompt::TokenIds(vec![1, 2, 3]);
    let source = Source::Local("/tmp/model".into());

    assert!(matches!(prompt, Prompt::TokenIds(ids) if ids == vec![1, 2, 3]));
    assert!(matches!(source, Source::Local(path) if path.ends_with("model")));
}
