use vllm_oxide::{
    default_dtype_from_config_json, is_offline_value, kv_cache_layout_shape, round_up,
    EngineOptions, Prompt, SamplingParams, Sequence, SequenceGroup, Source,
};

fn greedy_params() -> SamplingParams {
    SamplingParams {
        max_tokens: 64,
        ..SamplingParams::default()
    }
}

#[test]
fn default_sampling_params_are_deterministic() {
    let sp = SamplingParams::default();
    assert!(sp.temperature == 0.0);
    assert!(!sp.ignore_eos);
    assert_eq!(sp.top_k, None);
    assert_eq!(sp.top_p, None);
    assert!(sp.presence_penalty == 0.0);
    assert!(sp.frequency_penalty == 0.0);
    assert!(sp.repetition_penalty == 0.0);
}

#[test]
fn temperature_zero_and_top_k_one_both_greedy() {
    let sp_zero = SamplingParams {
        temperature: 0.0,
        top_k: None,
        ..SamplingParams::default()
    };
    let sp_topk1 = SamplingParams {
        temperature: 1.0,
        top_k: Some(1),
        ..SamplingParams::default()
    };
    assert!(sp_zero.temperature == 0.0);
    assert_eq!(sp_topk1.top_k, Some(1));
}

#[test]
fn default_engine_options_match_spec() {
    let opts = EngineOptions::default();
    assert_eq!(opts.max_num_batched_tokens, 16384);
    assert_eq!(opts.max_num_seqs, 512);
    assert!((opts.gpu_memory_utilization - 0.9).abs() < f32::EPSILON);
    assert!(opts.enforce_eager);
}

#[test]
fn new_sequence_is_not_finished() {
    let seq = Sequence::new(0, vec![1, 2, 3], &greedy_params());
    assert!(!seq.is_finished());
}

#[test]
fn sequence_num_completion_tokens_after_append() {
    let mut seq = Sequence::new(0, vec![10, 20, 30], &greedy_params());
    assert_eq!(seq.num_completion_tokens(), 0);
    seq.append_token(42);
    assert_eq!(seq.num_completion_tokens(), 1);
    seq.append_token(99);
    assert_eq!(seq.num_completion_tokens(), 2);
}

#[test]
fn sequence_prompt_and_completion_partitioning() {
    let mut seq = Sequence::new(0, vec![100, 200, 300], &greedy_params());
    assert_eq!(seq.prompt_token_ids(), &[100, 200, 300]);
    assert!(seq.completion_token_ids().is_empty());
    seq.append_token(400);
    seq.append_token(500);
    assert_eq!(seq.prompt_token_ids(), &[100, 200, 300]);
    assert_eq!(seq.completion_token_ids(), &[400, 500]);
}

#[test]
fn sequence_num_blocks_rounds_up() {
    let tokens: Vec<u32> = (0..257).collect();
    let seq = Sequence::new(0, tokens, &greedy_params());
    assert_eq!(seq.num_blocks(), 2);
}

#[test]
fn sequence_block_slicing() {
    let tokens: Vec<u32> = (0..300).collect();
    let seq = Sequence::new(0, tokens, &greedy_params());
    assert_eq!(seq.block(0).len(), 256);
    assert_eq!(seq.block(1).len(), 44);
}

#[test]
fn sequence_group_request_id() {
    let seq = Sequence::new(0, vec![1], &greedy_params());
    let group = SequenceGroup::new(42, seq);
    assert_eq!(group.request_id(), 42);
}

#[test]
fn sequence_group_delegates_is_finished() {
    let seq = Sequence::new(0, vec![1], &greedy_params());
    let group = SequenceGroup::new(0, seq);
    assert!(!group.is_finished());
}

#[test]
fn prompt_text_and_token_ids_are_distinct() {
    let text = Prompt::Text("hello".to_string());
    let tokens = Prompt::TokenIds(vec![1, 2, 3]);
    assert!(matches!(text, Prompt::Text(_)));
    assert!(matches!(tokens, Prompt::TokenIds(_)));
}

#[test]
fn hf_hub_offline_truthy() {
    for v in ["1", "true", "yes", "on", " 1 ", "TRUE", "YES"] {
        assert!(is_offline_value(Some(v)), "{v} must be truthy");
    }
}

#[test]
fn hf_hub_offline_falsy() {
    for v in ["0", "false", "no", "off", "", "garbage"] {
        assert!(!is_offline_value(Some(v)), "{v} must be falsy");
    }
}

#[test]
fn hf_hub_offline_unset() {
    assert!(!is_offline_value(None));
}

#[test]
// The panic is the deliberate wildcard arm of an exhaustive Source match —
// unreachable unless the variant is mis-constructed.
#[allow(clippy::panic)]
fn source_local_carries_path() {
    match Source::Local("/tmp/model".into()) {
        Source::Local(p) => assert_eq!(p, std::path::PathBuf::from("/tmp/model")),
        _ => panic!("expected Local"),
    }
}

#[test]
// The panic is the deliberate wildcard arm of an exhaustive Source match —
// unreachable unless the variant is mis-constructed.
#[allow(clippy::panic)]
fn source_hub_carries_repo() {
    match (Source::Hub {
        repo: "Qwen/Qwen3-0.6B".into(),
        revision: None,
    }) {
        Source::Hub { repo, .. } => assert_eq!(repo, "Qwen/Qwen3-0.6B"),
        _ => panic!("expected Hub"),
    }
}

#[test]
// The input JSON is a static, well-formed fixture; unwrap asserts the parser
// accepts "bfloat16".
#[allow(clippy::unwrap_used)]
fn default_dtype_parses_bfloat16() {
    let dtype =
        default_dtype_from_config_json(r#"{"torch_dtype": "bfloat16"}"#.as_bytes()).unwrap();
    assert_eq!(format!("{dtype:?}").to_lowercase(), "bf16");
}

#[test]
// The input JSON is a static, well-formed fixture; unwrap asserts the parser
// accepts "float16".
#[allow(clippy::unwrap_used)]
fn default_dtype_parses_float16() {
    let dtype = default_dtype_from_config_json(r#"{"torch_dtype": "float16"}"#.as_bytes()).unwrap();
    assert_eq!(format!("{dtype:?}").to_lowercase(), "f16");
}

#[test]
fn round_up_to_block_size() {
    assert_eq!(round_up(0, 256), 0);
    assert_eq!(round_up(1, 256), 256);
    assert_eq!(round_up(255, 256), 256);
    assert_eq!(round_up(256, 256), 256);
    assert_eq!(round_up(257, 256), 512);
}

#[test]
fn kv_cache_layout_shape_dimensions() {
    let shape = kv_cache_layout_shape(28, 10_000, 256, 8, 128);
    assert_eq!(shape[0], 2);
    assert_eq!(shape[1], 28);
    assert_eq!(shape[2], 10_000);
    assert_eq!(shape[3], 256);
    assert_eq!(shape[4], 8);
    assert_eq!(shape[5], 128);
}
