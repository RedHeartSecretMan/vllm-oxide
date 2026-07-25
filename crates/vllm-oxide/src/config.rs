//! `Source` enum, dtype resolution, and the `HF_HUB_OFFLINE` shim (ADR-0002).
//!
//! Three concerns live here:
//!
//! 1. [`Source`] — the loader's input: a local directory or a HuggingFace Hub
//!    repo id. Tag carries the data the loader needs to resolve shards.
//! 2. [`is_hf_hub_offline`] — verbatim port of mistral.rs's
//!    `mistralrs-core/src/pipeline/hf.rs:28-37` shim. hf-hub 0.5 lacks a
//!    native `HF_HUB_OFFLINE` (zero source matches at that rev); we mirror
//!    mistral.rs's accepted values so the behaviour is identical.
//! 3. [`default_dtype`] — reads `config.json`'s `torch_dtype` field. This is
//!    the fix for nano-vllm's latent bug (`model_runner.py:29` reads the
//!    nonexistent `hf_config.dtype` attribute). User-overridable via the
//!    `dtype` parameter on [`crate::loader::load_weights`].

use candle_core::DType;
use serde::Deserialize;

/// Env-var name consulted by [`is_hf_hub_offline`]. Mirrors `HF_HUB_OFFLINE`
/// in mistral.rs and the upstream Python `huggingface_hub` convention.
pub const HF_HUB_OFFLINE_ENV: &str = "HF_HUB_OFFLINE";

/// Where to load weights from.
///
/// The loader resolves either variant to a list of `*.safetensors` paths,
/// then mmaps them via candle's `ShardedSafeTensors::var_builder`. See
/// [`crate::loader::load_weights`].
///
/// `Local` is the parity anchor (golden comparison runs against a local
/// snapshot); `Hub` is for ad-hoc / interactive use. `Hub` honours
/// [`is_hf_hub_offline`] and short-circuits to the local cache when offline.
#[derive(Debug, Clone)]
pub enum Source {
    /// Absolute or relative path to a directory containing one of:
    /// `model.safetensors.index.json` (multi-shard), a single
    /// `model.safetensors`, or any `*.safetensors` files (bare layout).
    Local(std::path::PathBuf),

    /// HuggingFace Hub repo id (`"Qwen/Qwen3-0.6B"`) plus an optional
    /// revision (`"main"`, a commit SHA, a tag). `None` resolves to the
    /// Hub's default branch.
    Hub {
        repo: String,
        revision: Option<String>,
    },
}

/// True iff the caller has requested fully-offline Hub operation.
///
/// Accepted truthy values (case-insensitive, after `trim()`): `1`, `true`,
/// `yes`, `on`. Anything else — including unset — is online. Mirrors
/// mistral.rs's `is_hf_hub_offline()` verbatim (`mistralrs-core/src/pipeline/
/// hf.rs:28-37`); hf-hub 0.5 has no native equivalent.
pub fn is_hf_hub_offline() -> bool {
    is_offline_value(std::env::var(HF_HUB_OFFLINE_ENV).ok().as_deref())
}

/// Pure form of [`is_hf_hub_offline`] over an explicit env value. Exported so
/// tests can pin the parsing without mutating process-global env state
/// (which would race sibling test threads).
pub fn is_offline_value(raw: Option<&str>) -> bool {
    match raw.map(|v| v.trim().to_ascii_lowercase()) {
        Some(ref v) => matches!(v.as_str(), "1" | "true" | "yes" | "on"),
        None => false,
    }
}

/// HF `config.json` fields the loader cares about. Only `torch_dtype` is
/// read here; downstream tickets (T6) define their own architecture-specific
/// config structs and read additional fields. `#[serde(default)]` lets us
/// parse partial configs without `deny_unknown_fields`.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct HFConfig {
    /// `"bfloat16"`, `"float16"`, `"float32"`, ... — as written by
    /// `transformers`. The field name is `torch_dtype`, NOT `dtype`
    /// (nano-vllm reads `dtype` and silently misses this — bug fixed here).
    #[serde(default)]
    pub torch_dtype: Option<String>,
}

/// Resolve the checkpoint's default [`DType`] from a parsed [`HFConfig`].
///
/// Returns:
/// - `Ok(BF16)` for `"bfloat16"`
/// - `Ok(F16)` for `"float16"`
/// - `Ok(F32)` for `"float32"`
/// - `Err` if `torch_dtype` is missing or unrecognised — v0.1 refuses to
///   guess. The caller decides whether to surface the error or supply a
///   user override.
pub fn default_dtype(config: &HFConfig) -> anyhow::Result<DType> {
    let raw = config
        .torch_dtype
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("config.json has no `torch_dtype` field"))?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "bfloat16" | "bf16" => Ok(DType::BF16),
        "float16" | "fp16" | "half" => Ok(DType::F16),
        "float32" | "fp32" | "float" => Ok(DType::F32),
        "float64" | "fp64" | "double" => Ok(DType::F64),
        other => Err(anyhow::anyhow!(
            "config.json `torch_dtype={other}` is not a recognised v0.1 dtype \
             (supported: bfloat16, float16, float32, float64)"
        )),
    }
}

/// Convenience: parse `config.json` bytes and resolve the dtype in one call.
/// Used by `LLM::new` (T8) when the user has not supplied an explicit `dtype`.
pub fn default_dtype_from_config_json(config_json: &[u8]) -> anyhow::Result<DType> {
    let config: HFConfig = serde_json::from_slice(config_json)?;
    default_dtype(&config)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;

    mod is_offline_value {
        use super::*;

        #[test]
        fn unset_is_online() {
            assert!(!is_offline_value(None));
        }

        #[test]
        fn empty_is_online() {
            assert!(!is_offline_value(Some("")));
        }

        #[test]
        fn truthy_canonical_forms() {
            for v in ["1", "true", "yes", "on"] {
                assert!(is_offline_value(Some(v)), "{v:?} should be offline");
            }
        }

        #[test]
        fn truthy_case_insensitive() {
            for v in ["TRUE", "Yes", "On", "1"] {
                assert!(is_offline_value(Some(v)), "{v:?} should be offline");
            }
        }

        #[test]
        fn truthy_with_whitespace() {
            for v in ["  true  ", "\ton\n", " yes "] {
                assert!(is_offline_value(Some(v)), "{v:?} should be offline");
            }
        }

        #[test]
        fn non_truthy_values_stay_online() {
            for v in ["0", "false", "no", "off", "", "maybe", "y", "2"] {
                assert!(!is_offline_value(Some(v)), "{v:?} should be online");
            }
        }
    }

    mod default_dtype {
        use super::*;

        fn cfg(dtype: Option<&str>) -> HFConfig {
            HFConfig {
                torch_dtype: dtype.map(String::from),
            }
        }

        #[test]
        fn bfloat16_canonical_and_alias() {
            for s in ["bfloat16", "BF16", "  bf16  "] {
                assert_eq!(default_dtype(&cfg(Some(s))).unwrap(), DType::BF16);
            }
        }

        #[test]
        fn float16_canonical_and_aliases() {
            for s in ["float16", "FP16", "half", "  Float16 "] {
                assert_eq!(default_dtype(&cfg(Some(s))).unwrap(), DType::F16);
            }
        }

        #[test]
        fn float32_canonical_and_aliases() {
            for s in ["float32", "FP32", "float"] {
                assert_eq!(default_dtype(&cfg(Some(s))).unwrap(), DType::F32);
            }
        }

        #[test]
        fn float64_aliases() {
            for s in ["float64", "double"] {
                assert_eq!(default_dtype(&cfg(Some(s))).unwrap(), DType::F64);
            }
        }

        #[test]
        fn missing_torch_dtype_errors() {
            assert!(default_dtype(&cfg(None)).is_err());
        }

        #[test]
        fn unknown_dtype_errors() {
            assert!(default_dtype(&cfg(Some("int8"))).is_err());
            assert!(default_dtype(&cfg(Some("uf16"))).is_err());
            assert!(default_dtype(&cfg(Some(""))).is_err());
        }
    }

    mod default_dtype_from_config_json {
        use super::*;

        #[test]
        fn parses_full_qwen3_style_config() {
            // Truncated but representative: torch_dtype sits among many other
            // fields the loader must ignore.
            let json = br#"{
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "torch_dtype": "bfloat16",
                "use_cache": true
            }"#;
            assert_eq!(default_dtype_from_config_json(json).unwrap(), DType::BF16);
        }

        #[test]
        fn missing_torch_dtype_errors() {
            let json = br#"{"architectures":["X"],"hidden_size":1}"#;
            assert!(default_dtype_from_config_json(json).is_err());
        }

        #[test]
        fn malformed_json_errors() {
            assert!(default_dtype_from_config_json(b"{not json").is_err());
        }

        #[test]
        fn empty_input_errors() {
            assert!(default_dtype_from_config_json(b"").is_err());
        }
    }

    mod hf_config_serde {
        use super::*;

        #[test]
        fn ignores_unknown_fields() {
            // T6 will extend with hidden_size, num_attention_heads, etc. The
            // loader must not break when those fields appear.
            let json = br#"{"torch_dtype":"float16","hidden_size":4096}"#;
            let cfg: HFConfig = serde_json::from_slice(json).unwrap();
            assert_eq!(cfg.torch_dtype.as_deref(), Some("float16"));
        }

        #[test]
        fn empty_object_yields_none() {
            let cfg: HFConfig = serde_json::from_slice(b"{}").unwrap();
            assert!(cfg.torch_dtype.is_none());
        }
    }

    mod source_enum {
        use super::*;
        use std::path::PathBuf;

        #[test]
        fn local_carries_path() {
            match Source::Local(PathBuf::from("/tmp/x")) {
                Source::Local(p) => assert_eq!(p, PathBuf::from("/tmp/x")),
                _ => panic!("wrong variant"),
            }
        }

        #[test]
        fn hub_carries_repo_and_optional_revision() {
            let s = Source::Hub {
                repo: String::from("Qwen/Qwen3-0.6B"),
                revision: None,
            };
            match s {
                Source::Hub { repo, revision } => {
                    assert_eq!(repo, "Qwen/Qwen3-0.6B");
                    assert!(revision.is_none());
                }
                _ => panic!("wrong variant"),
            }
        }
    }
}
