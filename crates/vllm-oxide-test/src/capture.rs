//! Consumer for the default-off, diagnostic-only internal golden capture.
//!
//! This protocol exists solely for the GPU release gate. It is not a public
//! vllm-oxide generation interface and is validated fail-closed before L1/L2.
//! The consumer owns an independent wire declaration and parser rather than
//! sharing producer DTOs, so malformed producer output is checked across a
//! real trust seam instead of being accepted by the same implementation.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::os::unix::fs::{DirBuilderExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use vllm_oxide::{Prompt, RequestOutput, SamplingParams, LLM};

const TEMP_DIR_ENV: &str = "VLLM_OXIDE_INTERNAL_GOLDEN_TEMP_DIR";
const DESTINATION_ENV: &str = "VLLM_OXIDE_INTERNAL_GOLDEN_DESTINATION";
const CALL_ID_ENV: &str = "VLLM_OXIDE_INTERNAL_GOLDEN_CALL_ID";
const FORMAT_NAME: &str = "vllm-oxide-internal-golden-jsonl-v1";
const SCHEMA_VERSION: u32 = 1;
const LOGITS_DTYPE: &str = "F32";
const DESTINATION_NAME: &str = "capture.jsonl";
static WORKSPACE_NONCE: AtomicU64 = AtomicU64::new(0);
static CAPTURE_ENV_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug)]
pub(crate) struct CapturedGeneration {
    pub(crate) logits: Vec<f32>,
    pub(crate) tensor_shape: (usize, usize),
    pub(crate) tokens_by_input: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct RequestBinding {
    input_position: usize,
    request_id: usize,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ArtifactLine {
    Header {
        format: String,
        schema_version: u32,
        call_id: String,
        requests: Vec<RequestBinding>,
    },
    Row {
        row_index: usize,
        input_position: usize,
        request_id: usize,
        selected_token: u32,
        completion_step: usize,
        dtype: String,
        row_shape: [usize; 1],
        logits: Vec<f32>,
    },
    Trailer {
        complete: bool,
        row_count: usize,
        dtype: String,
        tensor_shape: [usize; 2],
    },
}

pub(crate) fn generate_with_capture(
    llm: &mut LLM,
    prompt: Prompt,
    max_tokens: usize,
    fixture_id: &str,
) -> Result<CapturedGeneration> {
    Ok(generate_with_capture_inner(llm, prompt, max_tokens, fixture_id, None)?.0)
}

pub(crate) fn generate_with_preserved_capture(
    llm: &mut LLM,
    prompt: Prompt,
    max_tokens: usize,
    fixture_id: &str,
    destination: &Path,
) -> Result<(CapturedGeneration, String)> {
    generate_with_capture_inner(llm, prompt, max_tokens, fixture_id, Some(destination))
}

fn generate_with_capture_inner(
    llm: &mut LLM,
    prompt: Prompt,
    max_tokens: usize,
    fixture_id: &str,
    preserve: Option<&Path>,
) -> Result<(CapturedGeneration, String)> {
    let call_id = unique_call_id(fixture_id);
    let workspace = CaptureWorkspace::new()?;
    let capture_env = CaptureEnvironment::install(&workspace.path, &call_id)?;
    let outputs = llm
        .generate(
            &[prompt],
            &[SamplingParams {
                temperature: 0.0,
                max_tokens,
                ignore_eos: true,
                ..SamplingParams::default()
            }],
        )
        .context("LLM::generate failed while diagnostic capture was active")?;
    drop(capture_env);
    let artifact_path = workspace.path.join(DESTINATION_NAME);
    let captured = read_and_validate_capture(&artifact_path, &call_id, &outputs)?;
    let bytes = std::fs::read(&artifact_path).context("reading validated capture for identity")?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    if let Some(destination) = preserve {
        if destination.exists() || destination.is_symlink() {
            bail!("preserved capture destination must be fresh and non-existing");
        }
        std::fs::hard_link(&artifact_path, destination).with_context(|| {
            format!(
                "publishing validated raw capture to {}",
                destination.display()
            )
        })?;
    }
    Ok((captured, sha256))
}

fn unique_call_id(fixture_id: &str) -> String {
    let digest = Sha256::digest(fixture_id.as_bytes());
    let encoded = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let nonce = WORKSPACE_NONCE.fetch_add(1, Ordering::Relaxed);
    format!("fixture-{encoded}-{nonce}")
}

struct CaptureWorkspace {
    path: PathBuf,
}

impl CaptureWorkspace {
    fn new() -> Result<Self> {
        let root = std::env::temp_dir()
            .canonicalize()
            .context("canonicalizing process temporary directory")?;
        for _ in 0..64 {
            let nonce = WORKSPACE_NONCE.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(".vllm-oxide-golden-{}-{nonce}", std::process::id()));
            match std::fs::DirBuilder::new().mode(0o700).create(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("creating private capture workspace {}", path.display())
                    })
                }
            }
        }
        bail!("could not allocate a unique private capture workspace")
    }
}

impl Drop for CaptureWorkspace {
    fn drop(&mut self) {
        let destination = self.path.join(DESTINATION_NAME);
        let _ = std::fs::remove_file(destination);
        let _ = std::fs::remove_dir(&self.path);
    }
}

struct CaptureEnvironment {
    _guard: MutexGuard<'static, ()>,
}

impl CaptureEnvironment {
    fn install(temp_dir: &Path, call_id: &str) -> Result<Self> {
        let guard = CAPTURE_ENV_LOCK.lock().map_err(|error| {
            anyhow!("internal golden capture environment lock poisoned: {error}")
        })?;
        let occupied = [TEMP_DIR_ENV, DESTINATION_ENV, CALL_ID_ENV]
            .into_iter()
            .filter(|name| std::env::var_os(name).is_some())
            .collect::<Vec<_>>();
        if !occupied.is_empty() {
            bail!(
                "reserved internal golden capture environment is already set: {}",
                occupied.join(", ")
            );
        }
        std::env::set_var(TEMP_DIR_ENV, temp_dir);
        std::env::set_var(DESTINATION_ENV, DESTINATION_NAME);
        std::env::set_var(CALL_ID_ENV, call_id);
        Ok(Self { _guard: guard })
    }
}

impl Drop for CaptureEnvironment {
    fn drop(&mut self) {
        std::env::remove_var(TEMP_DIR_ENV);
        std::env::remove_var(DESTINATION_ENV);
        std::env::remove_var(CALL_ID_ENV);
    }
}

pub(crate) fn read_and_validate_capture(
    path: &Path,
    expected_call_id: &str,
    outputs: &[RequestOutput],
) -> Result<CapturedGeneration> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("reading diagnostic capture metadata at {}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        bail!("diagnostic capture destination is not a regular non-symlink file");
    }
    if metadata.mode() & 0o077 != 0 {
        bail!("diagnostic capture artifact is not caller-private");
    }

    let reader = BufReader::new(
        File::open(path)
            .with_context(|| format!("opening diagnostic capture {}", path.display()))?,
    );
    let mut lines = reader.lines();
    let header = lines
        .next()
        .transpose()
        .context("reading diagnostic capture header")?
        .ok_or_else(|| anyhow!("diagnostic capture artifact is empty"))?;
    let bindings = match parse_line(&header, "header")? {
        ArtifactLine::Header {
            format,
            schema_version,
            call_id,
            requests,
        } => {
            if format != FORMAT_NAME
                || schema_version != SCHEMA_VERSION
                || call_id != expected_call_id
            {
                bail!("diagnostic capture header has a stale call identity or schema");
            }
            requests
        }
        _ => bail!("diagnostic capture must begin with a header"),
    };
    validate_bindings(&bindings, outputs)?;

    let mut expected_steps = vec![0usize; bindings.len()];
    let mut tokens_by_input = vec![Vec::<u32>::new(); bindings.len()];
    let mut logits = Vec::new();
    let mut row_count = 0usize;
    let mut vocab_size = None;
    let mut last_input_position = None;
    let mut trailer = None;

    for line in lines {
        let line = line.context("reading diagnostic capture line")?;
        match parse_line(&line, "body")? {
            ArtifactLine::Row {
                row_index,
                input_position,
                request_id,
                selected_token,
                completion_step,
                dtype,
                row_shape,
                logits: row_logits,
            } => {
                if trailer.is_some() {
                    bail!("diagnostic capture has additional rows after its trailer");
                }
                if row_index != row_count {
                    bail!("diagnostic capture row indices are duplicate or non-contiguous");
                }
                if last_input_position.is_some_and(|previous| input_position < previous) {
                    bail!("diagnostic capture row order is not request-major deterministic");
                }
                last_input_position = Some(input_position);
                let binding = bindings.get(input_position).ok_or_else(|| {
                    anyhow!(
                        "diagnostic capture contains additional input position {input_position}"
                    )
                })?;
                if request_id != binding.request_id {
                    bail!("diagnostic capture row has a stale stable request identity");
                }
                if completion_step != expected_steps[input_position] {
                    bail!("diagnostic capture completion steps are duplicate or have a gap");
                }
                let expected_token = outputs[input_position]
                    .token_ids
                    .get(completion_step)
                    .ok_or_else(|| anyhow!("diagnostic capture contains an additional row"))?;
                if selected_token != *expected_token {
                    bail!("diagnostic capture selected token disagrees with RequestOutput");
                }
                if dtype != LOGITS_DTYPE
                    || row_shape[0] == 0
                    || row_shape[0] != row_logits.len()
                    || row_logits.iter().any(|value| !value.is_finite())
                {
                    bail!("diagnostic capture row has an invalid dtype, shape, or value");
                }
                match vocab_size {
                    Some(width) if width != row_shape[0] => {
                        bail!("diagnostic capture rows have inconsistent vocabulary widths")
                    }
                    None => vocab_size = Some(row_shape[0]),
                    _ => {}
                }
                logits.extend(row_logits);
                tokens_by_input[input_position].push(selected_token);
                expected_steps[input_position] += 1;
                row_count += 1;
            }
            ArtifactLine::Trailer {
                complete,
                row_count: declared_rows,
                dtype,
                tensor_shape,
            } => {
                if trailer.is_some() {
                    bail!("diagnostic capture has duplicate trailers");
                }
                trailer = Some((complete, declared_rows, dtype, tensor_shape));
            }
            ArtifactLine::Header { .. } => bail!("diagnostic capture has an additional header"),
        }
    }
    let (complete, declared_rows, dtype, declared_shape) =
        trailer.ok_or_else(|| anyhow!("diagnostic capture is missing its completion trailer"))?;
    let vocab_size = vocab_size.unwrap_or(0);
    if !complete
        || declared_rows != row_count
        || dtype != LOGITS_DTYPE
        || declared_shape != [row_count, vocab_size]
        || logits.len() != row_count.saturating_mul(vocab_size)
    {
        bail!("diagnostic capture trailer count, shape, dtype, or completion marker is invalid");
    }
    for (input_position, output) in outputs.iter().enumerate() {
        if tokens_by_input[input_position] != output.token_ids {
            bail!("diagnostic capture is missing rows at input position {input_position}");
        }
    }

    Ok(CapturedGeneration {
        logits,
        tensor_shape: (row_count, vocab_size),
        tokens_by_input,
    })
}

fn parse_line(line: &str, description: &str) -> Result<ArtifactLine> {
    serde_json::from_str(line)
        .with_context(|| format!("parsing diagnostic capture {description} JSON line"))
}

fn validate_bindings(bindings: &[RequestBinding], outputs: &[RequestOutput]) -> Result<()> {
    if bindings.len() != outputs.len() {
        bail!("diagnostic capture request count does not match RequestOutput count");
    }
    let mut seen = HashSet::new();
    for (input_position, (binding, output)) in bindings.iter().zip(outputs).enumerate() {
        if binding.input_position != input_position
            || binding.request_id != output.request_id
            || !output.finished
            || !seen.insert(binding.request_id)
        {
            bail!("diagnostic capture request bindings are duplicate, stale, or unordered");
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;

    fn private_artifact(contents: &str) -> (tempfile::TempDir, PathBuf) {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

        let temp = tempfile::tempdir().unwrap();
        std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = temp.path().join("capture.jsonl");
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&path)
            .unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        (temp, path)
    }

    #[test]
    fn consumer_accepts_complete_bound_artifact() {
        let (_temp, path) = private_artifact(
            concat!(
                "{\"kind\":\"header\",\"format\":\"vllm-oxide-internal-golden-jsonl-v1\",\"schema_version\":1,\"call_id\":\"call-1\",\"requests\":[{\"input_position\":0,\"request_id\":7}]}\n",
                "{\"kind\":\"row\",\"row_index\":0,\"input_position\":0,\"request_id\":7,\"selected_token\":2,\"completion_step\":0,\"dtype\":\"F32\",\"row_shape\":[3],\"logits\":[0.0,1.0,2.0]}\n",
                "{\"kind\":\"trailer\",\"complete\":true,\"row_count\":1,\"dtype\":\"F32\",\"tensor_shape\":[1,3]}\n"
            ),
        );
        let outputs = [RequestOutput {
            request_id: 7,
            token_ids: vec![2],
            text: String::new(),
            finished: true,
        }];

        let capture = read_and_validate_capture(&path, "call-1", &outputs).unwrap();

        assert_eq!(capture.tensor_shape, (1, 3));
        assert_eq!(capture.logits, vec![0.0, 1.0, 2.0]);
        assert_eq!(capture.tokens_by_input, vec![vec![2]]);
    }

    #[test]
    fn long_fixture_id_maps_to_a_bounded_unique_call_identity() {
        let first = unique_call_id(&"x".repeat(10_000));
        let second = unique_call_id(&"x".repeat(10_000));

        assert!(first.len() <= 128);
        assert!(first
            .bytes()
            .all(|byte| { byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') }));
        assert_ne!(first, second);
    }

    #[test]
    fn consumer_rejects_completion_step_gap_before_comparison() {
        let (_temp, path) = private_artifact(
            concat!(
                "{\"kind\":\"header\",\"format\":\"vllm-oxide-internal-golden-jsonl-v1\",\"schema_version\":1,\"call_id\":\"call-1\",\"requests\":[{\"input_position\":0,\"request_id\":7}]}\n",
                "{\"kind\":\"row\",\"row_index\":0,\"input_position\":0,\"request_id\":7,\"selected_token\":2,\"completion_step\":1,\"dtype\":\"F32\",\"row_shape\":[3],\"logits\":[0.0,1.0,2.0]}\n",
                "{\"kind\":\"trailer\",\"complete\":true,\"row_count\":1,\"dtype\":\"F32\",\"tensor_shape\":[1,3]}\n"
            ),
        );
        let outputs = [RequestOutput {
            request_id: 7,
            token_ids: vec![2],
            text: String::new(),
            finished: true,
        }];

        let error = read_and_validate_capture(&path, "call-1", &outputs).unwrap_err();

        assert!(error.to_string().contains("duplicate or have a gap"));
    }

    #[test]
    fn consumer_rejects_duplicate_missing_additional_stale_and_shape_mismatched_rows() {
        let outputs = [RequestOutput {
            request_id: 7,
            token_ids: vec![2],
            text: String::new(),
            finished: true,
        }];
        let header = "{\"kind\":\"header\",\"format\":\"vllm-oxide-internal-golden-jsonl-v1\",\"schema_version\":1,\"call_id\":\"call-1\",\"requests\":[{\"input_position\":0,\"request_id\":7}]}\n";
        let valid_row = "{\"kind\":\"row\",\"row_index\":0,\"input_position\":0,\"request_id\":7,\"selected_token\":2,\"completion_step\":0,\"dtype\":\"F32\",\"row_shape\":[3],\"logits\":[0.0,1.0,2.0]}\n";
        let valid_trailer = "{\"kind\":\"trailer\",\"complete\":true,\"row_count\":1,\"dtype\":\"F32\",\"tensor_shape\":[1,3]}\n";
        let cases = [
            (
                "duplicate row index",
                format!(
                    "{header}{valid_row}{{\"kind\":\"row\",\"row_index\":0,\"input_position\":0,\"request_id\":7,\"selected_token\":2,\"completion_step\":1,\"dtype\":\"F32\",\"row_shape\":[3],\"logits\":[0.0,1.0,2.0]}}\n{{\"kind\":\"trailer\",\"complete\":true,\"row_count\":2,\"dtype\":\"F32\",\"tensor_shape\":[2,3]}}\n"
                ),
                "row indices",
            ),
            (
                "missing row",
                format!(
                    "{header}{{\"kind\":\"trailer\",\"complete\":true,\"row_count\":0,\"dtype\":\"F32\",\"tensor_shape\":[0,0]}}\n"
                ),
                "missing rows",
            ),
            (
                "additional row",
                format!(
                    "{header}{valid_row}{{\"kind\":\"row\",\"row_index\":1,\"input_position\":0,\"request_id\":7,\"selected_token\":2,\"completion_step\":1,\"dtype\":\"F32\",\"row_shape\":[3],\"logits\":[0.0,1.0,2.0]}}\n{{\"kind\":\"trailer\",\"complete\":true,\"row_count\":2,\"dtype\":\"F32\",\"tensor_shape\":[2,3]}}\n"
                ),
                "additional row",
            ),
            (
                "stale request",
                format!(
                    "{header}{{\"kind\":\"row\",\"row_index\":0,\"input_position\":0,\"request_id\":8,\"selected_token\":2,\"completion_step\":0,\"dtype\":\"F32\",\"row_shape\":[3],\"logits\":[0.0,1.0,2.0]}}\n{valid_trailer}"
                ),
                "stale stable request identity",
            ),
            (
                "shape mismatch",
                format!(
                    "{header}{{\"kind\":\"row\",\"row_index\":0,\"input_position\":0,\"request_id\":7,\"selected_token\":2,\"completion_step\":0,\"dtype\":\"F32\",\"row_shape\":[4],\"logits\":[0.0,1.0,2.0]}}\n{valid_trailer}"
                ),
                "invalid dtype, shape, or value",
            ),
        ];

        for (name, artifact, expected_error) in cases {
            let (_temp, path) = private_artifact(&artifact);
            let error = read_and_validate_capture(&path, "call-1", &outputs).unwrap_err();
            assert!(
                format!("{error:#}").contains(expected_error),
                "{name}: {error:#}"
            );
        }
    }

    #[test]
    fn consumer_rejects_stale_call_identity_and_noncanonical_request_order() {
        let stale = concat!(
            "{\"kind\":\"header\",\"format\":\"vllm-oxide-internal-golden-jsonl-v1\",\"schema_version\":1,\"call_id\":\"old-call\",\"requests\":[{\"input_position\":0,\"request_id\":7}]}\n",
            "{\"kind\":\"row\",\"row_index\":0,\"input_position\":0,\"request_id\":7,\"selected_token\":2,\"completion_step\":0,\"dtype\":\"F32\",\"row_shape\":[3],\"logits\":[0.0,1.0,2.0]}\n",
            "{\"kind\":\"trailer\",\"complete\":true,\"row_count\":1,\"dtype\":\"F32\",\"tensor_shape\":[1,3]}\n"
        );
        let outputs = [RequestOutput {
            request_id: 7,
            token_ids: vec![2],
            text: String::new(),
            finished: true,
        }];
        let (_temp, path) = private_artifact(stale);
        let error = read_and_validate_capture(&path, "call-1", &outputs).unwrap_err();
        assert!(error.to_string().contains("stale call identity"));

        let unordered = concat!(
            "{\"kind\":\"header\",\"format\":\"vllm-oxide-internal-golden-jsonl-v1\",\"schema_version\":1,\"call_id\":\"call-1\",\"requests\":[{\"input_position\":0,\"request_id\":7},{\"input_position\":1,\"request_id\":8}]}\n",
            "{\"kind\":\"row\",\"row_index\":0,\"input_position\":1,\"request_id\":8,\"selected_token\":3,\"completion_step\":0,\"dtype\":\"F32\",\"row_shape\":[3],\"logits\":[0.0,1.0,2.0]}\n",
            "{\"kind\":\"row\",\"row_index\":1,\"input_position\":0,\"request_id\":7,\"selected_token\":2,\"completion_step\":0,\"dtype\":\"F32\",\"row_shape\":[3],\"logits\":[0.0,1.0,2.0]}\n",
            "{\"kind\":\"trailer\",\"complete\":true,\"row_count\":2,\"dtype\":\"F32\",\"tensor_shape\":[2,3]}\n"
        );
        let outputs = [
            RequestOutput {
                request_id: 7,
                token_ids: vec![2],
                text: String::new(),
                finished: true,
            },
            RequestOutput {
                request_id: 8,
                token_ids: vec![3],
                text: String::new(),
                finished: true,
            },
        ];
        let (_temp, path) = private_artifact(unordered);
        let error = read_and_validate_capture(&path, "call-1", &outputs).unwrap_err();
        assert!(error.to_string().contains("request-major deterministic"));
    }
}
