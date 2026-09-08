//! Private, feature-gated raw-logit capture for the workspace release gate.
//!
//! This module is intentionally absent from the public Rust namespace. The
//! process-level protocol is unsupported diagnostic tooling defined by
//! ADR-0011, not a second generation interface.
//! Its wire structs and validator intentionally remain independent from the
//! consumer implementation so producer defects cannot become shared-decoder
//! common-mode acceptance failures.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

pub(crate) const TEMP_DIR_ENV: &str = "VLLM_OXIDE_INTERNAL_GOLDEN_TEMP_DIR";
pub(crate) const DESTINATION_ENV: &str = "VLLM_OXIDE_INTERNAL_GOLDEN_DESTINATION";
pub(crate) const CALL_ID_ENV: &str = "VLLM_OXIDE_INTERNAL_GOLDEN_CALL_ID";
pub(crate) mod behavior;
mod benchmark;
pub(crate) mod fixed_prefix;
pub(crate) mod layer_trace;
pub(crate) mod operators;
pub(crate) use benchmark::BenchmarkSession;
#[cfg(test)]
use benchmark::{BenchmarkConfig, BENCHMARK_DESTINATION_ENV};

/// Publish a small private JSON artifact without replacing any existing bytes.
pub(crate) fn write_atomic_json(destination: &Path, value: &impl Serialize) -> Result<()> {
    let parent =
        validate_private_temp_dir(destination.parent().context("artifact parent missing")?)?;
    let name = destination
        .file_name()
        .context("artifact filename missing")?;
    validate_destination_name(name)?;
    let destination = parent.join(name);
    let stage = destination.with_extension("partial");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&stage)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    std::fs::hard_link(&stage, &destination)?;
    std::fs::remove_file(&stage)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

#[derive(Debug)]
pub(crate) struct CaptureConfig {
    temp_dir: PathBuf,
    destination_name: OsString,
    call_id: String,
}

impl CaptureConfig {
    pub(crate) fn from_env() -> Result<Option<Self>> {
        Self::from_values(
            std::env::var_os(TEMP_DIR_ENV),
            std::env::var_os(DESTINATION_ENV),
            std::env::var_os(CALL_ID_ENV),
        )
    }

    fn from_values(
        temp_dir: Option<OsString>,
        destination: Option<OsString>,
        call_id: Option<OsString>,
    ) -> Result<Option<Self>> {
        if temp_dir.is_none() && destination.is_none() && call_id.is_none() {
            return Ok(None);
        }
        let missing = [
            (TEMP_DIR_ENV, temp_dir.is_none()),
            (DESTINATION_ENV, destination.is_none()),
            (CALL_ID_ENV, call_id.is_none()),
        ]
        .into_iter()
        .filter_map(|(name, is_missing)| is_missing.then_some(name))
        .collect::<Vec<_>>();
        if !missing.is_empty() {
            bail!(
                "internal golden capture configuration is incomplete; missing {}",
                missing.join(", ")
            );
        }

        let temp_dir =
            PathBuf::from(temp_dir.ok_or_else(|| anyhow!("missing capture temporary directory"))?);
        let destination_name = destination.ok_or_else(|| anyhow!("missing capture destination"))?;
        let call_id = call_id
            .ok_or_else(|| anyhow!("missing capture call identity"))?
            .into_string()
            .map_err(|_| anyhow!("{CALL_ID_ENV} must be valid UTF-8"))?;

        validate_call_id(&call_id)?;
        validate_destination_name(&destination_name)?;
        let temp_dir = validate_private_temp_dir(&temp_dir)?;
        let destination_path = temp_dir.join(&destination_name);
        match std::fs::symlink_metadata(&destination_path) {
            Ok(_) => bail!(
                "{} must name a fresh non-existing destination; {} already exists",
                DESTINATION_ENV,
                destination_path.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "checking capture destination {}",
                        destination_path.display()
                    )
                })
            }
        }

        Ok(Some(Self {
            temp_dir,
            destination_name,
            call_id,
        }))
    }
}

const FORMAT_NAME: &str = "vllm-oxide-internal-golden-jsonl-v1";
const SCHEMA_VERSION: u32 = 1;
const LOGITS_DTYPE: &str = "F32";
const STAGING_PREFIX: &[u8] = b".vllm-oxide-";
static STAGING_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RequestBinding {
    input_position: usize,
    request_id: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct SpoolRow {
    request_id: usize,
    selected_token: u32,
    completion_step: usize,
    logits: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug)]
pub(crate) struct HostCaptureRow {
    pub(crate) request_id: usize,
    pub(crate) selected_token: u32,
    pub(crate) completion_step: usize,
    pub(crate) logits: Vec<f32>,
}

struct RequestSpool {
    binding: RequestBinding,
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    next_completion_step: usize,
}

pub(crate) struct CaptureSession {
    config: CaptureConfig,
    final_stage_name: OsString,
    final_stage_path: Option<PathBuf>,
    final_stage_file: Option<File>,
    spools: Vec<RequestSpool>,
    request_to_spool: HashMap<usize, usize>,
    vocab_size: Option<usize>,
    failed: bool,
    published: bool,
}

impl CaptureSession {
    pub(crate) fn from_env() -> Result<Option<Self>> {
        CaptureConfig::from_env()?.map(Self::prepare).transpose()
    }

    fn prepare(config: CaptureConfig) -> Result<Self> {
        let (final_stage_name, final_stage_path, final_stage_file) = create_staging_file(
            &config.temp_dir,
            &config.call_id,
            "artifact",
            &config.destination_name,
        )?;
        Ok(Self {
            config,
            final_stage_name,
            final_stage_path: Some(final_stage_path),
            final_stage_file: Some(final_stage_file),
            spools: Vec::new(),
            request_to_spool: HashMap::new(),
            vocab_size: None,
            failed: false,
            published: false,
        })
    }

    pub(crate) fn bind_requests(&mut self, request_ids: &[usize]) -> Result<()> {
        if !self.spools.is_empty() || !self.request_to_spool.is_empty() {
            bail!("internal golden capture requests are already bound");
        }
        let unique = request_ids.iter().copied().collect::<HashSet<_>>();
        if unique.len() != request_ids.len() {
            bail!("internal golden capture received duplicate stable request ids");
        }

        for (input_position, &request_id) in request_ids.iter().enumerate() {
            let (_, path, file) = create_staging_file(
                &self.config.temp_dir,
                &self.config.call_id,
                &format!("request-{input_position}"),
                &self.config.destination_name,
            )?;
            self.request_to_spool.insert(request_id, input_position);
            self.spools.push(RequestSpool {
                binding: RequestBinding {
                    input_position,
                    request_id,
                },
                path,
                writer: Some(BufWriter::new(file)),
                next_completion_step: 0,
            });
        }
        Ok(())
    }

    pub(crate) fn record_host_rows(&mut self, rows: Vec<HostCaptureRow>) -> Result<()> {
        if self.failed {
            bail!("internal golden capture session already failed");
        }
        for row in rows {
            if row.logits.is_empty() {
                self.failed = true;
                bail!("captured logits row must have a non-zero vocabulary width");
            }
            if row.logits.iter().any(|value| !value.is_finite()) {
                self.failed = true;
                bail!(
                    "captured logits contain a non-finite value for request {} step {}",
                    row.request_id,
                    row.completion_step
                );
            }
            match self.vocab_size {
                Some(vocab_size) if vocab_size != row.logits.len() => {
                    self.failed = true;
                    bail!(
                        "captured logits vocabulary width changed from {vocab_size} to {}",
                        row.logits.len()
                    );
                }
                None => self.vocab_size = Some(row.logits.len()),
                _ => {}
            }

            let spool_index = self
                .request_to_spool
                .get(&row.request_id)
                .copied()
                .ok_or_else(|| {
                    anyhow!(
                        "captured logits reference unknown stable request id {}",
                        row.request_id
                    )
                })?;
            let spool = &mut self.spools[spool_index];
            if row.completion_step != spool.next_completion_step {
                self.failed = true;
                bail!(
                    "captured logits for request {} expected completion step {}, got {}",
                    row.request_id,
                    spool.next_completion_step,
                    row.completion_step
                );
            }
            let writer = spool
                .writer
                .as_mut()
                .ok_or_else(|| anyhow!("capture spool is already closed"))?;
            let spool_row = SpoolRow {
                request_id: row.request_id,
                selected_token: row.selected_token,
                completion_step: row.completion_step,
                logits: row.logits,
            };
            if let Err(error) = write_json_line(writer, &spool_row) {
                self.failed = true;
                return Err(error).context("writing private raw-logit capture spool");
            }
            spool.next_completion_step += 1;
        }
        Ok(())
    }

    pub(crate) fn record_engine_step(
        &mut self,
        step: crate::engine::EngineStepCapture,
    ) -> Result<()> {
        if step.rows.is_empty() {
            if step.logits.dims() != [0, 0] {
                self.failed = true;
                bail!("engine returned logits without capture row identities");
            }
            return Ok(());
        }
        let [row_count, vocab_size] = step.logits.dims() else {
            self.failed = true;
            bail!(
                "captured logits must have rank 2 [rows, vocab], got {:?}",
                step.logits.dims()
            );
        };
        if *row_count != step.rows.len() || *vocab_size == 0 {
            self.failed = true;
            bail!(
                "captured logits shape {:?} does not match {} row identities",
                step.logits.dims(),
                step.rows.len()
            );
        }
        if step.logits.dtype() != candle_core::DType::F32 {
            self.failed = true;
            bail!("captured logits must be F32, got {:?}", step.logits.dtype());
        }
        // This is the only full-logit device-to-host transfer. It is reached
        // only with an explicitly configured, default-off diagnostic session.
        let logits = step
            .logits
            .to_vec2::<f32>()
            .context("copying explicitly requested diagnostic logits to host")?;
        let rows = step
            .rows
            .into_iter()
            .zip(logits)
            .map(|(metadata, logits)| HostCaptureRow {
                request_id: metadata.request_id,
                selected_token: metadata.selected_token,
                completion_step: metadata.completion_step,
                logits,
            })
            .collect();
        self.record_host_rows(rows)
    }

    pub(crate) fn finish(mut self, outputs: &[crate::RequestOutput]) -> Result<PathBuf> {
        if self.failed {
            bail!("cannot publish a failed internal golden capture session");
        }
        for spool in &mut self.spools {
            let mut writer = spool
                .writer
                .take()
                .ok_or_else(|| anyhow!("capture spool is already closed"))?;
            writer.flush().context("flushing raw-logit capture spool")?;
            writer
                .get_ref()
                .sync_all()
                .context("synchronizing raw-logit capture spool")?;
        }

        let stage_file = self
            .final_stage_file
            .take()
            .ok_or_else(|| anyhow!("capture artifact staging file is unavailable"))?;
        let mut writer = BufWriter::new(stage_file);
        let bindings = self
            .spools
            .iter()
            .map(|spool| spool.binding.clone())
            .collect::<Vec<_>>();
        write_json_line(
            &mut writer,
            &ArtifactLine::Header {
                format: FORMAT_NAME.to_string(),
                schema_version: SCHEMA_VERSION,
                call_id: self.config.call_id.clone(),
                requests: bindings.clone(),
            },
        )
        .context("serializing internal golden capture header")?;

        let mut row_index = 0usize;
        for spool in &self.spools {
            let reader = BufReader::new(
                File::open(&spool.path)
                    .with_context(|| format!("opening capture spool {}", spool.path.display()))?,
            );
            for line in reader.lines() {
                let line = line.context("reading raw-logit capture spool")?;
                let row: SpoolRow = serde_json::from_str(&line)
                    .context("validating serialized raw-logit capture spool row")?;
                write_json_line(
                    &mut writer,
                    &ArtifactLine::Row {
                        row_index,
                        input_position: spool.binding.input_position,
                        request_id: row.request_id,
                        selected_token: row.selected_token,
                        completion_step: row.completion_step,
                        dtype: LOGITS_DTYPE.to_string(),
                        row_shape: [row.logits.len()],
                        logits: row.logits,
                    },
                )
                .context("serializing internal golden capture row")?;
                row_index += 1;
            }
        }
        let vocab_size = self.vocab_size.unwrap_or(0);
        write_json_line(
            &mut writer,
            &ArtifactLine::Trailer {
                complete: true,
                row_count: row_index,
                dtype: LOGITS_DTYPE.to_string(),
                tensor_shape: [row_index, vocab_size],
            },
        )
        .context("serializing internal golden capture trailer")?;
        writer.flush().context("flushing capture artifact")?;
        writer
            .get_ref()
            .sync_all()
            .context("synchronizing capture artifact")?;
        drop(writer);

        let stage_path = self
            .final_stage_path
            .as_ref()
            .ok_or_else(|| anyhow!("capture artifact staging path is unavailable"))?;
        validate_artifact(stage_path, &self.config.call_id, &bindings, outputs)?;

        for spool in &self.spools {
            std::fs::remove_file(&spool.path)
                .with_context(|| format!("removing capture spool {}", spool.path.display()))?;
        }
        self.spools.clear();

        let directory = rustix::fs::open(
            &self.config.temp_dir,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::CLOEXEC
                | rustix::fs::OFlags::NOFOLLOW,
            rustix::fs::Mode::empty(),
        )
        .with_context(|| {
            format!(
                "opening private capture directory {}",
                self.config.temp_dir.display()
            )
        })?;
        rustix::fs::renameat_with(
            &directory,
            &self.final_stage_name,
            &directory,
            &self.config.destination_name,
            rustix::fs::RenameFlags::NOREPLACE,
        )
        .with_context(|| {
            format!(
                "atomically publishing capture artifact as {} without replacement",
                self.config.destination_name.to_string_lossy()
            )
        })?;
        self.published = true;
        self.final_stage_path = None;
        Ok(self.config.temp_dir.join(&self.config.destination_name))
    }
}

impl Drop for CaptureSession {
    fn drop(&mut self) {
        for spool in &self.spools {
            let _ = std::fs::remove_file(&spool.path);
        }
        if !self.published {
            if let Some(path) = &self.final_stage_path {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

fn create_staging_file(
    temp_dir: &Path,
    call_id: &str,
    role: &str,
    destination_name: &std::ffi::OsStr,
) -> Result<(OsString, PathBuf, File)> {
    for _ in 0..64 {
        let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
        let name = OsString::from(format!(
            ".vllm-oxide-{call_id}-{}-{nonce}-{role}.staging",
            std::process::id()
        ));
        if name.as_os_str() == destination_name {
            continue;
        }
        let path = temp_dir.join(&name);
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
        {
            Ok(file) => return Ok((name, path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("creating private staging file {}", path.display()))
            }
        }
    }
    bail!("could not allocate a unique private capture staging filename")
}

fn write_json_line(writer: &mut impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, value).context("serializing JSON line")?;
    writer.write_all(b"\n").context("terminating JSON line")?;
    Ok(())
}

fn validate_artifact(
    path: &Path,
    expected_call_id: &str,
    expected_bindings: &[RequestBinding],
    outputs: &[crate::RequestOutput],
) -> Result<()> {
    if outputs.len() != expected_bindings.len() {
        bail!(
            "capture output count {} does not match request count {}",
            outputs.len(),
            expected_bindings.len()
        );
    }
    for (binding, output) in expected_bindings.iter().zip(outputs) {
        if binding.request_id != output.request_id || !output.finished {
            bail!(
                "capture output at input position {} has stale request identity or is unfinished",
                binding.input_position
            );
        }
    }

    let reader = BufReader::new(
        File::open(path).with_context(|| format!("opening staged artifact {}", path.display()))?,
    );
    let mut lines = reader.lines();
    let header = lines
        .next()
        .transpose()
        .context("reading capture header")?
        .ok_or_else(|| anyhow!("capture artifact is empty"))?;
    match serde_json::from_str::<ArtifactLine>(&header).context("parsing capture header")? {
        ArtifactLine::Header {
            format,
            schema_version,
            call_id,
            requests,
        } if format == FORMAT_NAME
            && schema_version == SCHEMA_VERSION
            && call_id == expected_call_id
            && requests == expected_bindings => {}
        _ => bail!("capture artifact header identity or schema mismatch"),
    }

    let mut expected_steps = vec![0usize; expected_bindings.len()];
    let mut selected_tokens = vec![Vec::<u32>::new(); expected_bindings.len()];
    let mut row_count = 0usize;
    let mut vocab_size = None;
    let mut last_input_position = None;
    let mut trailer = None;
    for line in lines {
        let line = line.context("reading capture artifact")?;
        let parsed: ArtifactLine =
            serde_json::from_str(&line).context("parsing capture artifact line")?;
        match parsed {
            ArtifactLine::Row {
                row_index,
                input_position,
                request_id,
                selected_token,
                completion_step,
                dtype,
                row_shape,
                logits,
            } => {
                if trailer.is_some() {
                    bail!("capture artifact contains an additional row after its trailer");
                }
                if row_index != row_count {
                    bail!("capture artifact row indices are duplicate or non-contiguous");
                }
                if last_input_position.is_some_and(|previous| input_position < previous) {
                    bail!("capture artifact row order is not request-major deterministic");
                }
                last_input_position = Some(input_position);
                let binding = expected_bindings.get(input_position).ok_or_else(|| {
                    anyhow!("capture artifact row has additional input position {input_position}")
                })?;
                if request_id != binding.request_id {
                    bail!("capture artifact row has stale stable request identity");
                }
                if completion_step != expected_steps[input_position] {
                    bail!("capture artifact completion steps are duplicate or non-contiguous");
                }
                if dtype != LOGITS_DTYPE
                    || row_shape[0] == 0
                    || row_shape[0] != logits.len()
                    || logits.iter().any(|value| !value.is_finite())
                {
                    bail!("capture artifact row has a dtype, shape, or value mismatch");
                }
                match vocab_size {
                    Some(width) if width != row_shape[0] => {
                        bail!("capture artifact rows have inconsistent vocabulary shapes")
                    }
                    None => vocab_size = Some(row_shape[0]),
                    _ => {}
                }
                expected_steps[input_position] += 1;
                selected_tokens[input_position].push(selected_token);
                row_count += 1;
            }
            ArtifactLine::Trailer {
                complete,
                row_count: declared_rows,
                dtype,
                tensor_shape,
            } => {
                if trailer.is_some() {
                    bail!("capture artifact has duplicate trailers");
                }
                trailer = Some((complete, declared_rows, dtype, tensor_shape));
            }
            ArtifactLine::Header { .. } => bail!("capture artifact has an additional header"),
        }
    }
    let (complete, declared_rows, dtype, tensor_shape) =
        trailer.ok_or_else(|| anyhow!("capture artifact is missing its completion trailer"))?;
    let width = vocab_size.unwrap_or(0);
    if !complete
        || declared_rows != row_count
        || dtype != LOGITS_DTYPE
        || tensor_shape != [row_count, width]
    {
        bail!("capture artifact trailer count, shape, dtype, or completion marker is invalid");
    }
    for (input_position, output) in outputs.iter().enumerate() {
        if selected_tokens[input_position] != output.token_ids {
            bail!(
                "capture artifact rows are missing, additional, or disagree with output at input position {input_position}"
            );
        }
    }
    Ok(())
}

fn validate_call_id(call_id: &str) -> Result<()> {
    if call_id.is_empty()
        || call_id.len() > 128
        || !call_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        bail!("{CALL_ID_ENV} must contain 1-128 ASCII letters, digits, '.', '_', or '-'");
    }
    Ok(())
}

fn validate_destination_name(destination: &std::ffi::OsStr) -> Result<()> {
    use std::os::unix::ffi::OsStrExt;

    let bytes = destination.as_bytes();
    if bytes.is_empty()
        || matches!(bytes, b"." | b"..")
        || bytes.contains(&b'/')
        || bytes.contains(&0)
        || bytes.starts_with(STAGING_PREFIX)
    {
        bail!("{DESTINATION_ENV} must be one non-reserved file basename without path separators");
    }
    Ok(())
}

fn validate_private_temp_dir(path: &Path) -> Result<PathBuf> {
    use std::os::unix::fs::MetadataExt;

    if !path.is_absolute() {
        bail!("{TEMP_DIR_ENV} must be an absolute path");
    }
    ensure_no_symlink_components(path)?;
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("reading capture temporary directory {}", path.display()))?;
    if !metadata.file_type().is_dir() {
        bail!("{TEMP_DIR_ENV} must name a real directory, not a symlink or file");
    }
    if metadata.uid() != rustix::process::geteuid().as_raw() {
        bail!("{TEMP_DIR_ENV} must be owned by the effective caller");
    }
    if metadata.mode() & 0o077 != 0 {
        bail!("{TEMP_DIR_ENV} must be caller-private (mode 0700 or stricter)");
    }
    path.canonicalize().with_context(|| {
        format!(
            "canonicalizing capture temporary directory {}",
            path.display()
        )
    })
}

fn ensure_no_symlink_components(path: &Path) -> Result<()> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        let metadata = std::fs::symlink_metadata(&current)
            .with_context(|| format!("checking path component {}", current.display()))?;
        if metadata.file_type().is_symlink() {
            bail!(
                "{TEMP_DIR_ENV} must not contain symlink components: {}",
                current.display()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn private_temp() -> tempfile::TempDir {
        let temp = tempfile::tempdir().unwrap();
        std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        temp
    }

    fn config(temp: &Path, destination: &str, call_id: &str) -> CaptureConfig {
        CaptureConfig::from_values(
            Some(temp.as_os_str().to_os_string()),
            Some(OsString::from(destination)),
            Some(OsString::from(call_id)),
        )
        .unwrap()
        .unwrap()
    }

    fn benchmark_config(temp: &Path, destination: &str, call_id: &str) -> BenchmarkConfig {
        BenchmarkConfig::from_values(
            Some(temp.as_os_str().to_owned()),
            Some(OsString::from(destination)),
            Some(OsString::from(call_id)),
        )
        .unwrap()
        .unwrap()
    }

    fn output(request_id: usize, token_ids: Vec<u32>) -> crate::RequestOutput {
        crate::RequestOutput {
            request_id,
            token_ids,
            text: String::new(),
            finished: true,
        }
    }

    #[test]
    fn absent_configuration_is_inert_without_touching_the_filesystem() {
        let config = CaptureConfig::from_values(None, None, None).unwrap();

        assert!(config.is_none());
    }

    #[test]
    fn partial_configuration_names_every_missing_variable() {
        let error = CaptureConfig::from_values(
            Some(OsString::from("/does/not/matter")),
            None,
            Some(OsString::from("call-1")),
        )
        .unwrap_err();

        assert!(error.to_string().contains(DESTINATION_ENV));
    }

    #[test]
    fn complete_configuration_accepts_a_private_directory_and_fresh_basename() {
        let temp = private_temp();
        let config = config(temp.path(), "capture.jsonl", "fixture-01");

        assert_eq!(config.temp_dir, temp.path().canonicalize().unwrap());
        assert_eq!(config.destination_name, "capture.jsonl");
        assert_eq!(config.call_id, "fixture-01");
    }

    #[test]
    fn artifact_is_published_in_request_then_completion_order() {
        let temp = private_temp();
        let config = config(temp.path(), "capture.jsonl", "fixture-ordered");
        let mut session = CaptureSession::prepare(config).unwrap();
        session.bind_requests(&[7, 9]).unwrap();
        session
            .record_host_rows(vec![
                HostCaptureRow {
                    request_id: 9,
                    selected_token: 21,
                    completion_step: 0,
                    logits: vec![0.0, 1.0, 2.0],
                },
                HostCaptureRow {
                    request_id: 7,
                    selected_token: 11,
                    completion_step: 0,
                    logits: vec![2.0, 1.0, 0.0],
                },
            ])
            .unwrap();
        session
            .record_host_rows(vec![HostCaptureRow {
                request_id: 7,
                selected_token: 12,
                completion_step: 1,
                logits: vec![1.0, 2.0, 0.0],
            }])
            .unwrap();
        let outputs = vec![output(7, vec![11, 12]), output(9, vec![21])];

        let destination = session.finish(&outputs).unwrap();
        let lines = std::fs::read_to_string(destination)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(lines[1]["input_position"], 0);
        assert_eq!(lines[1]["completion_step"], 0);
        assert_eq!(lines[2]["input_position"], 0);
        assert_eq!(lines[2]["completion_step"], 1);
        assert_eq!(lines[3]["input_position"], 1);
        assert_eq!(lines[3]["completion_step"], 0);
        assert_eq!(lines[4]["complete"], true);
    }

    #[test]
    fn engine_step_capture_binds_selected_token_and_logits_row() {
        let temp = private_temp();
        let config = config(temp.path(), "capture.jsonl", "fixture-engine-step");
        let mut session = CaptureSession::prepare(config).unwrap();
        session.bind_requests(&[42]).unwrap();
        let logits = candle_core::Tensor::from_vec(
            vec![1.0_f32, 3.0, 2.0],
            (1, 3),
            &candle_core::Device::Cpu,
        )
        .unwrap();

        session
            .record_engine_step(crate::engine::EngineStepCapture {
                rows: vec![crate::engine::EngineCaptureRow {
                    request_id: 42,
                    selected_token: 1,
                    completion_step: 0,
                }],
                phase: Some(crate::engine::StepPhase::Prefill),
                prefill_tokens: 1,
                logits,
            })
            .unwrap();
        let destination = session.finish(&[output(42, vec![1])]).unwrap();
        let row: serde_json::Value = serde_json::from_str(
            std::fs::read_to_string(destination)
                .unwrap()
                .lines()
                .nth(1)
                .unwrap(),
        )
        .unwrap();

        assert_eq!(row["selected_token"], 1);
        assert_eq!(row["logits"], serde_json::json!([1.0, 3.0, 2.0]));
    }

    #[test]
    fn configuration_rejects_unsafe_paths_and_identities() {
        let insecure = private_temp();
        std::fs::set_permissions(insecure.path(), std::fs::Permissions::from_mode(0o750)).unwrap();
        let error = CaptureConfig::from_values(
            Some(insecure.path().as_os_str().to_os_string()),
            Some(OsString::from("capture.jsonl")),
            Some(OsString::from("call-1")),
        )
        .unwrap_err();
        assert!(error.to_string().contains("caller-private"));

        let temp = private_temp();
        let error = CaptureConfig::from_values(
            Some(temp.path().as_os_str().to_os_string()),
            Some(OsString::from("../escape.jsonl")),
            Some(OsString::from("call-1")),
        )
        .unwrap_err();
        assert!(error.to_string().contains("basename"));

        let error = CaptureConfig::from_values(
            Some(temp.path().as_os_str().to_os_string()),
            Some(OsString::from("capture.jsonl")),
            Some(OsString::from("invalid/call")),
        )
        .unwrap_err();
        assert!(error.to_string().contains(CALL_ID_ENV));

        let real = temp.path().join("real");
        std::fs::create_dir(&real).unwrap();
        std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o700)).unwrap();
        let linked = temp.path().join("linked");
        std::os::unix::fs::symlink(&real, &linked).unwrap();
        let error = CaptureConfig::from_values(
            Some(linked.into_os_string()),
            Some(OsString::from("capture.jsonl")),
            Some(OsString::from("call-1")),
        )
        .unwrap_err();
        assert!(error.to_string().contains("symlink components"));
    }

    #[test]
    fn configuration_rejects_existing_file_or_symlink_destination() {
        let temp = private_temp();
        std::fs::write(temp.path().join("capture.jsonl"), b"existing").unwrap();
        let error = CaptureConfig::from_values(
            Some(temp.path().as_os_str().to_os_string()),
            Some(OsString::from("capture.jsonl")),
            Some(OsString::from("call-1")),
        )
        .unwrap_err();
        assert!(error.to_string().contains("already exists"));

        std::fs::remove_file(temp.path().join("capture.jsonl")).unwrap();
        std::os::unix::fs::symlink("missing-target", temp.path().join("capture.jsonl")).unwrap();
        let error = CaptureConfig::from_values(
            Some(temp.path().as_os_str().to_os_string()),
            Some(OsString::from("capture.jsonl")),
            Some(OsString::from("call-1")),
        )
        .unwrap_err();
        assert!(error.to_string().contains("already exists"));
    }

    #[test]
    fn destination_validation_rejects_raw_non_basename_spellings_before_admission() {
        use std::os::unix::ffi::OsStringExt;

        let temp = private_temp();
        for destination in [
            OsString::from(""),
            OsString::from("."),
            OsString::from(".."),
            OsString::from("capture.jsonl/"),
            OsString::from("capture.jsonl/."),
            OsString::from("./capture.jsonl"),
            OsString::from(".vllm-oxide-call-1-123-0-artifact.staging"),
            OsString::from_vec(b"capture\0.jsonl".to_vec()),
        ] {
            let error = CaptureConfig::from_values(
                Some(temp.path().as_os_str().to_os_string()),
                Some(destination),
                Some(OsString::from("call-1")),
            )
            .unwrap_err();

            assert!(error.to_string().contains("basename"));
        }
    }

    #[test]
    fn publication_collision_preserves_owner_file_and_cleans_private_staging() {
        let temp = private_temp();
        let mut session =
            CaptureSession::prepare(config(temp.path(), "capture.jsonl", "fixture-collision"))
                .unwrap();
        session.bind_requests(&[7]).unwrap();
        session
            .record_host_rows(vec![HostCaptureRow {
                request_id: 7,
                selected_token: 2,
                completion_step: 0,
                logits: vec![0.0, 1.0, 2.0],
            }])
            .unwrap();
        std::fs::write(temp.path().join("capture.jsonl"), b"owner-data").unwrap();

        let error = session.finish(&[output(7, vec![2])]).unwrap_err();

        assert!(format!("{error:#}").contains("without replacement"));
        assert_eq!(
            std::fs::read(temp.path().join("capture.jsonl")).unwrap(),
            b"owner-data"
        );
        assert!(std::fs::read_dir(temp.path()).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("staging")));
    }

    #[test]
    fn missing_capture_row_fails_self_check_without_publishing() {
        let temp = private_temp();
        let mut session =
            CaptureSession::prepare(config(temp.path(), "capture.jsonl", "fixture-missing"))
                .unwrap();
        session.bind_requests(&[7]).unwrap();
        session
            .record_host_rows(vec![HostCaptureRow {
                request_id: 7,
                selected_token: 2,
                completion_step: 0,
                logits: vec![0.0, 1.0, 2.0],
            }])
            .unwrap();

        let error = session.finish(&[output(7, vec![2, 1])]).unwrap_err();

        assert!(error
            .to_string()
            .contains("missing, additional, or disagree"));
        assert!(!temp.path().join("capture.jsonl").exists());
    }

    #[test]
    fn capture_rejects_gap_non_finite_values_and_engine_shape_mismatch() {
        let temp = private_temp();
        let mut session =
            CaptureSession::prepare(config(temp.path(), "capture.jsonl", "fixture-invalid-row"))
                .unwrap();
        session.bind_requests(&[7]).unwrap();
        let error = session
            .record_host_rows(vec![HostCaptureRow {
                request_id: 7,
                selected_token: 2,
                completion_step: 1,
                logits: vec![0.0, 1.0, 2.0],
            }])
            .unwrap_err();
        assert!(error.to_string().contains("expected completion step 0"));

        let temp = private_temp();
        let mut session =
            CaptureSession::prepare(config(temp.path(), "capture.jsonl", "fixture-non-finite"))
                .unwrap();
        session.bind_requests(&[7]).unwrap();
        let error = session
            .record_host_rows(vec![HostCaptureRow {
                request_id: 7,
                selected_token: 2,
                completion_step: 0,
                logits: vec![0.0, f32::NAN, 2.0],
            }])
            .unwrap_err();
        assert!(error.to_string().contains("non-finite"));

        let temp = private_temp();
        let mut session =
            CaptureSession::prepare(config(temp.path(), "capture.jsonl", "fixture-shape")).unwrap();
        session.bind_requests(&[7]).unwrap();
        let logits =
            candle_core::Tensor::zeros((2, 3), candle_core::DType::F32, &candle_core::Device::Cpu)
                .unwrap();
        let error = session
            .record_engine_step(crate::engine::EngineStepCapture {
                rows: vec![crate::engine::EngineCaptureRow {
                    request_id: 7,
                    selected_token: 2,
                    completion_step: 0,
                }],
                phase: Some(crate::engine::StepPhase::Prefill),
                prefill_tokens: 1,
                logits,
            })
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("does not match 1 row identities"));
    }

    #[test]
    fn spool_write_failure_cannot_publish_a_partial_destination() {
        let temp = private_temp();
        let mut session = CaptureSession::prepare(config(
            temp.path(),
            "capture.jsonl",
            "fixture-write-failure",
        ))
        .unwrap();
        session.bind_requests(&[7]).unwrap();
        let spool_path = session.spools[0].path.clone();
        drop(session.spools[0].writer.take());
        session.spools[0].writer =
            Some(BufWriter::with_capacity(1, File::open(spool_path).unwrap()));

        let error = session
            .record_host_rows(vec![HostCaptureRow {
                request_id: 7,
                selected_token: 2,
                completion_step: 0,
                logits: vec![0.0, 1.0, 2.0],
            }])
            .unwrap_err();

        assert!(format!("{error:#}").contains("serializing JSON line"));
        drop(session);
        assert!(!temp.path().join("capture.jsonl").exists());
    }

    #[test]
    fn benchmark_artifact_contains_only_small_synchronized_step_telemetry() {
        let temp = private_temp();
        let mut session = BenchmarkSession::prepare(benchmark_config(
            temp.path(),
            "benchmark.json",
            "canonical-04-repetition-1",
        ))
        .unwrap();
        session
            .record_inputs(serde_json::json!({"prompt_token_ids":[vec![1;8]]}))
            .unwrap();
        session.bind_requests(&[7]).unwrap();
        session
            .record_engine_step(
                crate::engine::EngineStepTelemetry {
                    phase: Some(crate::engine::StepPhase::Prefill),
                    prefill_tokens: 8,
                    emissions: vec![crate::engine::EngineCaptureRow {
                        request_id: 7,
                        selected_token: 11,
                        completion_step: 0,
                    }],
                },
                0,
                10_000_000,
            )
            .unwrap();
        session
            .record_engine_step(
                crate::engine::EngineStepTelemetry {
                    phase: Some(crate::engine::StepPhase::Decode),
                    prefill_tokens: 0,
                    emissions: vec![crate::engine::EngineCaptureRow {
                        request_id: 7,
                        selected_token: 12,
                        completion_step: 1,
                    }],
                },
                10_000_000,
                12_000_000,
            )
            .unwrap();

        let path = session.finish(&[output(7, vec![11, 12])]).unwrap();
        let bytes = std::fs::read_to_string(path).unwrap();
        let artifact: serde_json::Value = serde_json::from_str(&bytes).unwrap();

        assert_eq!(artifact["complete"], true);
        assert_eq!(artifact["telemetry"]["prefill_tokens"], 8);
        assert_eq!(artifact["telemetry"]["decode_tokens"], 1);
        assert!(!bytes.contains("logits"));
    }

    #[test]
    fn incomplete_benchmark_configuration_fails_before_request_admission() {
        let error = BenchmarkConfig::from_values(
            Some(OsString::from("/does/not/matter")),
            None,
            Some(OsString::from("call-1")),
        )
        .unwrap_err();

        assert!(error.to_string().contains(BENCHMARK_DESTINATION_ENV));
    }
}
