//! Private benchmark artifact producer, separate from raw-logit capture.

use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::{
    create_staging_file, validate_call_id, validate_destination_name, validate_private_temp_dir,
};

pub(super) const BENCHMARK_TEMP_DIR_ENV: &str = "VLLM_OXIDE_INTERNAL_BENCHMARK_TEMP_DIR";
pub(super) const BENCHMARK_DESTINATION_ENV: &str = "VLLM_OXIDE_INTERNAL_BENCHMARK_DESTINATION";
pub(super) const BENCHMARK_CALL_ID_ENV: &str = "VLLM_OXIDE_INTERNAL_BENCHMARK_CALL_ID";

#[derive(Debug)]
pub(super) struct BenchmarkConfig {
    temp_dir: PathBuf,
    destination_name: OsString,
    call_id: String,
}

impl BenchmarkConfig {
    fn from_env() -> Result<Option<Self>> {
        Self::from_values(
            std::env::var_os(BENCHMARK_TEMP_DIR_ENV),
            std::env::var_os(BENCHMARK_DESTINATION_ENV),
            std::env::var_os(BENCHMARK_CALL_ID_ENV),
        )
    }

    pub(super) fn from_values(
        temp_dir: Option<OsString>,
        destination: Option<OsString>,
        call_id: Option<OsString>,
    ) -> Result<Option<Self>> {
        if temp_dir.is_none() && destination.is_none() && call_id.is_none() {
            return Ok(None);
        }
        let missing = [
            (BENCHMARK_TEMP_DIR_ENV, temp_dir.is_none()),
            (BENCHMARK_DESTINATION_ENV, destination.is_none()),
            (BENCHMARK_CALL_ID_ENV, call_id.is_none()),
        ]
        .into_iter()
        .filter_map(|(name, missing)| missing.then_some(name))
        .collect::<Vec<_>>();
        if !missing.is_empty() {
            bail!(
                "internal benchmark telemetry configuration is incomplete; missing {}",
                missing.join(", ")
            );
        }
        let temp_dir = PathBuf::from(temp_dir.ok_or_else(|| anyhow!("missing telemetry temp"))?);
        let destination_name =
            destination.ok_or_else(|| anyhow!("missing telemetry destination"))?;
        let call_id = call_id
            .ok_or_else(|| anyhow!("missing telemetry call identity"))?
            .into_string()
            .map_err(|_| anyhow!("{BENCHMARK_CALL_ID_ENV} must be valid UTF-8"))?;
        validate_call_id(&call_id)?;
        validate_destination_name(&destination_name)?;
        let temp_dir = validate_private_temp_dir(&temp_dir)?;
        let destination_path = temp_dir.join(&destination_name);
        match std::fs::symlink_metadata(&destination_path) {
            Ok(_) => {
                bail!("{BENCHMARK_DESTINATION_ENV} must name a fresh non-existing destination")
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("checking benchmark telemetry destination"),
        }
        Ok(Some(Self {
            temp_dir,
            destination_name,
            call_id,
        }))
    }
}

const BENCHMARK_FORMAT: &str = "vllm-oxide-internal-benchmark-json-v1";

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct BenchmarkArtifact {
    format: String,
    schema_version: u32,
    call_id: String,
    request_ids: Vec<usize>,
    complete: bool,
    telemetry: crate::benchmark_telemetry::BenchmarkTelemetry,
}

pub(crate) struct BenchmarkSession {
    config: BenchmarkConfig,
    stage_name: OsString,
    stage_path: Option<PathBuf>,
    stage_file: Option<File>,
    request_ids: Vec<usize>,
    samples: Vec<crate::benchmark_telemetry::StepSample>,
    published: bool,
}

impl BenchmarkSession {
    pub(crate) fn from_env() -> Result<Option<Self>> {
        BenchmarkConfig::from_env()?.map(Self::prepare).transpose()
    }

    pub(super) fn prepare(config: BenchmarkConfig) -> Result<Self> {
        let (stage_name, stage_path, stage_file) = create_staging_file(
            &config.temp_dir,
            &config.call_id,
            "benchmark",
            &config.destination_name,
        )?;
        Ok(Self {
            config,
            stage_name,
            stage_path: Some(stage_path),
            stage_file: Some(stage_file),
            request_ids: Vec::new(),
            samples: Vec::new(),
            published: false,
        })
    }

    pub(crate) fn bind_requests(&mut self, request_ids: &[usize]) -> Result<()> {
        if !self.request_ids.is_empty() {
            bail!("internal benchmark requests are already bound");
        }
        if request_ids.iter().copied().collect::<HashSet<_>>().len() != request_ids.len() {
            bail!("internal benchmark received duplicate stable request ids");
        }
        self.request_ids.extend_from_slice(request_ids);
        Ok(())
    }

    pub(crate) fn record_engine_step(
        &mut self,
        step: crate::engine::EngineStepTelemetry,
        started_ns: u64,
        ended_ns: u64,
    ) -> Result<()> {
        let phase = match step
            .phase
            .ok_or_else(|| anyhow!("benchmark engine step has no executable plan"))?
        {
            crate::engine::StepPhase::Prefill => crate::benchmark_telemetry::StepPhase::Prefill,
            crate::engine::StepPhase::Decode => crate::benchmark_telemetry::StepPhase::Decode,
            crate::engine::StepPhase::Mixed => crate::benchmark_telemetry::StepPhase::Mixed,
        };
        let emissions = step
            .emissions
            .into_iter()
            .map(|emission| crate::benchmark_telemetry::Emission {
                request_id: emission.request_id,
                completion_step: emission.completion_step,
                sampled_at_ns: ended_ns,
            })
            .collect();
        self.samples.push(crate::benchmark_telemetry::StepSample {
            phase,
            started_ns,
            ended_ns,
            prefill_tokens: step.prefill_tokens,
            emissions,
        });
        Ok(())
    }

    pub(crate) fn finish(mut self, outputs: &[crate::RequestOutput]) -> Result<PathBuf> {
        if outputs.len() != self.request_ids.len()
            || outputs
                .iter()
                .zip(&self.request_ids)
                .any(|(output, request_id)| output.request_id != *request_id || !output.finished)
        {
            bail!("benchmark outputs do not match the bound request identities");
        }
        let telemetry = crate::benchmark_telemetry::BenchmarkTelemetry::from_samples(
            &self.request_ids,
            std::mem::take(&mut self.samples),
        )?;
        let artifact = BenchmarkArtifact {
            format: BENCHMARK_FORMAT.to_string(),
            schema_version: 1,
            call_id: self.config.call_id.clone(),
            request_ids: self.request_ids.clone(),
            complete: true,
            telemetry,
        };
        let file = self
            .stage_file
            .take()
            .ok_or_else(|| anyhow!("benchmark staging file is unavailable"))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &artifact)
            .context("serializing private benchmark telemetry")?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        let stage_path = self
            .stage_path
            .as_ref()
            .ok_or_else(|| anyhow!("benchmark staging path is unavailable"))?;
        let decoded: BenchmarkArtifact = serde_json::from_slice(
            &std::fs::read(stage_path).context("reading staged benchmark telemetry")?,
        )
        .context("validating staged benchmark telemetry")?;
        if decoded != artifact {
            bail!("staged benchmark telemetry failed its self-check");
        }
        let directory = rustix::fs::open(
            &self.config.temp_dir,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::CLOEXEC
                | rustix::fs::OFlags::NOFOLLOW,
            rustix::fs::Mode::empty(),
        )?;
        rustix::fs::renameat_with(
            &directory,
            &self.stage_name,
            &directory,
            &self.config.destination_name,
            rustix::fs::RenameFlags::NOREPLACE,
        )
        .context("atomically publishing benchmark telemetry without replacement")?;
        self.published = true;
        self.stage_path = None;
        Ok(self.config.temp_dir.join(&self.config.destination_name))
    }
}

impl Drop for BenchmarkSession {
    fn drop(&mut self) {
        if !self.published {
            if let Some(path) = &self.stage_path {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}
