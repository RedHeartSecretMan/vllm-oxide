//! Private fixed-history execution. Legacy capture remains an independent format.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

use crate::engine::{SequencePhase, StepPlan, StepResult};
use crate::{RequestOutput, SamplingParams};

pub(crate) const PLAN_ENV: &str = "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_PLAN";
pub(crate) const OUTPUT_ENV: &str = "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_OUTPUT";
pub(crate) const CONTROL_ENV: &str = "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_CONTROL";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Member {
    case_id: String,
    member_id: String,
    prompt: Vec<u32>,
    continuation: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    protocol: String,
    schema_version: u32,
    execution_group_id: String,
    call_id: String,
    vocab_size: usize,
    members: Vec<Member>,
}

pub(crate) struct ReplaySession {
    plan: Plan,
    request_ids: Vec<usize>,
    next_rows: Vec<usize>,
    advances: Vec<Vec<u32>>,
    control: bool,
    cache_blocks: Option<usize>,
    file: File,
    stage: PathBuf,
    destination: PathBuf,
    rows: usize,
    execution_events: Vec<serde_json::Value>,
    public_params: Vec<serde_json::Value>,
    public_binding: Option<serde_json::Value>,
}

impl ReplaySession {
    pub(crate) fn from_env(
        prompts: &[Vec<u32>],
        params: &[SamplingParams],
    ) -> Result<Option<Self>> {
        let (request, destination) = (std::env::var_os(PLAN_ENV), std::env::var_os(OUTPUT_ENV));
        let control = match std::env::var(CONTROL_ENV) {
            Ok(value) if value == "1" => true,
            Err(std::env::VarError::NotPresent) => false,
            _ => bail!("fixed-prefix control flag must be absent or exactly 1"),
        };
        if request.is_none() && destination.is_none() {
            if control {
                bail!("orphaned fixed-prefix control flag");
            }
            return Ok(None);
        }
        let request = request.context("fixed-prefix plan is missing")?;
        let destination = PathBuf::from(destination.context("fixed-prefix output is missing")?);
        let plan: Plan = serde_json::from_str(&std::fs::read_to_string(request)?)?;
        if plan.protocol != "layered-accuracy-v1"
            || plan.schema_version != 1
            || plan.execution_group_id.is_empty()
            || plan.call_id.is_empty()
            || plan.vocab_size == 0
            || plan.members.is_empty()
            || plan.members.len() != prompts.len()
            || params.len() != prompts.len()
            || plan
                .members
                .iter()
                .map(|m| &m.member_id)
                .collect::<HashSet<_>>()
                .len()
                != plan.members.len()
        {
            bail!("invalid fixed-prefix plan identity or members");
        }
        for ((member, prompt), params) in plan.members.iter().zip(prompts).zip(params) {
            if member.member_id.is_empty()
                || member.case_id.is_empty()
                || member.prompt != *prompt
                || member.prompt.is_empty()
                || member.continuation.is_empty()
                || member
                    .prompt
                    .iter()
                    .chain(&member.continuation)
                    .any(|&t| t as usize >= plan.vocab_size)
                || params.max_tokens != member.continuation.len()
                || !params.ignore_eos
                || params.temperature != 0.0
                || params.top_k.is_some()
                || params.top_p.is_some()
                || params.presence_penalty != 0.0
                || params.frequency_penalty != 0.0
                || params.repetition_penalty != 0.0
            {
                bail!("fixed-prefix prompt/continuation or neutral greedy/EOS policy mismatch");
            }
        }
        let parent = super::validate_private_temp_dir(
            destination.parent().context("fixed output has no parent")?,
        )?;
        let name = destination
            .file_name()
            .context("fixed output has no name")?;
        super::validate_destination_name(name)?;
        let destination = parent.join(name);
        if destination.exists() || destination.is_symlink() {
            bail!("fixed output must be fresh");
        }
        let stage = destination.with_extension("partial");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&stage)?;
        let header = serde_json::json!({"protocol":"layered-accuracy-v1","schema_version":1,
            "execution_group_id":plan.execution_group_id,"call_id":plan.call_id,
            "mode":if control {"collection_control"}else{"fixed_prefix"},"ignore_eos":true});
        let mut prefix = serde_json::to_string(&header)?;
        prefix.pop();
        write!(file, "{prefix},\"rows\":[")?;
        let next_rows = vec![0; plan.members.len()];
        let advances = vec![Vec::new(); plan.members.len()];
        let public_params=params.iter().map(|p|serde_json::json!({"max_tokens":p.max_tokens,"ignore_eos":p.ignore_eos,
            "temperature":p.temperature,"top_k":p.top_k,"top_p":p.top_p,"presence_penalty":p.presence_penalty,
            "frequency_penalty":p.frequency_penalty,"repetition_penalty":p.repetition_penalty})).collect();
        Ok(Some(Self {
            plan,
            request_ids: Vec::new(),
            next_rows,
            advances,
            control,
            cache_blocks: None,
            file,
            stage,
            destination,
            rows: 0,
            execution_events: Vec::new(),
            public_params,
            public_binding: None,
        }))
    }

    pub(crate) fn bind_requests(
        &mut self,
        ids: &[usize],
        cache_blocks: usize,
        eos: &[u32],
        max_model_len: usize,
        device: &candle_core::Device,
    ) -> Result<()> {
        if !self.request_ids.is_empty()
            || ids.len() != self.plan.members.len()
            || ids.iter().collect::<HashSet<_>>().len() != ids.len()
            || cache_blocks == 0
        {
            bail!("fixed request binding mismatch");
        }
        self.request_ids = ids.to_vec();
        self.cache_blocks = Some(cache_blocks);
        self.public_binding = Some(
            serde_json::json!({"protocol":"layered-accuracy-v1","schema_version":1,
            "mode":"behavior_binding","request_ids":ids,"prompt_lengths":self.plan.members.iter().map(|m|m.prompt.len()).collect::<Vec<_>>(),
            "eos_token_ids":eos,"max_model_len":max_model_len,"device":if device.is_cuda(){"cuda:0"}else{"cpu"},"forcing_enabled":!self.control}),
        );
        Ok(())
    }

    pub(crate) fn record_and_advance(
        &mut self,
        plan: &StepPlan,
        result: &mut StepResult,
        logits: &Tensor,
    ) -> Result<()> {
        let members = plan.sequences.iter().map(|s| serde_json::json!({
            "request_id":s.request_id,"sequence_id":s.sequence_id,
            "phase":match s.phase { SequencePhase::Prefill=>"prefill",SequencePhase::Decode=>"decode" },
            "completion_step":s.completion_step,"sampling_allowed":s.sampling_allowed,
            "input_token_ids":s.input_token_ids,"positions":[s.logical_positions.start,s.logical_positions.end],
            "cached_range":[s.cache.cached_token_range.start,s.cache.cached_token_range.end],
            "kv_length":s.cache.kv_length,"slot_mapping":s.cache.slot_mapping,"block_table":s.cache.block_table
        })).collect::<Vec<_>>();
        self.execution_events.push(serde_json::json!({"plan_id":plan.id,"token_budget":plan.token_budget,"members":members}));
        let expected = plan.sequences.iter().filter(|s| s.sampling_allowed).count();
        if expected == 0 {
            return Ok(());
        }
        if logits.dtype() != DType::F32 || logits.dims() != [expected, self.plan.vocab_size] {
            bail!("fixed-prefix raw-logit shape/dtype mismatch");
        }
        let host = logits.to_vec2::<f32>()?;
        let mut sample_row = 0;
        for (planned, executed) in plan.sequences.iter().zip(&mut result.sequences) {
            if !planned.sampling_allowed {
                continue;
            }
            let member_index = self
                .request_ids
                .iter()
                .position(|id| *id == planned.request_id)
                .context("unbound fixed request")?;
            let member = &self.plan.members[member_index];
            let step = self.next_rows[member_index];
            let frozen_advance = *member
                .continuation
                .get(step)
                .context("fixed-prefix execution exceeded frozen rows")?;
            let history: Vec<u32> = member
                .prompt
                .iter()
                .chain(&self.advances[member_index])
                .copied()
                .collect();
            if planned.completion_step != step
                || planned.token_history != history
                || planned.logical_positions.end != history.len()
                || Some(planned.input_token_ids.as_slice())
                    != history.get(planned.token_range.clone())
            {
                bail!("executed StepPlan differs from frozen causal history");
            }
            let raw = &host[sample_row];
            sample_row += 1;
            if raw.iter().any(|v| !v.is_finite()) {
                bail!("nonfinite fixed-prefix logits");
            }
            let greedy = raw
                .iter()
                .enumerate()
                .fold(0, |best, (id, v)| if *v > raw[best] { id } else { best });
            let predicted = executed
                .sampled_token
                .context("fixed-prefix prediction missing")?;
            if predicted as usize != greedy {
                bail!("raw predicted token violates greedy tie contract");
            }
            let advance = if self.control {
                predicted
            } else {
                frozen_advance
            };
            let mut digest = Sha256::new();
            for token in &history {
                digest.update(token.to_le_bytes());
            }
            let row = serde_json::json!({"kind":"prediction","case_id":member.case_id,"execution_group_id":self.plan.execution_group_id,"call_id":self.plan.call_id,
                "member_id":member.member_id,"request_id":planned.request_id,"step":step,
                "history_sha256":format!("{:x}",digest.finalize()),"position":history.len()-1,
                "phase":match planned.phase { SequencePhase::Prefill=>"prefill",SequencePhase::Decode=>"decode" },
                "effective_length":history.len(),"row_shape":[raw.len()],"logits":raw,
                "predicted_token_id":predicted,"advance_token_id":advance});
            if self.rows > 0 {
                self.file.write_all(b",")?;
            }
            serde_json::to_writer(&mut self.file, &row)?;
            self.file.flush()?;
            if !self.control {
                executed.sampled_token = Some(advance);
            }
            self.advances[member_index].push(advance);
            self.next_rows[member_index] += 1;
            self.rows += 1;
        }
        Ok(())
    }

    pub(crate) fn finish(mut self, outputs: &[RequestOutput]) -> Result<()> {
        if outputs.len() != self.plan.members.len() {
            bail!("fixed-prefix missing outputs");
        }
        for (index, member) in self.plan.members.iter().enumerate() {
            let output = outputs
                .iter()
                .find(|o| o.request_id == self.request_ids[index])
                .context("fixed output request missing")?;
            if self.next_rows[index] != member.continuation.len()
                || output.token_ids != self.advances[index]
                || !output.finished
            {
                bail!("fixed-prefix output/row completeness mismatch");
            }
        }
        self.file.write_all(b"],\"execution_events\":")?;
        serde_json::to_writer(&mut self.file, &self.execution_events)?;
        self.file.write_all(b",\"public_call\":")?;
        serde_json::to_writer(
            &mut self.file,
            &serde_json::json!({"call_id":self.plan.call_id,
            "prompts":self.plan.members.iter().map(|m|&m.prompt).collect::<Vec<_>>(),"params":self.public_params,
            "binding":self.public_binding.context("missing public request binding")?,"error":null,
            "outputs":outputs.iter().map(|o|serde_json::json!({"request_id":o.request_id,"token_ids":o.token_ids,"text":o.text,"finished":o.finished})).collect::<Vec<_>>()}),
        )?;
        writeln!(
            self.file,
            ",\"allocated_cache_blocks\":{},\"complete\":true}}",
            self.cache_blocks.context("missing actual cache capacity")?
        )?;
        self.file.sync_all()?;
        std::fs::hard_link(&self.stage, &self.destination)?;
        std::fs::remove_file(&self.stage)?;
        File::open(
            self.destination
                .parent()
                .context("fixed output parent missing")?,
        )?
        .sync_all()?;
        Ok(())
    }
}

impl crate::engine::GoldenStepControl for ReplaySession {
    fn observe_and_advance(
        &mut self,
        plan: &StepPlan,
        result: &mut StepResult,
        logits: &Tensor,
    ) -> candle_core::Result<()> {
        self.record_and_advance(plan, result, logits)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))
    }
}
