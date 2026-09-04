use std::ops::Range;

use crate::attention::AttnMetadata;
use crate::engine::kv_cache_manager::KvCacheError;
use crate::SamplingParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheOperation {
    Allocation,
    AllocationRollback,
    EmptyAllocationRollback,
    AppendAllocation,
    Deallocation,
    FailureCleanup,
}

impl std::fmt::Display for CacheOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allocation => f.write_str("allocation"),
            Self::AllocationRollback => f.write_str("allocation rollback"),
            Self::EmptyAllocationRollback => f.write_str("empty allocation rollback"),
            Self::AppendAllocation => f.write_str("append allocation"),
            Self::Deallocation => f.write_str("deallocation"),
            Self::FailureCleanup => f.write_str("failure cleanup"),
        }
    }
}

/// Aggregate execution phase for one immutable engine step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepPhase {
    Prefill,
    Decode,
    Mixed,
}

/// Execution phase for one sequence inside a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SequencePhase {
    Prefill,
    Decode,
}

/// Why the FIFO head could not be admitted in this scheduling decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdmissionBlockedReason {
    TokenBudget,
    SequenceLimit,
    KvCache,
}

impl std::fmt::Display for AdmissionBlockedReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenBudget => f.write_str("token budget"),
            Self::SequenceLimit => f.write_str("sequence limit"),
            Self::KvCache => f.write_str("KV cache"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockedAdmission {
    pub(crate) request_id: usize,
    pub(crate) reason: AdmissionBlockedReason,
}

/// Immutable description of one scheduler-selected engine step.
///
/// `token_budget` is the exact sum of positive per-sequence budgets and never
/// exceeds the Scheduler's configured global budget.
#[derive(Debug, Clone)]
pub(crate) struct StepPlan {
    pub(crate) id: u64,
    pub(crate) phase: StepPhase,
    pub(crate) sequences: Vec<SequenceStepPlan>,
    pub(crate) token_budget: usize,
    pub(crate) attention: AttnMetadata,
    /// Diagnostic only: this request contributed no work to the plan.
    pub(crate) blocked_admission: Option<BlockedAdmission>,
}

/// Immutable work assigned to one sequence within a [`StepPlan`].
#[derive(Debug, Clone)]
pub(crate) struct SequenceStepPlan {
    pub(crate) request_id: usize,
    pub(crate) sequence_id: usize,
    pub(crate) phase: SequencePhase,
    /// Exact newly scheduled model-input range. Cached logical-prefix tokens
    /// are not repeated here; their range is captured by
    /// `cache.cached_token_range`. Recovery prefill may extend through prior
    /// completion tokens.
    pub(crate) token_range: Range<usize>,
    /// Causal positions corresponding one-for-one with `token_range`.
    pub(crate) logical_positions: Range<usize>,
    pub(crate) token_budget: usize,
    pub(crate) cache: SequenceCachePlan,
    pub(crate) sampling_allowed: bool,
    pub(crate) input_token_ids: Vec<u32>,
    pub(crate) sampling_params: SamplingParams,
    pub(crate) token_history: Vec<u32>,
}

/// Cache state and mappings captured for one planned sequence.
#[derive(Debug, Clone)]
pub(crate) struct SequenceCachePlan {
    /// Logical tokens already represented in paged KV before this step.
    pub(crate) cached_token_range: Range<usize>,
    /// Full logical KV length after the newly scheduled input is written.
    pub(crate) kv_length: usize,
    pub(crate) block_table: Vec<usize>,
    pub(crate) slot_mapping: Vec<i64>,
}

/// Result returned by EngineCore for exactly one [`StepPlan`].
#[derive(Debug, Clone)]
pub(crate) struct StepResult {
    pub(crate) plan_id: u64,
    pub(crate) sequences: Vec<SequenceStepResult>,
}

/// Execution result for one planned sequence.
#[derive(Debug, Clone)]
pub(crate) struct SequenceStepResult {
    pub(crate) request_id: usize,
    pub(crate) sequence_id: usize,
    pub(crate) sampled_token: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StepPlanError {
    Invalid(String),
    PlanAlreadyInFlight {
        plan_id: u64,
    },
    NoPlanInFlight {
        result_plan_id: u64,
    },
    StaleResult {
        expected_plan_id: u64,
        result_plan_id: u64,
    },
    ResultMismatch {
        plan_id: u64,
        reason: String,
    },
    Cache {
        operation: CacheOperation,
        sequence_id: usize,
        source: KvCacheError,
    },
    NoProgress {
        waiting_sequences: usize,
        running_sequences: usize,
        token_budget: usize,
        free_blocks: usize,
        blocked_admission: Option<BlockedAdmission>,
    },
}

impl StepPlanError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    pub(crate) fn cache(
        operation: CacheOperation,
        sequence_id: usize,
        source: KvCacheError,
    ) -> Self {
        Self::Cache {
            operation,
            sequence_id,
            source,
        }
    }
}

impl std::fmt::Display for StepPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::PlanAlreadyInFlight { plan_id } => {
                write!(f, "step plan {plan_id} is still in flight")
            }
            Self::NoPlanInFlight { result_plan_id } => {
                write!(f, "step result {result_plan_id} has no plan in flight")
            }
            Self::StaleResult {
                expected_plan_id,
                result_plan_id,
            } => write!(
                f,
                "stale step result {result_plan_id}; expected plan {expected_plan_id}"
            ),
            Self::ResultMismatch { plan_id, reason } => {
                write!(f, "step result {plan_id} does not match its plan: {reason}")
            }
            Self::Cache {
                operation,
                sequence_id,
                source,
            } => write!(
                f,
                "cache {operation} failed for sequence {sequence_id}: {source}"
            ),
            Self::NoProgress {
                waiting_sequences,
                running_sequences,
                token_budget,
                free_blocks,
                blocked_admission,
            } => {
                write!(
                    f,
                    "scheduler made no progress: waiting={waiting_sequences}, running={running_sequences}, token_budget={token_budget}, free_blocks={free_blocks}"
                )?;
                if let Some(blocked) = blocked_admission {
                    write!(
                        f,
                        ", blocked_admission=request {} ({})",
                        blocked.request_id, blocked.reason
                    )?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for StepPlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cache { source, .. } => Some(source),
            _ => None,
        }
    }
}
