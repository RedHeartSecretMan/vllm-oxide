//! Record-only direct NVML memory method, with unchanged strict 50 ms completion gaps.
//! The worker owns NVML. A bounded data-only snapshot remains readable before join.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::measurement::MeasurementIdentity;
use anyhow::{bail, Context, Result};
use serde::Serialize;

use super::{observed_interval, MemoryMonitorEvidence, MemorySample};
#[path = "nvml_ffi.rs"]
mod nvml_ffi;
use nvml_ffi::{MemoryBytes, Reply, SamplingApi};

const INTERVAL: Duration = Duration::from_millis(5);
const LIMIT: Duration = Duration::from_millis(50);
const SAMPLE_LIMIT: usize = 20_000;

#[derive(Clone, Serialize)]
struct RawSample {
    api_enter_ns: u128,
    api_return_ns: Option<u128>,
    api_wall_span_ns: Option<u128>,
    observed_ns: Option<u128>,
    return_code: Option<i32>,
    bytes: Option<MemoryBytes>,
    error: Option<String>,
    final_sample: bool,
}

#[derive(Clone, Default, Serialize)]
struct Trace {
    library: Option<serde_json::Value>,
    raw_samples: Vec<RawSample>,
    projected_samples: Vec<MemorySample>,
    baseline_observed_ns: Option<u128>,
    maximum_gap_ns: u128,
    finish_requested_ns: Option<u128>,
    final_sample_enter_ns: Option<u128>,
    final_sample_return_ns: Option<u128>,
    final_sample_observed_ns: Option<u128>,
    worker_finished_ns: Option<u128>,
    primary_error: Option<String>,
    shutdown_error: Option<String>,
    shutdown_completed: bool,
    shutdown_enter_ns: Option<u128>,
    shutdown_return_ns: Option<u128>,
}

type SharedTrace = Arc<Mutex<Trace>>;

fn trace_lock(shared: &SharedTrace) -> MutexGuard<'_, Trace> {
    match shared.lock() {
        Ok(trace) => trace,
        Err(poisoned) => {
            let mut trace = poisoned.into_inner();
            trace
                .primary_error
                .get_or_insert_with(|| "NVML data snapshot mutex was poisoned".into());
            trace
        }
    }
}

fn duration(ns: u128) -> Result<Duration> {
    Ok(Duration::from_nanos(u64::try_from(ns)?))
}

fn validate_memory(
    code: i32,
    memory: Option<MemoryBytes>,
    error: Option<&str>,
) -> Result<MemoryBytes> {
    if code != 0 {
        bail!(
            "NVML query return code {code}: {}",
            error.unwrap_or("missing error text")
        );
    }
    let memory = memory.context("successful NVML query omitted memory bytes")?;
    if memory.total == 0 || memory.free.checked_add(memory.used) != Some(memory.total) {
        bail!("invalid NVML memory accounting");
    }
    Ok(memory)
}

fn retain_sample(trace: &mut Trace, sample: RawSample) -> Result<()> {
    let last = trace
        .raw_samples
        .last()
        .and_then(|sample| sample.observed_ns);
    let observed = sample.observed_ns;
    let span = sample.api_wall_span_ns;
    let code = sample.return_code;
    let memory = sample.bytes;
    let error = sample.error.clone();
    trace.raw_samples.push(sample);
    if let (Some(last), Some(observed)) = (last, observed) {
        let gap = observed
            .checked_sub(last)
            .context("nonmonotonic NVML sample")?;
        trace.maximum_gap_ns = trace.maximum_gap_ns.max(gap);
        observed_interval(duration(gap)?)?;
    }
    // Additional conservative method condition; ADR-0012's primary rule is completion gap.
    if duration(span.context("NVML call has no return/span")?)? > LIMIT {
        bail!("NVML API wall span exceeds 50 ms");
    }
    let memory = validate_memory(
        code.context("NVML call has no return code")?,
        memory,
        error.as_deref(),
    )?;
    let observed = observed.context("valid NVML sample omitted observation point")?;
    let baseline = *trace.baseline_observed_ns.get_or_insert(observed);
    trace.projected_samples.push(MemorySample {
        elapsed_ms: u64::try_from((observed - baseline) / 1_000_000)?,
        // Explicit record projection. Original bytes remain authoritative in this sidecar.
        used_mib: memory.used / (1024 * 1024),
    });
    Ok(())
}

fn sample_loop(
    api: &mut dyn SamplingApi,
    shared: &SharedTrace,
    origin: Instant,
    stop: &Receiver<u128>,
    ready: &SyncSender<std::result::Result<(), String>>,
    ready_sent: &mut bool,
    sample_limit: usize,
) -> Result<()> {
    // No FFI/provider call runs while the data snapshot mutex is held.
    api.initialize()?;
    trace_lock(shared).library = Some(api.metadata());
    let mut next = Instant::now();
    let mut finish_requested = None;
    loop {
        if finish_requested.is_none() {
            match stop.recv_timeout(next.saturating_duration_since(Instant::now())) {
                Ok(requested) => finish_requested = Some(requested),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => bail!("NVML stop sender disappeared"),
            }
        }
        if let Some(delay) = next.checked_duration_since(Instant::now()) {
            thread::sleep(delay);
        }
        if finish_requested.is_none() {
            if let Ok(requested) = stop.try_recv() {
                finish_requested = Some(requested);
            }
        }
        let (enter, final_sample) = {
            let mut trace = trace_lock(shared);
            trace.finish_requested_ns = finish_requested;
            if trace.raw_samples.len() == sample_limit {
                bail!("bounded NVML raw sample storage exhausted");
            }
            let enter = origin.elapsed().as_nanos();
            let final_sample = finish_requested.is_some_and(|requested| enter >= requested);
            // Register the real enter BEFORE the call. Unknown return/code/bytes stay NULL.
            trace.raw_samples.push(RawSample {
                api_enter_ns: enter,
                api_return_ns: None,
                api_wall_span_ns: None,
                observed_ns: None,
                return_code: None,
                bytes: None,
                error: None,
                final_sample,
            });
            (enter, final_sample)
        };
        let Reply {
            code,
            memory,
            error,
        } = api.query();
        let returned_at = Instant::now();
        let returned = returned_at.duration_since(origin).as_nanos();
        let validated = validate_memory(code, memory, error.as_deref());
        let observed_at = validated.as_ref().ok().map(|_| Instant::now());
        let observed = observed_at.map(|instant| instant.duration_since(origin).as_nanos());
        {
            let mut trace = trace_lock(shared);
            trace
                .raw_samples
                .pop()
                .context("NVML pending enter record missing")?;
            retain_sample(
                &mut trace,
                RawSample {
                    api_enter_ns: enter,
                    api_return_ns: Some(returned),
                    api_wall_span_ns: Some(returned - enter),
                    observed_ns: observed,
                    return_code: Some(code),
                    bytes: memory,
                    error,
                    final_sample,
                },
            )?;
            if final_sample {
                trace.final_sample_enter_ns = Some(enter);
                trace.final_sample_return_ns = Some(returned);
                trace.final_sample_observed_ns = observed;
            }
        }
        if !*ready_sent {
            ready
                .send(Ok(()))
                .context("NVML baseline receiver disappeared")?;
            *ready_sent = true;
        }
        if final_sample {
            return Ok(());
        }
        next = observed_at.context("valid NVML sample omitted scheduling observation")? + INTERVAL;
    }
}

fn worker<F>(
    factory: F,
    origin: Instant,
    stop: Receiver<u128>,
    ready: SyncSender<std::result::Result<(), String>>,
    sample_limit: usize,
    shared: SharedTrace,
) -> Trace
where
    F: FnOnce() -> Result<Box<dyn SamplingApi>>,
{
    let mut ready_sent = false;
    let mut api = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(factory)) {
        Ok(Ok(api)) => api,
        result => {
            let error = match result {
                Ok(Err(error)) => format!("{error:#}"),
                _ => "NVML factory panicked".into(),
            };
            trace_lock(&shared).primary_error = Some(error.clone());
            let _ = ready.send(Err(error));
            return trace_lock(&shared).clone();
        }
    };
    trace_lock(&shared).library = Some(api.metadata());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sample_loop(
            api.as_mut(),
            &shared,
            origin,
            &stop,
            &ready,
            &mut ready_sent,
            sample_limit,
        )
    }));
    trace_lock(&shared).primary_error = match result {
        Ok(result) => result.err().map(|error| format!("{error:#}")),
        Err(_) => Some("NVML sampling worker panicked".into()),
    };
    trace_lock(&shared).shutdown_enter_ns = Some(origin.elapsed().as_nanos());
    // The worker has finished query/unwind before shutdown; never hold the snapshot lock here.
    let closed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| api.close()));
    {
        let mut trace = trace_lock(&shared);
        trace.shutdown_return_ns = Some(origin.elapsed().as_nanos());
        match closed {
            Ok(Ok(())) => trace.shutdown_completed = true,
            Ok(Err(error)) => trace.shutdown_error = Some(format!("{error:#}")),
            Err(_) => {
                trace.shutdown_return_ns = None;
                trace.shutdown_error = Some("NVML shutdown panicked".into());
            }
        }
        trace.worker_finished_ns = Some(origin.elapsed().as_nanos());
    }
    let trace = trace_lock(&shared).clone();
    if !ready_sent {
        let _ = ready.send(Err(trace
            .primary_error
            .clone()
            .unwrap_or_else(|| "NVML baseline missing".into())));
    }
    trace
}

pub(super) struct ActiveMemoryMonitor {
    directory: PathBuf,
    call_id: String,
    source: MeasurementIdentity,
    origin: Instant,
    times: BTreeMap<&'static str, u128>,
    stop: Option<SyncSender<u128>>,
    worker: Option<JoinHandle<Trace>>,
    done: Receiver<()>,
    join_timeout: Duration,
    trace: Option<Trace>,
    shared: SharedTrace,
    primary_failure: Option<String>,
    joined: bool,
    saved: bool,
}

impl ActiveMemoryMonitor {
    pub(super) fn start(
        output: &Path,
        call_id: &str,
        origin: Instant,
        source: &MeasurementIdentity,
    ) -> Result<Self> {
        Self::start_with_factory(
            output,
            call_id,
            origin,
            source,
            nvml_ffi::open,
            SAMPLE_LIMIT,
        )
    }

    fn start_with_factory<F>(
        output: &Path,
        call_id: &str,
        origin: Instant,
        source: &MeasurementIdentity,
        factory: F,
        sample_limit: usize,
    ) -> Result<Self>
    where
        F: FnOnce() -> Result<Box<dyn SamplingApi>> + Send + 'static,
    {
        Self::start_with_schedule(
            output,
            call_id,
            origin,
            source,
            factory,
            sample_limit,
            || {},
        )
    }

    fn start_with_schedule<F, H>(
        output: &Path,
        call_id: &str,
        origin: Instant,
        source: &MeasurementIdentity,
        factory: F,
        sample_limit: usize,
        after_spawn: H,
    ) -> Result<Self>
    where
        F: FnOnce() -> Result<Box<dyn SamplingApi>> + Send + 'static,
        H: FnOnce(),
    {
        let directory = output.join(format!("{call_id}.memory-method"));
        std::fs::create_dir(&directory)
            .context("creating fresh direct NVML memory record directory")?;
        let (stop_sender, stop) = mpsc::sync_channel(1);
        let (ready, baseline) = mpsc::sync_channel(1);
        let (acknowledge, done) = mpsc::sync_channel(1);
        let shared = Arc::new(Mutex::new(Trace {
            raw_samples: Vec::with_capacity(sample_limit),
            projected_samples: Vec::with_capacity(sample_limit),
            ..Trace::default()
        }));
        let worker_shared = Arc::clone(&shared);
        // Capture the real shared-clock anchor before the worker can initialize/query.
        let monitor_started_ns = origin.elapsed().as_nanos();
        let worker = thread::Builder::new()
            .name("direct-nvml-monitor".into())
            .spawn(move || {
                let trace = worker(factory, origin, stop, ready, sample_limit, worker_shared);
                let _ = acknowledge.send(());
                trace
            })?;
        after_spawn();
        let mut monitor = Self {
            directory,
            call_id: call_id.to_owned(),
            source: source.clone(),
            origin,
            times: BTreeMap::from([("post_initialization_monitor_started_ns", monitor_started_ns)]),
            stop: Some(stop_sender),
            worker: Some(worker),
            done,
            join_timeout: Duration::from_secs(5),
            trace: None,
            shared,
            primary_failure: None,
            joined: false,
            saved: false,
        };
        let result = baseline
            .recv_timeout(Duration::from_secs(5))
            .context("NVML monitor did not produce a post-initialization baseline")
            .and_then(|value| value.map_err(|error| anyhow::anyhow!(error)));
        if let Err(error) = result {
            monitor.primary_failure = Some(format!("{error:#}"));
            monitor.close_and_join();
            return monitor.save_result(Err(error)).map(|_| monitor);
        }
        monitor.mark("baseline_ready_ns");
        Ok(monitor)
    }

    pub(super) fn mark(&mut self, event: &'static str) {
        self.times.insert(event, self.origin.elapsed().as_nanos());
    }

    fn request_stop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let requested = self.origin.elapsed().as_nanos();
            self.times.insert("finish_requested_ns", requested);
            let _ = stop.send(requested); // An already-failed worker is still joined below.
        }
    }

    fn close_and_join(&mut self) {
        self.request_stop();
        if let Some(worker) = self.worker.take() {
            if self.done.recv_timeout(self.join_timeout).is_err() {
                self.primary_failure.get_or_insert_with(|| "NVML worker completion acknowledgement deadline exceeded; external owner guard must bound join".into());
                self.mark("worker_ack_timeout_ns");
                let timeout = self.document();
                if let Err(error) = self.write("worker-timeout.json", &timeout) {
                    eprintln!("NVML_MEMORY_RECORD_TIMEOUT_PRESERVATION_FAILURE {error:#}");
                }
                // Never unload or detach a live API. The external process guard bounds a
                // genuinely stuck join; it must terminate this owner before any next owner.
            }
            match worker.join() {
                Ok(trace) => self.trace = Some(trace),
                Err(_) => {
                    self.primary_failure
                        .get_or_insert_with(|| "NVML worker join panicked".into());
                }
            };
            self.joined = true;
            self.mark("worker_join_completed_ns");
        }
    }

    fn evidence(&self) -> Result<MemoryMonitorEvidence> {
        if let Some(error) = &self.primary_failure {
            bail!("{error}");
        }
        let trace = self.trace.as_ref().context("NVML worker trace missing")?;
        if let Some(error) = &trace.primary_error {
            bail!("NVML sampler: {error}");
        }
        if let Some(error) = &trace.shutdown_error {
            bail!("NVML shutdown: {error}");
        }
        if !trace.shutdown_completed || !self.joined {
            bail!("NVML worker cleanup incomplete");
        }
        let requested = self
            .times
            .get("finish_requested_ns")
            .context("NVML finish request missing")?;
        if trace.finish_requested_ns != Some(*requested)
            || trace
                .final_sample_enter_ns
                .is_none_or(|enter| enter < *requested)
            || trace.final_sample_observed_ns.is_none()
        {
            bail!("NVML final query does not cover the final synchronization");
        }
        let interval = observed_interval(duration(trace.maximum_gap_ns)?)?;
        MemoryMonitorEvidence::validate(interval, false, &[], trace.projected_samples.clone())
    }

    pub(super) fn finish(mut self) -> Result<MemoryMonitorEvidence> {
        self.close_and_join();
        let result = self.evidence();
        self.primary_failure = result.as_ref().err().map(|error| format!("{error:#}"));
        self.save_result(result)
    }

    fn write(&self, name: &str, value: &serde_json::Value) -> Result<()> {
        let path = self.directory.join(name);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("creating {}", path.display()))?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        File::open(&self.directory)?.sync_all()?;
        Ok(())
    }

    fn document(&self) -> serde_json::Value {
        // Clone only bounded data under the mutex; serialize/fsync after releasing it.
        let trace = self
            .trace
            .clone()
            .unwrap_or_else(|| trace_lock(&self.shared).clone());
        serde_json::json!({"protocol":"memory-method-record-v1","schema_version":1,"method":"direct-nvml-v1","accepting":false,"record_only":true,
            "call_id":self.call_id,"source":{"commit":self.source.commit,"tree":self.source.tree},
            "benchmark_pid":std::process::id(),"build_source_id":env!("VLLM_OXIDE_BUILD_SOURCE_ID"),
            "clock":"same benchmark-stage Rust Instant origin; elapsed nanoseconds","times":self.times,
            "completion_observation":"observed_ns after successful API return and memory validation",
            "worker_ack_timeout_seconds":self.join_timeout.as_secs_f64(),
            "sampling_method":"direct nvmlDeviceGetMemoryInfo, device0; no CLI/Python sampler",
            "sampling_target_ms":5,"sampling_schedule":"previous validated observation + 5ms; no catch-up","completion_gap_limit_ns":50_000_000,"additional_api_wall_span_limit_ns":50_000_000,
            "projection":"used_mib=floor(used_bytes/2^20); raw bytes retained; not old native parser evidence",
            "raw_sample_limit":SAMPLE_LIMIT,"primary_failure":self.primary_failure,"worker_joined":self.joined,"trace":trace})
    }

    fn save_result(
        &mut self,
        result: Result<MemoryMonitorEvidence>,
    ) -> Result<MemoryMonitorEvidence> {
        match self.write("record.json", &self.document()) {
            Ok(()) => {
                self.saved = true;
                result
            }
            Err(secondary) => {
                let preservation = serde_json::json!({"accepting":false,"record_only":true,"method":"direct-nvml-v1",
                    "primary_failure":self.primary_failure,"secondary_preservation_failure":format!("{secondary:#}"),
                    "full_partial_record":self.document()});
                let fallback = self.write("preservation-error.json", &preservation);
                self.saved = true;
                let detail = format!("memory record preservation secondary failure: {secondary:#}; fallback={fallback:?}");
                eprintln!(
                    "NVML_MEMORY_RECORD_PRESERVATION_FAILURE primary={:?}; {detail}",
                    self.primary_failure
                );
                match result {
                    Err(primary) => Err(primary.context(detail)),
                    Ok(_) => Err(secondary.context(detail)),
                }
            }
        }
    }
}

impl Drop for ActiveMemoryMonitor {
    fn drop(&mut self) {
        if !self.saved {
            self.primary_failure
                .get_or_insert_with(|| "NVML monitor dropped before finish".into());
            self.close_and_join();
            let error = anyhow::anyhow!(self.primary_failure.clone().unwrap_or_default());
            let _ = self.save_result(Err(error));
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)] // Deliberate worker faults and assertion fixtures.
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::process::Command;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    enum Action {
        Good,
        Error,
        Slow,
        Panic,
        Blocked,
        SignalEntered(SyncSender<()>),
        Held(SyncSender<()>, Receiver<()>),
    }

    struct FakeApi {
        actions: VecDeque<Action>,
        closed: Arc<AtomicUsize>,
        init_error: bool,
        close_error: bool,
    }

    impl SamplingApi for FakeApi {
        fn metadata(&self) -> serde_json::Value {
            serde_json::json!({"synthetic_cpu_only":true})
        }
        fn initialize(&mut self) -> Result<()> {
            if self.init_error {
                bail!("synthetic initialization error");
            }
            Ok(())
        }
        fn query(&mut self) -> Reply {
            match self.actions.pop_front().unwrap_or(Action::Good) {
                Action::Error => {
                    return Reply {
                        code: 3,
                        memory: None,
                        error: Some("synthetic unsupported API".into()),
                    }
                }
                Action::Slow => thread::sleep(Duration::from_millis(80)),
                Action::Panic => panic!("synthetic worker fault"),
                Action::Blocked => loop {
                    thread::park();
                },
                Action::SignalEntered(entered) => {
                    entered.send(()).unwrap();
                }
                Action::Held(entered, release) => {
                    entered.send(()).unwrap();
                    release.recv().unwrap();
                }
                Action::Good => {}
            }
            Reply {
                code: 0,
                memory: Some(MemoryBytes {
                    total: 8 * 1024 * 1024,
                    free: 6 * 1024 * 1024 - 999,
                    used: 2 * 1024 * 1024 + 999,
                }),
                error: None,
            }
        }
        fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
            if std::env::var("OXIDE_NVML_CPU_BLOCK").as_deref() == Ok("shutdown") {
                loop {
                    thread::park();
                }
            }
            if self.close_error {
                bail!("synthetic shutdown error");
            }
            Ok(())
        }
    }

    fn start(
        directory: &Path,
        actions: Vec<Action>,
        closed: Arc<AtomicUsize>,
        init_error: bool,
        close_error: bool,
        limit: usize,
    ) -> Result<ActiveMemoryMonitor> {
        ActiveMemoryMonitor::start_with_factory(
            directory,
            "test",
            Instant::now(),
            &MeasurementIdentity {
                commit: "0".repeat(40),
                tree: "1".repeat(40),
            },
            move || {
                Ok(Box::new(FakeApi {
                    actions: actions.into(),
                    closed,
                    init_error,
                    close_error,
                }))
            },
            limit,
        )
    }

    fn saved(directory: &Path) -> serde_json::Value {
        serde_json::from_slice(
            &std::fs::read(directory.join("test.memory-method/record.json")).unwrap(),
        )
        .unwrap()
    }

    fn raw(enter: u128, returned: u128) -> RawSample {
        RawSample {
            api_enter_ns: enter,
            api_return_ns: Some(returned),
            api_wall_span_ns: Some(returned - enter),
            observed_ns: Some(returned),
            return_code: Some(0),
            bytes: Some(MemoryBytes {
                total: 100,
                free: 90,
                used: 10,
            }),
            error: None,
            final_sample: false,
        }
    }

    #[test]
    fn nvml_nanoseconds_reject_fifty_point_zero_zero_one_before_projection() {
        let mut trace = Trace::default();
        retain_sample(&mut trace, raw(1, 10)).unwrap();
        retain_sample(&mut trace, raw(49_000_010, 50_000_010)).unwrap();
        assert_eq!(trace.projected_samples[1].elapsed_ms, 50);
        assert!(retain_sample(&mut trace, raw(99_000_011, 100_000_011)).is_err());
        assert_eq!(trace.raw_samples.len(), 3); // Invalid sample remains in raw evidence.
        assert_eq!(trace.maximum_gap_ns, 50_000_001);
        assert_eq!(trace.projected_samples.len(), 2);
    }

    #[test]
    fn nvml_same_millisecond_projection_cannot_be_accepted_as_complete_evidence() {
        let mut trace = Trace::default();
        retain_sample(&mut trace, raw(1, 100)).unwrap();
        retain_sample(&mut trace, raw(200, 900_000)).unwrap();
        assert_eq!(trace.projected_samples[1].elapsed_ms, 0);
        assert!(MemoryMonitorEvidence::validate(1, false, &[], trace.projected_samples).is_err());
    }

    #[test]
    fn nvml_immediate_finish_waits_for_real_distinct_final_sample() {
        let directory = tempfile::tempdir().unwrap();
        let closed = Arc::new(AtomicUsize::new(0));
        let monitor = start(directory.path(), vec![], closed.clone(), false, false, 20).unwrap();
        let result = monitor.finish().unwrap();
        assert!(result.sample_count >= 2);
        assert_eq!(result.baseline_mib, 2); // +999 bytes are retained, then explicitly floored.
        assert!(result.samples[1].elapsed_ms >= 5);
        let data = saved(directory.path());
        let requested = data["times"]["finish_requested_ns"].as_u64().unwrap();
        assert!(data["trace"]["final_sample_enter_ns"].as_u64().unwrap() >= requested);
        assert_eq!(
            data["trace"]["raw_samples"][0]["bytes"]["used"],
            2 * 1024 * 1024 + 999
        );
        assert_eq!(closed.load(Ordering::SeqCst), 1);
        assert_eq!(data["worker_joined"], true);
        assert_eq!(data["trace"]["shutdown_completed"], true);
    }

    #[test]
    fn nvml_record_contract_names_method_and_observation_based_five_ms_target() {
        let directory = tempfile::tempdir().unwrap();
        let monitor = start(
            directory.path(),
            vec![],
            Arc::new(AtomicUsize::new(0)),
            false,
            false,
            20,
        )
        .unwrap();
        monitor.finish().unwrap();
        let data = saved(directory.path());
        assert_eq!(data["protocol"], "memory-method-record-v1");
        assert_eq!(data["schema_version"], 1);
        assert_eq!(data["method"], "direct-nvml-v1");
        assert_eq!(data["record_only"], true);
        assert_eq!(data["accepting"], false);
        assert_eq!(data["sampling_target_ms"], 5);
        assert_eq!(
            data["sampling_schedule"],
            "previous validated observation + 5ms; no catch-up"
        );
        let raw = data["trace"]["raw_samples"].as_array().unwrap();
        assert!(
            raw[1]["api_enter_ns"].as_u64().unwrap()
                >= raw[0]["observed_ns"].as_u64().unwrap() + 5_000_000
        );
    }

    #[test]
    fn nvml_start_anchor_precedes_first_query_when_parent_runs_after_worker() {
        let directory = tempfile::tempdir().unwrap();
        let (entered, observed) = mpsc::sync_channel(1);
        let closed = Arc::new(AtomicUsize::new(0));
        let api_closed = Arc::clone(&closed);
        let monitor = ActiveMemoryMonitor::start_with_schedule(
            directory.path(),
            "test",
            Instant::now(),
            &MeasurementIdentity {
                commit: "0".repeat(40),
                tree: "1".repeat(40),
            },
            move || {
                Ok(Box::new(FakeApi {
                    actions: vec![Action::SignalEntered(entered)].into(),
                    closed: api_closed,
                    init_error: false,
                    close_error: false,
                }))
            },
            20,
            // Deterministically realize the legal worker-first schedule, with no timestamp edits.
            move || {
                observed.recv_timeout(Duration::from_secs(1)).unwrap();
            },
        )
        .unwrap();
        monitor.finish().unwrap();
        let data = saved(directory.path());
        let start = data["times"]["post_initialization_monitor_started_ns"]
            .as_u64()
            .unwrap();
        let first_enter = data["trace"]["raw_samples"][0]["api_enter_ns"]
            .as_u64()
            .unwrap();
        assert!(start <= first_enter, "baseline sampling starts before post-initialization monitor: start={start}, enter={first_enter}");
        assert_eq!(closed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn nvml_inflight_return_after_stop_is_not_the_final_query() {
        let directory = tempfile::tempdir().unwrap();
        let closed = Arc::new(AtomicUsize::new(0));
        let (entered, pending) = mpsc::sync_channel(1);
        let (release, gate) = mpsc::sync_channel(1);
        let mut monitor = start(
            directory.path(),
            vec![Action::Good, Action::Held(entered, gate)],
            closed.clone(),
            false,
            false,
            20,
        )
        .unwrap();
        pending.recv_timeout(Duration::from_secs(1)).unwrap();
        monitor.request_stop();
        release.send(()).unwrap();
        monitor.finish().unwrap();
        let data = saved(directory.path());
        let samples = data["trace"]["raw_samples"].as_array().unwrap();
        assert!(!samples[1]["final_sample"].as_bool().unwrap());
        assert!(
            samples[1]["api_return_ns"].as_u64().unwrap()
                >= data["times"]["finish_requested_ns"].as_u64().unwrap()
        );
        assert!(samples.last().unwrap()["final_sample"].as_bool().unwrap());
        assert_eq!(closed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn nvml_actual_eighty_ms_call_preserves_partial_raw_and_fails_closed() {
        let directory = tempfile::tempdir().unwrap();
        let closed = Arc::new(AtomicUsize::new(0));
        let monitor = start(
            directory.path(),
            vec![Action::Good, Action::Slow],
            closed.clone(),
            false,
            false,
            20,
        )
        .unwrap();
        let error = monitor.finish().unwrap_err();
        assert!(format!("{error:#}").contains("at most 50 ms"));
        let data = saved(directory.path());
        assert!(data["trace"]["maximum_gap_ns"].as_u64().unwrap() > 50_000_000);
        assert!(
            data["trace"]["raw_samples"][1]["api_wall_span_ns"]
                .as_u64()
                .unwrap()
                >= 80_000_000
        );
        assert_eq!(
            data["trace"]["projected_samples"].as_array().unwrap().len(),
            1
        );
        assert_eq!(closed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn nvml_baseline_initialization_and_api_errors_are_retained() {
        for init_error in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let closed = Arc::new(AtomicUsize::new(0));
            assert!(start(
                directory.path(),
                vec![Action::Error],
                closed.clone(),
                init_error,
                false,
                20
            )
            .is_err());
            let data = saved(directory.path());
            assert!(data["trace"]["primary_error"].is_string());
            assert!(data["primary_failure"].is_string());
            assert_eq!(data["worker_joined"], true);
            assert_eq!(closed.load(Ordering::SeqCst), 1);
            if !init_error {
                assert_eq!(data["trace"]["raw_samples"][0]["return_code"], 3);
            }
        }
    }

    #[test]
    fn nvml_worker_api_error_and_panic_keep_baseline_and_join_shutdown() {
        for action in [Action::Error, Action::Panic] {
            let directory = tempfile::tempdir().unwrap();
            let closed = Arc::new(AtomicUsize::new(0));
            let monitor = start(
                directory.path(),
                vec![Action::Good, action],
                closed.clone(),
                false,
                false,
                20,
            )
            .unwrap();
            assert!(monitor.finish().is_err());
            let data = saved(directory.path());
            let raw = data["trace"]["raw_samples"].as_array().unwrap();
            assert_eq!(raw.len(), 2, "worker fault lost the entered API call");
            if raw[1]["return_code"].is_null() {
                assert!(raw[1]["api_enter_ns"].is_number());
                assert!(raw[1]["api_return_ns"].is_null());
                assert!(raw[1]["observed_ns"].is_null());
            }
            assert!(data["trace"]["primary_error"].is_string());
            assert_eq!(data["worker_joined"], true);
            assert_eq!(data["trace"]["shutdown_completed"], true);
            assert_eq!(closed.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn nvml_shutdown_failure_is_not_a_successful_measurement() {
        let directory = tempfile::tempdir().unwrap();
        let closed = Arc::new(AtomicUsize::new(0));
        let monitor = start(directory.path(), vec![], closed.clone(), false, true, 20).unwrap();
        assert!(monitor
            .finish()
            .unwrap_err()
            .to_string()
            .contains("synthetic shutdown error"));
        let data = saved(directory.path());
        assert_eq!(data["trace"]["shutdown_completed"], false);
        assert_eq!(data["worker_joined"], true);
        assert_eq!(closed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn nvml_drop_aborts_and_joins_instead_of_detaching_sampler() {
        let directory = tempfile::tempdir().unwrap();
        let closed = Arc::new(AtomicUsize::new(0));
        let monitor = start(directory.path(), vec![], closed.clone(), false, false, 20).unwrap();
        drop(monitor);
        let data = saved(directory.path());
        assert_eq!(
            data["primary_failure"],
            "NVML monitor dropped before finish"
        );
        assert_eq!(data["worker_joined"], true);
        assert_eq!(closed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn nvml_late_completion_ack_remains_invalid_but_worker_is_joined() {
        let directory = tempfile::tempdir().unwrap();
        let closed = Arc::new(AtomicUsize::new(0));
        let mut monitor =
            start(directory.path(), vec![], closed.clone(), false, false, 20).unwrap();
        // Final sample's normal target is 5ms; force only the CPU fixture's ack deadline.
        monitor.join_timeout = Duration::from_millis(1);
        assert!(monitor
            .finish()
            .unwrap_err()
            .to_string()
            .contains("acknowledgement deadline"));
        let data = saved(directory.path());
        assert!(directory
            .path()
            .join("test.memory-method/worker-timeout.json")
            .exists());
        assert_eq!(data["worker_joined"], true);
        assert_eq!(data["trace"]["shutdown_completed"], true);
        assert_eq!(closed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn nvml_storage_limit_invalidates_without_losing_collected_samples() {
        let directory = tempfile::tempdir().unwrap();
        let closed = Arc::new(AtomicUsize::new(0));
        let monitor = start(directory.path(), vec![], closed.clone(), false, false, 1).unwrap();
        assert!(monitor
            .finish()
            .unwrap_err()
            .to_string()
            .contains("storage exhausted"));
        let data = saved(directory.path());
        assert_eq!(data["trace"]["raw_samples"].as_array().unwrap().len(), 1);
        assert_eq!(closed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn nvml_preservation_failure_keeps_primary_and_complete_partial_trace() {
        let directory = tempfile::tempdir().unwrap();
        let closed = Arc::new(AtomicUsize::new(0));
        let monitor = start(
            directory.path(),
            vec![Action::Good, Action::Slow],
            closed.clone(),
            false,
            false,
            20,
        )
        .unwrap();
        let output = directory.path().join("test.memory-method");
        std::fs::write(output.join("record.json"), b"preexisting bytes\n").unwrap();
        let error = format!("{:#}", monitor.finish().unwrap_err());
        assert!(error.contains("at most 50 ms"));
        assert!(error.contains("preservation secondary failure"));
        assert_eq!(
            std::fs::read(output.join("record.json")).unwrap(),
            b"preexisting bytes\n"
        );
        let data: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output.join("preservation-error.json")).unwrap())
                .unwrap();
        assert!(data["primary_failure"]
            .as_str()
            .unwrap()
            .contains("at most 50 ms"));
        assert!(
            data["full_partial_record"]["trace"]["raw_samples"][1]["api_wall_span_ns"]
                .as_u64()
                .unwrap()
                >= 80_000_000
        );
        assert_eq!(closed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn nvml_v1_memory_layout_matches_verified_official_abi() {
        assert_eq!(std::mem::size_of::<MemoryBytes>(), 24);
        assert_eq!(std::mem::align_of::<MemoryBytes>(), 8);
        assert_eq!(std::mem::offset_of!(MemoryBytes, total), 0);
        assert_eq!(std::mem::offset_of!(MemoryBytes, free), 8);
        assert_eq!(std::mem::offset_of!(MemoryBytes, used), 16);
    }

    fn permanently_blocked_fixture(mode: &str, test_name: &str) {
        if std::env::var("OXIDE_NVML_CPU_BLOCK").as_deref() == Ok(mode) {
            let output = PathBuf::from(std::env::var("OXIDE_NVML_CPU_DIR").unwrap());
            let actions = if mode == "query" {
                vec![Action::Good, Action::Blocked]
            } else {
                vec![]
            };
            let mut monitor = start(
                &output,
                actions,
                Arc::new(AtomicUsize::new(0)),
                false,
                false,
                20,
            )
            .unwrap();
            monitor.join_timeout = Duration::from_millis(20);
            let _ = monitor.finish();
            panic!("permanently blocked worker unexpectedly returned");
        }
        let temporary = tempfile::tempdir().unwrap();
        let output = match std::env::var("OXIDE_NVML_CPU_FIXTURE_ARTIFACTS") {
            Ok(root) => {
                let output = PathBuf::from(root).join(mode);
                std::fs::create_dir_all(&output).unwrap();
                output
            }
            Err(_) => temporary.path().to_owned(),
        };
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([test_name, "--exact", "--nocapture", "--test-threads=1"])
            .env("OXIDE_NVML_CPU_BLOCK", mode)
            .env("OXIDE_NVML_CPU_DIR", &output)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        let path = output.join("test.memory-method/worker-timeout.json");
        let mut snapshot = None;
        while Instant::now() < deadline {
            if let Ok(bytes) = std::fs::read(&path) {
                if let Ok(data) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    snapshot = Some(data);
                    break;
                }
            }
            thread::sleep(Duration::from_millis(2));
        }
        // The blocked child is always killed/reaped before any assertion can panic.
        let pid = child.id();
        child.kill().unwrap();
        let status = child.wait().unwrap();
        println!(
            "NVML_CPU_PERMANENT_BLOCK mode={mode} pid={pid} status={status} snapshot={}",
            path.display()
        );
        assert!(!PathBuf::from(format!("/proc/{pid}")).exists());
        let data = snapshot
            .unwrap_or_else(|| panic!("blocked worker failed to save its finite timeout marker"));
        assert!(
            data["trace"].is_object(),
            "timeout lost the worker's completed baseline/raw"
        );
        let raw = data["trace"]["raw_samples"].as_array().unwrap();
        assert!(!raw.is_empty(), "timeout lost the baseline sample");
        assert_eq!(data["worker_joined"], false);
        assert_eq!(data["trace"]["shutdown_completed"], false);
        if mode == "query" {
            assert!(raw.last().unwrap()["api_enter_ns"].is_number());
            assert!(raw.last().unwrap()["api_return_ns"].is_null());
            assert!(raw.last().unwrap()["observed_ns"].is_null());
        } else {
            assert!(data["trace"]["shutdown_enter_ns"].is_number());
            assert!(data["trace"]["shutdown_return_ns"].is_null());
            assert!(raw.last().unwrap()["final_sample"].as_bool().unwrap());
        }
    }

    #[test]
    fn nvml_permanent_query_timeout_keeps_baseline_and_pending_enter() {
        permanently_blocked_fixture("query", "benchmark::runner::nvml_monitor::tests::nvml_permanent_query_timeout_keeps_baseline_and_pending_enter");
    }

    #[test]
    fn nvml_permanent_shutdown_timeout_keeps_complete_samples_and_pending_shutdown() {
        permanently_blocked_fixture("shutdown", "benchmark::runner::nvml_monitor::tests::nvml_permanent_shutdown_timeout_keeps_complete_samples_and_pending_shutdown");
    }
}
