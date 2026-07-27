//! Comparison report generation — aggregates L1, L2, and L3 results
//! and prints a human-readable summary, optionally as JSON.

use crate::l1::L1Result;
use crate::l2::L2Result;
use crate::l3::L3Result;
use crate::types::{
    KnownDeviation, ToleranceCalibration,
};

/// Aggregate results from a full golden comparison run.
#[derive(Debug, Default)]
pub struct ComparisonReport {
    pub manifest_path: String,
    pub model_path: String,
    pub l1_results: Vec<L1Result>,
    pub l2_results: Vec<L2Result>,
    pub l3_results: Vec<L3Result>,
    /// Fixtures that were skipped (e.g., no engine logits for regression).
    pub skipped_l2: Vec<String>,
    pub known_deviations: Vec<KnownDeviation>,
}

impl ComparisonReport {
    pub fn l1_passed(&self) -> bool {
        self.l1_results.iter().all(|r| r.passed)
    }

    pub fn l2_passed(&self) -> bool {
        self.l2_results.iter().all(|r| r.passed)
    }

    pub fn overall_passed(&self) -> bool {
        self.l1_passed() && self.l2_passed()
    }
}

/// Print a human-readable comparison report to stdout.
pub fn print_report(report: &ComparisonReport, tolerance: &ToleranceCalibration) {
    println!("══════════════════════════════════════════════════");
    println!("  vllm-oxide Golden Comparison Report");
    println!("══════════════════════════════════════════════════");
    println!("  Manifest:  {}", report.manifest_path);
    println!("  Model:     {}", report.model_path);
    println!("  Tolerance: atol={:.2e}, rtol={:.2e}", tolerance.atol, tolerance.rtol);
    if tolerance.observed_max_l2 > 1e-1 {
        println!(
            "  ⚠ WARNING: observed_max_l2 ({:.2e}) > 1e-1 — investigate oracle first (T8 Q8.2)",
            tolerance.observed_max_l2,
        );
    }
    println!();

    // ── L1 ──────────────────────────────────────────────
    print_l1_section(report);

    // ── L2 ──────────────────────────────────────────────
    print_l2_section(report);

    // ── L3 ──────────────────────────────────────────────
    if !report.l3_results.is_empty() {
        println!("── L3 (per-layer activations, debug-only) ──");
        for r in &report.l3_results {
            println!("  {}: {}", r.prompt_id, r.message);
        }
        println!();
    }

    // ── Known Deviations ────────────────────────────────
    if !report.known_deviations.is_empty() {
        println!("── Known Oracle Deviations ──");
        for dev in &report.known_deviations {
            println!(
                "  [{:?}] {}: max_l2={:.2e}, argmax_mismatches={}",
                dev.pair, dev.prompt_id, dev.max_l2, dev.argmax_mismatches,
            );
            println!("    {}", dev.note);
        }
        println!();
    }

    // ── Summary ─────────────────────────────────────────
    println!("══════════════════════════════════════════════════");
    let l1_status = if report.l1_passed() { "PASS" } else { "FAIL" };
    let l2_status = if report.l2_passed() { "PASS" } else { "FAIL" };
    let overall = if report.overall_passed() { "PASS" } else { "FAIL" };
    println!("  L1 (token match):   {}", l1_status);
    println!("  L2 (logits):        {}", l2_status);
    if !report.skipped_l2.is_empty() {
        println!("  L2 skipped:         {}", report.skipped_l2.join(", "));
    }
    println!("  ────────────────────────────────────────");
    println!("  OVERALL:            {}", overall);
    println!("══════════════════════════════════════════════");
}

fn print_l1_section(report: &ComparisonReport) {
    if report.l1_results.is_empty() {
        return;
    }

    println!("── L1 (greedy token-sequence exact match) ──");
    for r in &report.l1_results {
        let status = if r.passed { "✓" } else { "✗" };
        println!(
            "  {} {}: matches={}, near-tie-skips={}, mismatches={}, ε={:.2e}",
            status, r.prompt_id, r.exact_matches, r.near_tie_skips, r.mismatches, r.epsilon,
        );
        if let Some(pos) = r.first_mismatch {
            println!("    first mismatch at position {}", pos);
        }
    }
    println!();
}

fn print_l2_section(report: &ComparisonReport) {
    if report.l2_results.is_empty() {
        return;
    }

    println!("── L2 (per-step logits tensor comparison) ──");
    for r in &report.l2_results {
        let status = if r.passed { "✓" } else { "✗" };
        println!(
            "  {} {}: max_abs_diff={:.2e}, max_rel_diff={:.2e}, exceeding={}/{} elements",
            status,
            r.prompt_id,
            r.max_abs_diff,
            r.max_rel_diff,
            r.elements_exceeding_tol,
            r.total_elements,
        );
        if let Some(step) = r.max_abs_step {
            println!("    max abs diff at step {}", step);
        }
    }
    println!();
}

/// Generate a JSON report string.
pub fn json_report(report: &ComparisonReport, tolerance: &ToleranceCalibration) -> String {
    let mut json = String::from("{\n");

    json.push_str(&format!("  \"tolerance\": {{\"atol\": {:.10e}, \"rtol\": {:.10e}}},\n",
        tolerance.atol, tolerance.rtol));

    json.push_str("  \"l1\": [\n");
    for (i, r) in report.l1_results.iter().enumerate() {
        let comma = if i + 1 < report.l1_results.len() { "," } else { "" };
        json.push_str(&format!(
            "    {{\"prompt_id\": \"{}\", \"passed\": {}, \"exact_matches\": {}, \"near_tie_skips\": {}, \"mismatches\": {}, \"epsilon\": {:.10e}}}{}\n",
            r.prompt_id, r.passed, r.exact_matches, r.near_tie_skips, r.mismatches, r.epsilon, comma,
        ));
    }
    json.push_str("  ],\n");

    json.push_str("  \"l2\": [\n");
    for (i, r) in report.l2_results.iter().enumerate() {
        let comma = if i + 1 < report.l2_results.len() { "," } else { "" };
        json.push_str(&format!(
            "    {{\"prompt_id\": \"{}\", \"passed\": {}, \"max_abs_diff\": {:.10e}, \"max_rel_diff\": {:.10e}, \"elements_exceeding_tol\": {}}}{}\n",
            r.prompt_id, r.passed, r.max_abs_diff, r.max_rel_diff, r.elements_exceeding_tol, comma,
        ));
    }
    json.push_str("  ],\n");

    json.push_str(&format!(
        "  \"overall\": {}\n",
        report.overall_passed()
    ));
    json.push_str("}\n");
    json
}
