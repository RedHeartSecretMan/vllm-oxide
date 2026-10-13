//! Comparison report generation — aggregates L1, L2, and L3 results
//! and prints a human-readable summary, optionally as JSON.

use serde::Serialize;

use crate::l1::L1Result;
use crate::l2::L2Result;
use crate::l3::L3Result;
use crate::types::ToleranceCalibration;

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
    println!(
        "  Tolerance: atol={:.2e}, method={}",
        tolerance.atol, tolerance.method,
    );
    if tolerance.observed_max_abs_diff > 1e-1 {
        println!(
            "  ⚠ WARNING: observed_max_abs_diff ({:.2e}) > 1e-1 — investigate oracle first (T8 Q8.2)",
            tolerance.observed_max_abs_diff,
        );
    }
    println!();

    // ── L1 ──────────────────────────────────────────────
    print_l1_section(report);

    // ── L2 (same-prefix, no chain divergence) ───────────
    print_l2_section(report);

    // ── L3 ──────────────────────────────────────────────
    if !report.l3_results.is_empty() {
        println!("── L3 (per-layer activations, debug-only) ──");
        for r in &report.l3_results {
            println!("  {}: {}", r.prompt_id, r.message);
        }
        println!();
    }

    // ── Summary ─────────────────────────────────────────
    println!("══════════════════════════════════════════════════");
    let l1_status = if report.l1_passed() { "PASS" } else { "FAIL" };
    let l2_status = if report.l2_passed() { "PASS" } else { "FAIL" };
    let overall = if report.overall_passed() {
        "PASS"
    } else {
        "FAIL"
    };
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

    println!("── L2 (same-prefix logits, no chain divergence) ──");
    for r in &report.l2_results {
        let status = if r.passed { "✓" } else { "✗" };
        println!(
            "  {} {}: same-token steps={}, max_abs_diff={:.2e}, exceeding={}/{} elements",
            status,
            r.prompt_id,
            r.same_token_steps,
            r.max_abs_diff,
            r.elements_exceeding_tol,
            r.total_elements,
        );
        if r.diff_token_steps > 0 {
            println!("    {} steps skipped (token mismatch)", r.diff_token_steps,);
        }
    }
    println!();
}

// ── JSON serialization types ──────────────────────────────────────

#[derive(Serialize)]
struct JsonReportEntry<'a> {
    tolerance: &'a ToleranceCalibration,
    l1: Vec<JsonL1Entry>,
    l2: Vec<JsonL2Entry>,
    overall: bool,
}

#[derive(Serialize)]
struct JsonL1Entry {
    prompt_id: String,
    passed: bool,
    exact_matches: usize,
    near_tie_skips: usize,
    regression_skips: usize,
    mismatches: usize,
    epsilon: f64,
}

#[derive(Serialize)]
struct JsonL2Entry {
    prompt_id: String,
    passed: bool,
    max_abs_diff: f64,
    elements_exceeding_tol: usize,
}

/// Generate a JSON report string using serde serialization.
pub fn json_report(report: &ComparisonReport, tolerance: &ToleranceCalibration) -> String {
    let data = JsonReportEntry {
        tolerance,
        l1: report
            .l1_results
            .iter()
            .map(|r| JsonL1Entry {
                prompt_id: r.prompt_id.clone(),
                passed: r.passed,
                exact_matches: r.exact_matches,
                near_tie_skips: r.near_tie_skips,
                regression_skips: r.regression_skips,
                mismatches: r.mismatches,
                epsilon: r.epsilon,
            })
            .collect(),
        l2: report
            .l2_results
            .iter()
            .map(|r| JsonL2Entry {
                prompt_id: r.prompt_id.clone(),
                passed: r.passed,
                max_abs_diff: r.max_abs_diff,
                elements_exceeding_tol: r.elements_exceeding_tol,
            })
            .collect(),
        overall: report.overall_passed(),
    };
    serde_json::to_string_pretty(&data)
        .unwrap_or_else(|e| format!("{{ \"error\": \"serialization failed: {e}\" }}"))
}
