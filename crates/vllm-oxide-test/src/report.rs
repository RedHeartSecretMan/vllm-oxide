//! Comparison report generation — aggregates L1, L2, and L3 results
//! and prints a human-readable summary, optionally as JSON.

use serde::Serialize;

use crate::l1::L1Result;
use crate::l2::L2Result;
use crate::l3::L3Result;
use crate::types::{BaselineCalibration, TolerancePolicy};

/// Exact fixture lifecycle accounting for one validation run.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct LifecycleTotals {
    pub expected: usize,
    pub discovered: usize,
    pub generated: usize,
    pub compared: usize,
    pub missing: usize,
    pub unexpected: usize,
    pub skipped: usize,
    pub failed: usize,
}

impl LifecycleTotals {
    /// Release validation is fail-closed: every expected fixture must reach comparison.
    pub fn release_passed(&self) -> bool {
        self.expected > 0
            && self.discovered == self.expected
            && self.generated == self.expected
            && self.compared == self.expected
            && self.missing == 0
            && self.unexpected == 0
            && self.skipped == 0
            && self.failed == 0
    }
}

/// Aggregate results from a full golden comparison run.
#[derive(Debug, Default)]
pub struct ComparisonReport {
    pub manifest_path: String,
    pub model_path: String,
    pub l1_results: Vec<L1Result>,
    pub l2_results: Vec<L2Result>,
    pub l3_results: Vec<L3Result>,
    pub lifecycle: LifecycleTotals,
    pub failures: Vec<String>,
}

impl ComparisonReport {
    pub fn l1_passed(&self) -> bool {
        self.l1_results.iter().all(|r| r.passed)
    }

    pub fn l2_passed(&self) -> bool {
        self.l2_results.iter().all(|r| r.passed)
    }

    pub fn reference_passed(&self) -> bool {
        (!self.l1_results.is_empty() || !self.l2_results.is_empty())
            && self.l1_passed()
            && self.l2_passed()
    }

    pub fn overall_passed(&self) -> bool {
        self.reference_passed() && self.lifecycle.release_passed()
    }
}

/// Print a human-readable comparison report to stdout.
pub fn print_report(
    report: &ComparisonReport,
    policy: &TolerancePolicy,
    calibration: &BaselineCalibration,
    calibrated_fixtures: &[String],
) {
    println!("══════════════════════════════════════════════════");
    println!("  vllm-oxide Golden Comparison Report");
    println!("══════════════════════════════════════════════════");
    println!("  Manifest:  {}", report.manifest_path);
    println!("  Model:     {}", report.model_path);
    println!(
        "  Tolerance policy: version={}, dtype={}, kernel={}, L1 candidate-gap≤{:.2e}, L2 atol={:.2e}",
        policy.version,
        policy.dtype,
        policy.kernel,
        policy.l1_near_tie_max_abs_logit_gap,
        policy.l2_atol,
    );
    println!("  Policy rationale: {}", policy.rationale);
    println!("  Policy evidence: {}", policy.evidence.join(", "));
    println!(
        "  Baseline calibration observation: fixtures={}, candidate_atol={:.2e}, observed_max_abs_diff={:.2e}, method={}",
        calibrated_fixtures.len(),
        calibration.candidate_atol,
        calibration.observed_max_abs_diff,
        calibration.method,
    );
    if calibration.observed_max_abs_diff > 1e-1 {
        println!(
            "  ⚠ WARNING: observed_max_abs_diff ({:.2e}) > 1e-1 — investigate oracle first (T8 Q8.2)",
            calibration.observed_max_abs_diff,
        );
    }
    println!();

    println!("── Fixture lifecycle ──");
    println!(
        "  expected={} discovered={} generated={} compared={} skipped={} failed={}",
        report.lifecycle.expected,
        report.lifecycle.discovered,
        report.lifecycle.generated,
        report.lifecycle.compared,
        report.lifecycle.skipped,
        report.lifecycle.failed,
    );
    println!(
        "  missing={} unexpected={}",
        report.lifecycle.missing, report.lifecycle.unexpected,
    );
    for failure in &report.failures {
        println!("  ✗ {failure}");
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
    let l1_status = if report.l1_results.is_empty() {
        "NOT RUN"
    } else if report.l1_passed() {
        "PASS"
    } else {
        "FAIL"
    };
    let l2_status = if report.l2_results.is_empty() {
        "NOT RUN"
    } else if report.l2_passed() {
        "PASS"
    } else {
        "FAIL"
    };
    let overall = if report.overall_passed() {
        "PASS"
    } else {
        "FAIL"
    };
    println!("  L1 (token match):   {}", l1_status);
    println!("  L2 (logits):        {}", l2_status);
    println!("  ────────────────────────────────────────");
    println!("  OVERALL:            {}", overall);
    println!("══════════════════════════════════════════════");
}

fn print_l1_section(report: &ComparisonReport) {
    if report.l1_results.is_empty() {
        return;
    }

    println!("── L1 (reference token or explicit near tie) ──");
    for r in &report.l1_results {
        let status = if r.passed { "✓" } else { "✗" };
        println!(
            "  {} {}: matches={}, near-ties={}, mismatches={}, candidate-gap≤{:.2e}",
            status,
            r.prompt_id,
            r.exact_matches,
            r.near_ties,
            r.mismatches,
            r.near_tie_max_abs_logit_gap,
        );
        if let Some(pos) = r.first_divergence {
            println!(
                "    first divergence at position {}; {} later positions excluded",
                pos, r.excluded_positions,
            );
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
            "  {} {}: compared steps={}, max_abs_diff={:.2e}, exceeding={}/{} elements",
            status,
            r.prompt_id,
            r.compared_steps,
            r.max_abs_diff,
            r.elements_exceeding_tol,
            r.total_elements,
        );
        if let Some(position) = r.first_divergence {
            println!(
                "    first divergence at position {}; {} later steps excluded",
                position, r.excluded_steps,
            );
        }
    }
    println!();
}

// ── JSON serialization types ──────────────────────────────────────

#[derive(Serialize)]
struct JsonReportEntry<'a> {
    lifecycle: &'a LifecycleTotals,
    failures: &'a [String],
    reference_correctness: JsonReferenceCorrectness<'a>,
    baseline_calibration: JsonBaselineCalibration<'a>,
    overall: bool,
}

#[derive(Serialize)]
struct JsonReferenceCorrectness<'a> {
    tolerance_policy: &'a TolerancePolicy,
    l1: Vec<JsonL1Entry>,
    l2: Vec<JsonL2Entry>,
    passed: bool,
}

#[derive(Serialize)]
struct JsonBaselineCalibration<'a> {
    candidate_atol: f64,
    observed_max_abs_diff: f64,
    calibration_factor: f64,
    method: &'a str,
    calibrated_fixtures: &'a [String],
}

#[derive(Serialize)]
struct JsonL1Entry {
    prompt_id: String,
    passed: bool,
    exact_matches: usize,
    near_ties: usize,
    mismatches: usize,
    first_divergence: Option<usize>,
    excluded_positions: usize,
    policy_version: String,
    near_tie_max_abs_logit_gap: f64,
}

#[derive(Serialize)]
struct JsonL2Entry {
    prompt_id: String,
    passed: bool,
    compared_steps: usize,
    first_divergence: Option<usize>,
    excluded_steps: usize,
    total_elements: usize,
    max_abs_diff: f64,
    elements_exceeding_tol: usize,
}

/// Generate a JSON report string using serde serialization.
pub fn json_report(
    report: &ComparisonReport,
    policy: &TolerancePolicy,
    calibration: &BaselineCalibration,
    calibrated_fixtures: &[String],
) -> String {
    let data = JsonReportEntry {
        lifecycle: &report.lifecycle,
        failures: &report.failures,
        reference_correctness: JsonReferenceCorrectness {
            tolerance_policy: policy,
            l1: report
                .l1_results
                .iter()
                .map(|r| JsonL1Entry {
                    prompt_id: r.prompt_id.clone(),
                    passed: r.passed,
                    exact_matches: r.exact_matches,
                    near_ties: r.near_ties,
                    mismatches: r.mismatches,
                    first_divergence: r.first_divergence,
                    excluded_positions: r.excluded_positions,
                    policy_version: r.policy_version.clone(),
                    near_tie_max_abs_logit_gap: r.near_tie_max_abs_logit_gap,
                })
                .collect(),
            l2: report
                .l2_results
                .iter()
                .map(|r| JsonL2Entry {
                    prompt_id: r.prompt_id.clone(),
                    passed: r.passed,
                    compared_steps: r.compared_steps,
                    first_divergence: r.first_divergence,
                    excluded_steps: r.excluded_steps,
                    total_elements: r.total_elements,
                    max_abs_diff: r.max_abs_diff,
                    elements_exceeding_tol: r.elements_exceeding_tol,
                })
                .collect(),
            passed: report.reference_passed(),
        },
        baseline_calibration: JsonBaselineCalibration {
            candidate_atol: calibration.candidate_atol,
            observed_max_abs_diff: calibration.observed_max_abs_diff,
            calibration_factor: calibration.calibration_factor,
            method: &calibration.method,
            calibrated_fixtures,
        },
        overall: report.overall_passed(),
    };
    serde_json::to_string_pretty(&data)
        .unwrap_or_else(|e| format!("{{ \"error\": \"serialization failed: {e}\" }}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{json_report, ComparisonReport, LifecycleTotals};
    use crate::types::{BaselineCalibration, TolerancePolicy};

    #[test]
    fn empty_comparison_set_fails_closed() {
        let report = ComparisonReport::default();

        assert!(!report.overall_passed());
    }

    #[test]
    fn complete_lifecycle_is_release_accepted() {
        let totals = LifecycleTotals {
            expected: 2,
            discovered: 2,
            generated: 2,
            compared: 2,
            missing: 0,
            unexpected: 0,
            skipped: 0,
            failed: 0,
        };

        assert!(totals.release_passed());
    }

    #[test]
    fn json_report_contains_exact_lifecycle_totals() {
        let report = ComparisonReport {
            lifecycle: LifecycleTotals {
                expected: 8,
                discovered: 7,
                generated: 6,
                compared: 5,
                missing: 2,
                unexpected: 1,
                skipped: 1,
                failed: 3,
            },
            ..ComparisonReport::default()
        };
        let calibration = BaselineCalibration {
            candidate_atol: 0.01,
            observed_max_abs_diff: 0.005,
            calibration_factor: 2.0,
            method: "test".to_string(),
        };
        let policy = TolerancePolicy {
            version: "same-prefix-v1".to_string(),
            dtype: "bfloat16".to_string(),
            kernel: "sdpa".to_string(),
            l1_near_tie_max_abs_logit_gap: 0.02,
            l2_atol: 0.01,
            rationale: "Reviewed synthetic policy".to_string(),
            evidence: vec!["synthetic:report".to_string()],
        };
        let calibrated_fixtures = vec!["canonical_01.vllm".to_string()];

        let json: serde_json::Value = serde_json::from_str(&json_report(
            &report,
            &policy,
            &calibration,
            &calibrated_fixtures,
        ))
        .unwrap();

        assert_eq!(json["lifecycle"]["expected"], 8);
        assert_eq!(json["lifecycle"]["discovered"], 7);
        assert_eq!(json["lifecycle"]["generated"], 6);
        assert_eq!(json["lifecycle"]["compared"], 5);
        assert_eq!(json["lifecycle"]["missing"], 2);
        assert_eq!(json["lifecycle"]["unexpected"], 1);
        assert_eq!(json["lifecycle"]["skipped"], 1);
        assert_eq!(json["lifecycle"]["failed"], 3);
        assert_eq!(
            json["reference_correctness"]["tolerance_policy"]["version"],
            "same-prefix-v1"
        );
        assert_eq!(json["reference_correctness"]["passed"], false);
        assert_eq!(
            json["baseline_calibration"]["calibrated_fixtures"][0],
            "canonical_01.vllm"
        );
        assert_eq!(json["baseline_calibration"]["observed_max_abs_diff"], 0.005);
        assert_eq!(json["baseline_calibration"]["candidate_atol"], 0.01);
        assert_eq!(json["overall"], false);
    }
}
