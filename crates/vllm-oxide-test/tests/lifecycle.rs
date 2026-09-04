use vllm_oxide_test::LifecycleTracker;

#[test]
fn successful_lifecycle_reports_exact_totals_and_passes_release() {
    let mut lifecycle = LifecycleTracker::new([
        "canonical_01.transformers".to_string(),
        "canonical_01.vllm".to_string(),
    ]);
    for fixture_id in ["canonical_01.transformers", "canonical_01.vllm"] {
        lifecycle.record_discovered(fixture_id);
        lifecycle.record_generated(fixture_id);
        lifecycle.record_compared(fixture_id);
    }

    let totals = lifecycle.totals();

    assert_eq!(totals.expected, 2);
    assert_eq!(totals.discovered, 2);
    assert_eq!(totals.generated, 2);
    assert_eq!(totals.compared, 2);
    assert_eq!(totals.missing, 0);
    assert_eq!(totals.unexpected, 0);
    assert_eq!(totals.skipped, 0);
    assert_eq!(totals.failed, 0);
    assert!(totals.release_passed());
}

#[test]
fn incomplete_lifecycle_reports_each_fail_closed_total() {
    let mut lifecycle = LifecycleTracker::new([
        "canonical_01.transformers".to_string(),
        "canonical_01.vllm".to_string(),
    ]);
    lifecycle.record_discovered("canonical_01.transformers");
    lifecycle.record_generated("canonical_01.transformers");
    lifecycle.record_skipped("canonical_01.transformers");
    lifecycle.record_failed("canonical_01.vllm");
    lifecycle.record_generated("unexpected.transformers");

    let totals = lifecycle.totals();

    assert_eq!(totals.expected, 2);
    assert_eq!(totals.discovered, 1);
    assert_eq!(totals.generated, 1);
    assert_eq!(totals.compared, 0);
    assert_eq!(totals.missing, 1);
    assert_eq!(totals.unexpected, 1);
    assert_eq!(totals.skipped, 1);
    assert_eq!(totals.failed, 1);
    assert!(!totals.release_passed());
}
