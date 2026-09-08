#![allow(clippy::unwrap_used)]

use vllm_oxide_test::release_transport::parse_manifest;

#[test]
fn historical_manifest_never_becomes_layered_transport() {
    assert!(parse_manifest(br#"{"schema_version":4}"#).is_err());
}
