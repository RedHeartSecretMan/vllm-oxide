#!/usr/bin/env bash
# Release validation script for vllm-oxide v0.2.0
#
# Run on a GPU machine (sm_89+) before tagging a release.
# This is the RELEASE GATE — not CI.
#
# Usage (thresholds must come from reviewed release evidence):
#   TOLERANCE_POLICY_VERSION=same-prefix-v1 \
#   L1_NEAR_TIE_MAX_ABS_LOGIT_GAP=<value> L2_ATOL=<value> \
#   TOLERANCE_POLICY_RATIONALE=<text> TOLERANCE_POLICY_EVIDENCE=<uri> \
#   ./tools/validate-release.sh /path/to/Qwen3-0.6B [goldens-v0.2]

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model-path> [release-tag]}"
RELEASE_TAG="${2:-goldens-v0.2}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GOLDEN_OUTPUT="/tmp/vllm-oxide-goldens-release"
TOLERANCE_POLICY_VERSION="${TOLERANCE_POLICY_VERSION:?Set reviewed policy version}"
L1_NEAR_TIE_MAX_ABS_LOGIT_GAP="${L1_NEAR_TIE_MAX_ABS_LOGIT_GAP:?Set reviewed L1 threshold}"
L2_ATOL="${L2_ATOL:?Set the reviewed L2 threshold}"
TOLERANCE_POLICY_RATIONALE="${TOLERANCE_POLICY_RATIONALE:?Set reviewed policy rationale}"
TOLERANCE_POLICY_EVIDENCE="${TOLERANCE_POLICY_EVIDENCE:?Set reviewed evidence URI}"

if [[ "$RELEASE_TAG" != "goldens-v0.2" ]]; then
    echo "ERROR: schema-v4 assets require release tag goldens-v0.2" >&2
    exit 1
fi
BUNDLE_PARENT="$(mktemp -d /tmp/vllm-oxide-golden-bundle.XXXXXX)"
GOLDEN_BUNDLE="$BUNDLE_PARENT/goldens-v0.2"

echo "════════════════════════════════════════════"
echo "  vllm-oxide v0.2.0 Release Validation"
echo "════════════════════════════════════════════"
echo "  Model:      $MODEL_PATH"
echo "  Release:    $RELEASE_TAG"
echo "  Goldens:    $GOLDEN_OUTPUT"
echo "  Bundle:     $GOLDEN_BUNDLE"
echo ""

# Step 1: Generate golden fixtures (both oracles)
echo "── Step 1: Generating golden fixtures ──"
cd "$SCRIPT_DIR/tools/golden-gen"
uv sync
uv run python -m golden_gen generate \
    --output-dir "$GOLDEN_OUTPUT"
echo ""

# Step 2: Record calibration observations and explicit tolerance policy
echo "── Step 2: Calibrating tolerance ──"
uv run python -m golden_gen calibrate \
    --manifest-dir "$GOLDEN_OUTPUT" \
    --tolerance-policy-version "$TOLERANCE_POLICY_VERSION" \
    --l1-near-tie-max-abs-logit-gap "$L1_NEAR_TIE_MAX_ABS_LOGIT_GAP" \
    --l2-atol "$L2_ATOL" \
    --tolerance-policy-rationale "$TOLERANCE_POLICY_RATIONALE" \
    --tolerance-policy-evidence "$TOLERANCE_POLICY_EVIDENCE"
echo ""

# Step 3: Build and validate the exact-two local release bundle
echo "── Step 3: Building exact-two golden asset bundle ──"
uv run python -m golden_gen bundle \
    --fixture-dir "$GOLDEN_OUTPUT" \
    --release-dir "$GOLDEN_BUNDLE"
test -f "$GOLDEN_BUNDLE/manifest.json"
test -f "$GOLDEN_BUNDLE/goldens-v0.2.tar.gz"
test "$(find "$GOLDEN_BUNDLE" -mindepth 1 -maxdepth 1 | wc -l)" -eq 2
echo ""

# Step 4: Build the comparison crate (release mode)
echo "── Step 4: Building comparison crate ──"
cd "$SCRIPT_DIR"
cargo build --release -p vllm_oxide_test
echo ""

# Step 5: Run the comparison
echo "── Step 5: Running golden comparison ──"
cargo run --release -p vllm_oxide_test -- \
    --model-path "$MODEL_PATH" \
    --manifest "$GOLDEN_OUTPUT/manifest.json" \
    --prompts-dir "$SCRIPT_DIR/tools/golden-gen/prompts"
echo ""

# Step 6: Upload the exact two GitHub Release assets (if PASS)
echo "── Step 6: Uploading exact-two Release assets ──"
gh release create "$RELEASE_TAG" \
    --repo RedHeartSecretMan/vllm-oxide \
    --title "Golden fixtures — $RELEASE_TAG" \
    --notes "Schema-v4 golden asset bundle for vllm-oxide v0.2.0." \
    "$GOLDEN_BUNDLE/manifest.json" \
    "$GOLDEN_BUNDLE/goldens-v0.2.tar.gz"

echo ""
echo "════════════════════════════════════════════"
echo "  Release $RELEASE_TAG published"
echo "════════════════════════════════════════════"
