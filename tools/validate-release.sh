#!/usr/bin/env bash
# Local historical schema-1 validation; run manually on a suitable CUDA host.
# Usage: ./tools/validate-release.sh /path/to/Qwen3-0.6B

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <model-path>" >&2
    exit 2
fi
MODEL_PATH="$(cd -- "$1" && pwd -P)"
REPO_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vllm-oxide-v010-validation.XXXXXXXX")"
GOLDEN_OUTPUT="$RUN_DIR/fixtures"
export UV_PROJECT_ENVIRONMENT="$RUN_DIR/.venv"
export PYTHONDONTWRITEBYTECODE=1

echo "vllm-oxide v0.1 local validation"
echo "Model for Rust comparison: $MODEL_PATH"
echo "Local evidence and environment: $RUN_DIR"

cd "$REPO_ROOT/tools/golden-gen"
uv sync --locked --extra gpu 2>&1 | tee "$RUN_DIR/environment.log"
uv run --locked --extra gpu --no-sync python -m golden_gen generate \
    --output-dir "$GOLDEN_OUTPUT" 2>&1 | tee "$RUN_DIR/generation.log"
uv run --locked --extra gpu --no-sync python -m golden_gen calibrate \
    --manifest-dir "$GOLDEN_OUTPUT" 2>&1 | tee "$RUN_DIR/calibration.log"

cd "$REPO_ROOT"
cargo build --release -p vllm_oxide_test --features cuda \
    2>&1 | tee "$RUN_DIR/build.log"
cargo run --release -p vllm_oxide_test --features cuda -- \
    --model-path "$MODEL_PATH" \
    --manifest "$GOLDEN_OUTPUT/manifest.json" \
    --prompts-dir "$REPO_ROOT/tools/golden-gen/prompts" \
    2>&1 | tee "$RUN_DIR/comparison.log"

echo "Local validation passed. Evidence retained at $RUN_DIR"
