#!/usr/bin/env bash
# Fail-closed, separately invocable goldens-v0.2 stages (ADR-0012).

set -euo pipefail

STAGE="${1:-}"
RUN_ROOT="${2:-}"
MODEL_PATH="${3:-/home/wanghao/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/7e4ae267688d671ddfca3122e4528ee980cf3234}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GOLDEN_PROJECT="$SCRIPT_DIR/tools/golden-gen"
CARGO_TARGET_DIR=/home/wanghao/Projects/Codes/VibeCodings/vllm-oxide/target
CARGO_BUILD_JOBS=1
export CARGO_TARGET_DIR CARGO_BUILD_JOBS

usage() {
    echo "Usage: $0 {env|generate|calibrate|observe|authoritative|benchmark|report|bundle|verify-local|publish|verify} /tmp/vllm-oxide-dag-v0.2.0/t45-artifacts/<fresh-run> [model-path]" >&2
    exit 2
}

if [[ "$STAGE" == "publish" && "${VLLM_OXIDE_ALLOW_GOLDEN_PUBLISH:-}" != "goldens-v0.2" ]]; then
    echo "ERROR: publish requires separate VLLM_OXIDE_ALLOW_GOLDEN_PUBLISH=goldens-v0.2 authority" >&2
    exit 3
fi

case "$STAGE" in
    env|generate|calibrate|observe|authoritative|benchmark|report|bundle|verify-local|publish|verify) ;;
    *) usage ;;
esac

CANONICAL_RUN_ROOT="$(realpath -m -- "$RUN_ROOT")"
if [[ "$CANONICAL_RUN_ROOT" != "$RUN_ROOT" ]]; then
    echo "ERROR: run root must already be canonical and contain no traversal or symlink alias" >&2
    exit 2
fi
case "$CANONICAL_RUN_ROOT" in
    /tmp/vllm-oxide-dag-v0.2.0/t45-artifacts/*) ;;
    *) echo "ERROR: run root must be a fresh child of /tmp/vllm-oxide-dag-v0.2.0/t45-artifacts" >&2; exit 2 ;;
esac
RUN_ROOT="$CANONICAL_RUN_ROOT"
VENV="$RUN_ROOT/env/.venv"
PYTHON="$VENV/bin/python"
MARKERS="$RUN_ROOT/markers"

require_marker() {
    "$PYTHON" -m golden_gen verify-stage-marker \
        --run-root "$RUN_ROOT" \
        --stage "$1" \
        --repo-root "$SCRIPT_DIR"
}

complete_stage() {
    "$PYTHON" -m golden_gen stage-marker \
        --run-root "$RUN_ROOT" \
        --stage "$1" \
        --repo-root "$SCRIPT_DIR"
}

require_ram_guard() {
    local available_kib
    available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    if (( available_kib < 16 * 1024 * 1024 )); then
        echo "ERROR: available host RAM is below 16 GiB" >&2
        exit 4
    fi
}

set_measurement_identity() {
    if [[ -n "$(git -C "$SCRIPT_DIR" status --porcelain=v1 --untracked-files=all)" ]]; then
        echo "ERROR: measurement stages require a clean repository" >&2
        exit 4
    fi
    VLLM_OXIDE_MEASUREMENT_COMMIT="$(git -C "$SCRIPT_DIR" rev-parse HEAD)"
    VLLM_OXIDE_MEASUREMENT_TREE="$(git -C "$SCRIPT_DIR" rev-parse 'HEAD^{tree}')"
    export VLLM_OXIDE_MEASUREMENT_COMMIT VLLM_OXIDE_MEASUREMENT_TREE
}

case "$STAGE" in
env)
    set_measurement_identity
    test ! -e "$RUN_ROOT"
    mkdir -p "$RUN_ROOT/env"
    chmod 0700 "$RUN_ROOT" "$RUN_ROOT/env"
    UV_PROJECT_ENVIRONMENT="$VENV"
    export UV_PROJECT_ENVIRONMENT
    uv sync --extra gpu --frozen --no-build --no-install-project \
        --project "$GOLDEN_PROJECT" --python 3.12 --no-python-downloads \
        >"$RUN_ROOT/env/registry-install.log" 2>&1
    uv pip install --python "$PYTHON" --no-deps --editable "$GOLDEN_PROJECT" \
        >"$RUN_ROOT/env/project-install.log" 2>&1
    env PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        "$PYTHON" -m golden_gen preflight \
        --model-dir "$MODEL_PATH" \
        --repo-root "$SCRIPT_DIR" \
        --output "$RUN_ROOT/env/runtime.json"
    complete_stage env
    ;;
generate)
    require_marker env
    set_measurement_identity
    require_ram_guard
    test ! -e "$RUN_ROOT/generate"
    mkdir -m 0700 "$RUN_ROOT/generate"
    for oracle in transformers vllm; do
        for replay in primary replay; do
            "$PYTHON" -m golden_gen guard \
                --evidence "$RUN_ROOT/generate/$oracle-$replay-guard.json" -- \
            env PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
                "$PYTHON" -m golden_gen generate-oracle \
                --oracle "$oracle" \
                --runtime-record "$RUN_ROOT/env/runtime.json" \
                --prompts-dir "$GOLDEN_PROJECT/prompts" \
                --output-dir "$RUN_ROOT/generate/$oracle-$replay"
        done
        "$PYTHON" -m golden_gen verify-replay \
            --oracle "$oracle" \
            --primary-dir "$RUN_ROOT/generate/$oracle-primary" \
            --replay-dir "$RUN_ROOT/generate/$oracle-replay" \
            --output "$RUN_ROOT/generate/$oracle-replay-evidence.json"
    done
    "$PYTHON" -m golden_gen assemble \
        --runtime-record "$RUN_ROOT/env/runtime.json" \
        --reference-dir "$RUN_ROOT/generate/transformers-primary" \
        --baseline-dir "$RUN_ROOT/generate/vllm-primary" \
        --prompts-dir "$GOLDEN_PROJECT/prompts" \
        --output-dir "$RUN_ROOT/generate/fixtures"
    complete_stage generate
    ;;
calibrate)
    require_marker generate
    require_ram_guard
    test ! -e "$RUN_ROOT/calibrate"
    mkdir -m 0700 "$RUN_ROOT/calibrate"
    cp -R "$RUN_ROOT/generate/fixtures" "$RUN_ROOT/calibrate/fixtures"
    "$PYTHON" -m golden_gen calibrate-baseline \
        --manifest-dir "$RUN_ROOT/calibrate/fixtures"
    complete_stage calibrate
    ;;
observe)
    require_marker calibrate
    require_ram_guard
    set_measurement_identity
    test ! -e "$RUN_ROOT/observe"
    mkdir -m 0700 "$RUN_ROOT/observe"
    cargo build --manifest-path "$SCRIPT_DIR/Cargo.toml" --release -p vllm_oxide_test --features cuda --bin vllm-oxide-observe
    for replay in primary replay; do
        "$PYTHON" -m golden_gen guard \
            --evidence "$RUN_ROOT/observe/$replay-guard.json" -- \
        env PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
            "$CARGO_TARGET_DIR/release/vllm-oxide-observe" \
            --model-path "$MODEL_PATH" \
            --manifest "$RUN_ROOT/calibrate/fixtures/manifest.json" \
            --prompts-dir "$GOLDEN_PROJECT/prompts" \
            --repo-root "$SCRIPT_DIR" \
            --measurement-commit "$VLLM_OXIDE_MEASUREMENT_COMMIT" \
            --measurement-tree "$VLLM_OXIDE_MEASUREMENT_TREE" \
            --output-dir "$RUN_ROOT/observe/$replay"
    done
    set +e
    "$PYTHON" -m golden_gen observe \
        --manifest "$RUN_ROOT/calibrate/fixtures/manifest.json" \
        --primary-dir "$RUN_ROOT/observe/primary" \
        --replay-dir "$RUN_ROOT/observe/replay" \
        --output "$RUN_ROOT/observe/calibration-observation.json"
    observe_status=$?
    set -e
    test "$observe_status" -eq 3
    test -f "$RUN_ROOT/observe/calibration-observation.json"
    complete_stage observe
    ;;
authoritative)
    require_marker observe
    require_ram_guard
    set_measurement_identity
    APPROVED_OBSERVATION="$SCRIPT_DIR/docs/releases/goldens-v0.2-calibration-observation.json"
    test -f "$APPROVED_OBSERVATION"
    test ! -e "$RUN_ROOT/authoritative"
    mkdir -m 0700 "$RUN_ROOT/authoritative"
    cp -R "$RUN_ROOT/calibrate/fixtures" "$RUN_ROOT/authoritative/fixtures"
    "$PYTHON" -m golden_gen approve-policy \
        --manifest "$RUN_ROOT/authoritative/fixtures/manifest.json" \
        --observation "$APPROVED_OBSERVATION" \
        --repo-root "$SCRIPT_DIR"
    for replay in primary replay; do
        "$PYTHON" -m golden_gen guard \
            --evidence "$RUN_ROOT/authoritative/$replay-guard.json" -- \
        env PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        cargo run --manifest-path "$SCRIPT_DIR/Cargo.toml" --release -p vllm_oxide_test --features cuda -- \
            --mode authoritative \
            --approved-observation "$APPROVED_OBSERVATION" \
            --model-path "$MODEL_PATH" \
            --repo-root "$SCRIPT_DIR" \
            --measurement-commit "$VLLM_OXIDE_MEASUREMENT_COMMIT" \
            --measurement-tree "$VLLM_OXIDE_MEASUREMENT_TREE" \
            --manifest "$RUN_ROOT/authoritative/fixtures/manifest.json" \
            --prompts-dir "$GOLDEN_PROJECT/prompts" \
            --capture-dir "$RUN_ROOT/authoritative/captures-$replay" \
            --json >"$RUN_ROOT/authoritative/comparison-$replay.json"
    done
    "$PYTHON" -m golden_gen verify-candidate-replay \
        --primary-dir "$RUN_ROOT/authoritative/captures-primary" \
        --replay-dir "$RUN_ROOT/authoritative/captures-replay" \
        --output "$RUN_ROOT/authoritative/candidate-replay.json"
    cmp "$RUN_ROOT/authoritative/comparison-primary.json" \
        "$RUN_ROOT/authoritative/comparison-replay.json"
    cp "$RUN_ROOT/authoritative/comparison-primary.json" \
        "$RUN_ROOT/authoritative/comparison.json"
    complete_stage authoritative
    ;;
benchmark)
    require_marker authoritative
    require_ram_guard
    set_measurement_identity
    test ! -e "$RUN_ROOT/benchmark"
    mkdir -m 0700 "$RUN_ROOT/benchmark"
    "$PYTHON" -m golden_gen guard \
        --evidence "$RUN_ROOT/benchmark/guard.json" -- \
    env PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    cargo run --manifest-path "$SCRIPT_DIR/Cargo.toml" --release -p vllm_oxide_test --features cuda --bin vllm-oxide-benchmark -- \
        --model-path "$MODEL_PATH" \
        --prompts-dir "$GOLDEN_PROJECT/prompts" \
        --repo-root "$SCRIPT_DIR" \
        --measurement-commit "$VLLM_OXIDE_MEASUREMENT_COMMIT" \
        --measurement-tree "$VLLM_OXIDE_MEASUREMENT_TREE" \
        --output "$RUN_ROOT/benchmark/benchmark.json"
    complete_stage benchmark
    ;;
report)
    require_marker benchmark
    test ! -e "$RUN_ROOT/report"
    mkdir -m 0700 "$RUN_ROOT/report"
    "$PYTHON" -m golden_gen report \
        --manifest "$RUN_ROOT/authoritative/fixtures/manifest.json" \
        --observation "$SCRIPT_DIR/docs/releases/goldens-v0.2-calibration-observation.json" \
        --comparison "$RUN_ROOT/authoritative/comparison.json" \
        --benchmark "$RUN_ROOT/benchmark/benchmark.json" \
        --observation-commit "${VLLM_OXIDE_OBSERVATION_COMMIT:?Set observation commit}" \
        --observation-tree "${VLLM_OXIDE_OBSERVATION_TREE:?Set observation tree}" \
        --policy-checkpoint-commit "${VLLM_OXIDE_POLICY_CHECKPOINT_COMMIT:?Set policy checkpoint commit}" \
        --policy-checkpoint-tree "${VLLM_OXIDE_POLICY_CHECKPOINT_TREE:?Set policy checkpoint tree}" \
        --measurement-commit "${VLLM_OXIDE_MEASUREMENT_COMMIT:?Set measurement commit}" \
        --measurement-tree "${VLLM_OXIDE_MEASUREMENT_TREE:?Set measurement tree}" \
        --limitation "Single-GPU Qwen3 offline generation only; no cross-hardware performance claim." \
        --output "$RUN_ROOT/report/goldens-v0.2.md"
    complete_stage report
    ;;
bundle)
    require_marker report
    BUNDLE_DIR="$RUN_ROOT/bundle/goldens-v0.2"
    test ! -e "$RUN_ROOT/bundle"
    mkdir -m 0700 "$RUN_ROOT/bundle"
    "$PYTHON" -m golden_gen bundle \
        --fixture-dir "$RUN_ROOT/authoritative/fixtures" \
        --release-dir "$BUNDLE_DIR"
    test -f "$BUNDLE_DIR/manifest.json"
    test -f "$BUNDLE_DIR/goldens-v0.2.tar.gz"
    test "$(find "$BUNDLE_DIR" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 2
    complete_stage bundle
    ;;
publish)
    require_marker verify-local
    BUNDLE_DIR="$RUN_ROOT/bundle/goldens-v0.2"
    test ! -e "$RUN_ROOT/publish"
    mkdir -m 0700 "$RUN_ROOT/publish"
    CANDIDATE="${VLLM_OXIDE_GOLDEN_CANDIDATE:?Set the frozen reviewed Ticket candidate}"
    "$PYTHON" -m golden_gen publication --action prepare \
        --run-root "$RUN_ROOT" --repo-root "$SCRIPT_DIR" --candidate "$CANDIDATE"
    test "$(git -C "$SCRIPT_DIR" rev-parse HEAD)" = "$CANDIDATE"
    test -z "$(git -C "$SCRIPT_DIR" tag --list goldens-v0.2)"
    test -z "$(git -C "$SCRIPT_DIR" ls-remote --tags origin refs/tags/goldens-v0.2)"
    git -C "$SCRIPT_DIR" tag goldens-v0.2 "$CANDIDATE"
    git -C "$SCRIPT_DIR" push origin refs/tags/goldens-v0.2:refs/tags/goldens-v0.2
    gh release create goldens-v0.2 \
        --repo RedHeartSecretMan/vllm-oxide \
        --verify-tag \
        --title "Golden fixtures -- v0.2" \
        --notes-file "$RUN_ROOT/publish/release-notes.md" \
        "$BUNDLE_DIR/manifest.json" \
        "$BUNDLE_DIR/goldens-v0.2.tar.gz"
    git -C "$SCRIPT_DIR" ls-remote --tags origin refs/tags/goldens-v0.2 \
        >"$RUN_ROOT/publish/remote-tag.txt"
    gh release view goldens-v0.2 --repo RedHeartSecretMan/vllm-oxide \
        --json tagName,name,publishedAt,assets,url,isDraft,body \
        >"$RUN_ROOT/publish/release.json"
    "$PYTHON" -m golden_gen publication --action verify \
        --run-root "$RUN_ROOT" --repo-root "$SCRIPT_DIR" --candidate "$CANDIDATE"
    complete_stage publish
    ;;
verify-local)
    require_marker bundle
    test ! -e "$RUN_ROOT/verify-local"
    mkdir -m 0700 "$RUN_ROOT/verify-local"
    CARGO_TARGET_DIR=/tmp/vllm-oxide-dag-v0.2.0/shared-target \
    cargo run --manifest-path "$SCRIPT_DIR/Cargo.toml" --offline -p vllm_oxide_test \
        --bin vllm-oxide-verify-bundle -- \
        --bundle-dir "$RUN_ROOT/bundle/goldens-v0.2" \
        --cache-dir "$RUN_ROOT/verify-local/cache" \
        >"$RUN_ROOT/verify-local/consumer.json"
    complete_stage verify-local
    ;;
verify)
    require_marker publish
    require_ram_guard
    test ! -e "$RUN_ROOT/verify"
    mkdir -m 0700 "$RUN_ROOT/verify"
    CANDIDATE="${VLLM_OXIDE_GOLDEN_CANDIDATE:?Set the frozen reviewed Ticket candidate}"
    gh release download goldens-v0.2 --repo RedHeartSecretMan/vllm-oxide \
        --dir "$RUN_ROOT/verify/downloads"
    "$PYTHON" -m golden_gen publication --action verify-download \
        --run-root "$RUN_ROOT" --repo-root "$SCRIPT_DIR" --candidate "$CANDIDATE"
    CARGO_TARGET_DIR=/tmp/vllm-oxide-dag-v0.2.0/shared-target \
    cargo run --manifest-path "$SCRIPT_DIR/Cargo.toml" --offline -p vllm_oxide_test \
        --bin vllm-oxide-verify-bundle -- \
        --bundle-dir "$RUN_ROOT/verify/downloads" \
        --cache-dir "$RUN_ROOT/verify/cache" \
        >"$RUN_ROOT/verify/consumer.json"
    complete_stage verify
    ;;
esac
