"""Constants for golden fixture generation."""

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "7e4ae267688d671ddfca3122e4528ee980cf3234"
MODEL_DTYPE = "bfloat16"
ARCH = "Qwen3ForCausalLM"
VOCAB_SIZE = 151936
ATTN_IMPLEMENTATION = "eager"
CANONICAL_MAX_TOKENS = 64
REGRESSION_MAX_TOKENS = 32
TOP_K_REGRESSION = 5
TOLERANCE_CALIBRATION_FACTOR = 2.0
NANO_VLLM_TEMPERATURE = 1e-9
ORACLE_INVESTIGATION_THRESHOLD = 0.1  # Spec T8 Q8.2: divergence > 0.1 means investigate
DEVIATION_REPORT_THRESHOLD = 1e-6  # Log a KnownDeviation if L2 or argmax mismatch exceeds this


def max_tokens_for_category(category: str) -> int:
    """Max generation tokens for a prompt category."""
    return CANONICAL_MAX_TOKENS if category == "canonical" else REGRESSION_MAX_TOKENS
