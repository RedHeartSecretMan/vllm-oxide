"""Constants for golden fixture generation."""

from typing import Final, Literal

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "7e4ae267688d671ddfca3122e4528ee980cf3234"
MODEL_DTYPE = "bfloat16"
ARCH = "Qwen3ForCausalLM"
VOCAB_SIZE = 151936
ATTN_IMPLEMENTATION = "sdpa"
CANONICAL_MAX_TOKENS = 64
REGRESSION_MAX_TOKENS = 32
TOP_K_REGRESSION = 5
TOLERANCE_CALIBRATION_FACTOR = 2.0
PRODUCT_VERSION: Final[Literal["v0.2.0"]] = "v0.2.0"
GOLDEN_VERSION: Final[Literal["goldens-v0.2"]] = "goldens-v0.2"
ARCHIVE_FILENAME: Final[Literal["goldens-v0.2.tar.gz"]] = "goldens-v0.2.tar.gz"
