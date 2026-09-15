"""Every tunable number in one place. Structure lives with its code; only knobs belong here."""

from __future__ import annotations

MIN_NORMAL = 2**-126  # smallest normal bf16 and fp32 value; below it hardware returns zero

# Sweep sizing
TILE_WIDTH = 2**7
FP32_BLOCK = 2**6 * 2**9 * TILE_WIDTH
# Worst-of-group per row: keeps the max exact and an fp32 op the weight of a bf16 one.
FP32_GROUP = 2**16
B_CHUNK = 2**7  # second operands per dispatch, matching ttnn-eltwise-op-tester's batch
TERNARY_STRIDE = 2**9  # keeps a 3-operand sweep the same size as a 2-operand one
FP32_SAMPLE_SEED = 0  # fixed: the fp32 pair sample must not move between releases
FP32_MANTISSA_SAMPLES = 512  # mantissas per exponent when bisecting an fp32 domain

# Timing
SIDE = 2**12  # at 2**20 elements the op costs ~65us and host jitter moved it 70% per run
WARMUP = 5  # the first dispatches pay kernel lookup and cache population
BATCH = 5  # one synchronize per dispatch times the host's scheduler as much as the kernel
REPEATS = 10  # samples of the batch mean, so the median has something to choose between
FINITE = (-10.0, 10.0)  # timing range for an op whose derived domain is unbounded
NOISE_PCT = 5.0  # `us_min` moved 3.1% between two runs of one build; inside this is noise

# Scoring and charts
USABLE_ULP = 2.0  # what "still accurate here" means for the usable-range figure
ULP_CLIP = 1000.0  # chart clamp, and the CDF's right edge, so every chart compares
ULP_LINES = (1, 3, 10, 100)  # reference lines, drawn only once the data reaches the one before
N_BINS = 32
MIN_BIN = 8  # a bin of fewer points cannot carry a p99, so it is hidden rather than believed
CDF_POINTS = 200  # a log grid: an fp32 sweep holds 65k distinct ULP values, and draws as 200
OFFENDERS = 10  # worst points listed per variant: a maximum says how bad, these say where
MONOTONIC_TOP = 5  # worst ordering violations listed per variant
NONFINITE_DETAIL = 10  # non-finite points listed per variant, out of however many there are
