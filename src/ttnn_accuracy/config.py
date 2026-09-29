# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Every tunable number in one place. Structure lives with its code; only knobs belong here."""

from __future__ import annotations

MIN_NORMAL = 2**-126  # smallest normal bf16 and fp32 value; below it hardware returns zero
# Largest value each dtype holds; a domain edge here is the format ending, not the function.
MAX_FINITE = {"bf16": 3.3895313892515355e38, "fp32": 3.4028234663852886e38}

# Sweep sizing
TILE_WIDTH = 2**7
FP32_BLOCK = 2**6 * 2**9 * TILE_WIDTH
# Worst-of-group per row: keeps the max exact and an fp32 op the weight of a bf16 one.
FP32_GROUP = 2**16
B_CHUNK = 2**7  # second operands per dispatch, matching ttnn-eltwise-op-tester's batch
TERNARY_STRIDE = 2**9  # keeps a 3-operand sweep the same size as a 2-operand one
FP32_SAMPLE_SEED = 0  # fixed: the fp32 pair sample must not move between releases
FP32_MANTISSA_SAMPLES = 512  # mantissas per exponent when bisecting an fp32 domain

# Incomplete sweeps the pages must disclose. Unary and bf16 pairs are exhaustive.
SAMPLED = {
    (2, "fp32"): "both operands take 65,536 of the 2³² fp32 values, drawn once from a fixed seed",
    (3, "bf16"): (
        f"the first operand is exhaustive; the second and third take every "
        f"{TERNARY_STRIDE}th bf16 code, {2**16 // TERNARY_STRIDE} values each"
    ),
    (3, "fp32"): (
        f"the first operand takes 65,536 of the 2³² fp32 values, drawn once from a fixed "
        f"seed; the second and third take every {TERNARY_STRIDE}th of that sample"
    ),
}

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
# Reference lines, drawn only once the data reaches the one before. 0.5 is the whole point of
# measuring against a wider reference: at or below it the device is correctly rounded.
ULP_LINES = (0.5, 1, 3, 10, 100)
N_BINS = 32
MIN_BIN = 8  # a bin of fewer points cannot carry a p99, so it is hidden rather than believed
CDF_POINTS = 200  # a log grid: an fp32 sweep holds 65k distinct ULP values, and draws as 200
# Named ops a changed-path selection may ask for before its categories are cheaper to sweep.
SELECT_CAP = 12
OFFENDERS = 10  # worst points listed per variant: a maximum says how bad, these say where
MONOTONIC_TOP = 5  # worst ordering violations listed per variant
NONFINITE_DETAIL = 10  # non-finite points listed per variant, out of however many there are
