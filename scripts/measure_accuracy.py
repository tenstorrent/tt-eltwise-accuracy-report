#!/usr/bin/env python3
"""
Measure accuracy of ttnn eltwise ops vs torch/golden reference.

Usage:
    python measure_accuracy.py --arch wh --ops exp,gelu --dtype bf16
    python measure_accuracy.py --arch bh --ops all --dtype both
    python measure_accuracy.py --arch wh --category unary --dtype bf16

Outputs CSV files to data/{arch}/{dtype}/{op_name}/{variant}.csv

CSV columns: x, y, y_ref, max_ulp_error, mean_ulp_error,
             max_abs_error, mean_abs_error, max_rel_error, mean_rel_error
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# Resolve repo root regardless of invocation directory
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"

TERM_RED = "\033[91m"
TERM_GREEN = "\033[92m"
TERM_RESET = "\033[0m"

EPSILON = 2**-9  # small positive floor for relative-error denominator


# ---------------------------------------------------------------------------
# ULP helper (mirrors Nathan's approach, no external dependency)
# ---------------------------------------------------------------------------

def ulp_torch(x: torch.Tensor) -> torch.Tensor:
    """Return the ULP (unit in last place) for each element of x, always as float32.

    Keeping the result as float32 (not converting back to the input dtype) prevents
    bf16 underflow: ULP of bf16 min-normal is 2^-133 which is a valid fp32 subnormal
    but would flush to 0 if stored as bf16.
    """
    if x.dtype == torch.bfloat16:
        mantissa_bits = 7
    elif x.dtype == torch.float32:
        mantissa_bits = 23
    else:
        return ulp_torch(x.to(torch.float32))

    x_f32 = x.to(torch.float32).abs()
    x_f32 = torch.clamp(x_f32, min=torch.finfo(torch.float32).tiny)
    exp = torch.floor(torch.log2(x_f32))
    return torch.pow(2.0, exp - mantissa_bits)  # float32, never .to(dtype)


def flush_subnormals(t: torch.Tensor) -> torch.Tensor:
    min_normal = torch.finfo(t.dtype).tiny
    return torch.where(t.abs() < min_normal, torch.zeros_like(t), t)


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_with_golden(
    x_input: torch.Tensor,
    golden: torch.Tensor,
    calculated: torch.Tensor,
    group_size: int,
) -> pd.DataFrame:
    """Compute accuracy metrics, grouped by group_size consecutive elements."""
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "")

        compute_dtype = calculated.dtype  # bf16 or fp32 — must be saved before converting
        # Flush subnormals in both hardware output and golden, using hardware precision.
        # Hardware flushes subnormals to zero; applying the same flush to the golden prevents
        # artificially large ULP errors where one side is subnormal and the other is exactly 0.
        calc_flushed = flush_subnormals(calculated.to(compute_dtype)).to(torch.float32)
        calc = calc_flushed
        gold_downcast = flush_subnormals(golden.to(compute_dtype))  # ULP in hardware precision
        gold = flush_subnormals(golden.to(torch.float32))

        ulp = ulp_torch(gold_downcast).to(torch.float32)

        n = x_input.nelement()
        n_groups = n // group_size
        shape = [n_groups, group_size]

        x_np = x_input.to(torch.float32).flatten().numpy().reshape(shape)
        y_np = calc.flatten().numpy().reshape(shape)
        yr_np = gold.flatten().numpy().reshape(shape)
        ulp_np = ulp.flatten().numpy().reshape(shape)

        abs_err = np.abs(yr_np - y_np)
        rel_err = abs_err / np.maximum(np.abs(yr_np), EPSILON)
        # Use a very small floor (fp32 min subnormal ≈ 1e-45) so bf16 ULPs like 2^-133
        # are not clamped away, while still protecting against true zero division.
        ulp_err = abs_err / np.maximum(ulp_np, 1e-45)

        # When hardware gives 0 and the golden is at most min_normal of the compute dtype,
        # the error is due to hardware zeroing a result at the subnormal boundary (e.g.,
        # tanh(min_normal) via e^(2x)=1 in compute dtype). Treat these as 0 error.
        min_normal_compute = float(torch.finfo(compute_dtype).tiny)
        near_zero_mask = (np.abs(y_np) == 0.0) & (np.abs(yr_np) <= min_normal_compute)
        ulp_err = np.where(near_zero_mask, 0.0, ulp_err)
        abs_err = np.where(near_zero_mask, 0.0, abs_err)
        rel_err = np.where(near_zero_mask, 0.0, rel_err)

        x_repr = x_np[:, 0]
        y_repr = y_np[:, 0]
        yr_repr = yr_np[:, 0]

        return pd.DataFrame({
            "x":             x_repr,
            "y":             y_repr,
            "y_ref":         yr_repr,
            "ulp_error":     np.nanmax(ulp_err, axis=-1),   # per-input for bf16; worst-case per batch for fp32
            "abs_error":     np.nanmax(abs_err, axis=-1),
            "rel_error":     np.nanmax(rel_err, axis=-1),
        })


# ---------------------------------------------------------------------------
# BF16 exhaustive measurement
# ---------------------------------------------------------------------------

def measure_bf16(
    op_name: str,
    variant_name: str,
    ttnn_fn,
    golden_fn,
    device,
    out_dir: Path,
    lo: float = float("-inf"),
    hi: float = float("inf"),
):
    import ttnn

    TENSOR_HEIGHT = 2**9
    TENSOR_WIDTH = 2**7
    size = [TENSOR_HEIGHT, TENSOR_WIDTH]

    # All 65536 bf16 values
    input_np = np.arange(0, 2**16, dtype=np.uint32).astype(np.uint16)
    torch_val = torch.from_numpy(input_np.view(np.int16)).reshape(size)
    x_bf16 = torch_val.view(torch.bfloat16)

    # Mask to valid domain, excluding subnormals (TT hardware flushes them to 0)
    MIN_NORMAL_BF16 = 2**-126
    mask = torch.ones(TENSOR_HEIGHT * TENSOR_WIDTH, dtype=torch.bool)
    x_flat = x_bf16.flatten().to(torch.float32)
    mask &= (x_flat == 0.0) | (x_flat.abs() >= MIN_NORMAL_BF16)
    if lo != float("-inf"):
        mask &= x_flat >= lo
    if hi != float("inf"):
        mask &= x_flat <= hi

    x_flat_valid = x_bf16.flatten()[mask]
    if x_flat_valid.numel() == 0:
        print(f"  {TERM_RED}No valid bf16 inputs for {op_name}/{variant_name}{TERM_RESET}")
        return
    # Pad to multiple of TENSOR_WIDTH so tiling works even for narrow domains
    pad = (-x_flat_valid.numel()) % TENSOR_WIDTH
    if pad > 0:
        # Repeat data as needed to fill the padding
        pad_data = x_flat_valid.repeat((pad // max(x_flat_valid.numel(), 1)) + 1)[:pad]
        x_flat_valid = torch.cat([x_flat_valid, pad_data])
    x_valid = x_flat_valid.reshape(-1, TENSOR_WIDTH)

    actual_h = x_valid.shape[0]

    x_f64 = x_valid.to(torch.float64)
    with torch.no_grad():
        golden_f64 = torch.zeros_like(x_f64)
        golden_f64 = golden_fn(x_f64, out=golden_f64)

    ttnn_in = ttnn.from_torch(x_valid, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_result = ttnn_fn(ttnn_in)

    calc = ttnn.to_torch(ttnn_result)

    # group_size=1: store one row per individual bf16 input value for accurate per-point data
    df = compare_with_golden(x_valid, golden_f64, calc, group_size=1)
    df["op"] = op_name
    df["variant"] = variant_name
    df["dtype"] = "bf16"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{variant_name}.csv"
    df.to_csv(out_path, na_rep="NaN", index_label="index")
    print(f"  {TERM_GREEN}Saved {out_path} ({len(df)} rows){TERM_RESET}")


# ---------------------------------------------------------------------------
# FP32 measurement (stratified over valid range)
# ---------------------------------------------------------------------------

def measure_fp32(
    op_name: str,
    variant_name: str,
    ttnn_fn,
    golden_fn,
    device,
    out_dir: Path,
    lo: float = float("-inf"),
    hi: float = float("inf"),
):
    import ttnn

    BATCH = 2**6
    HEIGHT = 2**9
    WIDTH = 2**7
    GROUP_SIZE = HEIGHT

    shape = [BATCH, HEIGHT, WIDTH]
    num_elements = BATCH * HEIGHT * WIDTH
    num_tensors = 2**32 // num_elements

    all_dfs = []
    tensor = torch.arange(0, num_elements, dtype=torch.int64).reshape(shape)
    increment = num_elements

    print(f"    DEBUG: Starting fp32 measurement, num_tensors={num_tensors}", flush=True)

    for i in range(num_tensors):
        if i % 10 == 0:
            print(f"    DEBUG: Tensor {i}/{num_tensors}", flush=True)

        x_f32 = tensor.to(torch.int32).view(torch.float32)

        # Filter to valid range, excluding subnormals
        MIN_NORMAL_F32 = 2**-126
        mask = torch.ones(x_f32.numel(), dtype=torch.bool)
        x_flat = x_f32.flatten()
        mask &= (x_flat == 0.0) | (x_flat.abs() >= MIN_NORMAL_F32)
        if lo != float("-inf"):
            mask &= x_flat >= lo
        if hi != float("inf"):
            mask &= x_flat <= hi
        x_valid = x_flat[mask]

        if x_valid.numel() == 0:
            tensor += increment
            continue

        # Pad to multiple of WIDTH for tiling
        pad = (-x_valid.numel()) % WIDTH
        if pad > 0:
            # Repeat data as needed to fill the padding
            pad_data = x_valid.repeat((pad // max(x_valid.numel(), 1)) + 1)[:pad]
            x_valid = torch.cat([x_valid, pad_data])
        x_2d = x_valid.reshape(-1, WIDTH)

        print(f"      DEBUG: Golden computation for tensor {i}...", flush=True)
        x_f64 = x_2d.to(torch.float64)
        with torch.no_grad():
            golden_f64 = torch.zeros_like(x_f64)
            golden_f64 = golden_fn(x_f64, out=golden_f64)
        print(f"      DEBUG: Golden done, ttnn compute...", flush=True)

        ttnn_in = ttnn.from_torch(x_2d, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        ttnn_result = ttnn_fn(ttnn_in)
        calc = ttnn.to_torch(ttnn_result)
        print(f"      DEBUG: TTNN done, comparing...", flush=True)

        gs = min(GROUP_SIZE, x_2d.shape[-1])
        df = compare_with_golden(x_2d, golden_f64, calc, group_size=gs)
        all_dfs.append(df)
        print(f"      DEBUG: Tensor {i} complete, {len(df)} rows", flush=True)

        tensor += increment

    if not all_dfs:
        print(f"  {TERM_RED}No data for {op_name}/{variant_name} (fp32){TERM_RESET}")
        return

    result = pd.concat(all_dfs, ignore_index=True)
    result["op"] = op_name
    result["variant"] = variant_name
    result["dtype"] = "fp32"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{variant_name}.csv"
    result.to_csv(out_path, na_rep="NaN", index_label="index")
    print(f"  {TERM_GREEN}Saved {out_path} ({len(result)} rows){TERM_RESET}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_measurement(op_names: list[str], dtypes: list[str], arch: str, out_root: Path):
    import ttnn
    from ops_registry import get_registry, get_op

    device = ttnn.open_device(device_id=0)
    reg = get_registry()

    try:
        for op_name in op_names:
            if op_name not in reg:
                print(f"{TERM_RED}Unknown op: {op_name}{TERM_RESET}")
                continue

            entry = get_op(op_name)
            print(f"\n=== {op_name} ({entry.category}) ===")

            for dtype in dtypes:
                for variant in entry.variants:
                    vname = variant.params_desc.replace(" ", "_").replace(",", "_").replace("=", "")
                    out_dir = out_root / arch / dtype / op_name
                    print(f"  [{dtype}] variant={vname}")
                    try:
                        if dtype == "bf16":
                            measure_bf16(
                                op_name, vname,
                                variant.ttnn_fn, variant.golden_fn,
                                device, out_dir,
                                lo=entry.input_range.lo,
                                hi=entry.input_range.hi,
                            )
                        elif dtype == "fp32":
                            measure_fp32(
                                op_name, vname,
                                variant.ttnn_fn, variant.golden_fn,
                                device, out_dir,
                                lo=entry.input_range.lo,
                                hi=entry.input_range.hi,
                            )
                    except Exception:
                        print(f"  {TERM_RED}Failed: {traceback.format_exc()}{TERM_RESET}")
    finally:
        ttnn.close_device(device)


def parse_args() -> argparse.Namespace:
    from ops_registry import list_op_names

    p = argparse.ArgumentParser(description="Measure ttnn eltwise op accuracy")
    p.add_argument("--arch", required=True, choices=["wh", "bh"],
                   help="Target architecture")
    p.add_argument("--ops", default="all",
                   help="Comma-separated op names, or 'all'")
    p.add_argument("--category", choices=["unary", "binary", "unary_bw"],
                   help="Filter by op category (when --ops=all)")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "both"],
                   help="Data type(s) to measure")
    p.add_argument("--output-dir", default=str(DATA_DIR),
                   help="Root output directory (default: data/)")
    return p.parse_args()


def main():
    # Add scripts/ to path so ops_registry can be imported
    sys.path.insert(0, str(Path(__file__).parent))
    args = parse_args()

    from ops_registry import list_op_names

    if args.ops == "all":
        op_names = list_op_names(category=args.category)
    else:
        op_names = [o.strip() for o in args.ops.split(",")]

    dtypes = ["bf16", "fp32"] if args.dtype == "both" else [args.dtype]
    out_root = Path(args.output_dir)

    print(f"Measuring {len(op_names)} ops, dtypes={dtypes}, arch={args.arch}")
    print(f"Output: {out_root}")
    run_measurement(op_names, dtypes, args.arch, out_root)


if __name__ == "__main__":
    main()
