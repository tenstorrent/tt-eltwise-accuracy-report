#!/usr/bin/env python3
"""
Fix ULP error columns in bf16 CSVs.

Bug: compare_with_golden() converted `calculated` to float32 before saving the
dtype, so gold_downcast was cast to float32 and ULP was computed with 23-bit
fp32 mantissa instead of 7-bit bf16 mantissa.

Since ULP_bf16(x) = 2^16 × ULP_fp32(x) for any normal value x (same exponent,
16-bit mantissa difference), all bf16 ULP error columns are inflated by exactly
65536. The fix is a simple division — no hardware needed.

fp32 CSVs are unaffected (correct dtype was used).

Usage:
    python recompute_metrics.py              # fix all bf16 CSVs in data/
    python recompute_metrics.py --arch wh
    python recompute_metrics.py --dry-run    # show what would change, no writes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"

BF16_ULP_CORRECTION = 2 ** 16   # = 65536
ULP_COLUMNS = ["ulp_error"]


def fix_csv(csv_path: Path, dry_run: bool) -> bool:
    df = pd.read_csv(csv_path, index_col="index")
    if not all(c in df.columns for c in ULP_COLUMNS):
        return False

    old_max = df["ulp_error"].max()
    df[ULP_COLUMNS] = df[ULP_COLUMNS] / BF16_ULP_CORRECTION
    new_max = df["ulp_error"].max()

    print(f"  {csv_path.relative_to(REPO_ROOT)}: max_ulp {old_max:.3g} → {new_max:.3g}")

    if not dry_run:
        df.to_csv(csv_path, na_rep="NaN", index_label="index")
    return True


def main():
    p = argparse.ArgumentParser(description="Fix bf16 ULP columns (÷65536)")
    p.add_argument("--arch", help="Filter by architecture")
    p.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = p.parse_args()

    archs = [args.arch] if args.arch else sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    changed = total = 0

    for arch in archs:
        bf16_dir = DATA_DIR / arch / "bf16"
        if not bf16_dir.is_dir():
            continue
        for csv_path in sorted(bf16_dir.rglob("*.csv")):
            total += 1
            if fix_csv(csv_path, args.dry_run):
                changed += 1

    action = "would fix" if args.dry_run else "fixed"
    print(f"\n{action} {changed}/{total} bf16 CSVs")


if __name__ == "__main__":
    main()
