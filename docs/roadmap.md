# Roadmap

What GitHub readers and people running `measure` / `compare` / `check` get today, and
what the next published sweep will change. Kernel-generating loops are out of scope here.

## Published report vs the tool

The committed `report_index.json` and `reports/**` were scored before fractional ULP,
`faithful`, `rounded_frac`, and `bias_ulp`. The CSVs currently next to this checkout
predate that schema (`n_rounded`, `ulp_signed` missing), so `charts` cannot backfill —
only a new `measure` pass writes those columns.

Until that sweep is published:

- Pages and `ask.md` quote the stored verdicts and the four figures the index actually
  holds: `max_ulp`, `mean_ulp`, `usable_to`, `max_abs`, plus timings.
- `compare` / `check` still score `max_ulp` between two pre-`rounded_frac` indexes.
- `compare` / `check` will **not** treat `max_ulp` 0 → 0.5 as a regression when only the
  candidate has `rounded_frac`. That is the wider-reference definition, not a worse kernel.

The first full run on the new metric will move nearly every `max_ulp`. Reset the baseline
in that same commit; do not bisect tt-metal for it.

## Accuracy — open

| Gap | Size |
|---|---|
| row-major layout — 160 of 215 WH ops accept it, none measured | largest untested surface |
| exhaustive binary and ternary — fp32 pairs and all ternary sweeps sample, so those maxima are lower bounds | large |
| shape — one tile-aligned block; `check` requires tiling invariance, no published figure varies it | large |
| scalar breadth — 26 ops take one, 11 measured at more than one value | medium |
| integer eltwise under exact-match scoring: `bitwise_*`, shifts, `gcd`, `lcm` | medium |
| `bitcast` (uncallable golden signature), `where_bw` (fed a float where it wants a bool) | small |

## Performance — open

| Gap | Why |
|---|---|
| sweep the size | one 4096² point cannot separate dispatch overhead from throughput. Fit `t = a + b·N` |
| subtract a floor | time `identity`/`abs` moving the same bytes; everything above is the op's SFPU work |
| find the memory cliff | if bandwidth-bound, extra SFPU instructions are free until they exceed the shadow |
| host-aware `perf-diff` | already refuses two hosts; `check --perf` already diffs only on the same machine |

`us_median` today conflates dispatch, memory traffic, and SFPU work. Only the third moves
when a kernel changes. Size sweep + floor is what lets a page state a budget instead of a
single µs.

## Sequencing

1. Next complete `accuracy-report` sweep, then reset the published baseline in one commit.
2. Perf: size sweep and floor subtraction (same host, same run).
3. Whether these ops are bandwidth-bound, and where the cliff sits per op.
4. Row-major layout, the largest remaining accuracy surface.
