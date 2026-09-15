"""One page a reader hands to any assistant: the contract inlined beside every number."""

from __future__ import annotations

from ttnn_accuracy.ops.plan import params_desc
from ttnn_accuracy.paths import CONTRACT_FILE, RUNS_KEY
from ttnn_accuracy.report.index import _get_index, _us

ASK_HEADER = """# Ask an assistant about these measurements

Give your assistant this page's URL, or paste the file, then ask in plain language —
"is `exp` usable on Wormhole in bfloat16?", "which ops are worse than 2 ULP in fp32?".
Everything needed to answer is below: the definitions, then every measured result.

Answering rules, in force for whoever reads this: quote the `Verdict` column rather than
judging the numbers yourself, name the architecture and dtype in every answer, and if a
variant is not in the table say so instead of extrapolating from a neighbouring one. A timing
is only comparable to another taken on the same host, which the "Measured against" table names.

"""


def _ask_row(arch: str, dtype: str, op: str, variant: str, s: dict) -> str:
    perf = s.get("perf")
    us, rate = _us(perf), f"{perf['melem_per_s']:.0f}" if perf else "—"
    return (
        f"| {arch} | {dtype} | `{op}` | `{params_desc(op, variant)}` | {s.get('verdict', '—')} "
        f"| {s.get('max_ulp', '—')} | {s.get('mean_ulp', '—')} | {s.get('usable_to', '—')} "
        f"| {us} | {rate} |"
    )


def ask_page() -> str:
    index = _get_index()
    rows = [
        _ask_row(arch, dtype, op, variant, s)
        for arch in sorted(k for k in index if k != RUNS_KEY)
        for dtype, ops in sorted(index[arch].items())
        for op, variants in sorted(ops.items())
        for variant, s in sorted(variants.items())
    ]
    builds = [
        f"| {arch} | {dtype} | `{run.get('tt_metal_commit')}` | {run.get('ttnn_version')} "
        f"| {run.get('host', '—')} |"
        for arch, dtypes in sorted(index.get(RUNS_KEY, {}).items())
        for dtype, run in sorted(dtypes.items())
    ]
    return "\n".join(
        [
            ASK_HEADER,
            "## Measured against\n",
            "| Arch | Dtype | tt-metal | ttnn | Timed on |",
            "|---|---|---|---|---|",
            *builds,
            "",
            CONTRACT_FILE.read_text().partition("\n")[2].strip(),
            "",
            f"## Results — {len(rows)} variants\n",
            "| Arch | Dtype | Op | Parameters | Verdict | Max ULP | Mean ULP | Usable to "
            "| µs | Melem/s |",
            "|---|---|---|---|---|---|---|---|---|---|",
            *rows,
            "",
        ]
    )
