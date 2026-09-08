"""ttnn-accuracy: measure eltwise op accuracy, render charts, regenerate the report tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from ttnn_accuracy.paths import DATA_DIR, INDEX_FILE

CATEGORIES = [f"{n}{suffix}" for n in ("unary", "binary", "ternary") for suffix in ("", "_bw")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ttnn-accuracy", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="rebuild stats/ops_manifest.json from ttnn")
    discover.add_argument("--include-experimental", action="store_true")

    sub.add_parser("derive", help="add per-dtype domain bounds to the manifest")

    probe = sub.add_parser("probe", help="record which dtype and layout each op accepts")
    probe.add_argument("--device-id", type=int, default=0)

    measure = sub.add_parser("measure", help="run accuracy sweeps on a device")
    measure.add_argument("--arch", required=True, choices=["wh", "bh"])
    measure.add_argument("--ops", default="all", help="comma-separated op names, or 'all'")
    measure.add_argument("--category", choices=CATEGORIES, help="filter when --ops=all")
    measure.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "both"])
    measure.add_argument(
        "--params",
        help="measure named ops at a different scalar, e.g. relu_max=6,leaky_relu=0.2 — "
        "the value only; the op's own parameter name comes from ops/overrides.py",
    )
    measure.add_argument("--output-dir", type=Path, default=DATA_DIR)
    measure.add_argument("--device-id", type=int, default=0)

    check = sub.add_parser(
        "check", help="measure named ops and diff them against the published report"
    )
    check.add_argument("--arch", required=True, choices=["wh", "bh"])
    check.add_argument("--ops", required=True, help="comma-separated op names")
    check.add_argument("--dtype", default="both", choices=["bf16", "fp32", "both"])
    check.add_argument("--params", help="measure at a different scalar, e.g. relu_max=6")
    check.add_argument("--device-id", type=int, default=0)
    check.add_argument("--perf", action="store_true", help="also time them, and diff if same host")
    check.add_argument(
        "--max-ulp", type=float, help="fail any variant worse than this, whatever the baseline says"
    )
    check.set_defaults(category=None)

    perf = sub.add_parser("perf", help="time each op on a device; never scored, host-specific")
    perf.add_argument("--arch", required=True, choices=["wh", "bh"])
    perf.add_argument("--ops", default="all", help="comma-separated op names, or 'all'")
    perf.add_argument("--category", choices=CATEGORIES, help="filter when --ops=all")
    perf.add_argument("--dtype", default="both", choices=["bf16", "fp32", "both"])
    perf.add_argument("--device-id", type=int, default=0)
    perf.add_argument("--out", type=Path, help="where to write the timings (default stats/perf/)")
    perf.set_defaults(params=None)  # so _selection reads args.params for both commands

    perf_diff = sub.add_parser(
        "perf-diff", help="two timing files → what got slower; exit 1 when anything did"
    )
    perf_diff.add_argument("baseline", type=Path)
    perf_diff.add_argument("candidate", type=Path)
    perf_diff.add_argument("--findings", type=Path, help="also write the diff as a markdown page")

    charts = sub.add_parser("charts", help="render SVG charts from measured data")
    charts.add_argument("--arch")
    charts.add_argument("--dtype")
    charts.add_argument("--op")

    report = sub.add_parser("report", help="regenerate the markdown report tree")
    report.add_argument("--arch")
    report.add_argument("--dtype")
    report.add_argument("--categories", default="unary", help="comma-separated")

    refine = sub.add_parser(
        "refine", help="search exhaustively around a sampled worst point; exit counts looser bounds"
    )
    refine.add_argument("--arch", required=True, choices=["wh", "bh"])
    refine.add_argument("--ops", default="all", help="comma-separated op names, or 'all'")
    refine.add_argument("--category", choices=CATEGORIES, help="filter when --ops=all")
    refine.add_argument("--dtype", default="both", choices=["bf16", "fp32", "both"])
    refine.add_argument("--device-id", type=int, default=0)
    refine.add_argument("--findings", type=Path, help="also write the looser bounds as a page")
    refine.set_defaults(params=None)

    hist = sub.add_parser("history", help="when each number moved, from the index's git history")
    hist.add_argument("--op", help="one op name (default every op)")
    hist.add_argument("--findings", type=Path, help="also write the history as a markdown page")

    compare = sub.add_parser("compare", help="diff two report indexes; exit 1 on regression")
    compare.add_argument(
        "baseline", type=Path, help="report_index.json measured on the older build"
    )
    compare.add_argument("candidate", type=Path, nargs="?", default=INDEX_FILE)
    compare.add_argument("--findings", type=Path, help="also write the diff as a markdown page")

    return parser


def _values(params: str | None) -> dict[str, float | int]:
    """`relu_max=6` → {"relu_max": 6}; typed into a web form, so a mistake answers itself."""
    values: dict[str, float | int] = {}
    for pair in (p.strip() for p in (params or "").split(",") if p.strip()):
        op, _, raw = pair.partition("=")
        if not raw:
            raise SystemExit(f"--params: expected op=value, got `{pair}`")
        try:
            values[op.strip()] = int(raw) if raw.strip().lstrip("+-").isdigit() else float(raw)
        except ValueError:
            raise SystemExit(f"--params: `{raw.strip()}` is not a number, in `{pair}`") from None
    return values


def _selection(args) -> tuple[list, list[str], int]:
    """The ops and dtypes asked for, refusals logged — shared so measure and perf agree."""
    from ttnn_accuracy.ops import plan

    names = None if args.ops == "all" else [o.strip() for o in args.ops.split(",")]
    specs, problems = plan.resolve(names, args.category, args.arch, _values(args.params))
    for problem in problems:
        logger.error(problem)
    dtypes = ["bf16", "fp32"] if args.dtype == "both" else [args.dtype]
    return specs, dtypes, len(problems)


def _onboard(args) -> int:
    """Onboard ops the manifest lacks, before `check` takes the device — `probe` opens its own."""
    from ttnn_accuracy.ops.manifest import derive_domains, discover, load, probe_layouts

    known = {e["name"] for e in load()["ops"].values()}
    missing = [o.strip() for o in args.ops.split(",") if o.strip() not in known]
    if not missing:
        return 0

    logger.info("{} not in the manifest — discovering, deriving, probing", ", ".join(missing))
    for step in (discover, derive_domains):
        if rc := step():
            return rc
    if rc := probe_layouts(args.device_id):
        return rc

    from ttnn_accuracy.ops import plan

    plan._manifest.cache_clear()  # resolve() would otherwise read the pre-onboarding copy
    plan.describe.cache_clear()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    match args.command:
        case "discover":
            from ttnn_accuracy.ops.manifest import discover

            return discover(args.include_experimental)

        case "derive":
            from ttnn_accuracy.ops.manifest import derive_domains

            return derive_domains()

        case "probe":
            from ttnn_accuracy.ops.manifest import probe_layouts

            return probe_layouts(args.device_id)

        case "measure":
            from ttnn_accuracy.measure.runner import measure

            specs, dtypes, problems = _selection(args)
            if not specs:
                return 1
            return problems + measure(specs, dtypes, args.arch, args.output_dir, args.device_id)

        case "check":
            from ttnn_accuracy.measure.runner import check

            if rc := _onboard(args):
                return rc
            specs, dtypes, problems = _selection(args)
            if not specs:
                return 1
            return problems + check(
                specs, dtypes, args.arch, args.device_id, args.perf, args.max_ulp
            )

        case "perf":
            from ttnn_accuracy.measure.perf import measure_perf

            specs, dtypes, problems = _selection(args)
            if not specs:
                return 1
            return problems + measure_perf(specs, dtypes, args.arch, args.device_id, args.out)

        case "perf-diff":
            from ttnn_accuracy.measure.perf import perf_diff

            return perf_diff(args.baseline, args.candidate, args.findings)

        case "charts":
            from ttnn_accuracy.report.charts import generate_charts

            return generate_charts(args.arch, args.dtype, args.op)

        case "report":
            from ttnn_accuracy.report.pages import generate_reports

            cats = [c.strip() for c in args.categories.split(",")]
            return generate_reports(args.arch, args.dtype, cats)

        case "refine":
            from ttnn_accuracy.measure.runner import refine

            specs, dtypes, problems = _selection(args)
            if not specs:
                return 1
            return problems + refine(specs, dtypes, args.arch, args.device_id, args.findings)

        case "history":
            from ttnn_accuracy.report.compare import history

            return history(args.op, args.findings)

        case "compare":
            from ttnn_accuracy.report.compare import compare

            return compare(args.baseline, args.candidate, args.findings)


if __name__ == "__main__":  # `python -m ttnn_accuracy.cli`, for callers that must not install
    raise SystemExit(main())
