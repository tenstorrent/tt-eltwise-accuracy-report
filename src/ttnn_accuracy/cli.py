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
    measure.add_argument("--output-dir", type=Path, default=DATA_DIR)
    measure.add_argument("--device-id", type=int, default=0)
    measure.add_argument(
        "--source",
        default="manifest",
        choices=["manifest", "registry"],
        help="where op definitions and bounds come from",
    )

    charts = sub.add_parser("charts", help="render SVG charts from measured data")
    charts.add_argument("--arch")
    charts.add_argument("--dtype")
    charts.add_argument("--op")

    report = sub.add_parser("report", help="regenerate the markdown report tree")
    report.add_argument("--arch")
    report.add_argument("--dtype")
    report.add_argument("--categories", default="unary", help="comma-separated")

    compare = sub.add_parser("compare", help="diff two report indexes; exit 1 on regression")
    compare.add_argument(
        "baseline", type=Path, help="report_index.json measured on the older build"
    )
    compare.add_argument("candidate", type=Path, nargs="?", default=INDEX_FILE)

    return parser


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
            from ttnn_accuracy.ops import plan

            names = None if args.ops == "all" else [o.strip() for o in args.ops.split(",")]
            specs, problems = plan.resolve(args.source, names, args.category)
            for problem in problems:
                logger.error(problem)
            if not specs:
                return 1
            dtypes = ["bf16", "fp32"] if args.dtype == "both" else [args.dtype]
            return len(problems) + measure(
                specs, dtypes, args.arch, args.output_dir, args.device_id
            )

        case "charts":
            from ttnn_accuracy.report.charts import generate_charts

            return generate_charts(args.arch, args.dtype, args.op)

        case "report":
            from ttnn_accuracy.report.pages import generate_reports

            cats = [c.strip() for c in args.categories.split(",")]
            return generate_reports(args.arch, args.dtype, cats)

        case "compare":
            from ttnn_accuracy.report.compare import compare

            return compare(args.baseline, args.candidate)
