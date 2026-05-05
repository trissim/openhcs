#!/usr/bin/env python
"""Regenerate cppipe throughput figures from existing benchmark CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.reports.cppipe_scaling_figures import generate_cppipe_scaling_figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--throughput-batches",
        type=Path,
        required=True,
        help="Existing throughput_batches.csv produced by the throughput benchmark.",
    )
    parser.add_argument(
        "--native-summary-csv",
        type=Path,
        required=True,
        help="Existing CP-vs-OpenHCS summary.csv containing native CP timings.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        default=None,
        help="Output format. Repeat for multiple formats. Defaults to png and svg.",
    )
    parser.add_argument(
        "--no-average",
        action="store_true",
        help="Do not append the aggregate Average group.",
    )
    parser.add_argument(
        "--wrap-after",
        type=int,
        default=14,
        help="Split long grouped charts into two stacked panels above this count.",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        action="append",
        default=None,
        help="Only plot these sample counts. Repeat to include multiple counts.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        action="append",
        default=None,
        help="Only plot these OpenHCS job counts. Repeat to include multiple counts.",
    )
    args = parser.parse_args()

    outputs = generate_cppipe_scaling_figures(
        args.throughput_batches,
        output_dir=args.output_dir,
        native_summary_csv=args.native_summary_csv,
        output_formats=(
            tuple(args.formats) if args.formats is not None else ("png", "svg")
        ),
        include_average=not args.no_average,
        wrap_after=args.wrap_after,
        replica_counts=tuple(args.replicas or ()),
        worker_counts=tuple(args.workers or ()),
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
