#!/usr/bin/env python
"""Generate CP/OH cppipe benchmark comparison figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.reports.cppipe_figures import generate_cppipe_benchmark_figures
from benchmark.reports.cppipe_figures import parse_summary_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help=(
            "OpenHCS variant summary as LABEL=/path/to/summary.csv. "
            "Can be repeated for OH1..OH4."
        ),
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
        "--group-width-inches",
        type=float,
        default=0.82,
        help=(
            "Horizontal inches allocated per pipeline group. "
            "Use about 0.41 for twice-dense single-row lab-meeting figures."
        ),
    )
    args = parser.parse_args()

    outputs = generate_cppipe_benchmark_figures(
        tuple(parse_summary_source(value) for value in args.summary),
        output_dir=args.output_dir,
        output_formats=(
            tuple(args.formats) if args.formats is not None else ("png", "svg")
        ),
        include_average=not args.no_average,
        wrap_after=args.wrap_after,
        group_width_inches=args.group_width_inches,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
