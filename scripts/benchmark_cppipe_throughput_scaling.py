#!/usr/bin/env python
"""Benchmark throughput scaling for converted CellProfiler pipelines."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from benchmark.runtime_env import configure_headless_cpu_benchmark_runtime


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent converted-cppipe OpenHCS jobs sequentially and "
            "concurrently, then generate throughput-scaling figures."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_names",
        help="Case name to include. Repeat to select multiple cases. Defaults to all.",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        action="append",
        default=None,
        help=(
            "Independent sample/job count to run for each case. Repeat for "
            "sample-count scaling. Defaults to 4."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        action="append",
        dest="worker_counts",
        required=True,
        help="Concurrent independent OpenHCS jobs. Repeat for scaling curve.",
    )
    parser.add_argument("--native-reference-root", type=Path)
    parser.add_argument(
        "--native-summary-csv",
        type=Path,
        help=(
            "Existing CP-vs-OpenHCS summary.csv used to compute throughput "
            "speedup versus native CellProfiler."
        ),
    )
    parser.add_argument("--require-reference", action="store_true")
    parser.add_argument(
        "--skip-equivalence",
        action="store_true",
        help="Measure throughput only; do not compare against native references.",
    )
    parser.add_argument("--discard-openhcs-outputs", action="store_true")
    parser.add_argument("--openhcs-timeout-seconds", type=float)
    parser.add_argument(
        "--format",
        action="append",
        dest="output_formats",
        default=None,
        help="Figure format. Repeat for multiple formats. Defaults to png and svg.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("OPENHCS_BENCHMARK_LOG_LEVEL", "WARNING"),
    )
    args = parser.parse_args()

    configure_headless_cpu_benchmark_runtime(args.log_level)

    from benchmark.reports.cppipe_scaling_figures import (
        generate_cppipe_scaling_figures,
    )
    from benchmark.throughput_scaling import BATCH_ROWS_CSV
    from benchmark.throughput_scaling import load_scaling_cases
    from benchmark.throughput_scaling import run_throughput_scaling_suite

    cases = load_scaling_cases(
        args.manifest,
        case_names=tuple(args.case_names or ()),
        native_reference_root=args.native_reference_root,
    )
    if args.require_reference and not args.skip_equivalence:
        missing_references = tuple(
            case.name for case in cases if case.equivalence_reference_output_dir is None
        )
        if missing_references:
            raise FileNotFoundError(
                "Missing required native references: "
                + ", ".join(missing_references)
            )
    observations = run_throughput_scaling_suite(
        cases,
        output_root=args.output_dir,
        worker_counts=tuple(dict.fromkeys(args.worker_counts)),
        replicas=tuple(dict.fromkeys(args.replicas or (4,))),
        verify_equivalence=not args.skip_equivalence,
        compare_image_outputs=False,
        discard_outputs=args.discard_openhcs_outputs,
        openhcs_timeout_seconds=args.openhcs_timeout_seconds,
    )
    figure_paths = generate_cppipe_scaling_figures(
        args.output_dir / BATCH_ROWS_CSV,
        output_dir=args.output_dir / "figures",
        native_summary_csv=args.native_summary_csv,
        output_formats=tuple(args.output_formats or ("png", "svg")),
    )
    print(f"cases={len(cases)}")
    print(f"batch_observations={len(observations)}")
    print(f"summary_csv={args.output_dir / 'throughput_summary.csv'}")
    print(f"batches_csv={args.output_dir / BATCH_ROWS_CSV}")
    print(f"jobs_csv={args.output_dir / 'throughput_jobs.csv'}")
    print(f"figures={args.output_dir / 'figures'}")
    for path in figure_paths:
        print(path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
