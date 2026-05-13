#!/usr/bin/env python
"""Benchmark converted cppipes using native OpenHCS well-level multiprocessing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from benchmark.runtime_env import configure_headless_cpu_benchmark_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="case_names")
    parser.add_argument("--wells", type=int, action="append")
    parser.add_argument("--workers", type=int, action="append")
    parser.add_argument(
        "--mode",
        action="append",
        choices=("1w_1t", "8w_2c", "12w_3c", "16w_4c"),
        help=(
            "Paired native multiprocessing preset. May be repeated. "
            "Equivalent to 1 well/1 thread, 8 wells/2 workers, "
            "12 wells/3 workers, or 16 wells/4 workers."
        ),
    )
    parser.add_argument(
        "--native-summary-csv",
        type=Path,
        help=(
            "Official single-sample CP-vs-OpenHCS summary.csv used to project "
            "native CP execution time by multiplying each case baseline by the "
            "well count."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing well_throughput.csv rows and run only missing case/mode pairs.",
    )
    parser.add_argument(
        "--rerun-missing-memory",
        action="store_true",
        help=(
            "When used with --resume, rerun existing rows whose peak_memory_mb "
            "is blank and replace them with memory-recorded observations."
        ),
    )
    parser.add_argument(
        "--max-memory-mb",
        type=float,
        help=(
            "Skip the current case/mode by terminating OpenHCS worker children "
            "when process-tree RSS exceeds this limit."
        ),
    )
    parser.add_argument(
        "--skip-case-mode",
        action="append",
        metavar="CASE:MODE",
        help=(
            "Skip one exact case/mode observation while resuming, for example "
            "ExampleImagingFlowCytometryObjectsInGrid:16w_4c."
        ),
    )
    parser.add_argument(
        "--start-method",
        choices=("fork", "spawn", "forkserver"),
        default="fork",
        help="Multiprocessing start method used by OpenHCS worker execution.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("OPENHCS_BENCHMARK_LOG_LEVEL", "WARNING"),
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Generate throughput figures from well_throughput.csv after the run.",
    )
    parser.add_argument(
        "--figures-output-dir",
        type=Path,
        help="Directory for throughput figures. Defaults to OUTPUT_DIR/figures.",
    )
    args = parser.parse_args()

    configure_headless_cpu_benchmark_runtime(args.log_level)

    from benchmark.well_throughput_scaling import WELL_THROUGHPUT_ROWS_CSV
    from benchmark.well_throughput_scaling import WellThroughputBenchmarkPlan
    from benchmark.well_throughput_scaling import WellThroughputObservationKey
    from benchmark.well_throughput_scaling import WellThroughputPreset
    from benchmark.well_throughput_scaling import native_execution_baselines_from_summary_csv
    from benchmark.well_throughput_scaling import read_well_throughput_csv
    from benchmark.well_throughput_scaling import generate_well_throughput_figures
    from benchmark.well_throughput_scaling import run_well_throughput_suite
    from openhcs.core.config import MultiprocessingStartMethod

    try:
        plan = WellThroughputBenchmarkPlan.from_requested_modes(
            presets=tuple(WellThroughputPreset(mode) for mode in args.mode or ()),
            well_counts=tuple(args.wells or ()),
            worker_counts=tuple(args.workers or ()),
            manifest_path=args.manifest,
        )
    except ValueError:
        parser.error(
            "Specify one or more --mode values, both --wells and --workers, "
            "or a manifest with well_throughput_modes."
        )
    start_method = MultiprocessingStartMethod(args.start_method)
    if args.mode and start_method is not MultiprocessingStartMethod.FORK:
        parser.error("Preliminary multi-core modes must use --start-method fork.")
    native_execution_baselines = (
        native_execution_baselines_from_summary_csv(args.native_summary_csv)
        if args.native_summary_csv is not None
        else None
    )
    csv_path = args.output_dir / WELL_THROUGHPUT_ROWS_CSV
    existing_results = read_well_throughput_csv(csv_path) if args.resume else ()
    try:
        skipped_observations = tuple(
            _parse_observation_key(raw_value)
            for raw_value in args.skip_case_mode or ()
        )
    except ValueError as exc:
        parser.error(str(exc))
    results = run_well_throughput_suite(
        args.manifest,
        output_root=args.output_dir,
        case_names=tuple(args.case_names or ()),
        well_counts=tuple(args.wells or ()),
        worker_counts=tuple(args.workers or ()),
        start_method=start_method,
        plan=plan,
        native_execution_baselines=native_execution_baselines,
        existing_results=existing_results,
        skipped_observations=skipped_observations,
        rerun_missing_memory=args.rerun_missing_memory,
        max_memory_mb=args.max_memory_mb,
    )
    print(f"observations={len(results)}")
    print(f"csv={csv_path}")
    if args.figures:
        figures_output_dir = args.figures_output_dir or args.output_dir / "figures"
        if args.native_summary_csv is not None:
            from benchmark.reports.cppipe_figures import SummarySource
            from benchmark.reports.cppipe_figures import generate_cppipe_benchmark_figures

            for output in generate_cppipe_benchmark_figures(
                (SummarySource("OH1", args.native_summary_csv),),
                output_dir=figures_output_dir,
            ):
                print(output)
        figure_outputs = generate_well_throughput_figures(
            csv_path,
            figures_output_dir,
        )
        print(f"figures={figures_output_dir}")
        for output in figure_outputs:
            print(output)
    return 0


def _parse_observation_key(raw_value: str):
    from benchmark.well_throughput_scaling import WellThroughputObservationKey

    case_name, separator, mode_name = raw_value.partition(":")
    if not separator or not case_name or not mode_name:
        raise ValueError(
            f"Expected CASE:MODE observation key, got {raw_value!r}."
        )
    return WellThroughputObservationKey(case_name, mode_name)


if __name__ == "__main__":
    raise SystemExit(main())
