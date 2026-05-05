#!/usr/bin/env python
"""Run and plot native CellProfiler versus OpenHCS comparison benchmarks."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from benchmark.runtime_env import configure_headless_cpu_benchmark_runtime


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate CP-vs-OpenHCS runtime and parity benchmark artifacts."
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("OPENHCS_BENCHMARK_LOG_LEVEL", "WARNING"),
        help="Python logging level for benchmark harness and OpenHCS runtime logs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run benchmark cases.")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--native-reference-root", type=Path)
    run_parser.add_argument("--discard-openhcs-outputs", action="store_true")
    run_parser.add_argument("--continue-on-error", action="store_true")
    run_parser.add_argument("--suite-id")
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.add_argument(
        "--log-level",
        default=os.environ.get("OPENHCS_BENCHMARK_LOG_LEVEL", "WARNING"),
        help="Python logging level for benchmark harness and OpenHCS runtime logs.",
    )
    run_parser.add_argument(
        "--speedup-target",
        type=float,
        default=5.0,
        help="Minimum acceptable OpenHCS speedup recorded in summary artifacts.",
    )
    run_parser.add_argument(
        "--force-openhcs-run",
        action="store_true",
        help="Disable OpenHCS benchmark/runtime execution cache reuse.",
    )
    run_parser.set_defaults(handler=_run_command)

    manifest_parser = subparsers.add_parser(
        "official-cp3-manifest",
        help="Build a comparison manifest from a local CellProfiler examples checkout.",
    )
    manifest_parser.add_argument(
        "--examples-root",
        type=Path,
        default=Path(os.environ.get("CELLPROFILER_EXAMPLES_ROOT", "/tmp/cellprofiler_examples")),
    )
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--value-only", action="store_true")
    manifest_parser.add_argument("--microscope-type")
    manifest_parser.add_argument("--cellprofiler-timeout-seconds", type=float)
    manifest_parser.add_argument(
        "--log-level",
        default=os.environ.get("OPENHCS_BENCHMARK_LOG_LEVEL", "WARNING"),
        help="Python logging level for benchmark harness and OpenHCS runtime logs.",
    )
    manifest_parser.set_defaults(handler=_official_cp3_manifest_command)

    plot_parser = subparsers.add_parser("plot", help="Plot benchmark CSV output.")
    plot_parser.add_argument("--summary-csv", type=Path, required=True)
    plot_parser.add_argument("--output-dir", type=Path, required=True)
    plot_parser.add_argument(
        "--log-level",
        default=os.environ.get("OPENHCS_BENCHMARK_LOG_LEVEL", "WARNING"),
        help="Python logging level for benchmark harness and OpenHCS runtime logs.",
    )
    plot_parser.set_defaults(handler=_plot_command)
    args = parser.parse_args()
    return args.handler(args)


def _run_command(args: argparse.Namespace) -> int:
    configure_headless_cpu_benchmark_runtime(args.log_level)
    from benchmark.cellprofiler_comparison import (
        load_comparison_cases,
        run_comparison_suite,
    )

    suite_id = args.suite_id or datetime.now().strftime("cp_vs_openhcs_%Y%m%d_%H%M%S")
    observations = run_comparison_suite(
        load_comparison_cases(args.manifest),
        output_root=args.output_dir,
        suite_id=suite_id,
        repeats=args.repeats,
        reuse_openhcs_cache=not args.force_openhcs_run,
        speedup_target=args.speedup_target,
        native_reference_root=args.native_reference_root,
        discard_openhcs_outputs=args.discard_openhcs_outputs,
        continue_on_error=args.continue_on_error,
    )
    print(f"suite_id={suite_id}")
    print(f"observations={len(observations)}")
    print(f"summary_csv={args.output_dir / 'summary.csv'}")
    return 0


def _official_cp3_manifest_command(args: argparse.Namespace) -> int:
    configure_headless_cpu_benchmark_runtime(args.log_level)
    cppipe_dir = args.examples_root / "CellProfiler3Pipelines"
    if not cppipe_dir.is_dir():
        raise FileNotFoundError(f"CellProfiler3Pipelines directory not found: {cppipe_dir}")
    cases = []
    for cppipe_path in sorted(cppipe_dir.glob("*.cppipe")):
        dataset_name = _official_cellprofiler3_source_name_for_pipeline(
            args.examples_root,
            cppipe_path.stem,
        )
        case: dict[str, object] = {
            "name": cppipe_path.stem,
            "dataset_path": str(args.examples_root / dataset_name),
            "cppipe_path": str(cppipe_path),
            "dataset_id": dataset_name,
            "value_only": args.value_only,
        }
        if args.microscope_type is not None:
            case["microscope_type"] = args.microscope_type
        if args.cellprofiler_timeout_seconds is not None:
            case["cellprofiler_timeout_seconds"] = args.cellprofiler_timeout_seconds
        cases.append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"cases": cases}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(str(args.output))
    print(f"cases={len(cases)}")
    return 0


def _official_cellprofiler3_source_name_for_pipeline(
    examples_root: Path,
    pipeline_name: str,
) -> str:
    candidate_names = (
        pipeline_name,
        pipeline_name.removesuffix("URL"),
        pipeline_name.split("_", maxsplit=1)[0],
        f"{pipeline_name}Images",
        pipeline_name.replace("ExampleUntangleAnd", "Example"),
    )
    for candidate_name in candidate_names:
        if candidate_name and (examples_root / candidate_name).exists():
            return candidate_name
    raise FileNotFoundError(
        f"No source directory found for official pipeline {pipeline_name!r} "
        f"under {examples_root}."
    )


def _plot_command(args: argparse.Namespace) -> int:
    configure_headless_cpu_benchmark_runtime(args.log_level)
    plot_summary(args.summary_csv, args.output_dir)
    print(f"figures={args.output_dir}")
    return 0


def plot_summary(summary_csv: Path, output_dir: Path) -> None:
    """Create lab-meeting-ready runtime, speedup, parity, and memory figures."""
    from benchmark.reports.cppipe_figures import SummarySource
    from benchmark.reports.cppipe_figures import generate_cppipe_benchmark_figures

    generate_cppipe_benchmark_figures(
        (SummarySource("OH1", summary_csv),),
        output_dir=output_dir,
    )

if __name__ == "__main__":
    raise SystemExit(main())
