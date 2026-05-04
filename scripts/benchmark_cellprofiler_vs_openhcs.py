#!/usr/bin/env python
"""Run and plot native CellProfiler versus OpenHCS comparison benchmarks."""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from nominal_refactor_advisor.record_algebra import product_record

CASE_NAME_FIELD = "case_name"


BarSeries = product_record(
    "BarSeries",
    "values: tuple[float, ...]; color: str; label: str | None; offset: float; width: float",
    defaults={"label": None, "offset": 0.0, "width": 0.7},
    doc="Declarative bar-series rendering contract.",
    module_name=__name__,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate CP-vs-OpenHCS runtime and parity benchmark artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run benchmark cases.")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--suite-id")
    run_parser.add_argument("--repeats", type=int, default=1)
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

    plot_parser = subparsers.add_parser("plot", help="Plot benchmark CSV output.")
    plot_parser.add_argument("--summary-csv", type=Path, required=True)
    plot_parser.add_argument("--output-dir", type=Path, required=True)
    plot_parser.set_defaults(handler=_plot_command)
    args = parser.parse_args()
    return args.handler(args)


def _run_command(args: argparse.Namespace) -> int:
    from benchmark.cellprofiler_comparison import (
        load_comparison_cases,
        run_comparison_suite,
    )

    _configure_reproducible_runtime_env()
    suite_id = args.suite_id or datetime.now().strftime("cp_vs_openhcs_%Y%m%d_%H%M%S")
    observations = run_comparison_suite(
        load_comparison_cases(args.manifest),
        output_root=args.output_dir,
        suite_id=suite_id,
        repeats=args.repeats,
        reuse_openhcs_cache=not args.force_openhcs_run,
        speedup_target=args.speedup_target,
    )
    print(f"suite_id={suite_id}")
    print(f"observations={len(observations)}")
    print(f"summary_csv={args.output_dir / 'summary.csv'}")
    return 0


def _plot_command(args: argparse.Namespace) -> int:
    _configure_reproducible_runtime_env()
    plot_summary(args.summary_csv, args.output_dir)
    print(f"figures={args.output_dir}")
    return 0


def plot_summary(summary_csv: Path, output_dir: Path) -> None:
    """Create lab-meeting-ready runtime and parity figures from summary CSV."""
    import matplotlib.pyplot as plt

    rows = _summary_rows(summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [row[CASE_NAME_FIELD] for row in rows]
    native = [float(row["median_native_execution_seconds"]) for row in rows]
    openhcs = [float(row["median_openhcs_execution_seconds"]) for row in rows]
    speedups = [float(row["median_speedup"]) for row in rows]
    accuracy = [float(row["min_parity_accuracy"]) * 100.0 for row in rows]

    fig, axis = plt.subplots(figsize=(max(8.0, len(names) * 0.55), 4.8))
    x_positions = range(len(names))
    width = 0.38
    _plot_bar_series(
        axis,
        names=names,
        x_positions=tuple(x_positions),
        series=(
            BarSeries(
                tuple(native),
                "#333333",
                label="CellProfiler",
                offset=-width / 2,
                width=width,
            ),
            BarSeries(
                tuple(openhcs),
                "#0f8b8d",
                label="OpenHCS",
                offset=width / 2,
                width=width,
            ),
        ),
    )
    axis.set_ylabel("Execution time (s)")
    axis.set_title("Single-thread execution runtime")
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(names, rotation=45, ha="right")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_cellprofiler_vs_openhcs.png", dpi=240)
    fig.savefig(output_dir / "runtime_cellprofiler_vs_openhcs.svg")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(max(8.0, len(names) * 0.55), 4.8))
    bars = _plot_bar_series(
        axis,
        names=names,
        x_positions=tuple(range(len(names))),
        series=(BarSeries(tuple(speedups), "#d95f02"),),
    )[0]
    axis.axhline(1.0, color="#333333", linewidth=1.0)
    axis.set_ylabel("Speedup vs CellProfiler (x)")
    axis.set_title("OpenHCS execution speedup")
    axis.set_xticklabels(names, rotation=45, ha="right")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, speedups, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "speedup_openhcs_vs_cellprofiler.png", dpi=240)
    fig.savefig(output_dir / "speedup_openhcs_vs_cellprofiler.svg")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(max(8.0, len(names) * 0.55), 3.8))
    _plot_bar_series(
        axis,
        names=names,
        x_positions=tuple(range(len(names))),
        series=(BarSeries(tuple(accuracy), "#1b9e77"),),
    )
    axis.set_ylim(0, 105)
    axis.set_ylabel("Parity pass rate (%)")
    axis.set_title("Semantic parity at 1e-6 numeric tolerance")
    axis.set_xticklabels(names, rotation=45, ha="right")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "parity_accuracy.png", dpi=240)
    fig.savefig(output_dir / "parity_accuracy.svg")
    plt.close(fig)


def _plot_bar_series(
    axis,
    *,
    names: Sequence[str],
    x_positions: tuple[int, ...],
    series: tuple[BarSeries, ...],
):
    """Render bar series through one declarative plotting surface."""
    containers = []
    for spec in series:
        containers.append(
            axis.bar(
                [x + spec.offset for x in x_positions],
                spec.values,
                width=spec.width,
                label=spec.label,
                color=spec.color,
            )
        )
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(names, rotation=45, ha="right")
    return tuple(containers)


def _summary_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        CASE_NAME_FIELD,
        "median_native_execution_seconds",
        "median_openhcs_execution_seconds",
        "median_speedup",
        "min_parity_accuracy",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Summary CSV missing columns: {sorted(missing)!r}")
    return rows


def _configure_reproducible_runtime_env() -> None:
    """Set defaults that make GUI/logging imports deterministic in headless runs."""
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("XDG_DATA_HOME", "/tmp/openhcs-benchmark-xdg-data")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/openhcs-benchmark-xdg-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/openhcs-benchmark-mpl")


if __name__ == "__main__":
    raise SystemExit(main())
