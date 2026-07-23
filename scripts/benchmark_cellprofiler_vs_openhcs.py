#!/usr/bin/env python
"""Run and plot native CellProfiler versus OpenHCS comparison benchmarks."""

from __future__ import annotations

import argparse
import os

from benchmark.cellprofiler_benchmark_cli import BenchmarkCliCommand


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
    for command in BenchmarkCliCommand.registered_commands():
        command.configure(subparsers)
    args = parser.parse_args()
    return args.cli_command.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
