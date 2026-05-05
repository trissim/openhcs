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
    parser.add_argument("--wells", type=int, action="append", required=True)
    parser.add_argument("--workers", type=int, action="append", required=True)
    parser.add_argument(
        "--log-level",
        default=os.environ.get("OPENHCS_BENCHMARK_LOG_LEVEL", "WARNING"),
    )
    args = parser.parse_args()

    configure_headless_cpu_benchmark_runtime(args.log_level)

    from benchmark.well_throughput_scaling import WELL_THROUGHPUT_ROWS_CSV
    from benchmark.well_throughput_scaling import run_well_throughput_suite

    results = run_well_throughput_suite(
        args.manifest,
        output_root=args.output_dir,
        case_names=tuple(args.case_names or ()),
        well_counts=tuple(args.wells),
        worker_counts=tuple(args.workers),
    )
    csv_path = args.output_dir / WELL_THROUGHPUT_ROWS_CSV
    print(f"observations={len(results)}")
    print(f"csv={csv_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
