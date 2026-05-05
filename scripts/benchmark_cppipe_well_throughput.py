#!/usr/bin/env python
"""Benchmark converted cppipes using native OpenHCS well-level multiprocessing."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


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

    _configure_reproducible_runtime_env()
    _configure_benchmark_logging(args.log_level)

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


def _configure_reproducible_runtime_env() -> None:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("XDG_DATA_HOME", "/tmp/openhcs-benchmark-xdg-data")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/openhcs-benchmark-xdg-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/openhcs-benchmark-mpl")
    os.environ.setdefault("OPENHCS_CPU_ONLY", "true")
    os.environ.setdefault("OPENHCS_SUBPROCESS_NO_GPU", "1")
    os.environ.setdefault("POLYSTORE_SUBPROCESS_NO_GPU", "1")


def _configure_benchmark_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown log level: {log_level!r}")
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


if __name__ == "__main__":
    raise SystemExit(main())
