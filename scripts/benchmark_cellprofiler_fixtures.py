#!/usr/bin/env python3
"""Benchmark optimization candidates against captured real CellProfiler arrays."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import statistics
import time

import numpy as np


def _time_call(call: Callable[[], object], repeats: int) -> tuple[float, float]:
    samples: list[float] = []
    call()
    for _ in range(repeats):
        started_at = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started_at)
    return min(samples), statistics.median(samples)


def _benchmark_threshold_application(path: Path, repeats: int) -> tuple[float, float]:
    from openhcs.interop.cellprofiler.thresholding import (
        _threshold_application_smoothed_image,
    )

    fixture = np.load(path)
    smoothing = float(np.asarray(fixture["smoothing"]).reshape(-1)[0])

    def call() -> object:
        return _threshold_application_smoothed_image(
            fixture["image"],
            fixture["mask"],
            smoothing,
        )

    return _print_result(path, *_time_call(call, repeats))


def _benchmark_threshold_diagnostics(path: Path, repeats: int) -> tuple[float, float]:
    from openhcs.interop.cellprofiler.thresholding import (
        cellprofiler_threshold_diagnostics,
    )

    fixture = np.load(path)
    final_threshold = float(np.asarray(fixture["final_threshold"]).reshape(-1)[0])
    original_threshold = float(
        np.asarray(fixture["original_threshold"]).reshape(-1)[0]
    )

    def call() -> object:
        return cellprofiler_threshold_diagnostics(
            fixture["image"],
            fixture["binary"],
            final_threshold=final_threshold,
            original_threshold=original_threshold,
            mask=fixture["mask"],
        )

    return _print_result(path, *_time_call(call, repeats))


def _benchmark_ipo_fill_after(path: Path, repeats: int) -> tuple[float, float]:
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    fixture = np.load(path)
    morphology = MorphologyBackendStrategy.for_memory_type()

    def call() -> object:
        return morphology.fill_labeled_holes(fixture["labels"])

    return _print_result(path, *_time_call(call, repeats))


def _benchmark_ipo_declump_smooth(path: Path, repeats: int) -> tuple[float, float]:
    from openhcs.processing.backends.cellprofiler.morphology import (
        CellProfilerDeclumpMethod,
        MorphologyBackendStrategy,
    )

    fixture = np.load(path)
    morphology = MorphologyBackendStrategy.for_memory_type()
    smooth_size = float(np.asarray(fixture["smooth_size"]).reshape(-1)[0])
    suppress_size = float(np.asarray(fixture["suppress_size"]).reshape(-1)[0])
    min_diameter = float(np.asarray(fixture["min_diameter"]).reshape(-1)[0])

    def call() -> object:
        return morphology.smooth_image_for_declumping(
            fixture["image"],
            fixture["mask"],
            smooth_size,
            declump_method=CellProfilerDeclumpMethod.INTENSITY,
            suppress_size=suppress_size,
            min_diameter=min_diameter,
        )

    return _print_result(path, *_time_call(call, repeats))


def _benchmark_ipo_declump_seed_points(path: Path, repeats: int) -> tuple[float, float]:
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    fixture = np.load(path)
    morphology = MorphologyBackendStrategy.for_memory_type()
    image_resize_factor = float(
        np.asarray(fixture["image_resize_factor"]).reshape(-1)[0]
    )

    def call() -> object:
        return morphology.declumping_seed_points(
            fixture["maxima_image"],
            fixture["labeled_image"],
            fixture["maxima_mask"],
            image_resize_factor,
        )

    return _print_result(path, *_time_call(call, repeats))


def _benchmark_ipo_watershed_execute(path: Path, repeats: int) -> tuple[float, float]:
    from openhcs.processing.backends.cellprofiler.watershed import (
        cellprofiler_legacy_watershed,
    )

    fixture = np.load(path)
    connectivity = np.ones((3, 3), dtype=bool)

    def call() -> object:
        return cellprofiler_legacy_watershed(
            fixture["watershed_image"],
            markers=fixture["watershed_markers"],
            mask=fixture["mask"],
            connectivity=connectivity,
        )

    return _print_result(path, *_time_call(call, repeats))


def _benchmark_rank_median_compact_domain(
    path: Path,
    repeats: int,
) -> tuple[float, float]:
    from openhcs.processing.backends.cellprofiler.illumination import (
        NativeNumpyRankMedianSmoothingBackendStrategy,
    )

    fixture = np.load(path)
    strategy = NativeNumpyRankMedianSmoothingBackendStrategy()

    def call() -> object:
        return strategy._smooth_compact_rank_median(
            fixture["scaled"],
            fixture["footprint"],
        )

    return _print_result(path, *_time_call(call, repeats))


def _print_result(
    path: Path,
    min_seconds: float,
    median_seconds: float,
) -> tuple[float, float]:
    print(
        f"{path.name},min_seconds={min_seconds:.6f},"
        f"median_seconds={median_seconds:.6f}"
    )
    return min_seconds, median_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--prefix",
        action="append",
        help="Fixture prefix to benchmark. May be passed more than once.",
    )
    args = parser.parse_args()

    benchmarks: tuple[tuple[str, Callable[[Path, int], tuple[float, float]]], ...] = (
        ("rank_median_compact_domain", _benchmark_rank_median_compact_domain),
        ("threshold_application", _benchmark_threshold_application),
        ("threshold_diagnostics", _benchmark_threshold_diagnostics),
        ("ipo_declump_smooth", _benchmark_ipo_declump_smooth),
        ("ipo_declump_seed_points", _benchmark_ipo_declump_seed_points),
        ("ipo_watershed_execute", _benchmark_ipo_watershed_execute),
        ("ipo_fill_after", _benchmark_ipo_fill_after),
    )
    selected_prefixes = None if args.prefix is None else frozenset(args.prefix)
    for prefix, benchmark in benchmarks:
        if selected_prefixes is not None and prefix not in selected_prefixes:
            continue
        prefix_min_total = 0.0
        prefix_median_total = 0.0
        count = 0
        for path in sorted(args.fixture_dir.glob(f"{prefix}_*.npz")):
            min_seconds, median_seconds = benchmark(path, args.repeats)
            prefix_min_total += min_seconds
            prefix_median_total += median_seconds
            count += 1
        if count:
            print(
                f"{prefix},fixtures={count},"
                f"sum_min_seconds={prefix_min_total:.6f},"
                f"sum_median_seconds={prefix_median_total:.6f}"
            )


if __name__ == "__main__":
    main()
