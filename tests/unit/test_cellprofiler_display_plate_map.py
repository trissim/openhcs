"""DisplayPlatemap CellProfiler strategy semantics."""

from __future__ import annotations

import numpy as np

from openhcs.processing.backends.cellprofiler.display_modules import (
    AggregationMethod,
    PlateDimensionStrategy,
    PlateType,
    _aggregate_values,
)


def test_display_plate_map_plate_dimensions_are_strategy_backed() -> None:
    assert PlateDimensionStrategy.for_plate_type(PlateType.PLATE_96).dimensions() == (
        8,
        12,
    )
    assert PlateDimensionStrategy.for_plate_type(PlateType.PLATE_384).dimensions() == (
        16,
        24,
    )


def test_display_plate_map_aggregation_methods_are_strategy_backed() -> None:
    values = np.asarray([1.0, 2.0, 4.0, 5.0])

    assert _aggregate_values(values, AggregationMethod.AVG) == np.mean(values)
    assert _aggregate_values(values, AggregationMethod.STDEV) == np.std(values)
    assert _aggregate_values(values, AggregationMethod.MEDIAN) == np.median(values)
    assert _aggregate_values(values, AggregationMethod.CV) == np.std(values) / np.mean(
        values
    )


def test_display_plate_map_cv_zero_mean_is_nan() -> None:
    assert np.isnan(_aggregate_values(np.asarray([-1.0, 1.0]), AggregationMethod.CV))


def test_display_plate_map_empty_values_are_nan() -> None:
    assert np.isnan(_aggregate_values(np.asarray([]), AggregationMethod.AVG))
