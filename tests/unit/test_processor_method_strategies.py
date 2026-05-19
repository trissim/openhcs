"""Processor method strategy registration tests."""

from __future__ import annotations

import numpy as np

from openhcs.processing.backends.processors import numpy_processor


def test_numpy_processor_method_strategies_use_distinct_inherited_registries() -> None:
    assert set(numpy_processor.NumpySpatialBinStrategy.__registry__) == {
        "max",
        "mean",
        "min",
        "sum",
    }
    assert set(numpy_processor.NumpyStackProjectionStrategy.__registry__) == {
        "max_projection",
        "mean_projection",
    }
    assert (
        numpy_processor.NumpySpatialBinStrategy.__registry__
        is not numpy_processor.NumpyStackProjectionStrategy.__registry__
    )


def test_numpy_processor_method_strategies_dispatch_behavior() -> None:
    stack = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)

    assert numpy_processor.spatial_bin_2d(stack, 2, "mean").shape == (2, 2, 2)
    assert numpy_processor.spatial_bin_3d(stack, 2, "max").shape == (1, 2, 2)
    assert numpy_processor.create_projection(stack, "mean_projection").shape == (
        1,
        4,
        4,
    )
