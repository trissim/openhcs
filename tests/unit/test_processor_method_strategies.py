"""Processor method strategy registration tests."""

from __future__ import annotations

import numpy as np

from openhcs.core.callable_contract import CallableContract
from openhcs.processing.backends.lib_registry.openhcs_registry import OpenHCSRegistry
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
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
        "min_projection",
    }
    assert (
        numpy_processor.NumpySpatialBinStrategy.__registry__
        is not numpy_processor.NumpyStackProjectionStrategy.__registry__
    )


def test_numpy_processor_method_strategies_dispatch_behavior() -> None:
    stack = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)

    assert numpy_processor.spatial_bin_2d(stack, 2, "mean").shape == (2, 2, 2)
    assert numpy_processor.spatial_bin_3d(stack, 2, "max").shape == (1, 2, 2)
    assert numpy_processor.create_projection(stack, "mean_projection").shape == (4, 4)
    np.testing.assert_array_equal(
        numpy_processor.create_projection(stack, "min_projection"),
        np.min(stack, axis=0),
    )
    assert (
        CallableContract.from_callable(
            numpy_processor.create_projection
        ).require_processing_contract()
        is ProcessingContract.VOLUMETRIC_TO_SLICE
    )


def test_openhcs_registry_discovery_does_not_replace_public_processor_callables() -> (
    None
):
    declared_projection = numpy_processor.create_projection
    registry = OpenHCSRegistry()
    registry.MODULES_TO_SCAN = [numpy_processor.__name__]

    functions = registry.discover_functions()

    assert numpy_processor.create_projection is declared_projection
    assert (
        functions["processors_numpy_processor_create_projection"].func
        is not declared_projection
    )
    stack = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    assert numpy_processor.create_projection(stack, "mean_projection").shape == (4, 4)
