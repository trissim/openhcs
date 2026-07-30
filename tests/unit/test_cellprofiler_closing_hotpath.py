from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from skimage.morphology import closing as skimage_closing
from skimage.morphology import dilation as skimage_dilation
from skimage.morphology import disk
from skimage.morphology import erosion as skimage_erosion
from skimage.morphology import opening as skimage_opening

from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    NumpyMorphologyBackendStrategy,
    closing,
    dilate_image,
    erode_image,
    opening,
)
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
)


def _raw_function(function: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function


def test_closing_collapses_runtime_slice_stack_into_one_native_operation(
    monkeypatch,
) -> None:
    rng = np.random.default_rng(20260719)
    image = rng.random((12, 48, 52), dtype=np.float32)
    footprint = disk(7)
    expected = np.stack(
        tuple(skimage_closing(plane, footprint) for plane in image),
        axis=0,
    )
    observed_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    native_closing = NumpyMorphologyBackendStrategy.grayscale_closing

    def record_native_call(
        self,
        pixels: np.ndarray,
        structuring_element: np.ndarray,
    ) -> np.ndarray:
        observed_shapes.append((pixels.shape, structuring_element.shape))
        return native_closing(self, pixels, structuring_element)

    monkeypatch.setattr(
        NumpyMorphologyBackendStrategy,
        "grayscale_closing",
        record_native_call,
    )

    observed = _raw_function(closing)(
        image,
        structuring_element=StructuringElement.DISK,
        size=7,
    )

    np.testing.assert_array_equal(observed, expected)
    assert observed.dtype == image.dtype
    assert observed_shapes == [((12, 48, 52), (1, 15, 15))]


@pytest.mark.parametrize(
    "provider",
    (CellProfilerBackendProvider.NUMBA, CellProfilerBackendProvider.OPENCV),
)
def test_closing_stack_preserves_explicit_provider_semantics(
    provider: CellProfilerBackendProvider,
) -> None:
    rng = np.random.default_rng(13)
    image = rng.random((3, 19, 21), dtype=np.float32)
    footprint = disk(3)
    expected = np.stack(
        tuple(skimage_closing(plane, footprint) for plane in image),
        axis=0,
    )

    observed = _raw_function(closing)(
        image,
        structuring_element=StructuringElement.DISK,
        size=3,
        morphology_backend_provider=provider,
    )

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize(
    ("function", "reference"),
    (
        (closing, skimage_closing),
        (opening, skimage_opening),
        (dilate_image, skimage_dilation),
        (erode_image, skimage_erosion),
    ),
)
def test_image_morphology_stack_matches_planewise_reference(
    function: Callable[..., np.ndarray],
    reference: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    rng = np.random.default_rng(91)
    image = rng.random((4, 23, 25), dtype=np.float32)
    footprint = disk(3)
    expected = np.stack(
        tuple(reference(plane, footprint) for plane in image),
        axis=0,
    )

    observed = _raw_function(function)(
        image,
        structuring_element=StructuringElement.DISK,
        size=3,
    )

    np.testing.assert_array_equal(observed, expected)
