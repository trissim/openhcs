"""Exactness gates for the allocation-collapsed RescaleIntensity Stretch leaf."""

from __future__ import annotations

import ast
import inspect
import textwrap
import warnings

import numpy as np
import pytest

from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    AutomaticHigh,
    AutomaticLow,
    RescaleIntensityContext,
    RescaleMethod,
    RescaleMethodRunner,
    StretchRescaleMethodRunner,
    rescale_intensity,
)


def _legacy_stretch(data: np.ndarray) -> np.ndarray:
    source = np.asarray(data).astype(np.float32, copy=False)
    source_low = float(np.min(source))
    source_high = float(np.max(source))
    if source_low == source_high:
        return np.zeros_like(source)
    result = np.empty_like(source, dtype=np.float32)
    np.clip(source, source_low, source_high, out=result)
    result -= source_low
    result /= source_high - source_low
    result *= 1.0
    result += 0.0
    return result


def _rescale_stretch(image):
    return rescale_intensity.__wrapped__(
        image,
        rescale_method=RescaleMethod.STRETCH,
        automatic_low=AutomaticLow.CUSTOM,
        automatic_high=AutomaticHigh.CUSTOM,
        source_low=0.0,
        source_high=1.0,
        dest_low=0.0,
        dest_high=1.0,
    )


def test_stretch_is_bit_exact_for_masked_60_plane_payload() -> None:
    rng = np.random.default_rng(20260719)
    pixels = rng.uniform(0.001, 0.04, size=(60, 32, 34)).astype(np.float32)
    mask = np.indices(pixels.shape).sum(axis=0) % 5 != 0
    metadata = ImagePayloadMetadata.for_array(
        pixels,
        source_path="rescale-volume.tif",
    ).with_unit_interval_intensity_scale(65535)
    payload = metadata.payload_with(pixels, mask)

    result = _rescale_stretch(payload)

    expected = _legacy_stretch(pixels)
    np.testing.assert_array_equal(
        image_payload_data(result).view(np.uint32),
        expected.view(np.uint32),
    )
    assert image_payload_data(result).shape == pixels.shape
    assert image_payload_data(result).dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(image_payload_mask(result), mask)
    assert image_payload_metadata(result) == metadata


@pytest.mark.parametrize(
    "pixels",
    (
        np.full((3, 4, 5), 7.25, dtype=np.float32),
        np.array([-0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, np.nan, 1.0], dtype=np.float32),
        np.array([-np.inf, 0.0, np.inf], dtype=np.float32),
        np.array([0.0, 1.0, np.inf], dtype=np.float32),
        np.array([-np.inf, 0.0, 1.0], dtype=np.float32),
    ),
)
def test_stretch_preserves_legacy_edge_case_bits(pixels: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = _legacy_stretch(pixels)
        result = image_payload_data(_rescale_stretch(pixels))

    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))


def test_stretch_registry_leaf_owns_the_exact_reduction() -> None:
    runner = RescaleMethodRunner.for_method(RescaleMethod.STRETCH)
    assert isinstance(runner, StretchRescaleMethodRunner)

    tree = ast.parse(textwrap.dedent(inspect.getsource(type(runner).run)))
    calls = tuple(
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )
    assert calls.count("np.empty_like") == 1
    assert calls.count("np.subtract") == 1
    assert "np.clip" not in calls
    assert "context.linearly_rescaled" not in calls


def test_manual_ranges_remain_on_the_generic_clipping_owner() -> None:
    context = RescaleIntensityContext.from_settings(
        np.array([-1.0, 0.25, 2.0], dtype=np.float32),
        automatic_low=AutomaticLow.CUSTOM,
        automatic_high=AutomaticHigh.CUSTOM,
        source_low=0.0,
        source_high=1.0,
        dest_low=0.2,
        dest_high=0.8,
        divisor_value=1.0,
    )

    result = RescaleMethodRunner.for_method(RescaleMethod.MANUAL_IO_RANGE).run(context)

    expected = context.linearly_rescaled((0.0, 1.0), (0.2, 0.8))
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))
