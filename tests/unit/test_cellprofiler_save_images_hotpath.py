"""Exactness checks for the CellProfiler SaveImages conversion hot path."""

from __future__ import annotations

import numpy as np
import pytest

from openhcs.core.runtime_image_values import image_payload_data
from openhcs.processing.backends.cellprofiler.save_images import SaveImagesBitDepth


def _reference_uint16_conversion(payload: np.ndarray) -> np.ndarray:
    values = payload.astype(np.float64, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or (
        float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0
    ):
        values = values * 65535.0
    sanitized = np.nan_to_num(values, nan=0.0, posinf=65535.0, neginf=0.0)
    return np.rint(np.clip(sanitized, 0.0, 65535.0)).astype(np.uint16)


@pytest.mark.parametrize(
    "payload",
    (
        np.arange(60 * 256 * 256, dtype=np.int32).reshape(60, 256, 256) % 23,
        np.asarray((0, 1, 1, 0), dtype=np.int32),
        np.asarray((-1, 0, 1, 65535, 65536), dtype=np.int32),
        np.asarray((0, 1, 65535, 65536), dtype=np.uint32),
        np.asarray((), dtype=np.int32),
    ),
)
def test_save_images_uint16_integer_conversion_matches_reference(
    payload: np.ndarray,
) -> None:
    converted = SaveImagesBitDepth.UINT16.convert(payload)
    converted_data = np.asarray(image_payload_data(converted))

    assert converted_data.dtype == np.uint16
    np.testing.assert_array_equal(
        converted_data,
        _reference_uint16_conversion(payload),
    )


def test_save_images_uint16_conversion_does_not_mutate_integer_input() -> None:
    payload = np.asarray((0, 1, 2, 65535), dtype=np.int32)
    original = payload.copy()

    SaveImagesBitDepth.UINT16.convert(payload)

    np.testing.assert_array_equal(payload, original)
