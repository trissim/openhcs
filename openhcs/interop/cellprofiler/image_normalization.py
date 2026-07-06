"""CellProfiler image intensity normalization semantics."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_intensity_scale,
    image_payload_metadata,
    normalize_image_payload_intensity,
    with_image_payload_data,
)


def normalize_cellprofiler_image_payload(
    payload: Any,
    *,
    dtype: Any = np.float32,
    channel_index: int = 0,
    allow_unproven_uint8_float_domain: bool = False,
) -> Any:
    """Return payload in native CellProfiler's float image intensity domain."""
    array = np.asarray(image_payload_data(payload))
    target_dtype = np.dtype(dtype)
    intensity_scale = image_payload_intensity_scale(
        payload,
        channel_index=channel_index,
    )
    if (
        target_dtype == np.dtype(np.float32)
        and array.dtype == np.dtype(np.uint8)
        and intensity_scale == 255
        and _cellprofiler_uses_bioformats_uint8_domain(payload)
    ):
        normalized = _cellprofiler_uint8_float32_lut()[array]
        return with_image_payload_data(payload, normalized)
    if (
        target_dtype == np.dtype(np.float32)
        and np.issubdtype(array.dtype, np.floating)
        and image_payload_metadata(payload).common_unit_interval_intensity_scale()
        == 255
    ):
        remapped = _cellprofiler_float32_uint8_domain(
            array,
            accept_numpy_domain=_cellprofiler_uses_bioformats_uint8_domain(payload),
        )
        if remapped is not None:
            return with_image_payload_data(payload, remapped)
    if (
        allow_unproven_uint8_float_domain
        and target_dtype == np.dtype(np.float32)
        and np.issubdtype(array.dtype, np.floating)
    ):
        remapped = _cellprofiler_float32_uint8_domain(
            array,
            accept_numpy_domain=_cellprofiler_uses_bioformats_uint8_domain(payload),
        )
        if remapped is not None:
            return with_image_payload_data(payload, remapped)
    return normalize_image_payload_intensity(
        payload,
        dtype=target_dtype,
        channel_index=channel_index,
    )


@lru_cache(maxsize=1)
def _cellprofiler_uint8_float32_lut() -> np.ndarray:
    """Return CP 4.2/BioFormats uint8 source pixels as float32."""
    values = np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0
    downrounded_codes = _cellprofiler_bioformats_downrounded_uint8_codes()
    values[downrounded_codes] = np.nextafter(
        values[downrounded_codes],
        np.float32(0.0),
        dtype=np.float32,
    )
    values[255] = np.float32(1.0)
    values.setflags(write=False)
    return values


def _cellprofiler_bioformats_downrounded_uint8_codes() -> np.ndarray:
    """Return uint8 codes BioFormats rounds one float32 step below NumPy division."""
    return np.asarray(
        (
            7,
            14,
            23,
            28,
            39,
            46,
            49,
            56,
            59,
            78,
            81,
            92,
            98,
            101,
            109,
            112,
            115,
            118,
            121,
            156,
            159,
            162,
            165,
            181,
            184,
            187,
            193,
            196,
            199,
            202,
            205,
            218,
            221,
            224,
            230,
            233,
            236,
            239,
            242,
            245,
        ),
        dtype=np.uint8,
    )


def _cellprofiler_float32_uint8_domain(
    array: np.ndarray,
    *,
    accept_numpy_domain: bool,
) -> np.ndarray | None:
    """Return CP uint8-domain values when a float payload still encodes uint8."""
    image = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(image)
    if not np.all(finite):
        return None
    if image.size == 0 or np.min(image) < 0.0 or np.max(image) > 1.0:
        return None
    codes = np.rint(image * np.float32(255.0)).astype(np.uint8)
    cellprofiler_domain = _cellprofiler_uint8_float32_lut()[codes]
    numpy_domain = codes.astype(np.float32) / 255.0
    if np.array_equal(image, cellprofiler_domain):
        return cellprofiler_domain
    if not accept_numpy_domain or not np.array_equal(image, numpy_domain):
        return None
    return cellprofiler_domain


def _cellprofiler_uses_bioformats_uint8_domain(payload: Any) -> bool:
    """Return whether native CP loads this source through its PNG float codebook."""
    paths = image_payload_metadata(payload).source_image_paths
    return bool(paths) and all(path.lower().endswith(".png") for path in paths)
