import warnings

import numpy as np
import pytest
import scipy.ndimage as ndi
import skimage.transform

from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    ResizeGeometry,
    resize,
    resize_volumetric,
)


def _reference_mask_resize(
    mask: np.ndarray, output_shape: tuple[int, ...]
) -> np.ndarray:
    zoom = tuple(
        output_size / input_size
        for output_size, input_size in zip(
            output_shape, mask.shape, strict=True
        )
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return ndi.zoom(
            mask.astype(np.float32),
            zoom,
            order=0,
            mode="constant",
            grid_mode=True,
        ).astype(bool, copy=False)


@pytest.mark.parametrize(
    ("input_shape", "output_shape"),
    (
        ((1,), (4,)),
        ((4,), (1,)),
        ((3, 5), (7, 2)),
        ((2, 3, 4), (5, 1, 9)),
        ((2, 3, 4, 2), (3, 7, 2, 5)),
    ),
)
def test_resize_mask_matches_scipy_grid_constant_semantics(
    input_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> None:
    mask = np.indices(input_shape).sum(axis=0) % 3 != 0
    geometry = ResizeGeometry(output_shape=output_shape, interpolation_order=0)

    actual = geometry.resize_mask(mask)

    assert actual is not None
    assert actual.dtype == np.dtype(bool)
    assert actual.shape == output_shape
    np.testing.assert_array_equal(actual, _reference_mask_resize(mask, output_shape))


def test_resize_mask_uses_boolean_separable_projection(monkeypatch) -> None:
    mask = np.indices((3, 4, 5)).sum(axis=0) % 3 != 0
    geometry = ResizeGeometry(output_shape=(5, 2, 9), interpolation_order=0)
    take_calls: list[tuple[np.dtype, int]] = []
    numpy_take = np.take

    def tracked_take(array, indices, *, axis):
        take_calls.append((np.asarray(array).dtype, axis))
        return numpy_take(array, indices, axis=axis)

    monkeypatch.setattr(np, "take", tracked_take)

    actual = geometry.resize_mask(mask)

    assert actual is not None
    assert take_calls == [(np.dtype(bool), 0), (np.dtype(bool), 1), (np.dtype(bool), 2)]
    np.testing.assert_array_equal(actual, _reference_mask_resize(mask, (5, 2, 9)))


def test_resize_preserves_ordinary_image_payload_semantics() -> None:
    pixels = np.linspace(0.0, 1.0, 6 * 8, dtype=np.float32).reshape((6, 8))
    metadata = ImagePayloadMetadata.for_array(
        pixels, source_path="ordinary-source.tif"
    ).with_unit_interval_intensity_scale(65535)
    payload = metadata.payload_with(pixels, None)
    output_shape = (3, 12)

    result = resize.__wrapped__(
        payload,
        resizing_factor_x=1.5,
        resizing_factor_y=0.5,
        interpolation="bilinear",
    )

    expected_pixels = skimage.transform.resize(
        pixels,
        output_shape,
        order=1,
        mode="symmetric",
        preserve_range=True,
    ).astype(pixels.dtype, copy=False)
    expected_mask = _reference_mask_resize(
        np.ones(pixels.shape, dtype=bool), output_shape
    )
    np.testing.assert_allclose(
        image_payload_data(result), expected_pixels, rtol=0.0, atol=1e-6
    )
    np.testing.assert_array_equal(image_payload_mask(result), expected_mask)
    assert image_payload_data(result).dtype == pixels.dtype
    assert image_payload_data(result).shape == output_shape
    assert image_payload_metadata(result) == (
        metadata.with_spatial_resize(output_shape).without_unit_interval_intensity_scale()
    )


def test_resize_volumetric_preserves_masked_payload_semantics() -> None:
    pixels = np.linspace(0.0, 1.0, 3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
    mask = np.indices(pixels.shape).sum(axis=0) % 4 != 0
    metadata = ImagePayloadMetadata.for_array(
        pixels, source_path="volume-source.tif"
    ).with_unit_interval_intensity_scale(65535)
    payload = metadata.payload_with(pixels, mask)
    output_shape = (3, 8, 3)

    result = resize_volumetric.__wrapped__(
        payload,
        resizing_factor_x=0.6,
        resizing_factor_y=2.0,
        resizing_factor_z=1.0,
        interpolation="nearest_neighbor",
    )

    expected_pixels = skimage.transform.resize(
        pixels,
        output_shape,
        order=0,
        mode="symmetric",
        preserve_range=True,
    ).astype(pixels.dtype, copy=False)
    np.testing.assert_allclose(
        image_payload_data(result), expected_pixels, rtol=0.0, atol=1e-6
    )
    np.testing.assert_array_equal(
        image_payload_mask(result), _reference_mask_resize(mask, output_shape)
    )
    assert image_payload_data(result).dtype == pixels.dtype
    assert image_payload_data(result).shape == output_shape
    assert image_payload_metadata(result) == metadata.with_spatial_resize(
        output_shape[-2:]
    )
