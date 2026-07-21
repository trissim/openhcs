"""Exact Threshold entropy regressions for producer float32 codebooks."""

from __future__ import annotations

import numpy as np
import pytest

from openhcs.interop.cellprofiler.image_normalization import (
    _cellprofiler_uint8_float32_lut,
    normalize_cellprofiler_image_payload,
)
from openhcs.processing.backends.cellprofiler import thresholding
from openhcs.processing.backends.cellprofiler.thresholding import (
    NumbaNumpyThresholdDiagnosticsBackendStrategy,
    NumbaNumpyThresholdPrimitiveBackendStrategy,
    unit_interval_scale_for_threshold_selection,
)


def _codebook_image(*, seed: int) -> np.ndarray:
    codes = np.random.default_rng(seed).integers(
        0,
        255,
        size=(64, 64),
        dtype=np.uint8,
    )
    return _cellprofiler_uint8_float32_lut()[codes]


def test_virtual_source_uses_structured_physical_png_format() -> None:
    from openhcs.core.runtime_image_values import ImagePayloadMetadata, MaskedImagePayload
    from openhcs.core.source_metadata import (
        SOURCE_FILTER_PATHS_METADATA_FIELD,
        SourceFilterPathMetadata,
    )

    image = np.asarray([[7, 98, 128, 254, 255]], dtype=np.uint8)
    metadata = ImagePayloadMetadata.for_array(
        image,
        source_path="virtual/cho01.tif",
    ).replace_fields(
        source_component_metadata={
            SOURCE_FILTER_PATHS_METADATA_FIELD: SourceFilterPathMetadata.from_paths(
                ("cho01.png", "/physical/cho01.png")
            ).as_dict()
        }
    )
    payload = MaskedImagePayload(
        data=image,
        mask=np.ones(image.shape, dtype=np.bool_),
        metadata=metadata,
    )

    observed = np.asarray(normalize_cellprofiler_image_payload(payload))

    np.testing.assert_array_equal(observed, _cellprofiler_uint8_float32_lut()[image])


def test_full_stack_uses_retained_source_plane_dtype_scale() -> None:
    from openhcs.core.runtime_image_values import ImagePayloadMetadata

    image = _codebook_image(seed=7)[None, ...]
    metadata = ImagePayloadMetadata(
        source_plane_intensity_scales=(None,),
        source_plane_dtypes=("uint8",),
    )

    assert unit_interval_scale_for_threshold_selection(image, metadata) == 255


def test_full_stack_rejects_conflicting_source_plane_dtype_scales() -> None:
    from openhcs.core.runtime_image_values import ImagePayloadMetadata

    image = np.zeros((2, 4, 4), dtype=np.float32)
    metadata = ImagePayloadMetadata(
        source_plane_intensity_scales=(None, None),
        source_plane_dtypes=("uint8", "uint16"),
    )

    assert unit_interval_scale_for_threshold_selection(image, metadata) is None


def test_minimum_cross_entropy_preserves_producer_float32_codebook() -> None:
    image = _codebook_image(seed=1)
    expected = thresholding._li_threshold_float32_numpy(image.ravel())

    observed = NumbaNumpyThresholdPrimitiveBackendStrategy().minimum_cross_entropy_threshold(
        image,
        proven_unit_interval_scale=255,
    )

    assert expected == 0.4035060703754425
    assert observed == expected


def test_threshold_entropy_fast_path_preserves_producer_float32_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(17)
    codes = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    image = _cellprofiler_uint8_float32_lut()[codes]
    binary = rng.random(image.shape) > 0.5
    mask = np.ones(image.shape, dtype=np.bool_)
    expected = thresholding._numpy_threshold_sum_of_entropies(image, mask, binary)

    def forbidden_generic_kernel(*args: object, **kwargs: object) -> tuple[float, float]:
        del args, kwargs
        raise AssertionError("producer codebooks must retain the quantized fast path")

    monkeypatch.setattr(
        thresholding,
        "_threshold_diagnostics_unmasked_finite_numba",
        forbidden_generic_kernel,
    )
    _weighted_variance, observed = (
        NumbaNumpyThresholdDiagnosticsBackendStrategy().diagnostics(
            image,
            None,
            binary,
            proven_unit_interval_scale=255,
        )
    )

    assert expected == -13.022839004723934
    assert observed == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_minimum_cross_entropy_preserves_float32_reduction_order() -> None:
    codes = np.random.default_rng(0).integers(
        0,
        256,
        size=(64, 64),
        dtype=np.uint8,
    )
    image = codes.astype(np.float32) / np.float32(255)
    expected = thresholding._li_threshold_float32_numpy(image.ravel())
    observed = NumbaNumpyThresholdPrimitiveBackendStrategy().minimum_cross_entropy_threshold(
        image,
        proven_unit_interval_scale=255,
    )

    assert expected == 0.3884999454021454
    assert observed == expected
