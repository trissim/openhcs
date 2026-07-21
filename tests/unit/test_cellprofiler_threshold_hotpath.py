"""Exactness contracts for the CellProfiler Threshold hot path."""

from __future__ import annotations

import numpy as np
import pytest

from openhcs.core.config import DtypeConfig
from openhcs.core.runtime_image_values import image_payload_data
from openhcs.processing.backends.cellprofiler.thresholding import (
    CellProfilerOtsuMethod,
    CellProfilerThresholdMethod,
    CellProfilerThresholdScope,
    NumbaNumpyThresholdDiagnosticsBackendStrategy,
    cellprofiler_get_adaptive_threshold,
    cellprofiler_get_global_threshold,
    threshold,
    threshold_method_for_class_count,
)
from openhcs.processing.backends.cellprofiler.thresholding_threshold_numba_diagnostics import (
    _deterministic_normal_noise,
    _quantized_log_tables,
    _threshold_diagnostics_numba,
)
from openhcs.processing.backends.cellprofiler.thresholding_threshold_numba_diagnostics_quantized import (
    QuantizedThresholdDiagnosticContext,
    _threshold_diagnostics_unmasked_finite_quantized_numba,
    exact_quantized_threshold_codes,
)


UINT16_SCALE = int(np.iinfo(np.uint16).max)


def _quantized_stack() -> tuple[np.ndarray, np.ndarray]:
    raw = (
        (np.arange(60 * 16 * 16, dtype=np.uint32) * 37 + 11) % 4096
    ).astype(np.uint16)
    raw = raw.reshape((60, 16, 16))
    return raw, raw.astype(np.float32) / np.float32(UINT16_SCALE)


@pytest.mark.parametrize("partial_mask", (False, True))
def test_whole_image_diagnostics_preserve_singleton_width_results_exactly(
    partial_mask: bool,
) -> None:
    _raw, image = _quantized_stack()
    binary = image >= np.float32(np.mean(image))
    mask = None
    if partial_mask:
        mask = np.ones(image.shape, dtype=np.bool_)
        mask[:, :, :3] = False

    actual = NumbaNumpyThresholdDiagnosticsBackendStrategy().diagnostics(
        image,
        mask,
        binary,
        proven_unit_interval_scale=UINT16_SCALE,
    )

    flat_image = np.ascontiguousarray(image.reshape(-1, 1))
    flat_binary = np.ascontiguousarray(binary.reshape(-1, 1))
    noise = _deterministic_normal_noise(image.shape).reshape(-1, 1)
    if mask is None:
        codes = exact_quantized_threshold_codes(image, UINT16_SCALE)
        assert codes is not None
        log_tables = _quantized_log_tables(UINT16_SCALE)
        expected = _threshold_diagnostics_unmasked_finite_quantized_numba(
            QuantizedThresholdDiagnosticContext(
                codes=np.ascontiguousarray(codes.reshape(-1, 1)),
                binary_image=flat_binary,
                noise=noise,
                values=log_tables.values,
                weighted_log_values=log_tables.weighted_log_values,
                entropy_log_values=log_tables.entropy_log_values,
                entropy_log_delta_values=log_tables.entropy_log_delta_values,
            )
        )
    else:
        expected = _threshold_diagnostics_numba(
            flat_image,
            np.ascontiguousarray(mask.reshape(-1, 1)),
            flat_binary,
            noise,
        )

    assert actual == expected


@pytest.mark.parametrize(
    ("threshold_method", "class_count"),
    (
        (CellProfilerThresholdMethod.OTSU, CellProfilerOtsuMethod.TWO_CLASS),
        (CellProfilerThresholdMethod.OTSU, CellProfilerOtsuMethod.THREE_CLASS),
        (
            CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
            CellProfilerOtsuMethod.TWO_CLASS,
        ),
    ),
)
def test_full_stack_global_threshold_matches_its_declared_method_exactly(
    threshold_method: CellProfilerThresholdMethod,
    class_count: CellProfilerOtsuMethod,
) -> None:
    raw, image = _quantized_stack()
    effective_method = threshold_method_for_class_count(
        threshold_method,
        class_count,
    )
    expected_threshold = cellprofiler_get_global_threshold(
        image,
        threshold_method=effective_method,
        proven_unit_interval_scale=UINT16_SCALE,
    )

    output, rows = threshold(
        raw,
        threshold_method=threshold_method,
        otsu_class_count=class_count,
        smoothing=0.0,
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_array_equal(
        image_payload_data(output),
        (image >= expected_threshold).astype(np.float32),
    )
    assert rows.columns["final_threshold"] == (expected_threshold,)


def test_adaptive_threshold_applies_the_exact_masked_threshold_image() -> None:
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
    mask = np.ones(image.shape, dtype=np.bool_)
    mask[:4, :] = False
    mask[:, :3] = False
    thresholds = cellprofiler_get_adaptive_threshold(
        image,
        mask=mask,
        window_size=8,
    )

    output, _rows = threshold(
        image,
        mask=mask,
        threshold_scope=CellProfilerThresholdScope.ADAPTIVE,
        window_size=8,
        smoothing=0.0,
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_array_equal(
        image_payload_data(output),
        ((image >= thresholds) & mask).astype(np.float32),
    )
