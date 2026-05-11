"""Shared CellProfiler image normalization and threshold compatibility surface."""

from __future__ import annotations

import logging
import os

import numpy as np

from openhcs.processing.backends.cellprofiler.thresholding import (
    CELLPROFILER_BASIC_THRESHOLD_SMOOTHING_SCALE,
    CELLPROFILER_LOG_MULTI_OTSU_BIN_CENTER_OFFSET,
    CELLPROFILER_LOG_MULTI_OTSU_BINS,
    CELLPROFILER_MULTI_OTSU_BINS,
    CellProfilerAveragingMethod,
    CellProfilerOtsuMethod,
    CellProfilerThresholdAssignment,
    CellProfilerThresholdDiagnostics,
    CellProfilerThresholdMethod,
    CellProfilerThresholdScope,
    CellProfilerVarianceMethod,
    RobustBackgroundCenterStrategy,
    RobustBackgroundSpreadStrategy,
    RobustBackgroundThresholdSettings,
    cellprofiler_apply_threshold as _backend_cellprofiler_apply_threshold,
    cellprofiler_get_adaptive_threshold as _backend_cellprofiler_get_adaptive_threshold,
    cellprofiler_get_global_threshold as _backend_cellprofiler_get_global_threshold,
    cellprofiler_threshold as _backend_cellprofiler_threshold,
    cellprofiler_threshold_diagnostics as _backend_cellprofiler_threshold_diagnostics,
    get_threshold_robust_background as _get_threshold_robust_background,
    normalize_cellprofiler_image,
    threshold_histogram_bin_width as _threshold_histogram_bin_width,
    threshold_multiotsu as _threshold_multiotsu,
    unit_interval_scale_for_threshold_diagnostics,
)


_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


def cellprofiler_get_global_threshold(*args: object, **kwargs: object) -> float:
    """Compatibility wrapper for backend-owned global threshold semantics."""
    return _backend_cellprofiler_get_global_threshold(*args, **kwargs)


def cellprofiler_get_adaptive_threshold(*args: object, **kwargs: object) -> np.ndarray:
    """Compatibility wrapper for backend-owned adaptive threshold semantics."""
    kwargs.setdefault("global_threshold_function", cellprofiler_get_global_threshold)
    return _backend_cellprofiler_get_adaptive_threshold(*args, **kwargs)


def cellprofiler_apply_threshold(
    image: np.ndarray,
    *,
    threshold: float | np.ndarray,
    mask: np.ndarray | None = None,
    smoothing: float = 0,
) -> tuple[np.ndarray, float]:
    """Compatibility wrapper for backend-owned threshold application."""
    return _backend_cellprofiler_apply_threshold(
        image,
        threshold=threshold,
        mask=mask,
        smoothing=smoothing,
    )


def cellprofiler_threshold(*args: object, **kwargs: object) -> tuple[np.ndarray, float, float]:
    """Apply backend-owned threshold semantics through the benchmark boundary."""
    kwargs.setdefault("global_threshold_function", cellprofiler_get_global_threshold)
    kwargs.setdefault("adaptive_threshold_function", cellprofiler_get_adaptive_threshold)
    kwargs.setdefault("apply_threshold_function", cellprofiler_apply_threshold)
    kwargs.setdefault("log_profile_function", _log_profile)
    return _backend_cellprofiler_threshold(*args, **kwargs)


def cellprofiler_threshold_diagnostics(
    *args: object,
    **kwargs: object,
) -> CellProfilerThresholdDiagnostics:
    """Compatibility wrapper for backend-owned threshold diagnostics."""
    kwargs.setdefault("log_profile_function", _log_profile)
    return _backend_cellprofiler_threshold_diagnostics(*args, **kwargs)
