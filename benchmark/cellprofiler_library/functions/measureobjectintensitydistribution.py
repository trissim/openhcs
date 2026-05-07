"""Converted from CellProfiler: MeasureObjectIntensityDistribution"""

import logging
import numpy as np
import os
import time
from typing import Tuple, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import hashlib
from numba import njit, prange
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.backends.cellprofiler.intensity_distribution import (
    radial_distribution_backend,
)
from openhcs.processing.backends.cellprofiler.shape import shape_measurement_backend
from openhcs.processing.backends.cellprofiler.zernike import (
    intensity_zernike_moments,
)
from openhcs.processing.materialization import csv_materializer

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


class CenterChoice(Enum):
    SELF = "self"
    CENTERS_OF_OTHER = "centers_of_other"
    EDGES_OF_OTHER = "edges_of_other"


class ZernikeMode(Enum):
    NONE = "none"
    MAGNITUDES = "magnitudes"
    MAGNITUDES_AND_PHASE = "magnitudes_and_phase"


@dataclass
class RadialDistributionMeasurement:
    """Measurements for radial intensity distribution."""
    object_label: int
    bin_index: int
    bin_count: int
    frac_at_d: float
    mean_frac: float
    radial_cv: float


@dataclass
class ZernikeMeasurement:
    """Zernike moment measurements."""
    object_label: int
    n: int
    m: int
    magnitude: float
    phase: Optional[float] = None


@dataclass(frozen=True)
class _RadialLabelGeometry:
    d_to_edge: np.ndarray
    centers_i: np.ndarray
    centers_j: np.ndarray


_RADIAL_LABEL_GEOMETRY_CACHE: OrderedDict[
    tuple[str, tuple[int, ...], bytes],
    _RadialLabelGeometry,
] = OrderedDict()
_RADIAL_LABEL_GEOMETRY_CACHE_MAX_ENTRIES = 16


@numpy
@special_inputs("labels")
@special_outputs(
    ("radial_measurements", csv_materializer(
        fields=[
            "object_label",
            "bin_index",
            "bin_count",
            "frac_at_d",
            "mean_frac",
            "radial_cv",
            "feature_name",
            "result_value",
        ],
        analysis_type="radial_distribution"
    ))
)
def measure_object_intensity_distribution(
    image: np.ndarray,
    labels: np.ndarray,
    bin_count: int = 4,
    wants_scaled: bool = True,
    maximum_radius: int = 100,
    wants_zernikes: ZernikeMode = ZernikeMode.NONE,
    zernike_degree: int = 9,
    center_choice: CenterChoice = CenterChoice.SELF,
    radial_distribution_backend_provider: CellProfilerBackendProvider | None = None,
    zernike_backend_provider: CellProfilerBackendProvider | None = None,
) -> Tuple[np.ndarray, List[Any]]:
    """
    Measure the spatial distribution of intensities within each object.

    Measures intensity distribution from each object's center to its boundary
    within a set of bins (rings).

    Args:
        image: Input grayscale image, shape (D, H, W) or (H, W)
        labels: Object labels, same spatial shape as image
        bin_count: Number of radial bins
        wants_scaled: If True, scale bins per-object; if False, use fixed radius
        maximum_radius: Maximum radius for unscaled bins (pixels)
        wants_zernikes: Whether to calculate Zernike moments
        zernike_degree: Maximum Zernike radial moment
        center_choice: How to determine object centers

    Returns:
        Tuple of (original image, list of measurements)
    """
    total_started_at = time.perf_counter()
    labels = object_label_dense_array(labels, dtype=np.int32)
    # Handle dimensionality
    if image.ndim == 3:
        # Process first slice for now (2D module)
        img_2d = image[0]
        if labels.ndim == 3:
            labels_2d = labels[0]
        else:
            labels_2d = labels
    else:
        img_2d = image
        labels_2d = labels
    
    wants_zernikes = _coerce_zernike_mode(wants_zernikes)
    measurements: list[Any] = []
    
    nobjects = int(np.max(labels_2d))
    if nobjects == 0:
        # Return empty measurements
        return image, measurements
    
    phase_started_at = time.perf_counter()
    radial_geometry = _radial_label_geometry(labels_2d, nobjects)
    _log_profile(
        "idist_radial_geometry",
        time.perf_counter() - phase_started_at,
        function="measure_object_intensity_distribution",
        nobjects=nobjects,
    )

    phase_started_at = time.perf_counter()
    radial_arrays = radial_distribution_backend(
        backend_provider=radial_distribution_backend_provider,
    ).measure_from_centers(
        img_2d,
        labels_2d,
        radial_geometry.d_to_edge,
        radial_geometry.centers_i,
        radial_geometry.centers_j,
        bin_count=bin_count,
        wants_scaled=wants_scaled,
        maximum_radius=maximum_radius,
    )
    _log_profile(
        "idist_radial_backend",
        time.perf_counter() - phase_started_at,
        function="measure_object_intensity_distribution",
        nobjects=nobjects,
        bins=radial_arrays.n_bins,
    )

    def append_bin_measurements(bin_idx: int, radial_cv: np.ndarray) -> None:
        # Missing label IDs inside the dense object domain carry no radial
        # fraction; CP-style exports keep RadialCV at zero but mark
        # FracAtD/MeanFrac as NaN.
        for obj_idx in range(nobjects):
            frac_at_d = (
                float(radial_arrays.fraction_at_distance[obj_idx, bin_idx])
                if radial_arrays.object_has_pixels[obj_idx]
                else np.nan
            )
            mean_frac = (
                float(radial_arrays.mean_pixel_fraction[obj_idx, bin_idx])
                if radial_arrays.object_has_pixels[obj_idx]
                else np.nan
            )
            measurements.append(RadialDistributionMeasurement(
                object_label=obj_idx + 1,
                bin_index=bin_idx + 1,
                bin_count=bin_count,
                frac_at_d=frac_at_d,
                mean_frac=mean_frac,
                radial_cv=float(radial_cv[obj_idx])
            ))
    
    phase_started_at = time.perf_counter()
    for bin_idx in range(radial_arrays.n_bins):
        append_bin_measurements(
            bin_idx,
            radial_arrays.radial_cv_by_bin[bin_idx],
        )
    _log_profile(
        "idist_radial_rows",
        time.perf_counter() - phase_started_at,
        function="measure_object_intensity_distribution",
        rows=len(measurements),
    )

    if wants_zernikes != ZernikeMode.NONE:
        phase_started_at = time.perf_counter()
        measurements.extend(
            _zernike_measurement_rows(
                img_2d,
                labels_2d,
                wants_zernikes=wants_zernikes,
                zernike_degree=zernike_degree,
                backend_provider=zernike_backend_provider,
            )
        )
        _log_profile(
            "idist_zernike_rows",
            time.perf_counter() - phase_started_at,
            function="measure_object_intensity_distribution",
            rows=len(measurements),
        )

    _log_profile(
        "idist_total",
        time.perf_counter() - total_started_at,
        function="measure_object_intensity_distribution",
        rows=len(measurements),
    )
    return image, measurements


def _radial_label_geometry(
    labels: np.ndarray,
    nobjects: int,
) -> _RadialLabelGeometry:
    total_started_at = time.perf_counter()
    labels_2d = np.asarray(labels, dtype=np.int32)
    phase_started_at = time.perf_counter()
    key = _label_content_key(labels_2d)
    _log_profile(
        "idist_geometry_key",
        time.perf_counter() - phase_started_at,
        function="measure_object_intensity_distribution",
    )
    entry = _RADIAL_LABEL_GEOMETRY_CACHE.get(key)
    if entry is not None:
        _RADIAL_LABEL_GEOMETRY_CACHE.move_to_end(key)
        _log_profile(
            "idist_geometry_cache_hit",
            time.perf_counter() - total_started_at,
            function="measure_object_intensity_distribution",
        )
        return entry

    phase_started_at = time.perf_counter()
    d_to_edge = _distance_to_edge(labels_2d)
    _log_profile(
        "idist_distance_to_edge",
        time.perf_counter() - phase_started_at,
        function="measure_object_intensity_distribution",
    )
    phase_started_at = time.perf_counter()
    centers_i, centers_j = _find_object_centers(labels_2d, d_to_edge, nobjects)
    _log_profile(
        "idist_find_centers",
        time.perf_counter() - phase_started_at,
        function="measure_object_intensity_distribution",
    )
    geometry = _RadialLabelGeometry(
        d_to_edge=d_to_edge,
        centers_i=centers_i,
        centers_j=centers_j,
    )
    _RADIAL_LABEL_GEOMETRY_CACHE[key] = geometry
    _RADIAL_LABEL_GEOMETRY_CACHE.move_to_end(key)
    while len(_RADIAL_LABEL_GEOMETRY_CACHE) > _RADIAL_LABEL_GEOMETRY_CACHE_MAX_ENTRIES:
        _RADIAL_LABEL_GEOMETRY_CACHE.popitem(last=False)
    _log_profile(
        "idist_geometry_total",
        time.perf_counter() - total_started_at,
        function="measure_object_intensity_distribution",
    )
    return geometry


def _label_content_key(labels: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    contiguous = np.ascontiguousarray(labels)
    digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest()
    return str(contiguous.dtype), tuple(int(value) for value in contiguous.shape), digest


def _coerce_zernike_mode(value: ZernikeMode | str) -> ZernikeMode:
    if isinstance(value, ZernikeMode):
        return value
    normalized = str(value).strip().lower().replace(" ", "_")
    for mode in ZernikeMode:
        if normalized in {mode.name.lower(), mode.value}:
            return mode
    raise ValueError(f"Unknown Zernike mode: {value!r}.")


def _zernike_measurement_rows(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    wants_zernikes: ZernikeMode,
    zernike_degree: int,
    backend_provider: CellProfilerBackendProvider | None = None,
) -> list[dict[str, Any]]:
    """Return CellProfiler-compatible long-form Zernike measurement rows."""
    labels_int = object_label_dense_array(labels, dtype=np.int32)
    object_count = int(labels_int.max()) if labels_int.size else 0
    if object_count <= 0:
        return []

    object_ids = np.arange(1, object_count + 1, dtype=np.int32)
    zernike_indexes, magnitudes, phases = intensity_zernike_moments(
        image,
        labels_int,
        object_ids,
        max_order=int(zernike_degree),
        backend_provider=backend_provider,
    )
    if len(zernike_indexes) == 0:
        return []

    rows: list[dict[str, Any]] = []

    for index, (n, m) in enumerate(zernike_indexes):
        for object_label, magnitude in zip(
            object_ids,
            magnitudes[:, index],
            strict=True,
        ):
            rows.append(
                {
                    "object_label": int(object_label),
                    "feature_name": f"ZernikeMagnitude_{int(n)}_{int(m)}",
                    "result_value": float(magnitude),
                }
            )

        if wants_zernikes == ZernikeMode.MAGNITUDES_AND_PHASE:
            for object_label, phase in zip(
                object_ids,
                phases[:, index],
                strict=True,
            ):
                rows.append(
                    {
                        "object_label": int(object_label),
                        "feature_name": f"ZernikePhase_{int(n)}_{int(m)}",
                        "result_value": float(phase),
                    }
                )

    return rows


def _empty_zernike_measurement_rows(
    object_count: int,
    zernike_indexes: list[tuple[int, int]],
    *,
    wants_zernikes: ZernikeMode,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n, m in zernike_indexes:
        for object_label in range(1, object_count + 1):
            rows.append(
                {
                    "object_label": object_label,
                    "feature_name": f"ZernikeMagnitude_{int(n)}_{int(m)}",
                    "result_value": np.nan,
                }
            )
        if wants_zernikes == ZernikeMode.MAGNITUDES_AND_PHASE:
            for object_label in range(1, object_count + 1):
                rows.append(
                    {
                        "object_label": object_label,
                        "feature_name": f"ZernikePhase_{int(n)}_{int(m)}",
                        "result_value": np.nan,
                    }
                )
    return rows


def _prepare_measure_object_intensity_distribution() -> None:
    """Compile radial-distribution and intensity-Zernike kernels before execution."""
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[8:24, 8:24] = 1
    labels[32:56, 32:56] = 2
    measure_object_intensity_distribution.__wrapped__(
        image,
        labels,
        bin_count=4,
        wants_scaled=True,
        maximum_radius=100,
        wants_zernikes=ZernikeMode.MAGNITUDES_AND_PHASE,
        zernike_degree=9,
    )


def _distance_to_edge(labels: np.ndarray) -> np.ndarray:
    """Compute distance to edge for each labeled pixel."""
    return shape_measurement_backend().distance_to_edge(labels)


def _find_object_centers(labels: np.ndarray, d_to_edge: np.ndarray, nobjects: int):
    """Find the center of each object (point farthest from edge)."""
    if nobjects <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    indexes = np.arange(1, nobjects + 1, dtype=np.int32)
    centers_i, centers_j = shape_measurement_backend().maximum_position_of_labels(
        d_to_edge,
        object_label_dense_array(labels, dtype=np.int32),
        indexes,
    )
    return centers_i.astype(np.float64), centers_j.astype(np.float64)


def _compute_distance_from_centers(
    labels: np.ndarray,
    centers_i: np.ndarray,
    centers_j: np.ndarray,
    nobjects: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute distance from center for each pixel."""
    labels_int = object_label_dense_array(labels, dtype=np.int32)
    if nobjects <= 0:
        return (
            np.zeros(labels.shape, dtype=np.float64),
            np.zeros(labels.shape, dtype=np.int32),
        )

    return _compute_distance_from_centers_numba(
        np.ascontiguousarray(labels_int),
        np.ascontiguousarray(centers_i, dtype=np.float64),
        np.ascontiguousarray(centers_j, dtype=np.float64),
        int(nobjects),
    )


@njit(cache=True, parallel=True)
def _compute_distance_from_centers_numba(
    labels: np.ndarray,
    centers_i: np.ndarray,
    centers_j: np.ndarray,
    nobjects: int,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = labels.shape
    d_from_center = np.zeros((height, width), dtype=np.float64)
    center_labels = np.zeros((height, width), dtype=np.int32)
    center_valid = np.zeros(nobjects + 1, dtype=np.bool_)

    for label_id in range(1, nobjects + 1):
        center_i = int(centers_i[label_id - 1])
        center_j = int(centers_j[label_id - 1])
        if (
            center_i >= 0
            and center_i < height
            and center_j >= 0
            and center_j < width
            and labels[center_i, center_j] == label_id
        ):
            center_valid[label_id] = True

    for y in prange(height):
        for x in range(width):
            label_id = labels[y, x]
            if label_id <= 0 or label_id > nobjects or not center_valid[label_id]:
                continue
            center_i = centers_i[label_id - 1]
            center_j = centers_j[label_id - 1]
            dy = float(y) - center_i
            dx = float(x) - center_j
            d_from_center[y, x] = np.sqrt(dy * dy + dx * dx)
            center_labels[y, x] = label_id

    return d_from_center, center_labels


measure_object_intensity_distribution.__openhcs_prepare__ = (
    _prepare_measure_object_intensity_distribution
)
