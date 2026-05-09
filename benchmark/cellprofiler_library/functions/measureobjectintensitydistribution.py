"""Converted from CellProfiler: MeasureObjectIntensityDistribution"""

import logging
import numpy as np
import os
import time
from typing import Tuple, List, Any
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.runtime_semantics import (
    ObjectIntensityDistributionMeasurementFeature,
    ObjectMeasurementValueRow,
    ObjectZernikeDescriptorFeature,
    dense_object_label_extent_id_domain,
    indexed_object_intensity_distribution_feature_name,
    indexed_object_intensity_zernike_feature_name,
)
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.backends.cellprofiler.intensity_distribution import (
    radial_distribution_backend,
)
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


@numpy
@special_inputs("labels")
@special_outputs(
    ("radial_measurements", csv_materializer(
        fields=[
            "object_label",
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
    
    object_ids = dense_object_label_extent_id_domain(labels_2d)
    if not object_ids:
        # Return empty measurements
        return image, measurements
    
    phase_started_at = time.perf_counter()
    radial_arrays = radial_distribution_backend(
        backend_provider=radial_distribution_backend_provider,
    ).measure_self_centered(
        img_2d,
        labels_2d,
        bin_count=bin_count,
        wants_scaled=wants_scaled,
        maximum_radius=maximum_radius,
    )
    _log_profile(
        "idist_radial_backend",
        time.perf_counter() - phase_started_at,
        function="measure_object_intensity_distribution",
        nobjects=len(object_ids),
        bins=radial_arrays.n_bins,
    )

    def append_bin_measurements(bin_idx: int, radial_cv: np.ndarray) -> None:
        # Missing label IDs inside the dense object domain carry no radial
        # fraction; CP-style exports keep RadialCV at zero but mark
        # FracAtD/MeanFrac as NaN.
        bin_index = bin_idx + 1
        for object_label in object_ids:
            obj_idx = object_label - 1
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
            measurements.extend(
                (
                    ObjectMeasurementValueRow(
                        object_label=object_label,
                        feature_name=indexed_object_intensity_distribution_feature_name(
                            ObjectIntensityDistributionMeasurementFeature.FRACTION_AT_DISTANCE,
                            bin_index=bin_index,
                            bin_count=bin_count,
                        ),
                        result_value=frac_at_d,
                    ),
                    ObjectMeasurementValueRow(
                        object_label=object_label,
                        feature_name=indexed_object_intensity_distribution_feature_name(
                            ObjectIntensityDistributionMeasurementFeature.MEAN_FRACTION,
                            bin_index=bin_index,
                            bin_count=bin_count,
                        ),
                        result_value=mean_frac,
                    ),
                    ObjectMeasurementValueRow(
                        object_label=object_label,
                        feature_name=indexed_object_intensity_distribution_feature_name(
                            ObjectIntensityDistributionMeasurementFeature.RADIAL_CV,
                            bin_index=bin_index,
                            bin_count=bin_count,
                        ),
                        result_value=float(radial_cv[obj_idx]),
                    ),
                )
            )
    
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
) -> list[ObjectMeasurementValueRow]:
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

    rows: list[ObjectMeasurementValueRow] = []

    for index, (n, m) in enumerate(zernike_indexes):
        for object_label, magnitude in zip(
            object_ids,
            magnitudes[:, index],
            strict=True,
        ):
            rows.append(
                ObjectMeasurementValueRow(
                    object_label=int(object_label),
                    feature_name=indexed_object_intensity_zernike_feature_name(
                        ObjectZernikeDescriptorFeature.INTENSITY_MAGNITUDE,
                        degree=int(n),
                        repetition=int(m),
                    ),
                    result_value=float(magnitude),
                )
            )

        if wants_zernikes == ZernikeMode.MAGNITUDES_AND_PHASE:
            for object_label, phase in zip(
                object_ids,
                phases[:, index],
                strict=True,
            ):
                rows.append(
                    ObjectMeasurementValueRow(
                        object_label=int(object_label),
                        feature_name=indexed_object_intensity_zernike_feature_name(
                            ObjectZernikeDescriptorFeature.INTENSITY_PHASE,
                            degree=int(n),
                            repetition=int(m),
                        ),
                        result_value=float(phase),
                    )
                )

    return rows


def _empty_zernike_measurement_rows(
    object_count: int,
    zernike_indexes: list[tuple[int, int]],
    *,
    wants_zernikes: ZernikeMode,
) -> list[ObjectMeasurementValueRow]:
    rows: list[ObjectMeasurementValueRow] = []
    for n, m in zernike_indexes:
        for object_label in range(1, object_count + 1):
            rows.append(
                ObjectMeasurementValueRow(
                    object_label=object_label,
                    feature_name=indexed_object_intensity_zernike_feature_name(
                        ObjectZernikeDescriptorFeature.INTENSITY_MAGNITUDE,
                        degree=int(n),
                        repetition=int(m),
                    ),
                    result_value=np.nan,
                )
            )
        if wants_zernikes == ZernikeMode.MAGNITUDES_AND_PHASE:
            for object_label in range(1, object_count + 1):
                rows.append(
                    ObjectMeasurementValueRow(
                        object_label=object_label,
                        feature_name=indexed_object_intensity_zernike_feature_name(
                            ObjectZernikeDescriptorFeature.INTENSITY_PHASE,
                            degree=int(n),
                            repetition=int(m),
                        ),
                        result_value=np.nan,
                    )
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


measure_object_intensity_distribution.__openhcs_prepare__ = (
    _prepare_measure_object_intensity_distribution
)
