"""Converted from CellProfiler: MeasureObjectIntensityDistribution"""

import logging
import numpy as np
import os
import time
from collections.abc import Callable
from typing import Tuple, List, Any
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import (
    measurement_image_batch_executor,
    special_inputs,
    special_outputs,
)
from openhcs.core.runtime_invocation import RuntimeBatchInvocationRequest
from openhcs.core.runtime_semantics import (
    ObjectIntensityDistributionMeasurementRows,
    dense_object_label_extent_id_domain,
)
from openhcs.core.runtime_values import image_payload_data
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
)
from openhcs.processing.backends.cellprofiler.intensity_distribution import (
    IntensityDistributionPlaneInputs,
    radial_distribution_backend,
)
from openhcs.processing.backends.cellprofiler.zernike import (
    IntensityZernikeMeasurementRowsRequest,
)
from openhcs.interop.cellprofiler.intensity_distribution_settings import (
    IntensityDistributionCenterChoice as CenterChoice,
    IntensityDistributionZernikeMode as ZernikeMode,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
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
    radial_distribution_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    zernike_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
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
    img_2d, labels_2d = IntensityDistributionPlaneInputs(image, labels).arrays()
    
    wants_zernikes = coerce_cellprofiler_enum(ZernikeMode, wants_zernikes)
    object_ids = dense_object_label_extent_id_domain(labels_2d)
    if not object_ids:
        return image, []

    phase_started_at = time.perf_counter()
    radial_backend = radial_distribution_backend(
        backend_provider=radial_distribution_backend_provider,
    )
    radial_arrays = radial_backend.measure_self_centered(
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
    phase_started_at = time.perf_counter()
    measurements = ObjectIntensityDistributionMeasurementRows(
        radial_arrays=radial_arrays,
        object_ids=object_ids,
        bin_count=bin_count,
    ).rows()
    _log_profile(
        "idist_radial_rows",
        time.perf_counter() - phase_started_at,
        function="measure_object_intensity_distribution",
        rows=len(measurements),
    )

    if wants_zernikes != ZernikeMode.NONE:
        phase_started_at = time.perf_counter()
        measurements.extend(
            IntensityZernikeMeasurementRowsRequest(
                image=img_2d,
                labels=labels_2d,
                max_order=zernike_degree,
                include_phase=wants_zernikes == ZernikeMode.MAGNITUDES_AND_PHASE,
                backend_provider=zernike_backend_provider,
            ).rows()
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


def _measure_object_intensity_distribution_batch(
    func: Callable[..., Any],
    requests: tuple[RuntimeBatchInvocationRequest, ...],
    execute_request: Callable[[Callable[..., Any], RuntimeBatchInvocationRequest], Any],
) -> list[Any]:
    """Batch measurement-image invocations sharing one label geometry contract."""
    if len(requests) <= 1:
        return [execute_request(func, request) for request in requests]

    labels_2d_by_request: list[np.ndarray] = []
    images_2d_by_request: list[np.ndarray] = []
    for request in requests:
        labels = request.kwargs.get("labels")
        if labels is None:
            return [execute_request(func, item) for item in requests]
        image_2d, labels_2d = IntensityDistributionPlaneInputs(
            np.asarray(image_payload_data(request.image)),
            labels,
        ).arrays()
        images_2d_by_request.append(image_2d)
        labels_2d_by_request.append(labels_2d)

    first_labels = labels_2d_by_request[0]
    if any(labels.shape != first_labels.shape or not np.array_equal(labels, first_labels) for labels in labels_2d_by_request[1:]):
        return [execute_request(func, item) for item in requests]

    first_kwargs = requests[0].kwargs
    wants_zernikes = coerce_cellprofiler_enum(
        ZernikeMode,
        first_kwargs.get("wants_zernikes", ZernikeMode.NONE),
    )
    radial_backend = radial_distribution_backend(
        backend_provider=first_kwargs.get("radial_distribution_backend_provider"),
    )
    geometry = radial_backend.label_geometry(first_labels)
    object_ids = dense_object_label_extent_id_domain(first_labels)
    if not object_ids:
        return [(request.image, []) for request in requests]

    outputs: list[Any] = []
    for request, image_2d in zip(requests, images_2d_by_request, strict=True):
        kwargs = request.kwargs
        radial_arrays = radial_backend.measure(
            image_2d,
            first_labels,
            geometry.d_to_edge,
            geometry.center_fields.d_from_center,
            geometry.center_fields.center_labels,
            geometry.center_fields.centers_i,
            geometry.center_fields.centers_j,
            bin_count=int(kwargs.get("bin_count", 4)),
            wants_scaled=bool(kwargs.get("wants_scaled", True)),
            maximum_radius=int(kwargs.get("maximum_radius", 100)),
        )
        rows = ObjectIntensityDistributionMeasurementRows(
            radial_arrays=radial_arrays,
            object_ids=object_ids,
            bin_count=int(kwargs.get("bin_count", 4)),
        ).rows()
        if wants_zernikes != ZernikeMode.NONE:
            rows.extend(
                IntensityZernikeMeasurementRowsRequest(
                    image=image_2d,
                    labels=first_labels,
                    max_order=int(kwargs.get("zernike_degree", 9)),
                    include_phase=wants_zernikes == ZernikeMode.MAGNITUDES_AND_PHASE,
                    backend_provider=kwargs.get("zernike_backend_provider"),
                ).rows()
            )
        outputs.append((request.image, rows))
    return outputs


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
measurement_image_batch_executor(_measure_object_intensity_distribution_batch)(
    measure_object_intensity_distribution
)
