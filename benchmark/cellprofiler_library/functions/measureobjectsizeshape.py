"""
Converted from CellProfiler: MeasureObjectSizeShape
Original: measureobjectsizeshape
"""

import logging
import numpy as np
import os
import time
from typing import Any
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    object_label_measurement_execution,
    special_outputs,
)

from openhcs.core.runtime_semantics import (
    object_shape_measurement_all_field_names,
)
from openhcs.core.runtime_values import ObjectLabelSet
from openhcs.processing.backends.cellprofiler.shape import (
    ObjectSizeShapeMeasurementRowsRequest,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
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


@numpy
@object_label_measurement_execution(ObjectLabelMeasurementExecution.FULL_STACK)
@special_outputs(
    (
        "measurements",
        csv_materializer(fields=list(object_shape_measurement_all_field_names())),
    )
)
def measure_object_size_shape(
    image: np.ndarray,
    labels: np.ndarray | ObjectLabelSet,
    calculate_advanced: bool = True,
    calculate_zernikes: bool = True,
    shape_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    zernike_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Measure size and shape features of labeled objects.
    
    Args:
        image: Input intensity image (H, W)
        labels: Label image where each object has unique integer label (H, W)
        calculate_advanced: Whether to calculate advanced features like moments
        calculate_zernikes: Whether to calculate Zernike moments
    
    Returns:
        Tuple of (original image, list of measurement rows per object)
    """
    total_started_at = time.perf_counter()
    rows = ObjectSizeShapeMeasurementRowsRequest(
        labels=labels,
        calculate_advanced=calculate_advanced,
        calculate_zernikes=calculate_zernikes,
        shape_backend_provider=shape_backend_provider,
        zernike_backend_provider=zernike_backend_provider,
    ).rows()
    _log_profile(
        "moss_total",
        time.perf_counter() - total_started_at,
        function="measure_object_size_shape",
        objects=len(rows),
    )
    return image, rows


def _prepare_measure_object_size_shape() -> None:
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[8:24, 8:24] = 1
    measure_object_size_shape.__wrapped__(image, labels)


measure_object_size_shape.__openhcs_prepare__ = _prepare_measure_object_size_shape
