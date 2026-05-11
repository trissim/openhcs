"""
Converted from CellProfiler: MeasureObjectIntensity
Measures intensity features for identified objects in grayscale images.
"""

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    RuntimePure2DSliceBatchRequest,
    object_label_measurement_execution,
    pure_2d_batch_executor,
)
from openhcs.core.runtime_values import (
    DenseObjectLabelSliceStack,
    image_payload_data,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    ObjectIntensityLabelInput,
    ObjectIntensityMeasurement,
    ObjectIntensityMeasurementRequest,
)


@dataclass
class ObjectIntensityResults:
    """Collection of intensity measurements for all objects."""
    slice_index: int
    object_count: int
    measurements: List[ObjectIntensityMeasurement]


def _measure_object_intensity_batch(
    request: RuntimePure2DSliceBatchRequest,
) -> list[Any]:
    kwargs = request.kwargs
    label_stack = DenseObjectLabelSliceStack.from_payload(
        kwargs["labels"],
        slice_count=request.slice_count,
        dtype=np.int32,
    )
    if label_stack is None:
        return [
            request.execute_one(slice_index)
            for slice_index in range(request.slice_count)
        ]

    backend_provider = kwargs.get("object_intensity_backend_provider")
    results: list[Any] = []
    for slice_index, slice_2d in enumerate(request.slices_2d):
        measurements = ObjectIntensityMeasurementRequest(
            image=image_payload_data(slice_2d),
            labels=label_stack.slice(slice_index),
            slice_index=slice_index,
            backend_provider=backend_provider,
        ).measurements()
        results.append((slice_2d, measurements))
    return results


@numpy
@object_label_measurement_execution(ObjectLabelMeasurementExecution.FULL_STACK)
def measure_object_intensity(
    image: np.ndarray,
    labels: ObjectIntensityLabelInput,
    object_intensity_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[np.ndarray, List[ObjectIntensityMeasurement]]:
    """
    Measure intensity features for identified objects.
    
    Measures several intensity features for each labeled object including:
    - Integrated, mean, std, min, max intensity (whole object and edge)
    - Mass displacement
    - Quartile intensities and MAD
    - Center of mass and max intensity locations
    
    Args:
        image: Grayscale intensity image (H, W)
        labels: Label image where each object has unique integer label (H, W)
    
    Returns:
        Tuple of (original image, list of intensity measurements per object)
    """
    return image, ObjectIntensityMeasurementRequest(
        image=image,
        labels=labels,
        slice_index=0,
        backend_provider=object_intensity_backend_provider,
    ).measurements()


def _prepare_measure_object_intensity() -> None:
    """Compile object-intensity kernels before benchmark execution."""
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[8:24, 8:24] = 1
    labels[32:56, 32:56] = 2
    measure_object_intensity.__wrapped__(image, labels)


measure_object_intensity.__openhcs_prepare__ = _prepare_measure_object_intensity
pure_2d_batch_executor(_measure_object_intensity_batch)(measure_object_intensity)
