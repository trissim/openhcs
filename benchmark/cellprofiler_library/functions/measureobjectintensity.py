"""
Converted from CellProfiler: MeasureObjectIntensity
Measures intensity features for identified objects in grayscale images.
"""

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    RuntimePure2DSliceBatchRequest,
    object_label_measurement_execution,
    pure_2d_batch_executor,
)
from openhcs.core.runtime_values import (
    DenseObjectLabelSliceStack,
    ObjectLabelPayload,
    ObjectLabelSet,
    image_payload_data,
    object_label_dense_array,
)
from openhcs.processing.backends.cellprofiler.image_geometry import cellprofiler_grayscale_plane
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.cellprofiler.intensity import object_intensity_backend


@dataclass
class ObjectIntensityMeasurement:
    """Per-object intensity measurements."""
    slice_index: int
    object_label: int
    integrated_intensity: float
    mean_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    integrated_intensity_edge: float
    mean_intensity_edge: float
    std_intensity_edge: float
    min_intensity_edge: float
    max_intensity_edge: float
    mass_displacement: float
    lower_quartile_intensity: float
    median_intensity: float
    mad_intensity: float
    upper_quartile_intensity: float
    center_mass_intensity_x: float
    center_mass_intensity_y: float
    center_mass_intensity_z: float
    max_intensity_x: float
    max_intensity_y: float
    max_intensity_z: float

    @classmethod
    def from_backend_arrays(
        cls,
        arrays: Any,
        *,
        index: int,
        label: int,
        slice_index: int,
    ) -> "ObjectIntensityMeasurement":
        """Materialize one CP object-intensity row from backend arrays."""
        return cls(
            slice_index=slice_index,
            object_label=int(label),
            integrated_intensity=float(arrays.integrated_intensity[index]),
            mean_intensity=float(arrays.mean_intensity[index]),
            std_intensity=float(arrays.std_intensity[index]),
            min_intensity=float(arrays.min_intensity[index]),
            max_intensity=float(arrays.max_intensity[index]),
            integrated_intensity_edge=float(arrays.integrated_intensity_edge[index]),
            mean_intensity_edge=float(arrays.mean_intensity_edge[index]),
            std_intensity_edge=float(arrays.std_intensity_edge[index]),
            min_intensity_edge=float(arrays.min_intensity_edge[index]),
            max_intensity_edge=float(arrays.max_intensity_edge[index]),
            mass_displacement=float(arrays.mass_displacement[index]),
            lower_quartile_intensity=float(arrays.lower_quartile_intensity[index]),
            median_intensity=float(arrays.median_intensity[index]),
            mad_intensity=float(arrays.mad_intensity[index]),
            upper_quartile_intensity=float(arrays.upper_quartile_intensity[index]),
            center_mass_intensity_x=float(arrays.center_mass_intensity_x[index]),
            center_mass_intensity_y=float(arrays.center_mass_intensity_y[index]),
            center_mass_intensity_z=float(arrays.center_mass_intensity_z[index]),
            max_intensity_x=float(arrays.max_intensity_x[index]),
            max_intensity_y=float(arrays.max_intensity_y[index]),
            max_intensity_z=float(arrays.max_intensity_z[index]),
        )

    @classmethod
    def rows_from_backend_arrays(
        cls,
        arrays: Any,
        *,
        slice_index: int,
    ) -> list["ObjectIntensityMeasurement"]:
        """Materialize all CP object-intensity rows from backend arrays."""
        if arrays.object_labels.size == 0:
            return []
        return [
            cls.from_backend_arrays(
                arrays,
                index=index,
                label=int(label),
                slice_index=slice_index,
            )
            for index, label in enumerate(arrays.object_labels)
        ]


@dataclass
class ObjectIntensityResults:
    """Collection of intensity measurements for all objects."""
    slice_index: int
    object_count: int
    measurements: List[ObjectIntensityMeasurement]


ObjectIntensityLabelInput = np.ndarray | ObjectLabelPayload | ObjectLabelSet


@dataclass(frozen=True, slots=True)
class ObjectIntensityMeasurementRequest:
    """Executable request for one object-intensity image/label plane."""

    image: np.ndarray
    labels: ObjectIntensityLabelInput
    slice_index: int
    backend_provider: CellProfilerBackendProvider | None

    @property
    def measurement_image(self) -> np.ndarray:
        image = np.asarray(self.image)
        if image.ndim == 3 and not is_color_image_slice(image):
            return image
        return cellprofiler_grayscale_plane(image, "image")

    @property
    def dense_labels(self) -> np.ndarray:
        return object_label_dense_array(self.labels, dtype=np.int32)

    def measurements(self) -> list[ObjectIntensityMeasurement]:
        """Measure this image/label plane through the selected backend."""
        intensity_arrays = object_intensity_backend(
            backend_provider=self.backend_provider,
        ).measure(
            self.measurement_image,
            self.dense_labels,
        )
        return ObjectIntensityMeasurement.rows_from_backend_arrays(
            intensity_arrays,
            slice_index=self.slice_index,
        )


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
