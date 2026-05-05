"""
Converted from CellProfiler: MeasureObjectIntensity
Measures intensity features for identified objects in grayscale images.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Tuple

import numpy as np

from openhcs.core.memory import numpy
from openhcs.core.runtime_values import image_payload_data
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
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


@dataclass
class ObjectIntensityResults:
    """Collection of intensity measurements for all objects."""
    slice_index: int
    object_count: int
    measurements: List[ObjectIntensityMeasurement]


def _fixup_scipy_result(result):
    """Convert scipy.ndimage result to proper array format."""
    if np.isscalar(result):
        return np.array([result])
    return np.asarray(result)


def _first_scalar_position(position) -> int:
    """Return the first scalar index from scipy's nested position shapes."""
    if np.isscalar(position):
        return int(position)
    if isinstance(position, np.ndarray):
        return _first_scalar_position(position.tolist())
    if hasattr(position, "__len__") and len(position) > 0:
        return _first_scalar_position(position[0])
    raise ValueError(f"Cannot extract scalar position from {position!r}.")


def _measure_object_intensity_batch(
    func: Callable[..., Any],
    slices_2d: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    slice_count: int,
    execute_slice: Callable[[Callable[..., Any], Any, Mapping[str, Any], int, int], Any],
) -> list[Any]:
    label_stack = _measurement_label_stack(kwargs["labels"], slice_count)
    if label_stack is None:
        return [
            execute_slice(func, slice_2d, kwargs, slice_index, slice_count)
            for slice_index, slice_2d in enumerate(slices_2d)
        ]

    backend = object_intensity_backend(
        backend_provider=kwargs.get("object_intensity_backend_provider"),
    )
    results: list[Any] = []
    for slice_index, slice_2d in enumerate(slices_2d):
        image_plane = _single_plane(np.asarray(image_payload_data(slice_2d)), "image")
        arrays = backend.measure(
            image_plane,
            label_stack[slice_index],
        )
        measurements = _measurements_from_arrays(arrays, slice_index)
        results.append((slice_2d, measurements))
    return results


def _measurement_label_stack(labels: Any, slice_count: int) -> np.ndarray | None:
    label_array = np.asarray(image_payload_data(labels), dtype=np.int32)
    if label_array.ndim == 3 and label_array.shape[0] == slice_count:
        return np.ascontiguousarray(label_array)
    if label_array.ndim == 2:
        return np.ascontiguousarray(
            np.broadcast_to(label_array, (slice_count, *label_array.shape))
        )
    return None


def _measurements_from_arrays(arrays: Any, slice_index: int) -> list[ObjectIntensityMeasurement]:
    if arrays.object_labels.size == 0:
        return []
    return [
        ObjectIntensityMeasurement(
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
            center_mass_intensity_z=0.0,
            max_intensity_x=float(arrays.max_intensity_x[index]),
            max_intensity_y=float(arrays.max_intensity_y[index]),
            max_intensity_z=0.0,
        )
        for index, label in enumerate(arrays.object_labels)
    ]


@numpy
def measure_object_intensity(
    image: np.ndarray,
    labels: np.ndarray,
    object_intensity_backend_provider: CellProfilerBackendProvider | None = None,
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
    intensity_arrays = object_intensity_backend(
        backend_provider=object_intensity_backend_provider,
    ).measure(
        _single_plane(np.asarray(image), "image"),
        _single_plane(np.asarray(labels), "labels"),
    )
    if intensity_arrays.object_labels.size == 0:
        return image, []
    
    return image, _measurements_from_arrays(intensity_arrays, 0)


def _single_plane(array: np.ndarray, name: str) -> np.ndarray:
    """Return a 2-D plane from OpenHCS' singleton stack convention."""
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    raise ValueError(
        f"MeasureObjectIntensity expects a 2-D {name} plane or singleton stack, "
        f"got shape {array.shape!r}."
    )


def _prepare_measure_object_intensity() -> None:
    """Compile object-intensity kernels before benchmark execution."""
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[8:24, 8:24] = 1
    labels[32:56, 32:56] = 2
    measure_object_intensity.__wrapped__(image, labels)


measure_object_intensity.__openhcs_prepare__ = _prepare_measure_object_intensity
measure_object_intensity.__openhcs_pure_2d_batch_executor__ = (
    _measure_object_intensity_batch
)
