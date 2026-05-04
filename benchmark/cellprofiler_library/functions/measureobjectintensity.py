"""
Converted from CellProfiler: MeasureObjectIntensity
Measures intensity features for identified objects in grayscale images.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from openhcs.core.memory import numpy
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
    
    # Build measurement list
    measurements = []
    for i, label in enumerate(intensity_arrays.object_labels):
        measurements.append(ObjectIntensityMeasurement(
            slice_index=0,
            object_label=int(label),
            integrated_intensity=float(intensity_arrays.integrated_intensity[i]),
            mean_intensity=float(intensity_arrays.mean_intensity[i]),
            std_intensity=float(intensity_arrays.std_intensity[i]),
            min_intensity=float(intensity_arrays.min_intensity[i]),
            max_intensity=float(intensity_arrays.max_intensity[i]),
            integrated_intensity_edge=float(intensity_arrays.integrated_intensity_edge[i]),
            mean_intensity_edge=float(intensity_arrays.mean_intensity_edge[i]),
            std_intensity_edge=float(intensity_arrays.std_intensity_edge[i]),
            min_intensity_edge=float(intensity_arrays.min_intensity_edge[i]),
            max_intensity_edge=float(intensity_arrays.max_intensity_edge[i]),
            mass_displacement=float(intensity_arrays.mass_displacement[i]),
            lower_quartile_intensity=float(
                intensity_arrays.lower_quartile_intensity[i]
            ),
            median_intensity=float(intensity_arrays.median_intensity[i]),
            mad_intensity=float(intensity_arrays.mad_intensity[i]),
            upper_quartile_intensity=float(
                intensity_arrays.upper_quartile_intensity[i]
            ),
            center_mass_intensity_x=float(intensity_arrays.center_mass_intensity_x[i]),
            center_mass_intensity_y=float(intensity_arrays.center_mass_intensity_y[i]),
            center_mass_intensity_z=0.0,
            max_intensity_x=float(intensity_arrays.max_intensity_x[i]),
            max_intensity_y=float(intensity_arrays.max_intensity_y[i]),
            max_intensity_z=0.0,
        ))
    
    return image, measurements


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
