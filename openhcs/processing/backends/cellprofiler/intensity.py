"""Intensity-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
from typing import Any

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
import scipy.ndimage
import skimage.segmentation

from openhcs.constants.constants import MemoryType
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.memory import numpy as numpy_decorator
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    RuntimePure2DSliceBatchRequest,
    object_label_measurement_execution,
    pure_2d_batch_executor,
)
from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.runtime_values import (
    DenseObjectLabelSliceStack,
    ObjectLabelPayload,
    ObjectLabelSet,
    image_payload_data,
    object_label_dense_array,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    cellprofiler_grayscale_plane,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class RescaleMethod(Enum):
    STRETCH = "stretch"
    MANUAL_INPUT_RANGE = "manual_input_range"
    MANUAL_IO_RANGE = "manual_io_range"
    DIVIDE_BY_IMAGE_MINIMUM = "divide_by_image_minimum"
    DIVIDE_BY_IMAGE_MAXIMUM = "divide_by_image_maximum"
    DIVIDE_BY_VALUE = "divide_by_value"


class AutomaticLow(Enum):
    CUSTOM = "custom"
    EACH_IMAGE = "each_image"


class AutomaticHigh(Enum):
    CUSTOM = "custom"
    EACH_IMAGE = "each_image"


@dataclass(frozen=True, slots=True)
class ObjectIntensityArrays:
    """Dense per-object intensity measurement arrays."""

    object_labels: np.ndarray
    integrated_intensity: np.ndarray
    mean_intensity: np.ndarray
    std_intensity: np.ndarray
    min_intensity: np.ndarray
    max_intensity: np.ndarray
    integrated_intensity_edge: np.ndarray
    mean_intensity_edge: np.ndarray
    std_intensity_edge: np.ndarray
    min_intensity_edge: np.ndarray
    max_intensity_edge: np.ndarray
    mass_displacement: np.ndarray
    lower_quartile_intensity: np.ndarray
    median_intensity: np.ndarray
    mad_intensity: np.ndarray
    upper_quartile_intensity: np.ndarray
    center_mass_intensity_x: np.ndarray
    center_mass_intensity_y: np.ndarray
    center_mass_intensity_z: np.ndarray
    max_intensity_x: np.ndarray
    max_intensity_y: np.ndarray
    max_intensity_z: np.ndarray


@dataclass(frozen=True, slots=True)
class ImageIntensityPercentileSpec:
    """Percentile calculation policy for image-intensity rows."""

    enabled: bool = False
    raw_percentiles: str = "10,90"

    @property
    def values(self) -> list[int]:
        percentiles = []
        for percentile in self.raw_percentiles.replace(" ", "").split(","):
            if percentile == "":
                continue
            if percentile.isdigit() and 0 <= int(percentile) <= 100:
                percentiles.append(int(percentile))
        return sorted(set(percentiles))

    def measurements_for(self, pixels: np.ndarray) -> dict[int, float]:
        if not self.enabled:
            return {}
        parsed_percentiles = self.values
        if pixels.size == 0:
            return {percentile: 0.0 for percentile in parsed_percentiles}
        if not parsed_percentiles:
            return {}
        percentile_results = np.percentile(pixels, parsed_percentiles)
        return {
            percentile: float(value)
            for percentile, value in zip(parsed_percentiles, percentile_results)
        }


@dataclass
class ImageIntensityMeasurement:
    """CellProfiler-compatible intensity measurements for one image region."""

    slice_index: int
    total_intensity: float
    mean_intensity: float
    median_intensity: float
    std_intensity: float
    mad_intensity: float
    min_intensity: float
    max_intensity: float
    total_area: int
    percent_maximal: float
    lower_quartile_intensity: float
    upper_quartile_intensity: float
    percentile_values: str

    @classmethod
    def from_pixels(
        cls,
        pixels: np.ndarray,
        *,
        percentile_spec: ImageIntensityPercentileSpec,
    ) -> "ImageIntensityMeasurement":
        """Build the authoritative image-intensity measurement row."""
        pixels = pixels[np.isfinite(pixels)]
        pixel_count = pixels.size
        percentile_dict = percentile_spec.measurements_for(pixels)

        if pixel_count == 0:
            pixel_sum = 0.0
            pixel_mean = 0.0
            pixel_std = 0.0
            pixel_mad = 0.0
            pixel_median = 0.0
            pixel_min = 0.0
            pixel_max = 0.0
            pixel_pct_max = 0.0
            pixel_lower_qrt = 0.0
            pixel_upper_qrt = 0.0
        else:
            pixel_sum = float(np.sum(pixels))
            pixel_mean = pixel_sum / float(pixel_count)
            pixel_std = float(np.std(pixels))
            pixel_median = float(np.median(pixels))
            pixel_mad = float(np.median(np.abs(pixels - pixel_median)))
            pixel_min = float(np.min(pixels))
            pixel_max = float(np.max(pixels))
            pixel_pct_max = (
                100.0 * float(np.sum(pixels == pixel_max)) / float(pixel_count)
            )
            quartiles = np.percentile(pixels, [25, 75])
            pixel_lower_qrt = float(quartiles[0])
            pixel_upper_qrt = float(quartiles[1])

        return cls(
            slice_index=0,
            total_intensity=pixel_sum,
            mean_intensity=pixel_mean,
            median_intensity=pixel_median,
            std_intensity=pixel_std,
            mad_intensity=pixel_mad,
            min_intensity=pixel_min,
            max_intensity=pixel_max,
            total_area=int(pixel_count),
            percent_maximal=pixel_pct_max,
            lower_quartile_intensity=pixel_lower_qrt,
            upper_quartile_intensity=pixel_upper_qrt,
            percentile_values=json.dumps(percentile_dict),
        )


@dataclass
class ObjectIntensityMeasurement:
    """Per-object CellProfiler-compatible intensity measurements."""

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
        """Materialize one CellProfiler object-intensity row from backend arrays."""
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
        """Materialize all CellProfiler object-intensity rows from backend arrays."""
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


ObjectIntensityLabelInput = np.ndarray | ObjectLabelPayload | ObjectLabelSet


@dataclass(frozen=True, slots=True)
class ObjectIntensityResults:
    """Collection of intensity measurements for all objects."""

    slice_index: int
    object_count: int
    measurements: list["ObjectIntensityMeasurement"]


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


class ObjectIntensityBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Object-intensity operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def measure(
        self,
        image: np.ndarray,
        labels: np.ndarray,
    ) -> ObjectIntensityArrays:
        """Measure object intensity arrays for one image plane."""


class NumbaNumpyObjectIntensityBackendStrategy(ObjectIntensityBackendStrategy):
    """Numba-accelerated NumPy object-intensity backend."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def measure(
        self,
        image: np.ndarray,
        labels: np.ndarray,
    ) -> ObjectIntensityArrays:
        image_array = np.ascontiguousarray(image, dtype=np.float64)
        label_array = np.ascontiguousarray(labels, dtype=np.int64)
        if image_array.ndim != 2 or label_array.ndim != 2:
            if image_array.ndim == 3 and label_array.ndim == 3:
                return self._measure_3d(image_array, label_array)
            raise NotImplementedError(
                "NumPy object-intensity backend supports 2-D and 3-D arrays."
            )
        if image_array.shape != label_array.shape:
            raise ValueError("image and labels must have matching shapes.")

        max_label = int(label_array.max()) if label_array.size else 0
        object_labels = np.arange(1, max_label + 1, dtype=np.int64)
        object_count = int(object_labels.size)
        if object_count == 0:
            return _empty_intensity_arrays(object_labels)

        label_to_index = np.full(max_label + 1, -1, dtype=np.int64)
        label_to_index[1:] = np.arange(object_count, dtype=np.int64)

        arrays = _object_intensity_scan_numba(
            image_array,
            label_array,
            object_labels,
            label_to_index,
        )
        lower, median, upper, mad = _object_intensity_quantiles(
            image_array,
            label_array,
            object_labels,
            label_to_index,
            arrays[0].astype(np.int64, copy=False),
        )
        return ObjectIntensityArrays(
            object_labels=object_labels.astype(np.int32, copy=False),
            integrated_intensity=arrays[1],
            mean_intensity=arrays[2],
            std_intensity=arrays[3],
            min_intensity=arrays[4],
            max_intensity=arrays[5],
            integrated_intensity_edge=arrays[6],
            mean_intensity_edge=arrays[7],
            std_intensity_edge=arrays[8],
            min_intensity_edge=arrays[9],
            max_intensity_edge=arrays[10],
            mass_displacement=arrays[11],
            lower_quartile_intensity=lower,
            median_intensity=median,
            mad_intensity=mad,
            upper_quartile_intensity=upper,
            center_mass_intensity_x=arrays[12],
            center_mass_intensity_y=arrays[13],
            center_mass_intensity_z=np.zeros(object_count, dtype=np.float64),
            max_intensity_x=arrays[14],
            max_intensity_y=arrays[15],
            max_intensity_z=np.zeros(object_count, dtype=np.float64),
        )

    def _measure_3d(
        self,
        image_array: np.ndarray,
        label_array: np.ndarray,
    ) -> ObjectIntensityArrays:
        if image_array.shape != label_array.shape:
            raise ValueError("image and labels must have matching shapes.")
        max_label = int(label_array.max()) if label_array.size else 0
        object_labels = np.arange(1, max_label + 1, dtype=np.int64)
        object_count = int(object_labels.size)
        if object_count == 0:
            return _empty_intensity_arrays(object_labels)
        label_to_index = np.full(max_label + 1, -1, dtype=np.int64)
        label_to_index[1:] = np.arange(object_count, dtype=np.int64)
        arrays = _object_intensity_scan_3d_numba(
            np.ascontiguousarray(image_array),
            np.ascontiguousarray(label_array),
            object_labels,
            label_to_index,
        )
        lower, median, upper, mad = _object_intensity_quantiles_3d_numba(
            np.ascontiguousarray(image_array),
            np.ascontiguousarray(label_array),
            label_to_index,
            arrays[0].astype(np.int64, copy=False),
            1.0 / 3.0,
        )
        return ObjectIntensityArrays(
            object_labels=object_labels.astype(np.int32, copy=False),
            integrated_intensity=arrays[1],
            mean_intensity=arrays[2],
            std_intensity=arrays[3],
            min_intensity=arrays[4],
            max_intensity=arrays[5],
            integrated_intensity_edge=arrays[6],
            mean_intensity_edge=arrays[7],
            std_intensity_edge=arrays[8],
            min_intensity_edge=arrays[9],
            max_intensity_edge=arrays[10],
            mass_displacement=arrays[11],
            lower_quartile_intensity=lower,
            median_intensity=median,
            mad_intensity=mad,
            upper_quartile_intensity=upper,
            center_mass_intensity_x=arrays[12],
            center_mass_intensity_y=arrays[13],
            center_mass_intensity_z=arrays[14],
            max_intensity_x=arrays[15],
            max_intensity_y=arrays[16],
            max_intensity_z=arrays[17],
        )

    def prepare_backend(self) -> None:
        """Compile object-intensity kernels outside measured execution."""
        image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
        labels = np.zeros(image.shape, dtype=np.int32)
        labels[4:16, 4:16] = 1
        labels[16:28, 16:28] = 2
        self.measure(image, labels)
        image_3d = np.linspace(0.0, 1.0, 8 * 16 * 16, dtype=np.float32).reshape(
            (8, 16, 16)
        )
        labels_3d = np.zeros(image_3d.shape, dtype=np.int32)
        labels_3d[1:4, 3:9, 3:9] = 1
        labels_3d[4:7, 7:14, 7:14] = 2
        self.measure(image_3d, labels_3d)


def object_intensity_backend(
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> ObjectIntensityBackendStrategy:
    """Return the selected object-intensity backend."""
    return ObjectIntensityBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


def measure_object_intensity_batch(
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


@numpy_decorator
@object_label_measurement_execution(ObjectLabelMeasurementExecution.FULL_STACK)
def measure_object_intensity(
    image: np.ndarray,
    labels: ObjectIntensityLabelInput,
    object_intensity_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, list[ObjectIntensityMeasurement]]:
    """Measure CellProfiler intensity features for identified objects."""
    return image, ObjectIntensityMeasurementRequest(
        image=image,
        labels=labels,
        slice_index=0,
        backend_provider=object_intensity_backend_provider,
    ).measurements()


def prepare_measure_object_intensity() -> None:
    """Compile object-intensity kernels before benchmark execution."""
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[8:24, 8:24] = 1
    labels[32:56, 32:56] = 2
    measure_object_intensity.__wrapped__(image, labels)


@numpy_decorator
def measure_image_intensity(
    image: np.ndarray,
    calculate_percentiles: bool = False,
    percentiles: str = "10,90",
) -> tuple[np.ndarray, ImageIntensityMeasurement]:
    """Measure CellProfiler intensity features across an entire image."""
    measurements = ImageIntensityMeasurement.from_pixels(
        image.flatten(),
        percentile_spec=ImageIntensityPercentileSpec(
            enabled=calculate_percentiles,
            raw_percentiles=percentiles,
        ),
    )
    return image, measurements


@numpy_decorator
def measure_image_intensity_masked(
    image: np.ndarray,
    labels: np.ndarray,
    calculate_percentiles: bool = False,
    percentiles: str = "10,90",
) -> tuple[np.ndarray, ImageIntensityMeasurement]:
    """Measure aggregate image intensity within nonzero label regions."""
    mask = labels > 0
    measurements = ImageIntensityMeasurement.from_pixels(
        image[mask].flatten(),
        percentile_spec=ImageIntensityPercentileSpec(
            enabled=calculate_percentiles,
            raw_percentiles=percentiles,
        ),
    )
    return image, measurements


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy_decorator(contract=ProcessingContract.PURE_2D)
def rescale_intensity(
    image: np.ndarray,
    rescale_method: RescaleMethod = RescaleMethod.STRETCH,
    automatic_low: AutomaticLow = AutomaticLow.EACH_IMAGE,
    automatic_high: AutomaticHigh = AutomaticHigh.EACH_IMAGE,
    source_low: float = 0.0,
    source_high: float = 1.0,
    dest_low: float = 0.0,
    dest_high: float = 1.0,
    divisor_value: float = 1.0,
) -> np.ndarray:
    """Rescale CellProfiler image intensity using its declared range policy."""
    from skimage.exposure import rescale_intensity as skimage_rescale

    rescale_method = coerce_cellprofiler_enum(RescaleMethod, rescale_method)
    automatic_low = coerce_cellprofiler_enum(AutomaticLow, automatic_low)
    automatic_high = coerce_cellprofiler_enum(AutomaticHigh, automatic_high)
    data = image.astype(np.float64)

    if rescale_method == RescaleMethod.STRETCH:
        in_min = np.min(data)
        in_max = np.max(data)
        if in_min == in_max:
            return np.zeros_like(data)
        rescaled = skimage_rescale(
            data,
            in_range=(in_min, in_max),
            out_range=(0.0, 1.0),
        )
    elif rescale_method == RescaleMethod.MANUAL_INPUT_RANGE:
        rescaled = skimage_rescale(
            data,
            in_range=rescale_source_range(
                data,
                automatic_low,
                automatic_high,
                source_low,
                source_high,
            ),
            out_range=(0.0, 1.0),
        )
    elif rescale_method == RescaleMethod.MANUAL_IO_RANGE:
        rescaled = skimage_rescale(
            data,
            in_range=rescale_source_range(
                data,
                automatic_low,
                automatic_high,
                source_low,
                source_high,
            ),
            out_range=(dest_low, dest_high),
        )
    elif rescale_method == RescaleMethod.DIVIDE_BY_IMAGE_MINIMUM:
        src_min = np.min(data)
        if src_min == 0.0:
            raise ZeroDivisionError("Cannot divide pixel intensity by 0.")
        rescaled = data / src_min
    elif rescale_method == RescaleMethod.DIVIDE_BY_IMAGE_MAXIMUM:
        src_max = np.max(data)
        if src_max == 0.0:
            src_max = 1.0
        rescaled = data / src_max
    elif rescale_method == RescaleMethod.DIVIDE_BY_VALUE:
        if divisor_value == 0.0:
            raise ZeroDivisionError("Cannot divide pixel intensity by 0.")
        rescaled = data / divisor_value
    else:
        in_min = np.min(data)
        in_max = np.max(data)
        if in_min == in_max:
            return np.zeros_like(data)
        rescaled = skimage_rescale(
            data,
            in_range=(in_min, in_max),
            out_range=(0.0, 1.0),
        )

    return rescaled.astype(np.float32)


def rescale_source_range(
    data: np.ndarray,
    automatic_low: AutomaticLow,
    automatic_high: AutomaticHigh,
    source_low: float,
    source_high: float,
) -> tuple[float, float]:
    """Determine the CellProfiler source intensity range from settings."""
    src_min = float(np.min(data)) if automatic_low == AutomaticLow.EACH_IMAGE else source_low
    src_max = float(np.max(data)) if automatic_high == AutomaticHigh.EACH_IMAGE else source_high
    return src_min, src_max


@numpy_decorator
def rescale_intensity_match_maximum(
    image: np.ndarray,
) -> np.ndarray:
    """Scale image[0] so its maximum matches image[1]'s maximum."""
    input_data = image[0].astype(np.float64)
    reference_data = image[1].astype(np.float64)
    image_max = np.max(input_data)
    reference_max = np.max(reference_data)
    if image_max == 0:
        result = input_data
    else:
        result = (input_data * reference_max) / image_max
    return result.astype(np.float32)[np.newaxis, :, :]


measure_object_intensity.__openhcs_prepare__ = prepare_measure_object_intensity
pure_2d_batch_executor(measure_object_intensity_batch)(measure_object_intensity)


def _empty_intensity_arrays(object_labels: np.ndarray) -> ObjectIntensityArrays:
    empty = np.zeros(0, dtype=float)
    return ObjectIntensityArrays(
        object_labels=object_labels.astype(np.int32, copy=False),
        integrated_intensity=empty,
        mean_intensity=empty,
        std_intensity=empty,
        min_intensity=empty,
        max_intensity=empty,
        integrated_intensity_edge=empty,
        mean_intensity_edge=empty,
        std_intensity_edge=empty,
        min_intensity_edge=empty,
        max_intensity_edge=empty,
        mass_displacement=empty,
        lower_quartile_intensity=empty,
        median_intensity=empty,
        mad_intensity=empty,
        upper_quartile_intensity=empty,
        center_mass_intensity_x=empty,
        center_mass_intensity_y=empty,
        center_mass_intensity_z=empty,
        max_intensity_x=empty,
        max_intensity_y=empty,
        max_intensity_z=empty,
    )


def _object_intensity_nd_scipy(
    image: np.ndarray,
    labels: np.ndarray,
) -> ObjectIntensityArrays:
    """Measure CellProfiler-compatible object intensities for one 3-D domain."""
    if image.shape != labels.shape:
        raise ValueError("image and labels must have matching shapes.")
    max_label = int(labels.max()) if labels.size else 0
    object_labels = np.arange(1, max_label + 1, dtype=np.int32)
    if object_labels.size == 0:
        return _empty_intensity_arrays(object_labels)

    finite_mask = np.isfinite(image)
    masked_labels = labels.copy()
    masked_labels[~finite_mask] = 0
    object_mask = masked_labels > 0
    if not np.any(object_mask):
        return _empty_intensity_arrays(object_labels)

    masked_image = image.copy()
    masked_image[~finite_mask] = 0.0
    outlines = skimage.segmentation.find_boundaries(masked_labels, mode="inner")
    masked_outlines = outlines & object_mask
    mesh_z, mesh_y, mesh_x = np.mgrid[
        0 : image.shape[0],
        0 : image.shape[1],
        0 : image.shape[2],
    ]

    counts = _fixup_scipy_result(
        scipy.ndimage.sum(np.ones(int(object_mask.sum())), masked_labels[object_mask], object_labels)
    )
    integrated = _fixup_scipy_result(
        scipy.ndimage.sum(masked_image[object_mask], masked_labels[object_mask], object_labels)
    )
    means = np.divide(
        integrated,
        counts,
        out=np.zeros_like(integrated, dtype=np.float64),
        where=counts != 0,
    )
    stds = np.sqrt(
        _fixup_scipy_result(
            scipy.ndimage.mean(
                (masked_image[object_mask] - means[masked_labels[object_mask] - 1]) ** 2,
                masked_labels[object_mask],
                object_labels,
            )
        )
    )
    min_values = _fixup_scipy_result(
        scipy.ndimage.minimum(masked_image[object_mask], masked_labels[object_mask], object_labels)
    )
    max_values = _fixup_scipy_result(
        scipy.ndimage.maximum(masked_image[object_mask], masked_labels[object_mask], object_labels)
    )

    max_position = np.asarray(
        _fixup_scipy_result(
            scipy.ndimage.maximum_position(
                masked_image[object_mask],
                masked_labels[object_mask],
                object_labels,
            )
        ),
        dtype=int,
    ).reshape((object_labels.size,))
    label_values = masked_labels[object_mask]
    max_x = mesh_x[object_mask][max_position].astype(np.float64, copy=False)
    max_y = mesh_y[object_mask][max_position].astype(np.float64, copy=False)
    max_z = mesh_z[object_mask][max_position].astype(np.float64, copy=False)

    cm_x = _fixup_scipy_result(
        scipy.ndimage.mean(mesh_x[object_mask], label_values, object_labels)
    )
    cm_y = _fixup_scipy_result(
        scipy.ndimage.mean(mesh_y[object_mask], label_values, object_labels)
    )
    cm_z = _fixup_scipy_result(
        scipy.ndimage.mean(mesh_z[object_mask], label_values, object_labels)
    )
    weighted_x = _fixup_scipy_result(
        scipy.ndimage.sum(mesh_x[object_mask] * masked_image[object_mask], label_values, object_labels)
    )
    weighted_y = _fixup_scipy_result(
        scipy.ndimage.sum(mesh_y[object_mask] * masked_image[object_mask], label_values, object_labels)
    )
    weighted_z = _fixup_scipy_result(
        scipy.ndimage.sum(mesh_z[object_mask] * masked_image[object_mask], label_values, object_labels)
    )
    cmi_x = np.divide(weighted_x, integrated, out=np.zeros_like(weighted_x), where=integrated != 0)
    cmi_y = np.divide(weighted_y, integrated, out=np.zeros_like(weighted_y), where=integrated != 0)
    cmi_z = np.divide(weighted_z, integrated, out=np.zeros_like(weighted_z), where=integrated != 0)
    mass_displacement = np.sqrt(
        (cm_x - cmi_x) * (cm_x - cmi_x)
        + (cm_y - cmi_y) * (cm_y - cmi_y)
        + (cm_z - cmi_z) * (cm_z - cmi_z)
    )

    lower, median, upper, mad = _object_intensity_quantiles_nd(
        masked_image[object_mask],
        label_values,
        object_labels,
        mad_fraction=1.0 / float(image.ndim),
    )
    edge_sums, edge_means, edge_stds, edge_min, edge_max = _edge_intensity_arrays(
        masked_image,
        masked_labels,
        masked_outlines,
        object_labels,
    )
    return ObjectIntensityArrays(
        object_labels=object_labels,
        integrated_intensity=integrated,
        mean_intensity=means,
        std_intensity=stds,
        min_intensity=min_values,
        max_intensity=max_values,
        integrated_intensity_edge=edge_sums,
        mean_intensity_edge=edge_means,
        std_intensity_edge=edge_stds,
        min_intensity_edge=edge_min,
        max_intensity_edge=edge_max,
        mass_displacement=mass_displacement,
        lower_quartile_intensity=lower,
        median_intensity=median,
        mad_intensity=mad,
        upper_quartile_intensity=upper,
        center_mass_intensity_x=cmi_x,
        center_mass_intensity_y=cmi_y,
        center_mass_intensity_z=cmi_z,
        max_intensity_x=max_x,
        max_intensity_y=max_y,
        max_intensity_z=max_z,
    )


def _fixup_scipy_result(result: object) -> np.ndarray:
    if np.isscalar(result):
        return np.asarray([result], dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def _object_intensity_quantiles_nd(
    values: np.ndarray,
    labels: np.ndarray,
    object_labels: np.ndarray,
    *,
    mad_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.zeros(object_labels.size, dtype=np.float64)
    median = np.zeros(object_labels.size, dtype=np.float64)
    upper = np.zeros(object_labels.size, dtype=np.float64)
    mad = np.zeros(object_labels.size, dtype=np.float64)
    for index, label in enumerate(object_labels):
        group = values[labels == label]
        if group.size == 0:
            continue
        ordered = np.sort(group)
        lower[index] = _quantile_from_sorted_values(ordered, 0.25)
        median[index] = _quantile_from_sorted_values(ordered, 0.5)
        upper[index] = _quantile_from_sorted_values(ordered, 0.75)
        mad[index] = _quantile_from_sorted_values(
            np.sort(np.abs(group - median[index])),
            mad_fraction,
        )
    return lower, median, upper, mad


def _quantile_from_sorted_values(values: np.ndarray, fraction: float) -> float:
    qindex = values.size * fraction
    low = int(qindex)
    qfraction = qindex - low
    last = values.size - 1
    if low < last:
        return float(values[low] * (1.0 - qfraction) + values[low + 1] * qfraction)
    return float(values[last])


def _edge_intensity_arrays(
    image: np.ndarray,
    labels: np.ndarray,
    edge_mask: np.ndarray,
    object_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge_labels = labels[edge_mask]
    edge_values = image[edge_mask]
    edge_sums = _fixup_scipy_result(
        scipy.ndimage.sum(edge_values, edge_labels, object_labels)
    )
    edge_counts = _fixup_scipy_result(
        scipy.ndimage.sum(np.ones(edge_values.size), edge_labels, object_labels)
    )
    edge_means = np.divide(
        edge_sums,
        edge_counts,
        out=np.zeros_like(edge_sums),
        where=edge_counts != 0,
    )
    edge_stds = np.sqrt(
        _fixup_scipy_result(
            scipy.ndimage.mean(
                (edge_values - edge_means[edge_labels - 1]) ** 2,
                edge_labels,
                object_labels,
            )
        )
    )
    edge_min = _fixup_scipy_result(
        scipy.ndimage.minimum(edge_values, edge_labels, object_labels)
    )
    edge_max = _fixup_scipy_result(
        scipy.ndimage.maximum(edge_values, edge_labels, object_labels)
    )
    edge_min[edge_counts == 0] = 0.0
    edge_max[edge_counts == 0] = 0.0
    return edge_sums, edge_means, edge_stds, edge_min, edge_max


def _object_intensity_quantiles(
    image: np.ndarray,
    labels: np.ndarray,
    object_labels: np.ndarray,
    label_to_index: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _object_intensity_quantiles_grouped_numba(
        image,
        labels,
        label_to_index,
        counts.astype(np.int64, copy=False),
    )


@njit(cache=True)
def _object_intensity_quantiles_grouped_numba(
    image: np.ndarray,
    labels: np.ndarray,
    label_to_index: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    object_count = counts.size
    lower = np.zeros(object_count, dtype=np.float64)
    median = np.zeros(object_count, dtype=np.float64)
    upper = np.zeros(object_count, dtype=np.float64)
    mad = np.zeros(object_count, dtype=np.float64)

    total_count = 0
    for index in range(object_count):
        total_count += int(counts[index])
    if total_count <= 0:
        return lower, median, upper, mad

    offsets = np.empty(object_count + 1, dtype=np.int64)
    offsets[0] = 0
    for index in range(object_count):
        offsets[index + 1] = offsets[index] + int(counts[index])

    write_offsets = offsets[:-1].copy()
    values = np.empty(total_count, dtype=np.float64)
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0 or label >= label_to_index.size:
                continue
            index = int(label_to_index[label])
            if index < 0:
                continue
            value = float(image[y, x])
            if not np.isfinite(value):
                continue
            offset = write_offsets[index]
            values[offset] = value
            write_offsets[index] = offset + 1

    for index in range(object_count):
        start = int(offsets[index])
        count = int(counts[index])
        if count <= 0:
            continue
        group = values[start:start + count].copy()
        lower[index] = _quantile_from_dense_group_partition(group, 0.25)
        median[index] = _quantile_from_dense_group_partition(group, 0.5)
        upper[index] = _quantile_from_dense_group_partition(group, 0.75)

    write_offsets = offsets[:-1].copy()
    deviations = np.empty(total_count, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0 or label >= label_to_index.size:
                continue
            index = int(label_to_index[label])
            if index < 0:
                continue
            value = float(image[y, x])
            if not np.isfinite(value):
                continue
            offset = write_offsets[index]
            deviations[offset] = abs(value - median[index])
            write_offsets[index] = offset + 1

    for index in range(object_count):
        start = int(offsets[index])
        count = int(counts[index])
        if count <= 0:
            continue
        group = deviations[start:start + count].copy()
        mad[index] = _quantile_from_dense_group_partition(group, 0.5)

    return lower, median, upper, mad


@njit(cache=True)
def _quantile_from_dense_group_partition(
    values: np.ndarray,
    fraction: float,
) -> float:
    count = values.size
    if count <= 0:
        return 0.0
    qindex = count * fraction
    low = int(qindex)
    qfraction = qindex - low
    last = count - 1
    if low >= last:
        return _partition_value(values, last)
    low_value = _partition_value(values, low)
    high_value = _partition_value(values, low + 1)
    return low_value * (1.0 - qfraction) + high_value * qfraction


@njit(cache=True)
def _partition_value(values: np.ndarray, index: int) -> float:
    partitioned = np.partition(values.copy(), index)
    return float(partitioned[index])


@njit(cache=True)
def _quantile_from_dense_sorted_group(
    sorted_values: np.ndarray,
    fraction: float,
) -> float:
    count = sorted_values.size
    if count <= 0:
        return 0.0
    qindex = count * fraction
    low = int(qindex)
    qfraction = qindex - low
    last = count - 1
    if low < last:
        return sorted_values[low] * (1.0 - qfraction) + sorted_values[low + 1] * qfraction
    return sorted_values[last]


@njit(cache=True)
def _group_starts_numba(
    sorted_labels: np.ndarray,
    label_to_index: np.ndarray,
    starts: np.ndarray,
) -> None:
    for offset in range(sorted_labels.size):
        label = sorted_labels[offset]
        if label < 0 or label >= label_to_index.size:
            continue
        index = label_to_index[label]
        if index >= 0 and starts[index] < 0:
            starts[index] = offset


@njit(cache=True)
def _quartiles_from_sorted_numba(
    sorted_values: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_count = counts.size
    lower = np.zeros(object_count, dtype=np.float64)
    median = np.zeros(object_count, dtype=np.float64)
    upper = np.zeros(object_count, dtype=np.float64)
    for index in range(object_count):
        lower[index] = _quantile_from_sorted_group(
            sorted_values,
            starts[index],
            counts[index],
            0.25,
        )
        median[index] = _quantile_from_sorted_group(
            sorted_values,
            starts[index],
            counts[index],
            0.5,
        )
        upper[index] = _quantile_from_sorted_group(
            sorted_values,
            starts[index],
            counts[index],
            0.75,
        )
    return lower, median, upper


@njit(cache=True)
def _median_from_sorted_numba(
    sorted_values: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    object_count = counts.size
    output = np.zeros(object_count, dtype=np.float64)
    for index in range(object_count):
        output[index] = _quantile_from_sorted_group(
            sorted_values,
            starts[index],
            counts[index],
            0.5,
        )
    return output


@njit(cache=True)
def _quantile_from_sorted_group(
    sorted_values: np.ndarray,
    start: int,
    count: int,
    fraction: float,
) -> float:
    if count <= 0:
        return 0.0
    qindex = start + count * fraction
    low = int(qindex)
    qfraction = qindex - low
    last = start + count - 1
    if low < last:
        return sorted_values[low] * (1.0 - qfraction) + sorted_values[low + 1] * qfraction
    return sorted_values[last]


@njit(cache=True)
def _object_intensity_scan_numba(
    image: np.ndarray,
    labels: np.ndarray,
    object_labels: np.ndarray,
    label_to_index: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    height, width = image.shape
    object_count = object_labels.size
    counts = np.zeros(object_count, dtype=np.float64)
    sums = np.zeros(object_count, dtype=np.float64)
    sumsq = np.zeros(object_count, dtype=np.float64)
    min_values = np.full(object_count, np.inf, dtype=np.float64)
    max_values = np.full(object_count, -np.inf, dtype=np.float64)
    sum_x = np.zeros(object_count, dtype=np.float64)
    sum_y = np.zeros(object_count, dtype=np.float64)
    weighted_x = np.zeros(object_count, dtype=np.float64)
    weighted_y = np.zeros(object_count, dtype=np.float64)
    max_x = np.zeros(object_count, dtype=np.float64)
    max_y = np.zeros(object_count, dtype=np.float64)

    edge_counts = np.zeros(object_count, dtype=np.float64)
    edge_sums = np.zeros(object_count, dtype=np.float64)
    edge_sumsq = np.zeros(object_count, dtype=np.float64)
    edge_min_values = np.full(object_count, np.inf, dtype=np.float64)
    edge_max_values = np.full(object_count, -np.inf, dtype=np.float64)

    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label <= 0 or label >= label_to_index.size:
                continue
            index = label_to_index[label]
            if index < 0:
                continue
            value = image[y, x]
            if not np.isfinite(value):
                continue

            counts[index] += 1.0
            sums[index] += value
            sumsq[index] += value * value
            sum_x[index] += x
            sum_y[index] += y
            weighted_x[index] += x * value
            weighted_y[index] += y * value
            if value < min_values[index]:
                min_values[index] = value
            if value >= max_values[index]:
                max_values[index] = value
                max_x[index] = x
                max_y[index] = y

            if _is_inner_boundary_pixel(labels, y, x, label):
                edge_counts[index] += 1.0
                edge_sums[index] += value
                edge_sumsq[index] += value * value
                if value < edge_min_values[index]:
                    edge_min_values[index] = value
                if value > edge_max_values[index]:
                    edge_max_values[index] = value

    means = np.zeros(object_count, dtype=np.float64)
    stds = np.zeros(object_count, dtype=np.float64)
    edge_means = np.zeros(object_count, dtype=np.float64)
    edge_stds = np.zeros(object_count, dtype=np.float64)
    mass_displacement = np.zeros(object_count, dtype=np.float64)
    center_mass_x = np.zeros(object_count, dtype=np.float64)
    center_mass_y = np.zeros(object_count, dtype=np.float64)
    for index in range(object_count):
        if counts[index] > 0.0:
            means[index] = sums[index] / counts[index]
            variance = sumsq[index] / counts[index] - means[index] * means[index]
            if variance < 0.0 and variance > -1e-15:
                variance = 0.0
            stds[index] = np.sqrt(variance)
            center_x = sum_x[index] / counts[index]
            center_y = sum_y[index] / counts[index]
            if sums[index] != 0.0:
                center_mass_x[index] = weighted_x[index] / sums[index]
                center_mass_y[index] = weighted_y[index] / sums[index]
            diff_x = center_x - center_mass_x[index]
            diff_y = center_y - center_mass_y[index]
            mass_displacement[index] = np.sqrt(diff_x * diff_x + diff_y * diff_y)
        else:
            min_values[index] = 0.0
            max_values[index] = 0.0

        if edge_counts[index] > 0.0:
            edge_means[index] = edge_sums[index] / edge_counts[index]
            edge_variance = (
                edge_sumsq[index] / edge_counts[index]
                - edge_means[index] * edge_means[index]
            )
            if edge_variance < 0.0 and edge_variance > -1e-15:
                edge_variance = 0.0
            edge_stds[index] = np.sqrt(edge_variance)
        else:
            edge_min_values[index] = 0.0
            edge_max_values[index] = 0.0

    return (
        counts,
        sums,
        means,
        stds,
        min_values,
        max_values,
        edge_sums,
        edge_means,
        edge_stds,
        edge_min_values,
        edge_max_values,
        mass_displacement,
        center_mass_x,
        center_mass_y,
        max_x,
        max_y,
    )


@njit(cache=True)
def _is_inner_boundary_pixel(
    labels: np.ndarray,
    y: int,
    x: int,
    label: int,
) -> bool:
    height, width = labels.shape
    if y > 0 and labels[y - 1, x] != label:
        return True
    if y + 1 < height and labels[y + 1, x] != label:
        return True
    if x > 0 and labels[y, x - 1] != label:
        return True
    if x + 1 < width and labels[y, x + 1] != label:
        return True
    return False


@njit(cache=True)
def _object_intensity_quantiles_3d_numba(
    image: np.ndarray,
    labels: np.ndarray,
    label_to_index: np.ndarray,
    counts: np.ndarray,
    mad_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    object_count = counts.size
    lower = np.zeros(object_count, dtype=np.float64)
    median = np.zeros(object_count, dtype=np.float64)
    upper = np.zeros(object_count, dtype=np.float64)
    mad = np.zeros(object_count, dtype=np.float64)

    total_count = 0
    for index in range(object_count):
        total_count += int(counts[index])
    if total_count <= 0:
        return lower, median, upper, mad

    offsets = np.empty(object_count + 1, dtype=np.int64)
    offsets[0] = 0
    for index in range(object_count):
        offsets[index + 1] = offsets[index] + int(counts[index])

    write_offsets = offsets[:-1].copy()
    values = np.empty(total_count, dtype=np.float64)
    z_size, y_size, x_size = image.shape
    for z_index in range(z_size):
        for y_index in range(y_size):
            for x_index in range(x_size):
                label = int(labels[z_index, y_index, x_index])
                if label <= 0 or label >= label_to_index.size:
                    continue
                index = int(label_to_index[label])
                if index < 0:
                    continue
                value = float(image[z_index, y_index, x_index])
                if not np.isfinite(value):
                    continue
                offset = write_offsets[index]
                values[offset] = value
                write_offsets[index] = offset + 1

    for index in range(object_count):
        start = int(offsets[index])
        count = int(counts[index])
        if count <= 0:
            continue
        group = values[start:start + count].copy()
        lower[index] = _quantile_from_dense_group_partition(group, 0.25)
        median[index] = _quantile_from_dense_group_partition(group, 0.5)
        upper[index] = _quantile_from_dense_group_partition(group, 0.75)

    write_offsets = offsets[:-1].copy()
    deviations = np.empty(total_count, dtype=np.float64)
    for z_index in range(z_size):
        for y_index in range(y_size):
            for x_index in range(x_size):
                label = int(labels[z_index, y_index, x_index])
                if label <= 0 or label >= label_to_index.size:
                    continue
                index = int(label_to_index[label])
                if index < 0:
                    continue
                value = float(image[z_index, y_index, x_index])
                if not np.isfinite(value):
                    continue
                offset = write_offsets[index]
                deviations[offset] = abs(value - median[index])
                write_offsets[index] = offset + 1

    for index in range(object_count):
        start = int(offsets[index])
        count = int(counts[index])
        if count <= 0:
            continue
        group = deviations[start:start + count].copy()
        mad[index] = _quantile_from_dense_group_partition(group, mad_fraction)

    return lower, median, upper, mad


@njit(cache=True)
def _object_intensity_scan_3d_numba(
    image: np.ndarray,
    labels: np.ndarray,
    object_labels: np.ndarray,
    label_to_index: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    z_size, y_size, x_size = image.shape
    object_count = object_labels.size
    counts = np.zeros(object_count, dtype=np.float64)
    sums = np.zeros(object_count, dtype=np.float64)
    sumsq = np.zeros(object_count, dtype=np.float64)
    min_values = np.full(object_count, np.inf, dtype=np.float64)
    max_values = np.full(object_count, -np.inf, dtype=np.float64)
    sum_x = np.zeros(object_count, dtype=np.float64)
    sum_y = np.zeros(object_count, dtype=np.float64)
    sum_z = np.zeros(object_count, dtype=np.float64)
    weighted_x = np.zeros(object_count, dtype=np.float64)
    weighted_y = np.zeros(object_count, dtype=np.float64)
    weighted_z = np.zeros(object_count, dtype=np.float64)
    max_x = np.zeros(object_count, dtype=np.float64)
    max_y = np.zeros(object_count, dtype=np.float64)
    max_z = np.zeros(object_count, dtype=np.float64)

    edge_counts = np.zeros(object_count, dtype=np.float64)
    edge_sums = np.zeros(object_count, dtype=np.float64)
    edge_sumsq = np.zeros(object_count, dtype=np.float64)
    edge_min_values = np.full(object_count, np.inf, dtype=np.float64)
    edge_max_values = np.full(object_count, -np.inf, dtype=np.float64)

    for z_index in range(z_size):
        for y_index in range(y_size):
            for x_index in range(x_size):
                label = labels[z_index, y_index, x_index]
                if label <= 0 or label >= label_to_index.size:
                    continue
                index = label_to_index[label]
                if index < 0:
                    continue
                value = image[z_index, y_index, x_index]
                if not np.isfinite(value):
                    continue

                counts[index] += 1.0
                sums[index] += value
                sumsq[index] += value * value
                sum_x[index] += x_index
                sum_y[index] += y_index
                sum_z[index] += z_index
                weighted_x[index] += x_index * value
                weighted_y[index] += y_index * value
                weighted_z[index] += z_index * value
                if value < min_values[index]:
                    min_values[index] = value
                if value >= max_values[index]:
                    max_values[index] = value
                    max_x[index] = x_index
                    max_y[index] = y_index
                    max_z[index] = z_index

                if _is_inner_boundary_voxel(labels, z_index, y_index, x_index, label):
                    edge_counts[index] += 1.0
                    edge_sums[index] += value
                    edge_sumsq[index] += value * value
                    if value < edge_min_values[index]:
                        edge_min_values[index] = value
                    if value > edge_max_values[index]:
                        edge_max_values[index] = value

    means = np.zeros(object_count, dtype=np.float64)
    stds = np.zeros(object_count, dtype=np.float64)
    edge_means = np.zeros(object_count, dtype=np.float64)
    edge_stds = np.zeros(object_count, dtype=np.float64)
    mass_displacement = np.zeros(object_count, dtype=np.float64)
    center_mass_x = np.zeros(object_count, dtype=np.float64)
    center_mass_y = np.zeros(object_count, dtype=np.float64)
    center_mass_z = np.zeros(object_count, dtype=np.float64)
    for index in range(object_count):
        if counts[index] > 0.0:
            means[index] = sums[index] / counts[index]
            variance = sumsq[index] / counts[index] - means[index] * means[index]
            if variance < 0.0 and variance > -1e-15:
                variance = 0.0
            stds[index] = np.sqrt(variance)
            center_x = sum_x[index] / counts[index]
            center_y = sum_y[index] / counts[index]
            center_z = sum_z[index] / counts[index]
            if sums[index] != 0.0:
                center_mass_x[index] = weighted_x[index] / sums[index]
                center_mass_y[index] = weighted_y[index] / sums[index]
                center_mass_z[index] = weighted_z[index] / sums[index]
            diff_x = center_x - center_mass_x[index]
            diff_y = center_y - center_mass_y[index]
            diff_z = center_z - center_mass_z[index]
            mass_displacement[index] = np.sqrt(
                diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
            )
        else:
            min_values[index] = 0.0
            max_values[index] = 0.0

        if edge_counts[index] > 0.0:
            edge_means[index] = edge_sums[index] / edge_counts[index]
            edge_variance = (
                edge_sumsq[index] / edge_counts[index]
                - edge_means[index] * edge_means[index]
            )
            if edge_variance < 0.0 and edge_variance > -1e-15:
                edge_variance = 0.0
            edge_stds[index] = np.sqrt(edge_variance)
        else:
            edge_min_values[index] = 0.0
            edge_max_values[index] = 0.0

    return (
        counts,
        sums,
        means,
        stds,
        min_values,
        max_values,
        edge_sums,
        edge_means,
        edge_stds,
        edge_min_values,
        edge_max_values,
        mass_displacement,
        center_mass_x,
        center_mass_y,
        center_mass_z,
        max_x,
        max_y,
        max_z,
    )


@njit(cache=True)
def _is_inner_boundary_voxel(
    labels: np.ndarray,
    z_index: int,
    y_index: int,
    x_index: int,
    label: int,
) -> bool:
    z_size, y_size, x_size = labels.shape
    if z_index > 0 and labels[z_index - 1, y_index, x_index] != label:
        return True
    if z_index + 1 < z_size and labels[z_index + 1, y_index, x_index] != label:
        return True
    if y_index > 0 and labels[z_index, y_index - 1, x_index] != label:
        return True
    if y_index + 1 < y_size and labels[z_index, y_index + 1, x_index] != label:
        return True
    if x_index > 0 and labels[z_index, y_index, x_index - 1] != label:
        return True
    if x_index + 1 < x_size and labels[z_index, y_index, x_index + 1] != label:
        return True
    return False


__all__ = [
    "NumbaNumpyObjectIntensityBackendStrategy",
    "AutomaticHigh",
    "AutomaticLow",
    "ImageIntensityMeasurement",
    "ImageIntensityPercentileSpec",
    "ObjectIntensityMeasurement",
    "ObjectIntensityMeasurementRequest",
    "ObjectIntensityResults",
    "RescaleMethod",
    "ObjectIntensityArrays",
    "ObjectIntensityBackendStrategy",
    "ObjectIntensityLabelInput",
    "measure_image_intensity",
    "measure_image_intensity_masked",
    "measure_object_intensity",
    "measure_object_intensity_batch",
    "object_intensity_backend",
    "prepare_measure_object_intensity",
    "rescale_intensity",
    "rescale_intensity_match_maximum",
    "rescale_source_range",
]
