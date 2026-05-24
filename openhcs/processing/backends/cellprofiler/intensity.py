"""Intensity-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from nominal_refactor_advisor.descriptor_algebra import AliasProperty

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
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.runtime_semantics import (
    ConsecutiveObjectLabelIdProjection,
    MeasurementObjectRowIdentity,
    ObjectLabelDomainScope,
    dense_object_label_id_domain,
    dense_object_label_extent_id_domain,
)
from openhcs.core.runtime_artifact_queries import ConcatenatedColumnarRows
from openhcs.core.runtime_artifact_queries import MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD
from openhcs.core.runtime_values import (
    ColumnarRows,
    DenseObjectLabelPlaneDomainStack,
    DenseObjectLabelSliceStack,
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectLabelValue,
    image_payload_data,
    object_label_dense_array,
)
from openhcs.core.measurement_image_alignment import ReplicatedChannelMonochromeProjection
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.cellprofiler.intensity_object_quantiles_numba import (
    ObjectIntensityArrays,
    _empty_intensity_arrays,
    _object_intensity_nd_scipy,
    _object_intensity_quantiles,
    _object_intensity_quantiles_3d_numba,
    _object_intensity_scan_3d_numba,
    _object_intensity_scan_numba,
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class ObjectIntensityMeasurementRows(ColumnarRows):
    """Columnar object-intensity measurements for runtime lookup paths."""

    arrays: ObjectIntensityArrays
    slice_index: int
    object_domain: tuple[int, ...] | None = None
    row_identity: MeasurementObjectRowIdentity | None = None
    _columns: dict[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object_labels = np.asarray(
            self.object_domain
            if self.object_domain is not None
            else tuple(int(label) for label in self.arrays.object_labels),
            dtype=np.int32,
        )
        row_count = int(object_labels.size)
        measured_index_by_label = {
            int(label): index for index, label in enumerate(self.arrays.object_labels)
        }
        measured_indexes = np.asarray(
            [
                measured_index_by_label[label]
                if label in measured_index_by_label
                else -1
                for label in object_labels
            ],
            dtype=np.int64,
        )
        measured_mask = measured_indexes >= 0
        columns = {
            "slice_index": np.full(row_count, self.slice_index, dtype=np.int64),
            "image_number": np.full(row_count, self.slice_index + 1, dtype=np.int64),
            "object_label": object_labels,
            "integrated_intensity": _align_intensity_column(
                self.arrays.integrated_intensity, measured_indexes, measured_mask
            ),
            "mean_intensity": _align_intensity_column(
                self.arrays.mean_intensity, measured_indexes, measured_mask
            ),
            "std_intensity": _align_intensity_column(
                self.arrays.std_intensity, measured_indexes, measured_mask
            ),
            "min_intensity": _align_intensity_column(
                self.arrays.min_intensity, measured_indexes, measured_mask
            ),
            "max_intensity": _align_intensity_column(
                self.arrays.max_intensity, measured_indexes, measured_mask
            ),
            "integrated_intensity_edge": _align_intensity_column(
                self.arrays.integrated_intensity_edge,
                measured_indexes,
                measured_mask,
            ),
            "mean_intensity_edge": _align_intensity_column(
                self.arrays.mean_intensity_edge, measured_indexes, measured_mask
            ),
            "std_intensity_edge": _align_intensity_column(
                self.arrays.std_intensity_edge, measured_indexes, measured_mask
            ),
            "min_intensity_edge": _align_intensity_column(
                self.arrays.min_intensity_edge, measured_indexes, measured_mask
            ),
            "max_intensity_edge": _align_intensity_column(
                self.arrays.max_intensity_edge, measured_indexes, measured_mask
            ),
            "mass_displacement": _align_intensity_column(
                self.arrays.mass_displacement, measured_indexes, measured_mask
            ),
            "lower_quartile_intensity": _align_intensity_column(
                self.arrays.lower_quartile_intensity,
                measured_indexes,
                measured_mask,
            ),
            "median_intensity": _align_intensity_column(
                self.arrays.median_intensity, measured_indexes, measured_mask
            ),
            "mad_intensity": _align_intensity_column(
                self.arrays.mad_intensity, measured_indexes, measured_mask
            ),
            "upper_quartile_intensity": _align_intensity_column(
                self.arrays.upper_quartile_intensity,
                measured_indexes,
                measured_mask,
            ),
            "center_mass_intensity_x": _align_intensity_column(
                self.arrays.center_mass_intensity_x,
                measured_indexes,
                measured_mask,
            ),
            "center_mass_intensity_y": _align_intensity_column(
                self.arrays.center_mass_intensity_y,
                measured_indexes,
                measured_mask,
            ),
            "center_mass_intensity_z": _align_intensity_column(
                self.arrays.center_mass_intensity_z,
                measured_indexes,
                measured_mask,
            ),
            "max_intensity_x": _align_intensity_column(
                self.arrays.max_intensity_x, measured_indexes, measured_mask
            ),
            "max_intensity_y": _align_intensity_column(
                self.arrays.max_intensity_y, measured_indexes, measured_mask
            ),
            "max_intensity_z": _align_intensity_column(
                self.arrays.max_intensity_z, measured_indexes, measured_mask
            ),
        }
        if self.row_identity is not None:
            columns[MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD] = np.full(
                row_count,
                self.row_identity.value,
                dtype=object,
            )
        object.__setattr__(self, "_columns", columns)

    columns: ClassVar[AliasProperty[dict[str, Any]]] = AliasProperty("_columns")

    def __len__(self) -> int:
        return int(self._columns["object_label"].size)

    def __iter__(self):
        for row_index in range(len(self)):
            yield self[row_index]

    def __getitem__(self, index: int) -> ObjectIntensityMeasurement:
        columns = self._columns
        return ObjectIntensityMeasurement(
            slice_index=int(columns["slice_index"][index]),
            object_label=int(columns["object_label"][index]),
            integrated_intensity=float(columns["integrated_intensity"][index]),
            mean_intensity=float(columns["mean_intensity"][index]),
            std_intensity=float(columns["std_intensity"][index]),
            min_intensity=float(columns["min_intensity"][index]),
            max_intensity=float(columns["max_intensity"][index]),
            integrated_intensity_edge=float(
                columns["integrated_intensity_edge"][index]
            ),
            mean_intensity_edge=float(columns["mean_intensity_edge"][index]),
            std_intensity_edge=float(columns["std_intensity_edge"][index]),
            min_intensity_edge=float(columns["min_intensity_edge"][index]),
            max_intensity_edge=float(columns["max_intensity_edge"][index]),
            mass_displacement=float(columns["mass_displacement"][index]),
            lower_quartile_intensity=float(
                columns["lower_quartile_intensity"][index]
            ),
            median_intensity=float(columns["median_intensity"][index]),
            mad_intensity=float(columns["mad_intensity"][index]),
            upper_quartile_intensity=float(
                columns["upper_quartile_intensity"][index]
            ),
            center_mass_intensity_x=float(
                columns["center_mass_intensity_x"][index]
            ),
            center_mass_intensity_y=float(
                columns["center_mass_intensity_y"][index]
            ),
            center_mass_intensity_z=float(
                columns["center_mass_intensity_z"][index]
            ),
            max_intensity_x=float(columns["max_intensity_x"][index]),
            max_intensity_y=float(columns["max_intensity_y"][index]),
            max_intensity_z=float(columns["max_intensity_z"][index]),
        )


ObjectIntensityLabelInput = ObjectLabelValue


def _align_intensity_column(
    measured_values: np.ndarray,
    measured_indexes: np.ndarray,
    measured_mask: np.ndarray,
) -> np.ndarray:
    """Align measured object rows to a declared object-label domain."""
    values = np.asarray(measured_values)
    aligned = np.zeros(measured_indexes.size, dtype=values.dtype)
    if measured_mask.any():
        aligned[measured_mask] = values[measured_indexes[measured_mask]]
    return aligned


@dataclass(frozen=True, slots=True)
class ObjectIntensityResults:
    """Collection of intensity measurements for all objects."""

    slice_index: int
    object_count: int
    measurements: ObjectIntensityMeasurementRows


@dataclass(frozen=True, slots=True)
class ObjectIntensityMeasurementRequest:
    """Executable request for one object-intensity image/label plane."""

    image: np.ndarray
    labels: ObjectIntensityLabelInput
    slice_index: int
    backend_provider: CellProfilerBackendProvider | None
    object_domain: tuple[int, ...] | None = None
    row_identity: MeasurementObjectRowIdentity | None = None

    @property
    def measurement_image(self) -> np.ndarray:
        image = np.asarray(self.image)
        if image.ndim == 3 and not is_color_image_slice(image):
            return image
        return ReplicatedChannelMonochromeProjection().plane(image, name="image")

    @property
    def dense_labels(self) -> np.ndarray:
        label_array = object_label_dense_array(self.labels, dtype=np.int32)
        image_array = np.asarray(self.measurement_image)
        if (
            image_array.ndim == 3
            and label_array.ndim == 4
            and label_array.shape[0] == image_array.shape[0]
            and label_array.shape[1] == image_array.shape[0]
            and label_array.shape[-2:] == image_array.shape[-2:]
        ):
            return np.ascontiguousarray(
                np.stack(
                    tuple(label_array[index, index] for index in range(image_array.shape[0])),
                    axis=0,
                ),
                dtype=np.int32,
            )
        if (
            image_array.ndim == 2
            and label_array.ndim == 3
            and label_array.shape[-2:] == image_array.shape
            and 0 <= self.slice_index < label_array.shape[0]
        ):
            return np.asarray(label_array[self.slice_index], dtype=np.int32)
        return label_array

    @property
    def measurement_object_domain(self) -> tuple[int, ...]:
        """Return the object-intensity row domain without fabricating sparse IDs."""
        if self.object_domain is not None:
            return self.object_domain
        if getattr(self.labels, "declared_object_count", None) is not None:
            return dense_object_label_extent_id_domain(self.labels)
        declared_ids = tuple(int(label) for label in getattr(self.labels, "declared_object_ids", ()) or ())
        if declared_ids:
            if len(declared_ids) == 1:
                return declared_ids
            max_label = int(self.dense_labels.max()) if self.dense_labels.size else 0
            max_declared = max(declared_ids)
            if (
                max_declared > len(declared_ids)
                and max_declared >= max_label
                and max_declared <= max(len(declared_ids) * 4, len(declared_ids) + 16)
            ):
                return tuple(range(1, max_declared + 1))
            if max_label > len(declared_ids) and max_label <= len(declared_ids) + 16:
                return tuple(range(1, max_label + 1))
        return dense_object_label_id_domain(self.labels)

    def measurements(self) -> ObjectIntensityMeasurementRows:
        """Measure this image/label plane through the selected backend."""
        intensity_arrays = object_intensity_backend(
            backend_provider=self.backend_provider,
        ).measure(
            self.measurement_image,
            self.dense_labels,
        )
        return ObjectIntensityMeasurementRows(
            intensity_arrays,
            self.slice_index,
            object_domain=self.measurement_object_domain,
            row_identity=self.row_identity,
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

    backend_key = CellProfilerBackendAuthority.backend_key(
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
        if (
            image_array.ndim == 2
            and label_array.ndim == 3
            and label_array.shape[0] == 1
            and label_array.shape[1:] == image_array.shape
        ):
            label_array = np.ascontiguousarray(label_array[0], dtype=np.int64)
        if image_array.ndim != 2 or label_array.ndim != 2:
            if image_array.ndim == 3 and label_array.ndim == 3:
                return self._measure_3d(image_array, label_array)
            if image_array.ndim == 3 and label_array.ndim == 2:
                if image_array.shape[-2:] != label_array.shape:
                    raise ValueError("image and labels must have matching YX shapes.")
                broadcast_labels = np.broadcast_to(
                    label_array,
                    image_array.shape,
                )
                return self._measure_3d(
                    image_array,
                    np.ascontiguousarray(broadcast_labels, dtype=np.int64),
                )
            raise NotImplementedError(
                "NumPy object-intensity backend supports 2-D and 3-D arrays; "
                f"got image shape {image_array.shape!r} and label shape {label_array.shape!r}."
            )
        if image_array.shape != label_array.shape:
            raise ValueError("image and labels must have matching shapes.")

        label_projection = ConsecutiveObjectLabelIdProjection.from_dense_array(
            label_array
        )
        object_labels = label_projection.positive_label_ids
        object_count = int(object_labels.size)
        if object_count == 0:
            return _empty_intensity_arrays(object_labels)

        label_array = np.ascontiguousarray(
            label_projection.relabel_numpy_array(label_array, dtype=np.int64)
        )
        label_to_index = np.full(object_count + 1, -1, dtype=np.int64)
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
        label_projection = ConsecutiveObjectLabelIdProjection.from_dense_array(
            label_array
        )
        object_labels = label_projection.positive_label_ids
        object_count = int(object_labels.size)
        if object_count == 0:
            return _empty_intensity_arrays(object_labels)
        label_array = np.ascontiguousarray(
            label_projection.relabel_numpy_array(label_array, dtype=np.int64)
        )
        label_to_index = np.full(object_count + 1, -1, dtype=np.int64)
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
    plane_domain_stack = DenseObjectLabelPlaneDomainStack.from_payload(
        kwargs["labels"],
        dtype=np.int32,
    )
    if plane_domain_stack is not None and plane_domain_stack.plane_count == request.slice_count:
        backend_provider = kwargs.get("object_intensity_backend_provider")
        return [
            (
                slice_2d,
                ObjectIntensityMeasurementRequest(
                    image=image_payload_data(slice_2d),
                    labels=plane_domain_stack.plane(slice_index),
                    slice_index=slice_index,
                    backend_provider=backend_provider,
                    object_domain=plane_domain_stack.object_id_domains[slice_index],
                    row_identity=plane_domain_stack.measurement_row_identity,
                ).measurements(),
            )
            for slice_index, slice_2d in enumerate(request.slices_2d)
        ]

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
        labels_for_slice = label_stack.slice(slice_index)
        measurements = ObjectIntensityMeasurementRequest(
            image=image_payload_data(slice_2d),
            labels=labels_for_slice,
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
    slice_index: int = 0,
) -> tuple[np.ndarray, ObjectIntensityMeasurementRows]:
    """Measure CellProfiler intensity features for identified objects."""
    label_array = object_label_dense_array(labels, dtype=np.int32)
    if np.asarray(image).ndim == 2 and slice_index == 0:
        plane_domain_stack = DenseObjectLabelPlaneDomainStack.from_payload(
            labels,
            dtype=np.int32,
        )
        if plane_domain_stack is not None:
            rows = tuple(
                ObjectIntensityMeasurementRequest(
                    image=image,
                    labels=plane_domain_stack.plane(plane_index),
                slice_index=plane_index,
                backend_provider=object_intensity_backend_provider,
                object_domain=plane_domain_stack.object_id_domains[plane_index],
                row_identity=plane_domain_stack.measurement_row_identity,
            ).measurements()
                for plane_index in range(plane_domain_stack.plane_count)
            )
            return image, ConcatenatedColumnarRows(rows)
    if (
        np.asarray(image).ndim == 2
        and label_array.ndim == 3
        and label_array.shape[-2:] == np.asarray(image).shape
    ):
        label_stack = DenseObjectLabelSliceStack.from_payload(
            labels,
            slice_count=int(label_array.shape[0]),
            dtype=np.int32,
        )
        if label_stack is not None:
            projected_index = (
                slice_index
                if slice_index < label_stack.labels.shape[0]
                else 0
                if label_stack.labels.shape[0] == 1
                else None
            )
            if projected_index is not None:
                labels = label_stack.slice(projected_index)
    return image, ObjectIntensityMeasurementRequest(
        image=image,
        labels=labels,
        slice_index=slice_index,
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


__all__ = public_names_from_objects(
    NumbaNumpyObjectIntensityBackendStrategy,
    AutomaticHigh,
    AutomaticLow,
    ImageIntensityMeasurement,
    ImageIntensityPercentileSpec,
    ObjectIntensityMeasurement,
    ObjectIntensityMeasurementRows,
    ObjectIntensityMeasurementRequest,
    ObjectIntensityResults,
    RescaleMethod,
    "ObjectIntensityArrays",
    ObjectIntensityBackendStrategy,
    "ObjectIntensityLabelInput",
    measure_image_intensity,
    measure_image_intensity_masked,
    measure_object_intensity,
    measure_object_intensity_batch,
    object_intensity_backend,
    prepare_measure_object_intensity,
    rescale_intensity,
    rescale_intensity_match_maximum,
    rescale_source_range,
)
