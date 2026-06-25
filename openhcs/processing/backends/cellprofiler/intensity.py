"""Intensity-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import time
from typing import ClassVar, TypeAlias

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
    measurement_image_batch_executor,
    object_label_measurement_execution,
    pure_2d_batch_executor,
)
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.runtime_semantics import (
    ConsecutiveObjectLabelIdProjection,
    MeasurementRowAxisField,
    MeasurementObjectRowIdentity,
    RuntimePlaneAxis,
    dense_object_label_measurement_row_domain,
)
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    DenseObjectLabelPlaneDomainStackRequest,
    DenseObjectLabelSliceStackRequest,
    ImageMetadataPayload,
    MaskedImagePayload,
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectLabelValue,
    image_payload_data,
    image_payload_metadata,
    object_label_dense_array,
    with_image_payload_data,
)
from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.core.runtime_invocation import RuntimeBatchInvocationRequest
from openhcs.core.measurement_image_alignment import ReplicatedChannelMonochromeProjection
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendAuthority,
    CellProfilerBackendProvider,
    CellProfilerBackendProviderSelection,
    CellProfilerBackendStrategyMixin,
)
from openhcs.processing.backends.cellprofiler.enum_attributes import (
    CellProfilerEnumAttributeMixin,
)
from openhcs.processing.backends.cellprofiler.intensity_object_quantiles_numba import (
    ObjectIntensityArrays,
    ObjectIntensityFeatureValues,
    _object_intensity_quantiles,
    _object_intensity_quantiles_3d_batch_numba,
    _object_intensity_quantiles_3d_numba,
    _object_intensity_scan_3d_batch_numba,
    _object_intensity_scan_3d_numba,
    _object_intensity_scan_numba,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


logger = logging.getLogger(__name__)
ImageIntensityOutput: TypeAlias = np.ndarray | ImageMetadataPayload | MaskedImagePayload
ObjectIntensityRuntimeRequest: TypeAlias = (
    RuntimePure2DSliceBatchRequest | RuntimeBatchInvocationRequest
)


class RescaleMethod(CellProfilerEnumAttributeMixin, Enum):
    """Closed CellProfiler rescale modes with intensity-scale policy metadata."""

    __cellprofiler_attribute_names__ = (
        "_can_preserve_unit_interval_scale",
        "_requires_unit_destination_range",
    )

    STRETCH = ("stretch", True, False)
    MANUAL_INPUT_RANGE = ("manual_input_range", True, False)
    MANUAL_IO_RANGE = ("manual_io_range", True, True)
    DIVIDE_BY_IMAGE_MINIMUM = ("divide_by_image_minimum", False, False)
    DIVIDE_BY_IMAGE_MAXIMUM = ("divide_by_image_maximum", False, False)
    DIVIDE_BY_VALUE = ("divide_by_value", False, False)

    def preserves_unit_interval_intensity_scale(
        self,
        *,
        source_range: tuple[float, float],
        destination_range: tuple[float, float],
    ) -> bool:
        """Return whether this mode preserves a proven unit-interval scale."""
        if not self._can_preserve_unit_interval_scale:
            return False
        source_low, source_high = source_range
        if not (np.isclose(source_low, 0.0) and np.isclose(source_high, 1.0)):
            return False
        if not self._requires_unit_destination_range:
            return True
        destination_low, destination_high = destination_range
        return np.isclose(destination_low, 0.0) and np.isclose(destination_high, 1.0)


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
class ObjectIntensityMeasurement(ObjectIntensityFeatureValues[float]):
    """Per-object CellProfiler-compatible intensity measurements."""

    slice_index: int
    object_label: int

    @classmethod
    def from_backend_arrays(
        cls,
        arrays: ObjectIntensityArrays,
        *,
        index: int,
        label: int,
        slice_index: int,
    ) -> "ObjectIntensityMeasurement":
        """Materialize one CellProfiler object-intensity row from backend arrays."""
        return cls(
            slice_index=slice_index,
            object_label=int(label),
            **arrays.scalar_kwargs(index),
        )

    @classmethod
    def rows_from_backend_arrays(
        cls,
        arrays: ObjectIntensityArrays,
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


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectIntensityMeasurementAxisContext:
    """Runtime row-axis context for one object-intensity measurement plane."""

    slice_index: int
    object_domain: tuple[int, ...] | None = None
    row_identity: MeasurementObjectRowIdentity | None = None

    def object_labels_for(self, measured_labels: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.object_domain
            if self.object_domain is not None
            else tuple(int(label) for label in measured_labels),
            dtype=np.int32,
        )

    def axis_columns_for(self, row_count: int) -> dict[str, np.ndarray]:
        columns: dict[str, np.ndarray] = {
            MeasurementRowAxisField.SLICE_INDEX.value: np.full(
                row_count,
                self.slice_index,
                dtype=np.int64,
            ),
            MeasurementRowAxisField.IMAGE_NUMBER.value: np.full(
                row_count,
                self.slice_index + 1,
                dtype=np.int64,
            ),
        }
        if self.row_identity is not None:
            columns[MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD] = np.full(
                row_count,
                self.row_identity.value,
                dtype=object,
            )
        return columns


@dataclass(frozen=True, slots=True)
class ObjectIntensityMeasurementRows(
    ObjectIntensityMeasurementAxisContext,
    ColumnarRows,
):
    """Columnar object-intensity measurements for runtime lookup paths."""

    arrays: ObjectIntensityArrays
    _columns: dict[str, np.ndarray] = field(repr=False, compare=False)

    @classmethod
    def from_arrays(
        cls,
        arrays: ObjectIntensityArrays,
        *,
        slice_index: int,
        object_domain: tuple[int, ...] | None = None,
        row_identity: MeasurementObjectRowIdentity | None = None,
    ) -> "ObjectIntensityMeasurementRows":
        axis_context = ObjectIntensityMeasurementAxisContext(
            slice_index=slice_index,
            object_domain=object_domain,
            row_identity=row_identity,
        )
        object_labels = axis_context.object_labels_for(arrays.object_labels)
        row_count = int(object_labels.size)
        measured_index_by_label = {
            int(label): index for index, label in enumerate(arrays.object_labels)
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
        def align_column(values: np.ndarray) -> np.ndarray:
            return _align_intensity_column(
                values,
                object_labels,
                measured_indexes,
                measured_mask,
            )

        columns: dict[str, np.ndarray] = {
            **axis_context.axis_columns_for(row_count),
            MeasurementRowAxisField.OBJECT_LABEL.value: object_labels,
            **arrays.aligned_feature_columns(align_column),
        }
        return cls(
            arrays=arrays,
            slice_index=slice_index,
            object_domain=object_domain,
            row_identity=row_identity,
            _columns=columns,
        )

    columns: ClassVar[AliasProperty[dict[str, np.ndarray]]] = AliasProperty(
        "_columns"
    )

    @property
    def covers_declared_object_measurement_domain(self) -> bool:
        """Rows are aligned to the declared/prepared object domain at creation."""
        return True

    def __len__(self) -> int:
        return int(self._columns[MeasurementRowAxisField.OBJECT_LABEL.value].size)

    def __iter__(self):
        for row_index in range(len(self)):
            yield self[row_index]

    def __getitem__(self, index: int) -> ObjectIntensityMeasurement:
        columns = self._columns
        return ObjectIntensityMeasurement(
            slice_index=int(columns[MeasurementRowAxisField.SLICE_INDEX.value][index]),
            object_label=int(columns[MeasurementRowAxisField.OBJECT_LABEL.value][index]),
            **ObjectIntensityMeasurement.scalar_kwargs_from_columns(columns, index),
        )


ObjectIntensityLabelInput = ObjectLabelValue | np.ndarray
OBJECT_INTENSITY_DEFAULT_SLICE_INDEX = 0
OBJECT_INTENSITY_PREPARED_LABELS_KWARG = "object_intensity_prepared_labels"


def _align_intensity_column(
    measured_values: np.ndarray,
    object_labels: np.ndarray,
    measured_indexes: np.ndarray,
    measured_mask: np.ndarray,
) -> np.ndarray:
    """Align measured object rows to a declared object-label domain."""
    values = np.asarray(measured_values)
    aligned = np.zeros(
        measured_indexes.size,
        dtype=np.result_type(values.dtype, np.float64),
    )
    if measured_mask.any():
        aligned[measured_mask] = values[measured_indexes[measured_mask]]
        measured_extent = int(np.max(np.asarray(object_labels)[measured_mask]))
    else:
        measured_extent = 0
    aligned[np.asarray(object_labels) > measured_extent] = np.nan
    return aligned


@dataclass(frozen=True, slots=True)
class ObjectIntensityResults:
    """Collection of intensity measurements for all objects."""

    slice_index: int
    object_count: int
    measurements: ObjectIntensityMeasurementRows


@dataclass(frozen=True, slots=True)
class ObjectIntensityPreparedLabels:
    """Prepared object-label domain reused across same-label intensity images."""

    source: ObjectIntensityLabelInput
    dense_labels: np.ndarray
    object_domain: tuple[int, ...]
    projection: ConsecutiveObjectLabelIdProjection
    relabeled_labels: np.ndarray
    label_to_index: np.ndarray

    @classmethod
    def from_source(
        cls,
        labels: ObjectIntensityLabelInput,
        dense_labels: np.ndarray,
    ) -> "ObjectIntensityPreparedLabels":
        label_array = np.ascontiguousarray(dense_labels, dtype=np.int32)
        projection = ConsecutiveObjectLabelIdProjection.from_dense_array(label_array)
        relabeled_labels = np.ascontiguousarray(
            projection.relabel_numpy_array(label_array, dtype=np.int32),
            dtype=np.int32,
        )
        label_to_index = cls.label_to_index_for_projection(projection)
        return cls(
            source=labels,
            dense_labels=label_array,
            object_domain=dense_object_label_measurement_row_domain(
                labels,
                label_array,
            ),
            projection=projection,
            relabeled_labels=relabeled_labels,
            label_to_index=label_to_index,
        )

    @classmethod
    def from_measurement(
        cls,
        *,
        image: object,
        labels: ObjectIntensityLabelInput,
        slice_index: int,
    ) -> "ObjectIntensityPreparedLabels":
        return cls.from_source(
            labels,
            object_intensity_dense_labels(
                image=image,
                labels=labels,
                slice_index=slice_index,
            ),
        )

    @staticmethod
    def label_to_index_for_projection(
        projection: ConsecutiveObjectLabelIdProjection,
    ) -> np.ndarray:
        label_to_index = np.full(projection.object_count + 1, -1, dtype=np.int64)
        if projection.has_objects:
            label_to_index[1:] = np.arange(projection.object_count, dtype=np.int64)
        return label_to_index

    def with_relabeled_labels(
        self,
        relabeled_labels: np.ndarray,
    ) -> "ObjectIntensityPreparedLabels":
        return type(self)(
            source=self.source,
            dense_labels=self.dense_labels,
            object_domain=self.object_domain,
            projection=self.projection,
            relabeled_labels=np.ascontiguousarray(relabeled_labels, dtype=np.int32),
            label_to_index=self.label_to_index,
        )

    @property
    def object_labels(self) -> np.ndarray:
        return self.projection.positive_label_ids.astype(np.int32, copy=False)

    @property
    def object_count(self) -> int:
        return self.projection.object_count


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectIntensityMeasurementContext(ObjectIntensityMeasurementAxisContext):
    """Nominal measurement context for object-intensity execution."""

    labels: ObjectIntensityLabelInput
    backend_provider: CellProfilerBackendProviderSelection
    prepared_labels: ObjectIntensityPreparedLabels | None = None

    @classmethod
    def from_function_arguments(
        cls,
        *,
        labels: ObjectIntensityLabelInput,
        backend_provider: BackendProviderInput,
        slice_index: int,
        object_domain: tuple[int, ...] | None = None,
        row_identity: MeasurementObjectRowIdentity | None = None,
        prepared_labels: ObjectIntensityPreparedLabels | None = None,
    ) -> "ObjectIntensityMeasurementContext":
        return cls(
            labels=labels,
            backend_provider=CellProfilerBackendAuthority.provider_selection(
                backend_provider,
            ),
            slice_index=int(slice_index),
            object_domain=object_domain,
            row_identity=row_identity,
            prepared_labels=prepared_labels,
        )

    @classmethod
    def from_runtime_request(
        cls,
        request: ObjectIntensityRuntimeRequest,
    ) -> "ObjectIntensityMeasurementContext":
        kwargs = request.kwargs
        if "labels" not in kwargs:
            raise ValueError("Object-intensity runtime kwargs are missing 'labels'.")
        labels = kwargs["labels"]
        if not isinstance(labels, ObjectLabelValue | np.ndarray):
            raise TypeError(
                "Object-intensity labels must be an ObjectLabelValue or ndarray; "
                f"got {type(labels).__name__}."
            )
        backend_provider = (
            kwargs["object_intensity_backend_provider"]
            if "object_intensity_backend_provider" in kwargs
            else DEFAULT_CELLPROFILER_BACKEND_SELECTION
        )
        prepared_labels = (
            kwargs[OBJECT_INTENSITY_PREPARED_LABELS_KWARG]
            if OBJECT_INTENSITY_PREPARED_LABELS_KWARG in kwargs
            else None
        )
        if prepared_labels is None or isinstance(
            prepared_labels,
            ObjectIntensityPreparedLabels,
        ):
            return cls(
                labels=labels,
                backend_provider=CellProfilerBackendAuthority.provider_selection(
                    backend_provider,
                ),
                slice_index=(
                    int(kwargs["slice_index"])
                    if "slice_index" in kwargs
                    else OBJECT_INTENSITY_DEFAULT_SLICE_INDEX
                ),
                prepared_labels=prepared_labels,
            )
        raise TypeError(
            "object_intensity_prepared_labels must be ObjectIntensityPreparedLabels "
            f"or None; got {type(prepared_labels).__name__}."
        )

    def batch_key_items(self) -> tuple[tuple[str, Hashable], ...]:
        return (
            (
                "object_intensity_backend_provider",
                self.backend_provider.semantic_identity(),
            ),
            ("slice_index", self.slice_index),
        )


def object_intensity_dense_labels(
    *,
    image: object,
    labels: ObjectIntensityLabelInput,
    slice_index: int,
) -> np.ndarray:
    """Return object labels in the measurement image execution domain."""
    label_array = object_label_dense_array(labels, dtype=np.int32)
    image_array = np.asarray(image)
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
        and 0 <= slice_index < label_array.shape[0]
    ):
        return np.asarray(label_array[slice_index], dtype=np.int32)
    return np.asarray(label_array, dtype=np.int32)


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectIntensityMeasurementRequest(ObjectIntensityMeasurementContext):
    """Executable request for one object-intensity image/label plane."""

    image: np.ndarray

    @property
    def measurement_image(self) -> np.ndarray:
        image = np.asarray(self.image)
        if image.ndim == 3 and not is_color_image_slice(image):
            return image
        return ReplicatedChannelMonochromeProjection().plane(image, name="image")

    @property
    def dense_labels(self) -> np.ndarray:
        if self.prepared_labels is not None:
            return self.prepared_labels.dense_labels
        return object_intensity_dense_labels(
            image=self.measurement_image,
            labels=self.labels,
            slice_index=self.slice_index,
        )

    @property
    def measurement_object_domain(self) -> tuple[int, ...]:
        """Return the object-intensity row domain without fabricating sparse IDs."""
        if self.object_domain is not None:
            return self.object_domain
        if self.prepared_labels is not None:
            return self.prepared_labels.object_domain
        return dense_object_label_measurement_row_domain(
            self.labels,
            self.dense_labels,
        )

    def measurements(self) -> ObjectIntensityMeasurementRows:
        """Measure this image/label plane through the selected backend."""
        prepared_labels = (
            self.prepared_labels
            if self.prepared_labels is not None
            else ObjectIntensityPreparedLabels.from_source(
                self.labels,
                self.dense_labels,
            )
        )
        intensity_arrays = object_intensity_backend(
            backend_provider=self.backend_provider,
        ).measure_prepared(
            self.measurement_image,
            prepared_labels,
        )
        return ObjectIntensityMeasurementRows.from_arrays(
            intensity_arrays,
            slice_index=self.slice_index,
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

    def measure_prepared(
        self,
        image: np.ndarray,
        labels: ObjectIntensityPreparedLabels,
    ) -> ObjectIntensityArrays:
        """Measure object intensity arrays with a prepared label domain."""
        return self.measure(image, labels.dense_labels)

    def measure_prepared_batch(
        self,
        images: tuple[np.ndarray, ...],
        labels: ObjectIntensityPreparedLabels,
    ) -> tuple[ObjectIntensityArrays, ...]:
        """Measure multiple images that share one prepared label domain."""
        return tuple(self.measure_prepared(image, labels) for image in images)


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
        image_array = np.ascontiguousarray(image)
        label_array = np.ascontiguousarray(labels, dtype=np.int32)
        if (
            image_array.ndim == 2
            and label_array.ndim == 3
            and label_array.shape[0] == 1
            and label_array.shape[1:] == image_array.shape
        ):
            label_array = np.ascontiguousarray(label_array[0], dtype=np.int32)
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
                    np.ascontiguousarray(broadcast_labels, dtype=np.int32),
                )
            raise NotImplementedError(
                "NumPy object-intensity backend supports 2-D and 3-D arrays; "
                f"got image shape {image_array.shape!r} and label shape {label_array.shape!r}."
            )
        if image_array.shape != label_array.shape:
            raise ValueError("image and labels must have matching shapes.")

        return self.measure_prepared(
            image_array,
            ObjectIntensityPreparedLabels.from_source(label_array, label_array),
        )

    def measure_prepared(
        self,
        image: np.ndarray,
        labels: ObjectIntensityPreparedLabels,
    ) -> ObjectIntensityArrays:
        image_array = np.ascontiguousarray(image)
        label_array = labels.dense_labels
        if image_array.ndim != 2 or label_array.ndim != 2:
            if image_array.ndim == 3 and label_array.ndim == 3:
                return self._measure_3d_prepared(image_array, labels)
            if image_array.ndim == 3 and label_array.ndim == 2:
                if image_array.shape[-2:] != label_array.shape:
                    raise ValueError("image and labels must have matching YX shapes.")
                broadcast_labels = np.broadcast_to(
                    labels.relabeled_labels,
                    image_array.shape,
                )
                return self._measure_3d_prepared(
                    image_array,
                    labels.with_relabeled_labels(broadcast_labels),
                )
            raise NotImplementedError(
                "NumPy object-intensity backend supports 2-D and 3-D arrays; "
                f"got image shape {image_array.shape!r} and label shape {label_array.shape!r}."
            )
        if image_array.shape != label_array.shape:
            raise ValueError("image and labels must have matching shapes.")
        object_labels = labels.object_labels
        object_count = labels.object_count
        if object_count == 0:
            return ObjectIntensityArrays.empty(object_labels)
        arrays = _object_intensity_scan_numba(
            image_array,
            labels.relabeled_labels,
            object_labels,
            labels.label_to_index,
        )
        lower, median, upper, mad = _object_intensity_quantiles(
            image_array,
            labels.relabeled_labels,
            object_labels,
            labels.label_to_index,
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

    def measure_prepared_batch(
        self,
        images: tuple[np.ndarray, ...],
        labels: ObjectIntensityPreparedLabels,
    ) -> tuple[ObjectIntensityArrays, ...]:
        """Measure a homogeneous 3-D image batch against one label domain."""
        if not images:
            return ()
        image_arrays = tuple(np.ascontiguousarray(image) for image in images)
        if not all(image.ndim == 3 for image in image_arrays):
            return super().measure_prepared_batch(images, labels)

        image_shape = image_arrays[0].shape
        if any(image.shape != image_shape for image in image_arrays):
            raise ValueError("Batched object-intensity images must share shape.")

        relabeled_labels = labels.relabeled_labels
        if relabeled_labels.ndim == 2:
            if image_shape[-2:] != relabeled_labels.shape:
                raise ValueError("image and labels must have matching YX shapes.")
            relabeled_labels = np.broadcast_to(relabeled_labels, image_shape)
        elif relabeled_labels.ndim != 3:
            return super().measure_prepared_batch(images, labels)

        if image_shape != relabeled_labels.shape:
            raise ValueError("image and labels must have matching shapes.")

        object_labels = labels.object_labels
        if labels.object_count == 0:
            return tuple(ObjectIntensityArrays.empty(object_labels) for _image in images)

        image_batch = np.ascontiguousarray(np.stack(image_arrays, axis=0))
        label_array = np.ascontiguousarray(relabeled_labels, dtype=np.int32)

        scan_started_at = time.perf_counter()
        scan_result = _object_intensity_scan_3d_batch_numba(
            image_batch,
            label_array,
            object_labels,
            labels.label_to_index,
        )
        RuntimeProfileLogger.log(
            logger,
            "object_intensity_scan_3d_batch",
            time.perf_counter() - scan_started_at,
            images=len(images),
            objects=labels.object_count,
            voxels=image_arrays[0].size,
        )
        quantile_started_at = time.perf_counter()
        quantile_result = _object_intensity_quantiles_3d_batch_numba(
            image_batch,
            label_array,
            labels.label_to_index,
            scan_result[0].astype(np.int64, copy=False),
            1.0 / 3.0,
        )
        RuntimeProfileLogger.log(
            logger,
            "object_intensity_quantiles_3d_batch",
            time.perf_counter() - quantile_started_at,
            images=len(images),
            objects=labels.object_count,
            voxels=image_arrays[0].size,
        )
        return tuple(
            ObjectIntensityArrays.from_3d_scan_batch_result(
                object_labels=object_labels,
                scan_result=scan_result,
                quantile_result=quantile_result,
                image_index=image_index,
            )
            for image_index in range(len(images))
        )

    def _measure_3d(
        self,
        image_array: np.ndarray,
        label_array: np.ndarray,
    ) -> ObjectIntensityArrays:
        if image_array.shape != label_array.shape:
            raise ValueError("image and labels must have matching shapes.")
        return self._measure_3d_prepared(
            image_array,
            ObjectIntensityPreparedLabels.from_source(label_array, label_array),
        )

    def _measure_3d_prepared(
        self,
        image_array: np.ndarray,
        labels: ObjectIntensityPreparedLabels,
    ) -> ObjectIntensityArrays:
        if image_array.shape != labels.relabeled_labels.shape:
            raise ValueError("image and labels must have matching shapes.")
        object_labels = labels.object_labels
        if labels.object_count == 0:
            return ObjectIntensityArrays.empty(object_labels)
        scan_started_at = time.perf_counter()
        arrays = _object_intensity_scan_3d_numba(
            np.ascontiguousarray(image_array),
            np.ascontiguousarray(labels.relabeled_labels),
            object_labels,
            labels.label_to_index,
        )
        RuntimeProfileLogger.log(
            logger,
            "object_intensity_scan_3d",
            time.perf_counter() - scan_started_at,
            objects=labels.object_count,
            voxels=image_array.size,
        )
        quantile_started_at = time.perf_counter()
        quantile_result = _object_intensity_quantiles_3d_numba(
            np.ascontiguousarray(image_array),
            np.ascontiguousarray(labels.relabeled_labels),
            labels.label_to_index,
            arrays[0].astype(np.int64, copy=False),
            1.0 / 3.0,
        )
        RuntimeProfileLogger.log(
            logger,
            "object_intensity_quantiles_3d",
            time.perf_counter() - quantile_started_at,
            objects=labels.object_count,
            voxels=image_array.size,
        )
        return ObjectIntensityArrays.from_3d_scan_result(
            object_labels=object_labels,
            scan_result=arrays,
            quantile_result=quantile_result,
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
        prepared_3d = ObjectIntensityPreparedLabels.from_source(labels_3d, labels_3d)
        self.measure_prepared_batch(
            (
                image_3d,
                np.ascontiguousarray(1.0 - image_3d),
            ),
            prepared_3d,
        )


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
) -> list[object]:
    context = ObjectIntensityMeasurementContext.from_runtime_request(request)
    plane_domain_stack = DenseObjectLabelPlaneDomainStackRequest(
        context.labels,
        dtype=np.int32,
    ).stack()
    if plane_domain_stack is not None and plane_domain_stack.plane_count == request.slice_count:
        return [
            (
                slice_2d,
                ObjectIntensityMeasurementRequest(
                    image=image_payload_data(slice_2d),
                    labels=plane_domain_stack.plane(slice_index),
                    backend_provider=context.backend_provider,
                    slice_index=slice_index,
                    object_domain=plane_domain_stack.object_id_domains[slice_index],
                    row_identity=plane_domain_stack.measurement_row_identity,
                ).measurements(),
            )
            for slice_index, slice_2d in enumerate(request.slices_2d)
        ]

    label_stack = DenseObjectLabelSliceStackRequest(
        context.labels,
        slice_count=request.slice_count,
        dtype=np.int32,
    ).stack()
    if label_stack is None:
        return [
            request.execute_one(slice_index)
            for slice_index in range(request.slice_count)
        ]

    results: list[object] = []
    for slice_index, slice_2d in enumerate(request.slices_2d):
        labels_for_slice = label_stack.slice(slice_index)
        measurements = ObjectIntensityMeasurementRequest(
            image=image_payload_data(slice_2d),
            labels=labels_for_slice,
            backend_provider=context.backend_provider,
            slice_index=slice_index,
        ).measurements()
        results.append((slice_2d, measurements))
    return results


def measure_object_intensity_measurement_image_batch(
    func: Callable[..., object],
    requests: tuple[RuntimeBatchInvocationRequest, ...],
    execute_request: Callable[
        [Callable[..., object], RuntimeBatchInvocationRequest],
        object,
    ],
) -> list[object]:
    """Batch intensity measurement images that share one object-label domain."""
    outputs: list[object | None] = [None] * len(requests)
    for group in _object_intensity_batch_groups(requests):
        if len(group) <= 1:
            index, request = group[0]
            outputs[index] = execute_request(func, request)
            continue
        first_request = group[0][1]
        prepared_labels = _object_intensity_prepared_labels_for_batch_group(group)
        if prepared_labels is None:
            for index, request in group:
                outputs[index] = execute_request(func, request)
            continue
        backend = object_intensity_backend(
            backend_provider=ObjectIntensityMeasurementContext.from_runtime_request(
                first_request,
            ).backend_provider,
        )
        images = tuple(
            np.asarray(image_payload_data(request.image))
            for _index, request in group
        )
        batch_started_at = time.perf_counter()
        measurement_batches = backend.measure_prepared_batch(images, prepared_labels)
        RuntimeProfileLogger.log(
            logger,
            "object_intensity_prepared_batch",
            time.perf_counter() - batch_started_at,
            images=len(images),
            objects=prepared_labels.object_count,
        )
        for measurement_arrays, (index, request) in zip(
            measurement_batches,
            group,
            strict=True,
        ):
            rows_started_at = time.perf_counter()
            rows = ObjectIntensityMeasurementRows.from_arrays(
                measurement_arrays,
                slice_index=ObjectIntensityMeasurementContext.from_runtime_request(
                    request,
                ).slice_index,
                object_domain=prepared_labels.object_domain,
            )
            RuntimeProfileLogger.log(
                logger,
                "object_intensity_rows_from_arrays",
                time.perf_counter() - rows_started_at,
                rows=len(rows),
            )
            outputs[index] = (request.image, rows)
    return [
        output
        for output in outputs
        if output is not None
    ]


def _object_intensity_prepared_labels_for_batch_group(
    group: tuple[tuple[int, RuntimeBatchInvocationRequest], ...],
) -> ObjectIntensityPreparedLabels | None:
    first_request = group[0][1]
    context = ObjectIntensityMeasurementContext.from_runtime_request(first_request)
    labels = context.labels
    if not isinstance(labels, ObjectLabelValue):
        return None
    if labels.plane_axis is RuntimePlaneAxis.SOURCE_BINDING:
        return None
    return ObjectIntensityPreparedLabels.from_measurement(
        image=first_request.image,
        labels=labels,
        slice_index=context.slice_index,
    )


def _object_intensity_batch_groups(
    requests: tuple[RuntimeBatchInvocationRequest, ...],
) -> tuple[tuple[tuple[int, RuntimeBatchInvocationRequest], ...], ...]:
    grouped_requests: dict[
        tuple[tuple[str, Hashable], ...],
        list[tuple[int, RuntimeBatchInvocationRequest]],
    ] = {}
    singleton_groups: list[tuple[tuple[int, RuntimeBatchInvocationRequest], ...]] = []
    for index, request in enumerate(requests):
        key = _object_intensity_batch_key(request)
        if key is None:
            singleton_groups.append(((index, request),))
            continue
        if key not in grouped_requests:
            grouped_requests[key] = []
        grouped_requests[key].append((index, request))
    return (
        *(tuple(group) for group in grouped_requests.values()),
        *singleton_groups,
    )


def _object_intensity_batch_key(
    request: RuntimeBatchInvocationRequest,
) -> tuple[tuple[str, Hashable], ...] | None:
    semantic_group_key = request.semantic_group_key
    if semantic_group_key is None:
        return None
    context = ObjectIntensityMeasurementContext.from_runtime_request(request)
    return (
        ("semantic_group_key", semantic_group_key),
        *context.batch_key_items(),
    )


@numpy_decorator
@object_label_measurement_execution(ObjectLabelMeasurementExecution.FULL_STACK)
def measure_object_intensity(
    image: np.ndarray,
    labels: ObjectIntensityLabelInput,
    object_intensity_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    slice_index: int = OBJECT_INTENSITY_DEFAULT_SLICE_INDEX,
    object_intensity_prepared_labels: ObjectIntensityPreparedLabels | None = None,
) -> tuple[np.ndarray, ObjectIntensityMeasurementRows]:
    """Measure CellProfiler intensity features for identified objects."""
    context = ObjectIntensityMeasurementContext.from_function_arguments(
        labels=labels,
        backend_provider=object_intensity_backend_provider,
        slice_index=slice_index,
        prepared_labels=object_intensity_prepared_labels,
    )
    if context.prepared_labels is not None:
        return image, ObjectIntensityMeasurementRequest(
            image=image,
            labels=context.labels,
            backend_provider=context.backend_provider,
            slice_index=context.slice_index,
            object_domain=context.object_domain,
            row_identity=context.row_identity,
            prepared_labels=context.prepared_labels,
        ).measurements()

    label_array = object_label_dense_array(context.labels, dtype=np.int32)
    measurement_labels = context.labels
    measurement_label_array = label_array
    if np.asarray(image).ndim == 2 and context.slice_index == 0:
        plane_domain_stack = DenseObjectLabelPlaneDomainStackRequest(
            context.labels,
            dtype=np.int32,
        ).stack()
        if plane_domain_stack is not None:
            rows = tuple(
                ObjectIntensityMeasurementRequest(
                    image=image,
                    labels=plane_domain_stack.plane(plane_index),
                    backend_provider=context.backend_provider,
                    slice_index=plane_index,
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
        label_stack = DenseObjectLabelSliceStackRequest(
            context.labels,
            slice_count=int(label_array.shape[0]),
            dtype=np.int32,
        ).stack()
        if label_stack is not None:
            projected_index = (
                context.slice_index
                if context.slice_index < label_stack.labels.shape[0]
                else 0
                if label_stack.labels.shape[0] == 1
                else None
            )
            if projected_index is not None:
                measurement_labels = label_stack.slice(projected_index)
                measurement_label_array = object_label_dense_array(
                    measurement_labels,
                    dtype=np.int32,
                )
    return image, ObjectIntensityMeasurementRequest(
        image=image,
        labels=measurement_label_array,
        backend_provider=context.backend_provider,
        slice_index=context.slice_index,
        prepared_labels=context.prepared_labels,
        object_domain=dense_object_label_measurement_row_domain(
            measurement_labels,
            measurement_label_array,
        ),
    ).measurements()


def prepare_measure_object_intensity() -> None:
    """Prepare object-intensity backend kernels before measured execution."""
    ObjectIntensityBackendStrategy.prepare_registered_family()


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


@dataclass(frozen=True, slots=True)
class RescaleIntensityContext:
    """Normalized settings and image data for one intensity rescale operation."""

    data: np.ndarray
    automatic_low: AutomaticLow
    automatic_high: AutomaticHigh
    source_low: float
    source_high: float
    dest_low: float
    dest_high: float
    divisor_value: float

    @classmethod
    def from_settings(
        cls,
        image: np.ndarray,
        *,
        automatic_low: AutomaticLow,
        automatic_high: AutomaticHigh,
        source_low: float,
        source_high: float,
        dest_low: float,
        dest_high: float,
        divisor_value: float,
    ) -> "RescaleIntensityContext":
        source_data = np.asarray(image_payload_data(image))
        return cls(
            data=source_data.astype(np.float32, copy=False),
            automatic_low=coerce_cellprofiler_enum(AutomaticLow, automatic_low),
            automatic_high=coerce_cellprofiler_enum(AutomaticHigh, automatic_high),
            source_low=source_low,
            source_high=source_high,
            dest_low=dest_low,
            dest_high=dest_high,
            divisor_value=divisor_value,
        )

    @property
    def source_range(self) -> tuple[float, float]:
        return rescale_source_range(
            self.data,
            self.automatic_low,
            self.automatic_high,
            self.source_low,
            self.source_high,
        )

    def preserves_unit_interval_intensity_scale(
        self,
        rescale_method: RescaleMethod,
    ) -> bool:
        """Return whether declared rescale settings are a unit-interval identity."""
        return rescale_method.preserves_unit_interval_intensity_scale(
            source_range=self.source_range,
            destination_range=(self.dest_low, self.dest_high),
        )

    def linearly_rescaled(
        self,
        source_range: tuple[float, float],
        destination_range: tuple[float, float],
    ) -> np.ndarray:
        """Rescale with CellProfiler/skimage tuple-range clipping semantics."""
        source_low, source_high = source_range
        destination_low, destination_high = destination_range
        result = np.empty_like(self.data, dtype=np.float32)
        np.clip(self.data, source_low, source_high, out=result)
        if source_low == source_high:
            np.clip(result, destination_low, destination_high, out=result)
            return result
        result -= source_low
        result /= source_high - source_low
        result *= destination_high - destination_low
        result += destination_low
        return result

    def divided_by(self, divisor: float) -> np.ndarray:
        """Return image data divided by a scalar as a float32 result."""
        result = np.empty_like(self.data, dtype=np.float32)
        np.divide(self.data, divisor, out=result)
        return result


class RescaleMethodRunner(ABC, metaclass=AutoRegisterMeta):
    """Registered implementation for one CellProfiler rescale method."""

    __registry_key__ = "rescale_method"
    __skip_if_no_key__ = True
    rescale_method: ClassVar[RescaleMethod | None] = None

    @classmethod
    def for_method(cls, method: RescaleMethod) -> "RescaleMethodRunner":
        return cls.__registry__[method]()

    @abstractmethod
    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        """Return float32 rescaled image data."""


class StretchRescaleMethodRunner(RescaleMethodRunner):
    """Stretch image intensities to the unit interval."""

    rescale_method = RescaleMethod.STRETCH

    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        in_min = np.min(context.data)
        in_max = np.max(context.data)
        if in_min == in_max:
            return np.zeros_like(context.data)
        return context.linearly_rescaled((float(in_min), float(in_max)), (0.0, 1.0))


class ManualInputRangeRescaleMethodRunner(RescaleMethodRunner):
    """Rescale from a declared input range to the unit interval."""

    rescale_method = RescaleMethod.MANUAL_INPUT_RANGE

    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        return context.linearly_rescaled(context.source_range, (0.0, 1.0))


class ManualIoRangeRescaleMethodRunner(RescaleMethodRunner):
    """Rescale from a declared input range to a declared output range."""

    rescale_method = RescaleMethod.MANUAL_IO_RANGE

    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        return context.linearly_rescaled(
            context.source_range,
            (context.dest_low, context.dest_high),
        )


class DivideByImageMinimumRescaleMethodRunner(RescaleMethodRunner):
    """Divide image intensities by the image minimum."""

    rescale_method = RescaleMethod.DIVIDE_BY_IMAGE_MINIMUM

    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        src_min = np.min(context.data)
        if src_min == 0.0:
            raise ZeroDivisionError("Cannot divide pixel intensity by 0.")
        return context.divided_by(float(src_min))


class DivideByImageMaximumRescaleMethodRunner(RescaleMethodRunner):
    """Divide image intensities by the image maximum."""

    rescale_method = RescaleMethod.DIVIDE_BY_IMAGE_MAXIMUM

    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        src_max = np.max(context.data)
        if src_max == 0.0:
            src_max = 1.0
        return context.divided_by(float(src_max))


class DivideByValueRescaleMethodRunner(RescaleMethodRunner):
    """Divide image intensities by a declared scalar."""

    rescale_method = RescaleMethod.DIVIDE_BY_VALUE

    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        if context.divisor_value == 0.0:
            raise ZeroDivisionError("Cannot divide pixel intensity by 0.")
        return context.divided_by(context.divisor_value)


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
) -> ImageIntensityOutput:
    """Rescale CellProfiler image intensity using its declared range policy."""
    rescale_method = coerce_cellprofiler_enum(RescaleMethod, rescale_method)
    context = RescaleIntensityContext.from_settings(
        image,
        automatic_low=automatic_low,
        automatic_high=automatic_high,
        source_low=source_low,
        source_high=source_high,
        dest_low=dest_low,
        dest_high=dest_high,
        divisor_value=divisor_value,
    )
    rescaled = RescaleMethodRunner.for_method(rescale_method).run(
        context
    )
    metadata = image_payload_metadata(image)
    if not context.preserves_unit_interval_intensity_scale(rescale_method):
        metadata = metadata.without_unit_interval_intensity_scale()
    return with_image_payload_data(image, rescaled, metadata=metadata)


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
measurement_image_batch_executor(measure_object_intensity_measurement_image_batch)(
    measure_object_intensity
)
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
    ObjectIntensityPreparedLabels,
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
