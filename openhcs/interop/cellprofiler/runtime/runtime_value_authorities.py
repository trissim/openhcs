"""Runtime value authority helpers for the CellProfiler adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.memory import detect_memory_type, stack_slices
from openhcs.core.runtime_invocation import RuntimeSliceAlignedValues
from openhcs.core.runtime_semantics import RuntimePlaneAxis
from openhcs.core.runtime_stores import StoredRuntimeValue
from openhcs.core.runtime_values import (
    ImagePayloadMetadataCompositionRequest,
    ObjectLabelPure2DSliceAggregator,
    ObjectLabelSet,
    RuntimeImagePayloadContext,
    SparseIJVLabelRows,
    SpatialGrid,
    image_payload_data,
    image_payload_mask,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    DenseLabelPayload,
    ImagePayloadMaskValue,
    ImagePayloadValue,
)
from openhcs.interop.cellprofiler.runtime.source_identity import (
    CellProfilerParsedMetadataValue,
)

SpatialGridMappingValue = (
    CellProfilerParsedMetadataValue
    | tuple[int, ...]
    | tuple[float, ...]
)
SpatialGridInputValue = SpatialGrid | Mapping[str, SpatialGridMappingValue]
SpatialGridInput = SpatialGridInputValue | RuntimeSliceAlignedValues[SpatialGridInputValue]
SpatialGridGroupValues = tuple[
    SpatialGrid | RuntimeSliceAlignedValues[SpatialGrid],
    ...,
]
SpatialGridEquivalenceValue = (
    str
    | int
    | float
    | tuple[int, int]
    | tuple[float, ...]
    | tuple[tuple[int, int], ...]
)

_MAX_DENSE_LABEL_STACK_BYTES = 1 << 30


@dataclass(frozen=True, slots=True)
class DenseLabelShapeSet:
    """Shape-set authority for dense label memory preflight."""

    arrays: tuple[np.ndarray, ...]

    @property
    def is_uniform(self) -> bool:
        if not self.arrays:
            return False
        first_shape = tuple(self.arrays[0].shape)
        return all(tuple(array.shape) == first_shape for array in self.arrays[1:])


@dataclass(frozen=True, slots=True)
class DenseLabelStackRepeatPattern:
    """Classify dense label stacks that repeat their first plane."""

    label_array: np.ndarray

    @property
    def repeats_first_plane(self) -> bool:
        if self.label_array.ndim <= 2:
            return False
        first_plane = self.label_array[0]
        return all(
            np.array_equal(first_plane, self.label_array[index])
            for index in range(1, self.label_array.shape[0])
        )


@dataclass(frozen=True, slots=True)
class MatlabPayloadEntryName:
    """Nominal classifier for MATLAB metadata entries."""

    value: str

    @property
    def is_private_metadata(self) -> bool:
        return self.value[:2] == "__"


@dataclass(frozen=True, slots=True)
class SpatialGridSliceCount:
    """Slice-count authority for grouped spatial-grid values."""

    grid: SpatialGrid | RuntimeSliceAlignedValues[SpatialGrid]

    @property
    def value(self) -> int:
        if isinstance(self.grid, RuntimeSliceAlignedValues):
            return self.grid.slice_count
        return 1


class RuntimeRecordStackAuthority:
    """Stack grouped runtime image/object-label records with payload semantics."""

    @classmethod
    def stack_image_records(cls, records: tuple[StoredRuntimeValue, ...]) -> ImagePayloadValue:
        payloads = tuple(record.value.data for record in records)
        arrays = tuple(
            cls.grouped_image_array(image_payload_data(payload))
            for payload in payloads
        )
        data = cls.stack_grouped_planes(arrays)
        masks = tuple(image_payload_mask(payload) for payload in payloads)
        mask = cls.stack_grouped_masks(masks)
        return cast(
            ImagePayloadValue,
            RuntimeImagePayloadContext(
                cast(ImagePayloadValue, data),
                cast(ImagePayloadMaskValue, mask),
                ImagePayloadMetadataCompositionRequest(payloads).metadata(),
            ).payload(),
        )

    @classmethod
    def stack_grouped_planes(cls, values: tuple[ImagePayloadValue, ...]) -> ImagePayloadValue:
        """Stack grouped values as image planes when they are homogeneous slices."""
        if not values:
            raise ValueError("Cannot stack an empty grouped runtime value set.")
        memory_type = detect_memory_type(values[0])
        if cls.all_values_are_2d_arrays(values):
            return stack_slices(list(values), memory_type, 0)
        return np.stack(
            tuple(np.asarray(value) for value in values),
            axis=0,
        )

    @classmethod
    def stack_grouped_masks(cls, masks: tuple[ImagePayloadMaskValue, ...]) -> ImagePayloadMaskValue:
        """Stack grouped masks while rejecting partially masked image groups."""
        present_masks = tuple(mask for mask in masks if mask is not None)
        if present_masks and len(present_masks) != len(masks):
            raise ValueError("Cannot stack mixed masked and unmasked grouped image inputs.")
        if not present_masks:
            return None
        return cls.stack_grouped_planes(present_masks)

    @staticmethod
    def grouped_image_array(array: ImagePayloadValue) -> ImagePayloadValue:
        array_view = np.asarray(array)
        if (
            array_view.ndim == 3
            and not is_color_image_slice(array_view)
            and array_view.shape[0] == 1
        ):
            return array_view[0]
        return array

    @staticmethod
    def all_values_are_2d_arrays(values: tuple[ImagePayloadValue, ...]) -> bool:
        """Return whether every grouped value is a two-dimensional array-like plane."""
        return all(np.asarray(value).ndim == 2 for value in values)

    @staticmethod
    def stack_object_label_records(
        records: tuple[StoredRuntimeValue, ...],
    ) -> ObjectLabelSet:
        values = tuple(
            ObjectLabelSet.from_runtime_value(record.value)
            for record in records
        )
        first = values[0]
        representations = {value.representation for value in values}
        if len(representations) != 1:
            raise ValueError("Cannot stack grouped object labels with mixed representations.")
        return ObjectLabelPure2DSliceAggregator.aggregate(
            values,
            detect_memory_type(first.labels),
            force_plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )

    @classmethod
    def normalize_dense_object_label_payload(cls, labels: DenseLabelPayload) -> DenseLabelPayload:
        """Return dense object labels as one array payload, not slice lists."""
        if labels is None or isinstance(labels, SparseIJVLabelRows):
            return labels
        if not _is_sequence_payload(labels):
            return labels
        if not labels:
            return np.asarray(labels, dtype=np.int32)
        memory_type = detect_memory_type(labels[0])
        label_tuple = tuple(labels)
        DenseLabelSequenceMemoryBudget(label_tuple).validate()
        return ImageStackLayout.stack_slices_or_single_stack(
            labels,
            memory_type=memory_type,
            gpu_id=0,
        )

    @staticmethod
    def stack_dense_label_sequence(
        labels: Sequence[DenseLabelPayload],
        memory_type: str,
    ) -> DenseLabelPayload:
        """Stack a homogeneous dense-label sequence without image-slice assumptions."""
        label_list = list(labels)
        arrays = tuple(np.asarray(label) for label in label_list)
        shapes = {tuple(array.shape) for array in arrays}
        if len(shapes) == 1:
            _raise_if_dense_label_stack_too_large(arrays)
            return np.stack(arrays, axis=0)
        return stack_slices(label_list, memory_type, 0)


def _raise_if_dense_label_stack_too_large(arrays: tuple[np.ndarray, ...]) -> None:
    total_bytes = sum(array.nbytes for array in arrays)
    if total_bytes > _MAX_DENSE_LABEL_STACK_BYTES:
        raise MemoryError(
            "Refusing to materialize dense object-label stack larger than "
            f"{_MAX_DENSE_LABEL_STACK_BYTES} bytes; requested {total_bytes} bytes."
        )


@dataclass(frozen=True, slots=True)
class DenseLabelSequenceMemoryBudget:
    """Fail before dense label list normalization can bypass stack limits."""

    labels: tuple[DenseLabelPayload, ...]

    def validate(self) -> None:
        arrays = tuple(np.asarray(label) for label in self.labels)
        if DenseLabelShapeSet(arrays).is_uniform:
            _raise_if_dense_label_stack_too_large(arrays)


def _is_sequence_payload(labels: DenseLabelPayload) -> bool:
    return isinstance(labels, Sequence) and not isinstance(
        labels,
        (str, bytes, bytearray, Mapping),
    )


class SpatialGridValueAuthority:
    """Normalize, compare, and collapse grouped spatial-grid runtime values."""

    @staticmethod
    def native_value(name: str, value: SpatialGridInputValue) -> SpatialGrid:
        if isinstance(value, SpatialGrid):
            return value.with_name(name)
        if isinstance(value, Mapping):
            return SpatialGrid.from_mapping(name, value)
        raise TypeError(
            f"Spatial grid slice '{name}' must be SpatialGrid or mapping-backed, "
            f"got {type(value).__name__}."
        )

    @classmethod
    def input_value(
        cls,
        name: str,
        value: SpatialGridInput,
    ) -> SpatialGrid | RuntimeSliceAlignedValues[SpatialGrid]:
        if isinstance(value, RuntimeSliceAlignedValues):
            return RuntimeSliceAlignedValues(
                slices=tuple(cls.native_value(name, item) for item in value.slices)
            )
        return cls.native_value(name, value)

    @staticmethod
    def record_value(
        name: str,
        record: StoredRuntimeValue,
    ) -> SpatialGrid | RuntimeSliceAlignedValues[SpatialGrid]:
        data = record.value.data
        if isinstance(data, tuple | list) and all(
            isinstance(value, Mapping) for value in data
        ):
            return RuntimeSliceAlignedValues(
                slices=tuple(
                    SpatialGrid.from_mapping(name, value) for value in data
                )
            )
        return SpatialGrid.from_runtime_value(record.value)

    @classmethod
    def single_spatial_grid(
        cls,
        name: str,
        grids: SpatialGridGroupValues,
    ) -> SpatialGrid | RuntimeSliceAlignedValues[SpatialGrid]:
        if not grids:
            raise RuntimeError(f"Missing spatial grid artifact {name!r}.")
        if any(isinstance(grid, RuntimeSliceAlignedValues) for grid in grids):
            return cls.single_slice_aligned_spatial_grid(name, grids)
        first = grids[0]
        first_payload = cls.equivalence_payload(first)
        if all(cls.equivalence_payload(grid) == first_payload for grid in grids):
            return first.with_name(name)
        raise RuntimeError(
            f"Spatial grid artifact {name!r} resolved to non-identical grouped grids."
        )

    @classmethod
    def single_slice_aligned_spatial_grid(
        cls,
        name: str,
        grids: SpatialGridGroupValues,
    ) -> RuntimeSliceAlignedValues[SpatialGrid]:
        slice_count = max(SpatialGridSliceCount(grid).value for grid in grids)
        aligned_slices: list[SpatialGrid] = []
        for slice_index in range(slice_count):
            candidates = tuple(
                cls.for_aligned_slice(grid, slice_index, slice_count)
                for grid in grids
            )
            first = candidates[0]
            first_payload = cls.equivalence_payload(first)
            if not all(
                cls.equivalence_payload(candidate) == first_payload
                for candidate in candidates
            ):
                raise RuntimeError(
                    f"Spatial grid artifact {name!r} resolved to non-identical "
                    "slice-aligned grouped grids."
                )
            aligned_slices.append(first.with_name(name))
        return RuntimeSliceAlignedValues(slices=tuple(aligned_slices))

    @staticmethod
    def for_aligned_slice(
        grid: SpatialGrid | RuntimeSliceAlignedValues[SpatialGrid],
        slice_index: int,
        slice_count: int,
    ) -> SpatialGrid:
        if isinstance(grid, RuntimeSliceAlignedValues):
            if grid.slice_count == slice_count:
                return grid.value_for_slice(slice_index)
            if grid.slice_count == 1:
                return grid.value_for_slice(0)
            raise RuntimeError(
                "Spatial grid artifact resolved to incompatible slice-aligned "
                f"counts {grid.slice_count} and {slice_count}."
            )
        return grid

    @staticmethod
    def equivalence_payload(grid: SpatialGrid) -> dict[str, SpatialGridEquivalenceValue]:
        return {**grid.as_mapping(), "slice_index": 0}
