"""Relationship backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.core.registry_strategies import MostDerivedContextStrategyMixin
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.core.runtime_semantics import (
    ParentChildRelationshipPayload,
    aligned_dense_object_label_arrays,
)
from openhcs.core.runtime_values import (
    ObjectLabelRepresentation,
    ObjectLabelSet,
    SparseIJVLabelRows,
    object_label_dense_array,
)


@dataclass(frozen=True, slots=True)
class ObjectRelationshipPayloadRequest:
    """Inputs for deriving CellProfiler-compatible object relationship payloads."""

    backend: "ObjectRelationshipBackendStrategy"
    parent_labels: Any
    child_labels: Any


class ObjectRelationshipBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Object relationship operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def relate_children_to_parents(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        child_count: int,
    ) -> np.ndarray:
        """Assign each child to its parent object."""

    @abstractmethod
    def centroid_distances(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        parents_of: np.ndarray,
    ) -> np.ndarray:
        """Return child-parent centroid distances."""

    @abstractmethod
    def minimum_distances(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        parents_of: np.ndarray,
    ) -> np.ndarray:
        """Return child-centroid to parent-boundary distances."""

    @abstractmethod
    def label_centers(self, labels: np.ndarray) -> np.ndarray:
        """Return row/column centers indexed by dense positive label id."""

    def parent_child_payload_from_labels(
        self,
        parent_labels: Any,
        child_labels: Any,
    ) -> ParentChildRelationshipPayload:
        """Return parent-child ids using the labels' nominal representation."""
        return ObjectRelationshipPayloadStrategy.for_context(
            ObjectRelationshipPayloadRequest(
                backend=self,
                parent_labels=parent_labels,
                child_labels=child_labels,
            )
        ).payload()

    def parents_of_from_payload(
        self,
        payload: ParentChildRelationshipPayload,
        child_count: int,
    ) -> np.ndarray:
        """Return a dense parents-of-child vector from a relationship payload."""
        parents_of = np.zeros(child_count, dtype=np.int32)
        for parent_id, child_id in zip(
            payload.parent_ids,
            payload.child_ids,
            strict=True,
        ):
            if 0 < child_id <= child_count:
                parents_of[child_id - 1] = int(parent_id)
        return parents_of


class ObjectRelationshipPayloadStrategy(
    MostDerivedContextStrategyMixin[ObjectRelationshipPayloadRequest],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Derive parent-child payloads by nominal object-label representation."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[str | None] = None

    request: ObjectRelationshipPayloadRequest

    def __init__(
        self,
        request: ObjectRelationshipPayloadRequest | None = None,
    ) -> None:
        if request is not None:
            self.request = request

    @classmethod
    def for_context(
        cls,
        context: ObjectRelationshipPayloadRequest,
        *,
        required: bool = True,
        error_subject: str | None = None,
    ) -> "ObjectRelationshipPayloadStrategy":
        strategy = super().for_context(
            context,
            required=required,
            error_subject=error_subject,
        )
        if strategy is None:
            raise ValueError("Object relationship payload requires a strategy.")
        strategy.request = context
        return strategy

    @abstractmethod
    def matches(self, context: ObjectRelationshipPayloadRequest) -> bool:
        """Return whether this strategy owns the label representation pair."""

    @abstractmethod
    def payload(self) -> ParentChildRelationshipPayload:
        """Return parent-child ids for the strategy's representation contract."""

    @staticmethod
    def related_payload_from_parents_of(
        parents_of: np.ndarray,
        child_ids: np.ndarray,
    ) -> ParentChildRelationshipPayload:
        parent_ids: list[int] = []
        related_child_ids: list[int] = []
        for child_id in child_ids:
            if 0 < child_id <= len(parents_of):
                parent_id = int(parents_of[child_id - 1])
                if parent_id > 0:
                    parent_ids.append(parent_id)
                    related_child_ids.append(int(child_id))
        return ParentChildRelationshipPayload(
            parent_ids=tuple(parent_ids),
            child_ids=tuple(related_child_ids),
        )


class DenseObjectRelationshipPayloadStrategy(ObjectRelationshipPayloadStrategy):
    """Dense label images use maximum positive-pixel overlap."""

    strategy_key = "dense"

    def matches(self, context: ObjectRelationshipPayloadRequest) -> bool:
        del context
        return True

    def payload(self) -> ParentChildRelationshipPayload:
        parent_array, child_array = aligned_dense_object_label_arrays(
            object_label_dense_array(self.request.parent_labels, dtype=np.int32),
            object_label_dense_array(self.request.child_labels, dtype=np.int32),
        )
        child_count = int(child_array.max()) if child_array.size else 0
        if child_count <= 0:
            return ParentChildRelationshipPayload(parent_ids=(), child_ids=())
        present_children = _present_positive_labels_numba(child_array, child_count)
        parents_of = self.request.backend.relate_children_to_parents(
            parent_array,
            child_array,
            child_count,
        )
        return self.related_payload_from_parents_of(parents_of, present_children)


class SparseIJVObjectRelationshipPayloadStrategy(DenseObjectRelationshipPayloadStrategy):
    """Sparse IJV labels match CellProfiler's non-volumetric object relation."""

    strategy_key = "sparse_ijv"

    def matches(self, context: ObjectRelationshipPayloadRequest) -> bool:
        return self.is_sparse_ijv(context.parent_labels) or self.is_sparse_ijv(
            context.child_labels
        )

    def payload(self) -> ParentChildRelationshipPayload:
        parent_rows = self.sparse_rows(self.request.parent_labels)
        child_rows = self.sparse_rows(self.request.child_labels)
        parent_array = parent_rows.as_yx_label_array()
        child_array = child_rows.as_yx_label_array()
        parent_count = self.label_count(parent_array, parent_rows)
        child_count = self.label_count(child_array, child_rows)
        if parent_count <= 0 or child_count <= 0:
            return ParentChildRelationshipPayload(parent_ids=(), child_ids=())
        parents_of = _relate_sparse_ijv_children_to_parents_numba(
            np.asarray(parent_array, dtype=np.int64),
            np.asarray(child_array, dtype=np.int64),
            child_count,
            parent_count,
        )
        present_children = self.present_sparse_child_ids(child_array, child_rows)
        return self.related_payload_from_parents_of(parents_of, present_children)

    @classmethod
    def is_sparse_ijv(cls, labels: Any) -> bool:
        if isinstance(labels, SparseIJVLabelRows):
            return True
        return (
            isinstance(labels, ObjectLabelSet)
            and labels.representation is ObjectLabelRepresentation.SPARSE_IJV
        )

    @classmethod
    def sparse_rows(cls, labels: Any) -> SparseIJVLabelRows:
        if isinstance(labels, ObjectLabelSet):
            if labels.representation is not ObjectLabelRepresentation.SPARSE_IJV:
                return SparseIJVLabelRows.from_dense_labels(labels.labels)
            labels = labels.labels
        if isinstance(labels, SparseIJVLabelRows):
            return labels
        return SparseIJVLabelRows.from_dense_labels(labels)

    @staticmethod
    def label_count(
        array: np.ndarray,
        rows: SparseIJVLabelRows,
    ) -> int:
        if array.size == 0:
            return 0
        return int(np.max(array[:, rows.label_column]))

    @staticmethod
    def present_sparse_child_ids(
        child_array: np.ndarray,
        child_rows: SparseIJVLabelRows,
    ) -> np.ndarray:
        if child_array.size == 0:
            return np.empty(0, dtype=np.int32)
        return np.unique(child_array[:, child_rows.label_column]).astype(
            np.int32,
            copy=False,
        )


class NumbaNumpyObjectRelationshipBackendStrategy(
    ObjectRelationshipBackendStrategy
):
    """Numba-accelerated NumPy object relationship primitives."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def relate_children_to_parents(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        child_count: int,
    ) -> np.ndarray:
        parent_count = int(parent_labels.max()) if parent_labels.max() > 0 else 0
        parents_of = np.zeros(child_count, dtype=np.int32)
        if child_count == 0 or parent_count == 0:
            return parents_of

        return _relate_children_to_parents_numba(
            np.asarray(parent_labels),
            np.asarray(child_labels),
            child_count,
            parent_count,
        )

    def centroid_distances(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        parents_of: np.ndarray,
    ) -> np.ndarray:
        parent_count = int(parent_labels.max())
        return _calculate_centroid_distances_numba(
            np.ascontiguousarray(parent_labels),
            np.ascontiguousarray(child_labels),
            np.asarray(parents_of, dtype=np.int32),
            parent_count,
        )

    def minimum_distances(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        parents_of: np.ndarray,
    ) -> np.ndarray:
        parent_count = int(parent_labels.max())
        return _calculate_minimum_distances_numba(
            np.ascontiguousarray(parent_labels),
            np.ascontiguousarray(child_labels),
            np.asarray(parents_of, dtype=np.int32),
            parent_count,
        )

    def label_centers(self, labels: np.ndarray) -> np.ndarray:
        label_count = int(labels.max())
        if label_count == 0:
            return np.empty((0, 2), dtype=np.float64)
        centroids = _label_centroids_numba(
            np.ascontiguousarray(labels),
            label_count,
        )
        return centroids[1:]


def object_relationship_backend(
    *,
    backend_provider: BackendProviderInput | None = None,
) -> ObjectRelationshipBackendStrategy:
    """Return the selected CellProfiler object relationship backend."""
    return ObjectRelationshipBackendStrategy.for_memory_type(
        backend_provider=backend_provider,
    )


@njit(cache=True)
def _present_positive_labels_numba(
    labels: np.ndarray,
    label_count: int,
) -> np.ndarray:
    present = np.zeros(label_count + 1, dtype=np.bool_)
    height, width = labels.shape
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id > 0 and label_id <= label_count:
                present[label_id] = True
    count = 0
    for label_id in range(1, label_count + 1):
        if present[label_id]:
            count += 1
    labels_out = np.empty(count, dtype=np.int32)
    index = 0
    for label_id in range(1, label_count + 1):
        if present[label_id]:
            labels_out[index] = label_id
            index += 1
    return labels_out


@njit(cache=True)
def _relate_sparse_ijv_children_to_parents_numba(
    parent_ijv: np.ndarray,
    child_ijv: np.ndarray,
    child_count: int,
    parent_count: int,
) -> np.ndarray:
    counts = np.zeros((child_count + 1, parent_count + 1), dtype=np.int32)
    parent_linear = _sparse_ijv_linear_coordinates(parent_ijv, child_ijv)
    child_linear = _sparse_ijv_linear_coordinates(child_ijv, parent_ijv)
    parent_order = np.argsort(parent_linear)
    child_order = np.argsort(child_linear)
    parent_position = 0
    child_position = 0
    while parent_position < parent_order.size and child_position < child_order.size:
        parent_index = parent_order[parent_position]
        child_index = child_order[child_position]
        parent_coordinate = parent_linear[parent_index]
        child_coordinate = child_linear[child_index]
        if parent_coordinate < child_coordinate:
            parent_position += 1
            continue
        if child_coordinate < parent_coordinate:
            child_position += 1
            continue

        parent_end = parent_position + 1
        while (
            parent_end < parent_order.size
            and parent_linear[parent_order[parent_end]] == parent_coordinate
        ):
            parent_end += 1
        child_end = child_position + 1
        while (
            child_end < child_order.size
            and child_linear[child_order[child_end]] == child_coordinate
        ):
            child_end += 1
        for grouped_parent_position in range(parent_position, parent_end):
            grouped_parent_index = parent_order[grouped_parent_position]
            parent_id = int(parent_ijv[grouped_parent_index, 2])
            if parent_id <= 0 or parent_id > parent_count:
                continue
            for grouped_child_position in range(child_position, child_end):
                grouped_child_index = child_order[grouped_child_position]
                child_id = int(child_ijv[grouped_child_index, 2])
                if child_id > 0 and child_id <= child_count:
                    counts[child_id, parent_id] += 1
        parent_position = parent_end
        child_position = child_end

    parents_of = np.zeros(child_count, dtype=np.int32)
    for child_id in range(1, child_count + 1):
        best_parent = 0
        best_count = 0
        for parent_id in range(1, parent_count + 1):
            overlap = counts[child_id, parent_id]
            if overlap > best_count:
                best_count = overlap
                best_parent = parent_id
        parents_of[child_id - 1] = best_parent
    return parents_of


@njit(cache=True)
def _sparse_ijv_linear_coordinates(
    rows: np.ndarray,
    peer_rows: np.ndarray,
) -> np.ndarray:
    max_y = 0
    for index in range(rows.shape[0]):
        y = int(rows[index, 0])
        if y > max_y:
            max_y = y
    for index in range(peer_rows.shape[0]):
        y = int(peer_rows[index, 0])
        if y > max_y:
            max_y = y
    dim_y = max_y + 1
    linear = np.empty(rows.shape[0], dtype=np.int64)
    for index in range(rows.shape[0]):
        linear[index] = int(rows[index, 0]) + dim_y * int(rows[index, 1])
    return linear


@njit(cache=True)
def _relate_children_to_parents_numba(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    child_count: int,
    parent_count: int,
) -> np.ndarray:
    counts = np.zeros((child_count + 1, parent_count + 1), dtype=np.int32)
    height, width = child_labels.shape
    for row in range(height):
        for col in range(width):
            child_id = int(child_labels[row, col])
            parent_id = int(parent_labels[row, col])
            if (
                child_id > 0
                and child_id <= child_count
                and parent_id > 0
                and parent_id <= parent_count
            ):
                counts[child_id, parent_id] += 1

    parents_of = np.zeros(child_count, dtype=np.int32)
    for child_id in range(1, child_count + 1):
        best_parent = 0
        best_count = 0
        for parent_id in range(1, parent_count + 1):
            overlap = counts[child_id, parent_id]
            if overlap > best_count:
                best_count = overlap
                best_parent = parent_id
        parents_of[child_id - 1] = best_parent
    return parents_of


@njit(cache=True)
def _label_centroids_numba(
    labels: np.ndarray,
    label_count: int,
) -> np.ndarray:
    sums = np.zeros((label_count + 1, 2), dtype=np.float64)
    counts = np.zeros(label_count + 1, dtype=np.int64)
    height, width = labels.shape
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id > 0 and label_id <= label_count:
                sums[label_id, 0] += row
                sums[label_id, 1] += col
                counts[label_id] += 1

    centroids = np.empty((label_count + 1, 2), dtype=np.float64)
    for label_id in range(label_count + 1):
        if counts[label_id] == 0:
            centroids[label_id, 0] = np.nan
            centroids[label_id, 1] = np.nan
        else:
            centroids[label_id, 0] = sums[label_id, 0] / counts[label_id]
            centroids[label_id, 1] = sums[label_id, 1] / counts[label_id]
    return centroids


@njit(cache=True)
def _calculate_centroid_distances_numba(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    parents_of: np.ndarray,
    parent_count: int,
) -> np.ndarray:
    child_count = len(parents_of)
    distances = np.empty(child_count, dtype=np.float64)
    for child_idx in range(child_count):
        distances[child_idx] = np.nan

    if child_count == 0 or parent_count == 0:
        return distances

    parent_centroids = _label_centroids_numba(parent_labels, parent_count)
    child_centroids = _label_centroids_numba(child_labels, child_count)
    for child_idx in range(child_count):
        parent_id = int(parents_of[child_idx])
        child_id = child_idx + 1
        if parent_id > 0 and parent_id <= parent_count:
            child_row = child_centroids[child_id, 0]
            child_col = child_centroids[child_id, 1]
            parent_row = parent_centroids[parent_id, 0]
            parent_col = parent_centroids[parent_id, 1]
            if not (
                np.isnan(child_row)
                or np.isnan(child_col)
                or np.isnan(parent_row)
                or np.isnan(parent_col)
            ):
                row_delta = child_row - parent_row
                col_delta = child_col - parent_col
                distances[child_idx] = np.sqrt(
                    row_delta * row_delta + col_delta * col_delta
                )
    return distances


@njit(cache=True)
def _is_inner_boundary_pixel(
    labels: np.ndarray,
    row: int,
    col: int,
    label_id: int,
) -> bool:
    height, width = labels.shape
    if row > 0 and int(labels[row - 1, col]) != label_id:
        return True
    if row + 1 < height and int(labels[row + 1, col]) != label_id:
        return True
    if col > 0 and int(labels[row, col - 1]) != label_id:
        return True
    if col + 1 < width and int(labels[row, col + 1]) != label_id:
        return True
    return False


@njit(cache=True)
def _calculate_minimum_distances_numba(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    parents_of: np.ndarray,
    parent_count: int,
) -> np.ndarray:
    child_count = len(parents_of)
    distances = np.empty(child_count, dtype=np.float64)
    for child_idx in range(child_count):
        distances[child_idx] = np.nan

    if child_count == 0 or parent_count == 0:
        return distances

    child_centroids = _label_centroids_numba(child_labels, child_count)
    height, width = parent_labels.shape
    counts = np.zeros(parent_count + 1, dtype=np.int64)

    for row in range(height):
        for col in range(width):
            parent_id = int(parent_labels[row, col])
            if (
                parent_id > 0
                and parent_id <= parent_count
                and _is_inner_boundary_pixel(parent_labels, row, col, parent_id)
            ):
                counts[parent_id] += 1

    offsets = np.zeros(parent_count + 2, dtype=np.int64)
    for parent_id in range(1, parent_count + 1):
        offsets[parent_id + 1] = offsets[parent_id] + counts[parent_id]

    total = offsets[parent_count + 1]
    rows = np.empty(total, dtype=np.float64)
    cols = np.empty(total, dtype=np.float64)
    write_offsets = offsets.copy()
    for row in range(height):
        for col in range(width):
            parent_id = int(parent_labels[row, col])
            if (
                parent_id > 0
                and parent_id <= parent_count
                and _is_inner_boundary_pixel(parent_labels, row, col, parent_id)
            ):
                offset = write_offsets[parent_id]
                rows[offset] = row
                cols[offset] = col
                write_offsets[parent_id] += 1

    for child_idx in range(child_count):
        parent_id = int(parents_of[child_idx])
        child_id = child_idx + 1
        if parent_id <= 0 or parent_id > parent_count:
            continue
        child_row = child_centroids[child_id, 0]
        child_col = child_centroids[child_id, 1]
        if np.isnan(child_row) or np.isnan(child_col):
            continue
        start = offsets[parent_id]
        end = offsets[parent_id + 1]
        if start == end:
            continue
        min_distance_sq = np.inf
        for offset in range(start, end):
            row_delta = rows[offset] - child_row
            col_delta = cols[offset] - child_col
            distance_sq = row_delta * row_delta + col_delta * col_delta
            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
        distances[child_idx] = np.sqrt(min_distance_sq)

    return distances


__all__ = [
    "NumbaNumpyObjectRelationshipBackendStrategy",
    "ObjectRelationshipBackendStrategy",
    "object_relationship_backend",
]
