"""Nominal runtime relationship values."""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Self

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactPlan,
    ArtifactSpecRef,
    ArtifactSpecRelation,
    NamedArtifactPayload,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
)
from openhcs.core.runtime_plane_projection import RuntimeSliceProjectableValue
from openhcs.core.source_image_provenance import (
    SourceImageProvenance,
    SourceImageProvenanceFields,
)

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from openhcs.core.registry_strategies import MostDerivedContextStrategyMixin
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.source_spatial_domain import SourceSpatialDomainAdapter
from openhcs.core.runtime_object_label_domains import ObjectLabelDomainMetadataStrategy
from openhcs.core.runtime_object_label_domains import ObjectLabelDomainScope
from openhcs.core.runtime_object_label_domains import dense_object_label_id_domain
from openhcs.core.runtime_plane_projection import RuntimeSliceIdentityProjectableValue
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhcs.core.runtime_object_labels import ObjectLabelMeasurementSource
    from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows


@dataclass(frozen=True)
class ObjectRelationshipDeclaration(ArtifactSpecRelation):
    """Authoritative directed relationship semantics for one artifact output."""

    relation_key = "object_relationship_declaration"
    target_plan_type: ClassVar[type[ArtifactPlan] | None] = None
    target: ArtifactSpecRef
    relationship_type: str
    source_role: str
    target_role: str
    source_id_field: str
    target_id_field: str
    producer_module_number: int
    source_runtime_slice_offset: int = 0
    target_runtime_slice_offset: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source.artifact_type is not ObjectLabelsArtifactType:
            raise ValueError(
                "ObjectRelationshipDeclaration source must be object labels, got "
                f"{self.source.artifact_type.value}:{self.source.name}."
            )
        if not isinstance(self.target, ArtifactSpecRef):
            raise TypeError(
                "ObjectRelationshipDeclaration.target must be ArtifactSpecRef, got "
                f"{type(self.target).__name__}."
            )
        if self.target.artifact_type is not ObjectLabelsArtifactType:
            raise ValueError(
                "ObjectRelationshipDeclaration target must be object labels, got "
                f"{self.target.artifact_type.value}:{self.target.name}."
            )
        _require_name(self.relationship_type, "relationship_type")
        _require_name(self.source_role, "source_role")
        _require_name(self.target_role, "target_role")
        _require_name(self.source_id_field, "source_id_field")
        _require_name(self.target_id_field, "target_id_field")
        if type(self.source_runtime_slice_offset) is not int:
            raise TypeError(
                "ObjectRelationshipDeclaration.source_runtime_slice_offset must be int."
            )
        if type(self.target_runtime_slice_offset) is not int:
            raise TypeError(
                "ObjectRelationshipDeclaration.target_runtime_slice_offset must be int."
            )
        if (
            not isinstance(self.producer_module_number, int)
            or self.producer_module_number < 1
        ):
            raise ValueError(
                "ObjectRelationshipDeclaration.producer_module_number must be a "
                f"positive integer, got {self.producer_module_number!r}."
            )

    @classmethod
    def parent_child(
        cls,
        *,
        source: ArtifactSpecRef,
        target: ArtifactSpecRef,
        producer_module_number: int,
        relationship_type: str = "parent_child",
        source_runtime_slice_offset: int = 0,
        target_runtime_slice_offset: int = 0,
    ) -> "ObjectRelationshipDeclaration":
        """Declare canonical parent-to-child object lineage."""

        return cls(
            source=source,
            target=target,
            relationship_type=relationship_type,
            source_role="parent",
            target_role="child",
            source_id_field="parent_id",
            target_id_field="child_id",
            producer_module_number=producer_module_number,
            source_runtime_slice_offset=source_runtime_slice_offset,
            target_runtime_slice_offset=target_runtime_slice_offset,
        )

    def require_target_spec(self, spec: ArtifactSpec) -> None:
        super().require_target_spec(spec)
        if not issubclass(spec.artifact_type, ObjectLineageArtifactType):
            raise ValueError(
                "ObjectRelationshipDeclaration requires an object-lineage artifact "
                f"target, got {spec.artifact_type.value}:{spec.name}."
            )

    def for_plan_type(
        self,
        plan_type: type[ArtifactPlan],
    ) -> "ObjectRelationshipDeclaration":
        """Project endpoint identities with the relationship artifact role."""

        return replace(
            self,
            source=self.source.for_plan_type(plan_type),
            target=self.target.for_plan_type(plan_type),
        )

    def artifact_name(self) -> str:
        """Return the deterministic identity owned by this declaration."""

        return "_".join(
            (
                "module",
                str(self.producer_module_number),
                self.relationship_type,
                self.source.name,
                self.target.name,
                "relationships",
            )
        )

    def dependency_refs(self) -> tuple[ArtifactSpecRef, ...]:
        """Return both object-label endpoints required by this relationship."""

        return (self.source, self.target)

    def projects_parent_child_measurements(self) -> bool:
        """Return whether this declaration contributes Parent/Children rows."""

        return self.source_role == "parent" and self.target_role == "child"

    def endpoint_runtime_slice_indices(
        self,
        payload_slice_index: int,
        slice_count: int,
    ) -> tuple[int | None, int | None]:
        """Project one payload slice onto both declared endpoint slices."""

        source_index = int(payload_slice_index) + self.source_runtime_slice_offset
        target_index = int(payload_slice_index) + self.target_runtime_slice_offset
        return (
            source_index if 0 <= source_index < int(slice_count) else None,
            target_index if 0 <= target_index < int(slice_count) else None,
        )


@dataclass(slots=True, kw_only=True)
class ObjectRelationship(
    SourceImageProvenanceFields,
    RuntimeSliceProjectableValue,
    NamedArtifactPayload,
):
    """Native OpenHCS directed object relationship value."""

    name: str
    declaration: ObjectRelationshipDeclaration
    payload: DirectedObjectRelationshipPayload

    @classmethod
    def from_payload(
        cls,
        *,
        name: str,
        declaration: ObjectRelationshipDeclaration,
        payload: DirectedObjectRelationshipPayload,
        source_provenance: SourceImageProvenance = SourceImageProvenance(),
    ) -> Self:
        """Bind one endpoint-neutral payload to its compiled declaration."""

        if not isinstance(payload, DirectedObjectRelationshipPayload):
            raise TypeError(
                "ObjectRelationship.from_payload requires a directed relationship "
                f"payload, got {type(payload).__name__}."
            )
        return cls(
            name=name,
            declaration=declaration,
            payload=payload,
            source_provenance=source_provenance,
        )

    def __post_init__(self, *source_provenance_values: object) -> None:
        self.validate_artifact_name()
        self.absorb_explicit_source_provenance(
            SourceImageProvenance.from_init_values(source_provenance_values)
        )
        self.normalize_source_provenance_fields()
        if not isinstance(self.declaration, ObjectRelationshipDeclaration):
            raise TypeError(
                "ObjectRelationship.declaration must be ObjectRelationshipDeclaration, "
                f"got {type(self.declaration).__name__}."
            )
        if not isinstance(self.payload, DirectedObjectRelationshipPayload):
            raise TypeError(
                "ObjectRelationship.payload must be DirectedObjectRelationshipPayload, "
                f"got {type(self.payload).__name__}."
            )

    def project_runtime_slice(self, slice_index: int) -> "ObjectRelationship":
        """Return only relationship rows belonging to one runtime slice."""
        source_provenance = self.source_provenance.for_source_plane(slice_index)
        return ObjectRelationship(
            name=self.name,
            declaration=self.declaration,
            payload=self.payload.project_runtime_slice(slice_index),
            source_provenance=source_provenance,
        )

    def relationship_columns(self) -> dict[str, Any]:
        """Return semantic relationship columns without payload provenance."""

        columns = {
            "relationship_type": self.declaration.relationship_type,
            "source_role": self.declaration.source_role,
            "target_role": self.declaration.target_role,
            "source_object": self.declaration.source.name,
            "target_object": self.declaration.target.name,
            "producer_module_number": self.declaration.producer_module_number,
            self.declaration.source_id_field: self.payload.source_ids,
            self.declaration.target_id_field: self.payload.target_ids,
        }
        if self.payload.slice_indices:
            columns["slice_index"] = self.payload.slice_indices
        if self.payload.slice_count is not None:
            columns["slice_count"] = self.payload.slice_count
        return columns

    def as_table(self) -> dict[str, Any]:
        """Return table-like relationship columns for materialization."""

        table = self.relationship_columns()
        if self.source_path is not None:
            table["source_path"] = self.source_path
        if self.source_component_metadata is not None:
            table["source_component_metadata"] = self.source_component_metadata
        if self.source_image_provenance_planes.has_values:
            table["source_image_provenance_planes"] = (
                self.source_image_provenance_planes.records
            )
        return table

    def row_mappings(self) -> tuple[Mapping[str, Any], ...]:
        """Return canonical relationship columns expanded into row mappings."""

        table = self.relationship_columns()
        vector_columns = {
            self.declaration.source_id_field: self.payload.source_ids,
            self.declaration.target_id_field: self.payload.target_ids,
        }
        if self.payload.slice_indices:
            vector_columns["slice_index"] = self.payload.slice_indices
        return tuple(
            {
                field_name: (
                    vector_columns[field_name][row_index]
                    if field_name in vector_columns
                    else value
                )
                for field_name, value in table.items()
            }
            for row_index in range(len(self.payload.source_ids))
        )


def _directed_relationship_payload_fields(
    source_ids: Iterable[int],
    target_ids: Iterable[int],
    slice_indices: Iterable[int],
    slice_count: int | None,
    *,
    payload_name: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int | None]:
    """Normalize the shared axis contract for one directed relationship payload."""

    normalized_source_ids = tuple(int(source_id) for source_id in source_ids)
    normalized_target_ids = tuple(int(target_id) for target_id in target_ids)
    if len(normalized_source_ids) != len(normalized_target_ids):
        raise ValueError(
            f"{payload_name} source and target IDs must have equal length, got "
            f"{len(normalized_source_ids)} and {len(normalized_target_ids)}."
        )
    normalized_slice_indices = tuple(int(index) for index in slice_indices)
    if normalized_slice_indices and len(normalized_slice_indices) != len(
        normalized_source_ids
    ):
        raise ValueError(
            f"{payload_name} slice_indices must be empty or match the relationship "
            f"count, got {len(normalized_slice_indices)} for "
            f"{len(normalized_source_ids)} relationships."
        )
    if any(index < 0 for index in normalized_slice_indices):
        raise ValueError(f"{payload_name} slice_indices cannot be negative.")
    normalized_slice_count = None if slice_count is None else int(slice_count)
    if normalized_slice_count is not None and normalized_slice_count < 0:
        raise ValueError(f"{payload_name} slice_count cannot be negative.")
    if (
        normalized_slice_count is not None
        and normalized_slice_indices
        and max(normalized_slice_indices) >= normalized_slice_count
    ):
        raise ValueError(
            f"{payload_name} slice_indices must be smaller than slice_count "
            f"{normalized_slice_count}."
        )
    return (
        normalized_source_ids,
        normalized_target_ids,
        normalized_slice_indices,
        normalized_slice_count,
    )


@dataclass(frozen=True, slots=True)
class DirectedObjectRelationshipPayload(
    RuntimeSliceProjectableValue, RuntimeSliceIdentityProjectableValue
):
    """Endpoint-neutral directed object ID pairs awaiting contract binding."""

    source_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    slice_indices: tuple[int, ...] = ()
    slice_count: int | None = None

    def __post_init__(self) -> None:
        values = _directed_relationship_payload_fields(
            self.source_ids,
            self.target_ids,
            self.slice_indices,
            self.slice_count,
            payload_name=type(self).__name__,
        )
        for field_name, value in zip(
            ("source_ids", "target_ids", "slice_indices", "slice_count"),
            values,
            strict=True,
        ):
            object.__setattr__(self, field_name, value)

    def with_runtime_slice_identity(
        self, *, slice_index: int, slice_count: int
    ) -> "DirectedObjectRelationshipPayload":
        return type(self)(
            source_ids=self.source_ids,
            target_ids=self.target_ids,
            slice_indices=tuple(int(slice_index) for _ in self.source_ids),
            slice_count=int(slice_count),
        )

    def runtime_slice_pairs(
        self,
    ) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...] | None:
        """Return relationship pairs partitioned by their declared runtime slice."""

        if self.slice_count is None:
            return None
        pairs_by_slice: list[list[tuple[int, int]]] = [
            [] for _slice_index in range(self.slice_count)
        ]
        if self.slice_indices:
            for slice_index, source_id, target_id in zip(
                self.slice_indices,
                self.source_ids,
                self.target_ids,
                strict=True,
            ):
                pairs_by_slice[slice_index].append((source_id, target_id))
        elif self.source_ids:
            if self.slice_count != 1:
                raise ValueError(
                    "Multi-slice relationship pairs require explicit slice indices."
                )
            pairs_by_slice[0].extend(
                zip(self.source_ids, self.target_ids, strict=True)
            )
        return tuple(
            (slice_index, tuple(pairs))
            for slice_index, pairs in enumerate(pairs_by_slice)
        )

    def project_runtime_slice(
        self, slice_index: int
    ) -> "DirectedObjectRelationshipPayload":
        if not self.slice_indices:
            if (
                self.slice_count is not None
                and self.slice_count > 1
                and self.source_ids
            ):
                raise ValueError(
                    "Cannot slice a multi-plane DirectedObjectRelationshipPayload "
                    "without slice_indices."
                )
            return self
        pairs = tuple(
            (source_id, target_id)
            for source_id, target_id, relationship_slice_index in zip(
                self.source_ids,
                self.target_ids,
                self.slice_indices,
                strict=True,
            )
            if relationship_slice_index == int(slice_index)
        )
        return type(self)(
            source_ids=tuple(source_id for source_id, _target_id in pairs),
            target_ids=tuple(target_id for _source_id, target_id in pairs),
            slice_count=1,
        )


class ObjectRelationshipPayloadKernel(ABC):
    """Kernel contract used by semantic relationship payload policies."""

    def dominant_parent_ids_by_child(
        self, parent_array: Any, child_array: Any, context_array: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return child ids with their dominant parent ids by positive overlap."""
        children = np.asarray(child_array, dtype=np.int64)
        parents = np.asarray(parent_array, dtype=np.int64)
        context = np.asarray(context_array, dtype=np.int64)
        child_ids = np.unique(children[children > 0])
        if child_ids.size == 0:
            empty = np.zeros(0, dtype=np.int64)
            return (empty, empty)
        max_parent = int(np.max(parents)) if parents.size else 0
        parent_ids = np.zeros(child_ids.size, dtype=np.int64)
        valid = (context > 0) & (parents > 0)
        if not np.any(valid) or max_parent <= 0:
            return (child_ids, parent_ids)
        stride = max_parent + 1
        pair_keys = context[valid] * stride + parents[valid]
        counts = np.bincount(pair_keys)
        child_to_index = np.full(int(child_ids[-1]) + 1, -1, dtype=np.int64)
        child_to_index[child_ids] = np.arange(child_ids.size, dtype=np.int64)
        nonzero_keys = np.flatnonzero(counts)
        best_counts = np.zeros(child_ids.size, dtype=np.int64)
        for key in nonzero_keys:
            child_id = key // stride
            if child_id >= child_to_index.size:
                continue
            output_index = child_to_index[child_id]
            if output_index < 0:
                continue
            count = counts[key]
            parent_id = key % stride
            if count > best_counts[output_index]:
                best_counts[output_index] = count
                parent_ids[output_index] = parent_id
        return (child_ids, parent_ids)

    @abstractmethod
    def relate_children_to_parents(
        self, parent_labels: np.ndarray, child_labels: np.ndarray, child_count: int
    ) -> np.ndarray:
        """Assign each child label id to its dominant parent label id."""

    @abstractmethod
    def relate_sparse_ijv_children_to_parents(
        self,
        parent_rows: np.ndarray,
        child_rows: np.ndarray,
        child_count: int,
        parent_count: int,
    ) -> np.ndarray:
        """Assign sparse IJV child ids to parent ids."""


class DefaultObjectRelationshipPayloadKernel(ObjectRelationshipPayloadKernel):
    """Core NumPy relationship kernel used outside backend-specific execution."""

    def relate_children_to_parents(
        self, parent_labels: np.ndarray, child_labels: np.ndarray, child_count: int
    ) -> np.ndarray:
        del child_count
        child_ids, parent_ids = self.dominant_parent_ids_by_child(
            parent_labels, child_labels, child_labels
        )
        parents_of = np.zeros(
            int(child_labels.max()) if child_labels.size else 0, dtype=np.int32
        )
        for child_id, parent_id in zip(child_ids, parent_ids, strict=True):
            if 0 < child_id <= parents_of.size:
                parents_of[int(child_id) - 1] = int(parent_id)
        return parents_of

    def relate_sparse_ijv_children_to_parents(
        self,
        parent_rows: np.ndarray,
        child_rows: np.ndarray,
        child_count: int,
        parent_count: int,
    ) -> np.ndarray:
        del parent_count
        parents_of = np.zeros(child_count, dtype=np.int32)
        if parent_rows.size == 0 or child_rows.size == 0:
            return parents_of
        parent_by_yx = {
            (int(row[0]), int(row[1])): int(row[2])
            for row in np.asarray(parent_rows, dtype=np.int64)
            if int(row[2]) > 0
        }
        votes: dict[tuple[int, int], int] = {}
        for row in np.asarray(child_rows, dtype=np.int64):
            child_id = int(row[2])
            if child_id <= 0:
                continue
            parent_id = parent_by_yx.get((int(row[0]), int(row[1])), 0)
            if parent_id <= 0:
                continue
            votes[child_id, parent_id] = votes.get((child_id, parent_id), 0) + 1
        for child_id in range(1, child_count + 1):
            child_votes = tuple(
                (
                    (parent_id, count)
                    for (candidate_child, parent_id), count in votes.items()
                    if candidate_child == child_id
                )
            )
            if child_votes:
                parents_of[child_id - 1] = min(
                    child_votes, key=lambda item: (-item[1], item[0])
                )[0]
        return parents_of


DEFAULT_OBJECT_RELATIONSHIP_PAYLOAD_KERNEL = DefaultObjectRelationshipPayloadKernel()


@dataclass(frozen=True, slots=True)
class ObjectRelationshipPayloadRequest:
    """Semantic request for deriving parent-child payloads from label values."""

    parent_labels: ObjectLabelMeasurementSource
    child_labels: ObjectLabelMeasurementSource
    kernel: ObjectRelationshipPayloadKernel = DEFAULT_OBJECT_RELATIONSHIP_PAYLOAD_KERNEL


class ObjectRelationshipPayloadStrategy(
    MostDerivedContextStrategyMixin[ObjectRelationshipPayloadRequest], ABC
):
    """Derive parent-child payloads by nominal object-label representation."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[str | None] = None

    @abstractmethod
    def matches(self, context: ObjectRelationshipPayloadRequest) -> bool:
        """Return whether this strategy owns the label representation pair."""

    @abstractmethod
    def payload(
        self, context: ObjectRelationshipPayloadRequest
    ) -> DirectedObjectRelationshipPayload:
        """Return parent-child ids for the strategy's representation contract."""

    @staticmethod
    def related_payload_from_parents_of(
        parents_of: np.ndarray, child_ids: np.ndarray
    ) -> DirectedObjectRelationshipPayload:
        parent_ids: list[int] = []
        related_child_ids: list[int] = []
        for child_id in child_ids:
            if 0 < child_id <= len(parents_of):
                parent_id = int(parents_of[child_id - 1])
                if parent_id > 0:
                    parent_ids.append(parent_id)
                    related_child_ids.append(int(child_id))
        return DirectedObjectRelationshipPayload(
            source_ids=tuple(parent_ids), target_ids=tuple(related_child_ids)
        )

    @staticmethod
    def related_payload_from_dense_parent_vector(
        parents_of: np.ndarray,
    ) -> DirectedObjectRelationshipPayload:
        """Return related dense child ids directly from a 1-indexed parent vector."""
        parent_vector = np.asarray(parents_of, dtype=np.int64)
        related_child_indexes = np.flatnonzero(parent_vector > 0)
        return DirectedObjectRelationshipPayload(
            source_ids=tuple(
                (int(parent_vector[index]) for index in related_child_indexes)
            ),
            target_ids=tuple((int(index + 1) for index in related_child_indexes)),
        )


class DenseObjectRelationshipPayloadStrategy(ObjectRelationshipPayloadStrategy):
    """Dense label images use maximum positive-pixel overlap."""

    strategy_key = "dense"

    def matches(self, context: ObjectRelationshipPayloadRequest) -> bool:
        del context
        return True

    def payload(
        self, context: ObjectRelationshipPayloadRequest
    ) -> DirectedObjectRelationshipPayload:
        from openhcs.core.runtime_object_labels import (
            object_label_dense_array,
        )

        slice_count = self.relationship_slice_count(context)
        if slice_count is not None:
            (parent_stack, child_stack), _adapters = (
                SourceSpatialDomainAdapter.aligned_values(
                    (context.parent_labels, context.child_labels)
                )
            )
            if (
                parent_stack.ndim != 3
                or child_stack.ndim != 3
                or parent_stack.shape[0] != slice_count
                or child_stack.shape[0] != slice_count
            ):
                raise ValueError(
                    "Plane-scoped object-label relationships require dense stacks "
                    f"with exactly {slice_count} declared planes; got "
                    f"{parent_stack.shape!r} and {child_stack.shape!r}."
                )
            return self.stack_payload(
                context,
                parent_stack,
                child_stack,
                slice_count=slice_count,
            )
        aligned_values, _adapters = SourceSpatialDomainAdapter.aligned_values(
            (context.parent_labels, context.child_labels)
        )
        parent_array, child_array = (
            object_label_dense_array(labels, dtype=np.int32)
            for labels in aligned_values
        )
        child_count = int(child_array.max()) if child_array.size else 0
        if child_count <= 0:
            return DirectedObjectRelationshipPayload(source_ids=(), target_ids=())
        parents_of = context.kernel.relate_children_to_parents(
            parent_array, child_array, child_count
        )
        return self.related_payload_from_dense_parent_vector(parents_of)

    def relationship_slice_count(
        self, context: ObjectRelationshipPayloadRequest
    ) -> int | None:
        """Return the plane count for plane-scoped label relationships."""

        domains = tuple(
            (
                ObjectLabelDomainMetadataStrategy.for_value(labels).object_label_domain(
                    labels
                )
                for labels in (context.parent_labels, context.child_labels)
            )
        )
        if (
            ObjectLabelDomainScope.common((domain.scope for domain in domains))
            is not ObjectLabelDomainScope.PLANE
        ):
            return None
        plane_counts = tuple(
            labels.declared_plane_count()
            for labels in (context.parent_labels, context.child_labels)
        )
        if any(plane_count is None for plane_count in plane_counts):
            raise ValueError(
                "Plane-scoped object relationships require both endpoints to "
                "declare nominal label-plane stacks."
            )
        if len(set(plane_counts)) != 1:
            raise ValueError(
                "Plane-scoped object relationship endpoints must declare the same "
                f"plane count, got {plane_counts!r}."
            )
        return plane_counts[0]

    def stack_payload(
        self,
        context: ObjectRelationshipPayloadRequest,
        parent_stack: np.ndarray,
        child_stack: np.ndarray,
        *,
        slice_count: int,
    ) -> DirectedObjectRelationshipPayload:
        """Return parent-child ids with explicit runtime-slice identity."""
        parent_ids: list[int] = []
        child_ids: list[int] = []
        slice_indices: list[int] = []
        for slice_index, (parent_plane, child_plane) in enumerate(
            zip(parent_stack, child_stack, strict=True)
        ):
            child_count = int(child_plane.max()) if child_plane.size else 0
            if child_count <= 0:
                continue
            parents_of = context.kernel.relate_children_to_parents(
                parent_plane, child_plane, child_count
            )
            payload = self.related_payload_from_dense_parent_vector(parents_of)
            parent_ids.extend(payload.source_ids)
            child_ids.extend(payload.target_ids)
            slice_indices.extend((slice_index for _child_id in payload.target_ids))
        return DirectedObjectRelationshipPayload(
            source_ids=tuple(parent_ids),
            target_ids=tuple(child_ids),
            slice_indices=tuple(slice_indices),
            slice_count=slice_count,
        )


class SparseIJVObjectRelationshipPayloadStrategy(
    DenseObjectRelationshipPayloadStrategy
):
    """Sparse IJV labels derive parent-child ids through sparse rows."""

    strategy_key = "sparse_ijv"

    def matches(self, context: ObjectRelationshipPayloadRequest) -> bool:
        from openhcs.core.runtime_object_labels import (
            object_label_storage_is_sparse_ijv,
        )

        return any(
            (
                object_label_storage_is_sparse_ijv(labels)
                for labels in (context.parent_labels, context.child_labels)
            )
        )

    def payload(
        self, context: ObjectRelationshipPayloadRequest
    ) -> DirectedObjectRelationshipPayload:
        from openhcs.core.runtime_object_labels import (
            object_label_sparse_ijv_rows,
        )

        parent_rows = object_label_sparse_ijv_rows(context.parent_labels)
        child_rows = object_label_sparse_ijv_rows(context.child_labels)
        parent_array = parent_rows.as_yx_label_array()
        child_array = child_rows.as_yx_label_array()
        parent_count = self.label_count(parent_array, parent_rows)
        child_count = self.label_count(child_array, child_rows)
        if parent_count <= 0 or child_count <= 0:
            return DirectedObjectRelationshipPayload(source_ids=(), target_ids=())
        parents_of = context.kernel.relate_sparse_ijv_children_to_parents(
            np.asarray(parent_array, dtype=np.int64),
            np.asarray(child_array, dtype=np.int64),
            child_count,
            parent_count,
        )
        return self.related_payload_from_parents_of(
            parents_of, self.present_sparse_child_ids(child_array, child_rows)
        )

    @staticmethod
    def label_count(array: np.ndarray, rows: SparseIJVLabelRows) -> int:
        if array.size == 0:
            return 0
        return int(np.max(array[:, rows.label_column]))

    @staticmethod
    def present_sparse_child_ids(
        child_array: np.ndarray, child_rows: SparseIJVLabelRows
    ) -> np.ndarray:
        if child_array.size == 0:
            return np.empty(0, dtype=np.int32)
        return np.unique(child_array[:, child_rows.label_column]).astype(
            np.int32, copy=False
        )


@dataclass(frozen=True, slots=True)
class ObjectInstanceKey:
    """Typed identity for one object label inside an optional measurement plane."""

    object_id: int
    slice_index: int | None = None

    def __post_init__(self) -> None:
        object_id = int(self.object_id)
        if object_id <= 0:
            raise ValueError("ObjectInstanceKey.object_id must be positive.")
        slice_index = None if self.slice_index is None else int(self.slice_index)
        if slice_index is not None and slice_index < 0:
            raise ValueError("ObjectInstanceKey.slice_index cannot be negative.")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "slice_index", slice_index)

    @classmethod
    def from_measurement_row(
        cls,
        row: Mapping[str, Any],
        object_id: int,
        *,
        slice_index_field: MeasurementRowAxisField = MeasurementRowAxisField.SLICE_INDEX,
    ) -> "ObjectInstanceKey":
        """Build object identity from the row's runtime slice axis."""
        raw_slice_index = row.get(slice_index_field.value)
        if raw_slice_index is not None and str(raw_slice_index).strip() != "":
            return cls(object_id, slice_index=int(raw_slice_index))
        return cls(object_id)

    @classmethod
    def domain(
        cls, object_ids: Iterable[int], *, slice_index: int | None = None
    ) -> tuple["ObjectInstanceKey", ...]:
        """Return typed object identities for one optional measurement plane."""
        return tuple(
            (cls(object_id, slice_index=slice_index) for object_id in object_ids)
        )


@dataclass(frozen=True, slots=True)
class ObjectInstanceRelationship:
    """Parent-child relationships keyed by typed object instance identity."""

    source_keys: tuple[ObjectInstanceKey, ...]
    target_keys: tuple[ObjectInstanceKey, ...]
    slice_count: int | None = None

    @classmethod
    def from_id_columns(
        cls,
        source_ids: Iterable[int],
        target_ids: Iterable[int],
        *,
        slice_indices: Iterable[int] = (),
        slice_count: int | None = None,
    ) -> "ObjectInstanceRelationship":
        """Build typed relationship identity from id columns plus optional slices."""
        source_id_tuple = tuple((int(value) for value in source_ids))
        target_id_tuple = tuple((int(value) for value in target_ids))
        if len(source_id_tuple) != len(target_id_tuple):
            raise ValueError(
                f"ObjectInstanceRelationship source_ids and target_ids must have equal length, got {len(source_id_tuple)} and {len(target_id_tuple)}."
            )
        slice_index_tuple = tuple((int(value) for value in slice_indices))
        if slice_index_tuple and len(slice_index_tuple) != len(source_id_tuple):
            raise ValueError(
                f"ObjectInstanceRelationship slice_indices must be empty or match id columns, got {len(slice_index_tuple)} for {len(source_id_tuple)}."
            )
        resolved_slice_count = None if slice_count is None else int(slice_count)
        if resolved_slice_count is not None and resolved_slice_count < 0:
            raise ValueError(
                "ObjectInstanceRelationship.slice_count cannot be negative."
            )
        source_keys: list[ObjectInstanceKey] = []
        target_keys: list[ObjectInstanceKey] = []
        for index, (source_id, target_id) in enumerate(
            zip(source_id_tuple, target_id_tuple, strict=True)
        ):
            slice_index = slice_index_tuple[index] if slice_index_tuple else None
            if source_id > 0 and target_id > 0:
                source_keys.append(ObjectInstanceKey(source_id, slice_index))
                target_keys.append(ObjectInstanceKey(target_id, slice_index))
        return cls(
            source_keys=tuple(source_keys),
            target_keys=tuple(target_keys),
            slice_count=resolved_slice_count,
        )

    def source_domain(
        self, object_count: int = 0, *, declared_keys: Iterable[ObjectInstanceKey] = ()
    ) -> tuple[ObjectInstanceKey, ...]:
        """Return all source identities represented by this relationship domain."""
        return self._domain(
            self.source_keys,
            object_count=object_count,
            slice_count=self.slice_count,
            declared_keys=declared_keys,
        )

    def target_domain(
        self, object_count: int = 0, *, declared_keys: Iterable[ObjectInstanceKey] = ()
    ) -> tuple[ObjectInstanceKey, ...]:
        """Return all target identities represented by this relationship domain."""
        return self._domain(
            self.target_keys,
            object_count=object_count,
            slice_count=self.slice_count,
            declared_keys=declared_keys,
        )

    def child_keys_by_parent(
        self,
        *,
        source_object_count: int = 0,
        declared_source_keys: Iterable[ObjectInstanceKey] = (),
    ) -> dict[ObjectInstanceKey, tuple[ObjectInstanceKey, ...]]:
        """Return target identities grouped by source identity."""
        children: dict[ObjectInstanceKey, list[ObjectInstanceKey]] = {
            source_key: []
            for source_key in self.source_domain(
                source_object_count, declared_keys=declared_source_keys
            )
        }
        for source_key, target_key in zip(
            self.source_keys, self.target_keys, strict=True
        ):
            children.setdefault(source_key, []).append(target_key)
        return {
            source_key: tuple(
                sorted(
                    child_keys,
                    key=lambda key: (
                        key.slice_index is None,
                        key.slice_index or -1,
                        key.object_id,
                    ),
                )
            )
            for source_key, child_keys in sorted(
                children.items(),
                key=lambda item: (
                    item[0].slice_index is None,
                    item[0].slice_index or -1,
                    item[0].object_id,
                ),
            )
        }

    def parent_key_by_child(self) -> dict[ObjectInstanceKey, ObjectInstanceKey]:
        """Return source identity for each target identity."""
        return dict(zip(self.target_keys, self.source_keys, strict=True))

    @staticmethod
    def _domain(
        keys: tuple[ObjectInstanceKey, ...],
        *,
        object_count: int,
        slice_count: int | None,
        declared_keys: Iterable[ObjectInstanceKey],
    ) -> tuple[ObjectInstanceKey, ...]:
        declared_key_tuple = tuple(declared_keys)
        if declared_key_tuple:
            merged = {key: None for key in declared_key_tuple}
            merged.update({key: None for key in keys})
            return tuple(
                sorted(
                    merged,
                    key=lambda key: (
                        key.slice_index is None,
                        key.slice_index or -1,
                        key.object_id,
                    ),
                )
            )
        if object_count <= 0 and keys:
            return tuple(
                sorted(
                    dict.fromkeys(keys),
                    key=lambda key: (
                        key.slice_index is None,
                        key.slice_index or -1,
                        key.object_id,
                    ),
                )
            )
        max_object_id = max((key.object_id for key in keys), default=0)
        max_object_id = max(max_object_id, int(object_count))
        if max_object_id <= 0:
            return ()
        slice_indexes = (
            tuple(range(slice_count))
            if slice_count is not None and slice_count > 1
            else ()
        ) or tuple(
            dict.fromkeys(
                (key.slice_index for key in keys if key.slice_index is not None)
            )
        )
        if not slice_indexes:
            return ObjectInstanceKey.domain(range(1, max_object_id + 1))
        return tuple(
            (
                key
                for slice_index in slice_indexes
                for key in ObjectInstanceKey.domain(
                    range(1, max_object_id + 1), slice_index=slice_index
                )
            )
        )


@dataclass(frozen=True, slots=True)
class ObjectLabelInstanceDomains:
    """Typed object-instance domains keyed by object-label artifact name."""

    domains_by_name: Mapping[str, tuple[ObjectInstanceKey, ...]]

    @classmethod
    def from_named_plane_domains(
        cls, named_plane_domains: Iterable[tuple[str, tuple[tuple[int, ...], ...]]]
    ) -> "ObjectLabelInstanceDomains":
        """Build domains from per-plane object-id domains."""
        domains: dict[str, dict[ObjectInstanceKey, None]] = {}
        for object_name, plane_domains in named_plane_domains:
            slice_indexes = (
                (None,) if len(plane_domains) <= 1 else tuple(range(len(plane_domains)))
            )
            domain = domains.setdefault(str(object_name), {})
            for slice_index, object_ids in zip(
                slice_indexes, plane_domains, strict=True
            ):
                for object_id in object_ids:
                    domain[ObjectInstanceKey(object_id, slice_index=slice_index)] = None
        return cls(
            {
                object_name: cls._ordered_keys(domain)
                for object_name, domain in domains.items()
            }
        )

    def for_name(self, object_name: str) -> tuple[ObjectInstanceKey, ...]:
        """Return the typed object-instance domain for one object-label name."""
        return self.domains_by_name.get(object_name, ())

    @staticmethod
    def _ordered_keys(
        keys: Mapping[ObjectInstanceKey, None],
    ) -> tuple[ObjectInstanceKey, ...]:
        return tuple(
            sorted(
                keys,
                key=lambda key: (
                    key.slice_index is None,
                    key.slice_index or -1,
                    key.object_id,
                ),
            )
        )


def object_label_parent_child_payload(
    parent_labels: Any,
    child_labels: Any,
    *,
    child_region_labels: Any | None = None,
    kernel: ObjectRelationshipPayloadKernel = DEFAULT_OBJECT_RELATIONSHIP_PAYLOAD_KERNEL,
) -> DirectedObjectRelationshipPayload:
    """Derive parent-child ids from nominal object-label representations.

    ``child_region_labels`` lets callers use one label image to enumerate child
    ids while selecting the pixels that define each child's parent context.
    """
    if child_region_labels is None:
        request = ObjectRelationshipPayloadRequest(
            parent_labels=parent_labels, child_labels=child_labels, kernel=kernel
        )
        return ObjectRelationshipPayloadStrategy.for_context(request).payload(request)
    else:
        (parent_array, child_array, context_array), _adapters = (
            SourceSpatialDomainAdapter.aligned_values(
                (parent_labels, child_labels, child_region_labels)
            )
        )
    child_ids_array, parent_ids_array = kernel.dominant_parent_ids_by_child(
        parent_array, child_array, context_array
    )
    return DirectedObjectRelationshipPayload(
        source_ids=tuple((int(parent_id) for parent_id in parent_ids_array)),
        target_ids=tuple((int(child_id) for child_id in child_ids_array)),
    )


def object_label_identity_lineage_payload(
    parent_labels: Any, child_labels: Any
) -> DirectedObjectRelationshipPayload:
    """Derive lineage from preserved IDs for a declared geometry-changing transform."""
    parent_ids = set(dense_object_label_id_domain(parent_labels))
    child_ids = tuple(dense_object_label_id_domain(child_labels))
    related_ids = tuple(object_id for object_id in child_ids if object_id in parent_ids)
    return DirectedObjectRelationshipPayload(
        source_ids=related_ids,
        target_ids=related_ids,
    )


def _require_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")
