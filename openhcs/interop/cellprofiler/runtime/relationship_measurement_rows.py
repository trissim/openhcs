"""Relationship measurement row authorities for CellProfiler runtime outputs."""

from __future__ import annotations
from dataclasses import dataclass, field, replace
import time
from typing import TYPE_CHECKING

from openhcs.core.artifacts import (
    ArtifactSpec,
)
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    ObjectReferenceFeatureMarker,
    RuntimeMeasurementFeatureDeclaration,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomainScope,
)
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
)
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
    ObjectRelationshipDeclaration,
)
from openhcs.core.runtime_image_values import (
    image_payload_metadata,
)
from openhcs.interop.cellprofiler.measurement_lookup import (
    CellProfilerMeasurementFeature,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.output_record_request import (
        CellProfilerOutputRecordRequest,
    )
class DirectParentReferenceFeatureMarker(ObjectReferenceFeatureMarker):
    """Semantic marker for a child's direct parent-object reference."""


@dataclass(frozen=True, slots=True)
class DirectParentReferenceMeasurementFeature:
    """Nominal identity encoded by a ``Parent_<object>`` measurement name."""

    parent_object_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.parent_object_name, str) or not self.parent_object_name:
            raise ValueError("Direct parent-reference object name cannot be empty.")


class DirectParentReferenceFeatureDeclaration(RuntimeMeasurementFeatureDeclaration):
    """Parse and render direct parent references at their row-production owner."""

    declaration_key = "direct_parent_reference"
    semantic_marker_types = (DirectParentReferenceFeatureMarker,)
    prefix = "Parent_"

    @classmethod
    def from_feature_name(
        cls,
        feature_name: str,
    ) -> DirectParentReferenceMeasurementFeature | None:
        if not feature_name.startswith(cls.prefix):
            return None
        parent_object_name = feature_name[len(cls.prefix) :]
        if not parent_object_name:
            return None
        return DirectParentReferenceMeasurementFeature(parent_object_name)

    @classmethod
    def feature_name(cls, identity: object) -> str:
        if not isinstance(identity, DirectParentReferenceMeasurementFeature):
            raise TypeError(
                f"{cls.__name__}.feature_name requires "
                "DirectParentReferenceMeasurementFeature."
            )
        return f"{cls.prefix}{identity.parent_object_name}"


@dataclass(frozen=True, slots=True)
class RelationshipMeasurementRows:
    """Project parent-child relationship payloads into CP object measurement rows."""

    request: CellProfilerOutputRecordRequest
    _object_number_cache: dict[
        tuple[object, int | None, int | None],
        dict[int, int],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def for_request(
        cls, request: CellProfilerOutputRecordRequest
    ) -> "RelationshipMeasurementRows":
        from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

        module_type = CellProfilerModule.require_callable_contract_owner(
            request.callable_contract
        )
        rows = module_type.relationship_measurement_rows(request)
        if not isinstance(rows, RelationshipMeasurementRows):
            raise TypeError(
                f"{module_type.__name__}.relationship_measurement_rows must return RelationshipMeasurementRows."
            )
        return rows

    def rows(self) -> ColumnarRows:
        rows_started_at = time.perf_counter()
        callable_contract = self.request.callable_contract
        declared_artifacts = callable_contract.artifact_specs
        row_batches: list[ColumnarRows] = []
        for _relationship_spec, declaration, payload in self.output_entries():
            parent_spec = declared_artifacts.by_ref(declaration.source)
            child_spec = declared_artifacts.by_ref(declaration.target)
            if parent_spec is None or child_spec is None:
                raise ValueError(
                    f"Callable {callable_contract.function_name!r} relationship "
                    "declaration references endpoints outside its artifact "
                    f"contract: source={declaration.source!r}, "
                    f"target={declaration.target!r}."
                )
            row_batches.append(
                self.child_count_rows(
                    parent_spec=parent_spec,
                    child_spec=child_spec,
                    payload=payload,
                )
            )
            row_batches.append(
                self.parent_rows(
                    parent_spec=parent_spec,
                    child_spec=child_spec,
                    payload=payload,
                )
            )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "relationship_measurement_rows",
            time.perf_counter() - rows_started_at,
            function=callable_contract.function_name,
            rows=sum(row_batch.row_count() for row_batch in row_batches),
        )
        return ConcatenatedColumnarRows(tuple(row_batches))

    def output_entries(
        self,
    ) -> tuple[
        tuple[ArtifactSpec, ObjectRelationshipDeclaration, ObjectRelationship], ...
    ]:
        entries: list[
            tuple[ArtifactSpec, ObjectRelationshipDeclaration, ObjectRelationship]
        ] = []
        for spec, relation in self.request.callable_contract.artifact_outputs.relation_refs(
            ObjectRelationshipDeclaration
        ):
            declaration = relation
            if not declaration.projects_parent_child_measurements():
                continue
            entries.extend(
                (spec, declaration, relationship)
                for relationship in self.canonical_relationships(
                    spec,
                    declaration,
                    self.request.artifact_output_value(spec),
                )
            )
        return tuple(entries)

    def canonical_relationships(
        self,
        spec: ArtifactSpec,
        declaration: ObjectRelationshipDeclaration,
        value: RuntimeCallableArgument,
    ) -> tuple[ObjectRelationship, ...]:
        """Normalize one exact output without querying another runtime authority."""

        match value:
            case ObjectRelationship() as relationship:
                if relationship.declaration != declaration:
                    raise ValueError(
                        f"Relationship output {spec.ref()!r} payload declaration "
                        "differs from its callable artifact relation."
                    )
                return (relationship,)
            case DirectedObjectRelationshipPayload() as payload:
                return (
                    ObjectRelationship.from_payload(
                        name=spec.name,
                        declaration=declaration,
                        payload=payload,
                        source_provenance=image_payload_metadata(
                            self.request.source.payload
                        ).source_provenance,
                    ),
                )
            case RuntimeSliceAlignedValues(slices=slices):
                relationships: list[ObjectRelationship] = []
                slice_count = len(slices)
                for slice_index, slice_value in enumerate(slices):
                    for relationship in self.canonical_relationships(
                        spec,
                        declaration,
                        slice_value,
                    ):
                        if relationship.payload.slice_count not in (None, 1):
                            raise ValueError(
                                "RuntimeSliceAlignedValues relationship entries must "
                                "be payload-local before outer-axis projection."
                            )
                        local_indices = tuple(relationship.payload.slice_indices)
                        if local_indices and any(index != 0 for index in local_indices):
                            raise ValueError(
                                "RuntimeSliceAlignedValues relationship entries must "
                                "use payload-local slice_index 0."
                            )
                        relationships.append(
                            replace(
                                relationship,
                                payload=replace(
                                    relationship.payload,
                                    slice_indices=tuple(
                                        slice_index
                                        for _source_id in relationship.payload.source_ids
                                    ),
                                    slice_count=slice_count,
                                ),
                            )
                        )
                return tuple(relationships)
            case _:
                raise TypeError(
                    f"Callable {self.request.callable_contract.function_name!r} "
                    f"relationship output {spec.name!r} "
                    "must be ObjectRelationship or DirectedObjectRelationshipPayload, "
                    f"got {type(value).__name__}."
                )

    def child_count_rows(
        self,
        *,
        parent_spec: ArtifactSpec,
        child_spec: ArtifactSpec,
        payload: ObjectRelationship,
    ) -> ColumnarRows:
        rows_started_at = time.perf_counter()
        sliced_pairs = payload.payload.runtime_slice_pairs()
        if sliced_pairs is not None:
            slice_count = len(sliced_pairs)
            row_batches: list[ColumnarRows] = []
            for slice_index, pairs in sliced_pairs:
                parent_slice_index, _child_slice_index = (
                    payload.declaration.endpoint_runtime_slice_indices(
                        slice_index,
                        slice_count,
                    )
                )
                if parent_slice_index is None:
                    if pairs:
                        raise ValueError(
                            f"Relationship source endpoint {parent_spec.name!r} "
                            f"is outside the declared {slice_count}-slice domain "
                            f"for payload slice {slice_index}."
                        )
                    continue
                related_parent_ids = tuple(
                    (parent_id for parent_id, _child_id in pairs)
                )
                row_batches.append(
                    self.child_count_rows_for_ids(
                        parent_spec=parent_spec,
                        child_spec=child_spec,
                        related_parent_ids=related_parent_ids,
                        parent_slice_index=parent_slice_index,
                        slice_count=slice_count,
                    )
                )
            result = ConcatenatedColumnarRows(tuple(row_batches))
        else:
            result = self.child_count_rows_for_ids(
                parent_spec=parent_spec,
                child_spec=child_spec,
                related_parent_ids=tuple(
                    (int(parent_id) for parent_id in payload.payload.source_ids)
                ),
                parent_slice_index=None,
            )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "relationship_child_count_rows",
            time.perf_counter() - rows_started_at,
            function=self.request.callable_contract.function_name,
            parent=parent_spec.name,
            child=child_spec.name,
            rows=result.row_count(),
        )
        return result

    def child_count_rows_for_ids(
        self,
        *,
        parent_spec: ArtifactSpec,
        child_spec: ArtifactSpec,
        related_parent_ids: tuple[int, ...],
        parent_slice_index: int | None,
        slice_count: int | None = None,
    ) -> MeasurementSparseColumnarRows:
        related_parent_ids = tuple((int(parent_id) for parent_id in related_parent_ids))
        parent_numbers = self.object_numbers_by_label_id(
            parent_spec,
            slice_index=parent_slice_index,
            slice_count=slice_count,
        )
        counts = {parent_number: 0 for parent_number in parent_numbers.values()}
        for parent_id in related_parent_ids:
            if parent_id not in parent_numbers:
                raise ValueError(
                    f"Relationship parent ID {parent_id} is outside the declared "
                    f"domain for {parent_spec.name!r}."
                )
            parent_number = parent_numbers[parent_id]
            counts[parent_number] = counts[parent_number] + 1
        feature_name = CellProfilerMeasurementFeature.child_count(child_spec.name).name
        rows = tuple(
            {
                MeasurementRowAxisField.OBJECT_NAME.value: parent_spec.name,
                MeasurementRowAxisField.OBJECT_LABEL.value: parent_id,
                **(
                    {}
                    if parent_slice_index is None
                    else {MeasurementRowAxisField.SLICE_INDEX.value: parent_slice_index}
                ),
                feature_name: count,
            }
            for parent_id, count in counts.items()
        )
        return MeasurementSparseColumnarRows.from_rows(
            rows,
            fields=(
                FieldSpec(MeasurementRowAxisField.OBJECT_NAME.value, str),
                FieldSpec(MeasurementRowAxisField.OBJECT_LABEL.value, int),
                *(
                    ()
                    if parent_slice_index is None
                    else (FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),)
                ),
                FieldSpec(feature_name, int),
            ),
        )

    def parent_rows(
        self,
        *,
        parent_spec: ArtifactSpec,
        child_spec: ArtifactSpec,
        payload: ObjectRelationship,
    ) -> ColumnarRows:
        rows_started_at = time.perf_counter()
        sliced_pairs = payload.payload.runtime_slice_pairs()
        if sliced_pairs is not None:
            slice_count = len(sliced_pairs)
            row_batches: list[ColumnarRows] = []
            for slice_index, pairs in sliced_pairs:
                parent_slice_index, child_slice_index = (
                    payload.declaration.endpoint_runtime_slice_indices(
                        slice_index,
                        slice_count,
                    )
                )
                if child_slice_index is None:
                    if pairs:
                        raise ValueError(
                            f"Relationship target endpoint {child_spec.name!r} "
                            f"is outside the declared {slice_count}-slice domain "
                            f"for payload slice {slice_index}."
                        )
                    continue
                if parent_slice_index is None and pairs:
                    raise ValueError(
                        f"Relationship source endpoint {parent_spec.name!r} is "
                        f"outside the declared {slice_count}-slice domain for "
                        f"payload slice {slice_index}."
                    )
                row_batches.append(
                    self.parent_rows_for_pairs(
                        parent_spec=parent_spec,
                        child_spec=child_spec,
                        pairs=pairs,
                        parent_slice_index=parent_slice_index,
                        child_slice_index=child_slice_index,
                        slice_count=slice_count,
                    )
                )
            result = ConcatenatedColumnarRows(tuple(row_batches))
        else:
            result = self.parent_rows_for_pairs(
                parent_spec=parent_spec,
                child_spec=child_spec,
                pairs=tuple(
                    (
                        (int(parent_id), int(child_id))
                        for parent_id, child_id in zip(
                            payload.payload.source_ids,
                            payload.payload.target_ids,
                            strict=True,
                        )
                    )
                ),
                parent_slice_index=None,
                child_slice_index=None,
            )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "relationship_parent_rows",
            time.perf_counter() - rows_started_at,
            function=self.request.callable_contract.function_name,
            parent=parent_spec.name,
            child=child_spec.name,
            rows=result.row_count(),
        )
        return result

    def parent_rows_for_pairs(
        self,
        *,
        parent_spec: ArtifactSpec,
        child_spec: ArtifactSpec,
        pairs: tuple[tuple[int, int], ...],
        parent_slice_index: int | None,
        child_slice_index: int | None,
        slice_count: int | None = None,
    ) -> MeasurementSparseColumnarRows:
        parent_numbers = (
            {}
            if parent_slice_index is None and not pairs
            else self.object_numbers_by_label_id(
                parent_spec,
                slice_index=parent_slice_index,
                slice_count=slice_count,
            )
        )
        child_numbers = self.object_numbers_by_label_id(
            child_spec,
            slice_index=child_slice_index,
            slice_count=slice_count,
        )
        parent_by_child = {}
        for parent_id, child_id in pairs:
            parent_id = int(parent_id)
            child_id = int(child_id)
            if parent_id not in parent_numbers:
                raise ValueError(
                    f"Relationship parent ID {parent_id} is outside the declared "
                    f"domain for {parent_spec.name!r}."
                )
            if child_id not in child_numbers:
                raise ValueError(
                    f"Relationship child ID {child_id} is outside the declared "
                    f"domain for {child_spec.name!r}."
                )
            child_number = child_numbers[child_id]
            parent_number = parent_numbers[parent_id]
            if (
                child_number in parent_by_child
                and parent_by_child[child_number] != parent_number
            ):
                raise ValueError(
                    f"Relationship child ID {child_id} declares multiple parents."
                )
            parent_by_child[child_number] = parent_number
        feature_name = DirectParentReferenceFeatureDeclaration.feature_name(
            DirectParentReferenceMeasurementFeature(parent_spec.name)
        )
        rows = tuple(
            {
                MeasurementRowAxisField.OBJECT_NAME.value: child_spec.name,
                MeasurementRowAxisField.OBJECT_LABEL.value: child_id,
                **(
                    {}
                    if child_slice_index is None
                    else {MeasurementRowAxisField.SLICE_INDEX.value: child_slice_index}
                ),
                feature_name: (
                    parent_by_child[child_id] if child_id in parent_by_child else 0
                ),
            }
            for child_id in child_numbers.values()
        )
        return MeasurementSparseColumnarRows.from_rows(
            rows,
            fields=(
                FieldSpec(MeasurementRowAxisField.OBJECT_NAME.value, str),
                FieldSpec(MeasurementRowAxisField.OBJECT_LABEL.value, int),
                *(
                    ()
                    if child_slice_index is None
                    else (FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),)
                ),
                FieldSpec(feature_name, int),
            ),
        )

    def object_numbers_by_label_id(
        self,
        spec: ArtifactSpec,
        *,
        slice_index: int | None,
        slice_count: int | None = None,
    ) -> dict[int, int]:
        cache_key = (spec.ref(), slice_index, slice_count)
        cached = self._object_number_cache.get(cache_key)
        if cached is not None:
            return cached
        started_at = time.perf_counter()
        labels = self.object_label_domain_value(
            spec, slice_index=slice_index, slice_count=slice_count
        )
        if not isinstance(labels, ObjectLabelValue):
            raise TypeError(
                f"Relationship endpoint {spec.name!r} requires an ObjectLabelValue, "
                f"got {type(labels).__name__}."
            )
        domain = labels.object_label_domain()
        if domain.scope is not ObjectLabelDomainScope.PAYLOAD:
            raise ValueError(
                f"Relationship endpoint {spec.name!r} must be projected to one "
                "payload-scoped object-ID domain before row materialization."
            )
        label_ids = domain.explicit_id_domain()
        if label_ids is None:
            raise ValueError(
                f"Relationship endpoint {spec.name!r} requires an explicit "
                "object-ID domain."
            )
        numbers = {
            int(label_id): object_number
            for object_number, label_id in enumerate(label_ids, start=1)
        }
        self._object_number_cache[cache_key] = numbers
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "relationship_object_numbers_by_label_id",
            time.perf_counter() - started_at,
            function=self.request.callable_contract.function_name,
            object=spec.name,
            slice_index=slice_index,
            ids=len(numbers),
        )
        return numbers

    def object_label_domain_value(
        self,
        spec: ArtifactSpec,
        *,
        slice_index: int | None = None,
        slice_count: int | None = None,
    ) -> RuntimeCallableArgument:
        labels = self.unprojected_object_labels(spec)
        if slice_index is None:
            return RuntimeSliceProjection.object_label_endpoint(labels)
        context = RuntimeSliceProjection.context_for_value(
            labels,
            slice_index=slice_index,
            slice_count=slice_count,
            source_description=f"object labels {spec.name!r}",
        )
        return RuntimeSliceProjection.object_label_endpoint(labels, context=context)

    def object_labels(
        self,
        spec: ArtifactSpec,
        *,
        slice_index: int | None = None,
        slice_count: int | None = None,
    ) -> RuntimeCallableArgument:
        labels = self.object_label_domain_value(
            spec, slice_index=slice_index, slice_count=slice_count
        )
        if not isinstance(labels, ObjectLabelValue):
            raise TypeError(
                f"Relationship endpoint {spec.name!r} requires an ObjectLabelValue."
            )
        return labels.labels

    def unprojected_object_labels(
        self,
        spec: ArtifactSpec,
    ) -> RuntimeCallableArgument:
        return self.request.artifact_value(spec)
