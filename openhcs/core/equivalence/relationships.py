"""Relationship-derived runtime measurement projection."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar

from metaclass_registry import RegistryFamily, RegistryKeyAttribute
import numpy as np

from openhcs.core.artifacts import ArtifactScope, ArtifactType
from openhcs.core.equivalence.cells import runtime_cell_signature
from openhcs.core.equivalence.keys import (
    RuntimeAggregateFeatureIdentity,
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementSubjectKey,
)
from openhcs.core.equivalence.measurement_facts import (
    RuntimeDirectionalPairMeasurementDerivationContract,
    RuntimeMeasurementFactList,
    RuntimeMeasurementFacts,
    RuntimeMeasurementKeySet as _RuntimeMeasurementKeySet,
    RuntimeRequiredMeasurementKeys,
)
from openhcs.core.equivalence.measurement_requirements import (
    RequiredRuntimeMeasurementProjection,
)
from openhcs.core.equivalence.measurement_rows import (
    RuntimeIndexedRowValues,
    RuntimeCollapsedNumericQualifierCache,
    RuntimeImageNumberOffset,
    RuntimeMeasurementFeatureKeyCache,
    RuntimeMeasurementLongFormKeyCache,
    RuntimeMeasurementPaddingGroupCache,
    RuntimeMeasurementQualifierRenderCache,
    RuntimeMeasurementRequiredKeyIndex,
    RuntimeMeasurementRowMapping,
    RuntimeMeasurementRowSchemaCache,
    RuntimeMeasurementRowSubjectProjection,
    RuntimeMeasurementRowSubjectSchemaCache,
    RuntimeMeasurementWideFeatureIndexCache,
    RuntimeMeasurementWideFeaturePlanCache,
    RuntimeRowProjectionContext,
    runtime_measurement_row_subject_schema,
)
from openhcs.core.equivalence.object_label_measurements import (
    RuntimeObjectLabelInstanceCatalog,
    RuntimeObjectValuesByLabel,
)
from openhcs.core.equivalence.policy import (
    RuntimeEquivalencePolicy,
    RuntimeMeasurementDialect,
    normalize_runtime_identifier,
)
from openhcs.core.equivalence.tables import (
    measurement_table_padding_group,
    measurement_table_spans_multiple_transport_identities,
)
from openhcs.core.registry_strategies import MostDerivedContextStrategyMixin
from openhcs.core.measurement_row_materialization import (
    columnar_row_values,
    iter_measurement_rows,
    measurement_object_label,
    measurement_table_object_id_field,
    measurement_table_object_name,
)
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    MeasurementScope,
    MeasurementStatistic,
    ObjectCoreMeasurementFeature,
    ObjectInstanceKey,
    ObjectInstanceRelationship,
    measurement_row_mapping,
)
from openhcs.core.runtime_stores import StoredRuntimeValue
from openhcs.core.runtime_values import ColumnarRows, MeasurementTable, ObjectRelationship

_ObjectInstanceChildrenByParent = Mapping[
    ObjectInstanceKey,
    tuple[ObjectInstanceKey, ...],
]
_ObjectInstanceAggregateValues = dict[tuple[ObjectInstanceKey, ...], float]


@dataclass(frozen=True, slots=True)
class RuntimeRecordPlaneIdentity:
    """Plane identity contributed by runtime record scope rather than row payload."""

    slice_index: int
    authority: "RuntimeRecordPlaneIdentityAuthority"

    def object_instance_key(
        self,
        row: RuntimeMeasurementRowMapping,
        object_id: int,
        *,
        image_number_offset: RuntimeImageNumberOffset,
    ) -> ObjectInstanceKey:
        if (
            self.authority
            is RuntimeRecordPlaneIdentityAuthority.OVERRIDE_ROW_IDENTITY
        ):
            return ObjectInstanceKey(object_id, slice_index=self.slice_index)
        key = ObjectInstanceKey.from_measurement_row(
            row.row,
            object_id,
            image_number_offset=image_number_offset.value,
        )
        if key.slice_index is None:
            return ObjectInstanceKey(object_id, slice_index=self.slice_index)
        return key

    def relationship_for_projection(
        self,
        relationship: ObjectRelationship,
    ) -> ObjectRelationship:
        if (
            object_relationship_spans_multiple_row_planes(relationship)
            and self.authority
            is RuntimeRecordPlaneIdentityAuthority.OVERRIDE_ROW_IDENTITY
        ):
            return relationship
        if (
            relationship.slice_indices
            and self.authority
            is RuntimeRecordPlaneIdentityAuthority.FILL_MISSING_ROW_IDENTITY
        ):
            return relationship
        source_ids = tuple(
            int(value) for value in np.asarray(relationship.source_ids).ravel()
        )
        relationship_slice_count = (
            relationship.slice_count if relationship.slice_count is not None else 0
        )
        slice_count = max(
            relationship_slice_count,
            self.slice_index + 1,
        )
        return ObjectRelationship(
            name=relationship.name,
            source=relationship.source,
            target=relationship.target,
            source_ids=relationship.source_ids,
            target_ids=relationship.target_ids,
            relationship_type=relationship.relationship_type,
            slice_indices=tuple(self.slice_index for _ in source_ids),
            slice_count=slice_count,
        )


def object_relationship_spans_multiple_row_planes(
    relationship: ObjectRelationship,
) -> bool:
    """Return whether a relationship already carries multiple row-plane identities."""
    return len(frozenset(int(value) for value in relationship.slice_indices)) > 1


@dataclass(frozen=True, slots=True)
class RuntimeScopedMeasurementTable:
    """Measurement table plus runtime record-plane identity used for joins."""

    table: MeasurementTable
    plane_identity: RuntimeRecordPlaneIdentity | None = None
    record_identity: str | None = None
    spans_multiple_transport_identities: bool = False

    def object_instance_key(
        self,
        row: RuntimeMeasurementRowMapping,
        object_id: int,
        *,
        image_number_offset: RuntimeImageNumberOffset,
    ) -> ObjectInstanceKey:
        if (
            self.plane_identity is None
            or (
                self.plane_identity.authority
                is RuntimeRecordPlaneIdentityAuthority.OVERRIDE_ROW_IDENTITY
                and self.spans_multiple_transport_identities
            )
        ):
            return ObjectInstanceKey.from_measurement_row(
                row.row,
                object_id,
                image_number_offset=image_number_offset.value,
            )
        return self.plane_identity.object_instance_key(
            row,
            object_id,
            image_number_offset=image_number_offset,
        )


class RuntimeRecordPlaneIdentityAuthority(Enum):
    """How runtime record identity should interact with row-local identity."""

    FILL_MISSING_ROW_IDENTITY = auto()
    OVERRIDE_ROW_IDENTITY = auto()


@dataclass(slots=True)
class RuntimeAxisGroupPlaneIndex:
    """Stable per-axis mapping from runtime group scope to plane identity."""

    indices_by_group_key: dict[str | None, int] = field(default_factory=dict)

    def slice_index_for_scope(self, scope: ArtifactScope) -> int | None:
        if scope.group_key is None:
            return None
        return self.indices_by_group_key.setdefault(
            scope.group_key,
            len(self.indices_by_group_key),
        )


@dataclass(slots=True)
class RuntimeAxisRepeatedArtifactPlaneIndex:
    """Assign plane identity to repeated records without distinct runtime scope."""

    next_index_by_artifact_key: dict[tuple[type[ArtifactType], str], int] = field(
        default_factory=dict
    )

    def plane_identity_for_record(
        self,
        *,
        kind: type[ArtifactType],
        name: str,
    ) -> RuntimeRecordPlaneIdentity:
        key = (kind, name)
        slice_index = self.next_index_by_artifact_key.get(key, 0)
        self.next_index_by_artifact_key[key] = slice_index + 1
        return RuntimeRecordPlaneIdentity(
            slice_index,
            RuntimeRecordPlaneIdentityAuthority.OVERRIDE_ROW_IDENTITY,
        )


@dataclass(slots=True)
class RuntimeAxisRecordPlaneIdentityResolver:
    """Resolve runtime record plane identity from scope and repeated artifacts."""

    repeated_artifact_counts: Counter[tuple[ArtifactType, str]]
    group_plane_index: RuntimeAxisGroupPlaneIndex = field(
        default_factory=RuntimeAxisGroupPlaneIndex
    )
    repeated_artifact_plane_index: RuntimeAxisRepeatedArtifactPlaneIndex = field(
        default_factory=RuntimeAxisRepeatedArtifactPlaneIndex
    )

    @classmethod
    def from_records(
        cls,
        records: Iterable[StoredRuntimeValue],
    ) -> "RuntimeAxisRecordPlaneIdentityResolver":
        return cls(
            Counter(
                (record.key.artifact_type, record.key.name)
                for record in records
                if record.key.artifact_type.participates_in_axis_plane_identity
            )
        )

    def plane_identity_for_record(
        self,
        *,
        kind: ArtifactType,
        name: str,
        scope: ArtifactScope,
    ) -> RuntimeRecordPlaneIdentity | None:
        slice_index = self.group_plane_index.slice_index_for_scope(scope)
        if slice_index is not None:
            return RuntimeRecordPlaneIdentity(
                slice_index,
                RuntimeRecordPlaneIdentityAuthority.FILL_MISSING_ROW_IDENTITY,
            )
        artifact_key = (kind, name)
        if self.repeated_artifact_counts[artifact_key] > 1:
            return self.repeated_artifact_plane_index.plane_identity_for_record(
                kind=kind,
                name=name,
            )
        return None

    def plane_identity_for_runtime_record(
        self,
        record: StoredRuntimeValue,
    ) -> RuntimeRecordPlaneIdentity | None:
        """Resolve plane identity directly from a runtime artifact record."""
        return self.plane_identity_for_record(
            kind=record.key.artifact_type,
            name=record.key.name,
            scope=record.key.scope,
        )


@dataclass(frozen=True, slots=True)
class RuntimeScopedObjectRelationship:
    """Object relationship plus runtime scope identity used for relationship joins."""

    relationship: ObjectRelationship
    plane_identity: RuntimeRecordPlaneIdentity | None = None

    def relationship_for_projection(self) -> ObjectRelationship:
        if self.plane_identity is None:
            return self.relationship
        return self.plane_identity.relationship_for_projection(self.relationship)

    def identity_for_projection(self) -> "RuntimeObjectRelationshipIdentity":
        """Return exact semantic identity after plane projection."""
        return RuntimeObjectRelationshipIdentity.from_relationship(
            self.relationship_for_projection()
        )


@dataclass(frozen=True, slots=True)
class RuntimeObjectRelationshipIdentity:
    """Exact relationship identity used to collapse duplicate runtime artifacts."""

    name: str
    source_name: str
    target_name: str
    relationship_type: str
    instance_relationship: ObjectInstanceRelationship

    @classmethod
    def from_relationship(
        cls,
        relationship: ObjectRelationship,
    ) -> "RuntimeObjectRelationshipIdentity":
        return cls(
            name=relationship.name,
            source_name=normalize_runtime_identifier(relationship.source.name),
            target_name=normalize_runtime_identifier(relationship.target.name),
            relationship_type=normalize_runtime_identifier(
                relationship.relationship_type
            ),
            instance_relationship=ObjectInstanceRelationship.from_id_columns(
                tuple(int(value) for value in np.asarray(relationship.source_ids).ravel()),
                tuple(int(value) for value in np.asarray(relationship.target_ids).ravel()),
                slice_indices=relationship.slice_indices,
                slice_count=relationship.slice_count,
            ),
        )


def object_measurement_values_by_label(
    measurement_tables: tuple[RuntimeScopedMeasurementTable, ...],
    object_name: str,
    policy: RuntimeEquivalencePolicy,
    *,
    known_source_names: tuple[str, ...],
    required_keys: RuntimeRequiredMeasurementKeys = None,
) -> RuntimeObjectValuesByLabel:
    object_subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, object_name)
    values_by_feature: RuntimeObjectValuesByLabel = {}
    row_required_keys = RequiredRuntimeMeasurementProjection(
        required_keys,
        policy,
        known_source_names=known_source_names,
    ).input_keys()
    schema_cache: RuntimeMeasurementRowSchemaCache = {}
    key_cache: RuntimeMeasurementFeatureKeyCache = {}
    long_form_key_cache: RuntimeMeasurementLongFormKeyCache = {}
    wide_feature_index_cache: RuntimeMeasurementWideFeatureIndexCache = {}
    wide_feature_plan_cache: RuntimeMeasurementWideFeaturePlanCache = {}
    qualifier_render_cache: RuntimeMeasurementQualifierRenderCache = {}
    padding_group_cache: RuntimeMeasurementPaddingGroupCache = {}
    collapsed_numeric_qualifier_cache: RuntimeCollapsedNumericQualifierCache = {}
    subject_schema_cache: RuntimeMeasurementRowSubjectSchemaCache = {}
    required_key_index = RuntimeMeasurementRequiredKeyIndex.from_required_keys(
        row_required_keys
    )
    derive_directional_pair_facts = RuntimeDirectionalPairMeasurementDerivationContract(
        policy,
        known_source_names,
    ).required_keys_need_derivation(row_required_keys)
    normalized_object_name = normalize_runtime_identifier(object_name)
    for scoped_table in measurement_tables:
        table = scoped_table.table
        if not measurement_table_may_contain_object_name(
            table,
            normalized_object_name,
        ):
            continue
        table_subject = RuntimeMeasurementSubjectKey.from_table_subject(table.subject)
        object_id_field = measurement_table_object_id_field(table)
        table_padding_group = measurement_table_padding_group(table.name)
        image_number_offset = RuntimeImageNumberOffset.from_measurement_table(table)
        for row in iter_measurement_rows((table,)):
            row_mapping = measurement_row_mapping(row)
            runtime_row = RuntimeMeasurementRowMapping(row_mapping)
            try:
                object_label = measurement_object_label(
                    row_mapping,
                    object_id_field=object_id_field,
                )
            except (TypeError, ValueError):
                continue
            if object_label is None:
                continue
            row_values = RuntimeIndexedRowValues.from_row(runtime_row)
            row_subject_projection = RuntimeMeasurementRowSubjectProjection(
                table_subject,
                table.source_image_name,
                row_values,
                runtime_measurement_row_subject_schema(
                    runtime_row.header,
                    subject_schema_cache,
                ),
            )
            subject = row_subject_projection.subject()
            if subject != object_subject:
                continue
            source_qualification = subject.bind_row_source_identity(
                row_subject_projection.source_name()
            )
            row_context = RuntimeRowProjectionContext.from_row(
                runtime_row,
                subject,
                policy,
                source_name=source_qualification.feature_source_name,
                known_source_names=known_source_names,
                required_keys=row_required_keys,
                table_padding_group=table_padding_group,
                image_number_offset=image_number_offset,
                derive_directional_pair_facts=derive_directional_pair_facts,
                schema_cache=schema_cache,
                key_cache=key_cache,
                long_form_key_cache=long_form_key_cache,
                wide_feature_index_cache=wide_feature_index_cache,
                wide_feature_plan_cache=wide_feature_plan_cache,
                qualifier_render_cache=qualifier_render_cache,
                padding_group_cache=padding_group_cache,
                collapsed_numeric_qualifier_cache=collapsed_numeric_qualifier_cache,
                required_key_index=required_key_index,
            )
            for key, value in row_context.numeric_values():
                if row_required_keys is not None and key not in row_required_keys:
                    continue
                if key.statistic != MeasurementStatistic.VALUE.value:
                    continue
                if key not in values_by_feature:
                    values_by_feature[key] = {}
                object_instance_key = scoped_table.object_instance_key(
                    runtime_row,
                    object_label,
                    image_number_offset=image_number_offset,
                )
                values_by_feature[key][object_instance_key] = value
    return values_by_feature


def measurement_table_may_contain_object_name(
    table: MeasurementTable,
    normalized_object_name: str,
) -> bool:
    """Return whether table ownership can match a normalized object target."""
    declared_object_name = measurement_table_object_name(table)
    if declared_object_name is not None:
        return normalize_runtime_identifier(declared_object_name) == normalized_object_name
    rows = table.rows
    if not isinstance(rows, ColumnarRows):
        return True
    column_names = frozenset(str(column) for column in rows.columns)
    if MeasurementRowAxisField.OBJECT_NAME.value not in column_names:
        return True
    return any(
        normalize_runtime_identifier(str(value)) == normalized_object_name
        for value in columnar_row_values(rows, MeasurementRowAxisField.OBJECT_NAME.value)
        if value is not None
    )


@dataclass(frozen=True, slots=True)
class ObjectInstanceKeyPlaneAlignmentContext:
    """Object-instance relationship keys plus measured child value identity."""

    child_ids_by_parent: _ObjectInstanceChildrenByParent
    values_by_child_id: Mapping[ObjectInstanceKey, float]

    @property
    def child_slice_indices(self) -> frozenset[int | None]:
        return frozenset(
            child_id.slice_index
            for child_ids in self.child_ids_by_parent.values()
            for child_id in child_ids
        )

    @property
    def value_slice_indices(self) -> frozenset[int | None]:
        return frozenset(child_id.slice_index for child_id in self.values_by_child_id)

    @property
    def single_value_slice_index(self) -> int | None:
        slice_indices = self.value_slice_indices
        if len(slice_indices) != 1:
            return None
        slice_index = next(iter(slice_indices))
        return slice_index if slice_index is not None else None


class ObjectInstanceKeyPlaneAlignmentStrategy(
    MostDerivedContextStrategyMixin[ObjectInstanceKeyPlaneAlignmentContext],
    ABC,
):
    """Nominal projection between relationship and measurement-row plane identity."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_KEY)

    strategy_key: ClassVar[str | None] = None

    @classmethod
    def align_child_ids_by_parent(
        cls,
        child_ids_by_parent: _ObjectInstanceChildrenByParent,
        values_by_child_id: Mapping[ObjectInstanceKey, float],
    ) -> _ObjectInstanceChildrenByParent:
        """Align relationship child identities to the measured value domain."""
        context = ObjectInstanceKeyPlaneAlignmentContext(
            child_ids_by_parent=child_ids_by_parent,
            values_by_child_id=values_by_child_id,
        )
        strategy = cls.for_context(
            context,
            error_subject="Object instance plane alignment",
        )
        if strategy is None:
            raise ValueError("Object instance plane alignment requires a strategy.")
        return strategy.align(context)

    @abstractmethod
    def align(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> _ObjectInstanceChildrenByParent:
        """Return relationship child identities projected into value identity."""


class UnmodifiedObjectInstanceKeyPlaneAlignmentStrategy(
    ObjectInstanceKeyPlaneAlignmentStrategy
):
    """Preserve relationship identity when no plane projection is required."""

    strategy_key = "unmodified"

    def matches(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> bool:
        del context
        return True

    def align(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> _ObjectInstanceChildrenByParent:
        return context.child_ids_by_parent


class SingleSliceValueObjectInstanceKeyPlaneAlignmentStrategy(
    UnmodifiedObjectInstanceKeyPlaneAlignmentStrategy
):
    """Inject the single measured plane into unscoped relationship identities."""

    strategy_key = "single_slice_value"

    def matches(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> bool:
        return (
            context.child_slice_indices == frozenset({None})
            and context.single_value_slice_index is not None
        )

    def align(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> _ObjectInstanceChildrenByParent:
        slice_index = context.single_value_slice_index
        if slice_index is None:
            raise ValueError("Single-slice alignment requires a value slice index.")
        return {
            ObjectInstanceKey(parent_id.object_id, slice_index=slice_index): tuple(
                ObjectInstanceKey(child_id.object_id, slice_index=slice_index)
                for child_id in child_ids
            )
            for parent_id, child_ids in context.child_ids_by_parent.items()
        }


class MultiSliceValueObjectInstanceKeyPlaneAlignmentStrategy(
    UnmodifiedObjectInstanceKeyPlaneAlignmentStrategy
):
    """Expand unscoped relationship identities across measured child planes."""

    strategy_key = "multi_slice_value"

    def matches(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> bool:
        return (
            context.child_slice_indices == frozenset({None})
            and None not in context.value_slice_indices
            and len(context.value_slice_indices) > 1
        )

    def align(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> _ObjectInstanceChildrenByParent:
        value_keys_by_object_id: dict[int, list[ObjectInstanceKey]] = {}
        for value_key in context.values_by_child_id:
            value_keys_by_object_id.setdefault(value_key.object_id, []).append(value_key)
        return {
            parent_id: tuple(
                value_key
                for child_id in child_ids
                for value_key in value_keys_by_object_id.get(child_id.object_id, ())
            )
            for parent_id, child_ids in context.child_ids_by_parent.items()
        }


class UnscopedValueObjectInstanceKeyPlaneAlignmentStrategy(
    UnmodifiedObjectInstanceKeyPlaneAlignmentStrategy
):
    """Drop relationship plane identity when measured values are unscoped."""

    strategy_key = "unscoped_value"

    def matches(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> bool:
        return (
            context.value_slice_indices == frozenset({None})
            and None not in context.child_slice_indices
        )

    def align(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> _ObjectInstanceChildrenByParent:
        unscoped_children_by_parent: dict[ObjectInstanceKey, list[ObjectInstanceKey]] = {}
        for parent_id, child_ids in context.child_ids_by_parent.items():
            parent_key = ObjectInstanceKey(parent_id.object_id)
            unscoped_children_by_parent.setdefault(parent_key, []).extend(
                ObjectInstanceKey(child_id.object_id) for child_id in child_ids
            )
        return {
            parent_id: tuple(child_ids)
            for parent_id, child_ids in unscoped_children_by_parent.items()
        }


RELATIONSHIP_DISTANCE_FEATURE_NAMES = frozenset(
    (
        "distance_centroid",
        "distance_minimum",
    )
)


@dataclass(frozen=True, slots=True)
class RelationshipAggregateFeatureContext:
    """Semantic context for deriving parent-row aggregates from child rows."""

    source_name: str
    target_name: str
    feature_name: str


@dataclass(frozen=True, slots=True)
class RelationshipAggregateFeatureKeyProjection:
    """Parsed relationship aggregate measurement key."""

    feature: RuntimeMeasurementFeatureKey
    dialect: RuntimeMeasurementDialect

    def resolution(self) -> "RelationshipAggregateFeatureResolution":
        aggregate_identity = RuntimeAggregateFeatureIdentity.from_parts(
            tuple(part for part in self.feature.feature_name.split("_") if part),
            self.dialect,
        )
        if aggregate_identity is None:
            return RelationshipAggregateFeatureResolution(None, None)
        context = RelationshipAggregateFeatureContext(
            source_name=self.feature.subject.name or "",
            target_name=aggregate_identity.object_name,
            feature_name=aggregate_identity.feature_name,
        )
        semantics = RelationshipAggregateFeatureSemantics.for_context(
            context,
            required=False,
        )
        if semantics is None:
            return RelationshipAggregateFeatureResolution(context, None)
        return RelationshipAggregateFeatureResolution(context, semantics)

    def aggregate_child_feature_name(self) -> str | None:
        return self.resolution().aggregate_child_feature_name()


@dataclass(frozen=True, slots=True)
class RelationshipAggregateFeatureResolution:
    """Relationship aggregate feature context with its owning semantics."""

    context: RelationshipAggregateFeatureContext | None
    semantics: "RelationshipAggregateFeatureSemantics | None"

    @property
    def is_resolved(self) -> bool:
        return self.context is not None and self.semantics is not None

    def aggregate_child_feature_name(self) -> str | None:
        if not self.is_resolved:
            return None
        assert self.context is not None
        assert self.semantics is not None
        return self.semantics.aggregate_child_feature_name(self.context)


class RelationshipAggregateFeatureSemantics(
    MostDerivedContextStrategyMixin[RelationshipAggregateFeatureContext],
    ABC,
):
    """Map child measurement features onto relationship aggregate features."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_KEY)

    strategy_key: ClassVar[str | None] = None

    @abstractmethod
    def matches(self, context: RelationshipAggregateFeatureContext) -> bool:
        """Return whether this semantic strategy owns the feature context."""

    @abstractmethod
    def required_child_feature_names(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> tuple[str, ...]:
        """Return child-row features needed to synthesize the aggregate."""

    @abstractmethod
    def aggregate_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
        *,
        aggregate: str = MeasurementStatistic.MEAN.value,
    ) -> str:
        """Return the source-row aggregate feature emitted for a child feature."""

    def aggregate_child_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> str:
        """Return the child feature semantically represented by ``context``."""
        return normalize_runtime_identifier(context.feature_name)

    @staticmethod
    def parent_qualified_feature_name(feature_name: str, parent_name: str) -> str:
        return "_".join(
            part for part in (feature_name, normalize_runtime_identifier(parent_name)) if part
        )

    @staticmethod
    def parent_unqualified_feature_name(feature_name: str, parent_name: str) -> str:
        parent_suffix = f"_{normalize_runtime_identifier(parent_name)}"
        if feature_name.endswith(parent_suffix):
            return feature_name[: -len(parent_suffix)]
        return feature_name

    @staticmethod
    def target_aggregate_feature_name(
        target_name: str,
        child_feature_name: str,
        *,
        aggregate: str = MeasurementStatistic.MEAN.value,
    ) -> str:
        parts = (
            normalize_runtime_identifier(aggregate),
            normalize_runtime_identifier(target_name),
            normalize_runtime_identifier(child_feature_name),
        )
        return "_".join(part for part in parts if part)

    @classmethod
    def aggregate_child_feature_name_from_key(
        cls,
        feature: RuntimeMeasurementFeatureKey,
        dialect: RuntimeMeasurementDialect,
    ) -> str | None:
        """Return child feature represented by a relationship aggregate key."""
        del cls
        return RelationshipAggregateFeatureKeyProjection(
            feature,
            dialect,
        ).aggregate_child_feature_name()


class GenericRelationshipAggregateFeatureSemantics(
    RelationshipAggregateFeatureSemantics
):
    """Default relationship aggregates preserve the child feature identity."""

    strategy_key = "generic"

    def matches(self, context: RelationshipAggregateFeatureContext) -> bool:
        del context
        return True

    def required_child_feature_names(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> tuple[str, ...]:
        return (normalize_runtime_identifier(context.feature_name),)

    def aggregate_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
        *,
        aggregate: str = MeasurementStatistic.MEAN.value,
    ) -> str:
        return self.target_aggregate_feature_name(
            context.target_name,
            context.feature_name,
            aggregate=aggregate,
        )

    def aggregate_child_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> str:
        return normalize_runtime_identifier(context.feature_name)


class ParentQualifiedDistanceAggregateFeatureSemantics(
    GenericRelationshipAggregateFeatureSemantics
):
    """CellProfiler relationship distance rows qualify child features by parent."""

    strategy_key = "parent_qualified_distance"

    def matches(self, context: RelationshipAggregateFeatureContext) -> bool:
        feature_name = normalize_runtime_identifier(context.feature_name)
        return (
            feature_name in RELATIONSHIP_DISTANCE_FEATURE_NAMES
            or self.parent_unqualified_feature_name(
                feature_name,
                context.source_name,
            )
            in RELATIONSHIP_DISTANCE_FEATURE_NAMES
        )

    def required_child_feature_names(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> tuple[str, ...]:
        feature_name = normalize_runtime_identifier(context.feature_name)
        return (
            feature_name,
            self.parent_qualified_feature_name(feature_name, context.source_name),
        )

    def aggregate_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
        *,
        aggregate: str = MeasurementStatistic.MEAN.value,
    ) -> str:
        return self.target_aggregate_feature_name(
            context.target_name,
            self.parent_unqualified_feature_name(
                normalize_runtime_identifier(context.feature_name),
                context.source_name,
            ),
            aggregate=aggregate,
        )

    def aggregate_child_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> str:
        return self.parent_unqualified_feature_name(
            normalize_runtime_identifier(context.feature_name),
            context.source_name,
        )


@dataclass(frozen=True, slots=True)
class RelationshipMeasurementSemantics:
    """Measurement identity contract for a directed object relationship."""

    relationship: ObjectRelationship

    @property
    def source_name(self) -> str:
        return normalize_runtime_identifier(self.relationship.source.name)

    @property
    def target_name(self) -> str:
        return normalize_runtime_identifier(self.relationship.target.name)

    @property
    def source_subject(self) -> RuntimeMeasurementSubjectKey:
        return RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            self.source_name,
        )

    @property
    def target_subject(self) -> RuntimeMeasurementSubjectKey:
        return RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            self.target_name,
        )

    @property
    def instance_relationship(self) -> ObjectInstanceRelationship:
        return ObjectInstanceRelationship.from_id_columns(
            tuple(int(value) for value in np.asarray(self.relationship.source_ids).ravel()),
            tuple(int(value) for value in np.asarray(self.relationship.target_ids).ravel()),
            slice_indices=self.relationship.slice_indices,
            slice_count=self.relationship.slice_count,
        )

    @property
    def aggregate_prefix(self) -> str:
        return f"{MeasurementStatistic.MEAN.value}_{self.target_name}_"

    @property
    def child_count_key(self) -> RuntimeMeasurementFeatureKey:
        return RuntimeMeasurementFeatureKey(
            self.source_subject,
            f"{self.target_name}_count",
        )

    @property
    def parent_key(self) -> RuntimeMeasurementFeatureKey:
        return RuntimeMeasurementFeatureKey(
            self.target_subject,
            self.source_name,
        )

    @property
    def target_object_number_key(self) -> RuntimeMeasurementFeatureKey:
        return RuntimeMeasurementFeatureKey.from_subject_feature(
            self.target_subject,
            ObjectCoreMeasurementFeature.OBJECT_NUMBER.value,
        )

    def aggregate_feature_name(
        self,
        child_feature_name: str,
        *,
        aggregate: str = MeasurementStatistic.MEAN.value,
    ) -> str:
        context = RelationshipAggregateFeatureContext(
            source_name=self.source_name,
            target_name=self.target_name,
            feature_name=child_feature_name,
        )
        return RelationshipAggregateFeatureSemantics.for_context(
            context,
            error_subject="relationship aggregate feature",
        ).aggregate_feature_name(context, aggregate=aggregate)

    def required_child_measurement_keys(
        self,
        required_measurement_keys: RuntimeRequiredMeasurementKeys,
    ) -> RuntimeRequiredMeasurementKeys:
        """Return child measurements needed to synthesize required aggregates."""
        if required_measurement_keys is None:
            return None
        child_keys: set[RuntimeMeasurementFeatureKey] = set()
        for key in required_measurement_keys:
            if (
                key.subject != self.source_subject
                or key.statistic != MeasurementStatistic.VALUE.value
                or not key.feature_name.startswith(self.aggregate_prefix)
                or key.feature_name == self.aggregate_prefix
            ):
                continue
            aggregate_child_feature_name = key.feature_name.removeprefix(
                self.aggregate_prefix
            )
            context = RelationshipAggregateFeatureContext(
                source_name=self.source_name,
                target_name=self.target_name,
                feature_name=aggregate_child_feature_name,
            )
            semantics = RelationshipAggregateFeatureSemantics.for_context(
                context,
                error_subject="relationship aggregate child feature",
            )
            child_keys.update(
                RuntimeMeasurementFeatureKey.from_subject_feature(
                    self.target_subject,
                    child_feature_name,
                    source_name=key.source_name,
                )
                for child_feature_name in semantics.required_child_feature_names(
                    context
                )
            )
        return frozenset(child_keys)

    def measurement_facts(
        self,
        policy: RuntimeEquivalencePolicy,
        *,
        object_label_catalog: RuntimeObjectLabelInstanceCatalog,
    ) -> RuntimeMeasurementFacts:
        """Return direct relationship measurements under canonical object identity."""
        child_keys_by_parent = self.child_keys_by_parent(object_label_catalog)
        parent_key_by_child = self.parent_key_by_child()
        return (
            *(
                (
                    self.child_count_key,
                    runtime_cell_signature(
                        str(len(child_keys_by_parent.get(source_key, ()))),
                        policy,
                    ),
                )
                for source_key in self.source_domain(object_label_catalog)
            ),
            *(
                (
                    self.parent_key,
                    runtime_cell_signature(
                        str(
                            parent_key_by_child[target_key].object_id
                            if target_key in parent_key_by_child
                            else 0
                        ),
                        policy,
                    ),
                )
                for target_key in self.target_domain(object_label_catalog)
            ),
        )

    def aggregate_measurement_facts(
        self,
        child_values_by_feature: Mapping[
            RuntimeMeasurementFeatureKey,
            Mapping[ObjectInstanceKey, float],
        ],
        policy: RuntimeEquivalencePolicy,
        *,
        object_label_catalog: RuntimeObjectLabelInstanceCatalog,
        existing_measurement_keys: _RuntimeMeasurementKeySet = frozenset(),
        required_measurement_keys: RuntimeRequiredMeasurementKeys = None,
    ) -> RuntimeMeasurementFacts:
        """Return source-row aggregate measurements derived from target rows."""
        child_values = {
            key: dict(values_by_child_id)
            for key, values_by_child_id in child_values_by_feature.items()
        }
        child_values[self.target_object_number_key] = self.target_object_number_values(
            object_label_catalog
        )
        if not child_values:
            return ()

        child_ids_by_parent = self.child_keys_by_parent(object_label_catalog)
        aggregate_facts: RuntimeMeasurementFactList = []
        for child_key, values_by_child_id in child_values.items():
            if child_key.subject.scope is not MeasurementScope.OBJECT:
                continue
            if child_key.subject != self.target_subject:
                continue
            aggregate_key = RuntimeMeasurementFeatureKey.from_subject_feature(
                self.source_subject,
                self.aggregate_feature_name(child_key.feature_name),
                source_name=child_key.source_name,
            )
            if (
                aggregate_key in existing_measurement_keys
                and child_key.feature_name
                != ObjectCoreMeasurementFeature.OBJECT_NUMBER.value
            ):
                continue
            if (
                required_measurement_keys is not None
                and aggregate_key not in required_measurement_keys
            ):
                continue
            aligned_child_ids_by_parent = (
                ObjectInstanceKeyPlaneAlignmentStrategy.align_child_ids_by_parent(
                    child_ids_by_parent,
                    values_by_child_id,
                )
            )
            aggregate_values_by_parent = self.aggregate_values_by_parent(
                aligned_child_ids_by_parent,
                values_by_child_id,
            )
            for _parent_id, child_ids in aligned_child_ids_by_parent.items():
                aggregate_value = aggregate_values_by_parent[child_ids]
                if not math.isfinite(aggregate_value):
                    continue
                aggregate_facts.append(
                    (
                        aggregate_key,
                        runtime_cell_signature(
                            str(aggregate_value),
                            policy,
                        ),
                    )
                )
        return tuple(aggregate_facts)

    def source_domain(
        self,
        object_label_catalog: RuntimeObjectLabelInstanceCatalog,
    ) -> tuple[ObjectInstanceKey, ...]:
        """Return source-object identities represented by this relationship."""
        return self.instance_relationship.source_domain(
            object_label_catalog.count_for_subject(self.source_subject),
            declared_keys=object_label_catalog.domain_for_subject(self.source_subject),
        )

    def target_domain(
        self,
        object_label_catalog: RuntimeObjectLabelInstanceCatalog,
    ) -> tuple[ObjectInstanceKey, ...]:
        """Return target-object identities represented by this relationship."""
        return self.instance_relationship.target_domain(
            object_label_catalog.count_for_subject(self.target_subject),
            declared_keys=object_label_catalog.domain_for_subject(self.target_subject),
        )

    def child_keys_by_parent(
        self,
        object_label_catalog: RuntimeObjectLabelInstanceCatalog,
    ) -> dict[ObjectInstanceKey, tuple[ObjectInstanceKey, ...]]:
        """Return target identities grouped by source identity."""
        return self.instance_relationship.child_keys_by_parent(
            source_object_count=object_label_catalog.count_for_subject(
                self.source_subject
            ),
            declared_source_keys=object_label_catalog.domain_for_subject(
                self.source_subject
            ),
        )

    def parent_key_by_child(self) -> dict[ObjectInstanceKey, ObjectInstanceKey]:
        """Return source identity for each target identity."""
        return self.instance_relationship.parent_key_by_child()

    def target_object_number_values(
        self,
        object_label_catalog: RuntimeObjectLabelInstanceCatalog,
    ) -> dict[ObjectInstanceKey, float]:
        """Return target-object-number values keyed by target identity."""
        return {
            target_key: float(target_key.object_id)
            for target_key in self.target_domain(object_label_catalog)
        }

    def aggregate_values_by_parent(
        self,
        child_ids_by_parent: _ObjectInstanceChildrenByParent,
        values_by_child_id: Mapping[ObjectInstanceKey, float],
    ) -> _ObjectInstanceAggregateValues:
        """Return aggregate target values keyed by each parent child-domain."""
        means: _ObjectInstanceAggregateValues = {}
        for child_ids in child_ids_by_parent.values():
            if child_ids in means:
                continue
            means[child_ids] = self.mean_child_value(child_ids, values_by_child_id)
        return means

    def mean_child_value(
        self,
        child_ids: tuple[ObjectInstanceKey, ...],
        values_by_child_id: Mapping[ObjectInstanceKey, float],
    ) -> float:
        """Return the mean over child identities with available values."""
        del self
        values = tuple(
            values_by_child_id[child_id]
            for child_id in child_ids
            if child_id in values_by_child_id
        )
        if not values:
            return float("nan")
        return float(sum(values) / len(values))
