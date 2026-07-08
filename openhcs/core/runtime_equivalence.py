"""Semantic equivalence checks for runtime outputs."""

from __future__ import annotations

import hashlib
import inspect
import math
import sys
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
import numpy as np

import openhcs.core.runtime_artifact_queries as runtime_artifact_queries
import openhcs.core.measurement_feature_queries as measurement_feature_queries
import openhcs.core.measurement_row_materialization as measurement_row_materialization
import openhcs.core.equivalence.cells as equivalence_cells
import openhcs.core.equivalence.keys as equivalence_keys
import openhcs.core.equivalence.measurement_facts as measurement_facts
import openhcs.core.equivalence.measurement_features as measurement_features
import openhcs.core.equivalence.measurement_requirements as measurement_requirements
import openhcs.core.equivalence.measurement_rows as equivalence_measurement_rows
import openhcs.core.equivalence.object_label_measurements as object_label_measurements
import openhcs.core.equivalence.policy as equivalence_policy
import openhcs.core.equivalence.relationships as equivalence_relationships
import openhcs.core.equivalence.tables as equivalence_tables
import openhcs.core.runtime_semantics as runtime_semantics
from openhcs.core.artifacts import (
    ArtifactType,
    ArtifactTypeStrategyMatchMixin,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.measurement_row_materialization import (
    iter_measurement_rows,
    measurement_object_label,
    measurement_row_object_name,
    measurement_row_source_image_name,
)
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionObservation,
)
from openhcs.core.runtime_exports import RuntimeImageExportSpec
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    MeasurementObjectRowIdentity,
    MeasurementScope,
    MeasurementStatistic,
    ObjectCoreMeasurementFeature,
    ObjectCalculatedFeatureMarker,
    ObjectCountFeatureMarker,
    ObjectIdentifierFeatureMarker,
    ObjectIntensityFeatureMarker,
    ObjectInstanceKey,
    ObjectLocationFeatureMarker,
    ObjectShapeDescriptorFeatureMarker,
    measurement_row_mapping,
)
from openhcs.core.runtime_stores import StoredRuntimeValue
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    MostDerivedContextStrategyMixin,
)
from openhcs.core.runtime_values import ColumnarRows
from openhcs.core.runtime_values import MeasurementTable
from openhcs.core.runtime_values import ObjectLabelSet
from openhcs.core.runtime_values import ObjectRelationship
from openhcs.core.equivalence.policy import (
    DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    RuntimeEquivalencePolicy,
    RuntimeMeasurementFeatureNumericTolerance,
    RuntimeMeasurementSourceQualifiedFeature,
    normalize_runtime_identifier,
)
from openhcs.core.equivalence.keys import (
    RuntimeAggregateFeatureIdentity,
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementFeatureKeyProjection,
    RuntimeMeasurementFeatureKeySourceContext,
    RuntimeMeasurementFeatureNameProjection,
    RuntimeMeasurementSubjectKey,
)
from openhcs.core.equivalence.cells import (
    RuntimeCellMissingStrategy,
    RuntimeCellSignature,
    RuntimeCellValueKind,
    RuntimeMeasurementCellSignatureProjection,
    absolute_numeric_counters_equivalent as _absolute_numeric_counters_equivalent,
    finite_signature_number as _finite_signature_number,
    runtime_cell_signature,
    runtime_cell_signature_counters_equivalent,
    sparse_absolute_numeric_counters_equivalent as _sparse_absolute_numeric_counters_equivalent,
    sparse_numeric_counters_equivalent as _sparse_numeric_counters_equivalent,
)
from openhcs.core.equivalence.tables import (
    CSV_HEADER_CONTEXT_STOPWORDS,
    RuntimeTableSnapshot,
    RuntimeMeasurementTableIdentity,
    RuntimeMeasurementObjectSubtableSet,
    aggregate_measurement_table_semantic_key,
    complete_object_domain_measurement_table_key,
    dedupe_runtime_measurement_table_aggregate_rows,
    dedupe_runtime_measurement_table_object_subtable,
    exact_measurement_table_key,
    is_wide_measurement_table,
    measurement_table_padding_group,
    measurement_table_schema_has_object_identity,
    measurement_table_spans_multiple_transport_identities,
)
from openhcs.core.equivalence.measurement_rows import (
    ContextualMeasurementPaddingProjection,
    IMAGE_IDENTITY_FIELDS,
    RuntimeIndexedRowValues,
    RuntimeCollapsedNumericQualifierCache,
    RuntimeImageNumberOffset,
    RuntimeMeasurementFeatureKeyCache,
    RuntimeMeasurementFeatureCategoryPriority,
    RuntimeMeasurementLongFormKeyCache,
    RuntimeMeasurementPaddingGroupCache,
    RuntimeMeasurementQualifierRenderCache,
    RuntimeMeasurementRequiredKeyIndex,
    RuntimeMeasurementRowIdentity,
    RuntimeMeasurementRowIdentityOrMissing,
    RuntimeMeasurementRowMapping,
    RuntimeMeasurementRowSchemaCache,
    RuntimeMeasurementRowSubjectProjection,
    RuntimeMeasurementRowSubjectSchemaCache,
    RuntimeObjectMeasurementRowIdentity,
    RuntimeRowProjectionContext,
    image_number_reference_feature,
    runtime_measurement_category_priority,
    runtime_measurement_identity_field_matches,
    runtime_measurement_row_subject_schema,
    runtime_metadata_map_row_matches,
)
from openhcs.core.equivalence.measurement_facts import (
    RuntimeDirectionalPairMeasurementDerivationContract,
    RuntimeMeasurementFact,
    RuntimeExpectedMeasurementFactCompletion,
    RuntimeMeasurementFactCounterMap,
    RuntimeMeasurementFactCounterMapping,
    RuntimeMeasurementFactList as RuntimeMeasurementFactList,
    RuntimeMeasurementFacts,
    RuntimeMeasurementPaddingGroup,
    RuntimeRequiredMeasurementKeys,
    record_measurement_facts,
    runtime_measurement_fact_counter,
    spatial_grid_measurement_facts,
)
from openhcs.core.equivalence.measurement_features import (
    RuntimeMeasurementDescriptorSemantics,
    RuntimeMeasurementFeatureSemanticProfile,
    object_measurement_feature_matches_marker,
    object_measurement_feature_requires_sparse_boundary_object_count_stability,
    object_measurement_subject_row_identities_matching_marker,
    object_measurement_subjects_matching_marker,
)
from openhcs.core.equivalence.measurement_requirements import (
    RequiredRuntimeMeasurementProjection,
)
from openhcs.core.equivalence.object_label_measurements import (
    ObjectLabelMeasurementCompletion,
    RuntimeObjectLabelMeasurementAuthority,
    RuntimeObjectLabelInstanceCatalog,
    RuntimeObjectValuesByLabel,
    object_label_measurement_values_for_name,
)
from openhcs.core.equivalence.relationships import (
    RelationshipAggregateFeatureSemantics,
    RelationshipMeasurementSemantics,
    RuntimeAxisRecordPlaneIdentityResolver,
    RuntimeObjectRelationshipIdentity,
    RuntimeRecordPlaneIdentity,
    RuntimeRecordPlaneIdentityAuthority,
    RuntimeScopedMeasurementTable,
    RuntimeScopedObjectRelationship,
    object_measurement_values_by_label,
)
from openhcs.core.equivalence.images import RuntimeImageSnapshot
from openhcs.core.equivalence.outputs import RuntimeOutputSnapshot
from openhcs.core.equivalence.report import (
    RuntimeEquivalenceDifference,
    RuntimeEquivalenceDifferenceKind,
    RuntimeEquivalenceReport,
)
from openhcs.core.equivalence.comparison import (
    runtime_image_differences as _image_differences,
    runtime_table_differences as _table_differences,
)
from openhcs.core.source_image_provenance import SourceImageProvenanceIdentity

BENCHMARK_CACHE_DOMAINS = frozenset({"parity"})
_RUNTIME_MEASUREMENT_PROJECTION_MODULES = (
    sys.modules[__name__],
    equivalence_cells,
    equivalence_keys,
    equivalence_measurement_rows,
    equivalence_policy,
    equivalence_tables,
    measurement_facts,
    measurement_features,
    measurement_requirements,
    object_label_measurements,
    measurement_feature_queries,
    measurement_row_materialization,
    equivalence_relationships,
    runtime_artifact_queries,
    runtime_semantics,
)
_CACHE_MISS = object()


def runtime_measurement_projection_cache_identity() -> tuple[tuple[str, str], ...]:
    """Return the core semantic-projection identity for measurement caches."""
    return tuple(
        (module.__name__, _module_source_digest(module))
        for module in _RUNTIME_MEASUREMENT_PROJECTION_MODULES
    )


def _module_source_digest(module: ModuleType) -> str:
    try:
        source = inspect.getsource(module).encode("utf-8")
    except (OSError, TypeError):
        source = (
            Path(module.__file__).read_bytes()
            if module.__file__ is not None
            else repr(module).encode("utf-8")
        )
    return hashlib.sha256(source).hexdigest()


_RuntimeObjectValuesByObject = dict[
    tuple[str, RuntimeRequiredMeasurementKeys],
    RuntimeObjectValuesByLabel,
]
_RuntimeMeasurementRowMergeKey = tuple[
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementRowIdentity,
]
_RuntimeMeasurementRowMergeValue = tuple[int, int, RuntimeCellSignature]
_RuntimeMeasurementRowMergeCache = dict[
    _RuntimeMeasurementRowMergeKey,
    _RuntimeMeasurementRowMergeValue,
]
RuntimeMeasurementFactWithRowIdentity = tuple[
    RuntimeMeasurementFeatureKey,
    RuntimeCellSignature,
    RuntimeObjectMeasurementRowIdentity | None,
]
RuntimeMeasurementFactsWithRowIdentity = tuple[
    RuntimeMeasurementFactWithRowIdentity,
    ...,
]


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowPriorityCacheKey:
    """Identity for row-to-feature priority resolution within one equivalence pass."""

    row_fields: tuple[str, ...]
    long_form_feature: str | None
    feature_key: RuntimeMeasurementFeatureKey


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowFeaturePriorityCacheKey:
    """Identity for feature-category priority shared by equivalent row shapes."""

    row_fields: tuple[str, ...]
    long_form_feature: str | None


_RuntimeMeasurementRowPriorityCache = dict[RuntimeMeasurementRowPriorityCacheKey, int]
_RuntimeMeasurementRowFeaturePriorityCache = dict[
    RuntimeMeasurementRowFeaturePriorityCacheKey,
    int | None,
]
_RuntimeMeasurementPrimaryRowKey = tuple[
    RuntimeMeasurementSubjectKey,
    RuntimeObjectMeasurementRowIdentity,
]
_RuntimeMeasurementPrimaryRowSet = set[_RuntimeMeasurementPrimaryRowKey]
_RuntimeImageFeatureFactIdentity = tuple[
    RuntimeMeasurementRowIdentity,
    RuntimeRecordPlaneIdentity | None,
    SourceImageProvenanceIdentity,
    RuntimeMeasurementFeatureKey,
    RuntimeCellSignature,
]
_RuntimeImageFeatureFactRecordSet = dict[
    _RuntimeImageFeatureFactIdentity,
    set[str | None],
]
_RuntimeMeasurementFactCounterObjectCache = dict[
    int,
    tuple[RuntimeMeasurementFeatureKey, Counter[RuntimeCellSignature]],
]
_ContextualMeasurementPaddingGroup = tuple[str, tuple[str, ...], str | None]


@dataclass(slots=True)
class RuntimeAggregateMeanAccumulator:
    """Running aggregate mean state without retaining per-row values."""

    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def has_values(self) -> bool:
        return self.count > 0

    @property
    def mean(self) -> float:
        if self.count == 0:
            raise ValueError("Cannot compute mean without values.")
        return self.total / self.count


class SparseNumericCounterToleranceProfile(ABC, metaclass=AutoRegisterMeta):
    """Registered sparse numeric comparison tolerance profile."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    __registry_key__ = "profile_key"
    __skip_if_no_key__ = True

    profile_key: ClassVar[str | None] = None

    @classmethod
    def profile_type(cls, profile_key: str) -> type["SparseNumericCounterToleranceProfile"]:
        """Return the registered sparse numeric profile for ``profile_key``."""
        try:
            return cls.__registry__[profile_key]
        except KeyError as exc:
            registered = tuple(cls.__registry__)
            raise ValueError(
                f"Unknown sparse numeric tolerance profile {profile_key!r}; "
                f"registered profiles: {registered!r}."
            ) from exc

    @classmethod
    def profile_type_for_descriptor(
        cls,
        descriptor: object,
    ) -> type["SparseNumericCounterToleranceProfile"]:
        """Return the registered sparse numeric profile that owns ``descriptor``."""
        matches = tuple(
            profile_type
            for profile_type in cls.__registry__.values()
            if profile_type.matches_descriptor(descriptor)
        )
        if len(matches) != 1:
            names = tuple(profile_type.__name__ for profile_type in matches)
            raise ValueError(
                "Sparse numeric descriptor tolerance requires exactly one "
                f"matching profile for {descriptor!r}, got {names!r}."
            )
        return matches[0]

    @classmethod
    def matches_descriptor(cls, descriptor: object) -> bool:
        """Return whether this profile owns sparse tolerance for ``descriptor``."""
        del descriptor
        return False

    @classmethod
    def equivalent(
        cls,
        reference_values: Counter[RuntimeCellSignature],
        candidate_values: Counter[RuntimeCellSignature],
        policy: RuntimeEquivalencePolicy,
        *,
        descriptor: object | None = None,
    ) -> bool:
        """Return whether sparse numeric counters match under this profile."""
        tolerance = cls().tolerance(policy, descriptor=descriptor)
        return _sparse_numeric_counters_equivalent(
            reference_values,
            candidate_values,
            policy,
            abs_tolerance=tolerance[0],
            rel_tolerance=tolerance[1],
            max_unstable_values=tolerance[2],
            max_unstable_fraction=tolerance[3],
        )

    @abstractmethod
    def tolerance(
        self,
        policy: RuntimeEquivalencePolicy,
        *,
        descriptor: object | None,
    ) -> tuple[float, float, int, float]:
        """Return sparse numeric tolerance settings."""


class ObjectBoundarySparseNumericTolerance(SparseNumericCounterToleranceProfile):
    """Object-boundary sparse jitter tolerance."""

    profile_key = "object_boundary"

    def tolerance(
        self,
        policy: RuntimeEquivalencePolicy,
        *,
        descriptor: object | None,
    ) -> tuple[float, float, int, float]:
        del descriptor
        return (
            policy.object_boundary_jitter_abs_tolerance,
            policy.object_boundary_jitter_rel_tolerance,
            policy.object_boundary_jitter_max_unstable_values,
            policy.object_boundary_jitter_max_unstable_fraction,
        )


class ShapeDescriptorSparseNumericTolerance(SparseNumericCounterToleranceProfile):
    """Shape-descriptor sparse tolerance."""

    profile_key = "shape_descriptor"

    def tolerance(
        self,
        policy: RuntimeEquivalencePolicy,
        *,
        descriptor: object | None,
    ) -> tuple[float, float, int, float]:
        del descriptor
        return (
            policy.shape_descriptor_abs_tolerance,
            policy.shape_descriptor_rel_tolerance,
            policy.shape_descriptor_max_unstable_values,
            policy.shape_descriptor_max_unstable_fraction,
        )


class BinarySparseNumericTolerance(SparseNumericCounterToleranceProfile):
    """Binary numeric sparse tolerance."""

    profile_key = "binary_numeric"

    def tolerance(
        self,
        policy: RuntimeEquivalencePolicy,
        *,
        descriptor: object | None,
    ) -> tuple[float, float, int, float]:
        del descriptor
        return (
            policy.numeric_abs_tolerance,
            policy.numeric_rel_tolerance,
            policy.object_boundary_jitter_max_unstable_values,
            policy.object_boundary_jitter_max_unstable_fraction,
        )


@dataclass(frozen=True, slots=True)
class AggregateMeanKeyProjection:
    """Project row-level object values to image-scoped aggregate mean keys."""

    value_key: RuntimeMeasurementFeatureKey
    required_keys: RuntimeRequiredMeasurementKeys
    key_cache: _AggregateMeanKeyCache

    def key(self) -> RuntimeMeasurementFeatureKey | None:
        cache_key = (
            self.value_key.subject.scope,
            self.value_key.subject.name,
            self.value_key.feature_name,
            self.value_key.statistic,
            self.value_key.source_name,
        )
        mean_key = self.key_cache.get(cache_key, _CACHE_MISS)
        if mean_key is _CACHE_MISS:
            mean_key = RuntimeMeasurementFeatureKey.from_subject_feature(
                self.value_key.subject,
                self.value_key.feature_name,
                MeasurementStatistic.MEAN.value,
                source_name=self.value_key.source_name,
            )
            if self.required_keys is not None and mean_key not in self.required_keys:
                mean_key = None
            elif image_number_reference_feature(self.value_key):
                mean_key = None
            self.key_cache[cache_key] = mean_key
        return mean_key


_AggregateValuesByFeature = dict[
    tuple[RuntimeMeasurementFeatureKey, RuntimeMeasurementRowIdentity],
    RuntimeAggregateMeanAccumulator,
]
_AggregateMeanKeyCache = dict[
    tuple[MeasurementScope, str | None, str, str, str | None],
    RuntimeMeasurementFeatureKey | None,
]


@dataclass(slots=True, kw_only=True)
class RuntimeObjectMeasurementFactRowDomain:
    """Object row identities proven by emitted measurement facts."""

    identities_by_subject: dict[
        RuntimeMeasurementSubjectKey,
        set[RuntimeObjectMeasurementRowIdentity],
    ] = field(default_factory=dict)

    def record_row_facts(
        self,
        row: RuntimeMeasurementRowMapping,
        axis_key: str | None,
        scoped_table: RuntimeScopedMeasurementTable,
        image_number_offset: RuntimeImageNumberOffset,
        policy: RuntimeEquivalencePolicy,
        facts: Iterable[RuntimeMeasurementFact],
    ) -> None:
        if row.identity_role() is MeasurementObjectRowIdentity.ROW_SEQUENCE:
            return
        object_label = row.object_label()
        if object_label is None:
            return
        identity = RuntimeObjectMeasurementRowIdentity.from_object_instance(
            row,
            axis_key,
            policy,
            scoped_table.object_instance_key(
                row,
                object_label,
                image_number_offset=image_number_offset,
            ),
        )
        self.record_row_facts_for_identity(identity, facts)

    def record_row_facts_for_identity(
        self,
        identity: RuntimeObjectMeasurementRowIdentity,
        facts: Iterable[RuntimeMeasurementFact],
    ) -> None:
        subjects = frozenset(
            key.subject
            for key, _value in facts
            if key.subject.scope is MeasurementScope.OBJECT
        )
        for subject in subjects:
            self.identities_by_subject.setdefault(subject, set()).add(identity)

    def record_subject_row_identity(
        self,
        subject: RuntimeMeasurementSubjectKey,
        row_identity: RuntimeMeasurementRowIdentity,
    ) -> None:
        if subject.scope is not MeasurementScope.OBJECT:
            return
        self.identities_by_subject.setdefault(subject, set()).add(
            RuntimeObjectMeasurementRowIdentity(row_identity)
        )

    def record_runtime_row_merge_facts(
        self,
        measurement_fact_counts: RuntimeMeasurementFactCounterMap,
        row_merge_cache: _RuntimeMeasurementRowMergeCache,
        *,
        required_keys: RuntimeRequiredMeasurementKeys,
        policy: RuntimeEquivalencePolicy,
    ) -> None:
        self.record_row_merge_value_facts(
            measurement_fact_counts,
            row_merge_cache,
            required_keys=required_keys,
        )
        aggregate_values_by_identity: dict[
            tuple[RuntimeMeasurementFeatureKey, RuntimeMeasurementRowIdentity],
            RuntimeAggregateMeanAccumulator,
        ] = {}
        aggregate_key_cache: _AggregateMeanKeyCache = {}
        for (
            key,
            row_identity,
        ), (_priority, _row_priority, value) in row_merge_cache.items():
            object_identity = RuntimeObjectMeasurementRowIdentity(row_identity)
            mean_key = AggregateMeanKeyProjection(
                key,
                required_keys,
                aggregate_key_cache,
            ).key()
            value_required = required_keys is None or key in required_keys
            if not value_required and mean_key is None:
                continue
            if mean_key is None:
                continue
            numeric_value = _finite_numeric_runtime_cell_value(value)
            if numeric_value is None:
                continue
            image_row_identity = object_identity.image_identity
            aggregate_identity = (mean_key, image_row_identity)
            accumulator = aggregate_values_by_identity.get(aggregate_identity)
            if accumulator is None:
                accumulator = RuntimeAggregateMeanAccumulator()
                aggregate_values_by_identity[aggregate_identity] = accumulator
            accumulator.add(numeric_value)

        mean_keys = frozenset(
            mean_key for mean_key, _row_identity in aggregate_values_by_identity
        )
        for mean_key in mean_keys:
            measurement_fact_counts.pop(mean_key, None)
        for (
            mean_key,
            _row_identity,
        ), accumulator in aggregate_values_by_identity.items():
            if not accumulator.has_values:
                continue
            runtime_measurement_fact_counter(measurement_fact_counts, mean_key)[
                runtime_cell_signature(str(accumulator.mean), policy)
            ] += 1

    def record_row_merge_value_facts(
        self,
        measurement_fact_counts: RuntimeMeasurementFactCounterMap,
        row_merge_cache: _RuntimeMeasurementRowMergeCache,
        *,
        required_keys: RuntimeRequiredMeasurementKeys,
    ) -> None:
        for (
            key,
            row_identity,
        ), (_priority, _row_priority, value) in row_merge_cache.items():
            if required_keys is not None and key not in required_keys:
                continue
            runtime_measurement_fact_counter(measurement_fact_counts, key)[value] += 1
            self.record_subject_row_identity(key.subject, row_identity)

    def primary_row_keys(self) -> frozenset[_RuntimeMeasurementPrimaryRowKey]:
        return frozenset(
            (subject, identity)
            for subject, identities in self.identities_by_subject.items()
            for identity in identities
            if identity.has_image_identity
        )


def _runtime_aggregate_mean_accumulator(
    measurement_fact_counts: _AggregateValuesByFeature,
    key: RuntimeMeasurementFeatureKey,
    row_identity: RuntimeMeasurementRowIdentity,
) -> RuntimeAggregateMeanAccumulator:
    accumulator_key = (key, row_identity)
    accumulator = measurement_fact_counts.get(accumulator_key)
    if accumulator is None:
        accumulator = RuntimeAggregateMeanAccumulator()
        measurement_fact_counts[accumulator_key] = accumulator
    return accumulator


class RuntimeMeasurementStatisticDependencyStrategy(
    EnumKeyedStrategyMixin[MeasurementStatistic],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Declare input measurement keys required by one output statistic."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    __enum_member_attr__ = "statistic"

    statistic: ClassVar[MeasurementStatistic]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def required_input_keys(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        """Return semantic input keys needed to produce ``key``."""


class ValueMeasurementStatisticDependencyStrategy(
    RuntimeMeasurementStatisticDependencyStrategy
):
    """Value facts are their own projection input."""

    statistic = MeasurementStatistic.VALUE

    def required_input_keys(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        return (key,)


class MeanMeasurementStatisticDependencyStrategy(
    RuntimeMeasurementStatisticDependencyStrategy
):
    """Mean facts depend on row-identifiable value facts."""

    statistic = MeasurementStatistic.MEAN

    def required_input_keys(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        return (
            RuntimeMeasurementFeatureKey(
                key.subject,
                key.feature_name,
                MeasurementStatistic.VALUE.value,
                key.source_name,
            ),
        )


class CountMeasurementStatisticDependencyStrategy(
    RuntimeMeasurementStatisticDependencyStrategy
):
    """Count facts are explicit facts unless higher-level object-count logic applies."""

    statistic = MeasurementStatistic.COUNT

    def required_input_keys(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        return (key,)


def measurement_table_covers_declared_object_domain(
    table: MeasurementTable,
) -> bool:
    """Return whether row owner declares complete object-domain coverage."""
    return (
        isinstance(table.rows, ColumnarRows)
        and table.rows.covers_declared_object_measurement_domain
    )


class RuntimeObjectLocationRowMergeProjectionKey(Enum):
    """Registered object-location row-merge projection identities."""

    LOCATION = "location"
    AGGREGATE_LOCATION = "aggregate_location"


@dataclass(slots=True)
class RuntimeMeasurementObservationAxis:
    """Per-axis runtime measurement artifacts used for semantic projection."""

    axis_key: str
    object_label_records: list[StoredRuntimeValue] = field(default_factory=list)
    relationship_records: list[RuntimeScopedObjectRelationship] = field(
        default_factory=list
    )
    measurement_tables: list[RuntimeScopedMeasurementTable] = field(
        default_factory=list
    )
    seen_relationships: set[RuntimeObjectRelationshipIdentity] = field(
        default_factory=set
    )
    seen_exact_measurement_tables: set[RuntimeMeasurementTableIdentity] = field(
        default_factory=set
    )
    seen_measurement_table_replay_identities: set[
        tuple[RuntimeRecordPlaneIdentity | None, RuntimeMeasurementTableIdentity]
    ] = field(default_factory=set)
    seen_object_subtables: RuntimeMeasurementObjectSubtableSet = field(
        default_factory=set
    )

    def accept_measurement_table(
        self,
        record: StoredRuntimeValue,
        plane_identity_resolver: RuntimeAxisRecordPlaneIdentityResolver,
        seen_aggregate_measurement_tables: set[RuntimeMeasurementTableIdentity],
    ) -> RuntimeScopedMeasurementTable | None:
        record_identity = record.path
        table = MeasurementTable.from_runtime_value(record.value)
        plane_identity = plane_identity_resolver.plane_identity_for_runtime_record(
            record
        )
        replay_identity = (plane_identity, exact_measurement_table_key(table))
        if replay_identity in self.seen_measurement_table_replay_identities:
            return None
        self.seen_measurement_table_replay_identities.add(replay_identity)
        table_spans_multiple_transport_identities = (
            measurement_table_spans_multiple_transport_identities(table)
        )
        if (
            record.key.scope.group_key is None
            or table_spans_multiple_transport_identities
            or measurement_table_covers_declared_object_domain(table)
        ):
            exact_table_key = (
                complete_object_domain_measurement_table_key(table)
                if measurement_table_covers_declared_object_domain(table)
                else exact_measurement_table_key(table)
            )
            if exact_table_key in self.seen_exact_measurement_tables:
                return None
            self.seen_exact_measurement_tables.add(exact_table_key)
            table = dedupe_runtime_measurement_table_object_subtable(
                table,
                self.seen_object_subtables,
            )
        table = dedupe_runtime_measurement_table_aggregate_rows(table)
        aggregate_table_key = aggregate_measurement_table_semantic_key(table)
        if aggregate_table_key is not None:
            if aggregate_table_key in seen_aggregate_measurement_tables:
                return None
            seen_aggregate_measurement_tables.add(aggregate_table_key)
        scoped_table = RuntimeScopedMeasurementTable(
            table,
            plane_identity=plane_identity,
            record_identity=record_identity,
            spans_multiple_transport_identities=(
                table_spans_multiple_transport_identities
            ),
        )
        self.measurement_tables.append(scoped_table)
        return scoped_table

    def accept_object_label_record(self, record: StoredRuntimeValue) -> None:
        """Record an object-label artifact observed on this axis."""
        self.object_label_records.append(record)

    def accept_relationship_record(
        self,
        record: StoredRuntimeValue,
        plane_identity_resolver: RuntimeAxisRecordPlaneIdentityResolver,
    ) -> None:
        """Record a relationship artifact observed on this axis."""
        plane_identity = plane_identity_resolver.plane_identity_for_runtime_record(
            record
        )
        scoped_relationship = RuntimeScopedObjectRelationship(
            ObjectRelationship.from_runtime_value(record.value),
            plane_identity=plane_identity,
        )
        relationship_identity = scoped_relationship.identity_for_projection()
        if relationship_identity in self.seen_relationships:
            return
        self.seen_relationships.add(relationship_identity)
        self.relationship_records.append(scoped_relationship)

    def record_relationship_facts(
        self,
        state: "RuntimeMeasurementProjectionState",
        *,
        explicit_measurement_keys: frozenset[RuntimeMeasurementFeatureKey],
    ) -> None:
        """Record relationship-derived facts from this axis' local artifacts."""
        measurement_fact_counts = state.measurement_fact_counts
        policy = state.policy
        required_measurement_keys = state.required_measurement_keys
        axis_object_label_records = tuple(self.object_label_records)
        object_label_catalog = RuntimeObjectLabelInstanceCatalog.from_records(
            axis_object_label_records
        )
        measurement_tables = tuple(self.measurement_tables)
        object_label_values_by_object: _RuntimeObjectValuesByObject = {}
        child_measurement_values_by_object: _RuntimeObjectValuesByObject = {}

        for scoped_relationship in self.relationship_records:
            relationship = scoped_relationship.relationship_for_projection()
            relationship_measurement = RelationshipMeasurementSemantics(relationship)
            relationship_facts = relationship_measurement.measurement_facts(
                policy,
                object_label_catalog=object_label_catalog,
            )
            record_measurement_facts(
                measurement_fact_counts,
                (
                    (key, value)
                    for key, value in relationship_facts
                    if key not in explicit_measurement_keys
                ),
                required_keys=required_measurement_keys,
            )
            required_child_keys = (
                relationship_measurement.required_child_measurement_keys(
                    required_measurement_keys
                )
            )
            if required_measurement_keys is not None and not required_child_keys:
                continue
            normalized_target_name = normalize_runtime_identifier(
                relationship_measurement.target_name
            )
            child_values_cache_key = (normalized_target_name, required_child_keys)
            child_measurement_values = child_measurement_values_by_object.get(
                child_values_cache_key
            )
            if child_measurement_values is None:
                child_measurement_values = object_measurement_values_by_label(
                    measurement_tables,
                    relationship_measurement.target_name,
                    policy,
                    known_source_names=state.known_source_names,
                    required_keys=required_child_keys,
                )
                object_label_values = object_label_values_by_object.get(
                    child_values_cache_key
                )
                if object_label_values is None:
                    object_label_values = object_label_measurement_values_for_name(
                        axis_object_label_records,
                        relationship_measurement.target_name,
                        policy,
                        required_keys=required_child_keys,
                    )
                    object_label_values_by_object[child_values_cache_key] = (
                        object_label_values
                    )
                for key, values_by_child_id in object_label_values.items():
                    if (
                        required_child_keys is not None
                        and key not in required_child_keys
                    ):
                        continue
                    if key not in child_measurement_values:
                        child_measurement_values[key] = {}
                    child_measurement_values[key].update(values_by_child_id)
                child_measurement_values_by_object[child_values_cache_key] = (
                    child_measurement_values
                )
            aggregate_facts = relationship_measurement.aggregate_measurement_facts(
                child_measurement_values,
                policy,
                object_label_catalog=object_label_catalog,
                existing_measurement_keys=explicit_measurement_keys,
                required_measurement_keys=required_measurement_keys,
            )
            relationship_object_number_aggregate_key = (
                RuntimeMeasurementFeatureKey.from_subject_feature(
                    relationship_measurement.source_subject,
                    relationship_measurement.aggregate_feature_name(
                        ObjectCoreMeasurementFeature.OBJECT_NUMBER.value,
                    ),
                )
            )
            if any(
                key == relationship_object_number_aggregate_key
                for key, _value in aggregate_facts
            ):
                measurement_fact_counts.pop(
                    relationship_object_number_aggregate_key,
                    None,
                )
            record_measurement_facts(
                measurement_fact_counts,
                (
                    (key, value)
                    for key, value in aggregate_facts
                    if key not in explicit_measurement_keys
                    or key == relationship_object_number_aggregate_key
                ),
                required_keys=required_measurement_keys,
            )


def _remove_authoritative_object_label_location_facts(
    measurement_fact_counts: RuntimeMeasurementFactCounterMap,
    subjects: frozenset[RuntimeMeasurementSubjectKey],
    policy: RuntimeEquivalencePolicy,
) -> None:
    """Remove row-derived location facts owned by dimensional object-label payloads."""
    if not subjects:
        return
    for key in tuple(measurement_fact_counts):
        if key.subject not in subjects:
            continue
        location_role_key = RuntimeMeasurementFeatureKey(
            key.subject,
            key.feature_name,
            MeasurementStatistic.VALUE.value,
            key.source_name,
        )
        if object_measurement_feature_matches_marker(
            location_role_key,
            ObjectLocationFeatureMarker,
            policy,
        ):
            measurement_fact_counts.pop(key, None)


def _object_location_subjects_matching_statistic(
    measurement_fact_counts: RuntimeMeasurementFactCounterMap,
    statistic: MeasurementStatistic,
    policy: RuntimeEquivalencePolicy,
) -> frozenset[RuntimeMeasurementSubjectKey]:
    """Return subjects with explicit object-location facts for ``statistic``."""
    return frozenset(
        key.subject
        for key in measurement_fact_counts
        if key.statistic == statistic.value
        and object_measurement_feature_matches_marker(
            key,
            ObjectLocationFeatureMarker,
            policy,
        )
    )


@dataclass(slots=True, kw_only=True)
class RuntimeMeasurementProjectionState(RuntimeObjectMeasurementFactRowDomain):
    """Mutable authority for one runtime measurement projection."""

    policy: RuntimeEquivalencePolicy
    known_source_names: tuple[str, ...]
    required_measurement_keys: RuntimeRequiredMeasurementKeys = None
    measurement_fact_counts: RuntimeMeasurementFactCounterMap = field(
        default_factory=dict
    )
    row_merge_cache: _RuntimeMeasurementRowMergeCache = field(default_factory=dict)
    primary_row_identities: _RuntimeMeasurementPrimaryRowSet = field(
        default_factory=set
    )
    object_label_records: list[StoredRuntimeValue] = field(default_factory=list)
    axis_observations: dict[str, RuntimeMeasurementObservationAxis] = field(
        default_factory=dict
    )
    seen_aggregate_measurement_tables: set[RuntimeMeasurementTableIdentity] = field(
        default_factory=set
    )
    seen_image_feature_fact_records: _RuntimeImageFeatureFactRecordSet = field(
        default_factory=dict
    )
    measurement_fact_counter_object_cache: _RuntimeMeasurementFactCounterObjectCache = (
        field(default_factory=dict)
    )

    def measurement_fact_counter_for_projected_key(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> Counter[RuntimeCellSignature]:
        """Return a fact counter cached for this exact projected key object."""
        cache_key = id(key)
        cached = self.measurement_fact_counter_object_cache.get(cache_key)
        if cached is not None and cached[0] is key:
            return cached[1]
        counter = runtime_measurement_fact_counter(self.measurement_fact_counts, key)
        self.measurement_fact_counter_object_cache[cache_key] = (key, counter)
        return counter

    def record_spatial_grid(self, record: StoredRuntimeValue) -> None:
        record_measurement_facts(
            self.measurement_fact_counts,
            spatial_grid_measurement_facts(record.value, self.policy),
            required_keys=self.required_measurement_keys,
        )

    def record_runtime_aggregate_mean_facts(
        self,
        aggregate_measurement_fact_counts: _AggregateValuesByFeature,
        explicit_measurement_keys: frozenset[RuntimeMeasurementFeatureKey],
    ) -> None:
        for (
            mean_key,
            _row_identity,
        ), accumulator in aggregate_measurement_fact_counts.items():
            if not accumulator.has_values or mean_key in explicit_measurement_keys:
                continue
            if (
                self.required_measurement_keys is not None
                and mean_key not in self.required_measurement_keys
            ):
                continue
            runtime_measurement_fact_counter(self.measurement_fact_counts, mean_key)[
                runtime_cell_signature(str(accumulator.mean), self.policy)
            ] += 1

    def project_recorded_row_fact_counts(self) -> RuntimeMeasurementFactCounterMap:
        """Finalize projected row facts without artifact-owned completions."""
        self.record_runtime_row_merge_facts(
            self.measurement_fact_counts,
            self.row_merge_cache,
            required_keys=self.required_measurement_keys,
            policy=self.policy,
        )
        return self.measurement_fact_counts

    def record_measurement_table(
        self,
        scoped_table: RuntimeScopedMeasurementTable,
        axis_key: str | None,
    ) -> None:
        """Record one runtime measurement table into semantic fact counters."""
        table = scoped_table.table
        schema_cache: RuntimeMeasurementRowSchemaCache = {}
        key_cache: RuntimeMeasurementFeatureKeyCache = {}
        long_form_key_cache: RuntimeMeasurementLongFormKeyCache = {}
        wide_feature_index_cache: (
            equivalence_measurement_rows.RuntimeMeasurementWideFeatureIndexCache
        ) = {}
        wide_feature_plan_cache: (
            equivalence_measurement_rows.RuntimeMeasurementWideFeaturePlanCache
        ) = {}
        qualifier_render_cache: RuntimeMeasurementQualifierRenderCache = {}
        padding_group_cache: RuntimeMeasurementPaddingGroupCache = {}
        collapsed_numeric_qualifier_cache: RuntimeCollapsedNumericQualifierCache = {}
        subject_schema_cache: RuntimeMeasurementRowSubjectSchemaCache = {}
        aggregate_measurement_fact_counts: _AggregateValuesByFeature = {}
        aggregate_input_key_cache: _AggregateMeanKeyCache = {}
        explicit_measurement_keys: set[RuntimeMeasurementFeatureKey] = set()
        row_priority_cache: _RuntimeMeasurementRowPriorityCache = {}
        row_feature_priority_cache: _RuntimeMeasurementRowFeaturePriorityCache = {}
        required_projection = RequiredRuntimeMeasurementProjection(
            self.required_measurement_keys,
            self.policy,
            known_source_names=self.known_source_names,
        )
        row_required_keys = required_projection.input_keys()
        row_required_subjects = required_projection.subjects()
        row_required_key_index = RuntimeMeasurementRequiredKeyIndex.from_required_keys(
            row_required_keys
        )
        row_derive_directional_pair_facts = (
            RuntimeDirectionalPairMeasurementDerivationContract(
                self.policy,
                self.known_source_names,
            ).required_keys_need_derivation(row_required_keys)
        )

        def record_row_mapping(
            row: RuntimeMeasurementRowMapping,
            table_subject: RuntimeMeasurementSubjectKey,
            table_padding_group: str,
            image_number_offset: RuntimeImageNumberOffset,
        ) -> None:
            header = row.header
            row_values = RuntimeIndexedRowValues.from_row(row)
            row_subject_projection = RuntimeMeasurementRowSubjectProjection(
                table_subject,
                table.source_image_name,
                row_values,
                runtime_measurement_row_subject_schema(header, subject_schema_cache),
            )
            subject = row_subject_projection.subject()
            if (
                row_required_subjects is not None
                and subject.scope is MeasurementScope.OBJECT
                and subject not in row_required_subjects
            ):
                return
            source_qualification = subject.bind_row_source_identity(
                row_subject_projection.source_name()
            )
            recorder.record_primary_row_identity(row, subject)
            row_context = RuntimeRowProjectionContext.from_row(
                row,
                subject,
                self.policy,
                source_name=source_qualification.feature_source_name,
                known_source_names=self.known_source_names,
                required_keys=row_required_keys,
                table_padding_group=table_padding_group,
                image_number_offset=image_number_offset,
                derive_directional_pair_facts=row_derive_directional_pair_facts,
                schema_cache=schema_cache,
                key_cache=key_cache,
                long_form_key_cache=long_form_key_cache,
                wide_feature_index_cache=wide_feature_index_cache,
                wide_feature_plan_cache=wide_feature_plan_cache,
                qualifier_render_cache=qualifier_render_cache,
                padding_group_cache=padding_group_cache,
                collapsed_numeric_qualifier_cache=collapsed_numeric_qualifier_cache,
                required_key_index=row_required_key_index,
            )
            row_facts = row_context.facts()
            if row_facts:
                recorder.record_projected_row_facts(row, row_facts)

        table_subject = RuntimeMeasurementSubjectKey.from_table_subject(table.subject)
        table_padding_group = measurement_table_padding_group(table.name)
        image_number_offset = RuntimeImageNumberOffset.from_measurement_table(table)
        recorder = RuntimeTableRowProjectionRecorder(
            state=self,
            image_number_offset=image_number_offset,
            row_priority_cache=row_priority_cache,
            row_feature_priority_cache=row_feature_priority_cache,
            aggregate_measurement_fact_counts=aggregate_measurement_fact_counts,
            aggregate_input_key_cache=aggregate_input_key_cache,
            explicit_measurement_keys=explicit_measurement_keys,
            required_key_index=row_required_key_index,
            axis_key=axis_key,
            scoped_table=scoped_table,
        )
        for row in iter_measurement_rows((table,)):
            record_row_mapping(
                RuntimeMeasurementRowMapping(measurement_row_mapping(row)),
                table_subject,
                table_padding_group,
                image_number_offset,
            )
        self.record_runtime_aggregate_mean_facts(
            aggregate_measurement_fact_counts,
            frozenset(explicit_measurement_keys),
        )

    def project_measurement_fact_counts(self) -> RuntimeMeasurementFactCounterMap:
        """Finalize runtime observation records into semantic measurement facts."""
        measurement_fact_counts = self.measurement_fact_counts
        row_merge_cache = self.row_merge_cache
        primary_row_identities = self.primary_row_identities
        object_label_measurement_authority = (
            RuntimeObjectLabelMeasurementAuthority.from_object_label_records(
                self.object_label_records,
                self.policy,
            )
        )
        self.project_recorded_row_fact_counts()
        explicit_object_location_subjects = _object_location_subjects_matching_statistic(
            measurement_fact_counts,
            MeasurementStatistic.VALUE,
            self.policy,
        )
        explicit_object_location_aggregate_subjects = (
            _object_location_subjects_matching_statistic(
                measurement_fact_counts,
                MeasurementStatistic.MEAN,
                self.policy,
            )
        )
        explicit_location_subjects = (
            explicit_object_location_subjects
            | explicit_object_location_aggregate_subjects
        )
        object_label_owned_location_subjects = (
            object_label_measurement_authority.location_subjects.difference(
                explicit_location_subjects
            )
        )
        _remove_authoritative_object_label_location_facts(
            measurement_fact_counts,
            object_label_owned_location_subjects,
            self.policy,
        )
        object_location_subjects = explicit_object_location_subjects | (
            RuntimeObjectLocationRowMergeContract.registered_projection(
                RuntimeObjectLocationRowMergeProjectionKey.LOCATION,
                self.policy,
            )
            .subjects(row_merge_cache)
            .difference(
                object_label_owned_location_subjects
            )
        )
        object_location_aggregate_subjects = (
            explicit_object_location_aggregate_subjects
            | RuntimeObjectLocationRowMergeContract.registered_projection(
                RuntimeObjectLocationRowMergeProjectionKey.AGGREGATE_LOCATION,
                self.policy,
            )
            .subjects(row_merge_cache)
            .difference(
                object_label_owned_location_subjects
            )
        )
        record_measurement_facts(
            measurement_fact_counts,
            _primary_row_object_count_measurement_facts(
                primary_row_identities,
                row_merge_cache,
                self.policy,
                existing_subjects=(
                    object_label_measurement_authority.primary_row_reserved_count_subjects_from_features(
                        measurement_fact_counts,
                        self.policy,
                    )
                ),
                required_keys=self.required_measurement_keys,
            ),
            required_keys=self.required_measurement_keys,
        )
        primary_row_object_label_completion = ObjectLabelMeasurementCompletion.from_feature_state(
            policy=self.policy,
            measurement_fact_counts=measurement_fact_counts,
            object_identifier_subjects=(
                object_label_measurement_authority.primary_row_reserved_identifier_subjects_from_features(
                    measurement_fact_counts,
                    self.policy,
                )
            ),
            object_location_subjects=object_location_subjects,
            required_keys=self.required_measurement_keys,
            object_location_aggregate_subjects=object_location_aggregate_subjects,
        )
        record_measurement_facts(
            measurement_fact_counts,
            primary_row_object_label_completion.facts_for_primary_rows(
                self.primary_row_keys()
            ),
            required_keys=self.required_measurement_keys,
        )
        object_label_completion = ObjectLabelMeasurementCompletion.from_feature_state(
            policy=self.policy,
            measurement_fact_counts=measurement_fact_counts,
            object_location_subjects=object_location_subjects,
            required_keys=self.required_measurement_keys,
            object_location_aggregate_subjects=object_location_aggregate_subjects,
        )
        record_measurement_facts(
            measurement_fact_counts,
            object_label_completion.facts_for_records(self.object_label_records),
            required_keys=self.required_measurement_keys,
        )
        self.record_relationship_facts(
            explicit_measurement_keys=frozenset(measurement_fact_counts),
        )
        return measurement_fact_counts

    def record_relationship_facts(
        self,
        *,
        explicit_measurement_keys: frozenset[RuntimeMeasurementFeatureKey],
    ) -> None:
        for axis_observation in self.axis_observations.values():
            axis_observation.record_relationship_facts(
                self,
                explicit_measurement_keys=explicit_measurement_keys,
            )


@dataclass(slots=True, kw_only=True)
class RuntimeMeasurementRowProjectionRecorder(ABC):
    """Shared semantic recorder for projected measurement rows."""

    state: RuntimeMeasurementProjectionState
    image_number_offset: RuntimeImageNumberOffset
    row_priority_cache: _RuntimeMeasurementRowPriorityCache
    row_feature_priority_cache: _RuntimeMeasurementRowFeaturePriorityCache
    explicit_measurement_keys: set[RuntimeMeasurementFeatureKey]
    required_key_index: RuntimeMeasurementRequiredKeyIndex
    axis_key: str | None = None

    @abstractmethod
    def object_row_identity(
        self,
        row: RuntimeMeasurementRowMapping,
        subject: RuntimeMeasurementSubjectKey,
    ) -> RuntimeObjectMeasurementRowIdentity | None:
        """Return the object-row identity for this row source."""

    def row_has_primary_object_features(
        self,
        row: RuntimeMeasurementRowMapping,
    ) -> bool:
        primary_location_priority = _object_location_primary_row_priority(
            self.state.policy
        )
        priority = _runtime_row_feature_category_priority(
            row,
            self.state.policy,
            self.row_feature_priority_cache,
        )
        return priority is not None and priority <= primary_location_priority

    def record_primary_row_identity(
        self,
        row: RuntimeMeasurementRowMapping,
        subject: RuntimeMeasurementSubjectKey,
    ) -> None:
        if subject.scope is not MeasurementScope.OBJECT:
            return
        if row.identity_role() is MeasurementObjectRowIdentity.ROW_SEQUENCE:
            return
        if (
            row.identity_role() is not MeasurementObjectRowIdentity.LABEL_ID
            and not self.row_has_primary_object_features(row)
        ):
            return
        object_row_identity = self.object_row_identity(row, subject)
        if object_row_identity is None:
            return
        self.state.primary_row_identities.add((subject, object_row_identity))

    def record_projected_row_facts(
        self,
        row: RuntimeMeasurementRowMapping,
        facts: Iterable[RuntimeMeasurementFact],
    ) -> None:
        object_row_identities: dict[
            RuntimeMeasurementSubjectKey,
            RuntimeObjectMeasurementRowIdentity | None,
        ] = {}
        row_identities: dict[int, RuntimeMeasurementRowIdentityOrMissing] = {}
        emitted_row_facts_by_identity: dict[
            int,
            tuple[RuntimeObjectMeasurementRowIdentity, RuntimeMeasurementFactList],
        ] = {}
        seen_facts: set[RuntimeMeasurementFactWithRowIdentity] = set()
        for key, value, object_row_identity in self.merge_row_measurement_facts(
            row,
            facts,
            object_row_identities,
        ):
            fact_with_identity = (key, value, object_row_identity)
            if fact_with_identity in seen_facts:
                continue
            seen_facts.add(fact_with_identity)
            self.explicit_measurement_keys.add(key)
            if not self.accept_projected_measurement_fact(row, key, value):
                continue
            if key.subject.scope is MeasurementScope.OBJECT:
                object_row_identity = self.object_row_identity_for_subject(
                    row,
                    key.subject,
                    object_row_identities,
                )
            if self.required_key_index.requires_key(key):
                self.state.measurement_fact_counter_for_projected_key(key)[value] += 1
                if object_row_identity is not None:
                    object_row_identity_id = id(object_row_identity)
                    emitted_row_fact_group = emitted_row_facts_by_identity.get(
                        object_row_identity_id
                    )
                    if emitted_row_fact_group is None:
                        emitted_row_fact_group = (object_row_identity, [])
                        emitted_row_facts_by_identity[object_row_identity_id] = (
                            emitted_row_fact_group
                        )
                    emitted_row_fact_group[1].append((key, value))
            object_row_identity_id = id(object_row_identity)
            row_identity = self.record_projected_fact_side_effects(
                key,
                value,
                row,
                object_row_identity,
                row_identity=row_identities.get(object_row_identity_id),
            )
            row_identities[object_row_identity_id] = row_identity
        for (
            object_row_identity,
            emitted_row_facts,
        ) in emitted_row_facts_by_identity.values():
            self.state.record_row_facts_for_identity(
                object_row_identity,
                emitted_row_facts,
            )

    def merge_row_measurement_facts(
        self,
        row: RuntimeMeasurementRowMapping,
        facts: Iterable[RuntimeMeasurementFact],
        object_row_identities: dict[
            RuntimeMeasurementSubjectKey,
            RuntimeObjectMeasurementRowIdentity | None,
        ],
    ) -> RuntimeMeasurementFactsWithRowIdentity:
        remaining_facts: list[RuntimeMeasurementFactWithRowIdentity] = []
        row_merge_contract = RuntimeObjectLocationRowMergeContract(self.state.policy)
        for key, value in facts:
            if not row_merge_contract.owns_key(key):
                remaining_facts.append((key, value, None))
                continue
            object_row_identity = self.object_row_identity_for_subject(
                row,
                key.subject,
                object_row_identities,
            )
            if object_row_identity is None:
                remaining_facts.append((key, value, None))
                continue
            merge_key = (key, object_row_identity.row_identity)
            priority = _runtime_row_measurement_fact_priority(
                row,
                key,
                self.state.policy,
                self.row_priority_cache,
            )
            candidate = (
                priority,
                priority,
                value,
            )
            current = self.state.row_merge_cache.get(merge_key)
            if current is None or _runtime_row_merge_candidate_preferred(
                candidate,
                current,
            ):
                self.state.row_merge_cache[merge_key] = (
                    candidate[0],
                    candidate[1] if current is None else min(candidate[1], current[1]),
                    candidate[2],
                )
            elif current is not None:
                self.state.row_merge_cache[merge_key] = (
                    current[0],
                    min(current[1], priority),
                    current[2],
                )
        return tuple(remaining_facts)

    def object_row_identity_for_subject(
        self,
        row: RuntimeMeasurementRowMapping,
        subject: RuntimeMeasurementSubjectKey,
        object_row_identities: dict[
            RuntimeMeasurementSubjectKey,
            RuntimeObjectMeasurementRowIdentity | None,
        ],
    ) -> RuntimeObjectMeasurementRowIdentity | None:
        if subject not in object_row_identities:
            object_row_identities[subject] = self.object_row_identity(row, subject)
        return object_row_identities[subject]

    def record_projected_fact_side_effects(
        self,
        key: RuntimeMeasurementFeatureKey,
        value: RuntimeCellSignature,
        row: RuntimeMeasurementRowMapping,
        object_row_identity: RuntimeObjectMeasurementRowIdentity | None,
        *,
        row_identity: RuntimeMeasurementRowIdentityOrMissing,
    ) -> RuntimeMeasurementRowIdentityOrMissing:
        del key, value, row, object_row_identity
        return row_identity

    def accept_projected_measurement_fact(
        self,
        row: RuntimeMeasurementRowMapping,
        key: RuntimeMeasurementFeatureKey,
        value: RuntimeCellSignature,
    ) -> bool:
        del row, key, value
        return True


@dataclass(slots=True, kw_only=True)
class RuntimeTableRowProjectionRecorder(RuntimeMeasurementRowProjectionRecorder):
    """Projection recorder for typed runtime measurement tables."""

    scoped_table: RuntimeScopedMeasurementTable
    aggregate_measurement_fact_counts: _AggregateValuesByFeature
    aggregate_input_key_cache: _AggregateMeanKeyCache

    @staticmethod
    def _image_feature_duplicate_plane_identity(
        plane_identity: RuntimeRecordPlaneIdentity | None,
    ) -> RuntimeRecordPlaneIdentity | None:
        if (
            plane_identity is not None
            and plane_identity.authority
            is RuntimeRecordPlaneIdentityAuthority.FILL_MISSING_ROW_IDENTITY
        ):
            return None
        return plane_identity

    def object_row_identity_for_subject(
        self,
        row: RuntimeMeasurementRowMapping,
        subject: RuntimeMeasurementSubjectKey,
        object_row_identities: dict[
            RuntimeMeasurementSubjectKey,
            RuntimeObjectMeasurementRowIdentity | None,
        ],
    ) -> RuntimeObjectMeasurementRowIdentity | None:
        if object_row_identities:
            return next(iter(object_row_identities.values()))
        object_row_identity = self.object_row_identity(row, subject)
        object_row_identities[subject] = object_row_identity
        return object_row_identity

    def object_row_identity(
        self,
        row: RuntimeMeasurementRowMapping,
        subject: RuntimeMeasurementSubjectKey,
    ) -> RuntimeObjectMeasurementRowIdentity | None:
        del subject
        object_label = row.object_label()
        if object_label is None:
            return None
        return RuntimeObjectMeasurementRowIdentity.from_object_instance(
            row,
            self.axis_key,
            self.state.policy,
            self.scoped_table.object_instance_key(
                row,
                object_label,
                image_number_offset=self.image_number_offset,
            ),
        )

    def record_projected_fact_side_effects(
        self,
        key: RuntimeMeasurementFeatureKey,
        value: RuntimeCellSignature,
        row: RuntimeMeasurementRowMapping,
        object_row_identity: RuntimeObjectMeasurementRowIdentity | None,
        *,
        row_identity: RuntimeMeasurementRowIdentityOrMissing,
    ) -> RuntimeMeasurementRowIdentityOrMissing:
        if key.subject.scope is not MeasurementScope.OBJECT:
            return row_identity
        if key.statistic != "value":
            return row_identity
        mean_key = AggregateMeanKeyProjection(
            key,
            self.state.required_measurement_keys,
            self.aggregate_input_key_cache,
        ).key()
        if mean_key is None:
            return row_identity
        numeric_value = _finite_numeric_runtime_cell_value(value)
        if numeric_value is None:
            return row_identity
        if row_identity is None:
            if object_row_identity is None:
                row_identity = row.axis_scoped_identity(
                    self.axis_key,
                    self.state.policy.measurement_dialect,
                )
            else:
                row_identity = object_row_identity.image_identity
        _runtime_aggregate_mean_accumulator(
            self.aggregate_measurement_fact_counts,
            mean_key,
            row_identity,
        ).add(numeric_value)
        return row_identity

    def accept_projected_measurement_fact(
        self,
        row: RuntimeMeasurementRowMapping,
        key: RuntimeMeasurementFeatureKey,
        value: RuntimeCellSignature,
    ) -> bool:
        if key.subject.scope is not MeasurementScope.IMAGE:
            return True
        image_identity = row.axis_scoped_identity(
            self.axis_key,
            self.state.policy.measurement_dialect,
        )
        identity = (
            image_identity,
            self._image_feature_duplicate_plane_identity(
                self.scoped_table.plane_identity
            ),
            self.scoped_table.table.source_provenance.equality_identity,
            key,
            value,
        )
        record_identities = self.state.seen_image_feature_fact_records.get(identity)
        record_identity = self.scoped_table.record_identity
        if record_identities is None:
            self.state.seen_image_feature_fact_records[identity] = {record_identity}
            return True
        if record_identity not in record_identities:
            record_identities.add(record_identity)
            return False
        return True


@dataclass(slots=True, kw_only=True)
class RuntimeExportTableRowProjectionRecorder(RuntimeMeasurementRowProjectionRecorder):
    """Projection recorder for exported measurement-table snapshots."""

    table: RuntimeTableSnapshot
    contextual_identity_candidates_by_subject: dict[
        RuntimeMeasurementSubjectKey,
        tuple[RuntimeExportContextualObjectIdentityCandidate, ...],
    ] = field(default_factory=dict)

    def object_row_identity(
        self,
        row: RuntimeMeasurementRowMapping,
        subject: RuntimeMeasurementSubjectKey,
    ) -> RuntimeObjectMeasurementRowIdentity | None:
        if subject.scope is not MeasurementScope.OBJECT:
            return None
        object_label = self.contextual_object_label(row, subject)
        if object_label is None:
            object_label = row.object_label()
        if object_label is None:
            return None
        return RuntimeObjectMeasurementRowIdentity.from_object_instance(
            row,
            self.axis_key,
            self.state.policy,
            ObjectInstanceKey.from_measurement_row(
                row.row,
                object_label,
                image_number_offset=self.image_number_offset.value,
            ),
        )

    def contextual_object_label(
        self,
        row: RuntimeMeasurementRowMapping,
        subject: RuntimeMeasurementSubjectKey,
    ) -> int | None:
        for candidate in self.contextual_object_identity_candidates(subject):
            object_label = measurement_object_label(
                row.row,
                object_id_field=candidate.field_name,
            )
            if object_label is not None:
                return object_label
        return None

    def contextual_object_identity_candidates(
        self,
        subject: RuntimeMeasurementSubjectKey,
    ) -> tuple[RuntimeExportContextualObjectIdentityCandidate, ...]:
        candidates = self.contextual_identity_candidates_by_subject.get(subject)
        if candidates is not None:
            return candidates
        if not self.table.column_context:
            candidates = ()
        else:
            candidates = tuple(
                sorted(
                    (
                        candidate
                        for index, context in enumerate(self.table.column_context)
                        for candidate in (
                            RuntimeExportContextualObjectIdentityField(
                                table=self.table,
                                index=index,
                                context=context,
                                subject=subject,
                                policy=self.state.policy,
                                known_source_names=self.state.known_source_names,
                            ).candidate(),
                        )
                        if candidate is not None
                    ),
                    key=lambda item: item.specificity.value,
                )
            )
        self.contextual_identity_candidates_by_subject[subject] = candidates
        return candidates


class RuntimeMeasurementObservationRecordHandler(
    ArtifactTypeStrategyMatchMixin,
    MostDerivedContextStrategyMixin[type[ArtifactType]],
    ABC,
):
    """Artifact-type handler for runtime measurement observation recording."""

    artifact_type: ClassVar[type[ArtifactType] | None] = None

    @classmethod
    def for_artifact_type(
        cls,
        artifact_type: ArtifactType,
    ) -> "RuntimeMeasurementObservationRecordHandler | None":
        return cls.for_context(ArtifactType.coerce(artifact_type), required=False)

    @abstractmethod
    def record(
        self,
        state: RuntimeMeasurementProjectionState,
        axis_observation: RuntimeMeasurementObservationAxis,
        record: StoredRuntimeValue,
        plane_identity_resolver: RuntimeAxisRecordPlaneIdentityResolver,
    ) -> None:
        """Record one runtime artifact into measurement-observation state."""


class SpatialGridObservationRecordHandler(RuntimeMeasurementObservationRecordHandler):
    """Record spatial-grid artifacts as direct measurement facts."""

    artifact_type = SpatialGridArtifactType

    def record(
        self,
        state: RuntimeMeasurementProjectionState,
        axis_observation: RuntimeMeasurementObservationAxis,
        record: StoredRuntimeValue,
        plane_identity_resolver: RuntimeAxisRecordPlaneIdentityResolver,
    ) -> None:
        del axis_observation, plane_identity_resolver
        state.record_spatial_grid(record)


class MeasurementTableObservationRecordHandler(
    RuntimeMeasurementObservationRecordHandler
):
    """Record measurement-table artifacts for fact projection."""

    artifact_type = MeasurementsArtifactType

    def record(
        self,
        state: RuntimeMeasurementProjectionState,
        axis_observation: RuntimeMeasurementObservationAxis,
        record: StoredRuntimeValue,
        plane_identity_resolver: RuntimeAxisRecordPlaneIdentityResolver,
    ) -> None:
        scoped_table = axis_observation.accept_measurement_table(
            record,
            plane_identity_resolver,
            state.seen_aggregate_measurement_tables,
        )
        if scoped_table is None:
            return
        state.record_measurement_table(
            scoped_table,
            axis_observation.axis_key,
        )


class ObjectLabelsObservationRecordHandler(RuntimeMeasurementObservationRecordHandler):
    """Record object-label artifacts for object-domain completion."""

    artifact_type = ObjectLabelsArtifactType

    def record(
        self,
        state: RuntimeMeasurementProjectionState,
        axis_observation: RuntimeMeasurementObservationAxis,
        record: StoredRuntimeValue,
        plane_identity_resolver: RuntimeAxisRecordPlaneIdentityResolver,
    ) -> None:
        del plane_identity_resolver
        state.object_label_records.append(record)
        axis_observation.accept_object_label_record(record)


class RelationshipObservationRecordHandler(RuntimeMeasurementObservationRecordHandler):
    """Record relationship artifacts for axis-local relationship projection."""

    artifact_type = RelationshipsArtifactType

    def record(
        self,
        state: RuntimeMeasurementProjectionState,
        axis_observation: RuntimeMeasurementObservationAxis,
        record: StoredRuntimeValue,
        plane_identity_resolver: RuntimeAxisRecordPlaneIdentityResolver,
    ) -> None:
        del state
        axis_observation.accept_relationship_record(record, plane_identity_resolver)


def _finite_numeric_runtime_cell_value(value: RuntimeCellSignature) -> float | None:
    if value.kind is not RuntimeCellValueKind.NUMBER:
        return None
    numeric_value = float(value.value)
    return numeric_value if math.isfinite(numeric_value) else None


@dataclass(frozen=True, slots=True)
class TieSensitiveLocationValueFeatureContext:
    """Typed context for resolving max-location dependency features."""

    feature: RuntimeMeasurementFeatureKey
    policy: RuntimeEquivalencePolicy

    @property
    def feature_parts(self) -> tuple[str, ...]:
        return tuple(
            part
            for part in normalize_runtime_identifier(self.feature.feature_name).split(
                "_"
            )
            if part
        )

    def location_feature_family(
        self,
        feature_name: str,
    ) -> RuntimeMeasurementSourceQualifiedFeature | None:
        """Return the source-qualified location family for one feature name."""
        feature_family = (
            self.policy.measurement_dialect.source_feature_family_for_relation(
                measurement_features.TieSensitiveLocationValueFeatureRelation,
                feature_name,
                self.feature.source_name,
                self.feature.subject.scope,
            )
        )
        if feature_family is None:
            return None
        if not isinstance(feature_family, RuntimeMeasurementSourceQualifiedFeature):
            raise TypeError(
                "RuntimeMeasurementFeature relation returned an invalid "
                "source-qualified feature family."
            )
        return feature_family

    def value_key_from_location_family(
        self,
        feature_family: RuntimeMeasurementSourceQualifiedFeature,
    ) -> RuntimeMeasurementFeatureKey:
        """Return direct value key for one source-qualified location family."""
        value_family = (
            self.policy.measurement_dialect.target_family_for_relation_source_family(
                measurement_features.TieSensitiveLocationValueFeatureRelation,
                feature_family.feature_name,
            )
        )
        if value_family is None:
            raise ValueError(
                "RuntimeMeasurementFeature relation lost target-family ownership "
                f"for {feature_family.feature_name!r}."
            )
        value_identity = (
            self.policy.measurement_dialect.encode_source_qualified_feature(
                value_family,
                feature_family.source_name,
                self.feature.subject.scope,
            )
        )
        return RuntimeMeasurementFeatureKey(
            subject=self.feature.subject,
            feature_name=value_identity.feature_name,
            statistic=self.feature.statistic,
            source_name=value_identity.source_name,
        )


@dataclass(frozen=True, slots=True)
class AggregateTieSensitiveLocationFeature:
    """Parsed aggregate max-location feature identity."""

    aggregate: str
    object_name_parts: tuple[str, ...]
    feature_parts: tuple[str, ...]
    feature_family: RuntimeMeasurementSourceQualifiedFeature

    @classmethod
    def from_context(
        cls,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> "AggregateTieSensitiveLocationFeature | None":
        """Parse an aggregate location feature, if the context declares one."""
        for identity in RuntimeAggregateFeatureIdentity.candidates_from_parts(
            context.feature_parts
        ):
            feature_family = context.location_feature_family(identity.feature_name)
            if feature_family is None:
                continue
            return cls(
                aggregate=identity.aggregate,
                object_name_parts=identity.object_name_parts,
                feature_parts=identity.feature_parts,
                feature_family=feature_family,
            )
        return None

    @property
    def feature_name(self) -> str:
        return "_".join(self.feature_parts)

    def value_key_from_location_family(
        self,
        context: TieSensitiveLocationValueFeatureContext,
        feature_family: RuntimeMeasurementSourceQualifiedFeature,
    ) -> RuntimeMeasurementFeatureKey:
        """Return aggregate value key corresponding to a max-location aggregate."""
        direct_value_key = context.value_key_from_location_family(feature_family)
        return RuntimeMeasurementFeatureKey(
            subject=context.feature.subject,
            feature_name="_".join(
                (
                    self.aggregate,
                    *self.object_name_parts,
                    direct_value_key.feature_name,
                )
            ),
            statistic=context.feature.statistic,
            source_name=direct_value_key.source_name,
        )


class TieSensitiveLocationValueFeatureStrategy(
    MostDerivedContextStrategyMixin[TieSensitiveLocationValueFeatureContext],
    ABC,
):
    """Registered resolver for location-feature value dependencies."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_KEY)

    strategy_key: ClassVar[str | None] = None

    @classmethod
    def value_feature_key(
        cls,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> RuntimeMeasurementFeatureKey | None:
        """Resolve value dependency through most-derived registered strategy."""
        strategy = cls.for_context(
            context,
            required=False,
            error_subject="Tie-sensitive location value feature resolution",
        )
        return None if strategy is None else strategy.resolve(context)

    @abstractmethod
    def resolve(
        self,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> RuntimeMeasurementFeatureKey:
        """Return the value feature that gates location tie mismatches."""


class DirectTieSensitiveLocationValueFeatureStrategy(
    TieSensitiveLocationValueFeatureStrategy
):
    """Resolve direct max-location features to direct max-intensity values."""

    strategy_key = "direct"

    def matches(
        self,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> bool:
        return context.location_feature_family(context.feature.feature_name) is not None

    def resolve(
        self,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> RuntimeMeasurementFeatureKey:
        feature_family = context.location_feature_family(context.feature.feature_name)
        if feature_family is None:
            raise ValueError("Direct tie-sensitive location strategy lost ownership.")
        return context.value_key_from_location_family(feature_family)


class AggregateTieSensitiveLocationValueFeatureStrategy(
    TieSensitiveLocationValueFeatureStrategy
):
    """Resolve aggregate max-location features to aggregate max-intensity values."""

    strategy_key = "aggregate"

    def matches(
        self,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> bool:
        return AggregateTieSensitiveLocationFeature.from_context(context) is not None

    def resolve(
        self,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> RuntimeMeasurementFeatureKey:
        aggregate_feature = AggregateTieSensitiveLocationFeature.from_context(context)
        if aggregate_feature is None:
            raise ValueError(
                "Aggregate tie-sensitive location strategy lost ownership."
            )
        return aggregate_feature.value_key_from_location_family(
            context,
            aggregate_feature.feature_family,
        )


@dataclass(frozen=True, slots=True)
class TieSensitiveLocationFeatureContract:
    """Semantic value dependency for tie-sensitive max-location features."""

    feature: RuntimeMeasurementFeatureKey
    policy: RuntimeEquivalencePolicy

    def value_feature_key(self) -> RuntimeMeasurementFeatureKey | None:
        """Return the value feature that makes this location tie-insensitive."""
        return TieSensitiveLocationValueFeatureStrategy.value_feature_key(
            TieSensitiveLocationValueFeatureContext(
                feature=self.feature,
                policy=self.policy,
            )
        )


@dataclass(frozen=True, slots=True)
class ShapeDescriptorFeatureContext:
    """Measurement feature context for unstable shape descriptor equivalence."""

    feature: RuntimeMeasurementFeatureKey
    policy: RuntimeEquivalencePolicy

    def semantic_context(self) -> "ShapeDescriptorFeatureContext":
        """Return the descriptor context that owns comparison semantics."""
        child_feature_name = (
            RelationshipAggregateFeatureSemantics.aggregate_child_feature_name_from_key(
                self.feature,
                self.policy.measurement_dialect,
            )
        )
        if child_feature_name is None:
            return self
        return ShapeDescriptorFeatureContext(
            RuntimeMeasurementFeatureKey(
                subject=self.feature.subject,
                feature_name=child_feature_name,
                statistic=self.feature.statistic,
                source_name=self.feature.source_name,
            ),
            self.policy,
        )


class ShapeDescriptorFeatureSemantics(
    MostDerivedContextStrategyMixin[ShapeDescriptorFeatureContext],
    ABC,
):
    """Classify direct and derived shape descriptor measurement features."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_KEY)

    strategy_key: ClassVar[str | None] = None

    @abstractmethod
    def matches(self, context: ShapeDescriptorFeatureContext) -> bool:
        """Return whether this strategy owns the feature."""

    @abstractmethod
    def values_equivalent(
        self,
        context: ShapeDescriptorFeatureContext,
        reference_values: Counter[RuntimeCellSignature],
        candidate_values: Counter[RuntimeCellSignature],
    ) -> bool:
        """Return whether descriptor values are equivalent under policy."""

    def boundary_jitter_values_equivalent(
        self,
        context: ShapeDescriptorFeatureContext,
        reference_values: Counter[RuntimeCellSignature],
        candidate_values: Counter[RuntimeCellSignature],
    ) -> bool:
        """Return sparse-boundary equivalence for descriptor values."""
        return ObjectBoundarySparseNumericTolerance.equivalent(
            reference_values,
            candidate_values,
            context.policy,
        )


class AngularShapeDescriptorFeatureSemantics(ShapeDescriptorFeatureSemantics, ABC):
    """Shape descriptors whose values live on a circular angular domain."""

    def values_equivalent(
        self,
        context: ShapeDescriptorFeatureContext,
        reference_values: Counter[RuntimeCellSignature],
        candidate_values: Counter[RuntimeCellSignature],
    ) -> bool:
        if context.policy.allow_sparse_object_boundary_jitter:
            return _sparse_absolute_numeric_counters_equivalent(
                reference_values,
                candidate_values,
                context.policy,
                abs_tolerance=context.policy.object_boundary_jitter_abs_tolerance,
                rel_tolerance=context.policy.object_boundary_jitter_rel_tolerance,
                max_unstable_values=context.policy.object_boundary_jitter_max_unstable_values,
                max_unstable_fraction=context.policy.object_boundary_jitter_max_unstable_fraction,
            )
        return _absolute_numeric_counters_equivalent(
            reference_values,
            candidate_values,
            context.policy,
        )


@dataclass(slots=True, kw_only=True)
class RuntimeMeasurementObservationProjector(RuntimeMeasurementProjectionState):
    """Project runtime artifact observations into mutable measurement facts."""

    observation: RuntimeArtifactExecutionObservation

    def record_artifacts(self) -> None:
        for axis_key, records in self.observation.records_by_axis.items():
            self._record_axis(axis_key, tuple(records))

    def _record_axis(
        self,
        axis_key: str,
        axis_records: tuple[StoredRuntimeValue, ...],
    ) -> None:
        plane_identity_resolver = RuntimeAxisRecordPlaneIdentityResolver.from_records(
            axis_records
        )
        axis_observation = RuntimeMeasurementObservationAxis(axis_key)
        self.axis_observations[axis_key] = axis_observation
        for record in axis_records:
            record_handler = (
                RuntimeMeasurementObservationRecordHandler.for_artifact_type(
                    record.key.artifact_type
                )
            )
            if record_handler is not None:
                record_handler.record(
                    self,
                    axis_observation,
                    record,
                    plane_identity_resolver,
                )

    def project_measurement_fact_counts(self) -> RuntimeMeasurementFactCounterMap:
        """Project all runtime observation records into semantic measurement facts."""
        self.record_artifacts()
        return RuntimeMeasurementProjectionState.project_measurement_fact_counts(self)


RuntimeMeasurementSubjectCachePayload = tuple[str, str | None]
RuntimeMeasurementFeatureCachePayload = tuple[
    RuntimeMeasurementSubjectCachePayload,
    str,
    str,
    str | None,
]
RuntimeCellSignatureCachePayload = tuple[str, str]
RuntimeMeasurementSnapshotCachePayload = tuple[
    tuple[
        RuntimeMeasurementFeatureCachePayload,
        tuple[tuple[RuntimeCellSignatureCachePayload, int], ...],
    ],
    ...,
]


@dataclass(slots=True)
class RuntimeMeasurementSnapshot:
    """Semantic measurement facts independent of table layout."""

    measurement_fact_counts: RuntimeMeasurementFactCounterMapping

    @classmethod
    def from_output_snapshot(
        cls,
        snapshot: "RuntimeOutputSnapshot",
        *,
        policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
        known_source_names: tuple[str, ...] = (),
    ) -> "RuntimeMeasurementSnapshot":
        """Project exported tables into semantic measurement facts."""
        state = RuntimeMeasurementProjectionState(
            policy=policy,
            known_source_names=known_source_names,
        )
        for table in snapshot.tables:
            RuntimeTableSnapshotFactExtractor(
                table,
                known_source_names=known_source_names,
                policy=policy,
            ).record_measurement_table(state)
        return cls(measurement_fact_counts=state.project_recorded_row_fact_counts())

    @classmethod
    def from_artifact_execution_observation(
        cls,
        observation: RuntimeArtifactExecutionObservation,
        *,
        policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
        known_source_names: tuple[str, ...] = (),
        required_measurement_keys: RuntimeRequiredMeasurementKeys = None,
    ) -> "RuntimeMeasurementSnapshot":
        """Project typed runtime measurement artifacts into semantic facts."""
        return cls(
            measurement_fact_counts=RuntimeMeasurementObservationProjector(
                observation=observation,
                policy=policy,
                known_source_names=known_source_names,
                required_measurement_keys=required_measurement_keys,
            ).project_measurement_fact_counts()
        )

    def __post_init__(self) -> None:
        self.measurement_fact_counts = MappingProxyType(
            {
                key: Counter(values)
                for key, values in self.measurement_fact_counts.items()
            }
        )

    @property
    def is_empty(self) -> bool:
        return not self.measurement_fact_counts

    def to_cache_payload(
        self,
    ) -> RuntimeMeasurementSnapshotCachePayload:
        """Return a stable semantic cache payload for repeated equivalence checks."""
        return tuple(
            (
                feature.to_cache_payload(),
                tuple(
                    (value.to_cache_payload(), int(count))
                    for value, count in sorted(
                        values.items(),
                        key=lambda item: item[0].sort_key,
                    )
                ),
            )
            for feature, values in sorted(
                self.measurement_fact_counts.items(),
                key=lambda item: item[0].sort_key,
            )
        )

    @classmethod
    def from_cache_payload(
        cls,
        payload: RuntimeMeasurementSnapshotCachePayload,
    ) -> "RuntimeMeasurementSnapshot":
        """Rebuild a semantic measurement snapshot from cache payload data."""
        measurement_fact_counts: RuntimeMeasurementFactCounterMap = {}
        for feature_payload, values_payload in payload:  # type: ignore[union-attr]
            counter: Counter[RuntimeCellSignature] = Counter()
            for value_payload, count in values_payload:
                counter[RuntimeCellSignature.from_cache_payload(value_payload)] = int(
                    count
                )
            measurement_fact_counts[
                RuntimeMeasurementFeatureKey.from_cache_payload(feature_payload)
            ] = counter
        return cls(measurement_fact_counts=measurement_fact_counts)


@dataclass(slots=True)
class RuntimeMeasurementSnapshotAccumulator:
    """Accumulate semantic measurement facts from independently executed windows."""

    _measurement_fact_counts: RuntimeMeasurementFactCounterMap = field(
        default_factory=dict
    )

    def add(self, snapshot: RuntimeMeasurementSnapshot) -> None:
        """Merge one projected runtime window into this semantic accumulator."""
        for feature, values in snapshot.measurement_fact_counts.items():
            runtime_measurement_fact_counter(
                self._measurement_fact_counts,
                feature,
            ).update(values)

    def snapshot(self) -> RuntimeMeasurementSnapshot:
        """Freeze the accumulated semantic facts for equivalence comparison."""
        return RuntimeMeasurementSnapshot(
            measurement_fact_counts=self._measurement_fact_counts
        )


def runtime_output_equivalence(
    reference: RuntimeOutputSnapshot,
    candidate: RuntimeOutputSnapshot,
    *,
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
) -> RuntimeEquivalenceReport:
    """Compare two runtime output snapshots for semantic equivalence."""
    return RuntimeEquivalenceReport(
        differences=(
            *_table_differences(reference.tables, candidate.tables, policy),
            *_image_differences(reference.images, candidate.images, policy),
        )
    )


def runtime_output_root_equivalence(
    reference_output_root: Path,
    candidate_output_root: Path,
    *,
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
) -> RuntimeEquivalenceReport:
    """Compare two runtime output directories for semantic equivalence."""
    return runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_output_root),
        RuntimeOutputSnapshot.from_output_root(candidate_output_root),
        policy=policy,
    )


def runtime_measurement_equivalence(
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    *,
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
) -> RuntimeEquivalenceReport:
    """Compare precomputed semantic measurement snapshots."""
    return RuntimeEquivalenceReport(
        differences=_measurement_differences(reference, candidate, policy)
    )


def runtime_artifact_measurement_source_names(
    observation: RuntimeArtifactExecutionObservation,
) -> tuple[str, ...]:
    """Return source-image names carried by typed runtime artifacts."""
    return _measurement_source_names_from_artifact_execution(observation)


def runtime_reference_artifact_equivalence(
    reference: RuntimeOutputSnapshot,
    candidate: RuntimeArtifactExecutionObservation,
    *,
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
    candidate_image_artifact_names: frozenset[str] = frozenset(),
    candidate_image_export_specs: tuple[RuntimeImageExportSpec, ...] = (),
    candidate_image_snapshots: tuple[RuntimeImageSnapshot, ...] | None = None,
) -> RuntimeEquivalenceReport:
    """Compare an external output reference to typed runtime artifact execution."""
    known_source_names = runtime_artifact_measurement_source_names(candidate)
    reference_measurements = RuntimeMeasurementSnapshot.from_output_snapshot(
        reference,
        policy=policy,
        known_source_names=known_source_names,
    )
    candidate_measurements = (
        RuntimeMeasurementSnapshot.from_artifact_execution_observation(
            candidate,
            policy=policy,
            known_source_names=known_source_names,
            required_measurement_keys=frozenset(
                reference_measurements.measurement_fact_counts
            ),
        )
    )
    if reference_measurements.is_empty and candidate_measurements.is_empty:
        table_differences = (
            ()
            if not reference.tables
            else _table_differences(
                reference.tables,
                RuntimeOutputSnapshot.from_artifact_execution_observation(
                    candidate
                ).tables,
                policy,
            )
        )
    else:
        table_differences = runtime_measurement_equivalence(
            reference_measurements,
            candidate_measurements,
            policy=policy,
        ).differences

    image_differences = ()
    if reference.images:
        candidate_images = (
            candidate_image_snapshots
            if candidate_image_snapshots is not None
            else RuntimeOutputSnapshot.from_artifact_execution_observation(
                candidate,
                image_artifact_names=candidate_image_artifact_names,
                image_export_specs=candidate_image_export_specs,
            ).images
        )
        image_differences = _image_differences(
            reference.images,
            candidate_images,
            policy,
        )

    return RuntimeEquivalenceReport(
        differences=(
            *table_differences,
            *image_differences,
        )
    )


def runtime_artifact_execution_equivalence(
    reference: RuntimeArtifactExecutionObservation,
    candidate: RuntimeArtifactExecutionObservation,
    *,
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
) -> RuntimeEquivalenceReport:
    """Compare runtime artifact state and file outputs for semantic equivalence."""
    return RuntimeEquivalenceReport(
        differences=(
            *_runtime_artifact_count_differences(reference, candidate),
            *runtime_output_equivalence(
                RuntimeOutputSnapshot.from_artifact_execution_observation(reference),
                RuntimeOutputSnapshot.from_artifact_execution_observation(candidate),
                policy=policy,
            ).differences,
        )
    )


def _runtime_artifact_count_differences(
    reference: RuntimeArtifactExecutionObservation,
    candidate: RuntimeArtifactExecutionObservation,
) -> tuple[RuntimeEquivalenceDifference, ...]:
    reference_counts = _total_record_counts(reference)
    candidate_counts = _total_record_counts(candidate)
    if reference_counts == candidate_counts:
        return ()
    return (
        RuntimeEquivalenceDifference(
            RuntimeEquivalenceDifferenceKind.RUNTIME_ARTIFACT_COUNTS,
            "runtime artifact counts differ: "
            f"reference={_artifact_type_counts_repr(reference_counts)}, "
            f"candidate={_artifact_type_counts_repr(candidate_counts)}",
        ),
    )


def _artifact_type_counts_repr(counts: Mapping[type[ArtifactType], int]) -> str:
    rendered_items = (
        f"{artifact_type.diagnostic_label()}: {count}"
        for artifact_type, count in sorted(
            (
                (ArtifactType.coerce(artifact_type), count)
                for artifact_type, count in counts.items()
            ),
            key=lambda item: item[0].require_value(),
        )
    )
    return "{" + ", ".join(rendered_items) + "}"


def _total_record_counts(
    observation: RuntimeArtifactExecutionObservation,
) -> Counter[type[ArtifactType]]:
    counts: Counter[type[ArtifactType]] = Counter()
    for axis_counts in observation.record_counts_by_axis.values():
        counts.update(axis_counts)
    return counts


def _measurement_differences(
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> tuple[RuntimeEquivalenceDifference, ...]:
    differences: list[RuntimeEquivalenceDifference] = []
    reference_features = set(reference.measurement_fact_counts)
    candidate_features = set(candidate.measurement_fact_counts)
    for feature in sorted(
        reference_features - candidate_features,
        key=lambda key: key.sort_key,
    ):
        feature_label = RuntimeMeasurementFeatureSemantics(feature, policy).label
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.MEASUREMENT_FEATURE,
                f"candidate is missing measurement feature {feature_label}",
            )
        )
    if not policy.allow_extra_candidate_measurements:
        for feature in sorted(
            candidate_features - reference_features,
            key=lambda key: key.sort_key,
        ):
            feature_label = RuntimeMeasurementFeatureSemantics(feature, policy).label
            differences.append(
                RuntimeEquivalenceDifference(
                    RuntimeEquivalenceDifferenceKind.MEASUREMENT_FEATURE,
                    f"candidate has extra measurement feature {feature_label}",
                )
            )
    for feature in sorted(
        reference_features & candidate_features,
        key=lambda key: key.sort_key,
    ):
        if _measurement_feature_values_equivalent(
            feature,
            reference,
            candidate,
            policy,
        ):
            continue
        feature_label = RuntimeMeasurementFeatureSemantics(feature, policy).label
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.MEASUREMENT_CONTENT,
                f"measurement feature {feature_label} values differ",
            )
        )
    return tuple(differences)


def _measurement_feature_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    reference_values = reference.measurement_fact_counts[feature]
    candidate_values = candidate.measurement_fact_counts[feature]
    if runtime_cell_signature_counters_equivalent(
        reference_values, candidate_values, policy
    ):
        return True
    if _duplicate_object_location_values_equivalent(
        feature,
        reference,
        candidate,
        policy,
    ):
        return True
    if _duplicate_object_values_equivalent(feature, reference, candidate, policy):
        return True
    if _relationship_object_projection_values_equivalent(feature, reference, candidate):
        return True
    if _tie_sensitive_location_values_equivalent(
        feature,
        reference,
        candidate,
        policy,
    ):
        return True
    if _threshold_entropy_values_equivalent(feature, reference, candidate, policy):
        return True
    if _indexed_descriptor_values_equivalent(feature, reference, candidate, policy):
        return True
    if SparseObjectBoundaryEquivalence(
        feature,
        reference,
        candidate,
        policy,
    ).values_equivalent():
        return True
    if RuntimeThresholdSensitivePairToleranceContract(policy).values_equivalent(
        feature,
        reference,
        candidate,
    ):
        return True
    if _feature_numeric_tolerance_values_equivalent(
        feature,
        reference,
        candidate,
        policy,
    ):
        return True
    if _relationship_object_number_mean_values_equivalent(feature):
        return True
    if _aggregate_mean_values_equivalent(feature, reference, candidate, policy):
        return True
    return RuntimeMeasurementFeatureSemantics(
        feature,
        policy,
    ).unstable_shape_descriptor_values_equivalent(reference, candidate)


def _duplicate_object_location_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    """Allow repeated runtime projections of the same object-location facts."""
    if feature.subject.scope is not MeasurementScope.OBJECT:
        return False
    role_feature = feature
    if feature.statistic == MeasurementStatistic.MEAN.value:
        role_feature = RuntimeMeasurementFeatureKey.from_subject_feature(
            feature.subject,
            feature.feature_name,
            MeasurementStatistic.VALUE.value,
            source_name=feature.source_name,
        )
    if not object_measurement_feature_matches_marker(
        role_feature,
        ObjectLocationFeatureMarker,
        policy,
    ):
        return False
    reference_values = reference.measurement_fact_counts[feature]
    candidate_values = candidate.measurement_fact_counts[feature]
    if not reference_values or set(reference_values) != set(candidate_values):
        normalized_candidate_values = (
            _candidate_counter_without_uniform_duplicate_projection(
                reference_values,
                candidate_values,
            )
        )
        if normalized_candidate_values is None:
            return False
        if runtime_cell_signature_counters_equivalent(
            reference_values,
            normalized_candidate_values,
            policy,
        ):
            return True
        if not policy.allow_sparse_object_boundary_jitter:
            return False
        return _sparse_numeric_counters_equivalent(
            reference_values,
            normalized_candidate_values,
            policy,
            abs_tolerance=policy.object_boundary_jitter_abs_tolerance,
            rel_tolerance=policy.object_boundary_jitter_rel_tolerance,
            max_unstable_values=policy.object_boundary_jitter_max_unstable_values,
            max_unstable_fraction=policy.object_boundary_jitter_max_unstable_fraction,
        )
    return _candidate_counter_is_duplicate_projection(
        reference_values,
        candidate_values,
    )


def _duplicate_object_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    """Allow exact repeated projections of object-level measurement facts."""
    if feature.subject.scope is not MeasurementScope.OBJECT:
        return False
    reference_values = reference.measurement_fact_counts[feature]
    candidate_values = candidate.measurement_fact_counts[feature]
    if _candidate_counter_is_duplicate_projection(reference_values, candidate_values):
        return True
    normalized_candidate_values = (
        _candidate_counter_without_uniform_duplicate_projection(
            reference_values,
            candidate_values,
        )
    )
    return (
        normalized_candidate_values is not None
        and runtime_cell_signature_counters_equivalent(
            reference_values,
            normalized_candidate_values,
            policy,
        )
    )


def _relationship_object_projection_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
) -> bool:
    """Allow CP per-image relationship rows to match global object-domain rows."""
    if feature.subject.scope is not MeasurementScope.OBJECT:
        return False
    if _candidate_counter_matches_reference_without_zero_padding(
        reference.measurement_fact_counts[feature],
        candidate.measurement_fact_counts[feature],
    ):
        return True
    object_subject_names = {
        key.subject.name
        for snapshot in (reference, candidate)
        for key in snapshot.measurement_fact_counts
        if key.subject.scope is MeasurementScope.OBJECT
    }
    if not (
        feature.feature_name.endswith("_count")
        or feature.feature_name in object_subject_names
        or any(
            feature.feature_name.endswith(f"_{name}") for name in object_subject_names
        )
    ):
        return False
    return _reference_counter_is_object_projection(
        reference.measurement_fact_counts[feature],
        candidate.measurement_fact_counts[feature],
    )


def _candidate_counter_matches_reference_without_zero_padding(
    reference_values: Counter[RuntimeCellSignature],
    candidate_values: Counter[RuntimeCellSignature],
) -> bool:
    if not reference_values or not candidate_values:
        return False
    without_zero_padding = Counter(
        {
            signature: count
            for signature, count in candidate_values.items()
            if not _runtime_cell_signature_is_zero(signature)
        }
    )
    return without_zero_padding == reference_values


def _runtime_cell_signature_is_zero(signature: RuntimeCellSignature) -> bool:
    numeric = _finite_signature_number(signature)
    return numeric == 0.0 if numeric is not None else False


def _reference_counter_is_object_projection(
    reference_values: Counter[RuntimeCellSignature],
    candidate_values: Counter[RuntimeCellSignature],
) -> bool:
    if not reference_values or set(reference_values) != set(candidate_values):
        return False
    return all(
        reference_values[signature] >= candidate_count > 0
        for signature, candidate_count in candidate_values.items()
    )


def _candidate_counter_without_duplicate_projection(
    reference_values: Counter[RuntimeCellSignature],
    candidate_values: Counter[RuntimeCellSignature],
) -> Counter[RuntimeCellSignature] | None:
    if set(reference_values) != set(candidate_values):
        return None
    return _candidate_counter_without_uniform_duplicate_projection(
        reference_values,
        candidate_values,
    )


def _candidate_counter_without_uniform_duplicate_projection(
    reference_values: Counter[RuntimeCellSignature],
    candidate_values: Counter[RuntimeCellSignature],
) -> Counter[RuntimeCellSignature] | None:
    reference_total = sum(reference_values.values())
    candidate_total = sum(candidate_values.values())
    if reference_total <= 0:
        return None
    factor, remainder = divmod(candidate_total, reference_total)
    if remainder or factor <= 1:
        return None
    normalized: Counter[RuntimeCellSignature] = Counter()
    for signature, candidate_count in candidate_values.items():
        count, count_remainder = divmod(candidate_count, factor)
        if count_remainder:
            return None
        normalized[signature] = count
    return normalized


def _candidate_counter_is_duplicate_projection(
    reference_values: Counter[RuntimeCellSignature],
    candidate_values: Counter[RuntimeCellSignature],
) -> bool:
    if not reference_values or set(reference_values) != set(candidate_values):
        return False
    duplicate_factors: set[int] = set()
    for signature, reference_count in reference_values.items():
        candidate_count = candidate_values[signature]
        if candidate_count < reference_count:
            return False
        factor, remainder = divmod(candidate_count, reference_count)
        if remainder or factor < 1:
            return False
        duplicate_factors.add(factor)
    return len(duplicate_factors) == 1


def _relationship_object_number_mean_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
) -> bool:
    """Treat child ObjectNumber means as identifier aggregates, not measurements."""
    return (
        feature.subject.scope is MeasurementScope.OBJECT
        and feature.statistic == MeasurementStatistic.VALUE.value
        and feature.feature_name.startswith("mean_")
        and feature.feature_name.endswith("_object_number")
    )


def _aggregate_mean_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    """Treat aggregate means as redundant when value-level facts already match."""
    if feature.subject.scope is not MeasurementScope.OBJECT:
        return False
    if feature.statistic != MeasurementStatistic.MEAN.value:
        return False
    value_feature = RuntimeMeasurementFeatureKey.from_subject_feature(
        feature.subject,
        feature.feature_name,
        MeasurementStatistic.VALUE.value,
        source_name=feature.source_name,
    )
    reference_values = reference.measurement_fact_counts.get(value_feature)
    candidate_values = candidate.measurement_fact_counts.get(value_feature)
    if reference_values is None or candidate_values is None:
        return False
    return (
        runtime_cell_signature_counters_equivalent(
            reference_values,
            candidate_values,
            policy,
        )
        or _candidate_counter_is_duplicate_projection(
            reference_values,
            candidate_values,
        )
        or _relationship_object_projection_values_equivalent(
            value_feature,
            reference,
            candidate,
        )
    )


def _feature_numeric_tolerance_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    feature_semantics = RuntimeMeasurementFeatureSemantics(feature, policy)
    for tolerance in policy.resolved_feature_numeric_tolerances():
        if not feature_semantics.matches_numeric_tolerance(tolerance):
            continue
        if (
            tolerance.require_object_count_stability
            and not MeasurementFeatureStabilityPolicy(
                feature,
                reference,
                candidate,
                policy,
            ).object_count_values_stable()
        ):
            continue
        feature_policy = RuntimeEquivalencePolicy(
            numeric_decimal_places=policy.numeric_decimal_places,
            numeric_abs_tolerance=tolerance.numeric_abs_tolerance,
            numeric_rel_tolerance=tolerance.numeric_rel_tolerance,
            measurement_feature_name_mode=policy.measurement_feature_name_mode,
            measurement_dialect=policy.measurement_dialect,
        )
        if runtime_cell_signature_counters_equivalent(
            reference.measurement_fact_counts[feature],
            candidate.measurement_fact_counts[feature],
            feature_policy,
        ):
            return True
    return False


def _tie_sensitive_location_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if not policy.allow_tie_sensitive_location_mismatches:
        return False
    value_feature = TieSensitiveLocationFeatureContract(
        feature,
        policy,
    ).value_feature_key()
    if value_feature is None:
        return False
    reference_values = reference.measurement_fact_counts.get(value_feature)
    candidate_values = candidate.measurement_fact_counts.get(value_feature)
    if reference_values is None or candidate_values is None:
        return False
    if runtime_cell_signature_counters_equivalent(
        reference_values,
        candidate_values,
        policy,
    ):
        return True
    return SparseObjectBoundaryEquivalence(
        value_feature,
        reference,
        candidate,
        policy,
    ).values_equivalent()


def _threshold_entropy_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if policy.threshold_entropy_abs_tolerance == 0:
        return False
    if not feature.feature_name.startswith("sum_of_entropies"):
        return False

    entropy_policy = RuntimeEquivalencePolicy(
        numeric_decimal_places=policy.numeric_decimal_places,
        numeric_abs_tolerance=policy.threshold_entropy_abs_tolerance,
        numeric_rel_tolerance=policy.numeric_rel_tolerance,
        measurement_feature_name_mode=policy.measurement_feature_name_mode,
    )
    return runtime_cell_signature_counters_equivalent(
        reference.measurement_fact_counts[feature],
        candidate.measurement_fact_counts[feature],
        entropy_policy,
    )


@dataclass(frozen=True, slots=True)
class RuntimeThresholdSensitivePairToleranceContract:
    """SSOT for pair measurements allowed to drift with threshold placement."""

    policy: RuntimeEquivalencePolicy

    def values_equivalent(
        self,
        feature: RuntimeMeasurementFeatureKey,
        reference: RuntimeMeasurementSnapshot,
        candidate: RuntimeMeasurementSnapshot,
    ) -> bool:
        if (
            self.policy.threshold_sensitive_pair_abs_tolerance == 0
            and self.policy.threshold_sensitive_pair_rel_tolerance == 0
        ):
            return False
        if not self.owns_key(feature):
            return False

        pair_policy = RuntimeEquivalencePolicy(
            numeric_decimal_places=self.policy.numeric_decimal_places,
            numeric_abs_tolerance=self.policy.threshold_sensitive_pair_abs_tolerance,
            numeric_rel_tolerance=self.policy.threshold_sensitive_pair_rel_tolerance,
            measurement_feature_name_mode=self.policy.measurement_feature_name_mode,
        )
        if not runtime_cell_signature_counters_equivalent(
            reference.measurement_fact_counts[feature],
            candidate.measurement_fact_counts[feature],
            pair_policy,
        ):
            return False

        return any(
            runtime_cell_signature_counters_equivalent(
                reference.measurement_fact_counts[companion],
                candidate.measurement_fact_counts[companion],
                pair_policy,
            )
            for companion in self.companion_features(feature, reference, candidate)
        )

    def owns_key(self, feature: RuntimeMeasurementFeatureKey) -> bool:
        """Return whether this feature is a threshold-sensitive pair family."""
        return feature.belongs_to_source_qualified_feature_family(
            self.policy.measurement_dialect,
            self.policy.measurement_dialect.resolved_threshold_sensitive_pair_feature_names(),
        )

    def companion_features(
        self,
        feature: RuntimeMeasurementFeatureKey,
        reference: RuntimeMeasurementSnapshot,
        candidate: RuntimeMeasurementSnapshot,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        """Return comparable pair-orientation companions for ``feature``."""
        source_tokens = feature.source_token_counter(
            self.policy.measurement_dialect,
            self.policy.measurement_dialect.resolved_threshold_sensitive_pair_feature_names(),
        )
        if source_tokens is None:
            return ()

        comparable_features = set(reference.measurement_fact_counts) & set(
            candidate.measurement_fact_counts
        )
        return tuple(
            sorted(
                (
                    other
                    for other in comparable_features
                    if self.is_companion_feature(
                        feature,
                        other,
                        source_tokens=source_tokens,
                    )
                ),
                key=lambda key: key.sort_key,
            )
        )

    def is_companion_feature(
        self,
        feature: RuntimeMeasurementFeatureKey,
        other: RuntimeMeasurementFeatureKey,
        *,
        source_tokens: Counter[str],
    ) -> bool:
        """Return whether ``other`` is the opposite orientation for ``feature``."""
        if other == feature:
            return False
        if other.statistic != feature.statistic:
            return False
        if other.subject.scope is not feature.subject.scope:
            return False
        if (
            feature.subject.scope is not MeasurementScope.IMAGE
            and other.subject != feature.subject
        ):
            return False
        if (feature.source_name is not None or other.source_name is not None) and (
            other.subject != feature.subject
        ):
            return False

        feature_family = feature.source_qualified_feature_family(
            self.policy.measurement_dialect,
            self.policy.measurement_dialect.resolved_threshold_sensitive_pair_feature_names(),
        )
        other_family = other.source_qualified_feature_family(
            self.policy.measurement_dialect,
            self.policy.measurement_dialect.resolved_threshold_sensitive_pair_feature_names(),
        )
        if feature_family is None or other_family is None:
            return False
        if other_family.feature_name != feature_family.feature_name:
            return False
        return (
            other.source_token_counter(
                self.policy.measurement_dialect,
                self.policy.measurement_dialect.resolved_threshold_sensitive_pair_feature_names(),
            )
            == source_tokens
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureSemantics:
    """Feature-local runtime measurement semantics used during equivalence."""

    feature: RuntimeMeasurementFeatureKey
    policy: RuntimeEquivalencePolicy

    @property
    def label(self) -> str:
        """Return a stable human-readable feature label."""
        subject = self.feature.subject
        if subject.name is None:
            subject_label = subject.scope.value
        else:
            subject_label = f"{subject.scope.value}:{subject.name}"
        feature_label = self.feature.feature_name
        if self.feature.source_name is not None:
            feature_label = f"{feature_label}@{self.feature.source_name}"
        if self.feature.statistic == MeasurementStatistic.VALUE.value:
            return f"{subject_label}/{feature_label}"
        return f"{subject_label}/{self.feature.statistic}({feature_label})"

    def matches_numeric_tolerance(
        self,
        tolerance: RuntimeMeasurementFeatureNumericTolerance,
    ) -> bool:
        """Return whether ``tolerance`` applies to this feature."""
        if (
            tolerance.subject_scope is not None
            and self.feature.subject.scope is not tolerance.subject_scope
        ):
            return False
        if (
            tolerance.statistic is not None
            and self.feature.statistic != tolerance.statistic
        ):
            return False
        if self._feature_name_matches_numeric_tolerance(
            self.feature.feature_name,
            tolerance,
        ):
            return True
        aggregate_child_feature_name = (
            RelationshipAggregateFeatureSemantics.aggregate_child_feature_name_from_key(
                self.feature,
                self.policy.measurement_dialect,
            )
        )
        return (
            aggregate_child_feature_name is not None
            and self._feature_name_matches_numeric_tolerance(
                aggregate_child_feature_name,
                tolerance,
            )
        )

    def _feature_name_matches_numeric_tolerance(
        self,
        feature_name: str,
        tolerance: RuntimeMeasurementFeatureNumericTolerance,
    ) -> bool:
        if feature_name in tolerance.feature_names:
            return True
        if (
            tolerance.feature_names
            and self.policy.measurement_dialect.source_qualified_feature_family(
                feature_name,
                self.feature.source_name,
                self.feature.subject.scope,
                tolerance.feature_names,
            )
            is not None
        ):
            return True
        return any(
            feature_name.startswith(prefix)
            for prefix in tolerance.feature_name_prefixes
        ) or any(
            feature_name.endswith(suffix) for suffix in tolerance.feature_name_suffixes
        )

    def unstable_shape_descriptor_values_equivalent(
        self,
        reference: RuntimeMeasurementSnapshot,
        candidate: RuntimeMeasurementSnapshot,
    ) -> bool:
        """Return whether unstable shape-descriptor values are equivalent."""
        if not self.policy.allow_unstable_shape_descriptors:
            return False
        if self.feature.subject.scope is not MeasurementScope.OBJECT:
            return False
        if self.feature.statistic != "value":
            return False
        shape_descriptor_context = ShapeDescriptorFeatureContext(
            self.feature,
            self.policy,
        ).semantic_context()
        semantics = ShapeDescriptorFeatureSemantics.for_context(
            shape_descriptor_context,
            required=False,
        )
        if semantics is None:
            return False
        if not MeasurementFeatureStabilityPolicy(
            self.feature,
            reference,
            candidate,
            self.policy,
        ).shape_descriptor_geometry_is_stable():
            return False

        reference_values = reference.measurement_fact_counts[self.feature]
        candidate_values = candidate.measurement_fact_counts[self.feature]
        return semantics.values_equivalent(
            shape_descriptor_context,
            reference_values,
            candidate_values,
        )


@dataclass(frozen=True, slots=True)
class SparseObjectBoundaryEquivalence:
    """Sparse object-boundary equivalence for object measurement features."""

    feature: RuntimeMeasurementFeatureKey
    reference: RuntimeMeasurementSnapshot
    candidate: RuntimeMeasurementSnapshot
    policy: RuntimeEquivalencePolicy

    def values_equivalent(self) -> bool:
        if not self.policy.allow_sparse_object_boundary_jitter:
            return False
        if self.feature.subject.scope is not MeasurementScope.OBJECT:
            return False
        if self.feature.statistic not in MeasurementStatistic._value2member_map_:
            return False
        statistic = MeasurementStatistic(self.feature.statistic)
        return SparseObjectBoundaryStatisticEquivalence.for_enum_member(
            statistic
        ).values_equivalent(self)

    def boundary_numeric_counters_equivalent(self) -> bool:
        return ObjectBoundarySparseNumericTolerance.equivalent(
            self.reference.measurement_fact_counts[self.feature],
            self.candidate.measurement_fact_counts[self.feature],
            self.policy,
        )

    def shape_descriptor_values_equivalent(self) -> bool:
        shape_descriptor_context = ShapeDescriptorFeatureContext(
            self.feature,
            self.policy,
        ).semantic_context()
        shape_descriptor_semantics = ShapeDescriptorFeatureSemantics.for_context(
            shape_descriptor_context,
            required=False,
        )
        if shape_descriptor_semantics is not None:
            return shape_descriptor_semantics.boundary_jitter_values_equivalent(
                shape_descriptor_context,
                self.reference.measurement_fact_counts[self.feature],
                self.candidate.measurement_fact_counts[self.feature],
            )
        if _numeric_counters_are_binary(
            self.reference.measurement_fact_counts[self.feature],
            self.candidate.measurement_fact_counts[self.feature],
        ):
            return BinarySparseNumericTolerance.equivalent(
                self.reference.measurement_fact_counts[self.feature],
                self.candidate.measurement_fact_counts[self.feature],
                self.policy,
            )
        return self.boundary_numeric_counters_equivalent()

    def identifier_counters_equivalent(
        self,
        reference: Counter[RuntimeCellSignature],
        candidate: Counter[RuntimeCellSignature],
    ) -> bool:
        if reference == candidate:
            return True
        if any(
            signature.kind is not RuntimeCellValueKind.NUMBER for signature in reference
        ):
            return False
        if any(
            signature.kind is not RuntimeCellValueKind.NUMBER for signature in candidate
        ):
            return False

        unstable_cap = max(
            self.policy.object_boundary_jitter_max_unstable_values,
            math.ceil(
                sum(reference.values())
                * self.policy.object_boundary_jitter_max_unstable_fraction
            ),
        )
        missing = sum((reference - candidate).values())
        extra = sum((candidate - reference).values())
        return max(missing, extra) <= unstable_cap


class SparseObjectBoundaryStatisticEquivalence(
    EnumKeyedStrategyMixin[MeasurementStatistic],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Statistic-specific sparse object-boundary equivalence."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    __enum_member_attr__ = "statistic"

    statistic: ClassVar[MeasurementStatistic]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def values_equivalent(
        self,
        context: SparseObjectBoundaryEquivalence,
    ) -> bool:
        """Return whether the statistic-specific sparse boundary values match."""


class SparseObjectBoundaryCountEquivalence(SparseObjectBoundaryStatisticEquivalence):
    """Sparse boundary equivalence for object count facts."""

    statistic = MeasurementStatistic.COUNT

    def values_equivalent(
        self,
        context: SparseObjectBoundaryEquivalence,
    ) -> bool:
        return object_measurement_feature_matches_marker(
            context.feature,
            ObjectCountFeatureMarker,
            context.policy,
        ) and _object_count_counters_sparse_equivalent(
            context.reference.measurement_fact_counts[context.feature],
            context.candidate.measurement_fact_counts[context.feature],
            context.policy,
        )


class SparseObjectBoundaryValueEquivalence(SparseObjectBoundaryStatisticEquivalence):
    """Sparse boundary equivalence for object value facts."""

    statistic = MeasurementStatistic.VALUE

    def values_equivalent(
        self,
        context: SparseObjectBoundaryEquivalence,
    ) -> bool:
        if (
            object_measurement_feature_requires_sparse_boundary_object_count_stability(
                context.feature,
                context.policy,
            )
            and not MeasurementFeatureStabilityPolicy(
                context.feature,
                context.reference,
                context.candidate,
                context.policy,
            ).object_count_values_stable()
        ):
            return False
        if object_measurement_feature_matches_marker(
            context.feature,
            ObjectIdentifierFeatureMarker,
            context.policy,
        ):
            return context.identifier_counters_equivalent(
                context.reference.measurement_fact_counts[context.feature],
                context.candidate.measurement_fact_counts[context.feature],
            )
        if any(
            object_measurement_feature_matches_marker(
                context.feature,
                marker_type,
                context.policy,
            )
            for marker_type in (
                ObjectLocationFeatureMarker,
                ObjectIntensityFeatureMarker,
                ObjectCalculatedFeatureMarker,
            )
        ):
            return context.boundary_numeric_counters_equivalent()
        if not object_measurement_feature_matches_marker(
            context.feature,
            ObjectShapeDescriptorFeatureMarker,
            context.policy,
        ):
            return False
        return context.shape_descriptor_values_equivalent()


class SparseObjectBoundaryMeanEquivalence(SparseObjectBoundaryStatisticEquivalence):
    """Sparse boundary equivalence for object mean facts."""

    statistic = MeasurementStatistic.MEAN

    def values_equivalent(
        self,
        context: SparseObjectBoundaryEquivalence,
    ) -> bool:
        value_feature = RuntimeMeasurementFeatureKey(
            subject=context.feature.subject,
            feature_name=context.feature.feature_name,
            statistic=MeasurementStatistic.VALUE.value,
            source_name=context.feature.source_name,
        )
        if value_feature not in context.reference.measurement_fact_counts:
            return False
        if value_feature not in context.candidate.measurement_fact_counts:
            return False
        if not SparseObjectBoundaryStatisticEquivalence.for_enum_member(
            MeasurementStatistic.VALUE
        ).values_equivalent(
            SparseObjectBoundaryEquivalence(
                value_feature,
                context.reference,
                context.candidate,
                context.policy,
            )
        ):
            return False

        mean_policy = RuntimeEquivalencePolicy(
            numeric_decimal_places=context.policy.numeric_decimal_places,
            numeric_abs_tolerance=context.policy.object_boundary_jitter_aggregate_abs_tolerance,
            numeric_rel_tolerance=context.policy.object_boundary_jitter_aggregate_rel_tolerance,
            measurement_feature_name_mode=context.policy.measurement_feature_name_mode,
        )
        return runtime_cell_signature_counters_equivalent(
            context.reference.measurement_fact_counts[context.feature],
            context.candidate.measurement_fact_counts[context.feature],
            mean_policy,
        )


@dataclass(frozen=True, slots=True)
class MeasurementFeatureStabilityPolicy:
    """Evaluate supporting measurement stability for feature equivalence."""

    feature: RuntimeMeasurementFeatureKey
    reference: RuntimeMeasurementSnapshot
    candidate: RuntimeMeasurementSnapshot
    policy: RuntimeEquivalencePolicy

    def object_count_values_stable(self) -> bool:
        count_feature = RuntimeMeasurementFeatureKey(
            subject=self.feature.subject,
            feature_name=ObjectCoreMeasurementFeature.OBJECT_COUNT.value,
            statistic=MeasurementStatistic.COUNT.value,
        )
        reference_counts = self.reference.measurement_fact_counts.get(count_feature)
        candidate_counts = self.candidate.measurement_fact_counts.get(count_feature)
        if reference_counts is None or candidate_counts is None:
            reference_values = self.reference.measurement_fact_counts.get(self.feature)
            candidate_values = self.candidate.measurement_fact_counts.get(self.feature)
            return (
                reference_values is not None
                and candidate_values is not None
                and sum(reference_values.values()) == sum(candidate_values.values())
            )
        return runtime_cell_signature_counters_equivalent(
            reference_counts,
            candidate_counts,
            self.policy,
        )

    def shape_descriptor_geometry_is_stable(self) -> bool:
        stable_features = (
            self.object_measurement_marker_stable_features(
                ObjectLocationFeatureMarker,
            )
            | self.object_measurement_marker_exactly_stable_features(
                ObjectShapeDescriptorFeatureMarker,
            )
        )
        return len(stable_features) >= 3

    def object_measurement_marker_stable_features(
        self,
        marker_type: type[runtime_semantics.RuntimeMeasurementFeatureSemanticMarker],
    ) -> frozenset[RuntimeMeasurementFeatureKey]:
        stable_features: set[RuntimeMeasurementFeatureKey] = set()
        candidate_keys = (
            self.reference.measurement_fact_counts.keys()
            | self.candidate.measurement_fact_counts.keys()
        )
        for candidate_key in candidate_keys:
            if not self._candidate_key_matches_marker(candidate_key, marker_type):
                continue
            reference_values = self.reference.measurement_fact_counts.get(candidate_key)
            candidate_values = self.candidate.measurement_fact_counts.get(candidate_key)
            if reference_values is None or candidate_values is None:
                continue
            if not self._feature_values_stable(
                candidate_key,
                reference_values,
                candidate_values,
            ):
                continue
            stable_features.add(candidate_key)
        return frozenset(stable_features)

    def object_measurement_marker_exactly_stable_features(
        self,
        marker_type: type[runtime_semantics.RuntimeMeasurementFeatureSemanticMarker],
    ) -> frozenset[RuntimeMeasurementFeatureKey]:
        stable_features: set[RuntimeMeasurementFeatureKey] = set()
        candidate_keys = (
            self.reference.measurement_fact_counts.keys()
            | self.candidate.measurement_fact_counts.keys()
        )
        for candidate_key in candidate_keys:
            if candidate_key == self.feature:
                continue
            if not self._candidate_key_matches_marker(candidate_key, marker_type):
                continue
            reference_values = self.reference.measurement_fact_counts.get(candidate_key)
            candidate_values = self.candidate.measurement_fact_counts.get(candidate_key)
            if reference_values is None or candidate_values is None:
                continue
            if not runtime_cell_signature_counters_equivalent(
                reference_values,
                candidate_values,
                self.policy,
            ):
                continue
            stable_features.add(candidate_key)
        return frozenset(stable_features)

    def _feature_values_stable(
        self,
        feature: RuntimeMeasurementFeatureKey | None,
        reference_values: Counter[RuntimeCellSignature],
        candidate_values: Counter[RuntimeCellSignature],
    ) -> bool:
        if runtime_cell_signature_counters_equivalent(
            reference_values,
            candidate_values,
            self.policy,
        ):
            return True
        if feature is None:
            return False
        if not self.policy.allow_sparse_object_boundary_jitter:
            return False
        if feature.subject.scope is not MeasurementScope.OBJECT:
            return False
        if feature.statistic not in MeasurementStatistic._value2member_map_:
            return False
        return SparseObjectBoundaryStatisticEquivalence.for_enum_member(
            MeasurementStatistic(feature.statistic)
        ).values_equivalent(
            SparseObjectBoundaryEquivalence(
                feature,
                self.reference,
                self.candidate,
                self.policy,
            )
        )

    def _candidate_key_matches_marker(
        self,
        candidate_key: RuntimeMeasurementFeatureKey,
        marker_type: type[runtime_semantics.RuntimeMeasurementFeatureSemanticMarker],
    ) -> bool:
        if candidate_key.subject != self.feature.subject:
            return False
        if candidate_key.source_name is not None:
            return False
        if candidate_key.statistic != MeasurementStatistic.VALUE.value:
            return False
        return object_measurement_feature_matches_marker(
            candidate_key,
            marker_type,
            self.policy,
        )


def _object_count_counters_sparse_equivalent(
    reference: Counter[RuntimeCellSignature],
    candidate: Counter[RuntimeCellSignature],
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if reference == candidate:
        return True
    if any(
        signature.kind is not RuntimeCellValueKind.NUMBER for signature in reference
    ):
        return False
    if any(
        signature.kind is not RuntimeCellValueKind.NUMBER for signature in candidate
    ):
        return False

    unstable_cap = max(
        policy.object_boundary_jitter_max_unstable_values,
        math.ceil(
            sum(reference.values())
            * policy.object_boundary_jitter_max_unstable_fraction
        ),
    )
    missing = sum((reference - candidate).values())
    extra = sum((candidate - reference).values())
    return max(missing, extra) <= unstable_cap


def _numeric_counters_are_binary(
    reference: Counter[RuntimeCellSignature],
    candidate: Counter[RuntimeCellSignature],
) -> bool:
    numbers: set[float] = set()
    for counter in (reference, candidate):
        for signature in counter:
            numeric = _finite_signature_number(signature)
            if numeric is None:
                return False
            numbers.add(numeric)
    return numbers.issubset({0.0, 1.0})


def _indexed_descriptor_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    profile = RuntimeMeasurementFeatureSemanticProfile.for_feature_key(feature, policy)
    if not isinstance(profile, RuntimeMeasurementDescriptorSemantics):
        return False
    if not profile.descriptor_snapshots_comparable(
        feature,
        reference,
        candidate,
        policy,
    ):
        return False

    return profile.values_equivalent(
        feature,
        reference.measurement_fact_counts[feature],
        candidate.measurement_fact_counts[feature],
        policy,
    )


@dataclass(frozen=True, slots=True)
class RuntimeObjectLocationRowMergeContract(metaclass=AutoRegisterMeta):
    """SSOT for object-location value facts merged by runtime row identity."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)

    registry_key: ClassVar[str | None] = None
    policy: RuntimeEquivalencePolicy

    @classmethod
    def registered_projection(
        cls,
        registry_key: RuntimeObjectLocationRowMergeProjectionKey,
        policy: RuntimeEquivalencePolicy,
    ) -> "RuntimeObjectLocationRowMergeContract":
        """Return the registered row-merge projection for ``registry_key``."""
        try:
            projection_type = cls.__registry__[registry_key.value]
        except KeyError as exc:
            registered = tuple(cls.__registry__)
            raise ValueError(
                "Unknown runtime object-location row-merge projection "
                f"{registry_key.value!r}; registered projections: {registered!r}."
            ) from exc
        return projection_type(policy)

    def owns_key(self, key: RuntimeMeasurementFeatureKey) -> bool:
        return (
            key.subject.scope is MeasurementScope.OBJECT
            and key.statistic == MeasurementStatistic.VALUE.value
            and key.source_name is None
            and object_measurement_feature_matches_marker(
                key,
                ObjectLocationFeatureMarker,
                self.policy,
            )
        )

    def subjects(
        self,
        row_merge_cache: _RuntimeMeasurementRowMergeCache,
    ) -> frozenset[RuntimeMeasurementSubjectKey]:
        """Return subjects owned by this row-merge projection."""
        return frozenset(
            key.subject
            for key, row_identity in row_merge_cache
            if self.owns_row_identity(key, row_identity)
        )

    def owns_row_identity(
        self,
        key: RuntimeMeasurementFeatureKey,
        row_identity: RuntimeMeasurementRowIdentity,
    ) -> bool:
        """Return whether this projection owns a row identity."""
        del row_identity
        return self.owns_key(key)


def _runtime_row_measurement_fact_priority(
    row: RuntimeMeasurementRowMapping,
    key: RuntimeMeasurementFeatureKey,
    policy: RuntimeEquivalencePolicy,
    row_priority_cache: _RuntimeMeasurementRowPriorityCache,
) -> int:
    """Return dialect category priority for the row field that produced ``key``."""
    candidates: list[str] = []
    long_form_feature = row.first_value(
        MeasurementRowAxisField.feature_name_field_names_ordered()
    )
    cache_key = RuntimeMeasurementRowPriorityCacheKey(
        row_fields=row.header,
        long_form_feature=(
            str(long_form_feature) if long_form_feature is not None else None
        ),
        feature_key=key,
    )
    cached = row_priority_cache.get(cache_key)
    if cached is not None:
        return cached
    if long_form_feature is not None:
        candidates.append(str(long_form_feature))
    else:
        candidates.extend(row.header)

    priorities = tuple(
        priority
        for feature_name in candidates
        if (
            priority := _measurement_feature_source_priority(
                feature_name,
                key,
                policy,
            )
        )
        is not None
    )
    priority = min(priorities, default=sys.maxsize)
    row_priority_cache[cache_key] = priority
    return priority


def _runtime_row_feature_category_priority(
    row: RuntimeMeasurementRowMapping,
    policy: RuntimeEquivalencePolicy,
    row_feature_priority_cache: _RuntimeMeasurementRowFeaturePriorityCache,
) -> int | None:
    """Return best dialect category priority represented by this row shape."""
    long_form_feature = row.first_value(
        MeasurementRowAxisField.feature_name_field_names_ordered()
    )
    cache_key = RuntimeMeasurementRowFeaturePriorityCacheKey(
        row_fields=row.header,
        long_form_feature=(
            str(long_form_feature) if long_form_feature is not None else None
        ),
    )
    cached = row_feature_priority_cache.get(cache_key, _CACHE_MISS)
    if cached is not _CACHE_MISS:
        return cached
    candidates = (
        (str(long_form_feature),) if long_form_feature is not None else row.header
    )
    priorities = tuple(
        priority
        for feature_name in candidates
        if (
            priority := RuntimeMeasurementFeatureCategoryPriority(
                feature_name,
                policy.measurement_dialect,
            ).priority()
        )
        is not None
    )
    priority = min(priorities, default=None)
    row_feature_priority_cache[cache_key] = priority
    return priority


def _measurement_feature_source_priority(
    feature_name: str,
    key: RuntimeMeasurementFeatureKey,
    policy: RuntimeEquivalencePolicy,
) -> int | None:
    priority = RuntimeMeasurementFeatureCategoryPriority(
        feature_name,
        policy.measurement_dialect,
    ).priority()
    if priority is None:
        return None
    canonical_feature_name, canonical_source_name = (
        RuntimeMeasurementFeatureNameProjection.from_feature_name(
            feature_name,
            policy,
            None,
            (),
        ).project()
    )
    if (
        canonical_feature_name != key.feature_name
        or canonical_source_name != key.source_name
    ):
        return None
    return priority if canonical_feature_name else None


def _runtime_row_merge_candidate_preferred(
    candidate: _RuntimeMeasurementRowMergeValue,
    current: _RuntimeMeasurementRowMergeValue,
) -> bool:
    candidate_priority, _candidate_row_priority, candidate_value = candidate
    current_priority, _current_row_priority, current_value = current
    candidate_missing = RuntimeCellMissingStrategy.for_kind(
        candidate_value.kind
    ).is_missing(candidate_value)
    current_missing = RuntimeCellMissingStrategy.for_kind(
        current_value.kind
    ).is_missing(current_value)
    if current_missing and not candidate_missing:
        return True
    if candidate_missing and not current_missing:
        return False
    return candidate_priority < current_priority


class RuntimeRowMergeLocationSubjectProjection(RuntimeObjectLocationRowMergeContract):
    """Project subjects with measured location rows from the row-merge cache."""

    registry_key: ClassVar[str | None] = (
        RuntimeObjectLocationRowMergeProjectionKey.LOCATION.value
    )


class RuntimeRowMergeAggregateLocationSubjectProjection(
    RuntimeRowMergeLocationSubjectProjection
):
    """Project measured-location subjects that can derive image-scoped aggregates."""

    registry_key: ClassVar[str | None] = (
        RuntimeObjectLocationRowMergeProjectionKey.AGGREGATE_LOCATION.value
    )

    def owns_row_identity(
        self,
        key: RuntimeMeasurementFeatureKey,
        row_identity: RuntimeMeasurementRowIdentity,
    ) -> bool:
        return (
            super().owns_row_identity(
                key,
                row_identity,
            )
            and RuntimeObjectMeasurementRowIdentity(row_identity).has_image_identity
        )


def _primary_row_object_count_measurement_facts(
    primary_row_identities: _RuntimeMeasurementPrimaryRowSet,
    row_merge_cache: _RuntimeMeasurementRowMergeCache,
    policy: RuntimeEquivalencePolicy,
    *,
    existing_subjects: frozenset[RuntimeMeasurementSubjectKey],
    required_keys: RuntimeRequiredMeasurementKeys,
) -> RuntimeMeasurementFacts:
    counts_by_image: dict[
        tuple[RuntimeMeasurementSubjectKey, RuntimeMeasurementRowIdentity],
        set[object],
    ] = {}
    row_merge_subject_identities = (
        object_measurement_subject_row_identities_matching_marker(
            row_merge_cache,
            ObjectLocationFeatureMarker,
            policy,
        )
    )
    source_row_identities = (
        tuple(
            (subject, RuntimeObjectMeasurementRowIdentity(row_identity))
            for subject, row_identity in row_merge_subject_identities
        )
        if row_merge_subject_identities
        else tuple(primary_row_identities)
    )
    for subject, identity in source_row_identities:
        image_identity = identity.image_identity
        object_label = identity.object_label_signature
        if object_label is None:
            continue
        counts_by_image.setdefault((subject, image_identity), set()).add(object_label)

    facts: RuntimeMeasurementFactList = []
    for (subject, _image_identity), object_labels in counts_by_image.items():
        if subject in existing_subjects:
            continue
        key = RuntimeMeasurementFeatureKey(
            subject,
            ObjectCoreMeasurementFeature.OBJECT_COUNT.value,
            MeasurementStatistic.COUNT.value,
        )
        if required_keys is not None and key not in required_keys:
            continue
        facts.append((key, runtime_cell_signature(str(len(object_labels)), policy)))
    return tuple(facts)


def _object_location_primary_row_priority(policy: RuntimeEquivalencePolicy) -> int:
    location_priority = runtime_measurement_category_priority(
        ("location",),
        policy.measurement_dialect,
    )
    return sys.maxsize if location_priority is None else location_priority - 1


@dataclass(frozen=True, slots=True)
class RuntimeExportRowProjectionContext(RuntimeRowProjectionContext):
    """Projection context for exported table rows with contextual column subjects."""

    table: RuntimeTableSnapshot

    def wide_feature_indexes(
        self,
        header: tuple[str, ...],
        row_schema: equivalence_measurement_rows.RuntimeMeasurementRowSchema,
    ) -> tuple[int, ...]:
        cache_key: equivalence_measurement_rows.RuntimeMeasurementWideFeatureIndexCacheKey = (
            header,
            self.subject.scope,
            self.subject.name,
            self.source_name,
            id(self.required_key_index),
        )
        cached = self.wide_feature_index_cache.get(cache_key)
        if cached is not None:
            return cached
        indexes = tuple(
            index
            for index in row_schema.feature_indexes
            if not self.contextual_object_identity_field(index)
        )
        self.wide_feature_index_cache[cache_key] = indexes
        return indexes

    def supports_static_wide_projection(self) -> bool:
        """Export snapshots may derive subjects from row-contextual columns."""
        return False

    def subject_for_field_index(
        self,
        index: int,
    ) -> RuntimeMeasurementSubjectKey:
        return RuntimeExportColumnSubject(
            self.table,
            self.row,
            index,
            self.subject,
        ).subject()

    def contextual_object_identity_field(self, index: int) -> bool:
        """Return whether one contextual export column carries object identity."""
        if not self.table.column_context or index >= len(self.table.column_context):
            return False
        subject = self.subject_for_field_index(index)
        return (
            RuntimeExportContextualObjectIdentityField(
                table=self.table,
                index=index,
                context=self.table.column_context[index],
                subject=subject,
                policy=self.policy,
                known_source_names=self.known_source_names,
            ).candidate()
            is not None
        )

    def padding_indexes(
        self,
        row_schema: equivalence_measurement_rows.RuntimeMeasurementRowSchema,
        row_values: RuntimeIndexedRowValues,
    ) -> frozenset[int]:
        return ContextualMeasurementPaddingProjection(
            self.table.column_context,
            self.table.header,
            row_schema.feature_indexes,
            self.policy.measurement_dialect,
            self.known_source_names,
        ).padding_indexes(row_values)


@dataclass(frozen=True, slots=True)
class RuntimeTableSnapshotFactExtractor:
    """Project exported measurement table snapshots into semantic facts."""

    table: RuntimeTableSnapshot
    policy: RuntimeEquivalencePolicy
    known_source_names: tuple[str, ...] = ()

    def measurement_facts(self) -> RuntimeMeasurementFacts:
        """Project exported table rows into semantic facts."""
        counts = self.measurement_fact_counts()
        return tuple(
            (key, value)
            for key, counter in counts.items()
            for value, count in counter.items()
            for _ in range(count)
        )

    def measurement_fact_counts(self) -> RuntimeMeasurementFactCounterMap:
        """Project this exported table into semantic fact counters."""
        state = RuntimeMeasurementProjectionState(
            policy=self.policy,
            known_source_names=self.known_source_names,
        )
        self.record_measurement_table(state)
        return state.project_recorded_row_fact_counts()

    def record_measurement_table(
        self,
        state: RuntimeMeasurementProjectionState,
    ) -> None:
        """Record this exported table into a shared measurement projection state."""
        feature_indexes = tuple(
            index
            for index, field_name in enumerate(self.table.header)
            if not runtime_measurement_identity_field_matches(
                field_name,
                state.policy.measurement_dialect,
            )
        )
        if not feature_indexes:
            record_measurement_facts(
                state.measurement_fact_counts,
                self.identity_facts(),
                required_keys=state.required_measurement_keys,
            )
            return
        image_number_offset = RuntimeImageNumberOffset.from_table_rows(
            self.table.header,
            self.table.rows,
        )
        schema_cache: RuntimeMeasurementRowSchemaCache = {}
        key_cache: RuntimeMeasurementFeatureKeyCache = {}
        long_form_key_cache: RuntimeMeasurementLongFormKeyCache = {}
        wide_feature_index_cache: (
            equivalence_measurement_rows.RuntimeMeasurementWideFeatureIndexCache
        ) = {}
        wide_feature_plan_cache: (
            equivalence_measurement_rows.RuntimeMeasurementWideFeaturePlanCache
        ) = {}
        qualifier_render_cache: RuntimeMeasurementQualifierRenderCache = {}
        padding_group_cache: RuntimeMeasurementPaddingGroupCache = {}
        collapsed_numeric_qualifier_cache: RuntimeCollapsedNumericQualifierCache = {}
        required_key_index = RuntimeMeasurementRequiredKeyIndex.from_required_keys(
            state.required_measurement_keys
        )
        derive_directional_pair_facts = (
            RuntimeDirectionalPairMeasurementDerivationContract(
                state.policy,
                state.known_source_names,
            ).required_keys_need_derivation(state.required_measurement_keys)
        )
        explicit_measurement_keys: set[RuntimeMeasurementFeatureKey] = set()
        row_priority_cache: _RuntimeMeasurementRowPriorityCache = {}
        row_feature_priority_cache: _RuntimeMeasurementRowFeaturePriorityCache = {}
        recorder = RuntimeExportTableRowProjectionRecorder(
            state=state,
            image_number_offset=image_number_offset,
            row_priority_cache=row_priority_cache,
            row_feature_priority_cache=row_feature_priority_cache,
            explicit_measurement_keys=explicit_measurement_keys,
            required_key_index=required_key_index,
            table=self.table,
        )
        for row in self.table.rows:
            row_mapping = dict(zip(self.table.header, row, strict=True))
            runtime_row = RuntimeMeasurementRowMapping(row_mapping)
            row_subject = RuntimeExportRowSubject(
                self.table.path,
                runtime_row,
            ).subject()
            recorder.record_primary_row_identity(runtime_row, row_subject)
            source_name = measurement_row_source_image_name(row_mapping)
            row_context = RuntimeExportRowProjectionContext(
                row=runtime_row,
                subject=row_subject,
                policy=state.policy,
                source_name=source_name,
                known_source_names=state.known_source_names,
                required_keys=state.required_measurement_keys,
                table_padding_group=measurement_table_padding_group(
                    self.table.path.stem
                ),
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
                table=self.table,
            )
            row_facts = row_context.facts()
            if row_facts:
                recorder.record_projected_row_facts(runtime_row, row_facts)

    def identity_facts(self) -> RuntimeMeasurementFacts:
        """Project object identity-only exports into semantic object-number facts."""
        facts: RuntimeMeasurementFactList = []
        for row in self.table.rows:
            row_mapping = dict(zip(self.table.header, row, strict=True))
            runtime_row = RuntimeMeasurementRowMapping(row_mapping)
            row_subject = RuntimeExportRowSubject(
                self.table.path,
                runtime_row,
            ).subject()
            if row_subject.scope is not MeasurementScope.OBJECT:
                continue
            normalized_row_mapping = {
                normalize_runtime_identifier(field_name): value
                for field_name, value in row_mapping.items()
            }
            object_number = measurement_object_label(normalized_row_mapping)
            if object_number is None:
                continue
            facts.append(
                (
                    RuntimeMeasurementFeatureKey(
                        row_subject,
                        ObjectCoreMeasurementFeature.OBJECT_NUMBER.value,
                    ),
                    runtime_cell_signature(str(object_number), self.policy),
                )
            )
        return tuple(facts)


def _measurement_source_names_from_artifact_execution(
    observation: RuntimeArtifactExecutionObservation,
) -> tuple[str, ...]:
    """Return source-image names carried by typed runtime artifacts."""
    source_names: set[str] = set()
    for records in observation.records_by_axis.values():
        for record in records:
            schema_source_name = record.value.schema.source_image_name
            if schema_source_name is not None:
                source_names.update(_source_name_aliases(str(schema_source_name)))
            if record.key.artifact_type is not MeasurementsArtifactType:
                continue
            table = MeasurementTable.from_runtime_value(record.value)
            if table.source_image_name is not None:
                source_names.update(_source_name_aliases(table.source_image_name))
            for row in iter_measurement_rows((table,)):
                row_source_name = measurement_row_source_image_name(
                    measurement_row_mapping(row)
                )
                if row_source_name is not None:
                    source_names.update(_source_name_aliases(row_source_name))
    return tuple(sorted(source_names, key=normalize_runtime_identifier))


def _source_name_aliases(source_name: str) -> tuple[str, ...]:
    names = tuple(part for part in str(source_name).split("__") if part)
    if len(names) <= 1:
        return names
    return (source_name, *names)


@dataclass(frozen=True, slots=True)
class RuntimeExportRowSubject:
    """Resolve table-export row subject from table path and row identity."""

    path: Path
    row: RuntimeMeasurementRowMapping

    def subject(self) -> RuntimeMeasurementSubjectKey:
        object_name = self.row.object_name()
        if object_name is not None:
            return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, object_name)

        table_name = self.path.stem
        normalized_table_name = normalize_runtime_identifier(table_name)
        if self.row.has_object_identity() and normalized_table_name != "image":
            return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, table_name)
        if normalized_table_name == "experiment":
            return RuntimeMeasurementSubjectKey(MeasurementScope.EXPERIMENT, None)
        return RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")


@dataclass(frozen=True, slots=True)
class RuntimeExportColumnSubject:
    """Resolve table-export column subject from contextual column metadata."""

    table: RuntimeTableSnapshot
    row: RuntimeMeasurementRowMapping
    index: int
    fallback_subject: RuntimeMeasurementSubjectKey

    def subject(self) -> RuntimeMeasurementSubjectKey:
        if not self.table.column_context or self.index >= len(
            self.table.column_context
        ):
            return self.fallback_subject

        field_name = self.table.header[self.index]
        if runtime_measurement_identity_field_matches(
            field_name,
            DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
        ):
            return self.fallback_subject

        context, normalized_context = self.context_pair()
        row_object_name = self.row.object_name()
        normalized_row_object_name = (
            normalize_runtime_identifier(row_object_name)
            if row_object_name is not None
            else None
        )
        return RuntimeColumnContextSubject(
            context,
            normalized_context,
            row_object_name,
            normalized_row_object_name,
            fallback_subject=self.fallback_subject,
        ).subject()

    def context_pair(self) -> tuple[str | None, str | None]:
        if not self.table.column_context or self.index >= len(
            self.table.column_context
        ):
            return None, None
        context = self.table.column_context[self.index]
        if context is None:
            return None, None
        normalized_context = normalize_runtime_identifier(context)
        if not normalized_context:
            return None, None
        return context, normalized_context


@dataclass(frozen=True, slots=True)
class RuntimeExportContextualObjectIdentityField:
    """Resolve a contextual object-id column for one exported object subject."""

    table: RuntimeTableSnapshot
    index: int
    context: str | None
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    known_source_names: tuple[str, ...]

    def candidate(self) -> "RuntimeExportContextualObjectIdentityCandidate | None":
        if self.subject.scope is not MeasurementScope.OBJECT:
            return None
        if self.subject.name is None:
            return None
        if self.context is None:
            return None
        if normalize_runtime_identifier(self.context) != self.subject.name:
            return None
        field_name = self.table.header[self.index]
        contextual_feature_name = RuntimeExportContextualFeatureName(
            field_name,
            self.subject,
        )
        projected_feature_name, _source_name = (
            RuntimeMeasurementFeatureNameProjection.from_feature_name(
                contextual_feature_name.feature_name(),
                self.policy,
                None,
                self.known_source_names,
            ).project()
        )
        if projected_feature_name != ObjectCoreMeasurementFeature.OBJECT_NUMBER.value:
            return None
        return RuntimeExportContextualObjectIdentityCandidate(
            field_name=field_name,
            specificity=RuntimeExportContextualObjectIdentitySpecificity.from_field(
                field_name,
                contextual_feature_name,
                self.policy,
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeExportContextualObjectIdentityCandidate:
    """Contextual object-id field candidate with deterministic specificity."""

    field_name: str
    specificity: "RuntimeExportContextualObjectIdentitySpecificity"


class RuntimeExportContextualObjectIdentitySpecificity(Enum):
    """Specificity ordering for contextual object-id fields."""

    CONTEXT_SUFFIX = 0
    CONTEXT_MEASUREMENT = 1
    ROW_IDENTITY = 2

    @classmethod
    def from_field(
        cls,
        field_name: str,
        contextual_feature_name: "RuntimeExportContextualFeatureName",
        policy: RuntimeEquivalencePolicy,
    ) -> "RuntimeExportContextualObjectIdentitySpecificity":
        if contextual_feature_name.has_context_suffix():
            return cls.CONTEXT_SUFFIX
        if runtime_measurement_identity_field_matches(
            field_name,
            policy.measurement_dialect,
        ):
            return cls.ROW_IDENTITY
        return cls.CONTEXT_MEASUREMENT


@dataclass(frozen=True, slots=True)
class RuntimeExportContextualFeatureName:
    """Feature-name view after removing a suffix introduced by CSV context."""

    field_name: str
    subject: RuntimeMeasurementSubjectKey

    def feature_name(self) -> str:
        normalized_field_name = normalize_runtime_identifier(self.field_name)
        subject_suffix = self.subject_suffix()
        if subject_suffix is not None and normalized_field_name.endswith(
            subject_suffix
        ):
            return normalized_field_name[: -len(subject_suffix)]
        return normalized_field_name

    def has_context_suffix(self) -> bool:
        subject_suffix = self.subject_suffix()
        return subject_suffix is not None and normalize_runtime_identifier(
            self.field_name
        ).endswith(subject_suffix)

    def subject_suffix(self) -> str | None:
        subject_name = self.subject.name
        if subject_name is None:
            return None
        return f"_{subject_name}"


@dataclass(frozen=True, slots=True)
class RuntimeColumnContextSubject:
    """Resolve a subject implied by contextual wide-table column metadata."""

    context: str | None
    normalized_context: str | None
    row_object_name: str | None
    normalized_row_object_name: str | None
    fallback_subject: RuntimeMeasurementSubjectKey

    def subject(self) -> RuntimeMeasurementSubjectKey:
        if self.context is None or self.normalized_context is None:
            return self.fallback_subject
        if self.normalized_context in CSV_HEADER_CONTEXT_STOPWORDS:
            if self.normalized_context == "image":
                return RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")
            return self.fallback_subject

        if (
            self.row_object_name is not None
            and self.normalized_row_object_name == self.normalized_context
        ):
            return RuntimeMeasurementSubjectKey(
                MeasurementScope.OBJECT,
                self.row_object_name,
            )
        return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, self.context)


_MEASUREMENT_QUALIFIER_FIELDS = (
    "scale",
    "direction",
    "gray_levels",
)
_MEASUREMENT_QUALIFIER_FIELD_SET = frozenset(_MEASUREMENT_QUALIFIER_FIELDS)
_NON_MEASUREMENT_FIELD_PREFIXES = (
    "channel_",
    "execution_time_",
    "file_name_",
    "frame_",
    "group_",
    "height_",
    "image_quality_scaling_",
    "image_set_",
    "md_5_digest_",
    "md5_digest_",
    "module_error_",
    "path_name_",
    "scaling_",
    "series_",
    "url_",
    "width_",
)
