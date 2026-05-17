"""Thin CellProfiler-style view over OpenHCS runtime artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Hashable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from inspect import isabstract
import logging
import os
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Generic, TypeVar
from weakref import WeakKeyDictionary

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.constants.constants import Backend, FileFormat
from openhcs.core.artifacts import ArtifactInputPlan, ArtifactKind, ArtifactOutputPlan
from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.memory import detect_memory_type, stack_slices
from openhcs.core.aligned_image_payload import payload_slices_for_alignment
from openhcs.core.source_image_semantics import apply_source_image_loading_semantics
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingRuntimeContext,
    SourceBindingOrigin,
    SourceRuntimePathLookup,
)
from openhcs.core.source_schema_workspace import source_schema_auxiliary_payload
from openhcs.core.source_matching import (
    is_image_path,
    merge_source_metadata,
    metadata_from_rules,
    source_filters_match,
    SourceImageSetIdentity,
    source_component_metadata_values,
    semantic_source_metadata_value,
    source_metadata_component,
    source_metadata_values_equal,
    source_metadata_value,
    source_path_identity_key,
    source_paths_equal,
)
from openhcs.core.runtime_stores import (
    RuntimeArtifactLocation,
    RuntimeArtifactQuery,
    RuntimeValueStore,
    StoredRuntimeValue,
    replace_runtime_artifact_payload,
)
from openhcs.core.runtime_invocation import RuntimeSliceAlignedValues
from openhcs.core.runtime_slice_projection import (
    MeasurementTableRepeatedScalarGroupKey,
    RuntimeSliceProjection,
)
from openhcs.core.measurement_lookup_dialect import (
    RuntimeMeasurementLookupDialectLike,
    resolve_runtime_measurement_lookup_dialect,
)
from openhcs.core.runtime_artifact_queries import (
    MeasurementTableObjectFeatureSemantics,
    RuntimeArtifactQueryContext,
    measurement_feature_candidates,
    measurement_value_indexes_for_object_feature_batch,
    measurement_values_for_feature,
    measurement_values_for_label_plane,
    measurement_values_for_label_slices,
    measurement_tables_for_image_number,
    measurement_tables_for_slice,
    optional_measurement_value_index,
    normalize_measurement_token,
    runtime_measurement_tables,
    runtime_measurement_tables_for_scope,
    runtime_measurement_tables_for_object,
    runtime_relationship,
    runtime_spatial_grid,
)
from openhcs.core.runtime_semantics import (
    MeasurementScope,
    ObjectLabelMeasurementValues,
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    RelationshipSemantics,
    RuntimeObjectFeatureMeasurementQuery,
    RuntimeObjectMeasurementQuery,
    RuntimeObjectLabelMeasurementQuery,
    RuntimePlaneProjection,
    RuntimePlaneAxisProjector,
    dense_object_label_id_domain,
)
from openhcs.core.runtime_values import (
    FieldSpec,
    RuntimeArrayPayload,
    MeasurementTable,
    NamedImage,
    ObjectLabelPayload,
    ObjectLabelPure2DSliceAggregator,
    ObjectLabelSet,
    ObjectLabelRepresentation,
    ObjectRelationship,
    SparseIJVLabelRows,
    SpatialGrid,
    compose_image_payload_metadata,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_metadata_from_source,
    image_payload_with_context,
    normalize_artifact_value,
)
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)
T = TypeVar("T")
U = TypeVar("U")
_SOURCE_CANDIDATE_CACHE_LIMIT = 64
_SOURCE_CANDIDATE_PROCESS_CACHE: OrderedDict[
    tuple[Hashable, ...],
    tuple["ParsedSourceCandidate", ...],
] = OrderedDict()
_OBJECT_FEATURE_VALUE_PROCESS_CACHE: WeakKeyDictionary[
    RuntimeValueStore,
    dict[RuntimeObjectFeatureMeasurementQuery, Any],
] = WeakKeyDictionary()

MeasurementTableSelection = tuple[MeasurementTable, ...] | None
SpatialGridGroupValues = tuple[
    SpatialGrid | RuntimeSliceAlignedValues[SpatialGrid],
    ...,
]


@dataclass(frozen=True, slots=True)
class ObjectMeasurementTableCacheKey:
    """Semantic cache key for object-subject measurement table queries."""

    group_key: str | None
    match_group: bool
    object_name: str


@dataclass(frozen=True, slots=True)
class ObjectMeasurementTableIndexCacheKey:
    """Semantic cache key for object-subject measurement table indexes."""

    group_key: str | None
    match_group: bool


@dataclass(frozen=True, slots=True)
class ObjectMeasurementTableIndex:
    """Nominal object-subject measurement table index."""

    tables: tuple[MeasurementTable, ...]
    tables_by_object: Mapping[str, tuple[MeasurementTable, ...]]
    feature_names_by_table: Mapping[int, frozenset[str]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    repeated_scalar_group_sizes: Mapping[MeasurementTableRepeatedScalarGroupKey, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    complete: bool = False

    @classmethod
    def from_tables(
        cls,
        tables: tuple[MeasurementTable, ...],
    ) -> "ObjectMeasurementTableIndex":
        """Return a complete index over the provided measurement tables."""
        table_lists: dict[str, list[tuple[MeasurementTable, MeasurementTableObjectFeatureSemantics]]] = {}
        group_sizes: dict[MeasurementTableRepeatedScalarGroupKey, int] = {}
        for table in tables:
            table_semantics = MeasurementTableObjectFeatureSemantics.from_table(table)
            group_key = RuntimeSliceProjection.measurement_table_repeated_scalar_group_key(
                table
            )
            group_sizes[group_key] = group_sizes.get(group_key, 0) + 1
            for table_object_name in table_semantics.object_names:
                table_lists.setdefault(table_object_name, []).append(
                    (table, table_semantics)
                )
        indexed_tables: dict[str, tuple[MeasurementTable, ...]] = {}
        feature_names_by_table: dict[int, frozenset[str]] = {}
        for object_name, object_table_entries in table_lists.items():
            object_tables = tuple(table for table, _semantics in object_table_entries)
            projected_tables = (
                RuntimeSliceProjection.measurement_tables_with_repeated_scalar_slice_offsets(
                    object_tables
                )
            )
            indexed_tables[object_name] = projected_tables
            for projected_table, (_table, table_semantics) in zip(
                projected_tables,
                object_table_entries,
                strict=True,
            ):
                feature_names_by_table[id(projected_table)] = (
                    table_semantics.feature_names
                )
        return cls(
            tables=RuntimeSliceProjection.measurement_tables_with_repeated_scalar_slice_offsets(
                tables
            ),
            tables_by_object=MappingProxyType(indexed_tables),
            feature_names_by_table=MappingProxyType(feature_names_by_table),
            repeated_scalar_group_sizes=MappingProxyType(group_sizes),
            complete=True,
        )

    @staticmethod
    def table_object_names(table: MeasurementTable) -> tuple[str, ...]:
        """Return all object subjects declared by a measurement table."""
        return MeasurementTableObjectFeatureSemantics.from_table(table).object_names

    def for_object(self, object_name: str) -> MeasurementTableSelection:
        """Return indexed tables for one object, or ``None`` when unknown."""
        if not self.complete:
            return None
        return self.tables_by_object.get(object_name, ())

    def for_object_feature(
        self,
        object_name: str,
        feature_name: str,
        *,
        dialect: RuntimeMeasurementLookupDialectLike = (
            CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT
        ),
    ) -> MeasurementTableSelection:
        """Return indexed object tables that may carry one feature."""
        query_object_name = resolve_runtime_measurement_lookup_dialect(
            dialect
        ).feature_lookup(feature_name).query_object_name(object_name)
        tables = (
            self.tables
            if query_object_name is None
            else self.for_object(query_object_name)
        )
        if tables is None:
            return None
        candidates = measurement_feature_candidates(
            feature_name,
            dialect=dialect,
        )
        return tuple(
            table
            for table in tables
            if self._table_may_carry_feature(table, candidates)
        )

    def _table_may_carry_feature(
        self,
        table: MeasurementTable,
        candidates: frozenset[str],
    ) -> bool:
        feature_names = self.feature_names_by_table.get(id(table))
        if feature_names is None:
            feature_names = MeasurementTableObjectFeatureSemantics.from_table(
                table
            ).feature_names
        if not feature_names:
            return True
        return any(
            normalize_measurement_token(feature_name) in candidates
            for feature_name in feature_names
        )

@dataclass(frozen=True, slots=True)
class MeasurementTableCacheMutation:
    """Semantic cache mutation caused by one measurement-table write."""

    adapter: "CellProfilerRuntimeAdapter"
    table: MeasurementTable
    table_semantics: MeasurementTableObjectFeatureSemantics

    @property
    def object_names(self) -> tuple[str, ...]:
        return self.table_semantics.object_names

    @property
    def feature_names(self) -> frozenset[str]:
        return self.table_semantics.feature_names


class MeasurementTableCacheMutationPolicy(ABC, metaclass=AutoRegisterMeta):
    """Registered policy for measurement-table cache mutation side effects."""

    __registry_key__ = "policy_name"
    __skip_if_no_key__ = True

    policy_name: ClassVar[str | None] = None

    @classmethod
    def registered_policies(cls) -> tuple["MeasurementTableCacheMutationPolicy", ...]:
        """Return registered cache mutation policies in registry order."""
        return tuple(
            policy_type()
            for policy_type in cls.__registry__.values()
            if not isabstract(policy_type)
        )

    @abstractmethod
    def apply(self, mutation: MeasurementTableCacheMutation) -> None:
        """Apply this policy to a measurement-table cache mutation."""


class MeasurementQueryCacheMutationPolicy(MeasurementTableCacheMutationPolicy):
    """Shared mutation contract for measurement-query caches."""

    def apply(self, mutation: MeasurementTableCacheMutation) -> None:
        if not mutation.object_names:
            return
        self.apply_query_cache_mutation(mutation)

    @abstractmethod
    def apply_query_cache_mutation(self, mutation: MeasurementTableCacheMutation) -> None:
        """Apply a mutation known to affect at least one object domain."""


class MeasurementQueryCacheInvalidationPolicy(MeasurementQueryCacheMutationPolicy):
    """Shared object/feature invalidation for measurement-query caches."""

    entry_type: ClassVar[type[object]]
    feature_scoped: ClassVar[bool] = False
    adapter_cache_accessor: ClassVar[
        Callable[["CellProfilerRuntimeAdapter"], MutableMapping[object, object]]
        | None
    ] = None

    def cache(self, mutation: MeasurementTableCacheMutation) -> MutableMapping[object, object]:
        """Return the cache owned by this invalidation policy."""
        cache_accessor = type(self).adapter_cache_accessor
        if cache_accessor is None:
            raise RuntimeError(
                f"{type(self).__name__} must declare adapter_cache_accessor."
            )
        return cache_accessor(mutation.adapter)

    def cache_entries(self, mutation: MeasurementTableCacheMutation) -> tuple[object, ...]:
        """Return cache entries considered for this mutation."""
        return tuple(self.cache(mutation))

    def delete_entry(
        self,
        mutation: MeasurementTableCacheMutation,
        entry: object,
    ) -> None:
        """Delete one cache entry."""
        del self.cache(mutation)[entry]

    def entry_object_name(self, entry: object) -> str | None:
        """Return the object domain addressed by one cache entry."""
        if not isinstance(entry, type(self).entry_type):
            return None
        return entry.object_name

    def entry_feature_name(self, entry: object) -> str | None:
        """Return the feature addressed by one cache entry, when feature-scoped."""
        if not type(self).feature_scoped:
            return None
        if not isinstance(entry, type(self).entry_type):
            return None
        return entry.feature_name

    def apply_query_cache_mutation(self, mutation: MeasurementTableCacheMutation) -> None:
        for entry in self.cache_entries(mutation):
            entry_object_name = self.entry_object_name(entry)
            if entry_object_name not in mutation.object_names:
                continue
            entry_feature_name = self.entry_feature_name(entry)
            if (
                entry_feature_name is not None
                and mutation.feature_names
                and entry_feature_name not in mutation.feature_names
            ):
                continue
            self.delete_entry(mutation, entry)


@dataclass(frozen=True, slots=True)
class MeasurementQueryCacheInvalidationDeclaration:
    """Typed declaration for one measurement-query cache invalidation policy."""

    class_name: str
    policy_name: str
    entry_type: type[object]
    feature_scoped: bool = False
    doc: str = ""

    def materialize(self) -> type[MeasurementQueryCacheInvalidationPolicy]:
        namespace = {
            "__module__": __name__,
            "__doc__": self.doc,
            "policy_name": self.policy_name,
            "entry_type": self.entry_type,
            "feature_scoped": self.feature_scoped,
        }
        return type(
            self.class_name,
            (MeasurementQueryCacheInvalidationPolicy,),
            namespace,
        )


class MeasurementQueryCacheInvalidationFamily:
    """Authoritative materializer for measurement-query cache policy classes."""

    declarations: ClassVar[tuple[MeasurementQueryCacheInvalidationDeclaration, ...]] = (
        MeasurementQueryCacheInvalidationDeclaration(
            class_name="ObjectFeatureValueCacheInvalidationPolicy",
            policy_name="object_feature_value",
            entry_type=RuntimeObjectMeasurementQuery,
            feature_scoped=True,
            doc="Invalidate object-feature vector cache entries touched by a table write.",
        ),
        MeasurementQueryCacheInvalidationDeclaration(
            class_name="ObjectLabelMeasurementValuesCacheInvalidationPolicy",
            policy_name="object_label_measurement_values",
            entry_type=RuntimeObjectLabelMeasurementQuery,
            feature_scoped=True,
            doc="Invalidate label-aligned measurement vector cache entries touched by a write.",
        ),
        MeasurementQueryCacheInvalidationDeclaration(
            class_name="ObjectMeasurementTableCacheInvalidationPolicy",
            policy_name="object_measurement_table",
            entry_type=ObjectMeasurementTableCacheKey,
            doc="Invalidate object-subject measurement table query cache entries.",
        ),
    )

    @classmethod
    def materialize_exports(
        cls,
        namespace: MutableMapping[str, object],
    ) -> tuple[type[MeasurementQueryCacheInvalidationPolicy], ...]:
        policy_types = tuple(
            declaration.materialize()
            for declaration in cls.declarations
        )
        namespace.update(
            {policy_type.__name__: policy_type for policy_type in policy_types}
        )
        return policy_types

    @staticmethod
    def bind_adapter_caches(
        adapter_type: type["CellProfilerRuntimeAdapter"],
    ) -> None:
        ObjectFeatureValueCacheInvalidationPolicy.adapter_cache_accessor = (
            adapter_type.object_feature_value_cache
        )
        ObjectLabelMeasurementValuesCacheInvalidationPolicy.adapter_cache_accessor = (
            adapter_type.object_label_measurement_values_cache
        )
        ObjectMeasurementTableCacheInvalidationPolicy.adapter_cache_accessor = (
            adapter_type.object_measurement_table_cache
        )


(
    ObjectFeatureValueCacheInvalidationPolicy,
    ObjectLabelMeasurementValuesCacheInvalidationPolicy,
    ObjectMeasurementTableCacheInvalidationPolicy,
) = MeasurementQueryCacheInvalidationFamily.materialize_exports(globals())


class ObjectMeasurementTableIndexInvalidationPolicy(MeasurementQueryCacheMutationPolicy):
    """Invalidate object-subject indexes touched by measurement-table writes."""

    policy_name = "object_measurement_table_index"

    def apply_query_cache_mutation(self, mutation: MeasurementTableCacheMutation) -> None:
        object_table_index_cache = mutation.adapter.object_measurement_table_index_cache()
        for cache_key in (
            ObjectMeasurementTableIndexCacheKey(mutation.adapter.group_key, True),
            ObjectMeasurementTableIndexCacheKey(None, False),
        ):
            object_table_index_cache.pop(cache_key, None)


_OBJECT_MEASUREMENT_TABLE_PROCESS_CACHE: WeakKeyDictionary[
    RuntimeValueStore,
    dict[ObjectMeasurementTableCacheKey, tuple[MeasurementTable, ...]],
] = WeakKeyDictionary()
_OBJECT_MEASUREMENT_TABLE_INDEX_PROCESS_CACHE: WeakKeyDictionary[
    RuntimeValueStore,
    dict[ObjectMeasurementTableIndexCacheKey, ObjectMeasurementTableIndex],
] = WeakKeyDictionary()
_OBJECT_LABEL_MEASUREMENT_VALUES_PROCESS_CACHE: WeakKeyDictionary[
    RuntimeValueStore,
    dict[RuntimeObjectLabelMeasurementQuery, tuple[Any, ...]],
] = WeakKeyDictionary()
_PIPELINE_START_PAYLOAD_CACHE_LIMIT = 64
_PIPELINE_START_PAYLOAD_PROCESS_CACHE: OrderedDict[
    tuple[Hashable, ...],
    tuple[Any, ...],
] = OrderedDict()
_MAX_DENSE_LABEL_STACK_BYTES = 1 << 30


def _runtime_profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_adapter_profile(label: str, seconds: float, **fields: Any) -> None:
    if not _runtime_profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


class AdapterProfileLog:
    """Authoritative projection for runtime adapter profiling event fields."""

    @staticmethod
    def object_feature(
        label: str,
        seconds: float,
        *,
        object_name: str,
        feature_name: str,
        count: int | None = None,
        cached: object | None = None,
        ndim: int | None = None,
    ) -> None:
        fields: dict[str, object] = {
            "object": object_name,
            "feature": feature_name,
        }
        if count is not None:
            fields["count"] = count
        if cached is not None:
            fields["cached"] = cached
        if ndim is not None:
            fields["ndim"] = ndim
        _log_adapter_profile(label, seconds, **fields)

    @staticmethod
    def label_batch(
        label: str,
        seconds: float,
        *,
        feature_name: str,
        count: int | None = None,
        cached: object | None = None,
        requested: int | None = None,
        uncached: int | None = None,
        fallback: str | None = None,
    ) -> None:
        fields: dict[str, object] = {"feature": feature_name}
        if count is not None:
            fields["count"] = count
        if cached is not None:
            fields["cached"] = cached
        if requested is not None:
            fields["requested"] = requested
        if uncached is not None:
            fields["uncached"] = uncached
        if fallback is not None:
            fields["fallback"] = fallback
        _log_adapter_profile(label, seconds, **fields)

    @staticmethod
    def artifact(
        label: str,
        seconds: float,
        *,
        artifact_name: str,
        kind: ArtifactKind | str | None = None,
        payload_type: str | None = None,
        group_key: str | None = None,
    ) -> None:
        fields: dict[str, object] = {"artifact": artifact_name}
        if kind is not None:
            fields["kind"] = kind.value if isinstance(kind, ArtifactKind) else kind
        if payload_type is not None:
            fields["payload_type"] = payload_type
        if group_key is not None:
            fields["group_key"] = group_key
        _log_adapter_profile(label, seconds, **fields)

    @staticmethod
    def measurement_artifact(
        label: str,
        seconds: float,
        *,
        artifact_name: str,
        object_name: str | None = None,
        fields_declared: bool | None = None,
    ) -> None:
        fields: dict[str, object] = {"artifact": artifact_name}
        if object_name is not None:
            fields["object"] = object_name
        if fields_declared is not None:
            fields["fields"] = fields_declared
        _log_adapter_profile(label, seconds, **fields)

    @staticmethod
    def measurement_cache(
        label: str,
        seconds: float,
        *,
        object_count: int,
        feature_count: int,
    ) -> None:
        _log_adapter_profile(
            label,
            seconds,
            objects=object_count,
            features=feature_count,
        )

    @staticmethod
    def measurement_cache_policy(
        seconds: float,
        *,
        object_count: int,
        feature_count: int,
        policy_name: str,
    ) -> None:
        _log_adapter_profile(
            "adapter_measurement_cache_policy",
            seconds,
            policy=policy_name,
            objects=object_count,
            features=feature_count,
        )

    @staticmethod
    def measurement_slice_mismatch(
        seconds: float,
        *,
        object_name: str,
        measurement_slices: int,
        label_slices: int,
    ) -> None:
        _log_adapter_profile(
            "adapter_multiplane_measurement_slice_mismatch",
            seconds,
            object=object_name,
            measurement_slices=measurement_slices,
            label_slices=label_slices,
        )

    @staticmethod
    def source_candidates(
        label: str,
        seconds: float,
        *,
        count: int,
        alias: str | None = None,
        source: SourceBindingOrigin | None = None,
    ) -> None:
        fields: dict[str, object] = {"count": count}
        if alias is not None:
            fields["alias"] = alias
        if source is not None:
            fields["source"] = source.value
        _log_adapter_profile(label, seconds, **fields)


@dataclass(frozen=True, slots=True)
class NativeRecordProfileContext:
    """Profiling context for one native artifact materialization."""

    artifact_name: str
    kind: ArtifactKind

    def event(
        self,
        label: str,
        seconds: float,
        *,
        payload_type: str | None = None,
        group_key: str | None = None,
    ) -> None:
        AdapterProfileLog.artifact(
            label,
            seconds,
            artifact_name=self.artifact_name,
            kind=self.kind,
            payload_type=payload_type,
            group_key=group_key,
        )

    def group_event(
        self,
        label: str,
        seconds: float,
        group_key: str | None,
    ) -> None:
        self.event(label, seconds, group_key=group_key)

    def normalized_value(
        self,
        seconds: float,
        *,
        payload_type: str,
        group_key: str | None,
    ) -> None:
        self.event(
            "adapter_normalize_artifact_value",
            seconds,
            payload_type=payload_type,
            group_key=group_key,
        )


@dataclass(frozen=True, slots=True)
class ObjectFeatureProfileContext:
    """Profiling context for one object/feature measurement query."""

    object_name: str
    feature_name: str

    def domain(self, seconds: float, *, count: int, ndim: int | None = None) -> None:
        AdapterProfileLog.object_feature(
            "adapter_object_label_query_domain",
            seconds,
            object_name=self.object_name,
            feature_name=self.feature_name,
            count=count,
            ndim=ndim,
        )

    def get_objects(self, seconds: float) -> None:
        AdapterProfileLog.object_feature(
            "adapter_object_feature_get_objects",
            seconds,
            object_name=self.object_name,
            feature_name=self.feature_name,
        )

    def counted_event(self, label: str, seconds: float, *, count: int) -> None:
        AdapterProfileLog.object_feature(
            label,
            seconds,
            object_name=self.object_name,
            feature_name=self.feature_name,
            count=count,
        )

    def object_domain(self, seconds: float, *, count: int) -> None:
        self.counted_event(
            "adapter_object_feature_domain",
            seconds,
            count=count,
        )

    def query_tables(self, seconds: float, *, count: int) -> None:
        self.counted_event(
            "adapter_object_label_query_tables",
            seconds,
            count=count,
        )

    def feature_tables(self, seconds: float, *, count: int) -> None:
        self.counted_event(
            "adapter_object_feature_tables",
            seconds,
            count=count,
        )

    def query_extract(self, seconds: float, *, count: int) -> None:
        self.counted_event(
            "adapter_object_label_query_extract",
            seconds,
            count=count,
        )

    def feature_extract(self, seconds: float, *, count: int) -> None:
        self.counted_event(
            "adapter_object_feature_extract",
            seconds,
            count=count,
        )

    def query_values(self, seconds: float, *, cached: object) -> None:
        AdapterProfileLog.object_feature(
            "adapter_object_label_query_values",
            seconds,
            object_name=self.object_name,
            feature_name=self.feature_name,
            cached=cached,
        )

    def feature_values(self, seconds: float, *, cached: object) -> None:
        AdapterProfileLog.object_feature(
            "adapter_object_feature_values",
            seconds,
            object_name=self.object_name,
            feature_name=self.feature_name,
            cached=cached,
        )


@dataclass(frozen=True, slots=True)
class LabelBatchProfileContext:
    """Profiling context for one label-slice measurement batch."""

    feature_name: str

    def cache(self, seconds: float, *, requested: int, uncached: int) -> None:
        AdapterProfileLog.label_batch(
            "adapter_object_label_batch_cache",
            seconds,
            feature_name=self.feature_name,
            requested=requested,
            uncached=uncached,
        )

    def values(
        self,
        seconds: float,
        *,
        cached: object,
        count: int,
        fallback: str | None = None,
    ) -> None:
        AdapterProfileLog.label_batch(
            "adapter_object_label_batch_values",
            seconds,
            feature_name=self.feature_name,
            cached=cached,
            count=count,
            fallback=fallback,
        )

    def tables(self, seconds: float, *, count: int) -> None:
        AdapterProfileLog.label_batch(
            "adapter_object_label_batch_tables",
            seconds,
            feature_name=self.feature_name,
            count=count,
        )

    def indexes(self, seconds: float, *, count: int) -> None:
        AdapterProfileLog.label_batch(
            "adapter_object_label_batch_indexes",
            seconds,
            feature_name=self.feature_name,
            count=count,
        )

    def align(self, seconds: float, *, count: int) -> None:
        AdapterProfileLog.label_batch(
            "adapter_object_label_batch_align",
            seconds,
            feature_name=self.feature_name,
            count=count,
        )


@dataclass(frozen=True, slots=True)
class ObjectLabelSourceImageDomain:
    """Source-domain metadata inherited by object labels produced from an image."""

    spatial_origin_yx: tuple[int, int] | None = None
    source_spatial_shape_yx: tuple[int, int] | None = None

    @classmethod
    def for_adapter_source_image(
        cls,
        adapter: "CellProfilerRuntimeAdapter",
        source_image_name: str | None,
    ) -> "ObjectLabelSourceImageDomain":
        if source_image_name is None:
            return cls()
        try:
            source_image = adapter.get_image(source_image_name)
        except RuntimeError:
            return cls()
        metadata = image_payload_metadata(source_image.data)
        return cls(
            spatial_origin_yx=metadata.spatial_origin_yx,
            source_spatial_shape_yx=metadata.source_spatial_shape_yx,
        )

    def origin_or(self, existing: tuple[int, int] | None) -> tuple[int, int] | None:
        return existing if existing is not None else self.spatial_origin_yx

    def source_shape_or(
        self,
        existing: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        return existing if existing is not None else self.source_spatial_shape_yx


@dataclass(frozen=True, slots=True)
class CellProfilerRuntimeAdapter(RuntimePlaneAxisProjector):
    """CellProfiler-like API backed by typed OpenHCS runtime state.

    The adapter deliberately has no object/image/measurement dictionaries of its
    own. Writes require compiled output plans and a filemanager so the
    RuntimeValueStore record and VFS payload stay aligned with the normal
    FunctionStep runtime boundary.
    """

    runtime_value_store: RuntimeValueStore
    axis_id: str
    artifact_inputs: Mapping[str, ArtifactInputPlan] = field(default_factory=dict)
    artifact_outputs: Mapping[str, ArtifactOutputPlan] = field(default_factory=dict)
    source_binding_plan: CompiledSourceBindingPlan = field(
        default_factory=CompiledSourceBindingPlan.empty
    )
    source_binding_context: SourceBindingRuntimeContext = field(
        default_factory=SourceBindingRuntimeContext.empty
    )
    group_key: str | None = None
    processing_context: Any | None = None
    filemanager: Any | None = None
    backend: str = Backend.MEMORY.value
    plane_projection: RuntimePlaneProjection = field(
        default_factory=RuntimePlaneProjection.stack
    )
    _source_candidate_cache: dict[
        tuple[str, ...],
        tuple["ParsedSourceCandidate", ...],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _pipeline_start_payload_cache: dict[
        tuple[Hashable, ...],
        tuple[Any, ...],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _image_cache: dict[tuple[str | None, str], NamedImage] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _object_cache: dict[tuple[str | None, str], ObjectLabelSet] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _measurement_cache: dict[tuple[Hashable, ...], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _label_domain_cache: dict[int, tuple[int, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _source_order_cache: dict[tuple[Hashable, ...], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _artifact_availability_cache: dict[tuple[Hashable, ...], bool] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_value_store, RuntimeValueStore):
            raise TypeError(
                "CellProfilerRuntimeAdapter.runtime_value_store must be "
                f"RuntimeValueStore, got {type(self.runtime_value_store).__name__}."
            )

        if not self.axis_id:
            raise ValueError("CellProfilerRuntimeAdapter.axis_id cannot be empty.")
        if not self.backend:
            raise ValueError("CellProfilerRuntimeAdapter.backend cannot be empty.")
        if not isinstance(self.source_binding_plan, CompiledSourceBindingPlan):
            raise TypeError(
                "CellProfilerRuntimeAdapter.source_binding_plan must be "
                "CompiledSourceBindingPlan, got "
                f"{type(self.source_binding_plan).__name__}."
            )
        if not isinstance(self.source_binding_context, SourceBindingRuntimeContext):
            raise TypeError(
                "CellProfilerRuntimeAdapter.source_binding_context must be "
                "SourceBindingRuntimeContext, got "
                f"{type(self.source_binding_context).__name__}."
            )
        if not isinstance(self.plane_projection, RuntimePlaneProjection):
            raise TypeError(
                "CellProfilerRuntimeAdapter.plane_projection must be "
                "RuntimePlaneProjection, got "
                f"{type(self.plane_projection).__name__}."
            )

        outputs = dict(self.artifact_outputs)
        for name, plan in outputs.items():
            if not isinstance(plan, ArtifactOutputPlan):
                raise TypeError(
                    f"artifact_outputs['{name}'] must be ArtifactOutputPlan, "
                    f"got {type(plan).__name__}."
                )
            if name != plan.name:
                raise ValueError(
                    f"artifact_outputs key '{name}' does not match plan name "
                    f"'{plan.name}'."
                )
        object.__setattr__(self, "artifact_outputs", MappingProxyType(outputs))
        if self.group_key is not None:
            object.__setattr__(self, "group_key", str(self.group_key))

    def cellprofiler_source_order_path(self, path: str) -> str:
        """Return the source path identity used for CellProfiler image ordering."""
        source_paths = self.source_binding_context.step_input_source_paths
        mapped = source_paths.get(path) or path
        return source_path_identity_key(mapped)

    def object_feature_value_cache(self) -> dict[RuntimeObjectFeatureMeasurementQuery, Any]:
        """Return the process cache for stable object-feature vectors in this store."""
        return _OBJECT_FEATURE_VALUE_PROCESS_CACHE.setdefault(
            self.runtime_value_store,
            {},
        )

    def object_label_measurement_values_cache(
        self,
    ) -> dict[RuntimeObjectLabelMeasurementQuery, tuple[Any, ...]]:
        """Return the process cache for label-aligned object-feature vectors."""
        return _OBJECT_LABEL_MEASUREMENT_VALUES_PROCESS_CACHE.setdefault(
            self.runtime_value_store,
            {},
        )

    def object_measurement_table_cache(
        self,
    ) -> dict[ObjectMeasurementTableCacheKey, tuple[MeasurementTable, ...]]:
        """Return the process cache for object-subject measurement table queries."""
        return _OBJECT_MEASUREMENT_TABLE_PROCESS_CACHE.setdefault(
            self.runtime_value_store,
            {},
        )

    def object_measurement_table_index_cache(
        self,
    ) -> dict[ObjectMeasurementTableIndexCacheKey, ObjectMeasurementTableIndex]:
        """Return the process cache for object-subject measurement table indexes."""
        return _OBJECT_MEASUREMENT_TABLE_INDEX_PROCESS_CACHE.setdefault(
            self.runtime_value_store,
            {},
        )

    def cellprofiler_ordered_pipeline_image_paths(self) -> tuple[str, ...]:
        """Return loadable pipeline input paths in CellProfiler image order."""
        cache_key = ("ordered_pipeline_image_paths",)
        cached = self._source_order_cache.get(cache_key)
        if cached is not None:
            return cached
        ordered = tuple(
            dict.fromkeys(
                self.cellprofiler_source_order_path(path)
                for path in sorted(self.source_binding_context.pipeline_input_files)
                if is_image_path(path)
            )
        )
        self._source_order_cache[cache_key] = ordered
        return ordered

    def cellprofiler_axis_image_number_start(self) -> int:
        """Return CP's 1-based image number for this runtime axis."""
        cache_key = ("axis_image_number_start", str(self.axis_id))
        cached = self._source_order_cache.get(cache_key)
        if cached is not None:
            return int(cached)
        if self.processing_context is None:
            self._source_order_cache[cache_key] = 1
            return 1

        axis_id = str(self.axis_id)
        metadata_by_path = self.source_binding_context.source_metadata_by_path
        pipeline_paths = tuple(
            path
            for path in self.source_binding_context.pipeline_input_files
            if is_image_path(path)
        )
        for index, path in enumerate(sorted(pipeline_paths), start=1):
            metadata = (
                metadata_by_path.get(path)
                or metadata_by_path.get(str(Path(path).resolve()))
                or metadata_by_path.get(self.cellprofiler_source_order_path(path))
            )
            if metadata and any(str(value) == axis_id for value in metadata.values()):
                self._source_order_cache[cache_key] = index
                return index

        parser = _require_processing_context(self).microscope_handler.parser
        for index, path in enumerate(sorted(pipeline_paths), start=1):
            parsed = parser.parse_filename(Path(path).name) or {}
            if any(str(value) == axis_id for value in parsed.values()):
                self._source_order_cache[cache_key] = index
                return index

        self._source_order_cache[cache_key] = 1
        return 1

    def cellprofiler_image_number_for_payload(
        self,
        payload: Any,
    ) -> int | None:
        """Return the CellProfiler ImageNumber for a payload carrying source paths."""
        metadata = image_payload_metadata(payload)
        source_paths = tuple(
            str(path)
            for path in metadata.channel_source_paths
            if path is not None and str(path)
        )
        if not source_paths and metadata.source_path:
            source_paths = (metadata.source_path,)
        return CellProfilerImageNumberResolver.for_adapter(self).image_number_for_paths(
            source_paths
        )

    def source_image_payload_for_name(
        self,
        image_name: str,
        current_image: Any | None,
    ) -> Any | None:
        """Resolve an image name through source bindings or runtime image records."""
        if current_image is not None and self.has_source_binding(
            image_name,
            ArtifactKind.IMAGE,
        ):
            return self.resolve_source_image(image_name, current_image)
        if not self.has_runtime_artifact(name=image_name, kind=ArtifactKind.IMAGE):
            return None
        try:
            return self.get_image(image_name).data
        except RuntimeError:
            return None

    def invalidate_runtime_query_caches_for_kind(self, kind: ArtifactKind) -> None:
        """Invalidate adapter caches whose semantic domain can change for ``kind``."""
        RuntimeArtifactCacheInvalidationPolicy.for_kind(kind).invalidate(self)

    def require_artifact_available(self, *, name: str, kind: ArtifactKind) -> None:
        """Fail loudly unless a runtime artifact is declared, bound, or resolvable."""
        cache_key = (
            "artifact_available",
            self.runtime_value_store.revision,
            self.group_key,
            name,
            kind,
        )
        if self._artifact_availability_cache.get(cache_key):
            return
        if self.has_source_binding(name, kind):
            self._artifact_availability_cache[cache_key] = True
            return
        self._resolve_runtime_record(name=name, kind=kind)
        self._artifact_availability_cache[cache_key] = True

    def has_runtime_artifact(self, *, name: str, kind: ArtifactKind) -> bool:
        """Return whether this execution scope contains a runtime artifact."""
        return bool(self._query_context().find(name=name, kind=kind))

    def resolve_source_image(
        self,
        alias: str,
        current_image: Any,
    ) -> Any:
        request = self._source_resolution_request(
            alias,
            ArtifactKind.IMAGE,
            current_image,
        )
        return SourceBindingResolver.for_origin(request.binding.origin).resolve_image(
            request
        )

    def runtime_slice_plane_index(self) -> int | None:
        """Return the current axis-local runtime-slice plane index."""
        return self.plane_projection.runtime_slice_plane_index()

    def source_binding_axis_plane_index(
        self,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        """Return the current axis-local source-binding plane index."""
        indexes = tuple(
            index
            for alias in source_aliases
            for index in (self.source_binding_plane_index(alias),)
            if index is not None
        )
        unique_indexes = tuple(dict.fromkeys(indexes))
        if not unique_indexes:
            return None
        if len(unique_indexes) != 1:
            raise RuntimeError(
                "Source-binding plane resolution produced conflicting indexes: "
                f"{unique_indexes!r}."
            )
        return unique_indexes[0]

    def source_binding_plane_index(self, alias: str) -> int | None:
        """Return the current axis-local plane index for a source alias."""
        return (
            OptionalResolution(alias)
            .bind(self.source_binding_plane_candidate_context)
            .bind(self.source_binding_plane_matched_context)
            .bind(self.single_source_binding_plane_index)
            .value
        )

    def source_binding_plane_candidate_context(
        self,
        alias: str,
    ) -> OptionalResolution["SourceBindingPlaneCandidateContext"]:
        binding = self.pipeline_start_binding_for_alias(alias)
        if binding is None:
            return OptionalResolution(None)
        context = self.source_binding_context
        if not context.pipeline_input_files or not context.step_input_files:
            return OptionalResolution(None)

        step_candidates = self.source_candidates(context.step_input_files)
        pipeline_candidates = self.source_candidates(context.pipeline_input_files)
        axis_candidates = self.source_binding_axis_candidates(
            binding,
            pipeline_candidates,
            step_candidates,
        )
        if not axis_candidates:
            return OptionalResolution(None)
        return OptionalResolution(
            SourceBindingPlaneCandidateContext(
                request=SourceBindingRequestBase(alias=alias, binding=binding),
                axis_candidates=axis_candidates,
                step_candidates=step_candidates,
                pipeline_candidates=pipeline_candidates,
            )
        )

    def source_binding_plane_matched_context(
        self,
        context: "SourceBindingPlaneCandidateContext",
    ) -> OptionalResolution["SourceBindingPlaneMatchedContext"]:
        matched_indexes = self.source_binding_matched_axis_indexes(
            alias=context.request.alias,
            binding=context.request.binding,
            axis_candidates=context.axis_candidates,
            step_candidates=context.step_candidates,
            pipeline_candidates=context.pipeline_candidates,
        )
        if not matched_indexes:
            return OptionalResolution(None)
        return OptionalResolution(
            SourceBindingPlaneMatchedContext(
                alias=context.request.alias,
                matched_indexes=matched_indexes,
            )
        )

    def single_source_binding_plane_index(
        self,
        context: "SourceBindingPlaneMatchedContext",
    ) -> OptionalResolution[int]:
        matched_indexes = context.matched_indexes
        if len(matched_indexes) != 1:
            raise RuntimeError(
                f"Source binding alias {context.alias!r} matched multiple source planes "
                f"for axis {self.axis_id!r}: {matched_indexes!r}."
            )
        return OptionalResolution(matched_indexes[0])

    def pipeline_start_binding_for_alias(self, alias: str) -> NamedSourceBinding | None:
        binding = self.source_binding_plan.binding_for_alias(alias, self.group_key)
        if binding is None or binding.origin is not SourceBindingOrigin.PIPELINE_START:
            return None
        return binding

    def source_binding_axis_candidates(
        self,
        binding: NamedSourceBinding,
        pipeline_candidates: tuple["ParsedSourceCandidate", ...],
        step_candidates: tuple["ParsedSourceCandidate", ...],
    ) -> tuple["ParsedSourceCandidate", ...]:
        return SourceCandidateMatcher.axis_scoped_candidates(
            SourceCandidateMatcher.ordered_binding_candidates(
                binding=binding,
                candidates=pipeline_candidates,
            ),
            axis_id=self.axis_id,
            step_input_candidates=step_candidates,
        )

    def source_binding_matched_axis_indexes(
        self,
        *,
        alias: str,
        binding: NamedSourceBinding,
        axis_candidates: tuple["ParsedSourceCandidate", ...],
        step_candidates: tuple["ParsedSourceCandidate", ...],
        pipeline_candidates: tuple["ParsedSourceCandidate", ...],
    ) -> tuple[int, ...]:
        target_candidates = SourceCandidateMatcher.match_candidates(
            candidates=pipeline_candidates,
            binding=binding,
            inherit_components={},
        )
        current_candidates = SourceCandidateMatcher.match_image_set_candidates(
            alias,
            self.source_binding_plan.match_plan,
            step_candidates,
            target_candidates,
            pipeline_candidates,
            source_binding_plan=self.source_binding_plan,
            group_key=self.group_key,
        )
        current_paths = {candidate.resolved_path for candidate in current_candidates}
        return tuple(
            index
            for index, candidate in enumerate(axis_candidates)
            if candidate.resolved_path in current_paths
        )

    def resolve_source_objects(
        self,
        alias: str,
        current_image: Any,
    ) -> ObjectLabelSet:
        request = self._source_resolution_request(
            alias,
            ArtifactKind.OBJECT_LABELS,
            current_image,
        )
        labels = SourceBindingResolver.for_origin(request.binding.origin).resolve_image(
            request
        )
        return ObjectLabelSet(
            name=alias,
            labels=labels,
            source_image_name=alias,
        )

    def _source_resolution_request(
        self,
        alias: str,
        kind: ArtifactKind,
        current_image: Any,
    ) -> "SourceBindingResolutionRequest":
        return SourceBindingResolutionRequest(
            alias=alias,
            binding=self._require_source_binding(alias, kind),
            adapter=self,
            current_image=current_image,
        )

    def require_resolvable_source_aliases(
        self,
        aliases: tuple[str, ...],
    ) -> None:
        for alias in aliases:
            self._require_source_binding(alias, ArtifactKind.IMAGE)

    def has_source_binding(
        self,
        alias: str,
        kind: ArtifactKind | None = None,
    ) -> bool:
        binding = self.source_binding_plan.binding_for_alias(alias, self.group_key)
        return binding is not None and (
            kind is None or binding.artifact_kind is kind
        )

    def _require_source_binding(
        self,
        alias: str,
        kind: ArtifactKind,
    ) -> NamedSourceBinding:
        binding = self.source_binding_plan.binding_for_alias(alias, self.group_key)
        if binding is None:
            raise RuntimeError(
                f"Missing compiled source binding for CellProfiler "
                f"{kind.value} alias '{alias}' on axis '{self.axis_id}' and "
                f"group {self.group_key!r}."
            )
        if binding.artifact_kind is not kind:
            raise RuntimeError(
                f"CellProfiler source binding '{alias}' is declared as "
                f"{binding.artifact_kind.value}, not {kind.value}."
            )
        return binding

    def add_image(
        self,
        name: str,
        data: Any,
        *,
        dimensions: tuple[str, ...] = (),
        source_image_name: str | None = None,
    ) -> StoredRuntimeValue:
        return self._record_native_value(
            name,
            ArtifactKind.IMAGE,
            NamedImage(
                name=name,
                data=data,
                dimensions=dimensions,
                source_image_name=source_image_name,
            ),
        )

    def get_image(
        self,
        name: str,
        *,
        group_key: str | None = None,
    ) -> NamedImage:
        resolved_group_key = self.group_key if group_key is None else group_key
        cache_key = (resolved_group_key, name)
        cached = self._image_cache.get(cache_key)
        if cached is not None:
            return cached
        records = self._resolve_runtime_records(
            name=name,
            kind=ArtifactKind.IMAGE,
            group_key=group_key,
        )
        record = records[-1]
        data = (
            RuntimeRecordStackAuthority.stack_image_records(records)
            if len(records) > 1
            else record.value.data
        )
        schema = record.value.schema
        image = NamedImage(
            name=name,
            data=data,
            dimensions=schema.dimensions,
            source_image_name=schema.source_image_name,
        )
        self._image_cache[cache_key] = image
        return image

    def add_objects(
        self,
        name: str,
        labels: Any,
        *,
        source_image_name: str | None = None,
        dimensions: tuple[str, ...] = (),
        representation: ObjectLabelRepresentation = (
            ObjectLabelRepresentation.DENSE_LABELS
        ),
    ) -> StoredRuntimeValue:
        construct_started_at = time.perf_counter()
        if isinstance(labels, ObjectLabelSet):
            normalized_labels = RuntimeRecordStackAuthority.normalize_dense_object_label_payload(
                labels.labels
            )
            object_labels = ObjectLabelSet(
                name=name,
                labels=normalized_labels,
                unedited_labels=RuntimeRecordStackAuthority.normalize_dense_object_label_payload(
                    labels.unedited_labels
                ),
                small_removed_labels=RuntimeRecordStackAuthority.normalize_dense_object_label_payload(
                    labels.small_removed_labels
                ),
                declared_object_count=labels.declared_object_count,
                declared_object_ids=labels.declared_object_ids,
                declared_object_id_domains=labels.declared_object_id_domains,
                domain_scope=labels.domain_scope,
                spatial_origin_yx=labels.spatial_origin_yx,
                source_spatial_shape_yx=labels.source_spatial_shape_yx,
                source_image_name=source_image_name or labels.source_image_name,
                dimensions=dimensions or labels.dimensions,
                representation=labels.representation,
            )
        elif isinstance(labels, ObjectLabelPayload):
            normalized_labels = RuntimeRecordStackAuthority.normalize_dense_object_label_payload(
                labels.labels
            )
            object_labels = ObjectLabelSet(
                name=name,
                labels=normalized_labels,
                unedited_labels=RuntimeRecordStackAuthority.normalize_dense_object_label_payload(
                    labels.unedited_labels
                ),
                small_removed_labels=RuntimeRecordStackAuthority.normalize_dense_object_label_payload(
                    labels.small_removed_labels
                ),
                declared_object_count=labels.declared_object_count,
                declared_object_ids=labels.declared_object_ids,
                declared_object_id_domains=labels.declared_object_id_domains,
                domain_scope=labels.domain_scope,
                spatial_origin_yx=labels.spatial_origin_yx,
                source_spatial_shape_yx=labels.source_spatial_shape_yx,
                source_image_name=source_image_name,
                dimensions=dimensions,
                representation=representation,
            )
        else:
            source_domain = ObjectLabelSourceImageDomain.for_adapter_source_image(
                self,
                source_image_name,
            )
            object_labels = ObjectLabelSet(
                name=name,
                labels=RuntimeRecordStackAuthority.normalize_dense_object_label_payload(
                    labels
                ),
                spatial_origin_yx=source_domain.spatial_origin_yx,
                source_spatial_shape_yx=source_domain.source_spatial_shape_yx,
                source_image_name=source_image_name,
                dimensions=dimensions,
                representation=representation,
            )
        AdapterProfileLog.artifact(
            "adapter_construct_object_labels",
            time.perf_counter() - construct_started_at,
            artifact_name=name,
            kind=ArtifactKind.OBJECT_LABELS,
            payload_type=type(labels).__name__,
        )
        return self._record_native_value(
            name,
            ArtifactKind.OBJECT_LABELS,
            object_labels,
        )

    def get_objects(
        self,
        name: str,
        *,
        group_key: str | None = None,
        current_image: Any | None = None,
    ) -> ObjectLabelSet:
        resolved_group_key = self.runtime_input_group_key(
            name=name,
            kind=ArtifactKind.OBJECT_LABELS,
            group_key=group_key,
            current_image=current_image,
        )
        cache_key = (resolved_group_key, name)
        cached = self._object_cache.get(cache_key)
        if cached is not None:
            return cached
        records = self._resolve_runtime_records(
            name=name,
            kind=ArtifactKind.OBJECT_LABELS,
            group_key=group_key,
            current_image=current_image,
        )
        objects = (
            RuntimeRecordStackAuthority.stack_object_label_records(records)
            if len(records) > 1
            else ObjectLabelSet.from_runtime_value(records[0].value)
        )
        self._object_cache[cache_key] = objects
        return objects

    def runtime_input_group_key(
        self,
        *,
        name: str,
        kind: ArtifactKind,
        group_key: str | None = None,
        current_image: Any | None = None,
    ) -> str | None:
        """Return the artifact input group for this adapter/runtime context."""
        requested_group_key = self.group_key if group_key is None else group_key
        input_plan = self.artifact_inputs.get(name)
        if input_plan is None or input_plan.kind is not kind:
            return requested_group_key
        selected = input_plan.group_key_for_axis(
            axis_id=self.axis_id,
            requested_group_key=requested_group_key,
        )
        if selected not in (None, "default") or current_image is None:
            return selected
        group_keys = {str(group_key) for group_key in (input_plan.group_keys or ())}
        if not group_keys:
            return selected
        for candidate in self.source_candidates(
            self.source_binding_context.step_input_files
        ):
            if not self.current_image_matches_source_candidate(current_image, candidate):
                continue
            for value in candidate.metadata.values():
                normalized = str(value)
                if normalized in group_keys:
                    return normalized
        current_step_group = self.runtime_input_group_key_from_current_sources(
            group_keys
        )
        if current_step_group is not None:
            return current_step_group
        return selected

    def runtime_input_group_key_from_current_sources(
        self,
        group_keys: set[str],
    ) -> str | None:
        """Infer the active input group from this invocation's source files."""
        current_files = self.source_binding_context.current_step_input_files
        if not current_files:
            return None
        candidates = self.source_candidates(current_files)
        universe_files = (
            self.source_binding_context.pipeline_input_files
            or self.source_binding_context.step_input_files
            or current_files
        )
        universe_candidates = self.source_candidates(universe_files)
        field_values = self.current_source_group_field_values(
            candidates,
            universe_candidates,
            group_keys,
        )
        matched_groups = tuple(
            value
            for values in field_values.values()
            if len(values) == 1
            for value in values
        )
        logger.debug(
            "Resolved current source group candidates from fields %s for groups %s",
            field_values,
            sorted(group_keys),
        )
        if len(matched_groups) == 1:
            return matched_groups[0]
        return None

    @staticmethod
    def current_source_group_field_values(
        candidates: tuple["ParsedSourceCandidate", ...],
        universe_candidates: tuple["ParsedSourceCandidate", ...],
        group_keys: set[str],
    ) -> Mapping[str, frozenset[str]]:
        """Return candidate metadata fields whose values can select input groups."""
        return MappingProxyType(
            {
                field_name: frozenset(values)
                for field_name in tuple(
                    dict.fromkeys(
                        field_name
                        for candidate in candidates
                        for field_name in candidate.metadata
                    )
                )
                for values in (
                    tuple(
                        str(candidate.metadata[field_name])
                        for candidate in candidates
                        if field_name in candidate.metadata
                        and str(candidate.metadata[field_name]) in group_keys
                    ),
                )
                if (
                    values
                    and len(values) == len(candidates)
                    and group_keys.issubset(
                        {
                            str(candidate.metadata[field_name])
                            for candidate in universe_candidates
                            if field_name in candidate.metadata
                        }
                    )
                )
            }
        )

    def current_image_matches_source_candidate(
        self,
        current_image: Any,
        candidate: "ParsedSourceCandidate",
    ) -> bool:
        """Return whether the payload metadata names a parsed source candidate."""
        metadata = image_payload_metadata(current_image)
        source_paths = tuple(
            str(path)
            for path in (*metadata.channel_source_paths, metadata.source_path)
            if path is not None and str(path)
        )
        if not source_paths:
            return False
        candidate_paths = {
            self.cellprofiler_source_order_path(candidate.path),
            self.cellprofiler_source_order_path(candidate.resolved_path),
        }
        return any(
            self.cellprofiler_source_order_path(path) in candidate_paths
            for path in source_paths
        )

    def add_measurements(
        self,
        name: str,
        rows: Any,
        *,
        object_name: str | None = None,
        fields: tuple[FieldSpec, ...] = (),
        object_id_field: str | None = None,
        source_image_name: str | None = None,
    ) -> StoredRuntimeValue:
        validation_started_at = time.perf_counter()
        if object_name is not None:
            self.require_artifact_available(
                name=object_name,
                kind=ArtifactKind.OBJECT_LABELS,
            )
        AdapterProfileLog.measurement_artifact(
            "adapter_measurement_subject_validation",
            time.perf_counter() - validation_started_at,
            artifact_name=name,
            object_name=object_name,
        )
        table_started_at = time.perf_counter()
        measurement_table = MeasurementTable(
            name=name,
            rows=rows,
            object_name=object_name,
            fields=fields,
            object_id_field=object_id_field,
            source_image_name=source_image_name,
            validated_runtime_schema=bool(fields),
        )
        AdapterProfileLog.measurement_artifact(
            "adapter_measurement_table_construct",
            time.perf_counter() - table_started_at,
            artifact_name=name,
            object_name=object_name,
            fields_declared=bool(fields),
        )
        record_started_at = time.perf_counter()
        stored_value = self._record_native_value(
            name,
            ArtifactKind.MEASUREMENTS,
            measurement_table,
        )
        AdapterProfileLog.measurement_artifact(
            "adapter_measurement_record_native",
            time.perf_counter() - record_started_at,
            artifact_name=name,
            object_name=object_name,
        )
        return stored_value

    def get_measurements(
        self,
        name: str,
        *,
        group_key: str | None = None,
    ) -> MeasurementTable:
        record = self._resolve_runtime_record(
            name=name,
            kind=ArtifactKind.MEASUREMENTS,
            group_key=group_key,
        )
        return MeasurementTable.from_runtime_value(record.value)

    def measurement_tables_for_object(
        self,
        object_name: str,
        *,
        group_key: str | None = None,
        match_group: bool = True,
    ) -> tuple[MeasurementTable, ...]:
        """Return prior measurement tables whose subject is an object set."""
        resolved_group_key = self.group_key if group_key is None else group_key
        cache_key = ObjectMeasurementTableCacheKey(
            group_key=resolved_group_key if match_group else None,
            match_group=match_group,
            object_name=object_name,
        )
        cached = self._measurement_cache.get(cache_key)
        if cached is not None:
            return cached
        object_table_cache = self.object_measurement_table_cache()
        cached = object_table_cache.get(cache_key)
        if cached is not None:
            self._measurement_cache[cache_key] = cached
            return cached
        index_cache_key = ObjectMeasurementTableIndexCacheKey(
            group_key=resolved_group_key if match_group else None,
            match_group=match_group,
        )
        object_table_index_cache = self.object_measurement_table_index_cache()
        object_table_index = object_table_index_cache.get(index_cache_key)
        if object_table_index is None:
            object_table_index = ObjectMeasurementTableIndex.from_tables(
                runtime_measurement_tables(
                    self._measurement_query_context(
                        group_key=group_key,
                        match_group=match_group,
                    )
                )
            )
            object_table_index_cache[index_cache_key] = object_table_index
        tables = object_table_index.for_object(object_name)
        if tables is None:
            tables = runtime_measurement_tables_for_object(
                self._measurement_query_context(
                    group_key=group_key,
                    match_group=match_group,
                ),
                object_name,
            )
            tables = RuntimeSliceProjection.measurement_tables_with_repeated_scalar_slice_offsets(
                tables
            )
        object_table_cache[cache_key] = tables
        self._measurement_cache[cache_key] = tables
        return tables

    def measurement_tables_for_object_feature(
        self,
        object_name: str,
        feature_name: str,
        *,
        group_key: str | None = None,
        match_group: bool = True,
    ) -> tuple[MeasurementTable, ...]:
        """Return object tables that may carry the requested feature."""
        resolved_group_key = self.group_key if group_key is None else group_key
        index_cache_key = ObjectMeasurementTableIndexCacheKey(
            group_key=resolved_group_key if match_group else None,
            match_group=match_group,
        )
        object_table_index_cache = self.object_measurement_table_index_cache()
        object_table_index = object_table_index_cache.get(index_cache_key)
        if object_table_index is None:
            object_table_index = ObjectMeasurementTableIndex.from_tables(
                runtime_measurement_tables(
                    self._measurement_query_context(
                        group_key=group_key,
                        match_group=match_group,
                    )
                )
            )
            object_table_index_cache[index_cache_key] = object_table_index
        tables = object_table_index.for_object_feature(object_name, feature_name)
        if tables is None:
            return self.measurement_tables_for_object(
                object_name,
                group_key=group_key,
                match_group=match_group,
            )
        return tables

    def measurement_tables_for_object_feature_axis_scope(
        self,
        object_name: str,
        feature_name: str,
        *,
        group_key: str | None = None,
        image_number: int | None = None,
    ) -> tuple[MeasurementTable, ...]:
        """Return feature-bearing object tables in their declared runtime axis scope."""
        scoped_table_sets = (
            self.measurement_tables_for_object_feature(
                object_name,
                feature_name,
                group_key=group_key,
            ),
            self.measurement_tables_for_object_feature(
                object_name,
                feature_name,
                group_key=group_key,
                match_group=False,
            ),
        )
        runtime_slice_plane_index = self.runtime_slice_plane_index()
        candidates: list[tuple[MeasurementTable, ...]] = []
        for tables in scoped_table_sets:
            if image_number is not None:
                candidates.append(measurement_tables_for_image_number(tables, image_number))
            if runtime_slice_plane_index is not None:
                candidates.append(measurement_tables_for_slice(tables, runtime_slice_plane_index))
            candidates.append(tables)
        for candidate_tables in candidates:
            if optional_measurement_value_index(
                candidate_tables,
                feature_name,
                object_name=object_name,
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            ) is not None:
                return candidate_tables
        return scoped_table_sets[0]

    def measurement_values_for_label_slices(
        self,
        object_name: str,
        feature_name: str,
        labels: object,
        *,
        group_key: str | None = None,
        image_number: int | None = None,
    ) -> tuple[Any, ...]:
        """Return object measurements aligned to label planes with adapter caching."""
        started_at = time.perf_counter()
        profile = ObjectFeatureProfileContext(object_name, feature_name)
        label_array = np.asarray(labels)
        domain_started_at = time.perf_counter()
        label_domain = self._dense_label_domain(labels)
        profile.domain(
            time.perf_counter() - domain_started_at,
            count=len(label_domain),
            ndim=label_array.ndim,
        )
        resolved_group_key = self.group_key if group_key is None else group_key
        query = RuntimeObjectLabelMeasurementQuery(
            axis_id=self.axis_id,
            group_key=resolved_group_key,
            object_name=object_name,
            feature_name=feature_name,
            label_domain=label_domain,
            image_number=image_number,
        )
        cached = self._measurement_cache.get(query)
        if cached is not None:
            profile.query_values(
                time.perf_counter() - started_at,
                cached="adapter",
            )
            return cached
        object_label_values_cache = self.object_label_measurement_values_cache()
        cached = object_label_values_cache.get(query)
        if cached is not None:
            self._measurement_cache[query] = cached
            profile.query_values(
                time.perf_counter() - started_at,
                cached="store",
            )
            return cached
        if label_array.ndim <= 2:
            tables_started_at = time.perf_counter()
            tables = self.measurement_tables_for_object_feature_axis_scope(
                object_name,
                feature_name,
                group_key=group_key,
                image_number=image_number,
            )
            profile.query_tables(
                time.perf_counter() - tables_started_at,
                count=len(tables),
            )
            extract_started_at = time.perf_counter()
            values = (
                measurement_values_for_feature(
                    tables,
                    feature_name,
                    object_count=len(label_domain),
                    object_ids=label_domain,
                    object_name=object_name,
                    dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
                ),
            )
            profile.query_extract(
                time.perf_counter() - extract_started_at,
                count=len(values[0]),
            )
            self._measurement_cache[query] = values
            object_label_values_cache[query] = values
            profile.query_values(
                time.perf_counter() - started_at,
                cached=False,
            )
            return values
        tables_started_at = time.perf_counter()
        tables = self._measurement_tables_for_multiplane_labels(
            object_name,
            labels,
            group_key=group_key,
        )
        profile.query_tables(
            time.perf_counter() - tables_started_at,
            count=len(tables),
        )
        extract_started_at = time.perf_counter()
        if image_number is None or label_array.ndim > 2:
            values = measurement_values_for_label_slices(
                tables,
                feature_name,
                labels,
                object_name=object_name,
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            )
        else:
            label_planes = tuple(
                label_array[index] for index in range(label_array.shape[0])
            )
            values = tuple(
                measurement_values_for_feature(
                    measurement_tables_for_image_number(
                        tables,
                        image_number + slice_index,
                    ),
                    feature_name,
                    object_count=len(self._dense_label_domain(label_plane)),
                    object_ids=self._dense_label_domain(label_plane),
                    object_name=object_name,
                    dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
                )
                for slice_index, label_plane in enumerate(label_planes)
            )
        profile.query_extract(
            time.perf_counter() - extract_started_at,
            count=len(values),
        )
        self._measurement_cache[query] = values
        object_label_values_cache[query] = values
        profile.query_values(
            time.perf_counter() - started_at,
            cached=False,
        )
        return values

    def measurement_values_for_label_slice_batch(
        self,
        requests: Mapping[str, tuple[str, object, int | None]],
        *,
        feature_name: str,
        group_key: str | None = None,
    ) -> Mapping[str, tuple[Any, ...]]:
        """Return label-aligned object-feature vectors for one shared feature."""
        started_at = time.perf_counter()
        if not requests:
            return MappingProxyType({})
        profile = LabelBatchProfileContext(feature_name)

        resolved_group_key = self.group_key if group_key is None else group_key
        label_domains: dict[str, tuple[int, ...]] = {}
        label_arrays: dict[str, np.ndarray] = {}
        uncached_queries: dict[str, RuntimeObjectLabelMeasurementQuery] = {}
        values_by_object: dict[str, tuple[Any, ...]] = {}
        object_label_values_cache = self.object_label_measurement_values_cache()
        cache_started_at = time.perf_counter()
        for object_name, (_feature_name, labels, image_number) in requests.items():
            label_array = np.asarray(labels)
            label_arrays[object_name] = label_array
            label_domain = self._dense_label_domain(labels)
            label_domains[object_name] = label_domain
            query = RuntimeObjectLabelMeasurementQuery(
                axis_id=self.axis_id,
                group_key=resolved_group_key,
                object_name=object_name,
                feature_name=feature_name,
                label_domain=label_domain,
                image_number=image_number,
            )
            cached = self._measurement_cache.get(query)
            if cached is None:
                cached = object_label_values_cache.get(query)
                if cached is not None:
                    self._measurement_cache[query] = cached
            if cached is None:
                uncached_queries[object_name] = query
                continue
            values_by_object[object_name] = cached
        profile.cache(
            time.perf_counter() - cache_started_at,
            requested=len(requests),
            uncached=len(uncached_queries),
        )

        if not uncached_queries:
            profile.values(
                time.perf_counter() - started_at,
                cached=True,
                count=len(values_by_object),
            )
            return MappingProxyType(values_by_object)

        if any(label_arrays[object_name].ndim > 2 for object_name in uncached_queries):
            for object_name in uncached_queries:
                values_by_object[object_name] = self.measurement_values_for_label_slices(
                    object_name,
                    feature_name,
                    requests[object_name][1],
                    group_key=group_key,
                    image_number=requests[object_name][2],
                )
            profile.values(
                time.perf_counter() - started_at,
                cached=False,
                fallback="multiplane",
                count=len(values_by_object),
            )
            return MappingProxyType(values_by_object)

        tables_started_at = time.perf_counter()
        tables_by_object = {
            object_name: self.measurement_tables_for_object_feature_axis_scope(
                object_name,
                feature_name,
                group_key=group_key,
                image_number=image_number,
            )
            for object_name in uncached_queries
            for image_number in (requests[object_name][2],)
        }
        profile.tables(
            time.perf_counter() - tables_started_at,
            count=sum(len(tables) for tables in tables_by_object.values()),
        )
        indexes_started_at = time.perf_counter()
        try:
            value_indexes_by_object = measurement_value_indexes_for_object_feature_batch(
                tables_by_object,
                feature_name,
                object_names=tuple(uncached_queries),
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            )
        except ValueError as exc:
            context = self._measurement_query_context(
                group_key=group_key,
                match_group=True,
            )
            all_tables = runtime_measurement_tables(context)
            table_context = tuple(
                f"{table.name}:{table.object_name or '<none>'}:"
                f"{type(table.rows).__name__}"
                for table in all_tables
            )
            raise ValueError(
                f"{exc} Visible measurement tables in group "
                f"{context.group_key!r}: {table_context!r}."
            ) from exc
        profile.indexes(
            time.perf_counter() - indexes_started_at,
            count=len(value_indexes_by_object),
        )
        align_started_at = time.perf_counter()
        for object_name, query in uncached_queries.items():
            values = (
                measurement_values_for_feature(
                    tables_by_object[object_name],
                    feature_name,
                    object_count=len(label_domains[object_name]),
                    object_ids=label_domains[object_name],
                    object_name=object_name,
                    dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
                )
                if object_name not in value_indexes_by_object
                else (
                    ObjectLabelMeasurementValues.from_value_mapping(
                        label_domains[object_name],
                        value_indexes_by_object[object_name][0],
                    ).values
                    if value_indexes_by_object[object_name][0]
                    else ObjectLabelMeasurementValues.from_positional_values(
                        label_domains[object_name],
                        value_indexes_by_object[object_name][1],
                    ).values
                )
            )
            value_slices = (values,)
            self._measurement_cache[query] = value_slices
            object_label_values_cache[query] = value_slices
            values_by_object[object_name] = value_slices
        profile.align(
            time.perf_counter() - align_started_at,
            count=len(uncached_queries),
        )
        profile.values(
            time.perf_counter() - started_at,
            cached=False,
            count=len(values_by_object),
        )
        return MappingProxyType(values_by_object)

    def measurement_values_for_object_feature(
        self,
        object_name: str,
        feature_name: str,
        *,
        group_key: str | None = None,
    ) -> Any:
        """Return one object feature vector over the object's declared domain."""
        started_at = time.perf_counter()
        profile = ObjectFeatureProfileContext(object_name, feature_name)
        objects_started_at = time.perf_counter()
        objects = self.get_objects(object_name, group_key=group_key)
        profile.get_objects(time.perf_counter() - objects_started_at)
        domain_started_at = time.perf_counter()
        object_domain = dense_object_label_id_domain(objects)
        profile.object_domain(
            time.perf_counter() - domain_started_at,
            count=len(object_domain),
        )
        resolved_group_key = self.group_key if group_key is None else group_key
        cache_key = RuntimeObjectFeatureMeasurementQuery(
            group_key=resolved_group_key,
            object_name=object_name,
            feature_name=feature_name,
            object_domain=object_domain,
        )
        object_feature_cache = self.object_feature_value_cache()
        cached = object_feature_cache.get(cache_key)
        if cached is not None:
            profile.feature_values(
                time.perf_counter() - started_at,
                cached=True,
            )
            return cached
        tables_started_at = time.perf_counter()
        tables = self.measurement_tables_for_object(
            object_name,
            group_key=group_key,
        )
        profile.feature_tables(
            time.perf_counter() - tables_started_at,
            count=len(tables),
        )
        values_started_at = time.perf_counter()
        values = measurement_values_for_feature(
            tables,
            feature_name,
            object_count=len(object_domain),
            object_ids=object_domain,
            object_name=object_name,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )
        profile.feature_extract(
            time.perf_counter() - values_started_at,
            count=len(values),
        )
        object_feature_cache[cache_key] = values
        profile.feature_values(
            time.perf_counter() - started_at,
            cached=False,
        )
        return values

    def apply_measurement_table_cache_mutation(
        self,
        table: MeasurementTable,
    ) -> None:
        """Apply registered cache policies for one measurement-table write."""
        semantics_started_at = time.perf_counter()
        table_semantics = (
            MeasurementTableObjectFeatureSemantics.from_table(table)
        )
        AdapterProfileLog.measurement_cache(
            "adapter_measurement_table_semantics",
            time.perf_counter() - semantics_started_at,
            object_count=len(table_semantics.object_names),
            feature_count=len(table_semantics.feature_names),
        )
        mutation = MeasurementTableCacheMutation(
            adapter=self,
            table=table,
            table_semantics=table_semantics,
        )
        for policy in MeasurementTableCacheMutationPolicy.registered_policies():
            policy_started_at = time.perf_counter()
            policy.apply(mutation)
            AdapterProfileLog.measurement_cache_policy(
                time.perf_counter() - policy_started_at,
                policy_name=type(policy).__name__,
                object_count=len(table_semantics.object_names),
                feature_count=len(table_semantics.feature_names),
            )

    def _measurement_tables_for_multiplane_labels(
        self,
        object_name: str,
        labels: object,
        *,
        group_key: str | None,
    ) -> tuple[MeasurementTable, ...]:
        """Return object tables aligned from producer groups into label planes."""
        label_array = np.asarray(labels)
        cache_key = (
            "multiplane_object_tables",
            self.runtime_value_store.revision,
            self.axis_id,
            object_name,
            id(labels),
        )
        cached = self._measurement_cache.get(cache_key)
        if cached is not None:
            return cached

        records = self.runtime_value_store.find(
            kind=ArtifactKind.MEASUREMENTS,
            axis_id=self.axis_id,
            match_group=False,
        )
        scoped_tables: list[tuple[str | None, MeasurementTable]] = []
        for record in records:
            table = MeasurementTable.from_runtime_value(record.value)
            if RuntimeSliceProjection.measurement_table_matches_object(
                table,
                object_name,
            ):
                scoped_tables.append((record.key.scope.group_key, table))

        if not scoped_tables:
            tables = self.measurement_tables_for_object(
                object_name,
                group_key=group_key,
                match_group=False,
            )
            self._measurement_cache[cache_key] = tables
            return tables

        group_order = tuple(
            dict.fromkeys(
                group_key for group_key, _table in scoped_tables if group_key is not None
            )
        )
        group_slice_counts = {
            group_key: max(
                RuntimeSliceProjection.measurement_table_effective_slice_count(table)
                for table_group_key, table in scoped_tables
                if table_group_key == group_key
            )
            for group_key in group_order
        }
        if (
            label_array.ndim > 2
            and not group_order
            and len(scoped_tables) == 1
            and RuntimeSliceProjection.measurement_table_effective_slice_count(
                scoped_tables[0][1]
            )
            == 1
            and _label_stack_repeats_first_plane(label_array)
        ):
            tables = (
                RuntimeSliceProjection.measurement_table_broadcast_to_slice_count(
                    scoped_tables[0][1],
                    label_array.shape[0],
                ),
            )
            self._measurement_cache[cache_key] = tables
            return tables

        group_offsets: dict[str, int] = {}
        offset = 0
        for group_key in group_order:
            group_offsets[group_key] = offset
            offset += group_slice_counts[group_key]

        tables = tuple(
            (
                RuntimeSliceProjection.measurement_table_with_slice_offset(
                    table,
                    group_offsets[table_group_key],
                )
                if table_group_key is not None
                else table
            )
            for table_group_key, table in scoped_tables
        )
        if offset and label_array.ndim > 2 and offset != label_array.shape[0]:
            AdapterProfileLog.measurement_slice_mismatch(
                0.0,
                object_name=object_name,
                measurement_slices=offset,
                label_slices=label_array.shape[0],
            )
        self._measurement_cache[cache_key] = tables
        return tables

    def _dense_label_domain(self, labels: object) -> tuple[int, ...]:
        cache_key = id(labels)
        cached = self._label_domain_cache.get(cache_key)
        if cached is not None:
            return cached
        domain = dense_object_label_id_domain(labels)
        self._label_domain_cache[cache_key] = domain
        return domain

    def measurement_tables(
        self,
        *,
        group_key: str | None = None,
        match_group: bool = True,
    ) -> tuple[MeasurementTable, ...]:
        """Return measurement tables visible to the current runtime scope."""
        resolved_group_key = self.group_key if group_key is None else group_key
        cache_key = (
            "all",
            self.runtime_value_store.revision,
            resolved_group_key if match_group else None,
            match_group,
        )
        cached = self._measurement_cache.get(cache_key)
        if cached is not None:
            return cached
        tables = runtime_measurement_tables(
            self._measurement_query_context(
                group_key=group_key,
                match_group=match_group,
            )
        )
        self._measurement_cache[cache_key] = tables
        return tables

    def measurement_tables_for_scope(
        self,
        scope: MeasurementScope,
        *,
        name: str | None = None,
        group_key: str | None = None,
        match_group: bool = True,
    ) -> tuple[MeasurementTable, ...]:
        """Return measurement tables for one generic semantic scope."""
        resolved_group_key = self.group_key if group_key is None else group_key
        cache_key = (
            "scope",
            self.runtime_value_store.revision,
            resolved_group_key if match_group else None,
            match_group,
            scope,
            name,
        )
        cached = self._measurement_cache.get(cache_key)
        if cached is not None:
            return cached
        tables = runtime_measurement_tables_for_scope(
            self._measurement_query_context(
                group_key=group_key,
                match_group=match_group,
            ),
            scope,
            name,
        )
        self._measurement_cache[cache_key] = tables
        return tables

    def add_relationship(
        self,
        name: str,
        *,
        parent_object_name: str,
        child_object_name: str,
        parent_ids: Any,
        child_ids: Any,
        slice_indices: tuple[int, ...] = (),
        slice_count: int | None = None,
    ) -> StoredRuntimeValue:
        if not self._is_declared_output(name, ArtifactKind.RELATIONSHIPS):
            self._require_artifact_declared_or_available(
                name=parent_object_name,
                kind=ArtifactKind.OBJECT_LABELS,
            )
            self._require_artifact_declared_or_available(
                name=child_object_name,
                kind=ArtifactKind.OBJECT_LABELS,
            )
        semantics = RelationshipSemantics.parent_child(
            parent_object_name,
            child_object_name,
        )
        return self._record_native_value(
            name,
            ArtifactKind.RELATIONSHIPS,
            ObjectRelationship(
                name=name,
                source=semantics.source,
                target=semantics.target,
                source_ids=parent_ids,
                target_ids=child_ids,
                relationship_type=semantics.relationship_type,
                slice_indices=slice_indices,
                slice_count=slice_count,
            ),
        )

    def get_relationship(
        self,
        name: str,
        *,
        group_key: str | None = None,
    ) -> ObjectRelationship:
        return runtime_relationship(
            self._query_context(group_key),
            name=name,
        )

    def add_spatial_grid(
        self,
        name: str,
        grid: SpatialGrid | Mapping[str, Any] | RuntimeSliceAlignedValues[Any],
    ) -> StoredRuntimeValue:
        if isinstance(grid, RuntimeSliceAlignedValues):
            return self._record_native_value(
                name,
                ArtifactKind.SPATIAL_GRID,
                RuntimeSliceAlignedValues(
                    slices=tuple(
                        SpatialGridValueAuthority.native_value(name, value)
                        for value in grid.slices
                    )
                ),
            )
        spatial_grid = (
            grid.with_name(name)
            if isinstance(grid, SpatialGrid)
            else SpatialGrid.from_mapping(name, grid)
        )
        return self._record_native_value(
            name,
            ArtifactKind.SPATIAL_GRID,
            spatial_grid,
        )

    def get_spatial_grid(
        self,
        name: str,
        *,
        group_key: str | None = None,
    ) -> SpatialGrid | RuntimeSliceAlignedValues[SpatialGrid]:
        records = self._resolve_runtime_records(
            name=name,
            kind=ArtifactKind.SPATIAL_GRID,
            group_key=group_key,
        )
        grids = tuple(
            SpatialGridValueAuthority.record_value(name, record)
            for record in records
        )
        return SpatialGridValueAuthority.single_spatial_grid(name, grids)

    def _record_native_value(
        self,
        name: str,
        expected_kind: ArtifactKind,
        native_value: Any,
    ) -> StoredRuntimeValue:
        total_started_at = time.perf_counter()
        profile = NativeRecordProfileContext(name, expected_kind)
        plan_started_at = time.perf_counter()
        output_plan = self._require_output_plan(name, expected_kind)
        slice_count = RuntimeSliceProjection.slice_count_from_values((native_value,))
        output_group_keys = output_plan.runtime_slice_group_keys(
            requested_group_key=self.group_key,
            slice_count=slice_count,
        )
        profile.event(
            "adapter_require_output_plan",
            time.perf_counter() - plan_started_at,
        )
        store_started_at = time.perf_counter()
        stored_value: StoredRuntimeValue | None = None
        for slice_index, output_group_key in enumerate(output_group_keys):
            plan = output_plan.for_group(output_group_key)
            group_native_value = (
                RuntimeSliceProjection.value_for_slice(
                    native_value,
                    slice_index,
                    slice_count,
                )
                if slice_count is not None and len(output_group_keys) > 1
                else native_value
            )
            normalize_started_at = time.perf_counter()
            runtime_value = normalize_artifact_value(
                plan,
                group_native_value,
                axis_id=self.axis_id,
            )
            profile.normalized_value(
                time.perf_counter() - normalize_started_at,
                payload_type=type(runtime_value.data).__name__,
                group_key=output_group_key,
            )
            save_started_at = time.perf_counter()
            runtime_path = plan.path
            self._save_payload(runtime_value.data, runtime_path)
            profile.group_event(
                "adapter_save_payload",
                time.perf_counter() - save_started_at,
                output_group_key,
            )
            replace_started_at = time.perf_counter()
            stored_value = self.runtime_value_store.replace(
                runtime_value,
                path=runtime_path,
                backend=self.backend,
            )
            profile.group_event(
                "adapter_runtime_store_replace_only",
                time.perf_counter() - replace_started_at,
                output_group_key,
            )
            if expected_kind is ArtifactKind.MEASUREMENTS:
                table_started_at = time.perf_counter()
                measurement_table = MeasurementTable.from_runtime_value(runtime_value)
                profile.group_event(
                    "adapter_measurement_table_from_runtime_value",
                    time.perf_counter() - table_started_at,
                    output_group_key,
                )
                cache_mutation_started_at = time.perf_counter()
                self.apply_measurement_table_cache_mutation(measurement_table)
                profile.group_event(
                    "adapter_measurement_cache_mutation",
                    time.perf_counter() - cache_mutation_started_at,
                    output_group_key,
                )
        if stored_value is None:
            raise RuntimeError(
                f"No runtime artifact groups were selected for '{name}' "
                f"({expected_kind.value})."
            )
        invalidation_started_at = time.perf_counter()
        self.invalidate_runtime_query_caches_for_kind(expected_kind)
        profile.event(
            "adapter_runtime_query_cache_invalidation",
            time.perf_counter() - invalidation_started_at,
        )
        profile.event(
            "adapter_runtime_store_replace",
            time.perf_counter() - store_started_at,
        )
        profile.event(
            "adapter_record_native_value",
            time.perf_counter() - total_started_at,
        )
        return stored_value

    def clear_runtime_query_caches(self) -> None:
        """Clear every runtime query cache owned by this adapter."""
        self._image_cache.clear()
        self._object_cache.clear()
        self._measurement_cache.clear()
        self.object_feature_value_cache().clear()
        self.object_label_measurement_values_cache().clear()
        self.object_measurement_table_cache().clear()
        self.object_measurement_table_index_cache().clear()
        self._label_domain_cache.clear()
        self._artifact_availability_cache.clear()

    def clear_measurement_query_cache(self) -> None:
        """Clear adapter-local measurement queries after measurement writes.

        Process-wide object/feature measurement caches are mutated by
        MeasurementTableCacheMutationPolicy implementations so unrelated
        feature indexes survive derived-measurement writes.
        """
        self._measurement_cache.clear()

    def _require_output_plan(
        self,
        name: str,
        expected_kind: ArtifactKind,
    ) -> ArtifactOutputPlan:
        plan = self.artifact_outputs.get(name)
        if plan is None:
            raise RuntimeError(
                f"No compiled output plan for CellProfiler artifact '{name}' "
                f"({expected_kind.value})."
            )
        if plan.kind is not expected_kind:
            raise ValueError(
                f"CellProfiler artifact '{name}' expected output kind "
                f"{expected_kind.value}, got compiled kind {plan.kind.value}."
            )
        return plan

    def _query_context(
        self,
        group_key: str | None = None,
    ) -> RuntimeArtifactQueryContext:
        resolved_group_key = self.group_key if group_key is None else group_key
        return RuntimeArtifactQueryContext(
            self.runtime_value_store,
            self.axis_id,
            resolved_group_key,
        )

    def _resolve_runtime_record(
        self,
        *,
        name: str,
        kind: ArtifactKind,
        group_key: str | None = None,
        current_image: Any | None = None,
    ) -> StoredRuntimeValue:
        records = self._resolve_runtime_records(
            name=name,
            kind=kind,
            group_key=group_key,
            current_image=current_image,
        )
        if len(records) != 1:
            raise RuntimeError(
                f"CellProfiler runtime artifact input '{name}' resolved to "
                f"{len(records)} grouped records; use a typed grouped accessor."
            )
        return records[0]

    def _resolve_runtime_records(
        self,
        *,
        name: str,
        kind: ArtifactKind,
        group_key: str | None = None,
        current_image: Any | None = None,
    ) -> tuple[StoredRuntimeValue, ...]:
        input_plan = self.artifact_inputs.get(name)
        resolved_group_key = self.runtime_input_group_key(
            name=name,
            kind=kind,
            group_key=group_key,
            current_image=current_image,
        )
        if input_plan is not None:
            if input_plan.kind is not kind:
                raise ValueError(
                    f"CellProfiler artifact input '{name}' expected kind "
                    f"{kind.value}, got compiled kind {input_plan.kind.value}."
                )
            if _is_global_grouped_input_request(input_plan, resolved_group_key):
                return tuple(
                    self.runtime_value_store.resolve(
                        _runtime_query_for_input_plan(
                            input_plan,
                            axis_id=self.axis_id,
                            group_key=input_group_key,
                            backend=self.backend,
                        ),
                        purpose="CellProfiler grouped runtime artifact input",
                    )
                    for input_group_key in input_plan.group_keys
                )
            return (
                self.runtime_value_store.resolve(
                    _runtime_query_for_input_plan(
                        input_plan.for_group(resolved_group_key) or input_plan,
                        axis_id=self.axis_id,
                        group_key=resolved_group_key,
                        backend=self.backend,
                    ),
                    purpose="CellProfiler runtime artifact input",
                ),
            )
        try:
            return (self._query_context(group_key).resolve(name=name, kind=kind),)
        except RuntimeError:
            records = self.runtime_value_store.find(
                name=name,
                kind=kind,
                axis_id=self.axis_id,
            )
            if len(records) > 1 and resolved_group_key in (None, "default"):
                return records
            raise

    def _require_artifact_declared_or_available(
        self,
        *,
        name: str,
        kind: ArtifactKind,
    ) -> None:
        if name in self.artifact_outputs:
            plan = self._require_output_plan(name, kind)
            if plan.kind is kind:
                return
        self._resolve_runtime_records(name=name, kind=kind)

    def _is_declared_output(self, name: str, kind: ArtifactKind) -> bool:
        if name not in self.artifact_outputs:
            return False
        try:
            return self._require_output_plan(name, kind).kind is kind
        except Exception:
            return False

    def _measurement_query_context(
        self,
        *,
        group_key: str | None,
        match_group: bool,
    ) -> RuntimeArtifactQueryContext:
        resolved_group_key = self.group_key if group_key is None else group_key
        return RuntimeArtifactQueryContext(
            self.runtime_value_store,
            self.axis_id,
            resolved_group_key if match_group else None,
        )

    def _save_payload(self, data: Any, path: str) -> None:
        if self.filemanager is None:
            raise RuntimeError(
                "CellProfilerRuntimeAdapter.filemanager is required for writes; "
                "adapter writes must persist through the OpenHCS VFS boundary."
            )
        replace_runtime_artifact_payload(
            self.filemanager,
            data,
            RuntimeArtifactLocation(path=path, backend=self.backend),
        )

    def source_candidates(
        self,
        file_paths: tuple[str, ...],
    ) -> tuple["ParsedSourceCandidate", ...]:
        """Return parsed source candidates for this runtime source universe.

        Source resolution may query the same step-input and pipeline-start
        universes from separate CellProfiler runtime adapters. Parsing is pure
        for the path tuple, source-binding context, metadata rules and filename
        parser, so the cache key carries those semantic inputs explicitly.
        """
        candidates = self._source_candidate_cache.get(file_paths)
        if candidates is not None:
            return candidates
        cache_key = _source_candidate_cache_key(file_paths, self)
        candidates = None
        candidates = _SOURCE_CANDIDATE_PROCESS_CACHE.get(cache_key)
        if candidates is not None:
            _SOURCE_CANDIDATE_PROCESS_CACHE.move_to_end(cache_key)
            self._source_candidate_cache[file_paths] = candidates
        if candidates is None:
            started_at = time.perf_counter()
            candidates = _parse_source_candidates(file_paths, self)
            self._source_candidate_cache[file_paths] = candidates
            _SOURCE_CANDIDATE_PROCESS_CACHE[cache_key] = candidates
            _SOURCE_CANDIDATE_PROCESS_CACHE.move_to_end(cache_key)
            if len(_SOURCE_CANDIDATE_PROCESS_CACHE) > _SOURCE_CANDIDATE_CACHE_LIMIT:
                _SOURCE_CANDIDATE_PROCESS_CACHE.popitem(last=False)
            AdapterProfileLog.source_candidates(
                "source_candidates_parse",
                time.perf_counter() - started_at,
                count=len(candidates),
            )
        return candidates


MeasurementQueryCacheInvalidationFamily.bind_adapter_caches(CellProfilerRuntimeAdapter)


class RuntimeArtifactCacheInvalidationPolicy(
    EnumKeyedStrategyMixin[ArtifactKind],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal cache invalidation policy for one runtime artifact domain."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "kind"
    kind: ClassVar[ArtifactKind | None] = None

    @classmethod
    def for_kind(cls, kind: ArtifactKind) -> "RuntimeArtifactCacheInvalidationPolicy":
        policy_type = cls.__registry__.get(
            kind.value,
            FullRuntimeArtifactCacheInvalidationPolicy,
        )
        return policy_type()

    @abstractmethod
    def invalidate(self, adapter: CellProfilerRuntimeAdapter) -> None:
        """Invalidate adapter caches affected by this artifact kind."""


class FullRuntimeArtifactCacheInvalidationPolicy(RuntimeArtifactCacheInvalidationPolicy):
    """Conservative invalidation for artifact kinds without narrower semantics."""

    def invalidate(self, adapter: CellProfilerRuntimeAdapter) -> None:
        adapter.clear_runtime_query_caches()


class ImageRuntimeArtifactCacheInvalidationPolicy(FullRuntimeArtifactCacheInvalidationPolicy):
    """Image writes may affect image reads and image-derived measurement alignment."""

    kind = ArtifactKind.IMAGE

    def invalidate(self, adapter: CellProfilerRuntimeAdapter) -> None:
        adapter._image_cache.clear()
        adapter._artifact_availability_cache.clear()


class ObjectLabelRuntimeArtifactCacheInvalidationPolicy(
    FullRuntimeArtifactCacheInvalidationPolicy
):
    """Object writes may affect label reads, label domains, and measurement alignment."""

    kind = ArtifactKind.OBJECT_LABELS

    def invalidate(self, adapter: CellProfilerRuntimeAdapter) -> None:
        adapter._object_cache.clear()
        adapter._label_domain_cache.clear()
        adapter.object_feature_value_cache().clear()
        adapter.object_label_measurement_values_cache().clear()
        adapter._measurement_cache.clear()
        adapter._artifact_availability_cache.clear()


class MeasurementRuntimeArtifactCacheInvalidationPolicy(
    RuntimeArtifactCacheInvalidationPolicy
):
    """Measurement writes invalidate measurement queries without discarding labels/images."""

    kind = ArtifactKind.MEASUREMENTS

    def invalidate(self, adapter: CellProfilerRuntimeAdapter) -> None:
        adapter.clear_measurement_query_cache()


class RelationshipRuntimeArtifactCacheInvalidationPolicy(
    RuntimeArtifactCacheInvalidationPolicy
):
    """Relationship writes are independent of image/object/measurement read caches."""

    kind = ArtifactKind.RELATIONSHIPS

    def invalidate(self, adapter: CellProfilerRuntimeAdapter) -> None:
        return None


@dataclass(frozen=True, slots=True)
class SourceBindingRequestBase(ABC):
    """Shared nominal fields for source-binding request records."""

    alias: str
    binding: NamedSourceBinding


@dataclass(frozen=True, slots=True)
class SourceBindingResolutionRequest(SourceBindingRequestBase):
    """Source-binding resolution inputs for one external image alias."""

    adapter: CellProfilerRuntimeAdapter
    current_image: Any


@dataclass(frozen=True, slots=True)
class SourceBindingMatchPlanRequest:
    """Typed request for deriving target metadata from an image-set match plan."""

    alias: str
    plan: SourceBindingMatchPlan
    step_input_candidates: tuple["ParsedSourceCandidate", ...]
    target_candidates: tuple["ParsedSourceCandidate", ...]
    full_pipeline_candidates: tuple["ParsedSourceCandidate", ...]
    source_binding_plan: CompiledSourceBindingPlan
    group_key: str | None


class SourceBindingResolver(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for resolving typed source bindings."""

    __registry_key__ = "origin_key"
    __skip_if_no_key__ = True
    origin: ClassVar[SourceBindingOrigin | None] = None
    origin_key: ClassVar[str | None] = None

    @classmethod
    @lru_cache(maxsize=None)
    def for_origin(cls, origin: SourceBindingOrigin) -> "SourceBindingResolver":
        return cls.__registry__[origin.value]()

    @abstractmethod
    def resolve_image(self, request: SourceBindingResolutionRequest) -> Any:
        """Resolve one named source image binding."""

    def require_matched_candidates(
        self,
        request: MatchedSourceCandidatesRequest,
    ) -> tuple["ParsedSourceCandidate", ...]:
        """Return matched source candidates or raise with resolver context."""

        if request.matched:
            return request.matched
        candidate_summary = _source_candidate_summary(request.candidates)
        raise RuntimeError(
            f"CellProfiler source alias '{request.alias}' with selector "
            f"{request.binding.selector!r} matched no files in the "
            f"{request.source_description} source universe. "
            f"Candidate sample: {candidate_summary!r}."
        )


class StepInputSourceBindingResolver(SourceBindingResolver):
    """Resolve named images directly from the current FunctionStep input."""

    origin = SourceBindingOrigin.STEP_INPUT
    origin_key = SourceBindingOrigin.STEP_INPUT.value

    def resolve_image(self, request: SourceBindingResolutionRequest) -> Any:
        if not request.binding.requires_selector_resolution:
            return self.natural_step_input_payload(request.current_image)
        step_input_files = request.adapter.source_binding_context.step_input_files
        if not step_input_files:
            raise NotImplementedError(
                f"CellProfiler source alias '{request.alias}' needs step-input "
                "selector resolution, but no step input file universe was "
                "provided to the runtime adapter."
            )
        parsed_candidates = request.adapter.source_candidates(step_input_files)
        current_candidates = request.adapter.source_candidates(
            request.adapter.source_binding_context.current_step_input_files
        )
        match_started_at = time.perf_counter()
        matched = SourceCandidateMatcher.match_candidates(
            candidates=parsed_candidates,
            binding=request.binding,
            inherit_components=SourceCandidateMatcher.inherited_scope_components(
                current_candidates
            ),
        )
        AdapterProfileLog.source_candidates(
            "source_candidates_match",
            time.perf_counter() - match_started_at,
            alias=request.alias,
            source=SourceBindingOrigin.STEP_INPUT,
            count=len(matched),
        )
        selected_files = self.require_matched_candidates(
            MatchedSourceCandidatesRequest.from_resolution(
                request,
                matched=matched,
                candidates=parsed_candidates,
                source_description="step input",
            )
        )
        return _select_step_input_stack(
            request=request,
            selected_paths=tuple(candidate.path for candidate in selected_files),
        )

    @staticmethod
    def natural_step_input_payload(current_image: Any) -> Any:
        if not isinstance(current_image, (RuntimeArrayPayload, np.ndarray)):
            return current_image
        if current_image.ndim == 2:
            return current_image
        return RestackLikePayloadAuthority.restack(
            _unstack_payload(current_image),
            current_image,
        )


class PipelineStartSourceBindingResolver(SourceBindingResolver):
    """Resolve named images from the original pipeline-start source universe."""

    origin = SourceBindingOrigin.PIPELINE_START
    origin_key = SourceBindingOrigin.PIPELINE_START.value

    def resolve_image(self, request: SourceBindingResolutionRequest) -> Any:
        pipeline_input_files = request.adapter.source_binding_context.pipeline_input_files
        if not pipeline_input_files:
            raise NotImplementedError(
                f"CellProfiler source alias '{request.alias}' needs pipeline-start "
                "selector resolution, but no pipeline-start file universe was "
                "provided to the runtime adapter."
            )
        step_input_candidates = request.adapter.source_candidates(
            request.adapter.source_binding_context.current_step_input_files
        )
        inherit_components = SourceCandidateMatcher.pipeline_start_inherited_components(
            request.adapter.source_binding_plan,
            step_input_candidates,
        )
        parsed_candidates = request.adapter.source_candidates(pipeline_input_files)
        match_started_at = time.perf_counter()
        initially_matched = SourceCandidateMatcher.match_candidates(
            candidates=parsed_candidates,
            binding=request.binding,
            inherit_components=inherit_components,
        )
        matched = SourceCandidateMatcher.match_image_set_candidates(
            request.alias,
            request.adapter.source_binding_plan.match_plan,
            step_input_candidates,
            initially_matched,
            parsed_candidates,
            source_binding_plan=request.adapter.source_binding_plan,
            group_key=request.adapter.group_key,
        )
        AdapterProfileLog.source_candidates(
            "source_candidates_match",
            time.perf_counter() - match_started_at,
            alias=request.alias,
            source=SourceBindingOrigin.PIPELINE_START,
            count=len(matched),
        )
        selected_files = self.require_matched_candidates(
            MatchedSourceCandidatesRequest.from_resolution(
                request,
                matched=matched,
                candidates=parsed_candidates,
                source_description="pipeline start",
            )
        )
        load_started_at = time.perf_counter()
        payload = _load_pipeline_start_stack(
            adapter=request.adapter,
            selected_paths=tuple(candidate.path for candidate in selected_files),
            current_image=request.current_image,
        )
        AdapterProfileLog.source_candidates(
            "source_candidates_load",
            time.perf_counter() - load_started_at,
            alias=request.alias,
            count=len(selected_files),
        )
        return payload


@dataclass(frozen=True, slots=True)
class OptionalResolution(Generic[T]):
    """Typed carrier for adapter lookups where absence is a valid result."""

    value: T | None

    @classmethod
    def from_optional(cls, value: T | None) -> "OptionalResolution[T]":
        return cls(value)

    def bind(
        self,
        step: Callable[[T], "OptionalResolution[U]"],
    ) -> "OptionalResolution[U]":
        if self.value is None:
            return OptionalResolution(None)
        return step(self.value)


@dataclass(frozen=True, slots=True)
class CellProfilerImageNumberCandidateContext:
    """Candidate universe for resolving one source path to a CP image number."""

    source_path: str
    candidates: tuple["ParsedSourceCandidate", ...]


@dataclass(frozen=True, slots=True)
class CellProfilerImageNumberMatchedContext:
    """Matched source candidate with its image-set candidate universe."""

    matched_candidate: "ParsedSourceCandidate"
    candidates: tuple["ParsedSourceCandidate", ...]


@dataclass(frozen=True, slots=True)
class CellProfilerImageNumberResolver:
    """Resolve source paths to CellProfiler image numbers with explicit absence flow."""

    adapter: CellProfilerRuntimeAdapter

    @classmethod
    def for_adapter(
        cls,
        adapter: CellProfilerRuntimeAdapter,
    ) -> "CellProfilerImageNumberResolver":
        return cls(adapter)

    def image_number_for_paths(self, source_paths: tuple[str, ...]) -> int | None:
        return (
            OptionalResolution.from_optional(source_paths[0] if source_paths else None)
            .bind(self.candidate_context)
            .bind(self.matched_context)
            .bind(self.image_number)
            .value
        )

    def candidate_context(
        self,
        source_path: str,
    ) -> OptionalResolution[CellProfilerImageNumberCandidateContext]:
        pipeline_paths = self.pipeline_paths()
        if not pipeline_paths:
            return OptionalResolution(None)
        return OptionalResolution(
            CellProfilerImageNumberCandidateContext(
                source_path=source_path,
                candidates=self.adapter.source_candidates(pipeline_paths),
            )
        )

    def matched_context(
        self,
        context: CellProfilerImageNumberCandidateContext,
    ) -> OptionalResolution[CellProfilerImageNumberMatchedContext]:
        matched_candidate = self.matched_source_candidate(
            context.source_path,
            context.candidates,
        )
        if matched_candidate is None:
            return OptionalResolution(None)
        return OptionalResolution(
            CellProfilerImageNumberMatchedContext(
                matched_candidate=matched_candidate,
                candidates=context.candidates,
            )
        )

    def image_number(
        self,
        context: CellProfilerImageNumberMatchedContext,
    ) -> OptionalResolution[int]:
        return OptionalResolution.from_optional(
            self.image_numbers_by_set(context.candidates).get(
                SourceImageSetIdentity.from_metadata(
                    context.matched_candidate.metadata,
                    fallback_source_path=context.matched_candidate.resolved_path,
                )
            )
        )

    def pipeline_paths(self) -> tuple[str, ...]:
        return tuple(
            path
            for path in self.adapter.source_binding_context.pipeline_input_files
            if is_image_path(path)
        )

    def matched_source_candidate(
        self,
        source_path: str,
        candidates: tuple["ParsedSourceCandidate", ...],
    ) -> "ParsedSourceCandidate | None":
        first_source_path = self.adapter.cellprofiler_source_order_path(source_path)
        return next(
            (
                candidate
                for candidate in candidates
                if first_source_path
                in {
                    self.adapter.cellprofiler_source_order_path(candidate.path),
                    self.adapter.cellprofiler_source_order_path(candidate.resolved_path),
                }
            ),
            None,
        )

    @staticmethod
    def image_numbers_by_set(
        candidates: tuple["ParsedSourceCandidate", ...],
    ) -> Mapping[SourceImageSetIdentity, int]:
        image_numbers: dict[SourceImageSetIdentity, int] = {}
        for candidate in SourceCandidateMatcher.ordered_source_candidates(candidates):
            image_set_key = SourceImageSetIdentity.from_metadata(
                candidate.metadata,
                fallback_source_path=candidate.resolved_path,
            )
            if image_set_key not in image_numbers:
                image_numbers[image_set_key] = len(image_numbers) + 1
        return MappingProxyType(image_numbers)


@dataclass(frozen=True, slots=True)
class SourceBindingPlaneCandidateContext:
    """Candidate universe for resolving an alias to a current source plane."""

    request: SourceBindingRequestBase
    axis_candidates: tuple["ParsedSourceCandidate", ...]
    step_candidates: tuple["ParsedSourceCandidate", ...]
    pipeline_candidates: tuple["ParsedSourceCandidate", ...]


@dataclass(frozen=True, slots=True)
class SourceBindingPlaneMatchedContext:
    """Matched axis indexes for one source-binding alias."""

    alias: str
    matched_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PipelineStartSourceLoadRequest:
    """Typed request for loading pipeline-start source payloads."""

    adapter: CellProfilerRuntimeAdapter
    selected_paths: tuple[str, ...]
    backend: str


class PipelineStartSourceFileLoader(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for loading selected pipeline-start source files."""

    __registry_key__ = "loader_key"
    __skip_if_no_key__ = True
    loader_key: ClassVar[str | None] = None

    @classmethod
    @lru_cache(maxsize=None)
    def for_paths(
        cls,
        selected_paths: tuple[str, ...],
    ) -> "PipelineStartSourceFileLoader":
        matching_loaders = tuple(
            loader
            for loader in (
                loader_type() for loader_type in cls.__registry__.values()
            )
            if loader.accepts_all(selected_paths)
        )
        if len(matching_loaders) == 1:
            return matching_loaders[0]
        suffixes = sorted({Path(path).suffix.lower() for path in selected_paths})
        if not matching_loaders:
            raise RuntimeError(
                "Pipeline-start source resolution has no registered loader for "
                f"selected source suffixes {suffixes!r}."
            )
        raise RuntimeError(
            "Pipeline-start source resolution has ambiguous registered loaders for "
            f"selected source suffixes {suffixes!r}."
        )

    def accepts_all(self, selected_paths: tuple[str, ...]) -> bool:
        return bool(selected_paths) and all(
            self.accepts_path(path) for path in selected_paths
        )

    @abstractmethod
    def accepts_path(self, path: str) -> bool:
        """Return whether this loader owns one source file path."""

    @abstractmethod
    def load_slices(self, request: PipelineStartSourceLoadRequest) -> list[Any]:
        """Load selected source files as stackable image-like payloads."""

    def source_payload_with_metadata(
        self,
        payload: Any,
        *,
        source_path: str,
        request: PipelineStartSourceLoadRequest,
    ) -> Any:
        """Attach loader-owned source metadata to an image-like payload."""

        metadata = image_payload_metadata(payload)
        if not metadata.has_values:
            metadata = image_payload_metadata_from_source(
                payload,
                source_path=source_path,
                read_backend=request.backend,
                filemanager=_require_processing_context(request.adapter).filemanager,
            )
        return image_payload_with_context(
            image_payload_data(payload),
            mask=image_payload_mask(payload),
            metadata=metadata,
        )


def prepare_cellprofiler_runtime_adapter() -> None:
    """Materialize nominal runtime-adapter registries before execution."""
    for origin in SourceBindingOrigin:
        SourceBindingResolver.for_origin(origin)
    for method in SourceBindingMatchMethod:
        SourceBindingMatchPlanResolver.for_method(method)
    tuple(PipelineStartSourceFileLoader.__registry__.values())


def _label_stack_repeats_first_plane(label_array: np.ndarray) -> bool:
    if label_array.ndim <= 2:
        return False
    first_plane = label_array[0]
    return all(np.array_equal(first_plane, label_array[index]) for index in range(1, label_array.shape[0]))


class RuntimeRecordStackAuthority:
    """Stack grouped runtime image/object-label records with payload semantics."""

    @classmethod
    def stack_image_records(cls, records: tuple[StoredRuntimeValue, ...]) -> Any:
        payloads = tuple(record.value.data for record in records)
        arrays = tuple(
            cls.grouped_image_array(image_payload_data(payload))
            for payload in payloads
        )
        memory_type = detect_memory_type(arrays[0])
        data = (
            stack_slices(
                list(arrays),
                memory_type,
                0,
            )
            if all(getattr(array, "ndim", None) == 2 for array in arrays)
            else np.stack(
                tuple(np.asarray(array) for array in arrays),
                axis=0,
            )
        )
        masks = tuple(image_payload_mask(payload) for payload in payloads)
        present_masks = tuple(mask for mask in masks if mask is not None)
        if present_masks and len(present_masks) != len(masks):
            raise ValueError("Cannot stack mixed masked and unmasked grouped image inputs.")
        mask = (
            None
            if not present_masks
            else stack_slices(list(present_masks), memory_type, 0)
            if all(getattr(mask, "ndim", None) == 2 for mask in present_masks)
            else np.stack(tuple(np.asarray(mask) for mask in present_masks), axis=0)
        )
        return image_payload_with_context(
            data,
            mask=mask,
            metadata=compose_image_payload_metadata(payloads),
        )

    @staticmethod
    def grouped_image_array(array: Any) -> Any:
        if (
            getattr(array, "ndim", None) == 3
            and not is_color_image_slice(array)
            and getattr(array, "shape", ())[0] == 1
        ):
            return array[0]
        return array

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
        )

    @classmethod
    def normalize_dense_object_label_payload(cls, labels: Any) -> Any:
        """Return dense object labels as one array payload, not slice lists."""
        if labels is None or isinstance(labels, SparseIJVLabelRows):
            return labels
        if not _is_sequence_payload(labels):
            return labels
        if not labels:
            return np.asarray(labels, dtype=np.int32)
        memory_type = detect_memory_type(labels[0])
        try:
            return ImageStackLayout.stack_slices_or_single_stack(
                labels,
                memory_type=memory_type,
                gpu_id=0,
            )
        except ValueError:
            return cls.stack_dense_label_sequence(labels, memory_type)

    @staticmethod
    def stack_dense_label_sequence(labels: Sequence[Any], memory_type: str) -> Any:
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


def _is_sequence_payload(labels: Any) -> bool:
    return isinstance(labels, Sequence) and not isinstance(
        labels,
        (str, bytes, bytearray, Mapping),
    )


def _single_or_none(values: Any) -> Any | None:
    unique = tuple(dict.fromkeys(values))
    if len(unique) == 1:
        return unique[0]
    return None


def _is_global_grouped_input_request(
    input_plan: ArtifactInputPlan,
    group_key: str | None,
) -> bool:
    group_keys = tuple(input_plan.group_keys or ())
    if len(group_keys) <= 1:
        return False
    paths_by_group = input_plan.paths_by_group or {}
    return group_key in (None, "default") and group_key not in paths_by_group


class SpatialGridValueAuthority:
    """Normalize, compare, and collapse grouped spatial-grid runtime values."""

    @staticmethod
    def native_value(name: str, value: Any) -> SpatialGrid:
        if isinstance(value, SpatialGrid):
            return value.with_name(name)
        if isinstance(value, Mapping):
            return SpatialGrid.from_mapping(name, value)
        raise TypeError(
            f"Spatial grid slice '{name}' must be SpatialGrid or mapping-backed, "
            f"got {type(value).__name__}."
        )

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
        slice_count = max(
            grid.slice_count if isinstance(grid, RuntimeSliceAlignedValues) else 1
            for grid in grids
        )
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
    def equivalence_payload(grid: SpatialGrid) -> dict[str, Any]:
        return {**grid.as_mapping(), "slice_index": 0}


def _runtime_query_for_input_plan(
    input_plan: ArtifactInputPlan,
    *,
    axis_id: str,
    group_key: str | None,
    backend: str,
) -> RuntimeArtifactQuery:
    if input_plan.path != "self":
        return RuntimeArtifactQuery.by_location(
            name=input_plan.name,
            kind=input_plan.kind,
            axis_id=axis_id,
            location=RuntimeArtifactLocation(
                path=_input_plan_path_for_group(input_plan, group_key),
                backend=backend,
            ),
        )
    return RuntimeArtifactQuery.by_group(
        name=input_plan.name,
        kind=input_plan.kind,
        axis_id=axis_id,
        group_key=_single_input_plan_group_key(input_plan),
    )


def _input_plan_path_for_group(
    input_plan: ArtifactInputPlan,
    group_key: str | None,
) -> str:
    paths_by_group = input_plan.paths_by_group or {}
    if group_key in paths_by_group:
        return paths_by_group[group_key]
    if None in paths_by_group:
        return paths_by_group[None]
    return input_plan.path


def _single_input_plan_group_key(input_plan: ArtifactInputPlan) -> str | None:
    group_keys = input_plan.group_keys or (None,)
    if len(group_keys) == 1:
        return group_keys[0]
    raise RuntimeError(
        f"Artifact input '{input_plan.name}' uses self-location with multiple "
        f"producer groups: {group_keys!r}."
    )


class OpenHCSImageSourceFileLoader(PipelineStartSourceFileLoader):
    """Load normal image sources through the OpenHCS VFS filemanager."""

    loader_key = "openhcs_image"

    def accepts_path(self, path: str) -> bool:
        return is_image_path(path)

    def load_slices(self, request: PipelineStartSourceLoadRequest) -> list[Any]:
        context = _require_processing_context(request.adapter)
        load_kwargs: dict[str, Any] = {}
        if request.backend == Backend.ZARR.value:
            load_kwargs["zarr_config"] = context.global_config.zarr_config
        loaded_images = context.filemanager.load_batch(
            list(request.selected_paths),
            request.backend,
            **load_kwargs,
        )
        return [
            self.source_payload_with_metadata(
                _source_payload_for_declared_image_type(
                    payload,
                    source_path=source_path,
                    request=request,
                ),
                source_path=source_path,
                request=request,
            )
            for payload, source_path in zip(loaded_images, request.selected_paths)
        ]


class MatlabMatrixSourceFileLoader(PipelineStartSourceFileLoader):
    """Load CellProfiler MATLAB matrix image sources such as illumination files."""

    loader_key = "matlab_matrix"

    def accepts_path(self, path: str) -> bool:
        return Path(path).suffix.lower() == ".mat"

    def load_slices(self, request: PipelineStartSourceLoadRequest) -> list[Any]:
        return [
            self.source_payload_with_metadata(
                self._load_matrix(path),
                source_path=path,
                request=request,
            )
            for path in request.selected_paths
        ]

    def _load_matrix(self, path: str) -> Any:
        from scipy.io import loadmat

        payloads = _matlab_numeric_arrays(loadmat(path))
        if not payloads:
            raise RuntimeError(
                f"MATLAB source file {path!r} contains no numeric image arrays."
            )
        if len(payloads) == 1:
            return payloads[0][1]
        image_payloads = tuple(
            payload for name, payload in payloads if name.strip().lower() == "image"
        )
        if len(image_payloads) == 1:
            return image_payloads[0]
        names = tuple(name for name, _payload in payloads)
        raise RuntimeError(
            f"MATLAB source file {path!r} contains multiple numeric arrays "
            f"{names!r}; expected exactly one payload or one 'Image' payload."
        )


class NumpyArraySourceFileLoader(PipelineStartSourceFileLoader):
    """Load NumPy array image sources such as saved illumination functions."""

    loader_key = "numpy_array"

    def accepts_path(self, path: str) -> bool:
        return Path(path).suffix.lower() in FileFormat.NUMPY.value

    def load_slices(self, request: PipelineStartSourceLoadRequest) -> list[Any]:
        return [
            _numpy_array_source_payload_with_metadata(
                self._load_array(path, request),
                source_path=path,
            )
            for path in request.selected_paths
        ]

    def _load_array(
        self,
        path: str,
        request: PipelineStartSourceLoadRequest,
    ) -> Any:
        payload = source_schema_auxiliary_payload(path)
        if payload is None:
            if request.backend == Backend.DISK.value:
                payload = np.load(path)
            else:
                payload = _require_processing_context(
                    request.adapter
                ).filemanager.load_batch(
                    [path],
                    request.backend,
                )[0]
        if not _is_numeric_array_payload(payload):
            raise RuntimeError(
                f"NumPy source file {path!r} does not contain a numeric image array."
            )
        return payload


def _numpy_array_source_payload_with_metadata(
    payload: Any,
    *,
    source_path: str,
) -> Any:
    """Attach array payload metadata without image-file probing."""
    metadata = image_payload_metadata(payload)
    if not metadata.has_values:
        metadata = type(metadata).for_array_payload(
            image_payload_data(payload),
            source_path=source_path,
        )
    return image_payload_with_context(
        image_payload_data(payload),
        mask=image_payload_mask(payload),
        metadata=metadata,
    )


def _source_payload_for_declared_image_type(
    payload: Any,
    *,
    source_path: str,
    request: PipelineStartSourceLoadRequest,
) -> Any:
    """Apply setup-declared source image semantics before module execution."""

    return ContextSourceMetadataAuthority.apply_loading_semantics(
        payload,
        source_path=source_path,
        request=request,
    )


@dataclass(frozen=True, slots=True)
class ParsedSourceCandidate:
    """One parsed file candidate used for source-binding selector resolution."""

    path: str
    resolved_path: str
    filename: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MatchedSourceCandidatesRequest(SourceBindingRequestBase):
    """Typed request for fail-loud source-candidate selection."""

    matched: tuple[ParsedSourceCandidate, ...]
    candidates: tuple[ParsedSourceCandidate, ...]
    source_description: str

    @classmethod
    def from_resolution(
        cls,
        request: SourceBindingResolutionRequest,
        *,
        matched: tuple[ParsedSourceCandidate, ...],
        candidates: tuple[ParsedSourceCandidate, ...],
        source_description: str,
    ) -> "MatchedSourceCandidatesRequest":
        return cls(
            alias=request.alias,
            binding=request.binding,
            matched=matched,
            candidates=candidates,
            source_description=source_description,
        )


def _parse_source_candidates(
    file_paths: tuple[str, ...],
    adapter: CellProfilerRuntimeAdapter,
) -> tuple[ParsedSourceCandidate, ...]:
    parser = _require_processing_context(adapter).microscope_handler.parser
    candidates: list[ParsedSourceCandidate] = []
    for file_path in file_paths:
        resolved_path = _resolved_source_path(file_path, adapter)
        metadata = _candidate_metadata(
            file_path,
            resolved_path,
            adapter,
            parser,
        )
        candidates.append(
            ParsedSourceCandidate(
                path=str(file_path),
                resolved_path=str(resolved_path),
                filename=Path(resolved_path).name,
                metadata=MappingProxyType(dict(metadata)),
            )
        )
    return tuple(candidates)


def _source_candidate_cache_key(
    file_paths: tuple[str, ...],
    adapter: CellProfilerRuntimeAdapter,
) -> tuple[Hashable, ...]:
    context = adapter.source_binding_context
    parser = _require_processing_context(adapter).microscope_handler.parser
    return (
        tuple(file_paths),
        context.step_input_dir,
        tuple(sorted(context.step_input_source_paths.items())),
        context.source_metadata_identity,
        context.pipeline_input_backend,
        tuple(context.pipeline_input_files),
        adapter.source_binding_plan.metadata_rules,
        type(parser).__module__,
        type(parser).__qualname__,
        repr(parser),
    )


def _candidate_metadata(
    file_path: str,
    resolved_path: str,
    adapter: CellProfilerRuntimeAdapter,
    parser: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    context = adapter.source_binding_context
    virtual_path = _candidate_virtual_workspace_path(
        file_path,
        resolved_path,
        context,
    )
    context_paths = _candidate_metadata_paths(file_path, resolved_path, virtual_path)
    if ContextSourceMetadataAuthority.has_metadata_for_any(context_paths, context):
        ContextSourceMetadataAuthority.merge_into(metadata, context_paths, context)
        _merge_candidate_path_metadata(
            metadata,
            resolved_path,
            adapter,
            parser,
            strict=False,
        )
        if not source_paths_equal(file_path, resolved_path):
            _merge_candidate_path_metadata(
                metadata,
                file_path,
                adapter,
                parser,
                strict=False,
            )
        if virtual_path is not None and virtual_path not in {file_path, resolved_path}:
            _merge_candidate_path_metadata(
                metadata,
                virtual_path,
                adapter,
                parser,
                strict=False,
            )
        return metadata

    _merge_candidate_path_metadata(
        metadata,
        resolved_path,
        adapter,
        parser,
        strict=True,
    )
    if not source_paths_equal(file_path, resolved_path):
        _merge_candidate_path_metadata(
            metadata,
            file_path,
            adapter,
            parser,
            strict=SourceRuntimePathLookup(
                file_path,
                context.step_input_dir,
            ).first_value(context.step_input_source_paths) is None,
        )
    if virtual_path is not None and virtual_path not in {file_path, resolved_path}:
        _merge_candidate_path_metadata(
            metadata,
            virtual_path,
            adapter,
            parser,
            strict=False,
        )
    ContextSourceMetadataAuthority.merge_into(
        metadata,
        _candidate_metadata_paths(file_path, resolved_path, virtual_path),
        context,
    )
    return metadata


def _candidate_metadata_paths(
    file_path: str,
    resolved_path: str,
    virtual_path: str | None,
) -> tuple[str, ...]:
    paths = (file_path, resolved_path) if virtual_path is None else (
        file_path,
        resolved_path,
        virtual_path,
    )
    return tuple(dict.fromkeys(paths))


class ContextSourceMetadataAuthority:
    """Lookup authority for runtime source metadata attached to source paths."""

    @classmethod
    def apply_loading_semantics(
        cls,
        payload: Any,
        *,
        source_path: str,
        request: PipelineStartSourceLoadRequest,
    ) -> Any:
        source_metadata = SourceRuntimePathLookup(
            source_path,
            request.adapter.source_binding_context.step_input_dir,
        ).first_value(
            request.adapter.source_binding_context.source_metadata_by_path,
            include_native_path_fallback=True,
        )
        return apply_source_image_loading_semantics(
            payload,
            source_metadata=source_metadata,
            source_path=source_path,
            read_backend=request.backend,
            filemanager=_require_processing_context(request.adapter).filemanager,
        )

    @classmethod
    def has_metadata_for_any(
        cls,
        paths: tuple[str, ...],
        context: SourceBindingRuntimeContext,
    ) -> bool:
        return any(
            SourceRuntimePathLookup(path, context.step_input_dir).first_value(
                context.source_metadata_by_path,
                include_native_path_fallback=True,
            )
            is not None
            for path in paths
        )

    @classmethod
    def merge_into(
        cls,
        metadata: dict[str, Any],
        paths: tuple[str, ...],
        context: SourceBindingRuntimeContext,
    ) -> None:
        for path in dict.fromkeys(paths):
            context_metadata = SourceRuntimePathLookup(
                path,
                context.step_input_dir,
            ).first_value(
                context.source_metadata_by_path,
                include_native_path_fallback=True,
            )
            if context_metadata is not None:
                merge_source_metadata(metadata, context_metadata, path=path)


def _merge_candidate_path_metadata(
    metadata: dict[str, Any],
    metadata_path: str,
    adapter: CellProfilerRuntimeAdapter,
    parser: Any,
    *,
    strict: bool,
) -> None:
    parsed_metadata = parser.parse_filename(Path(metadata_path).name) or {}
    extracted_metadata = metadata_from_rules(
        metadata_path,
        adapter.source_binding_plan.metadata_rules,
    )
    if strict:
        merge_source_metadata(metadata, parsed_metadata, path=metadata_path)
        merge_source_metadata(metadata, extracted_metadata, path=metadata_path)
        return
    _merge_missing_source_metadata(metadata, parsed_metadata)
    _merge_missing_source_metadata(metadata, extracted_metadata)


def _merge_missing_source_metadata(
    metadata: dict[str, Any],
    additions: Mapping[str, Any],
) -> None:
    for key, value in additions.items():
        metadata.setdefault(key, str(value))


def _candidate_virtual_workspace_path(
    file_path: str,
    resolved_path: str,
    context: SourceBindingRuntimeContext,
) -> str | None:
    for key in SourceRuntimePathLookup(file_path, context.step_input_dir).keys():
        if key in context.step_input_source_paths:
            return key
    return _virtual_workspace_path_for_source(resolved_path, context)


def _virtual_workspace_path_for_source(
    resolved_path: str,
    context: SourceBindingRuntimeContext,
) -> str | None:
    resolved_key = source_path_identity_key(resolved_path)
    for virtual_path, source_path in context.step_input_source_paths.items():
        if source_path_identity_key(source_path) == resolved_key:
            return virtual_path
    return None


class SourceCandidateMatcher:
    """Nominal owner for source-binding candidate selection semantics."""

    @classmethod
    def match_candidates(
        cls,
        *,
        candidates: tuple[ParsedSourceCandidate, ...],
        binding: NamedSourceBinding,
        inherit_components: Mapping[str, str],
    ) -> tuple[ParsedSourceCandidate, ...]:
        cls.validate_metadata_selectors(candidates, binding)
        component_selectors = {
            selector.component.value: selector.value
            for selector in binding.selector.components
        }
        explicit_metadata_fields = {
            selector.field for selector in binding.selector.metadata
        }
        effective_components = (
            {
                **{
                    name: value
                    for name, value in inherit_components.items()
                    if name not in component_selectors
                    and name not in explicit_metadata_fields
                },
                **component_selectors,
            }
            if binding.selector.inherit_current_scope
            else component_selectors
        )
        inherited_component_items = tuple(
            (name, value)
            for name, value in effective_components.items()
            if name not in component_selectors
        )

        return tuple(
            candidate
            for candidate in candidates
            if cls.matches_explicit_components(candidate, component_selectors)
            and cls.matches_inherited_scope(
                candidate,
                inherited_component_items,
            )
            and cls.matches_metadata(candidate, binding.selector.metadata)
            and source_filters_match(candidate.resolved_path, binding.selector.filters)
        )

    @staticmethod
    def validate_metadata_selectors(
        candidates: tuple[ParsedSourceCandidate, ...],
        binding: NamedSourceBinding,
    ) -> None:
        metadata_fields = {selector.field for selector in binding.selector.metadata}
        if not metadata_fields:
            return
        unsupported = tuple(
            field
            for field in sorted(metadata_fields)
            if not any(
                source_metadata_value(candidate.metadata, field) is not None
                for candidate in candidates
            )
        )
        if unsupported:
            raise NotImplementedError(
                "Source-binding metadata selectors are only supported when the "
                "native OpenHCS filename parser exposes those fields. Missing "
                f"fields: {list(unsupported)}."
            )

    @classmethod
    def matches_explicit_components(
        cls,
        candidate: ParsedSourceCandidate,
        expected_components: Mapping[str, str],
    ) -> bool:
        return all(
            cls.matches_explicit_component(candidate, component_name, value)
            for component_name, value in expected_components.items()
        )

    @staticmethod
    def matches_explicit_component(
        candidate: ParsedSourceCandidate,
        component_name: str,
        expected_value: str,
    ) -> bool:
        component = source_metadata_component(component_name)
        if component is None:
            metadata_value = source_metadata_value(candidate.metadata, component_name)
            return metadata_value is not None and source_metadata_values_equal(
                metadata_value,
                expected_value,
            )
        return any(
            source_metadata_values_equal(metadata_value, expected_value)
            for metadata_value in source_component_metadata_values(
                candidate.metadata,
                component,
            )
        )

    @staticmethod
    def matches_inherited_scope(
        candidate: ParsedSourceCandidate,
        inherited_scope: tuple[tuple[str, str], ...],
    ) -> bool:
        return all(
            (
                metadata_value := semantic_source_metadata_value(
                    candidate.metadata,
                    field_name,
                )
            )
            is None
            or source_metadata_values_equal(metadata_value, value)
            for field_name, value in inherited_scope
        )

    @staticmethod
    def matches_metadata(
        candidate: ParsedSourceCandidate,
        metadata_selectors: tuple[Any, ...],
    ) -> bool:
        return all(
            (metadata_value := source_metadata_value(candidate.metadata, selector.field))
            is not None
            and source_metadata_values_equal(metadata_value, selector.value)
            for selector in metadata_selectors
        )

    @staticmethod
    def matches_image_set_metadata(
        candidate: ParsedSourceCandidate,
        image_set_metadata: Mapping[str, str],
    ) -> bool:
        return all(
            (
                metadata_value := semantic_source_metadata_value(
                    candidate.metadata,
                    field_name,
                )
            )
            is not None
            and source_metadata_values_equal(metadata_value, value)
            for field_name, value in image_set_metadata.items()
        )

    @staticmethod
    def ordered_source_candidates(
        candidates: tuple[ParsedSourceCandidate, ...],
    ) -> tuple[ParsedSourceCandidate, ...]:
        return tuple(sorted(candidates, key=lambda candidate: candidate.resolved_path))

    @classmethod
    def inherited_scope_components(
        cls,
        candidates: tuple[ParsedSourceCandidate, ...],
    ) -> Mapping[str, str]:
        if not candidates:
            return {}
        shared: dict[str, str] = {}
        first_metadata = candidates[0].metadata
        for field_name, value in first_metadata.items():
            if value is None:
                continue
            normalized_value = str(value)
            if all(
                (
                    candidate_value := semantic_source_metadata_value(
                        candidate.metadata,
                        field_name,
                    )
                )
                is not None
                and source_metadata_values_equal(candidate_value, normalized_value)
                for candidate in candidates[1:]
            ):
                shared[field_name] = normalized_value
        return MappingProxyType(shared)

    @classmethod
    def pipeline_start_inherited_components(
        cls,
        source_binding_plan: CompiledSourceBindingPlan,
        step_input_candidates: tuple[ParsedSourceCandidate, ...],
    ) -> Mapping[str, str]:
        if source_binding_plan.match_plan is not None:
            return MappingProxyType({})
        return cls.inherited_scope_components(step_input_candidates)

    @classmethod
    def match_image_set_candidates(
        cls,
        alias: str,
        match_plan: SourceBindingMatchPlan | None,
        step_input_candidates: tuple[ParsedSourceCandidate, ...],
        target_candidates: tuple[ParsedSourceCandidate, ...],
        full_pipeline_candidates: tuple[ParsedSourceCandidate, ...],
        *,
        source_binding_plan: CompiledSourceBindingPlan,
        group_key: str | None,
    ) -> tuple[ParsedSourceCandidate, ...]:
        if match_plan is None or not step_input_candidates or not target_candidates:
            return target_candidates
        return SourceBindingMatchPlanResolver.for_method(
            match_plan.method
        ).match_candidates(
            SourceBindingMatchPlanRequest(
                alias=alias,
                plan=match_plan,
                step_input_candidates=step_input_candidates,
                target_candidates=target_candidates,
                full_pipeline_candidates=full_pipeline_candidates,
                source_binding_plan=source_binding_plan,
                group_key=group_key,
            )
        )

    @classmethod
    def axis_scoped_candidates(
        cls,
        candidates: tuple[ParsedSourceCandidate, ...],
        *,
        axis_id: str,
        step_input_candidates: tuple[ParsedSourceCandidate, ...],
    ) -> tuple[ParsedSourceCandidate, ...]:
        axis_scope = cls.axis_scope_components(
            axis_id=axis_id,
            step_input_candidates=step_input_candidates,
        )
        if not axis_scope:
            return candidates
        axis_scope_items = tuple(axis_scope.items())
        return tuple(
            candidate
            for candidate in candidates
            if cls.matches_inherited_scope(candidate, axis_scope_items)
        )

    @staticmethod
    def axis_scope_components(
        *,
        axis_id: str,
        step_input_candidates: tuple[ParsedSourceCandidate, ...],
    ) -> Mapping[str, str]:
        constraints: dict[str, str] = {}
        for candidate in step_input_candidates:
            for field_name, value in candidate.metadata.items():
                if value is None:
                    continue
                normalized_value = str(value)
                if not source_metadata_values_equal(normalized_value, axis_id):
                    continue
                existing = constraints.get(field_name)
                if existing is not None and existing != normalized_value:
                    raise RuntimeError(
                        f"Conflicting axis scope values for field {field_name!r}: "
                        f"{existing!r} != {normalized_value!r}."
                    )
                constraints[field_name] = normalized_value
        return MappingProxyType(constraints)

    @classmethod
    def target_candidates_in_current_scope(
        cls,
        step_input_candidates: tuple[ParsedSourceCandidate, ...],
        target_candidates: tuple[ParsedSourceCandidate, ...],
    ) -> tuple[ParsedSourceCandidate, ...]:
        current_scope = cls.inherited_scope_components(step_input_candidates)
        if not current_scope:
            return ()
        current_scope_items = tuple(current_scope.items())
        return tuple(
            candidate
            for candidate in target_candidates
            if cls.matches_inherited_scope(candidate, current_scope_items)
        )

    @classmethod
    def order_match_indexes(
        cls,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[int, ...]:
        indexes = {
            index
            for candidate in request.step_input_candidates
            for index in (cls.source_alias_order_index(candidate=candidate, request=request),)
            if index is not None
        }
        return tuple(sorted(indexes))

    @classmethod
    def source_alias_order_index(
        cls,
        *,
        candidate: ParsedSourceCandidate,
        request: SourceBindingMatchPlanRequest,
    ) -> int | None:
        matched_indexes: set[int] = set()
        for binding in request.source_binding_plan.bindings_for_group(request.group_key):
            if binding.alias == request.alias:
                continue
            for index, ordered_candidate in enumerate(
                cls.ordered_binding_candidates(
                    binding=binding,
                    candidates=request.full_pipeline_candidates,
                )
            ):
                if ordered_candidate.resolved_path == candidate.resolved_path:
                    matched_indexes.add(index)
                    break
        if not matched_indexes:
            return None
        if len(matched_indexes) != 1:
            raise RuntimeError(
                f"Order-based image-set matching could not uniquely assign source file "
                f"{candidate.resolved_path!r} to one alias order index."
            )
        return next(iter(matched_indexes))

    @classmethod
    def ordered_binding_candidates(
        cls,
        *,
        binding: NamedSourceBinding,
        candidates: tuple[ParsedSourceCandidate, ...],
    ) -> tuple[ParsedSourceCandidate, ...]:
        return cls.ordered_source_candidates(
            cls.match_candidates(
                candidates=candidates,
                binding=binding,
                inherit_components={},
            )
        )

    @classmethod
    def dimension_match_value(
        cls,
        *,
        dimension: SourceBindingMatchDimension,
        request: SourceBindingMatchPlanRequest,
    ) -> str | None:
        target_alias = request.alias
        candidate_values = {
            value
            for field in dimension.fields
            if field.alias != target_alias
            for value in cls.source_match_field_values(field, request)
        }
        if not candidate_values:
            return None
        if len(candidate_values) > 1:
            raise RuntimeError(
                "Current step input candidates produce conflicting image-set match "
                f"values for alias {target_alias!r}: {sorted(candidate_values)!r}."
            )
        return next(iter(candidate_values))

    @classmethod
    def source_match_field_values(
        cls,
        field: SourceBindingMatchField,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[str, ...]:
        try:
            return cls.shared_candidate_values(field, request.step_input_candidates)
        except RuntimeError:
            alias_candidates = cls.alias_scoped_step_input_candidates(field.alias, request)
            if not alias_candidates:
                raise
            return cls.shared_candidate_values(field, alias_candidates)

    @classmethod
    def alias_scoped_step_input_candidates(
        cls,
        alias: str,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[ParsedSourceCandidate, ...]:
        binding = request.source_binding_plan.binding_for_alias(alias, request.group_key)
        if binding is None:
            return ()
        matched = cls.match_candidates(
            candidates=request.step_input_candidates,
            binding=binding,
            inherit_components={},
        )
        if matched == request.step_input_candidates:
            return ()
        return matched

    @staticmethod
    def shared_candidate_values(
        field: SourceBindingMatchField,
        step_input_candidates: tuple[ParsedSourceCandidate, ...],
    ) -> tuple[str, ...]:
        values = tuple(
            metadata_value
            for candidate in step_input_candidates
            for metadata_value in (
                source_metadata_value(candidate.metadata, field.metadata_field),
            )
            if metadata_value is not None
        )
        if not values:
            return ()
        shared_values = set(values)
        if len(shared_values) != 1:
            raise RuntimeError(
                "Current step input candidates do not share a single image-set match "
                f"value for metadata field {field.metadata_field!r}: "
                f"{sorted(shared_values)!r}."
            )
        return (values[0],)


def _source_candidate_summary(
    candidates: tuple[ParsedSourceCandidate, ...],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "path": candidate.path,
            "metadata": dict(candidate.metadata),
        }
        for candidate in candidates[:5]
    )


def _select_step_input_stack(
    *,
    request: SourceBindingResolutionRequest,
    selected_paths: tuple[str, ...],
) -> Any:
    context = request.adapter.source_binding_context
    current_step_input_files = context.current_step_input_files
    indexed_paths = {
        path: index for index, path in enumerate(current_step_input_files)
    }
    selected_indexes = tuple(
        indexed_paths[path]
        for path in current_step_input_files
        if path in selected_paths
    )
    current_image = request.current_image
    if len(selected_indexes) != len(selected_paths):
        return _load_step_input_stack(
            request=request,
            selected_paths=selected_paths,
        )
    if not selected_indexes:
        raise RuntimeError(
            f"CellProfiler source alias '{request.alias}' selected no step-input "
            "stack indexes after filename matching."
        )
    if len(current_step_input_files) == 1:
            return StepInputSourceBindingResolver.natural_step_input_payload(current_image)
    slices = _unstack_payload(current_image)
    selected_slices = [slices[index] for index in selected_indexes]
    return RestackLikePayloadAuthority.restack(selected_slices, current_image)


def _load_step_input_stack(
    *,
    request: SourceBindingResolutionRequest,
    selected_paths: tuple[str, ...],
) -> Any:
    context = request.adapter.source_binding_context
    if context.step_input_dir is None or context.step_input_backend is None:
        raise RuntimeError(
            "Step-input selector resolution needs step_input_dir and "
            "step_input_backend when selected files are outside the current stack."
        )
    full_paths = tuple(str(Path(context.step_input_dir) / path) for path in selected_paths)
    processing_context = _require_processing_context(request.adapter)
    loaded = processing_context.filemanager.load_batch(
        list(full_paths),
        context.step_input_backend,
    )
    if not loaded:
        raise RuntimeError(
            f"Step-input source resolution loaded no payloads from {list(full_paths)}."
        )
    return RestackLikePayloadAuthority.restack(list(loaded), request.current_image)


def _load_pipeline_start_stack(
    *,
    adapter: CellProfilerRuntimeAdapter,
    selected_paths: tuple[str, ...],
    current_image: Any,
) -> Any:
    if not selected_paths:
        raise RuntimeError("Pipeline-start source selection cannot load zero paths.")
    backend = adapter.source_binding_context.pipeline_input_backend
    if backend is None:
        raise RuntimeError(
            "Pipeline-start source resolution requires pipeline_input_backend."
        )
    cache_key = _pipeline_start_payload_cache_key(adapter, backend, selected_paths)
    loaded_payloads = adapter._pipeline_start_payload_cache.get(cache_key)
    if loaded_payloads is None:
        loaded_payloads = _PIPELINE_START_PAYLOAD_PROCESS_CACHE.get(cache_key)
        if loaded_payloads is not None:
            _PIPELINE_START_PAYLOAD_PROCESS_CACHE.move_to_end(cache_key)
            adapter._pipeline_start_payload_cache[cache_key] = loaded_payloads
    if loaded_payloads is None:
        loaded_payloads = tuple(
            PipelineStartSourceFileLoader.for_paths(selected_paths).load_slices(
                PipelineStartSourceLoadRequest(
                    adapter=adapter,
                    selected_paths=selected_paths,
                    backend=backend,
                )
            )
        )
        adapter._pipeline_start_payload_cache[cache_key] = loaded_payloads
        _PIPELINE_START_PAYLOAD_PROCESS_CACHE[cache_key] = loaded_payloads
        _PIPELINE_START_PAYLOAD_PROCESS_CACHE.move_to_end(cache_key)
        if len(_PIPELINE_START_PAYLOAD_PROCESS_CACHE) > _PIPELINE_START_PAYLOAD_CACHE_LIMIT:
            _PIPELINE_START_PAYLOAD_PROCESS_CACHE.popitem(last=False)
    if not loaded_payloads:
        raise RuntimeError(
            "Pipeline-start source resolution loaded no payloads from "
            f"{list(selected_paths)}."
        )
    return RestackLikePayloadAuthority.restack(list(loaded_payloads), current_image)


def _pipeline_start_payload_cache_key(
    adapter: CellProfilerRuntimeAdapter,
    backend: str,
    selected_paths: tuple[str, ...],
) -> tuple[Hashable, ...]:
    context = adapter.source_binding_context
    return (
        backend,
        _require_processing_context(adapter).filemanager,
        selected_paths,
        context.metadata_identity_for_paths(selected_paths),
    )


def _matlab_numeric_arrays(
    mat_payload: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (name, payload)
        for name, payload in mat_payload.items()
        if not name.startswith("__") and _is_numeric_array_payload(payload)
    )


def _is_numeric_array_payload(payload: Any) -> bool:
    if not isinstance(payload, (RuntimeArrayPayload, np.ndarray)):
        return False
    return payload.dtype.kind in {"b", "u", "i", "f", "c"} and payload.ndim >= 2


def _unstack_payload(payload: Any) -> list[Any]:
    return list(payload_slices_for_alignment(payload))


class RestackLikePayloadAuthority:
    """Restack selected image payload slices while preserving payload context."""

    @classmethod
    def restack(
        cls,
        slices: list[Any],
        reference_payload: Any,
    ) -> Any:
        if not slices:
            raise ValueError("Cannot restack an empty slice list.")
        if len(slices) == 1:
            return slices[0]
        slice_data = tuple(image_payload_data(slice_payload) for slice_payload in slices)
        memory_type = detect_memory_type(image_payload_data(reference_payload))
        stacked = ImageStackLayout.stack_slices_or_single_stack(
            slice_data,
            memory_type=memory_type,
            gpu_id=0,
        )
        return image_payload_with_context(
            stacked,
            mask=cls.stack_masks(slices, memory_type),
            metadata=compose_image_payload_metadata(slices),
        )

    @staticmethod
    def stack_masks(
        slices: list[Any],
        memory_type: str,
    ) -> Any | None:
        masks = tuple(image_payload_mask(slice_payload) for slice_payload in slices)
        if not any(mask is not None for mask in masks):
            return None
        slice_data = tuple(image_payload_data(slice_payload) for slice_payload in slices)
        resolved_masks = [
            np.ones(np.asarray(data).shape[:2], dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
            for data, mask in zip(slice_data, masks)
        ]
        return ImageStackLayout.stack_slices_or_single_stack(
            resolved_masks,
            memory_type=memory_type,
            gpu_id=0,
        )


class SourceBindingMatchPlanResolver(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for restricting target candidates to the current image set."""

    __registry_key__ = "method_key"
    __skip_if_no_key__ = True
    method: ClassVar[SourceBindingMatchMethod | None] = None
    method_key: ClassVar[str | None] = None

    @classmethod
    @lru_cache(maxsize=None)
    def for_method(
        cls,
        method: SourceBindingMatchMethod,
    ) -> "SourceBindingMatchPlanResolver":
        return cls.__registry__[method.value]()

    @abstractmethod
    def match_candidates(
        self,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[ParsedSourceCandidate, ...]:
        """Return target candidates belonging to the current image set."""


class MetadataSourceBindingMatchPlanResolver(SourceBindingMatchPlanResolver):
    method = SourceBindingMatchMethod.METADATA
    method_key = SourceBindingMatchMethod.METADATA.value

    def match_candidates(
        self,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[ParsedSourceCandidate, ...]:
        constraints: dict[str, str] = {}
        for dimension in request.plan.dimensions:
            target_field = dimension.field_for_alias(request.alias)
            if target_field is None:
                continue
            match_value = SourceCandidateMatcher.dimension_match_value(
                dimension=dimension,
                request=request,
            )
            if match_value is None:
                continue
            existing = constraints.get(target_field)
            if existing is not None and existing != match_value:
                raise RuntimeError(
                    f"Conflicting image-set match values for alias {request.alias!r} "
                    f"field {target_field!r}: {existing!r} != {match_value!r}."
                )
            constraints[target_field] = match_value
        metadata_constraints = MappingProxyType(constraints)
        return tuple(
            candidate
            for candidate in request.target_candidates
            if SourceCandidateMatcher.matches_image_set_metadata(
                candidate,
                metadata_constraints,
            )
        )


class OrderSourceBindingMatchPlanResolver(SourceBindingMatchPlanResolver):
    method = SourceBindingMatchMethod.ORDER
    method_key = SourceBindingMatchMethod.ORDER.value

    def match_candidates(
        self,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[ParsedSourceCandidate, ...]:
        current_indexes = SourceCandidateMatcher.order_match_indexes(request)
        if not current_indexes:
            scoped_candidates = SourceCandidateMatcher.target_candidates_in_current_scope(
                request.step_input_candidates,
                request.target_candidates,
            )
            return scoped_candidates or request.target_candidates
        ordered_target_candidates = SourceCandidateMatcher.ordered_source_candidates(
            request.target_candidates
        )
        return tuple(
            ordered_target_candidates[index]
            for index in current_indexes
            if index < len(ordered_target_candidates)
        )


def _require_processing_context(adapter: CellProfilerRuntimeAdapter) -> Any:
    if adapter.processing_context is None:
        raise RuntimeError(
            "CellProfilerRuntimeAdapter.processing_context is required for "
            "selector-bearing source resolution."
        )
    return adapter.processing_context


def _resolved_source_path(
    file_path: str,
    adapter: CellProfilerRuntimeAdapter,
) -> str:
    source_path = SourceRuntimePathLookup(
        file_path,
        adapter.source_binding_context.step_input_dir,
    ).first_value(adapter.source_binding_context.step_input_source_paths)
    if source_path is not None:
        return source_path
    path = Path(file_path)
    if path.is_absolute():
        return str(path)
    step_input_dir = adapter.source_binding_context.step_input_dir
    if step_input_dir is None:
        return str(path)
    return str(Path(step_input_dir) / path)
