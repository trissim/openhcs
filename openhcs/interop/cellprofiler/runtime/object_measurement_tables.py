"""Object-measurement table indexes and cache mutation policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from inspect import isabstract
from types import MappingProxyType
from typing import ClassVar
from weakref import WeakKeyDictionary

from metaclass_registry import AutoRegisterMeta

from openhcs.core.measurement_row_materialization import measurement_table_object_id_field
from openhcs.core.runtime_artifact_queries import (
    MeasurementTableUnion,
    runtime_measurement_tables_for_object,
)
from openhcs.core.measurement_feature_queries import (
    MeasurementFeatureQuery,
    MeasurementTableObjectFeatureSemantics,
)
from openhcs.core.measurement_lookup_dialect import (
    RuntimeMeasurementLookupDialectLike,
    resolve_runtime_measurement_lookup_dialect,
)
from openhcs.core.runtime_semantics import (
    RuntimeObjectLabelMeasurementQuery,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_slice_projection import (
    MeasurementTableRepeatedScalarGroupKey,
    RuntimeSliceProjection,
)
from openhcs.core.runtime_values import MeasurementTable
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.runtime.adapter_scope import (
    ObjectMeasurementTableCacheKey,
    ObjectMeasurementTableIndexCacheKey,
    RuntimeGroupMatchScope,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerMeasurementVector,
)
from openhcs.interop.cellprofiler.runtime.source_identity import (
    CellProfilerCurrentImage,
)

MeasurementTableSelection = tuple[MeasurementTable, ...] | None
CellProfilerMeasurementSliceValues = tuple[CellProfilerMeasurementVector, ...]
CellProfilerMeasurementCacheValue = (
    tuple[MeasurementTable, ...]
    | CellProfilerMeasurementVector
    | CellProfilerMeasurementSliceValues
)
MeasurementTablesByObject = Mapping[str, tuple[MeasurementTable, ...]]
MutableMeasurementTablesByObject = dict[str, tuple[MeasurementTable, ...]]


@dataclass(frozen=True, slots=True)
class ObjectMeasurementTableIndex:
    """Nominal object-subject measurement table index."""

    tables: tuple[MeasurementTable, ...]
    tables_by_object: MeasurementTablesByObject
    projected_tables_by_object: MeasurementTablesByObject = field(
        default_factory=lambda: MappingProxyType({})
    )
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
        logical_tables_by_name: dict[str, MeasurementTable] = {}
        for name in dict.fromkeys(table.name for table in tables):
            name_tables = tuple(table for table in tables if table.name == name)
            name_table_semantics = tuple(
                MeasurementTableObjectFeatureSemantics.from_table(table)
                for table in name_tables
            )
            object_names = {
                object_name
                for semantics in name_table_semantics
                for object_name in semantics.object_names
            }
            has_image_level_rows = any(
                not semantics.object_names for semantics in name_table_semantics
            )
            if len(object_names) > 1 or (object_names and has_image_level_rows):
                logical_tables_by_name[name] = MeasurementTableUnion(
                    name,
                    name_tables,
                ).as_table()
        for table in tables:
            table_semantics = MeasurementTableObjectFeatureSemantics.from_table(table)
            group_key = RuntimeSliceProjection.measurement_table_repeated_scalar_group_key(
                table
            )
            if group_key not in group_sizes:
                group_sizes[group_key] = 0
            group_sizes[group_key] += 1
            for table_object_name in table_semantics.object_names:
                if table_object_name not in table_lists:
                    table_lists[table_object_name] = []
                table_lists[table_object_name].append((table, table_semantics))
        indexed_tables: MutableMeasurementTablesByObject = {}
        projected_tables_by_object: MutableMeasurementTablesByObject = {}
        feature_names_by_table: dict[int, frozenset[str]] = {}
        for object_name, object_table_entries in table_lists.items():
            object_tables_list: list[MeasurementTable] = []
            emitted_logical_names: set[str] = set()
            for table, _semantics in object_table_entries:
                logical_table = logical_tables_by_name.get(table.name)
                if logical_table is None:
                    object_tables_list.append(table)
                    continue
                if table.name in emitted_logical_names:
                    continue
                object_tables_list.append(logical_table)
                emitted_logical_names.add(table.name)
            object_tables = tuple(object_tables_list)
            object_specific_tables = tuple(
                table for table, _semantics in object_table_entries
            )
            projected_tables = (
                RuntimeSliceProjection.measurement_tables_with_repeated_scalar_slice_offsets(
                    object_specific_tables
                )
            )
            if not emitted_logical_names:
                object_tables = projected_tables
            indexed_tables[object_name] = object_tables
            projected_tables_by_object[object_name] = projected_tables
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
            projected_tables_by_object=MappingProxyType(projected_tables_by_object),
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
        if object_name not in self.tables_by_object:
            return ()
        return self.tables_by_object[object_name]

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
        if query_object_name is None:
            tables = self.tables
        elif not self.complete:
            tables = None
        elif query_object_name in self.projected_tables_by_object:
            tables = self.projected_tables_by_object[query_object_name]
        else:
            tables = ()
        if tables is None:
            return None
        if not tables and query_object_name is not None:
            tables = self.unnamed_object_feature_tables()
        query = MeasurementFeatureQuery(feature_name, dialect=dialect)
        return tuple(
            table
            for table in tables
            if self._table_may_carry_feature(table, query)
        )

    def unnamed_object_feature_tables(self) -> tuple[MeasurementTable, ...]:
        """Return object-id tables that do not declare a specific object name."""
        return tuple(
            table
            for table in self.tables
            if measurement_table_object_id_field(table) is not None
            and not MeasurementTableObjectFeatureSemantics.from_table(table).object_names
        )

    def _table_may_carry_feature(
        self,
        table: MeasurementTable,
        query: MeasurementFeatureQuery,
    ) -> bool:
        feature_names = self.feature_names_by_table.get(id(table))
        semantics = None
        if feature_names is not None:
            semantics = MeasurementTableObjectFeatureSemantics(
                object_names=(),
                feature_names=feature_names,
            )
        return query.table_may_carry_feature(table, semantics)


ObjectMeasurementTableProcessCache = dict[
    ObjectMeasurementTableCacheKey,
    tuple[MeasurementTable, ...],
]
ObjectMeasurementTableIndexProcessCache = dict[
    ObjectMeasurementTableIndexCacheKey,
    ObjectMeasurementTableIndex,
]

_OBJECT_MEASUREMENT_TABLE_PROCESS_CACHE: WeakKeyDictionary[
    RuntimeValueStore,
    ObjectMeasurementTableProcessCache,
] = WeakKeyDictionary()
_OBJECT_MEASUREMENT_TABLE_INDEX_PROCESS_CACHE: WeakKeyDictionary[
    RuntimeValueStore,
    ObjectMeasurementTableIndexProcessCache,
] = WeakKeyDictionary()


def object_measurement_table_cache(
    store: RuntimeValueStore,
) -> ObjectMeasurementTableProcessCache:
    """Return the process cache for object-subject measurement table queries."""
    cache = _OBJECT_MEASUREMENT_TABLE_PROCESS_CACHE.get(store)
    if cache is None:
        cache = {}
        _OBJECT_MEASUREMENT_TABLE_PROCESS_CACHE[store] = cache
    return cache


def object_measurement_table_index_cache(
    store: RuntimeValueStore,
) -> ObjectMeasurementTableIndexProcessCache:
    """Return the process cache for object-subject measurement table indexes."""
    cache = _OBJECT_MEASUREMENT_TABLE_INDEX_PROCESS_CACHE.get(store)
    if cache is None:
        cache = {}
        _OBJECT_MEASUREMENT_TABLE_INDEX_PROCESS_CACHE[store] = cache
    return cache


def object_measurement_tables_for_object(
    adapter: "CellProfilerRuntimeAdapter",
    object_name: str,
    *,
    group_key: str | None = None,
    match_group: bool = True,
    current_image: CellProfilerCurrentImage | None = None,
) -> tuple[MeasurementTable, ...]:
    """Return prior measurement tables whose subject is an object set."""
    runtime_scope = RuntimeGroupMatchScope(
        group_key=group_key,
        match_group=match_group,
    ).runtime_scope(adapter, current_image=current_image)
    cache_key = ObjectMeasurementTableCacheKey(
        group_key=runtime_scope.group_cache_component,
        match_group=match_group,
        object_name=object_name,
        source_scope=runtime_scope.source_identity_cache_scope,
    )
    cached = adapter._measurement_cache.get(cache_key)
    if cached is not None:
        return cached
    object_table_cache = object_measurement_table_cache(adapter.runtime_value_store)
    cached = object_table_cache.get(cache_key)
    if cached is not None:
        adapter._measurement_cache[cache_key] = cached
        return cached
    object_table_index = object_measurement_table_index(
        adapter,
        group_key=group_key,
        match_group=match_group,
        current_image=current_image,
    )
    tables = object_table_index.for_object(object_name)
    if tables is None:
        tables = runtime_measurement_tables_for_object(
            runtime_scope.artifact_query_context(),
            object_name,
        )
        tables = RuntimeSliceProjection.measurement_tables_with_repeated_scalar_slice_offsets(
            tables
        )
    object_table_cache[cache_key] = tables
    adapter._measurement_cache[cache_key] = tables
    return tables


def object_measurement_table_index(
    adapter: "CellProfilerRuntimeAdapter",
    *,
    group_key: str | None = None,
    match_group: bool = True,
    current_image: CellProfilerCurrentImage | None = None,
) -> ObjectMeasurementTableIndex:
    """Return the cached object-subject measurement table index."""
    runtime_scope = RuntimeGroupMatchScope(
        group_key=group_key,
        match_group=match_group,
    ).runtime_scope(adapter, current_image=current_image)
    index_cache_key = ObjectMeasurementTableIndexCacheKey(
        group_key=runtime_scope.group_cache_component,
        match_group=match_group,
        source_scope=runtime_scope.source_identity_cache_scope,
    )
    object_table_index_cache = object_measurement_table_index_cache(
        adapter.runtime_value_store
    )
    object_table_index = object_table_index_cache.get(index_cache_key)
    if object_table_index is not None:
        return object_table_index
    source_tables = adapter.measurement_tables(
        group_key=group_key,
        match_group=match_group,
        current_image=current_image,
    )
    object_table_index = ObjectMeasurementTableIndex.from_tables(source_tables)
    object_table_index_cache[index_cache_key] = object_table_index
    return object_table_index


@dataclass(frozen=True, slots=True)
class MeasurementTableCacheMutation(MeasurementTableObjectFeatureSemantics):
    """Semantic cache mutation caused by one measurement-table write."""

    adapter: "CellProfilerRuntimeAdapter"
    table: MeasurementTable


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


MeasurementQueryCacheEntry = (
    RuntimeObjectLabelMeasurementQuery
    | ObjectMeasurementTableCacheKey
)
MeasurementQueryCacheValue = (
    CellProfilerMeasurementCacheValue
    | ObjectMeasurementTableIndex
)
MeasurementQueryCacheAccessor = Callable[
    [RuntimeValueStore],
    MutableMapping[MeasurementQueryCacheEntry, MeasurementQueryCacheValue],
]


class MeasurementQueryCacheInvalidationPolicy(MeasurementQueryCacheMutationPolicy):
    """Shared object/feature invalidation for measurement-query caches."""

    entry_type: ClassVar[type[MeasurementQueryCacheEntry]]
    feature_scoped: ClassVar[bool] = False
    store_cache_accessor: ClassVar[MeasurementQueryCacheAccessor]

    def apply_query_cache_mutation(self, mutation: MeasurementTableCacheMutation) -> None:
        cache = type(self).store_cache_accessor(mutation.adapter.runtime_value_store)
        for entry in tuple(cache):
            if not isinstance(entry, type(self).entry_type):
                continue
            if (
                type(self).feature_scoped
                and mutation.feature_names
                and entry.feature_name not in mutation.feature_names
            ):
                continue
            if entry.object_name in mutation.object_names:
                del cache[entry]


class ObjectMeasurementTableCacheInvalidationPolicy(
    MeasurementQueryCacheInvalidationPolicy
):
    """Invalidate object-subject measurement table query cache entries."""

    policy_name = "object_measurement_table"
    entry_type = ObjectMeasurementTableCacheKey
    store_cache_accessor = staticmethod(object_measurement_table_cache)


class ObjectMeasurementTableIndexInvalidationPolicy(MeasurementQueryCacheMutationPolicy):
    """Invalidate object-subject indexes touched by measurement-table writes."""

    policy_name = "object_measurement_table_index"

    def apply_query_cache_mutation(self, mutation: MeasurementTableCacheMutation) -> None:
        index_cache = object_measurement_table_index_cache(
            mutation.adapter.runtime_value_store
        )
        for cache_key in tuple(index_cache):
            match cache_key:
                case ObjectMeasurementTableIndexCacheKey(
                    group_key=group_key,
                    match_group=True,
                ) if group_key == mutation.adapter.group_key:
                    del index_cache[cache_key]
                case ObjectMeasurementTableIndexCacheKey(match_group=False):
                    del index_cache[cache_key]
