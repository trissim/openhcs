"""Object-measurement table indexes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from openhcs.core.measurement_feature_queries import (
    MeasurementFeatureQuery,
    MeasurementTableObjectFeatureSemantics,
)
from openhcs.core.measurement_lookup_dialect import (
    RuntimeMeasurementLookupDialectLike,
    resolve_runtime_measurement_lookup_dialect,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
MeasurementTableSelection = tuple[MeasurementTable, ...] | None
MeasurementTablesByObject = Mapping[str, tuple[MeasurementTable, ...]]
MutableMeasurementTablesByObject = dict[str, tuple[MeasurementTable, ...]]


@dataclass(frozen=True, slots=True)
class ObjectMeasurementTableIndex:
    """Nominal object-subject measurement table index."""

    tables: tuple[MeasurementTable, ...]
    tables_by_object: MeasurementTablesByObject
    feature_names_by_table: Mapping[int, frozenset[str]]
    complete: bool = False

    @classmethod
    def from_tables(
        cls,
        tables: tuple[MeasurementTable, ...],
    ) -> "ObjectMeasurementTableIndex":
        """Return a complete index over the provided measurement tables."""
        table_lists: dict[str, list[MeasurementTable]] = {}
        feature_names_by_table: dict[int, frozenset[str]] = {}
        for table in tables:
            table_semantics = MeasurementTableObjectFeatureSemantics.from_table(table)
            feature_names_by_table[id(table)] = table_semantics.feature_names
            for table_object_name in table_semantics.object_names:
                table_lists.setdefault(table_object_name, []).append(table)
        indexed_tables: MutableMeasurementTablesByObject = {
            object_name: tuple(object_tables)
            for object_name, object_tables in table_lists.items()
        }
        return cls(
            tables=tables,
            tables_by_object=MappingProxyType(indexed_tables),
            feature_names_by_table=MappingProxyType(feature_names_by_table),
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
        query_object_name = (
            resolve_runtime_measurement_lookup_dialect(dialect)
            .feature_lookup(feature_name)
            .query_object_name(object_name)
        )
        if query_object_name is None:
            tables = self.tables
        elif not self.complete:
            tables = None
        elif query_object_name in self.tables_by_object:
            tables = self.tables_by_object[query_object_name]
        else:
            tables = ()
        if tables is None:
            return None
        if not tables and query_object_name is not None:
            tables = self.unnamed_object_feature_tables()
        query = MeasurementFeatureQuery(feature_name, dialect=dialect)
        return tuple(
            table for table in tables if self._table_may_carry_feature(table, query)
        )

    def unnamed_object_feature_tables(self) -> tuple[MeasurementTable, ...]:
        """Return object-id tables that do not declare a specific object name."""
        return tuple(
            table
            for table in self.tables
            if table.subject.object_id_field is not None
            and not MeasurementTableObjectFeatureSemantics.from_table(
                table
            ).object_names
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
