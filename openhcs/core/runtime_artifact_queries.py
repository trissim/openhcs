"""Semantic queries over typed OpenHCS runtime artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.runtime_semantics import MeasurementScope
from openhcs.core.runtime_stores import (
    RuntimeValueStore,
    StoredRuntimeValue,
)
from openhcs.core.runtime_values import MeasurementTable, ObjectRelationship


MEASUREMENT_OBJECT_NAME_FIELD = "object_name"


@dataclass(frozen=True, slots=True)
class RuntimeArtifactQueryContext:
    """Execution-scope view over a RuntimeValueStore."""

    store: RuntimeValueStore
    axis_id: str
    group_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.store, RuntimeValueStore):
            raise TypeError(
                "RuntimeArtifactQueryContext.store must be RuntimeValueStore, "
                f"got {type(self.store).__name__}."
            )
        if not self.axis_id:
            raise ValueError("RuntimeArtifactQueryContext.axis_id cannot be empty.")

    @property
    def match_group(self) -> bool:
        return self.group_key is not None

    def find(
        self,
        *,
        kind: ArtifactKind | None = None,
        name: str | None = None,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Find runtime records in this execution scope."""
        return self.store.find(
            name=name,
            kind=kind,
            axis_id=self.axis_id,
            group_key=self.group_key,
            match_group=self.match_group,
        )

    def resolve(
        self,
        *,
        name: str,
        kind: ArtifactKind,
        purpose: str = "runtime artifact",
    ) -> StoredRuntimeValue:
        """Resolve exactly one runtime record in this execution scope."""
        records = self.find(name=name, kind=kind)
        if not records:
            raise RuntimeError(
                f"Missing {purpose} '{name}' ({kind.value}) on axis "
                f"'{self.axis_id}'."
            )
        if len(records) > 1:
            raise RuntimeError(
                f"Ambiguous {purpose} '{name}' ({kind.value}) on axis "
                f"'{self.axis_id}': {records!r}."
            )
        return records[0]


@dataclass(frozen=True, slots=True)
class MeasurementObjectQuery:
    """Query for measurement tables describing one object set."""

    object_name: str

    def __post_init__(self) -> None:
        if not self.object_name:
            raise ValueError("MeasurementObjectQuery.object_name cannot be empty.")

    def matches(self, table: MeasurementTable) -> bool:
        if table.subject.scope is MeasurementScope.OBJECT:
            return table.subject.name == self.object_name
        return any(
            measurement_row_object_name(measurement_row_mapping(row))
            == self.object_name
            for row in measurement_rows((table,))
        )


def runtime_measurement_tables(
    context: RuntimeArtifactQueryContext,
) -> tuple[MeasurementTable, ...]:
    """Return all measurement tables in a runtime query context."""
    return tuple(
        MeasurementTable.from_runtime_value(record.value)
        for record in context.find(kind=ArtifactKind.MEASUREMENTS)
    )


def runtime_measurement_tables_for_object(
    context: RuntimeArtifactQueryContext,
    object_name: str,
) -> tuple[MeasurementTable, ...]:
    """Return measurement tables whose subject is one object set."""
    query = MeasurementObjectQuery(object_name)
    return tuple(
        table
        for table in runtime_measurement_tables(context)
        if query.matches(table)
    )


def runtime_relationship(
    context: RuntimeArtifactQueryContext,
    name: str,
) -> ObjectRelationship:
    """Return one relationship artifact as native OpenHCS relationship value."""
    record = context.resolve(
        name=name,
        kind=ArtifactKind.RELATIONSHIPS,
        purpose="relationship artifact",
    )
    return ObjectRelationship.from_runtime_value(record.value)


def measurement_rows(
    measurement_tables: tuple[MeasurementTable, ...],
) -> tuple[object, ...]:
    """Flatten row payloads from measurement tables."""
    rows: list[object] = []
    for table in measurement_tables:
        if isinstance(table.rows, list | tuple):
            rows.extend(table.rows)
            continue
        rows.append(table.rows)
    return tuple(rows)


def measurement_row_mapping(row: object) -> Mapping[str, object]:
    """Return a mapping view for a supported measurement row payload."""
    if isinstance(row, Mapping):
        return row
    if is_dataclass(row):
        return {field.name: getattr(row, field.name) for field in fields(row)}
    try:
        return vars(row)
    except TypeError as exc:
        raise TypeError(
            f"Unsupported measurement row type {type(row).__name__}."
        ) from exc


def measurement_row_object_name(row: Mapping[str, object]) -> str | None:
    """Return the object-set owner encoded on one measurement row."""
    value = row.get(MEASUREMENT_OBJECT_NAME_FIELD)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def annotate_measurement_row_object(
    row: object,
    object_name: str,
) -> Mapping[str, object]:
    """Return a measurement row with explicit object-set ownership."""
    normalized_object_name = object_name.strip()
    if not normalized_object_name:
        raise ValueError("object_name cannot be empty.")
    return {
        **dict(measurement_row_mapping(row)),
        MEASUREMENT_OBJECT_NAME_FIELD: normalized_object_name,
    }
