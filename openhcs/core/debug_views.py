"""Renderer-independent debug inspector view models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping

from openhcs.core.debug import DebugArtifactRef, DebugInvocationParameter
from openhcs.core.runtime_stores import RuntimeValueStore, StoredRuntimeValue

DebugViewRowBuilder = Callable[[object], tuple[str, ...]]


class DebugViewTableProjection(Enum):
    """Closed table projection family used by debug inspectors."""

    ARTIFACT_REFS = "artifact_refs"
    INVOCATION_PARAMETERS = "invocation_parameters"
    RUNTIME_VALUE_RECORDS = "runtime_value_records"


@dataclass(frozen=True, slots=True)
class DebugViewTableProjectionSpec:
    """Authoritative constructor variant for a debug table projection."""

    projection: DebugViewTableProjection
    columns: tuple[str, ...]
    row_builder: DebugViewRowBuilder

    def table_for(self, values: tuple[object, ...]) -> "DebugViewTable":
        return DebugViewTable(
            columns=self.columns,
            rows=tuple(self.row_builder(value) for value in values),
        )


def artifact_ref_row(value: object) -> tuple[str, ...]:
    if not isinstance(value, DebugArtifactRef):
        raise TypeError(
            "artifact_ref_row requires DebugArtifactRef, "
            f"got {type(value).__name__}."
        )
    return (
        value.name,
        value.kind.value,
        value.storage_ref,
        "" if value.shape is None else "x".join(map(str, value.shape)),
        value.dtype or "",
    )


def invocation_parameter_row(value: object) -> tuple[str, ...]:
    if not isinstance(value, DebugInvocationParameter):
        raise TypeError(
            "invocation_parameter_row requires DebugInvocationParameter, "
            f"got {type(value).__name__}."
        )
    return (value.name, value.value_repr)


def runtime_value_record_row(value: object) -> tuple[str, ...]:
    if not isinstance(value, StoredRuntimeValue):
        raise TypeError(
            "runtime_value_record_row requires StoredRuntimeValue, "
            f"got {type(value).__name__}."
        )
    key = value.key
    scope = key.scope
    return (
        key.name,
        key.kind.value,
        scope.axis_id,
        scope.group_key or "",
        value.backend,
        value.path,
        key.semantic_id or "",
        type(value.value.data).__qualname__,
    )


DEBUG_VIEW_TABLE_PROJECTIONS: Mapping[
    DebugViewTableProjection,
    DebugViewTableProjectionSpec,
] = MappingProxyType(
    {
        DebugViewTableProjection.ARTIFACT_REFS: DebugViewTableProjectionSpec(
            projection=DebugViewTableProjection.ARTIFACT_REFS,
            columns=("Artifact", "Kind", "Storage ref", "Shape", "DType"),
            row_builder=artifact_ref_row,
        ),
        DebugViewTableProjection.INVOCATION_PARAMETERS: DebugViewTableProjectionSpec(
            projection=DebugViewTableProjection.INVOCATION_PARAMETERS,
            columns=("Parameter", "Value"),
            row_builder=invocation_parameter_row,
        ),
        DebugViewTableProjection.RUNTIME_VALUE_RECORDS: DebugViewTableProjectionSpec(
            projection=DebugViewTableProjection.RUNTIME_VALUE_RECORDS,
            columns=(
                "Name",
                "Kind",
                "Axis",
                "Group",
                "Backend",
                "Path",
                "Semantic",
                "Value type",
            ),
            row_builder=runtime_value_record_row,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class DebugViewTable:
    """Small table-like payload for debug inspectors."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    @classmethod
    def from_projection(
        cls,
        projection: DebugViewTableProjection,
        values: tuple[object, ...],
    ) -> "DebugViewTable":
        return DEBUG_VIEW_TABLE_PROJECTIONS[projection].table_for(values)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> "DebugViewTable":
        return cls(
            columns=tuple(str(column) for column in data["columns"]),
            rows=tuple(
                tuple(str(value) for value in row)
                for row in data["rows"]
            ),
        )


@dataclass(frozen=True, slots=True)
class DebugViewSection:
    """One named debug view section."""

    title: str
    table: DebugViewTable | None = None
    text: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "table": None if self.table is None else self.table.to_json_dict(),
            "text": self.text,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> "DebugViewSection":
        table = data["table"]
        if table is not None and not isinstance(table, Mapping):
            raise TypeError("DebugViewSection.table must be a mapping or None.")
        return cls(
            title=str(data["title"]),
            table=(
                None
                if table is None
                else DebugViewTable.from_json_dict(table)
            ),
            text=None if data["text"] is None else str(data["text"]),
        )


@dataclass(frozen=True, slots=True)
class DebugViewModel:
    """Renderer-independent debug inspector model."""

    title: str
    sections: tuple[DebugViewSection, ...]

    @classmethod
    def from_runtime_value_store(
        cls,
        store: RuntimeValueStore,
        *,
        title: str = "Runtime Values",
    ) -> "DebugViewModel":
        if not isinstance(store, RuntimeValueStore):
            raise TypeError(
                "DebugViewModel.from_runtime_value_store requires RuntimeValueStore, "
                f"got {type(store).__name__}."
            )
        return cls(
            title=title,
            sections=(
                DebugViewSection(
                    "Runtime Value Store",
                    table=DebugViewTable.from_projection(
                        DebugViewTableProjection.RUNTIME_VALUE_RECORDS,
                        store.values(),
                    ),
                ),
            ),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "sections": [section.to_json_dict() for section in self.sections],
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> "DebugViewModel":
        return cls(
            title=str(data["title"]),
            sections=tuple(
                DebugViewSection.from_json_dict(section)
                for section in data["sections"]
            ),
        )


def is_debug_view_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_debug_view_export(name, value)
)
