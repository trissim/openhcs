"""Renderer-independent debug inspector view models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping

from openhcs.core.debug import DebugArtifactRef, DebugInvocationParameter

DebugViewRowBuilder = Callable[[object], tuple[str, ...]]


class DebugViewTableProjection(Enum):
    """Closed table projection family used by debug inspectors."""

    ARTIFACT_REFS = "artifact_refs"
    INVOCATION_PARAMETERS = "invocation_parameters"


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


@dataclass(frozen=True, slots=True)
class DebugViewSection:
    """One named debug view section."""

    title: str
    table: DebugViewTable | None = None
    text: str | None = None


@dataclass(frozen=True, slots=True)
class DebugViewModel:
    """Renderer-independent debug inspector model."""

    title: str
    sections: tuple[DebugViewSection, ...]


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
