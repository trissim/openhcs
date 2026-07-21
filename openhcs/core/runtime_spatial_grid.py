"""Nominal runtime spatial-grid values."""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass, replace
from typing import Any, Self

import numpy as np

from openhcs.core.artifacts import NamedArtifactPayload
from openhcs.core.source_spatial_domain import SpatialShapeYX

from enum import Enum
from openhcs.core.alias_property import AliasProperty

SPATIAL_GRID_DEFAULT_SLICE_INDEX = 0


@dataclass(frozen=True, slots=True)
class SpatialGridAxis:
    """One physical axis of a rectangular spatial grid."""

    spacing: float
    origin: float
    locations: tuple[float, ...] | None = None

    def normalized(self, count: int, field_name: str) -> "SpatialGridAxis":
        """Return this axis with explicit center locations for every index."""
        spacing = float(self.spacing)
        origin = float(self.origin)
        if spacing <= 0:
            raise ValueError(f"{field_name}.spacing must be positive.")
        if self.locations is None:
            locations = tuple(origin + index * spacing for index in range(count))
        elif len(self.locations) != count:
            raise ValueError(f"{field_name}.locations must match axis length.")
        else:
            locations = tuple(float(value) for value in self.locations)
        return type(self)(spacing=spacing, origin=origin, locations=locations)


class SpatialGridOrdering(str, Enum):
    """Primary axis used when numbering positions in a spatial grid."""

    BY_ROWS = "rows"
    BY_COLUMNS = "columns"


class SpatialGridOrigin(str, Enum):
    """Corner used as the numbering origin for a spatial grid."""

    def __new__(cls, value: str, reverses_rows: bool, reverses_columns: bool):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._reverses_rows = reverses_rows
        obj._reverses_columns = reverses_columns
        return obj

    TOP_LEFT = ("top_left", False, False)
    BOTTOM_LEFT = ("bottom_left", True, False)
    TOP_RIGHT = ("top_right", False, True)
    BOTTOM_RIGHT = ("bottom_right", True, True)
    reverses_rows = AliasProperty[bool]("_reverses_rows")
    reverses_columns = AliasProperty[bool]("_reverses_columns")


@dataclass(slots=True, init=False)
class SpatialGrid(NamedArtifactPayload):
    """Native OpenHCS rectangular spatial grid definition."""

    name: str
    rows: int
    columns: int
    column_axis: SpatialGridAxis
    row_axis: SpatialGridAxis
    slice_index: int
    total_width: float | None
    total_height: float | None
    origin: SpatialGridOrigin
    ordering: SpatialGridOrdering
    spot_table: tuple[tuple[int, ...], ...] | None
    source_spatial_shape_yx: tuple[int, int] | None

    def __init__(
        self,
        *,
        name: str,
        rows: int,
        columns: int,
        x_spacing: float | None = None,
        y_spacing: float | None = None,
        x_origin: float | None = None,
        y_origin: float | None = None,
        slice_index: int = 0,
        total_width: float | None = None,
        total_height: float | None = None,
        origin: SpatialGridOrigin = SpatialGridOrigin.TOP_LEFT,
        ordering: SpatialGridOrdering = SpatialGridOrdering.BY_ROWS,
        x_locations: tuple[float, ...] | None = None,
        y_locations: tuple[float, ...] | None = None,
        column_axis: SpatialGridAxis | None = None,
        row_axis: SpatialGridAxis | None = None,
        spot_table: tuple[tuple[int, ...], ...] | None = None,
        source_spatial_shape_yx: tuple[int, int] | None = None,
    ) -> None:
        self.name = name
        self.rows = int(rows)
        self.columns = int(columns)
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("SpatialGrid dimensions must be positive.")
        self.slice_index = int(slice_index)
        self.total_width = total_width
        self.total_height = total_height
        self.origin = SpatialGridOrigin(
            origin,
        )
        self.ordering = SpatialGridOrdering(
            ordering,
        )
        if column_axis is None:
            if x_spacing is None or x_origin is None:
                raise TypeError(
                    "SpatialGrid requires column_axis or both x_spacing and x_origin."
                )
            column_axis = SpatialGridAxis(
                spacing=x_spacing,
                origin=x_origin,
                locations=x_locations,
            )
        if row_axis is None:
            if y_spacing is None or y_origin is None:
                raise TypeError(
                    "SpatialGrid requires row_axis or both y_spacing and y_origin."
                )
            row_axis = SpatialGridAxis(
                spacing=y_spacing,
                origin=y_origin,
                locations=y_locations,
            )
        self.column_axis = column_axis.normalized(
            count=self.columns,
            field_name="SpatialGrid.column_axis",
        )
        self.row_axis = row_axis.normalized(
            count=self.rows,
            field_name="SpatialGrid.row_axis",
        )
        if spot_table is None:
            self.spot_table = self.derived_spot_table()
        elif len(spot_table) != self.rows or any(
            len(row) != self.columns for row in spot_table
        ):
            raise ValueError("SpatialGrid.spot_table must match rows x columns.")
        else:
            self.spot_table = tuple(
                tuple(int(value) for value in row) for row in spot_table
            )
        self.source_spatial_shape_yx = source_spatial_shape_yx
        self.__post_init__()

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> Self:
        """Build a spatial grid from its canonical mapping representation."""
        rows = int(data["rows"])
        columns = int(data["columns"])
        x_spacing = float(data["x_spacing"])
        y_spacing = float(data["y_spacing"])
        x_origin = float(data["x_origin"])
        y_origin = float(data["y_origin"])

        def optional_float_tuple(field_name: str) -> tuple[float, ...] | None:
            value = data.get(field_name)
            if value is None:
                return None
            if not isinstance(value, Sequence) or isinstance(
                value, (str, bytes, bytearray)
            ):
                raise TypeError(
                    f"SpatialGrid.{field_name} must be a sequence, "
                    f"got {type(value).__name__}."
                )
            return tuple(float(item) for item in value)

        raw_spot_table = data.get("spot_table")
        if raw_spot_table is not None and (
            not isinstance(raw_spot_table, Sequence)
            or isinstance(raw_spot_table, (str, bytes, bytearray))
        ):
            raise TypeError(
                "SpatialGrid.spot_table must be a sequence, "
                f"got {type(raw_spot_table).__name__}."
            )
        return cls(
            name=name,
            rows=rows,
            columns=columns,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            x_origin=x_origin,
            y_origin=y_origin,
            slice_index=int(data.get("slice_index", SPATIAL_GRID_DEFAULT_SLICE_INDEX)),
            total_width=(
                None if data.get("total_width") is None else float(data["total_width"])
            ),
            total_height=(
                None
                if data.get("total_height") is None
                else float(data["total_height"])
            ),
            origin=SpatialGridOrigin(
                data.get("origin", SpatialGridOrigin.TOP_LEFT),
            ),
            ordering=SpatialGridOrdering(
                data.get("ordering", SpatialGridOrdering.BY_ROWS),
            ),
            x_locations=optional_float_tuple("x_locations"),
            y_locations=optional_float_tuple("y_locations"),
            spot_table=(
                None
                if raw_spot_table is None
                else tuple(tuple(int(item) for item in row) for row in raw_spot_table)
            ),
            source_spatial_shape_yx=(
                None
                if (
                    shape := SpatialShapeYX.optional_from_mapping(
                        data,
                        "source_spatial_shape_yx",
                    )
                )
                is None
                else shape.as_tuple()
            ),
        )

    @classmethod
    def from_runtime_value(cls, name: str, value: object) -> Self:
        """Return one nominal grid from an accepted scalar runtime value."""

        if isinstance(value, cls):
            return value if value.name == name else value.with_name(name)
        if isinstance(value, Mapping):
            return cls.from_mapping(name, value)
        raise TypeError(
            f"Spatial grid output {name!r} must be SpatialGrid or mapping-backed, "
            f"got {type(value).__name__}."
        )

    def __post_init__(self) -> None:
        self.validate_artifact_name()
        if self.total_width is None:
            self.total_width = self.x_spacing * self.columns
        if self.total_height is None:
            self.total_height = self.y_spacing * self.rows
        if self.source_spatial_shape_yx is not None:
            self.source_spatial_shape_yx = SpatialShapeYX.from_sequence(
                self.source_spatial_shape_yx,
                field_name="SpatialGrid.source_spatial_shape_yx",
            ).as_tuple()

    @property
    def x_spacing(self) -> float:
        return self.column_axis.spacing

    @property
    def y_spacing(self) -> float:
        return self.row_axis.spacing

    @property
    def x_origin(self) -> float:
        return self.column_axis.origin

    @property
    def y_origin(self) -> float:
        return self.row_axis.origin

    @property
    def x_locations(self) -> tuple[float, ...]:
        if self.column_axis.locations is None:
            raise ValueError("SpatialGrid.column_axis is not normalized.")
        return self.column_axis.locations

    @property
    def y_locations(self) -> tuple[float, ...]:
        if self.row_axis.locations is None:
            raise ValueError("SpatialGrid.row_axis is not normalized.")
        return self.row_axis.locations

    def with_name(self, name: str) -> Self:
        """Return the same grid under a different artifact name."""
        return replace(self, name=name)

    def derived_spot_table(self) -> tuple[tuple[int, ...], ...]:
        """Return the object-number topology declared by this grid."""
        object_ids = np.arange(1, self.rows * self.columns + 1, dtype=np.int32)
        if self.ordering is SpatialGridOrdering.BY_COLUMNS:
            table = object_ids.reshape(self.rows, self.columns)
        else:
            table = object_ids.reshape(self.columns, self.rows).T
        if self.origin.reverses_rows:
            table = table[::-1, :]
        if self.origin.reverses_columns:
            table = table[:, ::-1]
        return tuple(tuple(int(value) for value in row) for row in table)

    def as_mapping(self) -> dict[str, Any]:
        """Return a JSON/metadata-compatible grid payload."""
        return {
            "slice_index": self.slice_index,
            "rows": self.rows,
            "columns": self.columns,
            "x_spacing": self.x_spacing,
            "y_spacing": self.y_spacing,
            "x_origin": self.x_origin,
            "y_origin": self.y_origin,
            "total_width": self.total_width,
            "total_height": self.total_height,
            "origin": self.origin.value,
            "ordering": self.ordering.value,
            "x_locations": self.x_locations,
            "y_locations": self.y_locations,
            "spot_table": self.spot_table,
            "source_spatial_shape_yx": self.source_spatial_shape_yx,
        }
