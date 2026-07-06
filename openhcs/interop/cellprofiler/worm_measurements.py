"""CellProfiler worm measurement schema semantics."""

from __future__ import annotations
from openhcs.core.runtime_semantics import MeasurementRowAxisField

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np



WormMeasurementRows = tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class WormMeasurementRowSelection:
    """Object-scoped worm measurement rows with explicit absence semantics."""

    rows: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_rows(
        cls,
        rows: WormMeasurementRows,
        *,
        object_name: str | None,
    ) -> "WormMeasurementRowSelection":
        filtered_rows = tuple(
            row
            for row in rows
            if object_name is None
            or row.get(MeasurementRowAxisField.OBJECT_NAME.value) == object_name
        )
        return cls(
            tuple(
                sorted(
                    filtered_rows,
                    key=lambda row: int(row.get(MeasurementRowAxisField.OBJECT_NUMBER.value, 0)),
                )
            )
        )

    @property
    def is_empty(self) -> bool:
        return not self.rows

    def control_point_array(
        self,
        schema: "WormControlPointMeasurementSchema",
    ) -> np.ndarray:
        control_points = np.zeros(
            (len(self.rows), 2, schema.num_control_points),
            dtype=float,
        )
        for row_index, row in enumerate(self.rows):
            for control_point_index in range(schema.num_control_points):
                field_index = control_point_index + 1
                row_field = schema.field(WormControlPointAxis.ROW, field_index).name
                column_field = schema.field(
                    WormControlPointAxis.COLUMN,
                    field_index,
                ).name
                try:
                    row_value = row[row_field]
                    column_value = row[column_field]
                except KeyError as exc:
                    raise ValueError(
                        "UntangleWorms measurement rows are missing required "
                        f"control-point field {exc.args[0]!r}."
                    ) from exc
                control_points[row_index, 0, control_point_index] = float(row_value)
                control_points[row_index, 1, control_point_index] = float(column_value)
        return control_points


class WormControlPointAxis(str, Enum):
    """Axes encoded by UntangleWorms control-point measurement fields."""

    ROW = "y"
    COLUMN = "x"


@dataclass(frozen=True, slots=True)
class WormControlPointMeasurementField:
    """One CellProfiler worm control-point measurement field."""

    axis: WormControlPointAxis
    index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", WormControlPointAxis(self.axis))
        if self.index <= 0:
            raise ValueError("Worm control-point field indexes are 1-based.")

    @property
    def name(self) -> str:
        return f"worm_control_point_{self.axis.value}_{self.index}"


@dataclass(frozen=True, slots=True)
class WormControlPointMeasurementSchema:
    """Schema for UntangleWorms measurement rows consumed by StraightenWorms."""

    num_control_points: int

    def __post_init__(self) -> None:
        if self.num_control_points <= 0:
            raise ValueError("num_control_points must be positive.")

    def field(
        self,
        axis: WormControlPointAxis,
        index: int,
    ) -> WormControlPointMeasurementField:
        if index > self.num_control_points:
            raise ValueError(
                f"Control-point field index {index} exceeds schema size "
                f"{self.num_control_points}."
            )
        return WormControlPointMeasurementField(axis=axis, index=index)

    def row_fields(
        self,
        control_coords: np.ndarray,
    ) -> dict[str, float]:
        return {
            self.field(axis, index).name: float(value)
            for index, (row_coord, column_coord) in enumerate(control_coords, start=1)
            for axis, value in (
                (WormControlPointAxis.COLUMN, column_coord),
                (WormControlPointAxis.ROW, row_coord),
            )
        }

    def control_points_from_rows(
        self,
        rows: WormMeasurementRows,
        *,
        object_name: str | None = None,
    ) -> np.ndarray | None:
        selection = self.select_rows(rows, object_name=object_name)
        if selection.is_empty:
            return None
        return selection.control_point_array(self)

    def select_rows(
        self,
        rows: WormMeasurementRows,
        *,
        object_name: str | None,
    ) -> WormMeasurementRowSelection:
        return WormMeasurementRowSelection.from_rows(
            rows,
            object_name=object_name,
        )

    def rows_for_object(
        self,
        rows: WormMeasurementRows,
        *,
        object_name: str | None,
    ) -> tuple[Mapping[str, Any], ...]:
        return self.select_rows(rows, object_name=object_name).rows
