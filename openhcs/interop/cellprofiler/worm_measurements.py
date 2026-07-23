"""CellProfiler worm measurement schema semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.runtime_tabular_values import ColumnarRows


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
        rows: ColumnarRows,
        *,
        object_name: str | None = None,
    ) -> np.ndarray | None:
        descriptor_rows = self.rows_for_object(rows, object_name=object_name)
        if not descriptor_rows:
            return None
        control_points = np.zeros(
            (len(descriptor_rows), 2, self.num_control_points),
            dtype=float,
        )
        for row_index, row in enumerate(descriptor_rows):
            for control_point_index in range(self.num_control_points):
                field_index = control_point_index + 1
                row_field = self.field(WormControlPointAxis.ROW, field_index).name
                column_field = self.field(
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

    def rows_for_object(
        self,
        rows: ColumnarRows,
        *,
        object_name: str | None,
    ) -> tuple[Mapping[str, Any], ...]:
        object_number_field = MeasurementRowAxisField.OBJECT_NUMBER.value
        object_name_field = MeasurementRowAxisField.OBJECT_NAME.value
        descriptor_rows = (
            row
            for row in rows.iter_row_mappings()
            if object_number_field in row
            and (object_name is None or row.get(object_name_field) == object_name)
        )
        return tuple(
            sorted(
                descriptor_rows,
                key=lambda row: int(row[object_number_field]),
            )
        )
