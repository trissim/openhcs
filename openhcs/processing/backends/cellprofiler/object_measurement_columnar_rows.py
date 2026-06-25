"""Columnar row base classes for CellProfiler object measurements."""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence

from openhcs.core.runtime_semantics import (
    ObjectMeasurementSliceValueRow,
    ObjectMeasurementValueRow,
)
from openhcs.core.runtime_values import ColumnarRows


class ObjectMeasurementColumnarRows(ColumnarRows, ABC):
    """Columnar object measurement rows with row-object iteration compatibility."""

    slice_index: int | None

    @property
    def covers_declared_object_measurement_domain(self) -> bool:
        """OpenHCS object-measurement carriers are built on their label domain."""
        return True

    def __len__(self) -> int:
        return self.row_count()

    def __iter__(self):
        row_type = (
            ObjectMeasurementValueRow
            if self.slice_index is None
            else ObjectMeasurementSliceValueRow
        )
        columns = self.columns
        slice_values: Sequence[object] | None = columns.get("slice_index")
        for row_index in range(len(self)):
            kwargs = (
                {}
                if slice_values is None
                else {"slice_index": int(slice_values[row_index])}
            )
            yield row_type(
                object_label=int(columns["object_label"][row_index]),
                feature_name=str(columns["feature_name"][row_index]),
                result_value=float(columns["result_value"][row_index]),
                **kwargs,
            )
