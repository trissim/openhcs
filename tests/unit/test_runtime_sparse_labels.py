import numpy as np

from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows


def test_sparse_ijv_three_column_schema_is_exact_for_zero_rows() -> None:
    rows = SparseIJVLabelRows(np.zeros((0, 3), dtype=np.int32))

    assert rows.fields == (
        FieldSpec("y", int),
        FieldSpec("x", int),
        FieldSpec("label", int),
    )
    assert tuple(rows.columns) == ("y", "x", "label")
    assert all(column.dtype == np.int32 for column in rows.columns.values())


def test_sparse_ijv_four_column_schema_is_exact_for_zero_rows() -> None:
    rows = SparseIJVLabelRows(
        np.zeros((0, 4), dtype=np.int64),
        slice_count=0,
    )

    assert rows.fields == (
        FieldSpec("slice_index", int),
        FieldSpec("y", int),
        FieldSpec("x", int),
        FieldSpec("label", int),
    )
    assert tuple(rows.columns) == ("slice_index", "y", "x", "label")
    assert all(column.dtype == np.int64 for column in rows.columns.values())
