import ast
from collections.abc import Callable
import inspect
import textwrap

import numpy as np

from openhcs.core.measurement_row_materialization import MEASUREMENT_SPARSE_CELL
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelVariantData,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_completion import (
    LabelIdMeasurementObjectRowIdentityProjectionStrategy,
    ObjectMeasurementRowCompletionSchema,
    RowOrdinalMeasurementObjectRowIdentityProjectionStrategy,
    RowSequenceMeasurementObjectRowIdentityProjectionStrategy,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    CellProfilerObjectMeasurementRowPolicy,
    DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy,
)


class _AuthorityCallInventory(ast.NodeVisitor):
    """Record authority calls and their lexical loop depth."""

    def __init__(self) -> None:
        self.loop_depth = 0
        self.calls: list[tuple[int, int]] = []
        self.metadata_reads: list[tuple[int, int]] = []

    def _visit_loop(self, node: ast.AST) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    visit_AsyncFor = _visit_loop
    visit_DictComp = _visit_loop
    visit_For = _visit_loop
    visit_GeneratorExp = _visit_loop
    visit_ListComp = _visit_loop
    visit_SetComp = _visit_loop
    visit_While = _visit_loop

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "object_id_field_names"
        ):
            self.calls.append((node.lineno, self.loop_depth))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "metadata_fields":
            self.metadata_reads.append((node.lineno, self.loop_depth))
        self.generic_visit(node)


def _label_payload() -> ObjectLabelPayload:
    return ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[1, 0, 3]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1, 3)),
    )


def test_dense_schema_projection_preserves_declared_axis_order() -> None:
    schema = ObjectMeasurementRowCompletionSchema.from_fields(
        (
            FieldSpec("first_value", float),
            FieldSpec(MeasurementRowAxisField.DIRECTION.value, int),
            FieldSpec(MeasurementRowAxisField.OBJECT_LABEL.value, int),
            FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),
            FieldSpec("second_value", float),
        )
    )

    assert schema.object_id_field == MeasurementRowAxisField.OBJECT_LABEL.value
    assert schema.axis_fields == (
        MeasurementRowAxisField.DIRECTION.value,
        MeasurementRowAxisField.SLICE_INDEX.value,
    )


def test_missing_rows_preserve_sparse_identity_cells_and_exact_values() -> None:
    fields = (
        FieldSpec(MeasurementRowAxisField.OBJECT_NUMBER.value, int),
        FieldSpec(MeasurementRowAxisField.OBJECT_LABEL.value, int),
        FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),
        FieldSpec("value", float),
    )
    schema = ObjectMeasurementRowCompletionSchema(
        fields=fields,
        object_id_field=MeasurementRowAxisField.OBJECT_LABEL.value,
        axis_fields=(MeasurementRowAxisField.SLICE_INDEX.value,),
    )
    policy = CellProfilerObjectMeasurementRowPolicy()
    label_payload = _label_payload()

    columnar_rows = schema.missing_columnar_rows(
        missing_row_keys=((2, (0,)), (4, (1,))),
        label_payload=label_payload,
        row_policy=policy,
        object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
    )
    scalar_row = schema.missing_row(
        object_id=2,
        axis_key=(0,),
        label_payload=label_payload,
        row_policy=policy,
    )

    assert columnar_rows.fields == fields
    assert tuple(columnar_rows.columns) == tuple(field.name for field in fields)
    assert columnar_rows.object_row_identity is MeasurementObjectRowIdentity.LABEL_ID
    assert all(
        value is MEASUREMENT_SPARSE_CELL
        for value in columnar_rows.column_values(
            MeasurementRowAxisField.OBJECT_NUMBER.value
        )
    )
    assert tuple(
        columnar_rows.column_values(MeasurementRowAxisField.OBJECT_LABEL.value)
    ) == (2, 4)
    assert tuple(
        columnar_rows.column_values(MeasurementRowAxisField.SLICE_INDEX.value)
    ) == (0, 1)
    assert all(np.isnan(value) for value in columnar_rows.column_values("value"))

    first_columnar_row = columnar_rows.row_mappings()[0]
    assert MeasurementRowAxisField.OBJECT_NUMBER.value not in first_columnar_row
    assert scalar_row.keys() == first_columnar_row.keys()
    assert scalar_row[MeasurementRowAxisField.OBJECT_LABEL.value] == 2
    assert scalar_row[MeasurementRowAxisField.SLICE_INDEX.value] == 0
    assert np.isnan(scalar_row["value"])
    assert np.isnan(first_columnar_row["value"])


def test_owned_methods_query_object_id_authority_once_outside_field_loops() -> None:
    methods: tuple[Callable[..., object], ...] = (
        ObjectMeasurementRowCompletionSchema.object_id_field_from_fields,
        ObjectMeasurementRowCompletionSchema.axis_fields_from_fields,
        ObjectMeasurementRowCompletionSchema.missing_columnar_rows,
        ObjectMeasurementRowCompletionSchema.missing_row,
    )

    for method in methods:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        inventory = _AuthorityCallInventory()
        inventory.visit(tree)

        assert len(inventory.calls) == 1, method.__qualname__
        assert inventory.calls[0][1] == 0, method.__qualname__


def test_projection_reducers_resolve_metadata_once_outside_row_loops() -> None:
    methods: tuple[Callable[..., object], ...] = (
        LabelIdMeasurementObjectRowIdentityProjectionStrategy.project_rows,
        RowOrdinalMeasurementObjectRowIdentityProjectionStrategy.project_rows,
        RowSequenceMeasurementObjectRowIdentityProjectionStrategy.project_rows,
        DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy.project_completion_rows,
    )

    for method in methods:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        inventory = _AuthorityCallInventory()
        inventory.visit(tree)

        assert len(inventory.metadata_reads) == 1, method.__qualname__
        assert inventory.metadata_reads[0][1] == 0, method.__qualname__
