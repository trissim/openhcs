"""Exact producer-owned measurement slice-domain consumer tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    MeasurementsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
)
from openhcs.core.measurement_feature_queries import MeasurementFeatureQuery
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_measurements import MeasurementScope, MeasurementSubject
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_tables import (
    ObjectMeasurementTableIndex,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    MeasurementImageOperandVectorResolution,
)
from tests.unit.cellprofiler_runtime_test_support import (
    cellprofiler_runtime_input_edge_for_test,
    runtime_adapter_request_for_test,
)


def _measurement_consumer(image):
    return image


class _MeasurementInputAdapter:
    def __init__(
        self,
        runtime_values: tuple[object, ...],
        callable_contract: CallableContract,
        measurement_spec: ArtifactSpec,
    ) -> None:
        self._runtime_values = runtime_values
        self.measurement_record_queries = 0
        input_plan = ArtifactInputPlan(
            measurement_spec.name,
            "/memory/measurements.pkl",
            artifact_type=MeasurementsArtifactType,
        )
        edge = cellprofiler_runtime_input_edge_for_test(
            input_plan,
            spec=measurement_spec,
            invocation_scope=ComponentGroupScope.ungrouped(),
            producer_selection_scope=ComponentGroupScope.ungrouped(),
            component_scopes=(),
            consumer_variable_components=(),
        )
        self.request = runtime_adapter_request_for_test(
            runtime_value_store=RuntimeValueStore(),
            callable_contract=callable_contract,
            artifact_inputs={edge.key: edge},
            artifact_outputs={},
            axis_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=None,
                value=None,
            ),
        )

    def artifact_input_records(
        self,
        name: str,
        artifact_type: type[MeasurementsArtifactType],
    ) -> tuple[SimpleNamespace, ...]:
        self.measurement_record_queries += 1
        assert name == "measurements"
        assert artifact_type is MeasurementsArtifactType
        return tuple(SimpleNamespace(value=value) for value in self._runtime_values)

    def declared_measurement_input_records(
        self,
        *,
        group_key: str | None = None,
        match_group: bool = True,
        current_image: object | None = None,
    ) -> tuple[SimpleNamespace, ...]:
        del group_key, match_group, current_image
        return self.artifact_input_records("measurements", MeasurementsArtifactType)


def test_object_measurement_index_preserves_producer_slice_indexes() -> None:
    tables = tuple(
        MeasurementTable(
            name="measurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"slice_index": 0, "object_label": 1, "value": value},),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_label", int),
                    FieldSpec("value", float),
                ),
            ),
            subject=MeasurementSubject(
                MeasurementScope.OBJECT, "Cells", "object_label"
            ),
        )
        for value in (11.0, 13.0)
    )

    indexed = ObjectMeasurementTableIndex.from_tables(tables)

    assert tuple(
        table.rows[0]["slice_index"] for table in indexed.for_object("Cells")
    ) == (0, 0)


def test_object_measurement_index_preserves_mixed_payload_and_slice_rows() -> None:
    table = MeasurementTable(
        name="measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "object_label": 1, "value": 11.0},
                {"object_label": 2, "value": 13.0},
            ),
            fields=(
                FieldSpec("slice_index", int, required=False),
                FieldSpec("object_label", int),
                FieldSpec("value", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells", "object_label"),
    )

    indexed = ObjectMeasurementTableIndex.from_tables((table,))

    (indexed_table,) = indexed.for_object("Cells")
    assert tuple(indexed_table.rows) == tuple(table.rows)


def test_declared_measurement_binding_preserves_producer_slice_indexes() -> None:
    request = _measurement_input_request(
        tuple(
            MeasurementTable(
                name="measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"slice_index": 0, "value": value},),
                    fields=(
                        FieldSpec("slice_index", int),
                        FieldSpec("value", float),
                    ),
                ),
                subject=MeasurementSubject(MeasurementScope.ARTIFACT, "measurements"),
            )
            for value in (11.0, 13.0)
        )
    )

    tables = request.declared_measurement_tables()
    repeated = request.declared_measurement_tables()

    assert tuple(table.rows[0]["slice_index"] for table in tables) == (0, 0)
    assert tuple(table.rows[0]["slice_index"] for table in repeated) == (0, 0)
    assert request.adapter.measurement_record_queries == 2


def test_image_measurement_vector_record_lookup_has_no_replacement_cache() -> None:
    feature_name = "ImageQuality_FocusScore_DNA"
    request = _measurement_input_request(
        (
            MeasurementTable(
                name="measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"slice_index": 0, feature_name: 11.0},),
                    fields=(
                        FieldSpec("slice_index", int),
                        FieldSpec(feature_name, float),
                    ),
                ),
                subject=MeasurementSubject(MeasurementScope.ARTIFACT, "measurements"),
            ),
        )
    )
    query = MeasurementFeatureQuery(
        feature_name,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    first = MeasurementImageOperandVectorResolution.runtime_feature_tables(
        request.adapter,
        query,
        group_key=None,
        match_group=False,
    )
    second = MeasurementImageOperandVectorResolution.runtime_feature_tables(
        request.adapter,
        query,
        group_key=None,
        match_group=False,
    )

    assert tuple(first[0].rows) == ({"slice_index": 0, feature_name: 11.0},)
    assert second[0].rows == first[0].rows
    assert request.adapter.measurement_record_queries == 2


def test_declared_measurement_binding_accepts_axisless_payload_domain() -> None:
    request = _measurement_input_request(
        (
            MeasurementTable(
                name="measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"value": 11.0},),
                    fields=(FieldSpec("value", float),),
                ),
                subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
            ),
        )
    )

    tables = request.declared_measurement_tables()

    assert tuple(tables[0].rows) == ({"value": 11.0},)


def _measurement_input_request(
    tables: tuple[MeasurementTable, ...],
) -> RuntimeInputBindingRequest:
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    runtime_values = tuple(
        RuntimeValue.normalize(output_plan, table, axis_id="A01") for table in tables
    )
    measurement_spec = ArtifactSpec.input("measurements", MeasurementsArtifactType)
    raw_callable_contract = CallableContract.from_callable(_measurement_consumer)
    callable_contract = replace(
        raw_callable_contract,
        metadata=replace(
            raw_callable_contract.metadata,
            artifact_inputs=(measurement_spec,),
        ),
    )
    return RuntimeInputBindingRequest(
        adapter=_MeasurementInputAdapter(
            runtime_values,
            callable_contract,
            measurement_spec,
        ),
        kwargs={},
        current_image=object(),
    )
