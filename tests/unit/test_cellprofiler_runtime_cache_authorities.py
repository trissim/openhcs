"""Focused tests for retained CellProfiler runtime cache authorities."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from tests.unit.cellprofiler_runtime_test_support import (
    cellprofiler_runtime_adapter_for_test,
    cellprofiler_runtime_input_edge_for_test,
)

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    RelationshipsArtifactType,
)
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
)
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.measurement_row_materialization import MeasurementSparseColumnarRows
from openhcs.core.runtime_object_labels import (
    ObjectLabelSet,
    ObjectLabelVariantData,
)
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
    ObjectRelationshipDeclaration,
)
from openhcs.core.runtime_measurements import MeasurementScope, MeasurementSubject
from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
)
from openhcs.core.runtime_relationships import DirectedObjectRelationshipPayload
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.interop.cellprofiler.runtime.object_label_measurements import (
    ObjectLabelMeasurementSliceRequest,
    object_label_measurement_values_cache,
)

AXIS_ID = "A01"
DNA_IMAGE = "DNA"


def test_runtime_adapter_recomposes_images_from_runtime_value_store() -> None:
    store = RuntimeValueStore()
    output_plan = ArtifactOutputPlan(
        name=DNA_IMAGE,
        path="/memory/DNA.pkl",
        artifact_type=ImageArtifactType,
    )
    input_plan = ArtifactInputPlan(
        name=DNA_IMAGE,
        path=output_plan.path,
        artifact_type=ImageArtifactType,
    )
    input_spec = ArtifactSpec.input(DNA_IMAGE, ImageArtifactType)
    contract = CallableContract.from_callable(
        test_runtime_adapter_recomposes_images_from_runtime_value_store
    )
    contract = replace(
        contract,
        metadata=replace(contract.metadata, artifact_inputs=(input_spec,)),
    )
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=RuntimeExecutionAxisScope(axis_id=AXIS_ID),
        callable_contract=contract,
        artifact_inputs={
            edge.key: edge
            for edge in (
                cellprofiler_runtime_input_edge_for_test(
                    input_plan,
                    spec=input_spec,
                    invocation_scope=ComponentGroupScope.ungrouped(),
                    producer_selection_scope=ComponentGroupScope.ungrouped(),
                    component_scopes=(),
                    consumer_variable_components=(),
                ),
            )
        },
    )
    first_payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE
    ).payload_with(np.full((1, 2, 2), 1.0, dtype=np.float32), None)
    first_value = RuntimeValue.normalize(
        output_plan,
        first_payload,
        axis_id=AXIS_ID,
    )
    store.replace(first_value, path=output_plan.path, backend=adapter.backend)

    first = adapter.get_image(DNA_IMAGE)
    recomposed = adapter.get_image(DNA_IMAGE)

    assert recomposed is first
    np.testing.assert_array_equal(
        image_payload_data(recomposed),
        np.full((1, 2, 2), 1.0, dtype=np.float32),
    )

    replacement_payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE
    ).payload_with(np.full((1, 2, 2), 2.0, dtype=np.float32), None)
    replacement_value = RuntimeValue.normalize(
        output_plan,
        replacement_payload,
        axis_id=AXIS_ID,
    )
    store.replace(replacement_value, path=output_plan.path, backend=adapter.backend)

    refreshed = adapter.get_image(DNA_IMAGE)

    assert refreshed is not recomposed
    np.testing.assert_array_equal(
        image_payload_data(refreshed),
        np.full((1, 2, 2), 2.0, dtype=np.float32),
    )


def test_runtime_adapter_measurement_queries_follow_store_revisions() -> None:
    store = RuntimeValueStore()
    output_plan = ArtifactOutputPlan(
        name="Measurements",
        path="/memory/Measurements.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    input_plan = ArtifactInputPlan(
        name=output_plan.name,
        path=output_plan.path,
        artifact_type=MeasurementsArtifactType,
    )
    input_spec = ArtifactSpec.input(output_plan.name, MeasurementsArtifactType)
    contract = CallableContract.from_callable(
        test_runtime_adapter_measurement_queries_follow_store_revisions
    )
    contract = replace(
        contract,
        metadata=replace(contract.metadata, artifact_inputs=(input_spec,)),
    )
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=RuntimeExecutionAxisScope(axis_id=AXIS_ID),
        callable_contract=contract,
        artifact_inputs={
            edge.key: edge
            for edge in (
                cellprofiler_runtime_input_edge_for_test(
                    input_plan,
                    spec=input_spec,
                    invocation_scope=ComponentGroupScope.ungrouped(),
                    producer_selection_scope=ComponentGroupScope.ungrouped(),
                    component_scopes=(),
                    consumer_variable_components=(),
                ),
            )
        },
    )
    first_table = MeasurementTable(
        name=output_plan.name,
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"object_id": 1, "result_value": 10.0},),
            fields=(FieldSpec("object_id", int), FieldSpec("result_value", float)),
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, output_plan.name),
    )
    first_value = RuntimeValue.normalize(
        output_plan,
        first_table,
        axis_id=AXIS_ID,
    )
    store.replace(first_value, path=output_plan.path, backend=adapter.backend)

    first_tables = adapter.measurement_tables()

    replacement_table = MeasurementTable(
        name=output_plan.name,
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"object_id": 1, "result_value": 20.0},),
            fields=(FieldSpec("object_id", int), FieldSpec("result_value", float)),
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, output_plan.name),
    )
    replacement_value = RuntimeValue.normalize(
        output_plan,
        replacement_table,
        axis_id=AXIS_ID,
    )
    store.replace(replacement_value, path=output_plan.path, backend=adapter.backend)

    refreshed_tables = adapter.measurement_tables()

    assert tuple(first_tables[0].rows.iter_row_mappings()) == (
        {"object_id": 1, "result_value": 10.0},
    )
    assert tuple(refreshed_tables[0].rows.iter_row_mappings()) == (
        {"object_id": 1, "result_value": 20.0},
    )


def test_object_label_measurements_use_store_bound_values_cache() -> None:
    store = RuntimeValueStore()
    adapter = SimpleNamespace(
        request=SimpleNamespace(
            context=SimpleNamespace(runtime_value_store=store),
            axis_scope=SimpleNamespace(axis_id=AXIS_ID),
            group_key=None,
            artifact_inputs={},
        )
    )
    request = ObjectLabelMeasurementSliceRequest(
        object_name="Cells",
        feature_name="AreaShape_Area",
        group_key=None,
        slice_index=0,
        labels=ObjectLabelSet(
            name="Cells",
            variant_data=ObjectLabelVariantData(
                labels=np.asarray([[0, 1]], dtype=np.int32)
            ),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
    )
    query = request.measurement_query(adapter)
    expected = (np.asarray([11.0], dtype=np.float64),)
    object_label_measurement_values_cache(store)[query] = expected

    resolved = request.values(adapter)

    assert resolved is expected


def test_relationship_replacement_invalidates_cached_child_counts() -> None:
    store = RuntimeValueStore()
    relationship_name = "Cells_Nuclei_relationships"
    relationship_plan = ArtifactOutputPlan(
        name=relationship_name,
        path=f"/memory/{relationship_name}.pkl",
        artifact_type=RelationshipsArtifactType,
    )
    declaration = ObjectRelationshipDeclaration(
        source=ArtifactSpec.input("Cells", ObjectLabelsArtifactType).ref(),
        target=ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType).ref(),
        producer_module_number=1,
        relationship_type="parent_child",
        source_role="parent",
        target_role="child",
        source_id_field="parent_id",
        target_id_field="child_id",
        source_runtime_slice_offset=0,
        target_runtime_slice_offset=0,
    )
    relationship_spec = ArtifactSpec.input(
        relationship_name,
        RelationshipsArtifactType,
        relations=(declaration,),
    )
    relationship_input_plan = ArtifactInputPlan(
        name=relationship_spec.name,
        path=relationship_plan.path,
        artifact_type=relationship_spec.artifact_type,
    )
    relationship_edge = cellprofiler_runtime_input_edge_for_test(
        relationship_input_plan,
        spec=relationship_spec,
        invocation_scope=ComponentGroupScope.ungrouped(),
        producer_selection_scope=ComponentGroupScope.ungrouped(),
        component_scopes=(),
        consumer_variable_components=(),
    )

    def replace_relationship(source_ids: tuple[int, ...]) -> None:
        relationship = ObjectRelationship.from_payload(
            name=relationship_name,
            declaration=declaration,
            payload=DirectedObjectRelationshipPayload(
                source_ids=source_ids,
                target_ids=tuple(range(1, len(source_ids) + 1)),
            ),
        )
        store.replace(
            RuntimeValue.normalize(
                relationship_plan,
                relationship,
                axis_id=AXIS_ID,
            ),
            path=relationship_plan.path,
            backend="memory",
        )

    class RelationshipStoreAdapter:
        request = SimpleNamespace(
            context=SimpleNamespace(runtime_value_store=store),
            axis_scope=SimpleNamespace(axis_id=AXIS_ID),
            group_key=None,
            artifact_inputs={relationship_edge.key: relationship_edge},
        )

        def get_relationship(
            self,
            name: str,
            *,
            artifact_type: type = RelationshipsArtifactType,
            group_key: str | None = None,
        ) -> ObjectRelationship:
            del group_key
            records = store.find(
                name=name,
                artifact_type=artifact_type,
                axis_id=AXIS_ID,
            )
            assert len(records) == 1
            relationship = records[0].value.data
            assert isinstance(relationship, ObjectRelationship)
            return relationship

    request = ObjectLabelMeasurementSliceRequest(
        object_name="Cells",
        feature_name="Children_Nuclei_Count",
        group_key=None,
        slice_index=0,
        labels=ObjectLabelSet(
            name="Cells",
            variant_data=ObjectLabelVariantData(
                labels=np.asarray([[1, 2]], dtype=np.int32)
            ),
            domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
        ),
    )
    adapter = RelationshipStoreAdapter()

    replace_relationship((1, 1))
    first_revision = store.revision
    first = request.values(adapter)
    cached = request.values(adapter)

    assert cached is first
    np.testing.assert_array_equal(first[0], np.asarray([2.0, 0.0]))

    replace_relationship((1, 2))
    refreshed = request.values(adapter)

    assert store.revision > first_revision
    assert refreshed is not first
    np.testing.assert_array_equal(refreshed[0], np.asarray([1.0, 1.0]))
