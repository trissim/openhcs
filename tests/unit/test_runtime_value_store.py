import numpy as np
import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactInputProjectionPlan,
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
)
from openhcs.core.runtime_stores import (
    RuntimeArtifactAddress,
    RuntimeArtifactInput,
    RuntimeArtifactDynamicComponentTarget,
    RuntimeArtifactLocation,
    RuntimeArtifactLocationTarget,
    RuntimeArtifactQuery,
    RuntimeValueStore,
    StoredRuntimeValue,
)
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.runtime_equivalence import (
    RuntimeMeasurementObservationAxis,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
)
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
)
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.core.runtime_artifact_values import ArtifactKey, RuntimeValue


def _runtime_input_edge(
    storage_plan: ArtifactInputPlan,
    *,
    invocation_scope: ComponentGroupScope,
    producer_selection_scope: ComponentGroupScope,
    component_scopes: tuple[ComponentGroupScope, ...],
    consumer_variable_components: tuple[AllComponents, ...],
) -> InvocationArtifactInputEdgePlan:
    invocation_key = FunctionInvocationKey(
        "runtime_input_test",
        DEFAULT_GROUP_KEY,
        0,
    )
    projection = ArtifactInputProjectionPlan(
        invocation_scope=invocation_scope,
        producer_selection_scope=producer_selection_scope,
        component_scopes=component_scopes,
        consumer_variable_components=consumer_variable_components,
    )
    return InvocationArtifactInputEdgePlan(
        key=InvocationArtifactInputProjectionKey(
            invocation_key=invocation_key,
            input_index=0,
        ),
        spec=ArtifactSpec.input(
            storage_plan.name,
            storage_plan.artifact_type,
            sidecar_role=storage_plan.sidecar_role,
        ),
        storage_plan=storage_plan,
        projection=projection,
    )


def _runtime_value(name="measurements", path="/memory/measurements.pkl"):
    return RuntimeValue.normalize(
        ArtifactOutputPlan(
            name=name,
            path=path,
            artifact_type=MeasurementsArtifactType,
            group_keys=("DAPI",),
            group_component=AllComponents.CHANNEL,
        ),
        MeasurementTable(
            name=name,
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"object_id": 1},),
                fields=(FieldSpec("object_id", int),),
            ),
            subject=MeasurementSubject(
                MeasurementScope.ARTIFACT,
                name,
            ),
        ),
        axis_id="A01",
    )


def test_output_plan_uses_its_single_scope_for_ungrouped_invocation():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/Nuclei_1.pkl"},
    )

    resolved = output_plan.for_invocation_group(None)

    assert resolved.group_keys == ("1",)
    assert resolved.group_component is AllComponents.CHANNEL
    assert resolved.path == "/memory/Nuclei_1.pkl"


def test_ungrouped_output_plan_ignores_incidental_invocation_group():
    output_plan = ArtifactOutputPlan(
        name="RGBImage",
        path="/memory/RGBImage.pkl",
        artifact_type=ImageArtifactType,
        group_keys=(None,),
        group_component=None,
        paths_by_group={None: "/memory/RGBImage.pkl"},
    )

    resolved = output_plan.for_invocation_group("1")

    assert resolved.group_keys == (None,)
    assert resolved.group_component is None
    assert resolved.path == "/memory/RGBImage.pkl"


def test_runtime_artifact_address_round_trips_fixed_component_values():
    scope = RuntimeExecutionAxisScope.from_raw(
        "A01",
        component=AllComponents.CHANNEL,
        value="2",
        fixed_component_values=(
            (AllComponents.Z_INDEX, 3),
            (AllComponents.SITE, 1),
        ),
    )
    address = RuntimeArtifactAddress(
        key=ArtifactKey(
            name="measurements",
            artifact_type=MeasurementsArtifactType,
            scope=scope,
        ),
        location=RuntimeArtifactLocation(
            path="/memory/measurements.pkl",
            backend="memory",
        ),
        value_type="MeasurementTable",
    )

    restored = RuntimeArtifactAddress.from_dict(address.to_dict())

    assert restored == address
    assert restored.key.scope.fixed_component_values == scope.fixed_component_values


def test_runtime_artifact_key_canonicalizes_group_coordinate_value():
    numeric_scope = RuntimeExecutionAxisScope.from_raw(
        "A01",
        component=AllComponents.CHANNEL,
        value=2,
    )
    text_scope = RuntimeExecutionAxisScope.from_raw(
        "A01",
        component=AllComponents.CHANNEL,
        value="2",
    )

    numeric_key = ArtifactKey(
        name="measurements",
        artifact_type=MeasurementsArtifactType,
        scope=numeric_scope,
    )
    text_key = ArtifactKey(
        name="measurements",
        artifact_type=MeasurementsArtifactType,
        scope=text_scope,
    )

    assert numeric_scope == text_scope
    assert numeric_key == text_key
    assert hash(numeric_key) == hash(text_key)
    with pytest.raises(TypeError, match="value must be canonical text"):
        RuntimeExecutionAxisScope(
            axis_id="A01",
            component=AllComponents.CHANNEL,
            value=2,
        )


def test_dynamic_output_plan_requires_invocation_group():
    output_plan = ArtifactOutputPlan(
        name="ChannelImage",
        path="/memory/ChannelImage.pkl",
        artifact_type=ImageArtifactType,
        group_keys=(None,),
        group_component=AllComponents.CHANNEL,
        paths_by_group={None: "/memory/ChannelImage.pkl"},
    )

    with pytest.raises(ValueError, match="requires a concrete runtime key"):
        output_plan.for_invocation_group(None)


def test_runtime_value_store_records_and_finds_by_typed_identity():
    store = RuntimeValueStore()
    value = _runtime_value()

    record = store.record(
        value,
        path="/memory/measurements.pkl",
        backend="memory",
    )

    assert store.get(value.key) is record
    assert store.find(name="measurements") == (record,)
    assert store.find(
        name="measurements",
        artifact_type=MeasurementsArtifactType,
        axis_id="A01",
        group_key="DAPI",
        match_group=True,
    ) == (record,)
    assert store.find_by_location(
        path="/memory/measurements.pkl",
        backend="memory",
    ) == (record,)
    assert store.find(group_key="GFP", match_group=True) == ()


def test_runtime_value_store_keeps_fixed_component_artifacts_distinct() -> None:
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
    ).for_group("2")
    store = RuntimeValueStore()
    records = []
    for z_index in ("1", "2"):
        execution_scope = RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=AllComponents.CHANNEL,
            value="2",
            fixed_component_values=((AllComponents.Z_INDEX, z_index),),
        )
        value = RuntimeValue.normalize_for_execution_scope(
            output_plan,
            MeasurementTable(
                name="measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"z_index": z_index, "value": int(z_index)},),
                    fields=(
                        FieldSpec("z_index", str),
                        FieldSpec("value", int),
                    ),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT,
                    "measurements",
                ),
            ),
            execution_scope=execution_scope,
        )
        records.append(
            store.replace(
                value,
                path=output_plan.path,
                backend="memory",
            )
        )

    assert records[0].key != records[1].key
    assert store.values() == tuple(records)
    assert tuple(
        record.key.scope.value_text_for_component(AllComponents.Z_INDEX)
        for record in store.values()
    ) == ("1", "2")

    input_plan = ArtifactInputPlan(
        name="measurements",
        path=output_plan.path,
        artifact_type=MeasurementsArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
    )
    runtime_input = RuntimeArtifactInput(
        edge_plan=_runtime_input_edge(
            input_plan,
            invocation_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
            producer_selection_scope=input_plan.producer_group_scope(),
            component_scopes=(
                ComponentGroupScope.dynamic(AllComponents.CHANNEL),
            ),
            consumer_variable_components=(),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=AllComponents.CHANNEL,
            value="2",
            fixed_component_values=((AllComponents.Z_INDEX, "2"),),
        ),
        backend="memory",
    )

    assert runtime_input.records(store) == (records[1],)


def test_runtime_value_empty_fixed_scope_preserves_ordinary_key_identity() -> None:
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
    ).for_group("2")
    table = MeasurementTable(
        name="measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"value": 1},),
            fields=(FieldSpec("value", int),),
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, "measurements"),
    )

    ordinary = RuntimeValue.normalize(output_plan, table, axis_id="A01")
    exact_empty = RuntimeValue.normalize_for_execution_scope(
        output_plan,
        table,
        execution_scope=RuntimeExecutionAxisScope(axis_id="A01"),
    )

    assert exact_empty.key == ordinary.key
    assert exact_empty.key.scope.fixed_component_values == ()


def test_runtime_value_store_rejects_same_key_different_path():
    store = RuntimeValueStore()
    value = _runtime_value()
    store.record(value, path="/memory/measurements.pkl", backend="memory")

    with pytest.raises(ValueError, match="cannot overwrite"):
        store.record(value, path="/other/measurements.pkl", backend="memory")


def test_runtime_value_store_replace_updates_current_binding_and_keeps_locations():
    store = RuntimeValueStore()
    value = _runtime_value()
    original = store.record(value, path="/memory/measurements.pkl", backend="memory")

    replacement = store.replace(
        value,
        path="/other/measurements.pkl",
        backend="memory",
    )

    assert store.get(value.key) is replacement
    assert store.find_by_location(
        path="/memory/measurements.pkl",
        backend="memory",
    ) == (original,)
    assert store.find_by_location(
        path="/other/measurements.pkl",
        backend="memory",
    ) == (replacement,)


def test_runtime_value_store_find_matching_cache_invalidates_after_replace():
    store = RuntimeValueStore()
    value = _runtime_value()
    original = store.record(value, path="/memory/measurements.pkl", backend="memory")
    query = RuntimeArtifactQuery(
        name="measurements",
        artifact_type=MeasurementsArtifactType,
        axis_id="A01",
        target=RuntimeArtifactDynamicComponentTarget(AllComponents.CHANNEL),
    )

    assert store.find_matching(query) == (original,)

    replacement = store.replace(
        value,
        path="/other/measurements.pkl",
        backend="memory",
    )

    assert store.find_matching(query) == (original, replacement)


def test_runtime_artifact_query_from_input_plan_uses_group_path():
    query = RuntimeArtifactQuery.from_input_plan(
        ArtifactInputPlan(
            name="DNA",
            path="/memory/DNA.pkl",
            artifact_type=ImageArtifactType,
            group_keys=("1", "2"),
            group_component=AllComponents.SITE,
            paths_by_group={"1": "/memory/DNA_s1.pkl", "2": "/memory/DNA_s2.pkl"},
        ),
        axis_id="A01",
        backend="memory",
        group_key="2",
    )

    assert isinstance(query.target, RuntimeArtifactLocationTarget)
    assert query.target.location.path == "/memory/DNA_s2.pkl"
    assert query.target.location.backend == "memory"


def test_runtime_artifact_query_from_dynamic_input_plan_matches_discovered_groups():
    query = RuntimeArtifactQuery.from_input_plan(
        ArtifactInputPlan(
            name="measurements",
            path="/memory/measurements.pkl",
            artifact_type=MeasurementsArtifactType,
            group_component=AllComponents.CHANNEL,
            paths_by_group={None: "/memory/measurements.pkl"},
        ),
        axis_id="A01",
        backend="memory",
    )

    assert isinstance(query.target, RuntimeArtifactDynamicComponentTarget)
    assert query.matches(
        StoredRuntimeValue(
            value=_runtime_value(path="/memory/measurements_w1.pkl"),
            location=RuntimeArtifactLocation(
                path="/memory/measurements_w1.pkl",
                backend="memory",
            ),
        )
    )


def test_runtime_artifact_query_from_dynamic_output_plan_uses_runtime_group_path():
    query = RuntimeArtifactQuery.from_output_plan(
        ArtifactOutputPlan(
            name="RGBImage",
            path="/memory/A01_RGBImage.pkl",
            artifact_type=ImageArtifactType,
            group_component=AllComponents.SITE,
            paths_by_group={None: "/memory/A01_RGBImage.pkl"},
        ),
        axis_id="A01",
        backend="memory",
        group_key="3",
    )

    assert isinstance(query.target, RuntimeArtifactLocationTarget)
    assert query.target.location.path == "/memory/A01_w3_RGBImage.pkl"
    assert query.target.location.backend == "memory"


def test_runtime_artifact_input_projection_selects_same_component_group():
    store = RuntimeValueStore()
    paths = {
        "1": "/memory/image_site_1.pkl",
        "2": "/memory/image_site_2.pkl",
    }
    output_plan = ArtifactOutputPlan(
        name="image",
        path="/memory/image.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.SITE,
        paths_by_group=paths,
    )
    for group_key, value in (("1", 1.0), ("2", 2.0)):
        group_plan = output_plan.for_group(group_key)
        store.record(
            RuntimeValue.normalize(
                group_plan,
                np.full((2, 2), value, dtype=np.float32),
                axis_id="A01",
            ),
            path=group_plan.path,
            backend="memory",
        )

    storage_plan = ArtifactInputPlan(
        name="image",
        path="/memory/image.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.SITE,
        paths_by_group=paths,
    )
    records = RuntimeArtifactInput(
        edge_plan=_runtime_input_edge(
            storage_plan,
            invocation_scope=ComponentGroupScope.dynamic(AllComponents.SITE),
            producer_selection_scope=ComponentGroupScope.dynamic(AllComponents.SITE),
            component_scopes=(ComponentGroupScope.dynamic(AllComponents.SITE),),
            consumer_variable_components=(AllComponents.CHANNEL,),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=AllComponents.SITE,
            value="2",
        ),
        backend="memory",
    ).records(store)

    assert tuple(record.key.scope.value_text for record in records) == ("2",)


def test_runtime_artifact_input_collects_exact_complete_producer_scope():
    store = RuntimeValueStore()
    paths = {
        "1": "/memory/measurements_channel_1.pkl",
        "2": "/memory/measurements_channel_2.pkl",
    }
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        paths_by_group=paths,
    )
    for group_key in ("1", "2"):
        group_plan = output_plan.for_group(group_key)
        store.record(
            RuntimeValue.normalize(
                group_plan,
                MeasurementTable(
                    name="measurements",
                    rows=MeasurementSparseColumnarRows.from_rows(
                        ({"value": float(group_key)},),
                        fields=(FieldSpec("value", float),),
                    ),
                    subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
                ),
                axis_id="A01",
            ),
            path=group_plan.path,
            backend="memory",
        )

    storage_plan = ArtifactInputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        paths_by_group=paths,
    )
    runtime_input = RuntimeArtifactInput(
        edge_plan=_runtime_input_edge(
            storage_plan,
            invocation_scope=ComponentGroupScope(
                ("1",),
                component=AllComponents.CHANNEL,
            ),
            producer_selection_scope=storage_plan.producer_group_scope(),
            component_scopes=(),
            consumer_variable_components=(AllComponents.SITE,),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=AllComponents.CHANNEL,
            value="1",
        ),
        backend="memory",
    )
    records = runtime_input.records(store)

    assert tuple(record.key.scope.value_text for record in records) == ("1", "2")
    composed = runtime_input.composed_value(records)
    assert isinstance(composed, MeasurementTable)
    assert composed.name == "measurements"
    assert composed.subject == MeasurementSubject(MeasurementScope.IMAGE, "Image")
    assert tuple(composed.rows.column_values("value")) == (1.0, 2.0)


def test_runtime_artifact_input_projection_collects_variable_component_groups():
    store = RuntimeValueStore()
    output_plan = ArtifactOutputPlan(
        name="RGBImage",
        path="/memory/RGBImage.pkl",
        artifact_type=ImageArtifactType,
        group_component=AllComponents.SITE,
        paths_by_group={None: "/memory/RGBImage.pkl"},
    )
    for group_key, value in (("1", 1.0), ("2", 2.0)):
        group_plan = output_plan.for_group(group_key)
        store.record(
            RuntimeValue.normalize(
                group_plan,
                np.full((2, 2), value, dtype=np.float32),
                axis_id="A01",
            ),
            path=group_plan.path,
            backend="memory",
        )

    input_plan = ArtifactInputPlan(
        name="RGBImage",
        path="/memory/RGBImage.pkl",
        artifact_type=ImageArtifactType,
        group_component=AllComponents.SITE,
        paths_by_group={None: "/memory/RGBImage.pkl"},
    )
    runtime_input = RuntimeArtifactInput(
        edge_plan=_runtime_input_edge(
            input_plan,
            invocation_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
            producer_selection_scope=input_plan.producer_group_scope(),
            component_scopes=(ComponentGroupScope.dynamic(AllComponents.CHANNEL),),
            consumer_variable_components=(AllComponents.SITE,),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=AllComponents.CHANNEL,
            value="1",
        ),
        backend="memory",
    )
    records = runtime_input.records(store)
    payload = runtime_input.composed_value(records)

    assert tuple(record.key.scope.value_text for record in records) == ("1", "2")
    np.testing.assert_array_equal(
        payload,
        np.stack(
            (
                np.full((2, 2), 1.0, dtype=np.float32),
                np.full((2, 2), 2.0, dtype=np.float32),
            )
        ),
    )


def test_runtime_artifact_input_projection_transposes_producer_stack_axis() -> None:
    store = RuntimeValueStore()
    paths = {
        channel: f"/memory/image_channel_{channel}.pkl" for channel in ("1", "2", "3")
    }
    output_plan = ArtifactOutputPlan(
        name="image",
        path="/memory/image.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2", "3"),
        group_component=AllComponents.CHANNEL,
        variable_components=(AllComponents.SITE,),
        paths_by_group=paths,
    )
    for channel_index, channel in enumerate(("1", "2", "3"), start=1):
        group_plan = output_plan.for_group(channel)
        payload = ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=tuple(f"/source/site_{site}.tif" for site in (1, 2, 3)),
                    component_metadata=tuple(
                        {"site": str(site), "channel": channel} for site in (1, 2, 3)
                    ),
                )
            ),
        ).payload_with(
            np.stack(
                tuple(
                    np.full((2, 2), channel_index * 10 + site, dtype=np.float32)
                    for site in (1, 2, 3)
                )
            ),
            None,
        )
        store.record(
            RuntimeValue.normalize(group_plan, payload, axis_id="A01"),
            path=group_plan.path,
            backend="memory",
        )

    storage_plan = ArtifactInputPlan(
        name="image",
        path="/memory/image.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2", "3"),
        group_component=AllComponents.CHANNEL,
        variable_components=(AllComponents.SITE,),
        paths_by_group=paths,
    )
    runtime_input = RuntimeArtifactInput(
        edge_plan=_runtime_input_edge(
            storage_plan,
            invocation_scope=ComponentGroupScope.dynamic(AllComponents.SITE),
            producer_selection_scope=storage_plan.producer_group_scope(),
            component_scopes=(ComponentGroupScope.dynamic(AllComponents.SITE),),
            consumer_variable_components=(AllComponents.CHANNEL,),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=AllComponents.SITE,
            value="2",
        ),
        backend="memory",
    )

    payload = runtime_input.composed_value(runtime_input.records(store))

    np.testing.assert_array_equal(
        image_payload_data(payload),
        np.stack(
            tuple(
                np.full((2, 2), channel * 10 + 2, dtype=np.float32)
                for channel in (1, 2, 3)
            )
        ),
    )


def test_runtime_artifact_input_projection_ignores_scalar_pixel_contributors() -> None:
    path = "/memory/scalar_image.pkl"
    output_plan = ArtifactOutputPlan(
        name="image",
        path=path,
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.SITE,),
    )
    storage_plan = ArtifactInputPlan(
        name="image",
        path=path,
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.SITE,),
    )
    runtime_input = RuntimeArtifactInput(
        edge_plan=_runtime_input_edge(
            storage_plan,
            invocation_scope=ComponentGroupScope.dynamic(AllComponents.SITE),
            producer_selection_scope=ComponentGroupScope.ungrouped(),
            component_scopes=(ComponentGroupScope.dynamic(AllComponents.SITE),),
            consumer_variable_components=(),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=AllComponents.SITE,
            value="2",
        ),
        backend="memory",
    )
    runtime_planes = SourceImageProvenancePlanes.from_components(
        paths=("/source/site_1.tif", "/source/site_2.tif"),
        component_metadata=({"site": "1"}, {"site": "2"}),
    )

    scalar_store = RuntimeValueStore()
    scalar_payload = ImagePayloadMetadata(
        source_component_metadata={"site": "2"},
        source_image_provenance_planes=runtime_planes.as_contributors(),
    ).payload_with(np.full((2, 2), 20, dtype=np.float32), None)
    scalar_store.record(
        RuntimeValue.normalize(output_plan, scalar_payload, axis_id="A01"),
        path=path,
        backend="memory",
    )

    resolved_scalar = runtime_input.resolve_value(scalar_store)

    np.testing.assert_array_equal(
        image_payload_data(resolved_scalar),
        np.full((2, 2), 20, dtype=np.float32),
    )
    scalar_planes = image_payload_metadata(
        resolved_scalar
    ).source_image_provenance_planes
    assert scalar_planes.runtime_component_metadata == ()
    assert scalar_planes.contributor_count == 2

    stacked_store = RuntimeValueStore()
    stacked_payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=runtime_planes,
    ).payload_with(
        np.stack(
            (
                np.full((2, 2), 10, dtype=np.float32),
                np.full((2, 2), 20, dtype=np.float32),
            )
        ),
        None,
    )
    stacked_store.record(
        RuntimeValue.normalize(output_plan, stacked_payload, axis_id="A01"),
        path=path,
        backend="memory",
    )

    resolved_runtime_plane = runtime_input.resolve_value(stacked_store)

    np.testing.assert_array_equal(
        image_payload_data(resolved_runtime_plane),
        np.full((2, 2), 20, dtype=np.float32),
    )
    assert (
        runtime_planes.runtime_component_metadata == runtime_planes.component_metadata
    )


def test_runtime_artifact_input_projection_collapses_excluded_singleton_axis() -> None:
    store = RuntimeValueStore()
    paths = {site: f"/memory/image_site_{site}.pkl" for site in ("1", "2")}
    output_plan = ArtifactOutputPlan(
        name="image",
        path="/memory/image.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.SITE,
        variable_components=(AllComponents.CHANNEL,),
        paths_by_group=paths,
    )
    for site_index, site in enumerate(("1", "2"), start=1):
        group_plan = output_plan.for_group(site)
        payload = ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=(f"/source/site_{site}.tif",),
                    component_metadata=({"site": site, "channel": "1"},),
                )
            ),
        ).payload_with(np.full((1, 2, 2), site_index, dtype=np.float32), None)
        store.record(
            RuntimeValue.normalize(group_plan, payload, axis_id="A01"),
            path=group_plan.path,
            backend="memory",
        )

    storage_plan = ArtifactInputPlan(
        name="image",
        path="/memory/image.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.SITE,
        variable_components=(AllComponents.CHANNEL,),
        paths_by_group=paths,
    )
    runtime_input = RuntimeArtifactInput(
        edge_plan=_runtime_input_edge(
            storage_plan,
            invocation_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
            producer_selection_scope=storage_plan.producer_group_scope(),
            component_scopes=(
                ComponentGroupScope(
                    ("1",),
                    component=AllComponents.CHANNEL,
                ),
            ),
            consumer_variable_components=(AllComponents.SITE,),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=AllComponents.CHANNEL,
            value="1",
        ),
        backend="memory",
    )

    payload = runtime_input.resolve_value(store)

    assert image_payload_data(payload).shape == (2, 2, 2)
    np.testing.assert_array_equal(
        image_payload_data(payload),
        np.stack(
            (
                np.full((2, 2), 1, dtype=np.float32),
                np.full((2, 2), 2, dtype=np.float32),
            )
        ),
    )
    assert image_payload_metadata(payload).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_runtime_artifact_input_reconstructs_singleton_producer_group_axis() -> None:
    path = "/memory/image_site_1.pkl"
    output_plan = ArtifactOutputPlan(
        name="image",
        path=path,
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.SITE,
        paths_by_group={"1": path},
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue.normalize(
            output_plan.for_group("1"),
            np.full((2, 2), 1, dtype=np.float32),
            axis_id="A01",
        ),
        path=path,
        backend="memory",
    )
    storage_plan = ArtifactInputPlan(
        name="image",
        path=path,
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.SITE,
        paths_by_group={"1": path},
    )
    runtime_input = RuntimeArtifactInput(
        edge_plan=_runtime_input_edge(
            storage_plan,
            invocation_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
            producer_selection_scope=storage_plan.producer_group_scope(),
            component_scopes=(ComponentGroupScope.dynamic(AllComponents.CHANNEL),),
            consumer_variable_components=(AllComponents.SITE,),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=AllComponents.CHANNEL,
            value="1",
        ),
        backend="memory",
    )

    payload = runtime_input.resolve_value(store)

    assert image_payload_data(payload).shape == (1, 2, 2)
    np.testing.assert_array_equal(
        image_payload_data(payload),
        np.full((1, 2, 2), 1, dtype=np.float32),
    )
    assert image_payload_metadata(payload).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_runtime_artifact_input_keeps_singleton_scalar_selection_unstacked() -> None:
    path = "/memory/image_channel_1.pkl"
    output_plan = ArtifactOutputPlan(
        name="image",
        path=path,
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": path},
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue.normalize(
            output_plan.for_group("1"),
            np.full((2, 2), 1, dtype=np.float32),
            axis_id="A01",
        ),
        path=path,
        backend="memory",
    )
    storage_plan = ArtifactInputPlan(
        name="image",
        path=path,
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": path},
    )
    runtime_input = RuntimeArtifactInput(
        edge_plan=_runtime_input_edge(
            storage_plan,
            invocation_scope=ComponentGroupScope.ungrouped(),
            producer_selection_scope=storage_plan.producer_group_scope(),
            component_scopes=(
                ComponentGroupScope(("1",), component=AllComponents.CHANNEL),
            ),
            consumer_variable_components=(AllComponents.SITE,),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
        backend="memory",
    )

    payload = runtime_input.resolve_value(store)

    assert image_payload_data(payload).shape == (2, 2)
    assert image_payload_metadata(payload).plane_axis is None


def test_runtime_artifact_input_projection_selects_compiler_owned_group_for_ungrouped_invocation():
    storage_plan = ArtifactInputPlan(
        name="image",
        path="/memory/image.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
    )
    runtime_input = RuntimeArtifactInput(
        edge_plan=_runtime_input_edge(
            storage_plan,
            invocation_scope=ComponentGroupScope.ungrouped(),
            producer_selection_scope=ComponentGroupScope(
                ("2",),
                component=AllComponents.CHANNEL,
            ),
            component_scopes=(
                ComponentGroupScope(
                    ("2",),
                    component=AllComponents.CHANNEL,
                ),
            ),
            consumer_variable_components=(AllComponents.SITE,),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
        backend="memory",
    )

    assert (
        runtime_input.edge_plan.projection.producer_selection_scope
        == ComponentGroupScope(
            ("2",),
            component=AllComponents.CHANNEL,
        )
    )


def test_runtime_artifact_input_projection_keeps_equal_keys_component_typed():
    store = RuntimeValueStore()
    for component, path, value in (
        (AllComponents.SITE, "/memory/site_image.pkl", 1.0),
        (AllComponents.CHANNEL, "/memory/channel_image.pkl", 2.0),
    ):
        plan = ArtifactOutputPlan(
            name="image",
            path=path,
            artifact_type=ImageArtifactType,
            group_keys=("1",),
            group_component=component,
            paths_by_group={"1": path},
        )
        store.record(
            RuntimeValue.normalize(
                plan,
                np.full((2, 2), value, dtype=np.float32),
                axis_id="A01",
            ),
            path=path,
            backend="memory",
        )

    storage_plan = ArtifactInputPlan(
        name="image",
        path="/memory/site_image.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.SITE,
        paths_by_group={"1": "/memory/site_image.pkl"},
    )
    records = RuntimeArtifactInput(
        edge_plan=_runtime_input_edge(
            storage_plan,
            invocation_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
            producer_selection_scope=storage_plan.producer_group_scope(),
            component_scopes=(ComponentGroupScope.dynamic(AllComponents.CHANNEL),),
            consumer_variable_components=(AllComponents.SITE,),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=AllComponents.CHANNEL,
            value="1",
        ),
        backend="memory",
    ).records(store)

    assert len(records) == 1
    assert records[0].key.scope.component is AllComponents.SITE
    np.testing.assert_array_equal(records[0].value.data, np.full((2, 2), 1.0))


def test_runtime_artifact_input_projection_uses_compiled_group_for_plane_scope():
    store = RuntimeValueStore()
    path = "/memory/rgb_channel_3.pkl"
    output_plan = ArtifactOutputPlan(
        name="RGBImage",
        path=path,
        artifact_type=ImageArtifactType,
        group_keys=("3",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"3": path},
    )
    payload = ImagePayloadMetadata(
        source_component_metadata={"channel": "3"},
    ).payload_with(np.full((2, 2), 3.0, dtype=np.float32), None)
    store.record(
        RuntimeValue.normalize(output_plan.for_group("3"), payload, axis_id="A01"),
        path=path,
        backend="memory",
    )
    storage_plan = ArtifactInputPlan(
        name="RGBImage",
        path=path,
        artifact_type=ImageArtifactType,
        group_keys=("3",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"3": path},
    )
    runtime_input = RuntimeArtifactInput(
        edge_plan=_runtime_input_edge(
            storage_plan,
            invocation_scope=ComponentGroupScope.ungrouped(),
            producer_selection_scope=ComponentGroupScope(
                ("3",),
                component=AllComponents.CHANNEL,
            ),
            component_scopes=(
                ComponentGroupScope(
                    ("3",),
                    component=AllComponents.CHANNEL,
                ),
            ),
            consumer_variable_components=(AllComponents.SITE,),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
        backend="memory",
    )

    assert (
        runtime_input.edge_plan.projection.producer_selection_scope
        == ComponentGroupScope(
            ("3",),
            component=AllComponents.CHANNEL,
        )
    )
    resolved = runtime_input.resolve_value(store)

    np.testing.assert_array_equal(
        image_payload_data(resolved), image_payload_data(payload)
    )


def test_runtime_value_store_observation_cursor_returns_delta_only():
    store = RuntimeValueStore()
    first_value = _runtime_value(name="first", path="/memory/first.pkl")
    first_record = store.record(
        first_value,
        path="/memory/first.pkl",
        backend="memory",
    )
    cursor = store.observation_cursor()
    second_value = _runtime_value(name="second", path="/memory/second.pkl")
    second_record = store.record(
        second_value,
        path="/memory/second.pkl",
        backend="memory",
    )

    assert store.observed_values_after(cursor) == (second_record,)
    assert store.observed_values == (first_record, second_record)


def test_runtime_value_store_observation_cursor_rejects_invalid_index():
    store = RuntimeValueStore()

    with pytest.raises(ValueError, match="beyond the current observation stream"):
        store.observed_values_after(
            store.observation_cursor().__class__(index=1, revision=0)
        )


def test_runtime_value_store_preserves_same_artifact_measurement_subjects():
    store = RuntimeValueStore()
    output_plan = ArtifactOutputPlan(
        name="RelateObjects_measurements",
        path="/memory/RelateObjects_measurements.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    parent_value = RuntimeValue.normalize(
        output_plan,
        MeasurementTable(
            name="RelateObjects_measurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"object_label": 1, "children_count": 2},),
                fields=(
                    FieldSpec("object_label", int),
                    FieldSpec("children_count", int),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, "ParentObjects"),
        ),
        axis_id="A01",
    )
    child_value = RuntimeValue.normalize(
        output_plan,
        MeasurementTable(
            name="RelateObjects_measurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"object_label": 1, "parent_id": 1},),
                fields=(
                    FieldSpec("object_label", int),
                    FieldSpec("parent_id", int),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, "ChildObjects"),
        ),
        axis_id="A01",
    )

    parent_record = store.replace(
        parent_value,
        path="/memory/RelateObjects_measurements.pkl",
        backend="memory",
    )
    child_record = store.replace(
        child_value,
        path="/memory/RelateObjects_measurements.pkl",
        backend="memory",
    )

    assert parent_value.key != child_value.key
    assert store.find(
        name="RelateObjects_measurements",
        artifact_type=MeasurementsArtifactType,
        axis_id="A01",
    ) == (parent_record, child_record)


def test_runtime_value_store_preserves_same_subject_measurement_sources():
    store = RuntimeValueStore()
    name = "MeasureObjectIntensityDistribution_16_measurements"
    path = f"/memory/{name}.pkl"
    output_plan = ArtifactOutputPlan(
        name=name,
        path=path,
        artifact_type=MeasurementsArtifactType,
    )
    subject = MeasurementSubject(MeasurementScope.OBJECT, "Cells")
    records = []
    for source_image_name in ("Syto", "OrigSyto"):
        value = RuntimeValue.normalize(
            output_plan,
            MeasurementTable(
                name=name,
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"object_label": 1, "radial_fraction": 0.5},),
                    fields=(
                        FieldSpec("object_label", int),
                        FieldSpec("radial_fraction", float),
                    ),
                ),
                source_image_name=source_image_name,
                subject=subject,
            ),
            axis_id="A01",
        )
        records.append(
            store.replace(
                value,
                path=path,
                backend="memory",
            )
        )

    assert records[0].key != records[1].key
    assert store.find(
        name=name,
        artifact_type=MeasurementsArtifactType,
        axis_id="A01",
    ) == tuple(records)


def _same_location_measurement_subject_records() -> tuple[
    RuntimeValueStore,
    ArtifactInputPlan,
    dict[str, StoredRuntimeValue],
]:
    name = "MeasureObjectIntensity_6_measurements"
    path = f"/memory/{name}.pkl"
    output_plan = ArtifactOutputPlan(
        name=name,
        path=path,
        artifact_type=MeasurementsArtifactType,
    )
    store = RuntimeValueStore()
    records = {}
    for value, object_name in enumerate(("Nuclei", "Cells", "Cytoplasm"), start=1):
        runtime_value = RuntimeValue.normalize(
            output_plan,
            MeasurementTable(
                name=name,
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"object_label": 1, "mean_intensity": float(value)},),
                    fields=(
                        FieldSpec("object_label", int),
                        FieldSpec("mean_intensity", float),
                    ),
                ),
                subject=MeasurementSubject(MeasurementScope.OBJECT, object_name),
            ),
            axis_id="A01",
        )
        records[object_name] = store.replace(
            runtime_value,
            path=path,
            backend="memory",
        )
    return (
        store,
        ArtifactInputPlan(
            name=name,
            path=path,
            artifact_type=MeasurementsArtifactType,
        ),
        records,
    )


def _ungrouped_runtime_artifact_input(
    storage_plan: ArtifactInputPlan,
    *,
    axis_scope: RuntimeExecutionAxisScope = RuntimeExecutionAxisScope(axis_id="A01"),
) -> RuntimeArtifactInput:
    ungrouped = ComponentGroupScope.ungrouped()
    return RuntimeArtifactInput(
        edge_plan=_runtime_input_edge(
            storage_plan,
            invocation_scope=ungrouped,
            producer_selection_scope=ungrouped,
            component_scopes=(),
            consumer_variable_components=(),
        ),
        axis_scope=axis_scope,
        backend="memory",
    )


def test_runtime_artifact_input_preserves_same_scope_semantic_partitions():
    store, storage_plan, records = _same_location_measurement_subject_records()

    assert _ungrouped_runtime_artifact_input(storage_plan).records(store) == tuple(
        records.values()
    )


def test_runtime_artifact_input_accepts_consumer_coordinate_absent_from_producer():
    store = RuntimeValueStore()
    storage_plan = ArtifactInputPlan(
        name="positions",
        path="/memory/positions.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    output_plan = ArtifactOutputPlan(
        name=storage_plan.name,
        path=storage_plan.path,
        artifact_type=storage_plan.artifact_type,
    )
    value = RuntimeValue.normalize_for_execution_scope(
        output_plan,
        MeasurementTable(
            name=storage_plan.name,
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"position": 1.0},),
                fields=(FieldSpec("position", float),),
            ),
            subject=MeasurementSubject(MeasurementScope.ARTIFACT),
        ),
        execution_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
            fixed_component_values=((AllComponents.TIMEPOINT, "1"),),
        ),
    )
    record = store.replace(value, path=storage_plan.path, backend="memory")
    consumer_scope = RuntimeExecutionAxisScope.from_raw(
        "A01",
        component=None,
        value=None,
        fixed_component_values=(
            (AllComponents.Z_INDEX, "1"),
            (AllComponents.TIMEPOINT, "1"),
        ),
    )

    assert _ungrouped_runtime_artifact_input(
        storage_plan,
        axis_scope=consumer_scope,
    ).records(store) == (record,)


def test_runtime_artifact_input_rejects_conflicting_declared_fixed_coordinate():
    store = RuntimeValueStore()
    storage_plan = ArtifactInputPlan(
        name="positions",
        path="/memory/positions.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    output_plan = ArtifactOutputPlan(
        name=storage_plan.name,
        path=storage_plan.path,
        artifact_type=storage_plan.artifact_type,
    )
    value = RuntimeValue.normalize_for_execution_scope(
        output_plan,
        MeasurementTable(
            name=storage_plan.name,
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"position": 1.0},),
                fields=(FieldSpec("position", float),),
            ),
            subject=MeasurementSubject(MeasurementScope.ARTIFACT),
        ),
        execution_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
            fixed_component_values=((AllComponents.TIMEPOINT, "2"),),
        ),
    )
    store.replace(value, path=storage_plan.path, backend="memory")
    consumer_scope = RuntimeExecutionAxisScope.from_raw(
        "A01",
        component=None,
        value=None,
        fixed_component_values=(
            (AllComponents.Z_INDEX, "1"),
            (AllComponents.TIMEPOINT, "1"),
        ),
    )

    with pytest.raises(RuntimeError, match="Missing RuntimeValueStore record"):
        _ungrouped_runtime_artifact_input(
            storage_plan,
            axis_scope=consumer_scope,
        ).records(store)


def test_runtime_artifact_input_rejects_unprojected_fixed_scope_partitions():
    store = RuntimeValueStore()
    storage_plan = ArtifactInputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    output_plan = ArtifactOutputPlan(
        name=storage_plan.name,
        path=storage_plan.path,
        artifact_type=storage_plan.artifact_type,
    )
    for z_index in ("1", "2"):
        value = RuntimeValue.normalize_for_execution_scope(
            output_plan,
            MeasurementTable(
                name=storage_plan.name,
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"value": float(z_index)},),
                    fields=(FieldSpec("value", float),),
                ),
                subject=MeasurementSubject(MeasurementScope.ARTIFACT),
            ),
            execution_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=None,
                value=None,
                fixed_component_values=((AllComponents.Z_INDEX, z_index),),
            ),
        )
        store.replace(value, path=storage_plan.path, backend="memory")

    with pytest.raises(RuntimeError, match="Ambiguous RuntimeValueStore records"):
        _ungrouped_runtime_artifact_input(storage_plan).records(store)


def test_runtime_value_store_merges_observed_records_from_worker_boundary():
    worker_store = RuntimeValueStore()
    value = _runtime_value()
    record = worker_store.record(
        value,
        path="/memory/measurements.pkl",
        backend="memory",
    )

    parent_store = RuntimeValueStore()
    parent_store.merge_observed_values(worker_store.observed_values)
    parent_store.merge_observed_values(worker_store.observed_values)

    assert parent_store.get(value.key) == record
    assert parent_store.observed_values == (record,)


def test_runtime_measurement_observation_axis_accepts_table_record_once():
    value = _runtime_value()
    record = StoredRuntimeValue(
        value=value,
        location=RuntimeArtifactLocation(
            path="/memory/measurements.pkl",
            backend="memory",
        ),
    )
    axis = RuntimeMeasurementObservationAxis("A01")

    axis.accept_measurement_table(record)

    assert len(axis.measurement_tables) == 1
    scoped_table = axis.measurement_tables[0]
    assert scoped_table.table is value.data
    assert scoped_table.record_identity == record.path
    assert scoped_table.execution_scope == record.key.scope


def test_runtime_value_store_clear_releases_records_and_advances_revision():
    store = RuntimeValueStore()
    value = _runtime_value()
    store.record(value, path="/memory/measurements.pkl", backend="memory")
    revision = store.revision

    store.clear()

    assert store.revision > revision
    assert store.values() == ()
    assert store.observed_values == ()
