from collections import Counter

import pytest

from openhcs.core.artifacts import (
    ArtifactScope,
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.runtime_stores import (
    RuntimeArtifactLocation,
    RuntimeArtifactGroupTarget,
    RuntimeArtifactLocationTarget,
    RuntimeArtifactQuery,
    RuntimeValueStore,
    StoredRuntimeValue,
)
from openhcs.core.runtime_equivalence import (
    RuntimeAxisRecordPlaneIdentityResolver,
    RuntimeMeasurementObservationAxis,
)
from openhcs.core.equivalence.relationships import (
    RuntimeRecordPlaneIdentityAuthority,
)
from openhcs.core.runtime_values import MeasurementTable, normalize_artifact_value


def _runtime_value(name="measurements", path="/memory/measurements.pkl"):
    return normalize_artifact_value(
        ArtifactOutputPlan(
            name=name,
            path=path,
            artifact_type=MeasurementsArtifactType,
            group_keys=("DAPI",),
        ),
        [{"object_id": 1}],
        axis_id="A01",
    )


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
        target=RuntimeArtifactGroupTarget("DAPI"),
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
            paths_by_group={"1": "/memory/DNA_s1.pkl", "2": "/memory/DNA_s2.pkl"},
        ),
        axis_id="A01",
        backend="memory",
        group_key="2",
    )

    assert isinstance(query.target, RuntimeArtifactLocationTarget)
    assert query.target.location.path == "/memory/DNA_s2.pkl"
    assert query.target.location.backend == "memory"


def test_measurement_record_groups_ignore_payload_slice_count():
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("1", "3", "2"),
    )

    assert output_plan.runtime_record_group_keys(
        requested_group_key=None,
        scoped_group_key=None,
        slice_count=2,
    ) == ("1", "3", "2")


def test_runtime_artifact_query_from_self_input_plan_requires_single_group():
    query = RuntimeArtifactQuery.from_input_plan(
        ArtifactInputPlan(
            name="Nuclei",
            path="self",
            artifact_type=ObjectLabelsArtifactType,
            group_keys=("DNA",),
        ),
        axis_id="A01",
        backend="memory",
    )

    assert isinstance(query.target, RuntimeArtifactGroupTarget)
    assert query.target.group_key == "DNA"

    with pytest.raises(RuntimeError, match="requires one group key"):
        RuntimeArtifactQuery.from_input_plan(
            ArtifactInputPlan(
                name="Nuclei",
                path="self",
                artifact_type=ObjectLabelsArtifactType,
                group_keys=("DNA", "Mito"),
            ),
            axis_id="A01",
            backend="memory",
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
    parent_value = normalize_artifact_value(
        output_plan,
        MeasurementTable(
            name="RelateObjects_measurements",
            rows=({"object_label": 1, "children_count": 2},),
            object_name="ParentObjects",
        ),
        axis_id="A01",
    )
    child_value = normalize_artifact_value(
        output_plan,
        MeasurementTable(
            name="RelateObjects_measurements",
            rows=({"object_label": 1, "parent_id": 1},),
            object_name="ChildObjects",
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
    resolver = RuntimeAxisRecordPlaneIdentityResolver.from_records((record,))
    seen_aggregate_tables = set()

    assert axis.accept_measurement_table(
        record,
        resolver,
        seen_aggregate_tables,
    ) is not None
    assert axis.accept_measurement_table(
        record,
        resolver,
        seen_aggregate_tables,
    ) is None


def test_runtime_axis_record_plane_identity_prefers_group_scope():
    resolver = RuntimeAxisRecordPlaneIdentityResolver(
        repeated_artifact_counts=Counter({(MeasurementsArtifactType, "measurements"): 2})
    )

    first = resolver.plane_identity_for_record(
        kind=MeasurementsArtifactType,
        name="measurements",
        scope=ArtifactScope(axis_id="A01", group_key="2"),
    )
    second = resolver.plane_identity_for_record(
        kind=MeasurementsArtifactType,
        name="measurements",
        scope=ArtifactScope(axis_id="A01", group_key="1"),
    )
    first_replay = resolver.plane_identity_for_record(
        kind=MeasurementsArtifactType,
        name="measurements",
        scope=ArtifactScope(axis_id="A01", group_key="2"),
    )

    assert first is not None
    assert second is not None
    assert first.authority is RuntimeRecordPlaneIdentityAuthority.FILL_MISSING_ROW_IDENTITY
    assert second.authority is RuntimeRecordPlaneIdentityAuthority.FILL_MISSING_ROW_IDENTITY
    assert first.slice_index != second.slice_index
    assert first_replay == first


def test_runtime_value_store_clear_releases_records_and_advances_revision():
    store = RuntimeValueStore()
    value = _runtime_value()
    store.record(value, path="/memory/measurements.pkl", backend="memory")
    revision = store.revision

    store.clear()

    assert store.revision > revision
    assert store.values() == ()
    assert store.observed_values == ()
