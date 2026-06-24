import pytest

from openhcs.core.artifacts import ArtifactInputPlan, ArtifactKind, ArtifactOutputPlan
from openhcs.core.runtime_stores import (
    RuntimeArtifactGroupTarget,
    RuntimeArtifactLocationTarget,
    RuntimeArtifactQuery,
    RuntimeValueStore,
)
from openhcs.core.runtime_values import MeasurementTable, normalize_artifact_value


def _runtime_value(name="measurements", path="/memory/measurements.pkl"):
    return normalize_artifact_value(
        ArtifactOutputPlan(
            name=name,
            path=path,
            kind=ArtifactKind.MEASUREMENTS,
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
        kind=ArtifactKind.MEASUREMENTS,
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
        kind=ArtifactKind.MEASUREMENTS,
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
            kind=ArtifactKind.IMAGE,
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


def test_runtime_artifact_query_from_self_input_plan_requires_single_group():
    query = RuntimeArtifactQuery.from_input_plan(
        ArtifactInputPlan(
            name="Nuclei",
            path="self",
            kind=ArtifactKind.OBJECT_LABELS,
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
                kind=ArtifactKind.OBJECT_LABELS,
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
        kind=ArtifactKind.MEASUREMENTS,
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
        kind=ArtifactKind.MEASUREMENTS,
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


def test_runtime_value_store_clear_releases_records_and_advances_revision():
    store = RuntimeValueStore()
    value = _runtime_value()
    store.record(value, path="/memory/measurements.pkl", backend="memory")
    revision = store.revision

    store.clear()

    assert store.revision > revision
    assert store.values() == ()
    assert store.observed_values == ()
