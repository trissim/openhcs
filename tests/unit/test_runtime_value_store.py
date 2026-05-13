import pytest

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import normalize_artifact_value


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


def test_runtime_value_store_merges_observed_records_from_worker_boundary():
    worker_store = RuntimeValueStore()
    value = _runtime_value()
    record = worker_store.record(
        value,
        path="/memory/measurements.pkl",
        backend="memory",
    )

    parent_store = RuntimeValueStore()
    parent_store.merge_observed_values(worker_store.observed_values())
    parent_store.merge_observed_values(worker_store.observed_values())

    assert parent_store.get(value.key) == record
    assert parent_store.observed_values() == (record,)


def test_runtime_value_store_clear_releases_records_and_advances_revision():
    store = RuntimeValueStore()
    value = _runtime_value()
    store.record(value, path="/memory/measurements.pkl", backend="memory")
    revision = store.revision

    store.clear()

    assert store.revision > revision
    assert store.values() == ()
    assert store.observed_values() == ()
