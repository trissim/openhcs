from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import RuntimeArrayPayload, normalize_artifact_value
from openhcs.core.steps.function_artifact_materialization import (
    materialize_artifact_outputs,
)
from openhcs.processing.materialization import CsvOptions, JsonOptions, csv_only


class FileManagerStub:
    def __init__(self):
        self.memory = {}
        self.directories = set()

    def exists(self, path, backend):
        return path in self.memory

    def ensure_directory(self, path, backend):
        self.directories.add((str(path), backend))

    def load(self, path, backend):
        return self.memory[path]


class ArrayLike(RuntimeArrayPayload):
    shape = (2, 2)

    def array_payload_data(self):
        return np.zeros(self.shape, dtype=np.int32)

    def with_data(self, data):
        return data


def _plan(output_plan):
    return SimpleNamespace(
        artifact_outputs={output_plan.name: output_plan},
        streaming_configs=(),
        artifact_analysis_output_dir=Path("/analysis"),
        artifact_images_dir="/images",
        step_name="measure",
        axis_id="A01",
        pipeline_position=7,
        get_paths_for_axis=lambda *_args: [],
        output_dir=Path("/tmp/output"),
        input_dir=Path("/tmp/input"),
        read_backend="memory",
        group_by_value=None,
    )


def _context(filemanager):
    return SimpleNamespace(
        filemanager=filemanager,
        runtime_value_store=RuntimeValueStore(),
    )


def test_materialize_artifact_outputs_uses_runtime_store_payload(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="positions",
        path="/memory/positions.pkl",
        materialization=csv_only(),
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = {"x": "from-vfs"}
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, {"x": "from-runtime"}, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(_spec, data, path, *_args, **_kwargs):
        materialized.append((data, path))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(filemanager, _plan(output_plan), "disk", context)

    assert materialized == [
        ({"x": "from-runtime"}, "/analysis/A01_positions_step7.roi.zip")
    ]


def test_materialize_artifact_outputs_requires_runtime_store_record():
    output_plan = ArtifactOutputPlan(
        name="positions",
        path="/memory/positions.pkl",
        materialization=object(),
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = {"x": 1}
    context = _context(filemanager)

    with pytest.raises(RuntimeError, match="Missing RuntimeValueStore record"):
        materialize_artifact_outputs(filemanager, _plan(output_plan), "disk", context)


def test_materialize_artifact_outputs_does_not_require_vfs_payload_for_store_record(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="positions",
        path="/memory/positions.pkl",
        materialization=csv_only(),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, {"x": 1}, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(_spec, data, path, *_args, **_kwargs):
        materialized.append((data, path))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(filemanager, _plan(output_plan), "disk", context)

    assert materialized == [
        ({"x": 1}, "/analysis/A01_positions_step7.roi.zip")
    ]


def test_materialize_artifact_outputs_defaults_measurements_to_existing_csv_spec(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        kind=ArtifactKind.MEASUREMENTS,
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = [{"object_id": 1, "area": 42}]
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(
            output_plan,
            [{"object_id": 1, "area": 42}],
            axis_id="A01",
        ),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(spec, data, path, *_args, **_kwargs):
        materialized.append((spec, data, path))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(filemanager, _plan(output_plan), "disk", context)

    spec, data, path = materialized[0]
    assert isinstance(spec.outputs[0], CsvOptions)
    assert spec.outputs[0].filename_suffix == ".csv"
    assert data == [{"object_id": 1, "area": 42}]
    assert path == "/analysis/A01_measurements_step7.roi.zip"


def test_materialize_artifact_outputs_uses_actual_group_records(monkeypatch):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/A01_measurements_step7.pkl",
        kind=ArtifactKind.MEASUREMENTS,
        group_keys=("1", "2"),
        paths_by_group={
            "1": "/memory/A01_w1_measurements_step7.pkl",
            "2": "/memory/A01_w2_measurements_step7.pkl",
        },
    )
    group_plan = output_plan.for_group("1")
    filemanager = FileManagerStub()
    filemanager.memory[group_plan.path] = [{"site": "1", "area": 42}]
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(
            group_plan,
            [{"site": "1", "area": 42}],
            axis_id="A01",
        ),
        path=group_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(spec, data, path, *_args, **_kwargs):
        materialized.append((spec, data, path))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(filemanager, _plan(output_plan), "disk", context)

    assert len(materialized) == 1
    spec, data, path = materialized[0]
    assert isinstance(spec.outputs[0], CsvOptions)
    assert data == [{"site": "1", "area": 42}]
    assert path == "/analysis/A01_w1_measurements_step7.roi.zip"


def test_materialize_artifact_outputs_defaults_metadata_to_existing_json_spec(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="metadata",
        path="/memory/metadata.pkl",
        kind=ArtifactKind.METADATA,
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = {"plate": "A"}
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, {"plate": "A"}, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(spec, data, path, *_args, **_kwargs):
        materialized.append((spec, data, path))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(filemanager, _plan(output_plan), "disk", context)

    spec, data, _path = materialized[0]
    assert isinstance(spec.outputs[0], JsonOptions)
    assert data == {"plate": "A"}


def test_materialize_artifact_outputs_skips_special_without_explicit_spec(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="positions",
        path="/memory/positions.pkl",
        kind=ArtifactKind.SPECIAL,
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = {"x": 1}
    context = _context(filemanager)
    materialized = []

    def fake_materialize(*args, **kwargs):
        materialized.append((args, kwargs))

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(filemanager, _plan(output_plan), "disk", context)

    assert materialized == []


def test_materialize_artifact_outputs_fails_for_semantic_kind_without_default():
    output_plan = ArtifactOutputPlan(
        name="labels",
        path="/memory/labels.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    array_like = ArrayLike()
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = array_like
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, array_like, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )

    with pytest.raises(ValueError, match="No default materialization registered"):
        materialize_artifact_outputs(filemanager, _plan(output_plan), "disk", context)
