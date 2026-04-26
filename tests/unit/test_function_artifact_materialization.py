from pathlib import Path
from types import SimpleNamespace

import pytest

from openhcs.core.artifacts import ArtifactOutputPlan
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import normalize_artifact_value
from openhcs.core.steps.function_artifact_materialization import (
    materialize_artifact_outputs,
)


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


def test_materialize_artifact_outputs_loads_vfs_payload_through_store_record(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="positions",
        path="/memory/positions.pkl",
        materialization=object(),
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
        ({"x": "from-vfs"}, "/analysis/A01_positions_step7.roi.zip")
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


def test_materialize_artifact_outputs_requires_vfs_payload_for_store_record():
    output_plan = ArtifactOutputPlan(
        name="positions",
        path="/memory/positions.pkl",
        materialization=object(),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, {"x": 1}, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )

    with pytest.raises(RuntimeError, match="VFS payload is missing"):
        materialize_artifact_outputs(filemanager, _plan(output_plan), "disk", context)
