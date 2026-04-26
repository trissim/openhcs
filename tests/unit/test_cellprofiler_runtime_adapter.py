import pytest

from benchmark.cellprofiler_compat import (
    CellProfilerModuleExecutor,
    CellProfilerRuntimeAdapter,
)
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan, ArtifactSpec
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import FieldSpec, RuntimeArrayPayload


class ArrayLike(RuntimeArrayPayload):
    shape = (2, 2)


class FileManagerStub:
    def __init__(self):
        self.saved = {}

    def save(self, data, path, backend):
        self.saved[(backend, path)] = data


def _plan(name, kind):
    return ArtifactOutputPlan(name=name, path=f"/memory/{name}.pkl", kind=kind)


def _adapter(outputs):
    filemanager = FileManagerStub()
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id="A01",
        artifact_outputs=outputs,
        filemanager=filemanager,
    )
    return adapter, filemanager


def test_cellprofiler_adapter_adds_and_reads_objects_through_runtime_store():
    adapter, filemanager = _adapter(
        {"Nuclei": _plan("Nuclei", ArtifactKind.OBJECT_LABELS)}
    )
    labels = ArrayLike()

    record = adapter.add_objects(
        "Nuclei",
        labels,
        source_image_name="DNA",
        dimensions=("y", "x"),
    )
    objects = adapter.get_objects("Nuclei")

    assert record.value.schema.object_name == "Nuclei"
    assert objects.labels is labels
    assert objects.source_image_name == "DNA"
    assert objects.dimensions == ("y", "x")
    assert filemanager.saved[("memory", "/memory/Nuclei.pkl")] is labels


def test_cellprofiler_adapter_adds_measurements_after_object_reference_exists():
    adapter, _filemanager = _adapter(
        {
            "Nuclei": _plan("Nuclei", ArtifactKind.OBJECT_LABELS),
            "NucleiMeasurements": _plan(
                "NucleiMeasurements",
                ArtifactKind.MEASUREMENTS,
            ),
        }
    )
    adapter.add_objects("Nuclei", ArrayLike())
    rows = [{"object_id": 1, "area": 42.0}]

    adapter.add_measurements(
        "NucleiMeasurements",
        rows,
        object_name="Nuclei",
        fields=(FieldSpec("object_id"), FieldSpec("area")),
        object_id_field="object_id",
    )
    measurements = adapter.get_measurements("NucleiMeasurements")

    assert measurements.rows is rows
    assert measurements.object_name == "Nuclei"
    assert measurements.object_id_field == "object_id"
    assert measurements.fields == (FieldSpec("object_id"), FieldSpec("area"))


def test_cellprofiler_adapter_adds_relationships_after_objects_exist():
    adapter, _filemanager = _adapter(
        {
            "Cells": _plan("Cells", ArtifactKind.OBJECT_LABELS),
            "Nuclei": _plan("Nuclei", ArtifactKind.OBJECT_LABELS),
            "ParentChild": _plan("ParentChild", ArtifactKind.RELATIONSHIPS),
        }
    )
    adapter.add_objects("Cells", ArrayLike())
    adapter.add_objects("Nuclei", ArrayLike())

    adapter.add_relationship(
        "ParentChild",
        parent_object_name="Cells",
        child_object_name="Nuclei",
        parent_ids=[10, 11],
        child_ids=[1, 2],
    )
    relationship = adapter.get_relationship("ParentChild")

    assert relationship.source.name == "Cells"
    assert relationship.target.name == "Nuclei"
    assert relationship.source_ids == [10, 11]
    assert relationship.target_ids == [1, 2]


def test_cellprofiler_adapter_write_requires_compiled_output_plan():
    adapter, _filemanager = _adapter({})

    with pytest.raises(RuntimeError, match="No compiled output plan"):
        adapter.add_objects("Nuclei", ArrayLike())


def test_cellprofiler_adapter_write_rejects_output_kind_mismatch():
    adapter, _filemanager = _adapter(
        {"Nuclei": _plan("Nuclei", ArtifactKind.MEASUREMENTS)}
    )

    with pytest.raises(ValueError, match="expected output kind object_labels"):
        adapter.add_objects("Nuclei", ArrayLike())


def test_cellprofiler_adapter_write_requires_filemanager_vfs_boundary():
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id="A01",
        artifact_outputs={"Nuclei": _plan("Nuclei", ArtifactKind.OBJECT_LABELS)},
    )

    with pytest.raises(RuntimeError, match="filemanager is required for writes"):
        adapter.add_objects("Nuclei", ArrayLike())


def test_cellprofiler_adapter_measurements_require_object_reference():
    adapter, _filemanager = _adapter(
        {
            "NucleiMeasurements": _plan(
                "NucleiMeasurements",
                ArtifactKind.MEASUREMENTS,
            ),
        }
    )

    with pytest.raises(RuntimeError, match="Missing CellProfiler runtime artifact"):
        adapter.add_measurements(
            "NucleiMeasurements",
            [{"object_id": 1}],
            object_name="Nuclei",
        )


def test_cellprofiler_module_executor_records_object_output_through_adapter():
    adapter, _filemanager = _adapter(
        {"Nuclei": _plan("Nuclei", ArtifactKind.OBJECT_LABELS)}
    )
    image = ArrayLike()
    labels = ArrayLike()
    executor = CellProfilerModuleExecutor(
        {
            "module_name": "IdentifyPrimaryObjects",
            "runtime_artifact_inputs": (),
            "external_image_inputs": ("DNA",),
            "outputs": (ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),),
        }
    )

    def identify(image_arg, *, min_diameter):
        assert image_arg is image
        assert min_diameter == 8
        return image_arg, {"object_count": 1}, labels

    result = executor.run(
        identify,
        image,
        cellprofiler_runtime=adapter,
        min_diameter=8,
    )

    assert result is image
    assert adapter.get_objects("Nuclei").labels is labels


def test_cellprofiler_module_executor_reads_objects_for_measurements():
    adapter, _filemanager = _adapter(
        {
            "Nuclei": _plan("Nuclei", ArtifactKind.OBJECT_LABELS),
            "NucleiMeasurements": _plan(
                "NucleiMeasurements",
                ArtifactKind.MEASUREMENTS,
            ),
        }
    )
    image = ArrayLike()
    labels = ArrayLike()
    rows = [{"object_id": 1, "area": 12.0}]
    adapter.add_objects("Nuclei", labels)
    executor = CellProfilerModuleExecutor(
        {
            "module_name": "MeasureObjectSizeShape",
            "runtime_artifact_inputs": (
                ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),
            ),
            "external_image_inputs": ("DNA",),
            "outputs": (
                ArtifactSpec("NucleiMeasurements", ArtifactKind.MEASUREMENTS),
            ),
        }
    )

    def measure(image_arg, *, labels):
        assert image_arg is image
        assert labels is adapter.get_objects("Nuclei").labels
        return image_arg, rows

    executor.run(measure, image, cellprofiler_runtime=adapter)
    measurements = adapter.get_measurements("NucleiMeasurements")

    assert measurements.rows is rows
    assert measurements.object_name == "Nuclei"
    assert measurements.source_image_name == "DNA"


def test_cellprofiler_module_executor_combines_multi_object_measurements():
    adapter, _filemanager = _adapter(
        {
            "Nuclei": _plan("Nuclei", ArtifactKind.OBJECT_LABELS),
            "Cells": _plan("Cells", ArtifactKind.OBJECT_LABELS),
            "Measurements": _plan("Measurements", ArtifactKind.MEASUREMENTS),
        }
    )
    image = ArrayLike()
    nuclei = ArrayLike()
    cells = ArrayLike()
    adapter.add_objects("Nuclei", nuclei)
    adapter.add_objects("Cells", cells)
    executor = CellProfilerModuleExecutor(
        {
            "module_name": "MeasureObjectIntensity",
            "runtime_artifact_inputs": (
                ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),
            ),
            "external_image_inputs": ("DNA",),
            "outputs": (ArtifactSpec("Measurements", ArtifactKind.MEASUREMENTS),),
        }
    )

    def measure(image_arg, *, labels):
        if labels is nuclei:
            return image_arg, [{"object": "Nuclei"}]
        if labels is cells:
            return image_arg, [{"object": "Cells"}]
        raise AssertionError("unexpected labels")

    executor.run(measure, image, cellprofiler_runtime=adapter)
    measurements = adapter.get_measurements("Measurements")

    assert measurements.rows == [{"object": "Nuclei"}, {"object": "Cells"}]
    assert measurements.object_name is None
    assert measurements.source_image_name == "DNA"
