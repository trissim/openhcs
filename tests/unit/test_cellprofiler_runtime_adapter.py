import pytest
import numpy as np
from types import SimpleNamespace

from benchmark.cellprofiler_compat import (
    CellProfilerModuleContract,
    CellProfilerModuleExecutor,
    CellProfilerRuntimeAdapter,
)
from benchmark.cellprofiler_library import get_function
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan, ArtifactSpec
from openhcs.core.config import DtypeConfig
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    ComponentSelector,
    GroupedSourceBindings,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingOrigin,
    SourceBindingRuntimeContext,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import FieldSpec, RuntimeArrayPayload
from openhcs.constants.constants import AllComponents
from openhcs.microscopes.imagexpress import ImageXpressFilenameParser


AXIS_ID = "A01"
DNA_IMAGE = "DNA"
NUCLEI = "Nuclei"
CELLS = "Cells"
PARENT_CHILD = "ParentChild"
MEASUREMENTS = "Measurements"
NUCLEI_MEASUREMENTS = "NucleiMeasurements"
IDENTIFY_PRIMARY_OBJECTS = "IdentifyPrimaryObjects"
IDENTIFY_SECONDARY_OBJECTS = "IdentifySecondaryObjects"
IDENTIFY_TERTIARY_OBJECTS = "IdentifyTertiaryObjects"
MEASURE_OBJECT_INTENSITY = "MeasureObjectIntensity"
MEASURE_OBJECT_NEIGHBORS = "MeasureObjectNeighbors"
MEASURE_OBJECT_SIZE_SHAPE = "MeasureObjectSizeShape"
MEASURE_IMAGE_INTENSITY = "MeasureImageIntensity"


class ArrayLike(RuntimeArrayPayload):
    shape = (2, 2)


class FileManagerStub:
    def __init__(self):
        self.saved = {}
        self.directories = []
        self.loaded_batches = []

    def save(self, data, path, backend):
        self.saved[(backend, path)] = data

    def ensure_directory(self, path, backend):
        self.directories.append((backend, path))

    def load_batch(self, paths, backend, **kwargs):
        self.loaded_batches.append((tuple(paths), backend, dict(kwargs)))
        return [self.saved[(backend, path)] for path in paths]


class ContextStub:
    def __init__(self, filemanager):
        self.filemanager = filemanager
        self.input_dir = "/plate/Images"
        self.global_config = SimpleNamespace(zarr_config=None)
        self.microscope_handler = SimpleNamespace(
            parser=ImageXpressFilenameParser(),
            get_primary_backend=lambda plate_path, filemanager: "memory",
        )


def _plan(name, kind):
    return ArtifactOutputPlan(name=name, path=f"/memory/{name}.pkl", kind=kind)


def _adapter(
    outputs,
    *,
    source_bindings=StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(bindings=(NamedSourceBinding(alias=DNA_IMAGE),)),
        )
    ),
    source_binding_context=SourceBindingRuntimeContext.empty(),
    processing_context=None,
):
    filemanager = FileManagerStub()
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        artifact_outputs=outputs,
        source_binding_plan=CompiledSourceBindingPlan.from_config(source_bindings),
        source_binding_context=source_binding_context,
        processing_context=processing_context,
        filemanager=filemanager,
    )
    return adapter, filemanager


def _executor(
    module_name,
    outputs,
    *,
    runtime_artifact_inputs=(),
    inputs=(ArtifactSpec(DNA_IMAGE, ArtifactKind.IMAGE),),
):
    return CellProfilerModuleExecutor(
        CellProfilerModuleContract(
            module_name=module_name,
            inputs=inputs,
            runtime_artifact_inputs=runtime_artifact_inputs,
            outputs=outputs,
        )
    )


def test_cellprofiler_adapter_adds_and_reads_objects_through_runtime_store():
    adapter, filemanager = _adapter(
        {NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)}
    )
    labels = ArrayLike()

    record = adapter.add_objects(
        NUCLEI,
        labels,
        source_image_name=DNA_IMAGE,
        dimensions=("y", "x"),
    )
    objects = adapter.get_objects(NUCLEI)

    assert record.value.schema.object_name == NUCLEI
    assert objects.labels is labels
    assert objects.source_image_name == DNA_IMAGE
    assert objects.dimensions == ("y", "x")
    assert filemanager.saved[("memory", "/memory/Nuclei.pkl")] is labels


def test_cellprofiler_adapter_adds_measurements_after_object_reference_exists():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            NUCLEI_MEASUREMENTS: _plan(
                NUCLEI_MEASUREMENTS,
                ArtifactKind.MEASUREMENTS,
            ),
        }
    )
    adapter.add_objects(NUCLEI, ArrayLike())
    rows = [{"object_id": 1, "area": 42.0}]

    adapter.add_measurements(
        NUCLEI_MEASUREMENTS,
        rows,
        object_name=NUCLEI,
        fields=(FieldSpec("object_id"), FieldSpec("area")),
        object_id_field="object_id",
    )
    measurements = adapter.get_measurements(NUCLEI_MEASUREMENTS)

    assert measurements.rows is rows
    assert measurements.object_name == NUCLEI
    assert measurements.object_id_field == "object_id"
    assert measurements.fields == (FieldSpec("object_id"), FieldSpec("area"))


def test_cellprofiler_adapter_adds_relationships_after_objects_exist():
    adapter, _filemanager = _adapter(
        {
            CELLS: _plan(CELLS, ArtifactKind.OBJECT_LABELS),
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            PARENT_CHILD: _plan(PARENT_CHILD, ArtifactKind.RELATIONSHIPS),
        }
    )
    adapter.add_objects(CELLS, ArrayLike())
    adapter.add_objects(NUCLEI, ArrayLike())

    adapter.add_relationship(
        PARENT_CHILD,
        parent_object_name=CELLS,
        child_object_name=NUCLEI,
        parent_ids=[10, 11],
        child_ids=[1, 2],
    )
    relationship = adapter.get_relationship(PARENT_CHILD)

    assert relationship.source.name == CELLS
    assert relationship.target.name == NUCLEI
    assert relationship.source_ids == [10, 11]
    assert relationship.target_ids == [1, 2]


def test_cellprofiler_adapter_write_requires_compiled_output_plan():
    adapter, _filemanager = _adapter({})

    with pytest.raises(RuntimeError, match="No compiled output plan"):
        adapter.add_objects(NUCLEI, ArrayLike())


def test_cellprofiler_adapter_write_rejects_output_kind_mismatch():
    adapter, _filemanager = _adapter(
        {NUCLEI: _plan(NUCLEI, ArtifactKind.MEASUREMENTS)}
    )

    with pytest.raises(ValueError, match="expected output kind object_labels"):
        adapter.add_objects(NUCLEI, ArrayLike())


def test_cellprofiler_adapter_write_requires_filemanager_vfs_boundary():
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        artifact_outputs={NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)},
    )

    with pytest.raises(RuntimeError, match="filemanager is required for writes"):
        adapter.add_objects(NUCLEI, ArrayLike())


def test_cellprofiler_adapter_measurements_require_object_reference():
    adapter, _filemanager = _adapter(
        {
            NUCLEI_MEASUREMENTS: _plan(
                NUCLEI_MEASUREMENTS,
                ArtifactKind.MEASUREMENTS,
            ),
        }
    )

    with pytest.raises(RuntimeError, match="Missing CellProfiler runtime artifact"):
        adapter.add_measurements(
            NUCLEI_MEASUREMENTS,
            [{"object_id": 1}],
            object_name=NUCLEI,
        )


def test_cellprofiler_adapter_resolves_step_input_channel_selector_from_current_stack():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias=DNA_IMAGE,
                        selector=SourceSelector(
                            components=(
                                ComponentSelector(AllComponents.CHANNEL, "1"),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
        )
    )
    filemanager = FileManagerStub()
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        artifact_outputs={},
        source_binding_plan=CompiledSourceBindingPlan.from_config(source_bindings),
        source_binding_context=source_binding_context,
        processing_context=ContextStub(filemanager),
        filemanager=filemanager,
    )
    fallback_stack = np.stack(
        [
            np.full((2, 2), 1.0, dtype=np.float32),
            np.full((2, 2), 2.0, dtype=np.float32),
        ]
    )

    resolved = adapter.resolve_source_image(DNA_IMAGE, fallback_stack)

    assert resolved.shape == (1, 2, 2)
    np.testing.assert_array_equal(resolved[0], fallback_stack[0])


def test_cellprofiler_adapter_resolves_pipeline_start_component_selector_with_inherited_scope():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="Actin",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            components=(
                                ComponentSelector(AllComponents.CHANNEL, "2"),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
        ),
        pipeline_input_files=(
            "/plate/Images/A01_s001_w1_z001_t001.tif",
            "/plate/Images/A01_s001_w2_z001_t001.tif",
            "/plate/Images/A01_s002_w1_z001_t001.tif",
            "/plate/Images/A01_s002_w2_z001_t001.tif",
        ),
        pipeline_input_backend="memory",
    )
    filemanager = FileManagerStub()
    expected = np.full((2, 2), 12.0, dtype=np.float32)
    filemanager.saved[("memory", "/plate/Images/A01_s001_w1_z001_t001.tif")] = np.full(
        (2, 2),
        11.0,
        dtype=np.float32,
    )
    filemanager.saved[("memory", "/plate/Images/A01_s001_w2_z001_t001.tif")] = expected
    filemanager.saved[("memory", "/plate/Images/A01_s002_w1_z001_t001.tif")] = np.full(
        (2, 2),
        21.0,
        dtype=np.float32,
    )
    filemanager.saved[("memory", "/plate/Images/A01_s002_w2_z001_t001.tif")] = np.full(
        (2, 2),
        22.0,
        dtype=np.float32,
    )
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        artifact_outputs={},
        source_binding_plan=CompiledSourceBindingPlan.from_config(source_bindings),
        source_binding_context=source_binding_context,
        processing_context=ContextStub(filemanager),
        filemanager=filemanager,
    )
    fallback_stack = np.stack(
        [
            np.full((2, 2), 1.0, dtype=np.float32),
            np.full((2, 2), 2.0, dtype=np.float32),
        ]
    )

    resolved = adapter.resolve_source_image("Actin", fallback_stack)

    assert resolved.shape == (1, 2, 2)
    np.testing.assert_array_equal(resolved[0], expected)
    assert filemanager.loaded_batches == [
        (
            ("/plate/Images/A01_s001_w2_z001_t001.tif",),
            "memory",
            {},
        )
    ]


def test_cellprofiler_adapter_rejects_metadata_selector_fields_not_exposed_by_parser():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="IllumBlue",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            metadata=(MetadataSelector("illum", "DAPI"),),
                        ),
                    ),
                ),
            ),
        )
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=("A01_s001_w1_z001_t001.tif",),
        pipeline_input_files=("/plate/Images/A01_s001_w1_z001_t001.tif",),
        pipeline_input_backend="memory",
    )
    filemanager = FileManagerStub()
    filemanager.saved[("memory", "/plate/Images/A01_s001_w1_z001_t001.tif")] = np.full(
        (2, 2),
        11.0,
        dtype=np.float32,
    )
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        artifact_outputs={},
        source_binding_plan=CompiledSourceBindingPlan.from_config(source_bindings),
        source_binding_context=source_binding_context,
        processing_context=ContextStub(filemanager),
        filemanager=filemanager,
    )

    with pytest.raises(NotImplementedError, match="filename parser exposes"):
        adapter.resolve_source_image(
            "IllumBlue",
            np.full((2, 2), 1.0, dtype=np.float32),
        )


def test_cellprofiler_module_executor_records_object_output_through_adapter():
    adapter, _filemanager = _adapter(
        {NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)}
    )
    image = ArrayLike()
    labels = ArrayLike()
    executor = _executor(
        IDENTIFY_PRIMARY_OBJECTS,
        (ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
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
    assert adapter.get_objects(NUCLEI).labels is labels


@pytest.mark.parametrize(
    "module_name",
    [
        IDENTIFY_PRIMARY_OBJECTS,
        IDENTIFY_SECONDARY_OBJECTS,
        IDENTIFY_TERTIARY_OBJECTS,
        MEASURE_OBJECT_INTENSITY,
        MEASURE_OBJECT_NEIGHBORS,
        MEASURE_OBJECT_SIZE_SHAPE,
        MEASURE_IMAGE_INTENSITY,
    ],
)
def test_core_cellprofiler_functions_resolve_with_numpy_memory_contract(module_name):
    func = get_function(module_name)

    assert callable(func)
    assert func.input_memory_type == "numpy"
    assert func.output_memory_type == "numpy"


def test_cellprofiler_module_executor_runs_resolved_identify_primary_objects():
    adapter, filemanager = _adapter(
        {NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)}
    )
    image = np.zeros((64, 64), dtype=np.float32)
    image[18:28, 18:28] = 0.95
    image[40:50, 40:50] = 0.85
    executor = _executor(
        IDENTIFY_PRIMARY_OBJECTS,
        (ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
    )
    identify_primary_objects = get_function(IDENTIFY_PRIMARY_OBJECTS)

    result = executor.run(
        identify_primary_objects,
        image,
        cellprofiler_runtime=adapter,
        dtype_config=DtypeConfig(),
        min_diameter=4,
        max_diameter=20,
        exclude_border_objects=False,
    )

    objects = adapter.get_objects(NUCLEI)
    assert result.shape == image.shape
    assert objects.labels.shape == image.shape
    assert objects.labels.max() == 2
    assert filemanager.saved[("memory", "/memory/Nuclei.pkl")].shape == image.shape


def test_cellprofiler_module_executor_reads_objects_for_measurements():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            NUCLEI_MEASUREMENTS: _plan(
                NUCLEI_MEASUREMENTS,
                ArtifactKind.MEASUREMENTS,
            ),
        }
    )
    image = ArrayLike()
    labels = ArrayLike()
    rows = [{"object_id": 1, "area": 12.0}]
    adapter.add_objects(NUCLEI, labels)
    executor = _executor(
        MEASURE_OBJECT_SIZE_SHAPE,
        (ArtifactSpec(NUCLEI_MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        runtime_artifact_inputs=(
            ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),
        ),
    )

    def measure(image_arg, *, labels):
        assert image_arg is image
        assert labels is adapter.get_objects(NUCLEI).labels
        return image_arg, rows

    executor.run(measure, image, cellprofiler_runtime=adapter)
    measurements = adapter.get_measurements(NUCLEI_MEASUREMENTS)

    assert measurements.rows is rows
    assert measurements.object_name == NUCLEI
    assert measurements.source_image_name == DNA_IMAGE


def test_cellprofiler_module_executor_combines_multi_object_measurements():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            CELLS: _plan(CELLS, ArtifactKind.OBJECT_LABELS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    image = ArrayLike()
    nuclei = ArrayLike()
    cells = ArrayLike()
    adapter.add_objects(NUCLEI, nuclei)
    adapter.add_objects(CELLS, cells)
    executor = _executor(
        MEASURE_OBJECT_INTENSITY,
        (ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        runtime_artifact_inputs=(
            ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),
            ArtifactSpec(CELLS, ArtifactKind.OBJECT_LABELS),
        ),
    )

    def measure(image_arg, *, labels):
        if labels is nuclei:
            return image_arg, [{"object": NUCLEI}]
        if labels is cells:
            return image_arg, [{"object": CELLS}]
        raise AssertionError("unexpected labels")

    executor.run(measure, image, cellprofiler_runtime=adapter)
    measurements = adapter.get_measurements(MEASUREMENTS)

    assert measurements.rows == [{"object": NUCLEI}, {"object": CELLS}]
    assert measurements.object_name is None
    assert measurements.source_image_name == DNA_IMAGE
