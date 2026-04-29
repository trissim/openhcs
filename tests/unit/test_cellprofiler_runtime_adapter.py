import pytest
import numpy as np
from scipy.io import savemat
from types import SimpleNamespace

from benchmark.cellprofiler_compat import (
    CellProfilerModuleContract,
    CellProfilerModuleExecutor,
    CellProfilerRelationshipPayload,
    CellProfilerRuntimeAdapter,
)
from benchmark.cellprofiler_library import get_function
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan, ArtifactSpec
from openhcs.core.config import DtypeConfig
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    ComponentSelector,
    GroupedSourceBindings,
    MetadataExtractionRule,
    MetadataSource,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceBindingRuntimeContext,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import FieldSpec, RuntimeArrayPayload
from openhcs.constants.constants import AllComponents
from openhcs.microscopes.imagexpress import ImageXpressFilenameParser
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


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
MEASURE_COLOCALIZATION = "MeasureColocalization"
MEASURE_IMAGE_INTENSITY = "MeasureImageIntensity"
RELATE_OBJECTS = "RelateObjects"


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


def _pipeline_start_contains_binding(alias):
    return NamedSourceBinding(
        alias=alias,
        selector=SourceSelector(
            filters=(
                SourceFilterClause(
                    SourceFilterSubject.FILE,
                    SourceFilterMatchType.CONTAINS,
                    alias,
                ),
            )
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )


def _source_bound_image_adapter(outputs, images):
    filemanager = FileManagerStub()
    paths = tuple(f"/src/{alias}.tif" for alias in images)
    for alias, image in images.items():
        filemanager.saved[("memory", f"/src/{alias}.tif")] = image
    context = ContextStub(filemanager)
    return CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        artifact_outputs=outputs,
        source_binding_plan=CompiledSourceBindingPlan.from_config(
            StepSourceBindingsConfig(
                groups=(
                    GroupedSourceBindings(
                        bindings=tuple(
                            _pipeline_start_contains_binding(alias)
                            for alias in images
                        )
                    ),
                )
            )
        ),
        source_binding_context=SourceBindingRuntimeContext(
            step_input_files=paths,
            step_input_dir="/src",
            pipeline_input_files=paths,
            pipeline_input_backend="memory",
        ),
        processing_context=context,
        filemanager=filemanager,
    )


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


def test_cellprofiler_adapter_resolves_source_bound_objects():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias=NUCLEI,
                        artifact_kind=ArtifactKind.OBJECT_LABELS,
                    ),
                )
            ),
        )
    )
    adapter, _filemanager = _adapter({}, source_bindings=source_bindings)
    labels = np.ones((3, 3), dtype=np.uint16)

    objects = adapter.resolve_source_objects(NUCLEI, labels)

    assert objects.name == NUCLEI
    np.testing.assert_array_equal(objects.labels, labels)
    assert objects.source_image_name == NUCLEI


def test_cellprofiler_adapter_allows_measurements_for_source_bound_objects():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias=NUCLEI,
                        artifact_kind=ArtifactKind.OBJECT_LABELS,
                    ),
                )
            ),
        )
    )
    adapter, _filemanager = _adapter(
        {
            NUCLEI_MEASUREMENTS: _plan(
                NUCLEI_MEASUREMENTS,
                ArtifactKind.MEASUREMENTS,
            ),
        },
        source_bindings=source_bindings,
    )
    rows = [{"object_id": 1, "area": 42.0}]

    adapter.add_measurements(
        NUCLEI_MEASUREMENTS,
        rows,
        object_name=NUCLEI,
    )

    measurements = adapter.get_measurements(NUCLEI_MEASUREMENTS)
    assert measurements.object_name == NUCLEI


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


def test_cellprofiler_adapter_lists_measurement_tables_for_object_subject():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            NUCLEI_MEASUREMENTS: _plan(
                NUCLEI_MEASUREMENTS,
                ArtifactKind.MEASUREMENTS,
            ),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    adapter.add_objects(NUCLEI, ArrayLike())
    rows = [{"object_id": 1, "area": 42.0}]
    adapter.add_measurements(
        NUCLEI_MEASUREMENTS,
        rows,
        object_name=NUCLEI,
    )
    adapter.add_measurements(MEASUREMENTS, [{"image_area": 100.0}])

    tables = adapter.measurement_tables_for_object(NUCLEI)

    assert [table.name for table in tables] == [NUCLEI_MEASUREMENTS]
    assert tables[0].rows is rows
    assert tables[0].object_name == NUCLEI


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

    assert resolved.shape == (2, 2)
    np.testing.assert_array_equal(resolved, fallback_stack[0])


def test_cellprofiler_adapter_resolves_singleton_step_input_selector_to_natural_2d_view():
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
        step_input_files=("A01_s001_w1_z001_t001.tif",)
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
        [np.full((2, 2), 1.0, dtype=np.float32)]
    )

    resolved = adapter.resolve_source_image(DNA_IMAGE, fallback_stack)

    assert resolved.shape == (2, 2)
    np.testing.assert_array_equal(resolved, fallback_stack[0])


def test_cellprofiler_adapter_resolves_singleton_alias_only_step_input_to_natural_2d_view():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(bindings=(NamedSourceBinding(alias=DNA_IMAGE),)),
        )
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=("A01_s001_w1_z001_t001.tif",)
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
        [np.full((2, 2), 1.0, dtype=np.float32)]
    )

    resolved = adapter.resolve_source_image(DNA_IMAGE, fallback_stack)

    assert resolved.shape == (2, 2)
    np.testing.assert_array_equal(resolved, fallback_stack[0])


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

    assert resolved.shape == expected.shape
    np.testing.assert_array_equal(resolved, expected)
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


def test_cellprofiler_adapter_resolves_metadata_selector_via_compiled_rules(tmp_path):
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
        ),
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FOLDER_NAME,
                pattern=r".*/(?P<folder>plate[A-Z])/Images$",
                filters=(
                    SourceFilterClause(
                        subject=SourceFilterSubject.FILE,
                        match_type=SourceFilterMatchType.CONTAINS_REGEX,
                        value=r"\.tif$",
                    ),
                ),
            ),
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"(?P<folder_illum>plate[A-Z])_Illum(?P<illum>.+)\.mat",
                filters=(
                    SourceFilterClause(
                        subject=SourceFilterSubject.FILE,
                        match_type=SourceFilterMatchType.CONTAINS_REGEX,
                        value=r"_Illum.+\.mat$",
                    ),
                ),
            ),
        ),
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField(
                            alias=DNA_IMAGE,
                            metadata_field="folder",
                        ),
                        SourceBindingMatchField(
                            alias="IllumBlue",
                            metadata_field="folder_illum",
                        ),
                    ),
                ),
            ),
        ),
    )
    filemanager = FileManagerStub()
    expected = np.full((2, 2), 31.0, dtype=np.float32)
    plate_a = tmp_path / "plateA_IllumDAPI.mat"
    plate_b = tmp_path / "plateB_IllumDAPI.mat"
    savemat(plate_a, {"Image": expected})
    savemat(
        plate_b,
        {"Image": np.full((2, 2), 41.0, dtype=np.float32)},
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=("A01_s001_w1_z001_t001.tif",),
        step_input_dir="/plate/plateA/Images",
        pipeline_input_files=(str(plate_a), str(plate_b)),
        pipeline_input_backend="disk",
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

    resolved = adapter.resolve_source_image(
        "IllumBlue",
        np.full((2, 2), 1.0, dtype=np.float32),
    )

    assert resolved.shape == expected.shape
    np.testing.assert_array_equal(resolved, expected)
    assert filemanager.loaded_batches == []


def test_cellprofiler_adapter_resolves_step_input_source_filters_without_metadata():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="rawGFP",
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    subject=SourceFilterSubject.FILE,
                                    match_type=SourceFilterMatchType.CONTAINS,
                                    value="Channel1-",
                                ),
                                SourceFilterClause(
                                    subject=SourceFilterSubject.EXTENSION,
                                    match_type=SourceFilterMatchType.IS_TIF,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "plate-Channel1-A01.tif",
            "plate-Channel2-A01.tif",
        ),
        step_input_dir="/plate/Images",
    )
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        artifact_outputs={},
        source_binding_plan=CompiledSourceBindingPlan.from_config(source_bindings),
        source_binding_context=source_binding_context,
        processing_context=ContextStub(FileManagerStub()),
        filemanager=FileManagerStub(),
    )
    channel_1 = np.full((2, 2), 11.0, dtype=np.float32)
    channel_2 = np.full((2, 2), 22.0, dtype=np.float32)
    image_stack = np.stack((channel_1, channel_2), axis=0)

    resolved = adapter.resolve_source_image("rawGFP", image_stack)

    np.testing.assert_array_equal(resolved, channel_1)


def test_cellprofiler_adapter_resolves_order_based_pipeline_start_match_plan(tmp_path):
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
                    NamedSourceBinding(
                        alias="Actin",
                        selector=SourceSelector(
                            components=(
                                ComponentSelector(AllComponents.CHANNEL, "2"),
                            ),
                        ),
                    ),
                    NamedSourceBinding(
                        alias="IllumBlue",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            metadata=(MetadataSelector("illum", "DAPI"),),
                        ),
                    ),
                ),
            ),
        ),
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"plateA_Illum(?P<illum>.+)_(?P<illum_index>\d+)\.mat",
                filters=(
                    SourceFilterClause(
                        subject=SourceFilterSubject.FILE,
                        match_type=SourceFilterMatchType.CONTAINS_REGEX,
                        value=r"_Illum.+\.mat$",
                    ),
                ),
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    filemanager = FileManagerStub()
    first_mat = tmp_path / "plateA_IllumDAPI_1.mat"
    second_mat = tmp_path / "plateA_IllumDAPI_2.mat"
    savemat(
        first_mat,
        {"Image": np.full((2, 2), 31.0, dtype=np.float32)},
    )
    expected = np.full((2, 2), 41.0, dtype=np.float32)
    savemat(second_mat, {"Image": expected})
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "A01_s002_w1_z001_t001.tif",
            "A01_s002_w2_z001_t001.tif",
        ),
        step_input_dir="/plate/Images",
        pipeline_input_files=(
            "/plate/Images/A01_s001_w1_z001_t001.tif",
            "/plate/Images/A01_s001_w2_z001_t001.tif",
            "/plate/Images/A01_s002_w1_z001_t001.tif",
            "/plate/Images/A01_s002_w2_z001_t001.tif",
            str(first_mat),
            str(second_mat),
        ),
        pipeline_input_backend="disk",
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

    resolved = adapter.resolve_source_image(
        "IllumBlue",
        np.stack(
            [
                np.full((2, 2), 1.0, dtype=np.float32),
                np.full((2, 2), 2.0, dtype=np.float32),
            ]
        ),
    )

    assert resolved.shape == expected.shape
    np.testing.assert_array_equal(resolved, expected)
    assert filemanager.loaded_batches == []


def test_cellprofiler_adapter_uses_virtual_workspace_source_provenance_for_order_matching():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="Sytox",
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    SourceFilterSubject.FILE,
                                    SourceFilterMatchType.CONTAINS,
                                    "_w1",
                                ),
                            ),
                        ),
                    ),
                    NamedSourceBinding(
                        alias="BrightFieldImage",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    SourceFilterSubject.FILE,
                                    SourceFilterMatchType.CONTAINS,
                                    "_w2",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"plate_(?P<well>C\d{2})_w(?P<channel>\d)\.tif",
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    filemanager = FileManagerStub()
    first_brightfield = "/real/plate_C01_w2.tif"
    expected_brightfield = "/real/plate_C20_w2.tif"
    filemanager.saved[("memory", first_brightfield)] = np.full(
        (2, 2),
        12.0,
        dtype=np.float32,
    )
    expected = np.full((2, 2), 22.0, dtype=np.float32)
    filemanager.saved[("memory", expected_brightfield)] = expected
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=("A01_s001_w1_z001_t001.tif",),
        step_input_dir="/workspace",
        step_input_source_paths={
            "A01_s001_w1_z001_t001.tif": "/real/plate_C20_w1.tif",
        },
        pipeline_input_files=(
            "/real/plate_C01_w1.tif",
            first_brightfield,
            "/real/plate_C20_w1.tif",
            expected_brightfield,
        ),
        pipeline_input_backend="memory",
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

    resolved = adapter.resolve_source_image(
        "BrightFieldImage",
        np.full((2, 2), 1.0, dtype=np.float32),
    )

    np.testing.assert_array_equal(resolved, expected)
    assert filemanager.loaded_batches == [
        ((expected_brightfield,), "memory", {}),
    ]


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

    assert measurements.rows == rows
    assert measurements.object_name == NUCLEI
    assert measurements.source_image_name == DNA_IMAGE


def test_cellprofiler_module_executor_measures_each_declared_image_for_single_object():
    dna = np.full((4, 5), 3.0, dtype=np.float32)
    ph3 = np.full((4, 5), 9.0, dtype=np.float32)
    nuclei = np.ones((4, 5), dtype=np.int32)
    adapter = _source_bound_image_adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        },
        {DNA_IMAGE: dna, "PH3": ph3},
    )
    adapter.add_objects(NUCLEI, nuclei)
    executor = _executor(
        MEASURE_OBJECT_INTENSITY,
        (ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        inputs=(
            ArtifactSpec(DNA_IMAGE, ArtifactKind.IMAGE),
            ArtifactSpec("PH3", ArtifactKind.IMAGE),
            ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),
        ),
        runtime_artifact_inputs=(
            ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),
        ),
    )
    seen = []

    def measure(image_arg, *, labels):
        seen.append((float(image_arg.mean()), int(labels.max())))
        return image_arg, [{"mean": float(image_arg.mean()), "label": int(labels.max())}]

    result = executor.run(
        measure,
        np.stack((dna, ph3)),
        cellprofiler_runtime=adapter,
    )
    measurements = adapter.get_measurements(MEASUREMENTS)

    np.testing.assert_array_equal(result, np.stack((dna, ph3)))
    assert seen == [(3.0, 1), (9.0, 1)]
    assert measurements.rows == [
        {"mean": 3.0, "label": 1},
        {"mean": 9.0, "label": 1},
    ]
    assert measurements.object_name == NUCLEI
    assert measurements.source_image_name is None


def test_cellprofiler_module_executor_keeps_coupled_measurement_images_composed():
    dna = np.full((4, 5), 3.0, dtype=np.float32)
    ph3 = np.full((4, 5), 9.0, dtype=np.float32)
    nuclei = np.ones((4, 5), dtype=np.int32)
    adapter = _source_bound_image_adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        },
        {DNA_IMAGE: dna, "PH3": ph3},
    )
    adapter.add_objects(NUCLEI, nuclei)
    executor = _executor(
        MEASURE_COLOCALIZATION,
        (ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        inputs=(
            ArtifactSpec(DNA_IMAGE, ArtifactKind.IMAGE),
            ArtifactSpec("PH3", ArtifactKind.IMAGE),
            ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),
        ),
        runtime_artifact_inputs=(
            ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),
        ),
    )
    seen = []

    def measure(image_arg, *, labels):
        seen.append((image_arg.shape, labels.shape))
        return image_arg[0], [{"object_count": int(labels.max())}]

    result = executor.run(
        measure,
        np.stack((dna, ph3)),
        cellprofiler_runtime=adapter,
    )
    measurements = adapter.get_measurements(MEASUREMENTS)

    np.testing.assert_array_equal(result, np.stack((dna, ph3)))
    assert seen == [((2, 4, 5), (4, 5))]
    assert measurements.rows == [{"object_count": 1}]
    assert measurements.object_name == NUCLEI
    assert measurements.source_image_name is None


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


def test_cellprofiler_module_executor_preserves_main_stack_for_measurements():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            CELLS: _plan(CELLS, ArtifactKind.OBJECT_LABELS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    image = np.stack(
        [
            np.full((4, 5), 3.0, dtype=np.float32),
            np.full((4, 5), 9.0, dtype=np.float32),
        ]
    )
    nuclei = np.ones((4, 5), dtype=np.int32)
    cells = np.full((4, 5), 2, dtype=np.int32)
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
    seen_images = []

    def measure(image_arg, *, labels):
        seen_images.append((image_arg.copy(), labels.copy()))
        return image_arg, [{"object_count": int(labels.max())}]

    result = executor.run(measure, image, cellprofiler_runtime=adapter)
    measurements = adapter.get_measurements(MEASUREMENTS)

    assert len(seen_images) == 2
    for measurement_image, measurement_labels in seen_images:
        assert measurement_image.shape == measurement_labels.shape == (4, 5)
    np.testing.assert_array_equal(result, image)
    assert measurements.rows == [{"object_count": 1}, {"object_count": 2}]


def test_cellprofiler_module_executor_measures_each_declared_image_and_object():
    dna = np.full((4, 5), 3.0, dtype=np.float32)
    ph3 = np.full((4, 5), 9.0, dtype=np.float32)
    nuclei = np.ones((4, 5), dtype=np.int32)
    cells = np.full((4, 5), 2, dtype=np.int32)
    adapter = _source_bound_image_adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            CELLS: _plan(CELLS, ArtifactKind.OBJECT_LABELS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        },
        {DNA_IMAGE: dna, "PH3": ph3},
    )
    adapter.add_objects(NUCLEI, nuclei)
    adapter.add_objects(CELLS, cells)
    executor = _executor(
        MEASURE_OBJECT_INTENSITY,
        (ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        inputs=(
            ArtifactSpec(DNA_IMAGE, ArtifactKind.IMAGE),
            ArtifactSpec("PH3", ArtifactKind.IMAGE),
            ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),
            ArtifactSpec(CELLS, ArtifactKind.OBJECT_LABELS),
        ),
        runtime_artifact_inputs=(
            ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),
            ArtifactSpec(CELLS, ArtifactKind.OBJECT_LABELS),
        ),
    )
    seen = []

    def measure(image_arg, *, labels):
        seen.append((float(image_arg.mean()), int(labels.max())))
        return image_arg, [{"mean": float(image_arg.mean()), "label": int(labels.max())}]

    result = executor.run(
        measure,
        np.stack((dna, ph3)),
        cellprofiler_runtime=adapter,
    )
    measurements = adapter.get_measurements(MEASUREMENTS)

    np.testing.assert_array_equal(result, np.stack((dna, ph3)))
    assert seen == [(3.0, 1), (3.0, 2), (9.0, 1), (9.0, 2)]
    assert measurements.rows == [
        {"mean": 3.0, "label": 1},
        {"mean": 3.0, "label": 2},
        {"mean": 9.0, "label": 1},
        {"mean": 9.0, "label": 2},
    ]
    assert measurements.source_image_name is None


def test_cellprofiler_object_only_executor_does_not_iterate_image_stack():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            CELLS: _plan(CELLS, ArtifactKind.OBJECT_LABELS),
            "Cytoplasm": _plan("Cytoplasm", ArtifactKind.OBJECT_LABELS),
        }
    )
    nuclei = np.ones((4, 5), dtype=np.int32)
    cells = np.full((4, 5), 2, dtype=np.int32)
    adapter.add_objects(NUCLEI, nuclei)
    adapter.add_objects(CELLS, cells)
    executor = _executor(
        IDENTIFY_TERTIARY_OBJECTS,
        (ArtifactSpec("Cytoplasm", ArtifactKind.OBJECT_LABELS),),
        inputs=(
            ArtifactSpec(CELLS, ArtifactKind.OBJECT_LABELS),
            ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),
        ),
        runtime_artifact_inputs=(
            ArtifactSpec(CELLS, ArtifactKind.OBJECT_LABELS),
            ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),
        ),
    )
    seen_images = []

    def identify_tertiary(image_arg, *, primary_labels, secondary_labels):
        seen_images.append(image_arg.shape)
        return image_arg, secondary_labels - primary_labels

    identify_tertiary.__processing_contract__ = ProcessingContract.PURE_2D

    result = executor.run(
        identify_tertiary,
        np.zeros((3, 4, 5), dtype=np.float32),
        cellprofiler_runtime=adapter,
    )
    cytoplasm = adapter.get_objects("Cytoplasm")

    assert seen_images == [(4, 5)]
    assert result.shape == (3, 4, 5)
    assert cytoplasm.labels.shape == (4, 5)


def test_cellprofiler_module_executor_records_relationship_and_measurement_outputs():
    adapter, _filemanager = _adapter(
        {
            CELLS: _plan(CELLS, ArtifactKind.OBJECT_LABELS),
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            PARENT_CHILD: _plan(PARENT_CHILD, ArtifactKind.RELATIONSHIPS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        },
        source_bindings=StepSourceBindingsConfig(),
    )
    image = ArrayLike()
    cells = np.array([[1, 1], [0, 0]], dtype=np.int32)
    nuclei = np.array([[1, 0], [2, 0]], dtype=np.int32)
    adapter.add_objects(CELLS, cells)
    adapter.add_objects(NUCLEI, nuclei)
    executor = _executor(
        RELATE_OBJECTS,
        (
            ArtifactSpec(PARENT_CHILD, ArtifactKind.RELATIONSHIPS),
            ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        ),
        runtime_artifact_inputs=(
            ArtifactSpec(CELLS, ArtifactKind.OBJECT_LABELS),
            ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),
        ),
        inputs=(),
    )

    @special_inputs("parent_labels", "child_labels")
    def relate(image_arg, *, parent_labels, child_labels):
        assert image_arg is image
        assert parent_labels is cells
        assert child_labels is nuclei
        return (
            image_arg,
            CellProfilerRelationshipPayload(parent_ids=(1, 1), child_ids=(1, 2)),
            {"mean_children_per_parent": 2.0},
        )

    result = executor.run(relate, image, cellprofiler_runtime=adapter)
    relationship = adapter.get_relationship(PARENT_CHILD)
    measurements = adapter.get_measurements(MEASUREMENTS)

    assert result is image
    assert relationship.source.name == CELLS
    assert relationship.target.name == NUCLEI
    assert relationship.source_ids == (1, 1)
    assert relationship.target_ids == (1, 2)
    assert measurements.object_name == CELLS
    assert measurements.rows == [
        {"mean_children_per_parent": 2.0},
        {"object_label": 1, "Children_Nuclei_Count": 2},
    ]
    assert adapter.measurement_tables_for_object(CELLS) == (measurements,)
