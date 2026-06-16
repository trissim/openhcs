from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import pytest
import numpy as np
from scipy.io import savemat
from nominal_refactor_advisor.descriptor_algebra import AliasProperty

from openhcs.interop.cellprofiler.runtime import (
    CellProfilerRelationshipPayload,
    CellProfilerRuntimeAdapter,
)
from openhcs.interop.cellprofiler.runtime.adapter import (
    CellProfilerImageNumberResolver,
    CurrentSourceObjectLabelPayloadProjection,
    ObjectLabelMeasurementSliceBatchResolver,
    ObjectLabelMeasurementSliceRequest,
    ObjectMeasurementTableIndex,
    ParsedSourceCandidate,
    ParsedSourceCandidateCollection,
    SpatialGridValueAuthority,
    SourceBindingAxisPlaneResolution,
    SourceImageSetIdentityCompatibility,
)
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerModuleExecutor,
    ConcatenatedMeasurementColumnarRows,
    CurrentObjectShapeFeatureVectorSourceStrategy,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.core.runtime_artifact_queries import (
    MeasurementTableAxisQuery,
    measurement_rows,
    measurement_values_for_label_slices,
    measurement_values_for_feature,
)
from benchmark.cellprofiler_library import get_function
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactKind,
    ArtifactOutputPlan,
    ArtifactSpec,
)
from openhcs.core.config import DtypeConfig
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.pipeline_image_schema import SOURCE_IMAGE_TYPE_METADATA_FIELD
from openhcs.core.source_image_semantics import SourceImagePayloadSemantics
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
from openhcs.core.source_matching import SourceImageSetIdentity
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_invocation import RuntimeSliceAlignedValues
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    ObjectLabelDomainScope,
    RelationshipSemantics,
    RuntimePlaneAxis,
    RuntimePlaneProjection,
)
from openhcs.core.runtime_values import (
    FieldSpec,
    ImagePayloadMetadata,
    MeasurementTable,
    ObjectRelationship,
    ObjectLabelPayload,
    ObjectLabelSet,
    ColumnarRows,
    RuntimeArrayPayload,
    SpatialGrid,
    image_payload_data,
    image_payload_metadata,
    image_payload_mask,
    image_payload_with_context,
    normalize_artifact_value,
    object_label_dense_array,
)
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


def _measurement_rows_for_assertion(measurements):
    rows = []
    for row in measurements.rows:
        normalized = dict(row)
        normalized.setdefault("slice_index", 0)
        rows.append(normalized)
    return rows


def test_parsed_source_candidate_collection_deduplicates_virtual_and_resolved_paths():
    metadata = {
        "well": "A01",
        "site": "1",
        "channel": "1",
        "timepoint": "0",
    }
    candidates = ParsedSourceCandidateCollection(
        (
            ParsedSourceCandidate(
                path="A01_s001_w1_z001_t000.tif",
                resolved_path="/source/frame0.tif",
                filename="frame0.tif",
                metadata=metadata,
            ),
            ParsedSourceCandidate(
                path="/source/frame0.tif",
                resolved_path="/source/frame0.tif",
                filename="frame0.tif",
                metadata=metadata,
            ),
        )
    )

    assert candidates.deduplicated() == (candidates.candidates[0],)


def test_parsed_source_candidate_collection_preserves_distinct_resolved_paths():
    metadata = {
        "well": "A01",
        "site": "1",
        "channel": "1",
        "timepoint": "0",
    }
    candidates = ParsedSourceCandidateCollection(
        (
            ParsedSourceCandidate(
                path="A01_s001_w1_z001_t000.tif",
                resolved_path="/source/channel1.tif",
                filename="channel1.tif",
                metadata=metadata,
            ),
            ParsedSourceCandidate(
                path="A01_s001_w2_z001_t000.tif",
                resolved_path="/source/channel2.tif",
                filename="channel2.tif",
                metadata=metadata,
            ),
        )
    )

    assert candidates.deduplicated() == candidates.candidates


def test_source_identity_compatibility_matches_partial_channel_scope():
    record_identity = SourceImageSetIdentity((("channel", "1"),))
    current_identity = SourceImageSetIdentity(
        (
            ("site", "001"),
            ("channel", "1"),
            ("z", "001"),
            ("timepoint", "001"),
        )
    )
    other_current = SourceImageSetIdentity(
        (
            ("site", "001"),
            ("channel", "2"),
            ("z", "001"),
            ("timepoint", "001"),
        )
    )

    assert SourceImageSetIdentityCompatibility(
        record_identity,
        current_identity,
    ).matches()
    assert not SourceImageSetIdentityCompatibility(
        record_identity,
        other_current,
    ).matches()


def test_image_number_resolver_uses_group_start_for_multi_source_payloads():
    candidates = (
        ParsedSourceCandidate(
            path="A01_s001_w1_z001_t001.tif",
            resolved_path="/source/site1.tif",
            filename="site1.tif",
            metadata={"well": "A01", "site": "1", "channel": "1", "timepoint": "1"},
        ),
        ParsedSourceCandidate(
            path="A01_s002_w1_z001_t001.tif",
            resolved_path="/source/site2.tif",
            filename="site2.tif",
            metadata={"well": "A01", "site": "2", "channel": "1", "timepoint": "1"},
        ),
    )
    adapter = SimpleNamespace(
        source_binding_context=SimpleNamespace(
            pipeline_input_files=tuple(candidate.path for candidate in candidates)
        ),
        source_candidates=lambda _paths: candidates,
        cellprofiler_source_order_path=lambda path: path,
    )

    resolver = CellProfilerImageNumberResolver.for_adapter(adapter)

    assert resolver.image_number_for_paths(("/source/site2.tif", "/source/site1.tif")) == 2
    assert (
        resolver.image_number_start_for_paths(
            ("/source/site2.tif", "/source/site1.tif")
        )
        == 1
    )


def test_image_number_resolver_groups_channels_into_image_sets():
    candidates = (
        ParsedSourceCandidate(
            path="A01_s001_w1_z001_t001.tif",
            resolved_path="/source/Ch1_1.tif",
            filename="Ch1_1.tif",
            metadata={"well": "A01", "site": "1", "channel": "1", "timepoint": "1"},
        ),
        ParsedSourceCandidate(
            path="A01_s001_w2_z001_t001.tif",
            resolved_path="/source/Ch6_1.tif",
            filename="Ch6_1.tif",
            metadata={"well": "A01", "site": "1", "channel": "2", "timepoint": "1"},
        ),
        ParsedSourceCandidate(
            path="A01_s002_w1_z001_t001.tif",
            resolved_path="/source/Ch1_2.tif",
            filename="Ch1_2.tif",
            metadata={"well": "A01", "site": "2", "channel": "1", "timepoint": "1"},
        ),
    )

    image_numbers = CellProfilerImageNumberResolver.image_numbers_by_set(candidates)

    assert tuple(image_numbers.values()) == (1, 2)


def test_axis_image_number_start_matches_declared_axis_component_only():
    context = SourceBindingRuntimeContext(
        pipeline_input_files=(
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
            "A01_s002_w1_z001_t001.tif",
            "A01_s002_w2_z001_t001.tif",
        ),
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {
                "well": "A01",
                "site": "1",
                "channel": "1",
                "timepoint": "1",
            },
            "A01_s001_w2_z001_t001.tif": {
                "well": "A01",
                "site": "1",
                "channel": "2",
                "timepoint": "1",
            },
            "A01_s002_w1_z001_t001.tif": {
                "well": "A01",
                "site": "2",
                "channel": "1",
                "timepoint": "1",
            },
            "A01_s002_w2_z001_t001.tif": {
                "well": "A01",
                "site": "2",
                "channel": "2",
                "timepoint": "1",
            },
        },
    )
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id="2",
        source_binding_context=context,
        axis_component="site",
        processing_context=SimpleNamespace(
            microscope_handler=SimpleNamespace(
                parser=SimpleNamespace(parse_filename=lambda _name: {})
            )
        ),
    )

    assert adapter.cellprofiler_axis_image_number_start() == 2


def test_axis_image_number_start_matches_cellprofiler_metadata_spelling():
    context = SourceBindingRuntimeContext(
        pipeline_input_files=(
            "plate1_A14_site1_Ch1.tif",
            "plate1_A14_site1_Ch4.tif",
            "plate1_A14_site2_Ch1.tif",
            "plate1_A14_site2_Ch4.tif",
        ),
        source_metadata_by_path={
            "plate1_A14_site1_Ch1.tif": {
                "Well": "A14",
                "Site": "1",
                "ChannelNumber": "1",
            },
            "plate1_A14_site1_Ch4.tif": {
                "Well": "A14",
                "Site": "1",
                "ChannelNumber": "4",
            },
            "plate1_A14_site2_Ch1.tif": {
                "Well": "A14",
                "Site": "2",
                "ChannelNumber": "1",
            },
            "plate1_A14_site2_Ch4.tif": {
                "Well": "A14",
                "Site": "2",
                "ChannelNumber": "4",
            },
        },
    )
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id="2",
        source_binding_context=context,
        axis_component="site",
        processing_context=SimpleNamespace(
            microscope_handler=SimpleNamespace(
                parser=SimpleNamespace(parse_filename=lambda _name: {})
            )
        ),
    )

    assert adapter.cellprofiler_axis_image_number_start() == 2


def test_axis_image_number_start_uses_explicit_axis_component_value():
    context = SourceBindingRuntimeContext(
        pipeline_input_files=(
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
            "A01_s002_w1_z001_t001.tif",
            "A01_s002_w2_z001_t001.tif",
        ),
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {
                "well": "A01",
                "site": "1",
                "channel": "1",
            },
            "A01_s001_w2_z001_t001.tif": {
                "well": "A01",
                "site": "1",
                "channel": "2",
            },
            "A01_s002_w1_z001_t001.tif": {
                "well": "A01",
                "site": "2",
                "channel": "1",
            },
            "A01_s002_w2_z001_t001.tif": {
                "well": "A01",
                "site": "2",
                "channel": "2",
            },
        },
    )
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id="A01",
        group_key="default",
        source_binding_context=context,
        axis_component="site",
        axis_component_value="2",
        processing_context=SimpleNamespace(
            microscope_handler=SimpleNamespace(
                parser=SimpleNamespace(parse_filename=lambda _name: {})
            )
        ),
    )

    assert adapter.cellprofiler_axis_image_number_start() == 2


@dataclass(frozen=True, slots=True)
class SimpleColumnarRows(ColumnarRows):
    data: Mapping[str, tuple[object, ...]]

    columns: ClassVar[AliasProperty[Mapping[str, tuple[object, ...]]]] = (
        AliasProperty("data")
    )
MEASURE_COLOCALIZATION = "MeasureColocalization"
MEASURE_IMAGE_INTENSITY = "MeasureImageIntensity"
RELATE_OBJECTS = "RelateObjects"
CALCULATE_MATH = "CalculateMath"


class ArrayLike(RuntimeArrayPayload):
    shape = (2, 2)

    @property
    def ndim(self):
        return len(self.shape)

    def array_payload_data(self):
        return np.zeros(self.shape, dtype=np.int32)

    def with_data(self, data):
        return data


def declared_processing_contract(contract: ProcessingContract):
    def decorator(func):
        func.__processing_contract__ = contract
        return func

    return decorator


class FileManagerStub:
    def __init__(self):
        self.saved = {}
        self.directories = []
        self.loaded_batches = []
        self.deleted = []

    def save(self, data, path, backend):
        self.saved[(backend, path)] = data

    def exists(self, path, backend):
        return (backend, path) in self.saved

    def delete(self, path, backend):
        self.deleted.append((backend, path))
        self.saved.pop((backend, path))

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
        ModuleArtifactContract(
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
    saved_payload = filemanager.saved[("memory", "/memory/Nuclei.pkl")]
    assert saved_payload.labels is labels
    assert saved_payload.source_image_names == (DNA_IMAGE,)


def test_cellprofiler_adapter_contextualizes_source_aligned_object_label_stack():
    adapter, filemanager = _adapter(
        {NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)}
    )
    source_image = image_payload_with_context(
        np.zeros((2, 5, 6), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            channel_source_paths=(
                "/src/A01_s001_w1_z001_t001.tif",
                "/src/A01_s002_w1_z001_t001.tif",
            ),
            channel_source_component_metadata=(
                {"well": "A01", "site": "1", "channel": "1"},
                {"well": "A01", "site": "2", "channel": "1"},
            ),
        ),
    )
    labels = ObjectLabelPayload(
        labels=np.ones((2, 5, 6), dtype=np.int32),
    )

    record = adapter.add_objects(
        NUCLEI,
        labels,
        source_image_name=DNA_IMAGE,
        source_image_payload=source_image,
    )

    assert isinstance(record.value.data, ObjectLabelPayload)
    assert record.value.data.channel_source_paths == (
        "/src/A01_s001_w1_z001_t001.tif",
        "/src/A01_s002_w1_z001_t001.tif",
    )
    assert tuple(
        dict(metadata)
        for metadata in record.value.data.channel_source_component_metadata
        if metadata is not None
    ) == (
        {"well": "A01", "site": "1", "channel": "1"},
        {"well": "A01", "site": "2", "channel": "1"},
    )
    saved_payload = filemanager.saved[("memory", "/memory/Nuclei.pkl")]
    assert saved_payload.channel_source_paths == record.value.data.channel_source_paths


def test_cellprofiler_adapter_rejects_unaddressable_source_aligned_label_stack():
    adapter, _filemanager = _adapter(
        {NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)}
    )
    source_image = image_payload_with_context(
        np.zeros((2, 5, 6), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            channel_source_paths=(None, None),
        ),
    )
    labels = ObjectLabelPayload(
        labels=np.ones((2, 5, 6), dtype=np.int32),
    )

    with pytest.raises(ValueError, match="neither source_path"):
        adapter.add_objects(
            NUCLEI,
            labels,
            source_image_name=DNA_IMAGE,
            source_image_payload=source_image,
        )


def test_cellprofiler_adapter_does_not_cache_current_image_object_selection():
    store = RuntimeValueStore()
    outputs = {NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)}
    source_paths = (
        "/src/A01_s001_w1_z001_t001.tif",
        "/src/A01_s002_w1_z001_t001.tif",
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=source_paths,
        step_input_dir="/src",
        pipeline_input_files=source_paths,
    )
    filemanager = FileManagerStub()
    processing_context = ContextStub(filemanager)

    for group_key, source_path, label_value in (
        ("1", source_paths[0], 1),
        ("2", source_paths[1], 2),
    ):
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            artifact_outputs=outputs,
            source_binding_context=source_binding_context,
            group_key=group_key,
            processing_context=processing_context,
            filemanager=filemanager,
        )
        producer.add_objects(
            NUCLEI,
            ObjectLabelSet(
                name=NUCLEI,
                labels=np.full((2, 2), label_value, dtype=np.int32),
                source_path=source_path,
            ),
        )

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs=outputs,
        source_binding_context=source_binding_context,
        processing_context=processing_context,
        filemanager=filemanager,
    )
    current_images = tuple(
        image_payload_with_context(
            np.zeros((2, 2), dtype=np.float32),
            metadata=ImagePayloadMetadata(source_path=source_path),
        )
        for source_path in source_paths
    )

    first = consumer.get_objects(NUCLEI, current_image=current_images[0])
    second = consumer.get_objects(NUCLEI, current_image=current_images[1])

    np.testing.assert_array_equal(first.labels, np.full((2, 2), 1, dtype=np.int32))
    np.testing.assert_array_equal(second.labels, np.full((2, 2), 2, dtype=np.int32))


def test_cellprofiler_adapter_does_not_source_scope_default_image_records():
    store = RuntimeValueStore()
    outputs = {DNA_IMAGE: _plan(DNA_IMAGE, ArtifactKind.IMAGE)}
    filemanager = FileManagerStub()

    for group_key, source_path, value in (
        ("1", "/src/A01_s001_w1.tif", 1.0),
        ("2", "/src/A01_s002_w1.tif", 2.0),
    ):
        source_metadata = {"site": group_key}
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            artifact_outputs=outputs,
            group_key=group_key,
            filemanager=filemanager,
        )
        producer.add_image(
            DNA_IMAGE,
            image_payload_with_context(
                np.full((2, 2), value, dtype=np.float32),
                metadata=ImagePayloadMetadata(
                    source_path=source_path,
                    source_component_metadata=source_metadata,
                ),
            ),
        )

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs=outputs,
        source_binding_context=SourceBindingRuntimeContext(
            step_input_files=("/src/A01_s003_w1.tif",),
        ),
        filemanager=filemanager,
    )
    current_image = image_payload_with_context(
        np.zeros((2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/src/A01_s003_w1.tif",
            source_component_metadata={"site": "3"},
        ),
    )

    image = consumer.get_image(DNA_IMAGE, current_image=current_image)

    assert image.data.shape == (2, 2, 2)
    np.testing.assert_array_equal(image.data[0], np.full((2, 2), 1.0))
    np.testing.assert_array_equal(image.data[1], np.full((2, 2), 2.0))


def test_cellprofiler_adapter_stacks_declared_default_image_input_runtime_groups():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    context = ContextStub(filemanager)
    image_name = "Hoechst"
    image_path = "/memory/Hoechst.pkl"
    source_paths = (
        "/plate/Images/A01_s001_w1_z001_t001.tif",
        "/plate/Images/A01_s002_w1_z001_t001.tif",
    )
    output_plan = ArtifactOutputPlan(
        name=image_name,
        path=image_path,
        kind=ArtifactKind.IMAGE,
        group_keys=(None,),
    )

    for group_key, source_path, value in (
        ("1", source_paths[0], 1.0),
        ("2", source_paths[1], 2.0),
    ):
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            group_key=group_key,
            artifact_outputs={image_name: output_plan},
            filemanager=filemanager,
        )
        producer.add_image(
            image_name,
            image_payload_with_context(
                np.full((2, 2), value, dtype=np.float32),
                metadata=ImagePayloadMetadata(
                    source_path=source_path,
                    source_component_metadata={"site": group_key},
                ),
            ),
        )

    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=source_paths,
        current_step_input_files=source_paths,
        pipeline_input_files=source_paths,
    )
    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        artifact_inputs={
            image_name: ArtifactInputPlan(
                name=image_name,
                path=image_path,
                kind=ArtifactKind.IMAGE,
                group_keys=(None,),
            )
        },
        source_binding_context=source_binding_context,
        processing_context=context,
        filemanager=filemanager,
    )
    current_image = image_payload_with_context(
        np.zeros((2, 2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            channel_source_paths=source_paths,
            channel_source_component_metadata=(
                {"site": "1"},
                {"site": "2"},
            ),
        ),
    )

    image = consumer.get_image(image_name, current_image=current_image)

    assert image.data.shape == (2, 2, 2)
    np.testing.assert_array_equal(image.data[0], np.full((2, 2), 1.0))
    np.testing.assert_array_equal(image.data[1], np.full((2, 2), 2.0))


def test_cellprofiler_adapter_keeps_multisource_current_image_grouped_when_files_are_narrow():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    context = ContextStub(filemanager)
    image_name = "Hoechst"
    image_path = "/memory/Hoechst.pkl"
    source_paths = (
        "/plate/Images/A01_s1_w1.tif",
        "/plate/Images/A01_s2_w1.tif",
    )
    source_metadata_by_path = {
        source_paths[0]: {"Site": "1", "ChannelNumber": "1"},
        source_paths[1]: {"Site": "2", "ChannelNumber": "1"},
    }
    output_plan = ArtifactOutputPlan(
        name=image_name,
        path=image_path,
        kind=ArtifactKind.IMAGE,
        group_keys=(None,),
    )

    for group_key, source_path, value in (
        ("1", source_paths[0], 1.0),
        ("2", source_paths[1], 2.0),
    ):
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            group_key=group_key,
            artifact_outputs={image_name: output_plan},
            filemanager=filemanager,
        )
        producer.add_image(
            image_name,
            image_payload_with_context(
                np.full((2, 2), value, dtype=np.float32),
                metadata=ImagePayloadMetadata(
                    source_path=source_path,
                    source_component_metadata=source_metadata_by_path[source_path],
                ),
            ),
        )

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        artifact_inputs={
            image_name: ArtifactInputPlan(
                name=image_name,
                path=image_path,
                kind=ArtifactKind.IMAGE,
                group_keys=(None,),
            )
        },
        source_binding_context=SourceBindingRuntimeContext(
            step_input_files=source_paths,
            current_step_input_files=(source_paths[0],),
            pipeline_input_files=source_paths,
            source_metadata_by_path=source_metadata_by_path,
        ),
        processing_context=context,
        filemanager=filemanager,
    )
    current_image = image_payload_with_context(
        np.zeros((2, 2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(channel_source_paths=source_paths),
    )

    image = consumer.get_image(image_name, current_image=current_image)
    metadata = image_payload_metadata(image.data)

    assert image.data.shape == (2, 2, 2)
    np.testing.assert_array_equal(image.data[0], np.full((2, 2), 1.0))
    np.testing.assert_array_equal(image.data[1], np.full((2, 2), 2.0))
    assert metadata.channel_source_paths == source_paths


def test_cellprofiler_adapter_stacks_declared_image_input_for_pattern_group():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    context = ContextStub(filemanager)
    image_name = "Hoechst"
    image_path = "/memory/Hoechst.pkl"
    source_paths = (
        "/plate/Images/A01_s001_w1_z001_t001.tif",
        "/plate/Images/A01_s002_w1_z001_t001.tif",
    )
    output_plan = ArtifactOutputPlan(
        name=image_name,
        path=image_path,
        kind=ArtifactKind.IMAGE,
        group_keys=(None,),
    )

    for group_key, source_path, value in (
        ("1", source_paths[0], 1.0),
        ("2", source_paths[1], 2.0),
    ):
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            group_key=group_key,
            artifact_outputs={image_name: output_plan},
            filemanager=filemanager,
        )
        producer.add_image(
            image_name,
            image_payload_with_context(
                np.full((2, 2), value, dtype=np.float32),
                metadata=ImagePayloadMetadata(
                    source_path=source_path,
                    source_component_metadata={"site": group_key},
                ),
            ),
        )

    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=source_paths,
        current_step_input_files=source_paths,
        pipeline_input_files=source_paths,
    )
    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="A01_s{iii}_w1_z001_t001.tif",
        artifact_inputs={
            image_name: ArtifactInputPlan(
                name=image_name,
                path=image_path,
                kind=ArtifactKind.IMAGE,
                group_keys=(None,),
            )
        },
        source_binding_context=source_binding_context,
        processing_context=context,
        filemanager=filemanager,
    )
    current_image = image_payload_with_context(
        np.zeros((2, 2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            channel_source_paths=source_paths,
            channel_source_component_metadata=(
                {"site": "1"},
                {"site": "2"},
            ),
        ),
    )

    image = consumer.get_image(image_name, current_image=current_image)

    assert image.data.shape == (2, 2, 2)
    np.testing.assert_array_equal(image.data[0], np.full((2, 2), 1.0))
    np.testing.assert_array_equal(image.data[1], np.full((2, 2), 2.0))


def test_cellprofiler_adapter_projects_source_bound_runtime_image_to_group_plane():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    image_name = "MaskedMito"
    image_path = "/memory/MaskedMito.pkl"
    source_paths = (
        "/plate/Images/A01_s1_w5.tif",
        "/plate/Images/A01_s2_w5.tif",
    )
    source_metadata = (
        {"site": "1", "channel": "5"},
        {"site": "2", "channel": "5"},
    )
    producer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs={
            image_name: ArtifactOutputPlan(
                name=image_name,
                path=image_path,
                kind=ArtifactKind.IMAGE,
                group_keys=(None,),
                paths_by_group={None: image_path},
            )
        },
        filemanager=filemanager,
    )
    producer.add_image(
        image_name,
        image_payload_with_context(
            np.stack(
                (
                    np.full((2, 3), 1.0, dtype=np.float32),
                    np.full((2, 3), 2.0, dtype=np.float32),
                ),
                axis=0,
            ),
            metadata=ImagePayloadMetadata(
                channel_source_paths=source_paths,
                channel_source_component_metadata=source_metadata,
            ),
        ),
    )

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="2",
        axis_component="site",
        axis_component_value="2",
        plane_projection=RuntimePlaneProjection.group(1),
        artifact_inputs={
            image_name: ArtifactInputPlan(
                name=image_name,
                path=image_path,
                kind=ArtifactKind.IMAGE,
                group_keys=("2",),
                paths_by_group={"2": image_path},
            )
        },
        source_binding_context=SourceBindingRuntimeContext(
            source_metadata_by_path=dict(
                zip(source_paths, source_metadata, strict=True)
            ),
        ),
        filemanager=filemanager,
    )

    image = consumer.get_image(image_name)
    metadata = image_payload_metadata(image.data)

    assert image_payload_data(image.data).shape == (2, 3)
    np.testing.assert_array_equal(
        image_payload_data(image.data),
        np.full((2, 3), 2.0, dtype=np.float32),
    )
    assert metadata.source_path == source_paths[1]
    assert metadata.source_component_metadata == source_metadata[1]


def test_cellprofiler_adapter_deduplicates_grouped_runtime_image_input_locations():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    image_name = "MaskedMito"
    image_path = "/memory/MaskedMito.pkl"
    source_paths = (
        "/plate/Images/A01_s1_w5.tif",
        "/plate/Images/A01_s2_w5.tif",
    )
    source_metadata = (
        {"site": "1", "channel": "5"},
        {"site": "2", "channel": "5"},
    )
    producer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs={
            image_name: ArtifactOutputPlan(
                name=image_name,
                path=image_path,
                kind=ArtifactKind.IMAGE,
                group_keys=(None,),
                paths_by_group={None: image_path},
            )
        },
        filemanager=filemanager,
    )
    producer.add_image(
        image_name,
        image_payload_with_context(
            np.stack(
                (
                    np.full((2, 3), 1.0, dtype=np.float32),
                    np.full((2, 3), 2.0, dtype=np.float32),
                ),
                axis=0,
            ),
            metadata=ImagePayloadMetadata(
                channel_source_paths=source_paths,
                channel_source_component_metadata=source_metadata,
            ),
        ),
    )
    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        axis_component="site",
        axis_component_value="2",
        plane_projection=RuntimePlaneProjection.group(1),
        artifact_inputs={
            image_name: ArtifactInputPlan(
                name=image_name,
                path=image_path,
                kind=ArtifactKind.IMAGE,
                group_keys=("1", "2"),
                paths_by_group={"1": image_path, "2": image_path},
            )
        },
        source_binding_context=SourceBindingRuntimeContext(
            source_metadata_by_path=dict(
                zip(source_paths, source_metadata, strict=True)
            ),
        ),
        filemanager=filemanager,
    )

    image = consumer.get_image(image_name)
    metadata = image_payload_metadata(image.data)

    assert image_payload_data(image.data).shape == (2, 3)
    np.testing.assert_array_equal(
        image_payload_data(image.data),
        np.full((2, 3), 2.0, dtype=np.float32),
    )
    assert metadata.source_path == source_paths[1]
    assert metadata.source_component_metadata == source_metadata[1]


def test_cellprofiler_adapter_does_not_project_channel_stack_for_site_group():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    image_name = "Corrected"
    image_path = "/memory/Corrected.pkl"
    source_paths = (
        "/plate/Images/A01_s2_w1.tif",
        "/plate/Images/A01_s2_w2.tif",
    )
    source_metadata = (
        {"site": "2", "channel": "1"},
        {"site": "2", "channel": "2"},
    )
    producer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs={
            image_name: ArtifactOutputPlan(
                name=image_name,
                path=image_path,
                kind=ArtifactKind.IMAGE,
                group_keys=(None,),
                paths_by_group={None: image_path},
            )
        },
        filemanager=filemanager,
    )
    producer.add_image(
        image_name,
        image_payload_with_context(
            np.stack(
                (
                    np.full((2, 3), 1.0, dtype=np.float32),
                    np.full((2, 3), 2.0, dtype=np.float32),
                ),
                axis=0,
            ),
            metadata=ImagePayloadMetadata(
                channel_source_paths=source_paths,
                channel_source_component_metadata=source_metadata,
            ),
        ),
    )
    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="2",
        axis_component="site",
        axis_component_value="2",
        plane_projection=RuntimePlaneProjection.group(1),
        artifact_inputs={
            image_name: ArtifactInputPlan(
                name=image_name,
                path=image_path,
                kind=ArtifactKind.IMAGE,
                group_keys=("2",),
                paths_by_group={"2": image_path},
            )
        },
        filemanager=filemanager,
    )

    image = consumer.get_image(image_name)
    metadata = image_payload_metadata(image.data)

    assert image_payload_data(image.data).shape == (2, 2, 3)
    np.testing.assert_array_equal(
        image_payload_data(image.data)[0],
        np.full((2, 3), 1.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        image_payload_data(image.data)[1],
        np.full((2, 3), 2.0, dtype=np.float32),
    )
    assert metadata.channel_source_paths == source_paths


def test_cellprofiler_adapter_projects_stack_with_current_image_plane_context():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    image_name = "Masked"
    image_path = "/memory/Masked.pkl"
    producer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs={
            image_name: ArtifactOutputPlan(
                name=image_name,
                path=image_path,
                kind=ArtifactKind.IMAGE,
                group_keys=(None,),
                paths_by_group={None: image_path},
            )
        },
        filemanager=filemanager,
    )
    producer.add_image(
        image_name,
        image_payload_with_context(
            np.stack(
                (
                    np.full((2, 3), 1.0, dtype=np.float32),
                    np.full((2, 3), 2.0, dtype=np.float32),
                ),
                axis=0,
            ),
            metadata=ImagePayloadMetadata(
                source_path="/plate/Images/A01_s1_w1.tif",
                source_component_metadata={"site": "1", "channel": "1"},
            ),
        ),
    )
    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="2",
        axis_component="site",
        axis_component_value="2",
        plane_projection=RuntimePlaneProjection.group(1),
        artifact_inputs={
            image_name: ArtifactInputPlan(
                name=image_name,
                path=image_path,
                kind=ArtifactKind.IMAGE,
                group_keys=("2",),
                paths_by_group={"2": image_path},
            )
        },
        filemanager=filemanager,
    )
    current_image = image_payload_with_context(
        np.zeros((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/plate/Images/A01_s2_w1.tif",
            source_component_metadata={"site": "2", "channel": "1"},
        ),
    )

    image = consumer.get_image(image_name, current_image=current_image)
    metadata = image_payload_metadata(image.data)

    assert image_payload_data(image.data).shape == (2, 3)
    np.testing.assert_array_equal(
        image_payload_data(image.data),
        np.full((2, 3), 2.0, dtype=np.float32),
    )
    assert metadata.source_path == "/plate/Images/A01_s2_w1.tif"
    assert metadata.source_component_metadata == {"site": "2", "channel": "1"}


def test_cellprofiler_adapter_keeps_template_scoped_object_records_grouped():
    store = RuntimeValueStore()
    outputs = {NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)}
    filemanager = FileManagerStub()

    for group_key, source_path, value in (
        ("1", "/src/A01_s001_w2.tif", 1),
        ("2", "/src/A01_s002_w2.tif", 2),
    ):
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            artifact_outputs=outputs,
            group_key=group_key,
            filemanager=filemanager,
        )
        producer.add_objects(
            NUCLEI,
            ObjectLabelSet(
                name=NUCLEI,
                labels=np.full((2, 2), value, dtype=np.int32),
                source_path=source_path,
            ),
        )

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs=outputs,
        source_binding_context=SourceBindingRuntimeContext(
            step_input_files=("/src/A01_s{iii}_w1.tif",),
        ),
        filemanager=filemanager,
    )
    current_image = image_payload_with_context(
        np.zeros((2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_path="/src/A01_s{iii}_w1.tif"),
    )

    objects = consumer.get_objects(NUCLEI, current_image=current_image)

    assert objects.labels.shape == (2, 2, 2)
    np.testing.assert_array_equal(objects.labels[0], np.full((2, 2), 1))
    np.testing.assert_array_equal(objects.labels[1], np.full((2, 2), 2))


def test_cellprofiler_adapter_reads_object_labels_across_producer_groups():
    store = RuntimeValueStore()
    outputs = {NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)}
    filemanager = FileManagerStub()

    for group_key, label_value in (("1", 1), ("2", 2)):
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            artifact_outputs=outputs,
            group_key=group_key,
            filemanager=filemanager,
        )
        producer.add_objects(
            NUCLEI,
            ObjectLabelSet(
                name=NUCLEI,
                labels=np.full((2, 2), label_value, dtype=np.int32),
                source_image_names=(f"source_{group_key}",),
            ),
        )

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs=outputs,
        filemanager=filemanager,
    )

    objects = consumer.get_objects_across_groups(NUCLEI)

    assert objects.labels.shape == (2, 2, 2)
    assert objects.domain_scope is ObjectLabelDomainScope.PLANE
    assert objects.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert objects.source_image_names == ("source_1", "source_2")
    np.testing.assert_array_equal(objects.labels[0], np.full((2, 2), 1, dtype=np.int32))
    np.testing.assert_array_equal(objects.labels[1], np.full((2, 2), 2, dtype=np.int32))


def test_cellprofiler_adapter_projects_stacked_objects_by_current_source_plane():
    adapter, _filemanager = _adapter(
        {NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)}
    )
    source_paths = (
        "/plate/Images/A01_s1_w1.tif",
        "/plate/Images/A01_s2_w1.tif",
    )
    labels = np.stack(
        (
            np.full((2, 2), 1, dtype=np.int32),
            np.full((2, 2), 2, dtype=np.int32),
        ),
        axis=0,
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            labels=labels,
            channel_source_paths=source_paths,
            channel_source_component_metadata=({"site": "1"}, {"site": "2"}),
            source_image_names=(DNA_IMAGE,),
            domain_scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (2,)),
        ),
    )
    current_image = image_payload_with_context(
        np.zeros((2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path=source_paths[1],
            source_component_metadata={"site": "2"},
        ),
    )

    objects = adapter.get_objects(NUCLEI, current_image=current_image)

    np.testing.assert_array_equal(objects.labels, labels[1])
    assert objects.declared_object_ids == (2,)
    assert objects.source_image_names == (DNA_IMAGE,)


def test_cellprofiler_adapter_trusts_nominal_object_label_domain_over_source_fallback():
    adapter, _filemanager = _adapter(
        {
            DNA_IMAGE: _plan(DNA_IMAGE, ArtifactKind.IMAGE),
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            CELLS: _plan(CELLS, ArtifactKind.OBJECT_LABELS),
        }
    )
    source_image = image_payload_with_context(
        np.zeros((4, 4), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            spatial_origin_yx=(3, 5),
            source_spatial_shape_yx=(16, 16),
        ),
    )
    adapter.add_image(DNA_IMAGE, source_image)

    raw_labels = np.ones((2, 2), dtype=np.int32)
    adapter.add_objects(NUCLEI, raw_labels, source_image_name=DNA_IMAGE)
    raw_objects = adapter.get_objects(NUCLEI)
    assert raw_objects.spatial_origin_yx == (3, 5)
    assert raw_objects.source_spatial_shape_yx == (16, 16)

    transformed_labels = ObjectLabelPayload(
        labels=np.ones((1, 1), dtype=np.int32),
    )
    adapter.add_objects(CELLS, transformed_labels, source_image_name=DNA_IMAGE)
    transformed_objects = adapter.get_objects(CELLS)
    assert transformed_objects.spatial_origin_yx is None
    assert transformed_objects.source_spatial_shape_yx is None


def test_cellprofiler_adapter_requires_declared_source_image_coordinate_for_labels():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    source_plan = ArtifactOutputPlan(
        name=DNA_IMAGE,
        path="/memory/DNA.pkl",
        kind=ArtifactKind.IMAGE,
    )
    producer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs={DNA_IMAGE: source_plan},
        filemanager=filemanager,
    )
    producer.add_image(DNA_IMAGE, np.zeros((2, 2), dtype=np.float32))
    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_inputs={
            DNA_IMAGE: ArtifactInputPlan(
                name=DNA_IMAGE,
                path=source_plan.path,
                kind=ArtifactKind.IMAGE,
            ),
        },
        artifact_outputs={NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)},
        filemanager=filemanager,
    )

    with pytest.raises(RuntimeError, match="require source coordinate metadata"):
        consumer.add_objects(
            NUCLEI,
            np.ones((2, 2), dtype=np.int32),
            source_image_name=DNA_IMAGE,
        )


def test_cellprofiler_adapter_inherits_declared_source_image_coordinate_for_labels():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    source_plan = ArtifactOutputPlan(
        name=DNA_IMAGE,
        path="/memory/DNA.pkl",
        kind=ArtifactKind.IMAGE,
    )
    source_image = image_payload_with_context(
        np.zeros((2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/plate/A01_s001_w1_z001_t001.tif",
            source_component_metadata={"well": "A01", "site": "001", "channel": "1"},
        ),
    )
    producer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs={DNA_IMAGE: source_plan},
        filemanager=filemanager,
    )
    producer.add_image(DNA_IMAGE, source_image)
    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_inputs={
            DNA_IMAGE: ArtifactInputPlan(
                name=DNA_IMAGE,
                path=source_plan.path,
                kind=ArtifactKind.IMAGE,
            ),
        },
        artifact_outputs={NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)},
        filemanager=filemanager,
    )

    consumer.add_objects(
        NUCLEI,
        np.ones((2, 2), dtype=np.int32),
        source_image_name=DNA_IMAGE,
    )
    objects = consumer.get_objects(NUCLEI)

    assert objects.source_path == "/plate/A01_s001_w1_z001_t001.tif"
    assert objects.source_component_metadata == {
        "well": "A01",
        "site": "001",
        "channel": "1",
    }


def test_cellprofiler_adapter_records_dense_object_label_slice_lists_as_stacks():
    adapter, filemanager = _adapter(
        {NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)}
    )
    labels = [
        np.full((3, 4), 1, dtype=np.int32),
        np.full((3, 4), 2, dtype=np.int32),
    ]

    adapter.add_objects(NUCLEI, labels)
    objects = adapter.get_objects(NUCLEI)

    assert objects.labels.shape == (2, 3, 4)
    np.testing.assert_array_equal(objects.labels, np.stack(labels))
    np.testing.assert_array_equal(
        filemanager.saved[("memory", "/memory/Nuclei.pkl")],
        np.stack(labels),
    )


def test_cellprofiler_adapter_records_dense_object_label_volume_lists_as_stacks():
    adapter, _filemanager = _adapter(
        {NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)}
    )
    labels = [
        np.full((2, 3, 4), 1, dtype=np.int32),
        np.full((2, 3, 4), 2, dtype=np.int32),
    ]

    adapter.add_objects(NUCLEI, labels)
    objects = adapter.get_objects(NUCLEI)

    assert objects.labels.shape == (2, 2, 3, 4)
    np.testing.assert_array_equal(objects.labels, np.stack(labels))


def test_cellprofiler_adapter_refuses_explosive_dense_object_label_sequences(monkeypatch):
    import openhcs.interop.cellprofiler.runtime.adapter as runtime_adapter

    adapter, _filemanager = _adapter(
        {NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)}
    )
    monkeypatch.setattr(runtime_adapter, "_MAX_DENSE_LABEL_STACK_BYTES", 16)
    labels = []
    for slice_index in range(3):
        stack = np.zeros((3, 2, 4, 5), dtype=np.int32)
        stack[slice_index] = slice_index + 1
        labels.append(stack)

    with pytest.raises(MemoryError, match="Refusing to materialize"):
        adapter.add_objects(NUCLEI, labels)


def test_cellprofiler_adapter_reads_declared_inputs_by_compiled_location():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    producer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs={NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)},
        filemanager=filemanager,
    )
    labels = ArrayLike()
    producer.add_objects(NUCLEI, labels)

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        artifact_inputs={
            NUCLEI: ArtifactInputPlan(
                name=NUCLEI,
                path="/memory/Nuclei.pkl",
                kind=ArtifactKind.OBJECT_LABELS,
                group_keys=(None,),
            )
        },
        filemanager=filemanager,
    )

    assert consumer.get_objects(NUCLEI).labels is labels


def test_cellprofiler_adapter_availability_accepts_grouped_runtime_inputs():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    group_paths = {
        "1": "/memory/Nuclei_s1.pkl",
        "2": "/memory/Nuclei_s2.pkl",
    }
    for group_key, path in group_paths.items():
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            group_key=group_key,
            artifact_outputs={
                NUCLEI: ArtifactOutputPlan(
                    name=NUCLEI,
                    path=path,
                    kind=ArtifactKind.OBJECT_LABELS,
                    group_keys=(group_key,),
                    paths_by_group={group_key: path},
                )
            },
            filemanager=filemanager,
        )
        producer.add_objects(NUCLEI, np.full((2, 2), int(group_key), dtype=np.int32))

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        artifact_inputs={
            NUCLEI: ArtifactInputPlan(
                name=NUCLEI,
                path="/memory/Nuclei.pkl",
                kind=ArtifactKind.OBJECT_LABELS,
                group_keys=tuple(group_paths),
                paths_by_group=group_paths,
            )
        },
        filemanager=filemanager,
    )

    consumer.require_artifact_available(
        name=NUCLEI,
        kind=ArtifactKind.OBJECT_LABELS,
    )


def test_cellprofiler_adapter_resolves_current_image_object_input_by_artifact_group():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    second_group = "B01"
    first_labels = np.full((2, 2), 1, dtype=np.int32)
    second_labels = np.full((2, 2), 2, dtype=np.int32)
    group_paths = {
        AXIS_ID: "/memory/Nuclei_A01.pkl",
        second_group: "/memory/Nuclei_B01.pkl",
    }
    for group_key, labels in (
        (AXIS_ID, first_labels),
        (second_group, second_labels),
    ):
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            group_key=group_key,
            artifact_outputs={
                NUCLEI: ArtifactOutputPlan(
                    name=NUCLEI,
                    path=group_paths[group_key],
                    kind=ArtifactKind.OBJECT_LABELS,
                    group_keys=(group_key,),
                    paths_by_group={group_key: group_paths[group_key]},
                )
            },
            filemanager=filemanager,
        )
        producer.add_objects(NUCLEI, labels)

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        artifact_inputs={
            NUCLEI: ArtifactInputPlan(
                name=NUCLEI,
                path="/memory/Nuclei.pkl",
                kind=ArtifactKind.OBJECT_LABELS,
                group_keys=(AXIS_ID, second_group),
                paths_by_group=group_paths,
            )
        },
        filemanager=filemanager,
    )
    current_image = image_payload_with_context(
        np.zeros((2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/plate/Images/A01_s001_w1_z001_t001.tif",
        ),
    )

    objects = consumer.get_objects(NUCLEI, current_image=current_image)

    np.testing.assert_array_equal(objects.labels, first_labels)


def test_cellprofiler_adapter_resolves_object_input_by_current_source_context():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    first_labels = np.full((2, 2), 1, dtype=np.int32)
    second_labels = np.full((2, 2), 2, dtype=np.int32)
    group_paths = {
        "1": "/memory/Nuclei_s1.pkl",
        "2": "/memory/Nuclei_s2.pkl",
    }
    for group_key, labels in (("1", first_labels), ("2", second_labels)):
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            group_key=group_key,
            artifact_outputs={
                NUCLEI: ArtifactOutputPlan(
                    name=NUCLEI,
                    path=group_paths[group_key],
                    kind=ArtifactKind.OBJECT_LABELS,
                    group_keys=(group_key,),
                    paths_by_group={group_key: group_paths[group_key]},
                )
            },
            filemanager=filemanager,
        )
        producer.add_objects(NUCLEI, labels)

    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "/plate/Images/A01_s002_w1_z001_t001.tif",
            "/plate/Images/A01_s002_w2_z001_t001.tif",
        ),
        pipeline_input_files=(
            "/plate/Images/A01_s001_w1_z001_t001.tif",
            "/plate/Images/A01_s001_w2_z001_t001.tif",
            "/plate/Images/A01_s002_w1_z001_t001.tif",
            "/plate/Images/A01_s002_w2_z001_t001.tif",
        ),
        current_step_input_files=(
            "/plate/Images/A01_s002_w1_z001_t001.tif",
            "/plate/Images/A01_s002_w2_z001_t001.tif",
        ),
    )
    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        artifact_inputs={
            NUCLEI: ArtifactInputPlan(
                name=NUCLEI,
                path="/memory/Nuclei.pkl",
                kind=ArtifactKind.OBJECT_LABELS,
                group_keys=("1", "2"),
                paths_by_group=group_paths,
            )
        },
        source_binding_context=source_binding_context,
        processing_context=ContextStub(filemanager),
        filemanager=filemanager,
    )

    objects = consumer.get_objects(
        NUCLEI,
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    np.testing.assert_array_equal(objects.labels, second_labels)


def test_cellprofiler_adapter_projects_default_runtime_slice_output_to_group_paths():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    group_paths = {
        "1": "/memory/Nuclei_s1.pkl",
        "2": "/memory/Nuclei_s2.pkl",
    }
    producer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        artifact_outputs={
            NUCLEI: ArtifactOutputPlan(
                name=NUCLEI,
                path="/memory/Nuclei.pkl",
                kind=ArtifactKind.OBJECT_LABELS,
                group_keys=("1", "2"),
                paths_by_group=group_paths,
            )
        },
        filemanager=filemanager,
    )
    labels = ObjectLabelPayload(
        labels=np.stack(
            (
                np.full((2, 2), 1, dtype=np.int32),
                np.full((2, 2), 2, dtype=np.int32),
            )
        ),
        declared_object_id_domains=((1,), (2,)),
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    producer.add_objects(NUCLEI, labels)

    assert ("memory", group_paths["1"]) in filemanager.saved
    assert ("memory", group_paths["2"]) in filemanager.saved
    assert ("memory", "/memory/Nuclei.pkl") not in filemanager.saved

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        artifact_inputs={
            NUCLEI: ArtifactInputPlan(
                name=NUCLEI,
                path="/memory/Nuclei.pkl",
                kind=ArtifactKind.OBJECT_LABELS,
                group_keys=("1", "2"),
                paths_by_group=group_paths,
            )
        },
        filemanager=filemanager,
    )

    objects = consumer.get_objects(NUCLEI)

    assert objects.labels.shape == (2, 2, 2)
    assert objects.declared_object_id_domains == ((1,), (2,))
    np.testing.assert_array_equal(objects.labels[0], np.full((2, 2), 1))
    np.testing.assert_array_equal(objects.labels[1], np.full((2, 2), 2))


def test_cellprofiler_adapter_stacks_unplanned_global_grouped_images():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    first = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="1",
        artifact_outputs={
            DNA_IMAGE: ArtifactOutputPlan(
                name=DNA_IMAGE,
                path="/memory/DNA_s1.pkl",
                kind=ArtifactKind.IMAGE,
                group_keys=("1",),
                paths_by_group={"1": "/memory/DNA_s1.pkl"},
            )
        },
        filemanager=filemanager,
    )
    second = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="2",
        artifact_outputs={
            DNA_IMAGE: ArtifactOutputPlan(
                name=DNA_IMAGE,
                path="/memory/DNA_s2.pkl",
                kind=ArtifactKind.IMAGE,
                group_keys=("2",),
                paths_by_group={"2": "/memory/DNA_s2.pkl"},
            )
        },
        filemanager=filemanager,
    )
    first.add_image(DNA_IMAGE, np.full((2, 3), 1.0, dtype=np.float32))
    second.add_image(DNA_IMAGE, np.full((2, 3), 2.0, dtype=np.float32))

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        filemanager=filemanager,
    )

    image = consumer.get_image(DNA_IMAGE)

    assert image.data.shape == (2, 2, 3)
    np.testing.assert_array_equal(image.data[0], np.full((2, 3), 1.0))
    np.testing.assert_array_equal(image.data[1], np.full((2, 3), 2.0))


def test_cellprofiler_adapter_relationships_validate_declared_inputs_by_location():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    producer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs={
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            CELLS: _plan(CELLS, ArtifactKind.OBJECT_LABELS),
        },
        filemanager=filemanager,
    )
    producer.add_objects(NUCLEI, ArrayLike())
    producer.add_objects(CELLS, ArrayLike())

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        artifact_inputs={
            NUCLEI: ArtifactInputPlan(
                name=NUCLEI,
                path="/memory/Nuclei.pkl",
                kind=ArtifactKind.OBJECT_LABELS,
                group_keys=(None,),
            ),
            CELLS: ArtifactInputPlan(
                name=CELLS,
                path="/memory/Cells.pkl",
                kind=ArtifactKind.OBJECT_LABELS,
                group_keys=(None,),
            ),
        },
        artifact_outputs={
            PARENT_CHILD: _plan(PARENT_CHILD, ArtifactKind.RELATIONSHIPS)
        },
        filemanager=filemanager,
    )

    relationship = consumer.add_relationship(
        PARENT_CHILD,
        parent_object_name=NUCLEI,
        child_object_name=CELLS,
        parent_ids=np.array([1]),
        child_ids=np.array([2]),
    )

    assert relationship.value.schema.kind is ArtifactKind.RELATIONSHIPS


def test_cellprofiler_adapter_declared_relationship_allows_pruned_child_endpoint():
    filemanager = FileManagerStub()
    relationship_name = "Nuclei_FilteredCells_relationships"
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        group_key="default",
        artifact_outputs={
            relationship_name: _plan(relationship_name, ArtifactKind.RELATIONSHIPS),
        },
        filemanager=filemanager,
    )

    relationship = adapter.add_relationship(
        relationship_name,
        parent_object_name=NUCLEI,
        child_object_name="FilteredCells",
        parent_ids=np.array([1]),
        child_ids=np.array([2]),
    )

    assert relationship.value.schema.kind is ArtifactKind.RELATIONSHIPS
    assert relationship.value.schema.relationship.target.name == "FilteredCells"


def test_cellprofiler_adapter_relationships_accept_grouped_parent_inputs():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    first = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="1",
        artifact_outputs={
            CELLS: ArtifactOutputPlan(
                name=CELLS,
                path="/memory/Cells_s1.pkl",
                kind=ArtifactKind.OBJECT_LABELS,
                group_keys=("1",),
                paths_by_group={"1": "/memory/Cells_s1.pkl"},
            )
        },
        filemanager=filemanager,
    )
    second = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="2",
        artifact_outputs={
            CELLS: ArtifactOutputPlan(
                name=CELLS,
                path="/memory/Cells_s2.pkl",
                kind=ArtifactKind.OBJECT_LABELS,
                group_keys=("2",),
                paths_by_group={"2": "/memory/Cells_s2.pkl"},
            )
        },
        filemanager=filemanager,
    )
    first.add_objects(CELLS, np.zeros((2, 3), dtype=np.int32))
    second.add_objects(CELLS, np.ones((2, 3), dtype=np.int32))

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        artifact_outputs={
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            PARENT_CHILD: _plan(PARENT_CHILD, ArtifactKind.RELATIONSHIPS),
        },
        filemanager=filemanager,
    )
    consumer.add_objects(NUCLEI, np.zeros((2, 2, 3), dtype=np.int32))

    relationship = consumer.add_relationship(
        PARENT_CHILD,
        parent_object_name=CELLS,
        child_object_name=NUCLEI,
        parent_ids=np.array([1]),
        child_ids=np.array([2]),
    )

    assert relationship.value.schema.kind is ArtifactKind.RELATIONSHIPS


def test_cellprofiler_adapter_relationships_allow_same_invocation_child_output():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    producer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs={NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)},
        filemanager=filemanager,
    )
    producer.add_objects(NUCLEI, ArrayLike())

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        artifact_inputs={
            NUCLEI: ArtifactInputPlan(
                name=NUCLEI,
                path="/memory/Nuclei.pkl",
                kind=ArtifactKind.OBJECT_LABELS,
                group_keys=(None,),
            )
        },
        artifact_outputs={
            CELLS: _plan(CELLS, ArtifactKind.OBJECT_LABELS),
            PARENT_CHILD: _plan(PARENT_CHILD, ArtifactKind.RELATIONSHIPS),
        },
        filemanager=filemanager,
    )

    relationship = consumer.add_relationship(
        PARENT_CHILD,
        parent_object_name=NUCLEI,
        child_object_name=CELLS,
        parent_ids=np.array([1]),
        child_ids=np.array([2]),
    )

    assert relationship.value.schema.kind is ArtifactKind.RELATIONSHIPS


def test_cellprofiler_adapter_adds_and_reads_spatial_grid_artifacts():
    adapter, filemanager = _adapter({"Grid": _plan("Grid", ArtifactKind.SPATIAL_GRID)})
    grid = SpatialGrid(
        name="grid_info",
        rows=30,
        columns=30,
        x_spacing=55.0,
        y_spacing=55.0,
        x_origin=27.0,
        y_origin=27.0,
    )

    adapter.add_spatial_grid("Grid", grid)
    stored = adapter.get_spatial_grid("Grid")

    assert stored.name == "Grid"
    assert stored.rows == 30
    assert stored.columns == 30
    assert stored.x_origin == 27.0
    assert filemanager.saved[("memory", "/memory/Grid.pkl")]["rows"] == 30


def test_cellprofiler_adapter_adds_and_reads_slice_aligned_spatial_grids():
    adapter, filemanager = _adapter({"Grid": _plan("Grid", ArtifactKind.SPATIAL_GRID)})
    grids = RuntimeSliceAlignedValues(
        slices=(
            SpatialGrid(
                name="grid_info",
                rows=2,
                columns=2,
                x_spacing=8.0,
                y_spacing=8.0,
                x_origin=1.0,
                y_origin=4.0,
            ),
            SpatialGrid(
                name="grid_info",
                rows=2,
                columns=2,
                x_spacing=8.0,
                y_spacing=8.0,
                x_origin=2.0,
                y_origin=4.0,
            ),
        )
    )

    adapter.add_spatial_grid("Grid", grids)
    stored = adapter.get_spatial_grid("Grid")

    assert isinstance(stored, RuntimeSliceAlignedValues)
    assert [grid.name for grid in stored.slices] == ["Grid", "Grid"]
    assert [grid.x_origin for grid in stored.slices] == [1.0, 2.0]
    assert [grid["x_origin"] for grid in filemanager.saved[("memory", "/memory/Grid.pkl")]] == [
        1.0,
        2.0,
    ]


def test_cellprofiler_spatial_grid_resolver_broadcasts_identical_scalar_grid():
    scalar = SpatialGrid(
        name="grid_info",
        rows=2,
        columns=2,
        x_spacing=8.0,
        y_spacing=8.0,
        x_origin=1.0,
        y_origin=4.0,
    )
    aligned = RuntimeSliceAlignedValues(
        slices=(
            scalar,
            SpatialGrid(
                name="grid_info",
                rows=2,
                columns=2,
                x_spacing=8.0,
                y_spacing=8.0,
                x_origin=1.0,
                y_origin=4.0,
                slice_index=1,
            ),
        )
    )

    resolved = SpatialGridValueAuthority.single_spatial_grid("Grid", (scalar, aligned))

    assert isinstance(resolved, RuntimeSliceAlignedValues)
    assert [grid.name for grid in resolved.slices] == ["Grid", "Grid"]
    assert [grid.x_origin for grid in resolved.slices] == [1.0, 1.0]


def test_cellprofiler_adapter_replaces_existing_payload_with_latest_binding():
    adapter, filemanager = _adapter(
        {NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS)}
    )
    first = np.ones((2, 2), dtype=np.uint16)
    second = np.full((2, 2), 2, dtype=np.uint16)

    adapter.add_objects(NUCLEI, first)
    record = adapter.add_objects(NUCLEI, second)

    assert record.value.data is second
    assert filemanager.deleted == [("memory", "/memory/Nuclei.pkl")]
    assert filemanager.saved[("memory", "/memory/Nuclei.pkl")] is second


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


def test_cellprofiler_adapter_merges_same_artifact_measurement_subjects():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            CELLS: _plan(CELLS, ArtifactKind.OBJECT_LABELS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    adapter.add_objects(NUCLEI, np.array([[1]], dtype=np.int32))
    adapter.add_objects(CELLS, np.array([[1]], dtype=np.int32))
    adapter.add_measurements(
        MEASUREMENTS,
        ({"object_name": NUCLEI, "object_label": 1, "area": 10.0},),
        object_name=NUCLEI,
    )
    adapter.add_measurements(
        MEASUREMENTS,
        ({"object_name": CELLS, "object_label": 1, "area": 20.0},),
        object_name=CELLS,
    )

    measurements = adapter.get_measurements(MEASUREMENTS)

    assert measurements.name == MEASUREMENTS
    assert measurements.object_name is None
    assert tuple(measurements.rows) == (
        {"object_name": NUCLEI, "object_label": 1, "area": 10.0},
        {"object_name": CELLS, "object_label": 1, "area": 20.0},
    )


def test_cellprofiler_adapter_selects_current_source_measurement_record():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    object_group_paths = {
        "1": "/memory/A01_s001_w1_z001_t001_Nuclei.pkl",
        "2": "/memory/A01_s002_w1_z001_t001_Nuclei.pkl",
    }
    group_paths = {
        "1": "/memory/A01_s001_w1_z001_t001_NucleiMeasurements.pkl",
        "2": "/memory/A01_s002_w1_z001_t001_NucleiMeasurements.pkl",
    }
    for group_key, rows in (
        ("1", [{"object_id": 1, "area": 10.0}]),
        ("2", [{"object_id": 1, "area": 20.0}]),
    ):
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            group_key=group_key,
            artifact_outputs={
                NUCLEI: ArtifactOutputPlan(
                    name=NUCLEI,
                    path=object_group_paths[group_key],
                    kind=ArtifactKind.OBJECT_LABELS,
                    group_keys=(group_key,),
                    paths_by_group={group_key: object_group_paths[group_key]},
                ),
                NUCLEI_MEASUREMENTS: ArtifactOutputPlan(
                    name=NUCLEI_MEASUREMENTS,
                    path=group_paths[group_key],
                    kind=ArtifactKind.MEASUREMENTS,
                    group_keys=(group_key,),
                    paths_by_group={group_key: group_paths[group_key]},
                )
            },
            filemanager=filemanager,
        )
        producer.add_objects(NUCLEI, np.array([[1]], dtype=np.int32))
        producer.add_measurements(
            NUCLEI_MEASUREMENTS,
            rows,
            object_name=NUCLEI,
        )

    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "/plate/Images/A01_s002_w1_z001_t001.tif",
        ),
        pipeline_input_files=(
            "/plate/Images/A01_s001_w1_z001_t001.tif",
            "/plate/Images/A01_s002_w1_z001_t001.tif",
        ),
        current_step_input_files=(
            "/plate/Images/A01_s002_w1_z001_t001.tif",
        ),
    )
    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="default",
        artifact_inputs={
            NUCLEI_MEASUREMENTS: ArtifactInputPlan(
                name=NUCLEI_MEASUREMENTS,
                path="/memory/NucleiMeasurements.pkl",
                kind=ArtifactKind.MEASUREMENTS,
                group_keys=("1", "2"),
                paths_by_group=group_paths,
            )
        },
        source_binding_context=source_binding_context,
        processing_context=ContextStub(filemanager),
        filemanager=filemanager,
    )

    measurements = consumer.get_measurements(
        NUCLEI_MEASUREMENTS,
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    assert measurements.rows == [{"object_id": 1, "area": 20.0}]
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


def test_cellprofiler_adapter_offsets_repeated_scalar_measurement_tables():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    source_binding_plan = CompiledSourceBindingPlan.from_config(
        StepSourceBindingsConfig(
            groups=(GroupedSourceBindings(bindings=(NamedSourceBinding(alias=DNA_IMAGE),)),)
        )
    )

    for index, value in enumerate((11.0, 13.0), start=1):
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            artifact_outputs={
                NUCLEI_MEASUREMENTS: ArtifactOutputPlan(
                    name=NUCLEI_MEASUREMENTS,
                    path=f"/memory/{NUCLEI_MEASUREMENTS}_{index}.pkl",
                    kind=ArtifactKind.MEASUREMENTS,
                ),
            },
            source_binding_plan=source_binding_plan,
            filemanager=filemanager,
        )
        producer.add_measurements(
            NUCLEI_MEASUREMENTS,
            [
                {
                    "slice_index": 0,
                    "object_label": 1,
                    "object_name": NUCLEI,
                    "area": value,
                }
            ],
        )

    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs={},
        source_binding_plan=source_binding_plan,
        filemanager=filemanager,
    )

    first, second = adapter.measurement_tables_for_object(NUCLEI)

    assert first.rows[0]["slice_index"] == 0
    assert second.rows[0]["slice_index"] == 1


def test_cellprofiler_adapter_aligns_multiplane_measurements_across_groups():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    outputs = {
        NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
        NUCLEI_MEASUREMENTS: _plan(NUCLEI_MEASUREMENTS, ArtifactKind.MEASUREMENTS),
    }
    source_binding_plan = CompiledSourceBindingPlan.from_config(
        StepSourceBindingsConfig(
            groups=(GroupedSourceBindings(bindings=(NamedSourceBinding(alias=DNA_IMAGE),)),)
        )
    )

    for group_key, value in (("site1", 5.0), ("site2", 7.0)):
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            artifact_outputs=outputs,
            source_binding_plan=source_binding_plan,
            filemanager=filemanager,
            group_key=group_key,
        )
        producer.add_measurements(
            NUCLEI_MEASUREMENTS,
            [
                {
                    "slice_index": 0,
                    "object_label": 1,
                    "mean_intensity": value,
                    "object_name": NUCLEI,
                }
            ],
            source_image_name="rawGFP",
        )

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs=outputs,
        source_binding_plan=source_binding_plan,
        filemanager=filemanager,
        group_key="collapsed",
    )
    labels = np.array([[[1]], [[1]]], dtype=np.int32)

    values = consumer.measurement_values_for_label_slices(
        NUCLEI,
        "Intensity_MeanIntensity_rawGFP",
        labels,
    )

    np.testing.assert_allclose(values[0], [5.0])
    np.testing.assert_allclose(values[1], [7.0])


def test_cellprofiler_adapter_broadcasts_single_slice_measurements_for_repeated_labels():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            NUCLEI_MEASUREMENTS: _plan(NUCLEI_MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    adapter.add_measurements(
        NUCLEI_MEASUREMENTS,
        [
            {
                "slice_index": 0,
                "object_label": 1,
                "mean_intensity": 5.0,
                "object_name": NUCLEI,
            }
        ],
        source_image_name="rawGFP",
    )
    labels = np.array([[[1]], [[1]], [[1]]], dtype=np.int32)

    values = adapter.measurement_values_for_label_slices(
        NUCLEI,
        "Intensity_MeanIntensity_rawGFP",
        labels,
    )

    assert len(values) == 3
    for value in values:
        np.testing.assert_allclose(value, [5.0])


def test_measurement_lookup_uses_table_source_for_source_qualified_object_rows():
    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASURE_OBJECT_INTENSITY,
                rows=(
                    {"object_name": NUCLEI, "object_label": 1, "mean_intensity": 9.0},
                ),
                source_image_name=DNA_IMAGE,
            ),
            MeasurementTable(
                name=MEASURE_OBJECT_INTENSITY,
                rows=(
                    {"object_name": NUCLEI, "object_label": 1, "mean_intensity": 5.0},
                ),
                source_image_name="rawGFP",
            ),
        ),
        "Intensity_MeanIntensity_rawGFP",
        object_count=1,
        object_ids=(1,),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [5.0])


def test_measurement_lookup_table_source_owns_columnar_source_domain():
    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASURE_OBJECT_INTENSITY,
                rows=SimpleColumnarRows(
                    {
                        "object_name": (NUCLEI,),
                        "object_label": (1,),
                        "source_image_name": ("auxiliary-row-owner",),
                        "mean_intensity": (5.0,),
                    }
                ),
                source_image_name="rawGFP",
            ),
        ),
        "Intensity_MeanIntensity_rawGFP",
        object_count=1,
        object_ids=(1,),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [5.0])


def test_measurement_lookup_normalizes_columnar_object_domain():
    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASURE_OBJECT_INTENSITY,
                rows=SimpleColumnarRows(
                    {
                        "object_name": ("nuclei",),
                        "object_label": (1,),
                        "mean_intensity": (5.0,),
                    }
                ),
            ),
        ),
        "Intensity_MeanIntensity_rawGFP",
        object_count=1,
        object_ids=(1,),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [5.0])


def test_cellprofiler_adapter_multiplane_measurement_alignment_is_feature_scoped():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    outputs = {
        NUCLEI_MEASUREMENTS: _plan(NUCLEI_MEASUREMENTS, ArtifactKind.MEASUREMENTS),
    }
    source_binding_plan = CompiledSourceBindingPlan.from_config(
        StepSourceBindingsConfig(
            groups=(GroupedSourceBindings(bindings=(NamedSourceBinding(alias=DNA_IMAGE),)),)
        )
    )

    for group_key, feature_value in (
        ("dna_site1", 90.0),
        ("gfp_site1", 5.0),
        ("dna_site2", 95.0),
        ("gfp_site2", 7.0),
    ):
        producer = CellProfilerRuntimeAdapter(
            runtime_value_store=store,
            axis_id=AXIS_ID,
            artifact_outputs=outputs,
            source_binding_plan=source_binding_plan,
            filemanager=filemanager,
            group_key=group_key,
        )
        producer.add_measurements(
            NUCLEI_MEASUREMENTS,
            (
                {
                    "object_name": NUCLEI,
                    "object_label": 1,
                    "mean_intensity": feature_value,
                },
            ),
            source_image_name=DNA_IMAGE if group_key.startswith("dna") else "rawGFP",
        )

    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_outputs=outputs,
        source_binding_plan=source_binding_plan,
        filemanager=filemanager,
        group_key="collapsed",
    )

    values = consumer.measurement_values_for_label_slices(
        NUCLEI,
        "Intensity_MeanIntensity_rawGFP",
        np.array([[[1]], [[1]]], dtype=np.int32),
    )

    np.testing.assert_allclose(values[0], [5.0])
    np.testing.assert_allclose(values[1], [7.0])


def test_cellprofiler_adapter_measurement_query_cache_is_store_scoped():
    first, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            NUCLEI_MEASUREMENTS: _plan(NUCLEI_MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    second, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            NUCLEI_MEASUREMENTS: _plan(NUCLEI_MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    labels = np.array([[1, 2], [0, 0]], dtype=np.int32)
    for adapter, values in (
        (first, (20.0, 80.0)),
        (second, (4.0, 12.0)),
    ):
        adapter.add_objects(NUCLEI, labels)
        adapter.add_measurements(
            NUCLEI_MEASUREMENTS,
            [
                {
                    "object_name": NUCLEI,
                    "object_label": object_label,
                    "feature_name": "AreaShape_Area",
                    "result_value": value,
                }
                for object_label, value in enumerate(values, start=1)
            ],
            object_name=NUCLEI,
        )

    np.testing.assert_allclose(
        first.measurement_values_for_label_slices(
            NUCLEI,
            "AreaShape_Area",
            labels,
        )[0],
        [20.0, 80.0],
    )
    np.testing.assert_allclose(
        second.measurement_values_for_label_slices(
            NUCLEI,
            "AreaShape_Area",
            labels,
        )[0],
        [4.0, 12.0],
    )


def test_cellprofiler_adapter_projects_duplicate_object_labels_to_current_runtime_slice():
    filemanager = FileManagerStub()
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        artifact_outputs={
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            NUCLEI_MEASUREMENTS: _plan(NUCLEI_MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        },
        source_binding_plan=CompiledSourceBindingPlan.from_config(
            StepSourceBindingsConfig(
                groups=(
                    GroupedSourceBindings(
                        bindings=(NamedSourceBinding(alias=DNA_IMAGE),)
                    ),
                )
            )
        ),
        filemanager=filemanager,
        plane_projection=RuntimePlaneProjection.group(1),
    )
    labels = np.array([[1, 2], [0, 0]], dtype=np.int32)
    adapter.add_objects(NUCLEI, labels)
    adapter.add_measurements(
        NUCLEI_MEASUREMENTS,
        [
            {
                "slice_index": slice_index,
                "object_name": NUCLEI,
                "object_label": object_label,
                "feature_name": "AreaShape_Area",
                "result_value": value,
            }
            for slice_index, values in enumerate(((100.0, 200.0), (500.0, 600.0)))
            for object_label, value in enumerate(values, start=1)
        ],
        object_name=NUCLEI,
    )

    value_slices = adapter.measurement_values_for_label_slices(
        NUCLEI,
        "AreaShape_Area",
        labels,
    )

    assert len(value_slices) == 1
    np.testing.assert_allclose(value_slices[0], [500.0, 600.0])


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

    with pytest.raises(RuntimeError, match="Missing runtime artifact"):
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


def test_cellprofiler_adapter_source_binding_plane_uses_group_order_for_volumes():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(alias="origDNA"),
                    NamedSourceBinding(alias="origMemb"),
                ),
            ),
        )
    )
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        source_binding_plan=CompiledSourceBindingPlan.from_config(source_bindings),
    )

    assert adapter.source_binding_axis_size(("origDNA",)) == 2
    assert adapter.source_binding_axis_plane_index(("origDNA",)) == 0
    assert adapter.source_binding_axis_plane_index(("origMemb",)) == 1
    assert adapter.source_binding_axis_plane_index(("origDNA", "origMemb")) is None


def test_source_binding_axis_plane_resolution_keeps_composed_alias_axis():
    resolution = SourceBindingAxisPlaneResolution(
        source_aliases=("OrigDNA", "OrigER", "OrigMito", "OrigRNA"),
        indexes=(36, 6),
    )

    assert resolution.plane_index() is None


def test_cellprofiler_adapter_source_binding_plane_uses_single_group_alias_when_unqualified():
    single_source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="OrigComet",
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    subject=SourceFilterSubject.FILE,
                                    match_type=SourceFilterMatchType.CONTAINS,
                                    value=".tif",
                                ),
                            )
                        ),
                    ),
                )
            ),
        )
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "A01_s001_w1_z001_t001.tif",
            "A01_s002_w1_z001_t001.tif",
        ),
        current_step_input_files=("A01_s002_w1_z001_t001.tif",),
    )
    filemanager = FileManagerStub()
    single_adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        source_binding_plan=CompiledSourceBindingPlan.from_config(single_source_bindings),
        source_binding_context=source_binding_context,
        processing_context=ContextStub(filemanager),
        filemanager=filemanager,
    )
    multi_source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(alias="OrigComet"),
                    NamedSourceBinding(alias="Other"),
                ),
            ),
        )
    )
    multi_adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        source_binding_plan=CompiledSourceBindingPlan.from_config(multi_source_bindings),
    )

    assert single_adapter.source_binding_axis_plane_index(()) == 1
    assert multi_adapter.source_binding_axis_plane_index(()) is None


def test_source_binding_plane_prefers_current_virtual_path_over_duplicate_metadata():
    dna_virtual_path = "A14_s001_w1_z001_t001.tif"
    dna_real_path = "/plate/images_Illum-corrected/plate1_A14_site1_Ch1.tif"
    dna_archive_en_path = (
        "/plate/Archive_EN/images_Illum-corrected/plate1_A14_site1_Ch1.tif"
    )
    dna_archive_es_path = (
        "/plate/Archive_ES/images_Illum-corrected/plate1_A14_site1_Ch1.tif"
    )
    er_virtual_path = "A14_s001_w2_z001_t001.tif"
    er_real_path = "/plate/images_Illum-corrected/plate1_A14_site1_Ch2.tif"
    er_archive_en_path = (
        "/plate/Archive_EN/images_Illum-corrected/plate1_A14_site1_Ch2.tif"
    )
    er_archive_es_path = (
        "/plate/Archive_ES/images_Illum-corrected/plate1_A14_site1_Ch2.tif"
    )
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="OrigDNA",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    SourceFilterSubject.FILE,
                                    SourceFilterMatchType.CONTAINS,
                                    "Ch1",
                                ),
                            )
                        ),
                    ),
                    NamedSourceBinding(
                        alias="OrigER",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    SourceFilterSubject.FILE,
                                    SourceFilterMatchType.CONTAINS,
                                    "Ch2",
                                ),
                            )
                        ),
                    ),
                )
            ),
        ),
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField("OrigDNA", "Well"),
                        SourceBindingMatchField("OrigER", "Well"),
                    )
                ),
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField("OrigDNA", "Site"),
                        SourceBindingMatchField("OrigER", "Site"),
                    )
                ),
            ),
        ),
    )
    dna_metadata = {
        "Plate": "plate1",
        "Well": "A14",
        "Site": "1",
        "ChannelNumber": "1",
    }
    er_metadata = {
        "Plate": "plate1",
        "Well": "A14",
        "Site": "1",
        "ChannelNumber": "2",
    }
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(dna_virtual_path,),
        current_step_input_files=(dna_virtual_path,),
        step_input_source_paths={
            dna_virtual_path: dna_real_path,
            er_virtual_path: er_real_path,
        },
        source_metadata_by_path={
            dna_virtual_path: dna_metadata,
            dna_real_path: dna_metadata,
            dna_archive_en_path: dna_metadata,
            dna_archive_es_path: dna_metadata,
            er_virtual_path: er_metadata,
            er_real_path: er_metadata,
            er_archive_en_path: er_metadata,
            er_archive_es_path: er_metadata,
        },
        pipeline_input_files=(
            dna_archive_en_path,
            dna_archive_es_path,
            dna_real_path,
            er_archive_en_path,
            er_archive_es_path,
            er_real_path,
        ),
        pipeline_input_backend="memory",
    )
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id="A14",
        axis_component="site",
        axis_component_value="1",
        source_binding_plan=CompiledSourceBindingPlan.from_config(source_bindings),
        source_binding_context=source_binding_context,
        processing_context=ContextStub(FileManagerStub()),
        filemanager=FileManagerStub(),
    )

    dna_plane_index = adapter.source_binding_axis_plane_index(("OrigDNA",))
    dna_candidate_context = adapter.source_binding_plane_candidate_context("OrigDNA")
    er_plane_index = adapter.source_binding_axis_plane_index(("OrigER",))
    er_candidate_context = adapter.source_binding_plane_candidate_context("OrigER")

    assert dna_candidate_context is not None
    assert dna_candidate_context.axis_candidates[dna_plane_index].path == dna_virtual_path
    assert er_candidate_context is not None
    assert er_candidate_context.axis_candidates[er_plane_index].path == er_virtual_path


def test_cellprofiler_adapter_single_source_alias_keeps_runtime_site_stack_unprojected():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="DF_image",
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    subject=SourceFilterSubject.FILE,
                                    match_type=SourceFilterMatchType.CONTAINS,
                                    value="Ch6",
                                ),
                            )
                        ),
                    ),
                )
            ),
        )
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "A01_s001_w6_z001_t001.tif",
            "A01_s002_w6_z001_t001.tif",
        ),
    )
    filemanager = FileManagerStub()
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        source_binding_plan=CompiledSourceBindingPlan.from_config(source_bindings),
        source_binding_context=source_binding_context,
        processing_context=ContextStub(filemanager),
        filemanager=filemanager,
    )

    assert adapter.source_binding_axis_plane_index(("DF_image",)) == 0


def test_current_source_object_labels_project_source_binding_axis_without_plane_metadata():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="OrigComet",
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    subject=SourceFilterSubject.FILE,
                                    match_type=SourceFilterMatchType.CONTAINS,
                                    value=".tif",
                                ),
                            )
                        ),
                    ),
                )
            ),
        )
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "A01_s001_w1_z001_t001.tif",
            "A01_s002_w1_z001_t001.tif",
        ),
        current_step_input_files=("A01_s002_w1_z001_t001.tif",),
    )
    filemanager = FileManagerStub()
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        source_binding_plan=CompiledSourceBindingPlan.from_config(source_bindings),
        source_binding_context=source_binding_context,
        processing_context=ContextStub(filemanager),
        filemanager=filemanager,
    )
    labels = ObjectLabelSet(
        name="Comet",
        labels=np.asarray(
            (
                [[1, 0], [0, 0]],
                [[0, 0], [0, 2]],
            ),
            dtype=np.int32,
        ),
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )

    projected = CurrentSourceObjectLabelPayloadProjection(
        adapter,
        current_image=np.zeros((2, 2), dtype=np.float32),
    ).project(labels)

    np.testing.assert_array_equal(
        object_label_dense_array(projected),
        np.asarray([[0, 0], [0, 2]], dtype=np.int32),
    )


def test_current_source_object_labels_project_duplicate_source_stack_by_provenance():
    virtual_path = "A14_s001_w4_z001_t001.tif"
    real_path = "/plate/images_Illum-corrected/plate1_A14_site1_Ch4.tif"
    archive_en_path = (
        "/plate/Archive_EN/images_Illum-corrected/plate1_A14_site1_Ch4.tif"
    )
    archive_es_path = (
        "/plate/Archive_ES/images_Illum-corrected/plate1_A14_site1_Ch4.tif"
    )
    archive_pt_path = (
        "/plate/Archive_PT/images_Illum-corrected/plate1_A14_site1_Ch4.tif"
    )
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id="A14",
        axis_component="site",
        axis_component_value="1",
        source_binding_context=SourceBindingRuntimeContext(
            step_input_source_paths={virtual_path: real_path},
        ),
    )
    labels = ObjectLabelSet(
        name="Cells",
        labels=np.asarray(
            (
                [[1, 0], [0, 0]],
                [[0, 2], [0, 0]],
                [[0, 0], [3, 0]],
                [[0, 0], [0, 4]],
            ),
            dtype=np.int32,
        ),
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        channel_source_paths=(
            real_path,
            archive_en_path,
            archive_es_path,
            archive_pt_path,
        ),
        source_image_names=(
            "OrigActin_Golgi_Membrane",
            "OrigActin_Golgi_Membrane",
            "OrigActin_Golgi_Membrane",
            "OrigActin_Golgi_Membrane",
        ),
        source_image_name="OrigActin_Golgi_Membrane",
    )

    projected = CurrentSourceObjectLabelPayloadProjection(
        adapter,
        current_image=np.zeros((2, 2), dtype=np.float32),
    ).project(labels)

    np.testing.assert_array_equal(
        object_label_dense_array(projected),
        np.asarray([[1, 0], [0, 0]], dtype=np.int32),
    )


def test_cellprofiler_adapter_preserves_step_input_image_metadata():
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
    fallback_stack = image_payload_with_context(
        np.stack(
            [
                np.full((2, 2), 1.0, dtype=np.float32),
                np.full((2, 2), 2.0, dtype=np.float32),
            ]
        ),
        metadata=ImagePayloadMetadata(
            channel_intensity_scales=(65535.0, 255.0),
            channel_source_dtypes=("uint16", "uint8"),
        ),
    )

    resolved = adapter.resolve_source_image(DNA_IMAGE, fallback_stack)

    np.testing.assert_array_equal(resolved, np.asarray(fallback_stack)[0])
    metadata = image_payload_metadata(resolved)
    assert metadata.intensity_scale == 65535.0
    assert metadata.source_dtype == "uint16"


def test_cellprofiler_adapter_resolves_source_metadata_from_runtime_context():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias=DNA_IMAGE,
                        selector=SourceSelector(
                            metadata=(MetadataSelector("Compound", "DMSO"),),
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
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"Compound": "Vehicle"},
            "A01_s001_w2_z001_t001.tif": {"Compound": "DMSO"},
        },
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
    current_stack = np.stack(
        [
            np.full((2, 2), 1.0, dtype=np.float32),
            np.full((2, 2), 2.0, dtype=np.float32),
        ]
    )

    resolved = adapter.resolve_source_image(DNA_IMAGE, current_stack)

    np.testing.assert_array_equal(resolved, current_stack[1])


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


def test_cellprofiler_adapter_matches_explicit_component_alias_metadata():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="TypeI",
                        selector=SourceSelector(
                            components=(
                                ComponentSelector(AllComponents.CHANNEL, "00"),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "source_s001_w1_z001_t001.tif",
            "source_s001_w2_z001_t001.tif",
        ),
        source_metadata_by_path={
            "source_s001_w1_z001_t001.tif": {"ChannelNumber": "00"},
            "source_s001_w2_z001_t001.tif": {"ChannelNumber": "01"},
        },
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
    current_stack = np.stack(
        [
            np.full((2, 2), 7.0, dtype=np.float32),
            np.full((2, 2), 9.0, dtype=np.float32),
        ]
    )

    resolved = adapter.resolve_source_image("TypeI", current_stack)

    np.testing.assert_array_equal(resolved, current_stack[0])


def test_cellprofiler_adapter_explicit_component_overrides_inherited_scope():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="TypeI",
                        selector=SourceSelector(
                            components=(
                                ComponentSelector(AllComponents.CHANNEL, "00"),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "source_s001_w1_z001_t001.tif",
            "source_s001_w2_z001_t001.tif",
        ),
        source_metadata_by_path={
            "source_s001_w1_z001_t001.tif": {
                "channel": "1",
                "ChannelNumber": "00",
            },
            "source_s001_w2_z001_t001.tif": {
                "channel": "2",
                "ChannelNumber": "01",
            },
        },
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
    current_stack = np.stack(
        [
            np.full((2, 2), 7.0, dtype=np.float32),
            np.full((2, 2), 9.0, dtype=np.float32),
        ]
    )

    resolved = adapter.resolve_source_image("TypeI", current_stack)

    np.testing.assert_array_equal(resolved, current_stack[0])


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


def test_cellprofiler_adapter_scopes_match_plan_values_by_source_alias_selector():
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
                            components=(
                                ComponentSelector(AllComponents.CHANNEL, "3"),
                            ),
                        ),
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
                            metadata_field="site",
                        ),
                        SourceBindingMatchField(
                            alias="IllumBlue",
                            metadata_field="site",
                        ),
                    ),
                ),
            ),
        ),
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "A01_s001_w1_z001_t001.tif",
            "A01_s002_w2_z001_t001.tif",
        ),
        pipeline_input_files=(
            "/plate/Images/A01_s001_w3_z001_t001.tif",
            "/plate/Images/A01_s002_w3_z001_t001.tif",
        ),
        pipeline_input_backend="memory",
    )
    filemanager = FileManagerStub()
    expected = np.full((2, 2), 31.0, dtype=np.float32)
    filemanager.saved[("memory", "/plate/Images/A01_s001_w3_z001_t001.tif")] = expected
    filemanager.saved[("memory", "/plate/Images/A01_s002_w3_z001_t001.tif")] = np.full(
        (2, 2),
        41.0,
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


def test_cellprofiler_adapter_ignores_unbound_match_plan_aliases():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="OrigDNA",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    subject=SourceFilterSubject.FILE,
                                    match_type=SourceFilterMatchType.CONTAINS,
                                    value="Ch1",
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
                pattern=r"(?P<Well>[A-P]{1}[0-9]{2})_site(?P<Site>[0-9])_Ch(?P<ChannelNumber>[1-5])\.tif",
            ),
        ),
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField(
                            alias="OrigDNA",
                            metadata_field="Well",
                        ),
                        SourceBindingMatchField(
                            alias="OrigRNA",
                            metadata_field="Well",
                        ),
                    ),
                ),
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField(
                            alias="OrigDNA",
                            metadata_field="Site",
                        ),
                        SourceBindingMatchField(
                            alias="OrigRNA",
                            metadata_field="Site",
                        ),
                    ),
                ),
            ),
        ),
    )
    files = tuple(
        f"A14_site{site}_Ch{channel}.tif"
        for site in (1, 2)
        for channel in (1, 2)
    )
    filemanager = FileManagerStub()
    for index, path in enumerate(files):
        filemanager.saved[("memory", path)] = np.full((2, 2), index, dtype=np.float32)
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_id=AXIS_ID,
        artifact_outputs={},
        source_binding_plan=CompiledSourceBindingPlan.from_config(source_bindings),
        source_binding_context=SourceBindingRuntimeContext(
            current_step_input_files=files,
            pipeline_input_files=files,
            pipeline_input_backend="memory",
        ),
        processing_context=ContextStub(filemanager),
        filemanager=filemanager,
    )

    resolved = adapter.resolve_source_image(
        "OrigDNA",
        np.stack([np.full((2, 2), index, dtype=np.float32) for index in range(4)]),
    )

    np.testing.assert_array_equal(
        resolved,
        np.stack(
            (
                filemanager.saved[("memory", "A14_site1_Ch1.tif")],
                filemanager.saved[("memory", "A14_site2_Ch1.tif")],
            )
        ),
    )


def test_cellprofiler_adapter_matches_metadata_keys_by_semantic_identity(tmp_path):
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="IllumBlue",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            metadata=(MetadataSelector("Metadata_Illum", "DAPI"),),
                        ),
                    ),
                ),
            ),
        ),
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"(?P<metadataillum>.+)_illum\.mat",
            ),
        ),
    )
    expected = np.full((2, 2), 31.0, dtype=np.float32)
    illum_path = tmp_path / "DAPI_illum.mat"
    savemat(illum_path, {"Image": expected})
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=("A01_s001_w1_z001_t001.tif",),
        step_input_dir="/plate/Images",
        pipeline_input_files=(str(illum_path),),
        pipeline_input_backend="disk",
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

    resolved = adapter.resolve_source_image(
        "IllumBlue",
        np.full((2, 2), 1.0, dtype=np.float32),
    )

    np.testing.assert_array_equal(resolved, expected)


def test_cellprofiler_adapter_resolves_pipeline_start_npy_source(tmp_path):
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="IllumBlue",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    SourceFilterSubject.FILE,
                                    SourceFilterMatchType.ENDS_WITH,
                                    "IllumBlue.npy",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    expected = np.full((2, 2), 31.0, dtype=np.float32)
    illum_path = tmp_path / "IllumBlue.npy"
    np.save(illum_path, expected)
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=("A01_s001_w1_z001_t001.tif",),
        step_input_dir="/plate/Images",
        pipeline_input_files=(str(illum_path),),
        pipeline_input_backend="disk",
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


def test_cellprofiler_adapter_resolves_step_input_color_stack_source_filters():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="orig_color",
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    subject=SourceFilterSubject.FILE,
                                    match_type=SourceFilterMatchType.CONTAINS,
                                    value="t0",
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
            "DMSO_B5_t0.JPG",
            "DMSO_B5_t24.JPG",
        ),
        step_input_dir="/plate/images",
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
    t0 = np.zeros((3, 4, 3), dtype=np.float32)
    t24 = np.ones((3, 4, 3), dtype=np.float32)
    image_stack = np.stack((t0, t24), axis=0)

    resolved = adapter.resolve_source_image("orig_color", image_stack)

    assert resolved.shape == (3, 4, 3)
    np.testing.assert_array_equal(resolved, t0)


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


def test_cellprofiler_adapter_order_match_returns_all_current_image_sets():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="BF_image",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    SourceFilterSubject.FILE,
                                    SourceFilterMatchType.CONTAINS,
                                    "Ch1",
                                ),
                            ),
                        ),
                    ),
                    NamedSourceBinding(
                        alias="DF_image",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    SourceFilterSubject.FILE,
                                    SourceFilterMatchType.CONTAINS,
                                    "Ch6",
                                ),
                            ),
                        ),
                    ),
                    NamedSourceBinding(
                        alias="Marker_image",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    SourceFilterSubject.FILE,
                                    SourceFilterMatchType.CONTAINS,
                                    "Ch7",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    source_paths = (
        "/plate/images/Ch1_1.tif",
        "/plate/images/Ch1_2.tif",
        "/plate/images/Ch6_1.tif",
        "/plate/images/Ch6_2.tif",
        "/plate/images/Ch7_1.tif",
        "/plate/images/Ch7_2.tif",
    )
    filemanager = FileManagerStub()
    for index, path in enumerate(source_paths):
        filemanager.saved[("memory", path)] = np.full((2, 2), index, dtype=np.float32)
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=source_paths,
        step_input_dir="/plate/images",
        pipeline_input_files=source_paths,
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
    current_stack = np.stack(
        tuple(filemanager.saved[("memory", path)] for path in source_paths)
    )

    resolved = adapter.resolve_source_image("BF_image", current_stack)

    assert resolved.shape == (2, 2, 2)
    np.testing.assert_array_equal(
        resolved,
        np.stack(
            (
                filemanager.saved[("memory", "/plate/images/Ch1_1.tif")],
                filemanager.saved[("memory", "/plate/images/Ch1_2.tif")],
            )
        ),
    )
    assert filemanager.loaded_batches == [
        (
            ("/plate/images/Ch1_1.tif", "/plate/images/Ch1_2.tif"),
            "memory",
            {},
        )
    ]


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


def test_cellprofiler_adapter_resolves_single_alias_order_source_from_current_scope():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="RawData",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    SourceFilterSubject.FILE,
                                    SourceFilterMatchType.CONTAINS,
                                    ".png",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    filemanager = FileManagerStub()
    expected_source = "/real/fat_orig.png"
    other_source = "/real/WT_orig.png"
    expected = np.full((2, 2), 31.0, dtype=np.float32)
    filemanager.saved[("memory", expected_source)] = expected
    filemanager.saved[("memory", other_source)] = np.full(
        (2, 2),
        41.0,
        dtype=np.float32,
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=("A01_s001_w1_z001_t001.png",),
        step_input_dir="/workspace",
        step_input_source_paths={
            "A01_s001_w1_z001_t001.png": expected_source,
            "A02_s001_w1_z001_t001.png": other_source,
        },
        pipeline_input_files=(expected_source, other_source),
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
        "RawData",
        np.full((2, 2), 1.0, dtype=np.float32),
    )

    np.testing.assert_array_equal(resolved, expected)
    assert filemanager.loaded_batches == [
        ((expected_source,), "memory", {}),
    ]


def test_cellprofiler_adapter_attaches_source_metadata_to_pipeline_start_image():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="RawData",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    SourceFilterSubject.FILE,
                                    SourceFilterMatchType.CONTAINS,
                                    ".png",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    filemanager = FileManagerStub()
    expected_source = "/real/fat_orig.png"
    expected = np.full((2, 2), 31, dtype=np.uint16)
    filemanager.saved[("memory", expected_source)] = expected
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=("A01_s001_w1_z001_t001.png",),
        step_input_dir="/workspace",
        pipeline_input_files=(expected_source,),
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
        "RawData",
        np.full((2, 2), 1.0, dtype=np.float32),
    )

    np.testing.assert_array_equal(resolved, expected)
    metadata = image_payload_metadata(resolved)
    assert metadata.intensity_scale == 65535.0
    assert metadata.source_dtype == "uint16"
    assert metadata.source_path == expected_source


def test_cellprofiler_adapter_converts_declared_grayscale_rgb_sources():
    from skimage.color import rgb2gray

    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="RawData",
                        origin=SourceBindingOrigin.PIPELINE_START,
                        selector=SourceSelector(),
                    ),
                ),
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    filemanager = FileManagerStub()
    expected_source = "/real/color_declared_gray.png"
    source_pixels = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    filemanager.saved[("memory", expected_source)] = source_pixels
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=("A01_s001_w1_z001_t001.png",),
        step_input_dir="/workspace",
        source_metadata_by_path={
            expected_source: {
                SOURCE_IMAGE_TYPE_METADATA_FIELD: "Grayscale image",
            },
        },
        pipeline_input_files=(expected_source,),
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
        "RawData",
        np.full((2, 2), 1.0, dtype=np.float32),
    )

    np.testing.assert_allclose(resolved, rgb2gray(source_pixels))
    metadata = image_payload_metadata(resolved)
    assert metadata.intensity_scale is None
    assert metadata.source_dtype == "float64"
    assert metadata.source_path == expected_source


def test_cellprofiler_source_image_semantics_materializes_full_validity_mask():
    payload = np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6)

    resolved = SourceImagePayloadSemantics.from_source_metadata(
        {SOURCE_IMAGE_TYPE_METADATA_FIELD: "Grayscale image"},
        "/workspace/source.tif",
    ).apply(payload)

    np.testing.assert_array_equal(resolved, payload)
    np.testing.assert_array_equal(
        image_payload_mask(resolved),
        np.ones(payload.shape, dtype=bool),
    )
    assert image_payload_metadata(resolved).source_path == "/workspace/source.tif"


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

    @declared_processing_contract(ProcessingContract.PURE_2D)
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

    @declared_processing_contract(ProcessingContract.FLEXIBLE)
    def measure(image_arg, *, labels):
        assert image_arg is image
        assert labels is adapter.get_objects(NUCLEI).labels
        return image_arg, rows

    executor.run(measure, image, cellprofiler_runtime=adapter)
    measurements = adapter.get_measurements(NUCLEI_MEASUREMENTS)

    assert _measurement_rows_for_assertion(measurements) == [
        {"object_id": 1, "area": 12.0, "slice_index": 0},
    ]
    assert measurements.object_name == NUCLEI
    assert measurements.source_image_name is None


def test_cellprofiler_object_only_measurement_uses_label_domain_reference_image():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            NUCLEI_MEASUREMENTS: _plan(
                NUCLEI_MEASUREMENTS,
                ArtifactKind.MEASUREMENTS,
            ),
        }
    )
    image = np.zeros((1006, 1000), dtype=np.float32)
    labels = np.ones((199, 199), dtype=np.int32)
    rows = [{"object_id": 1, "area": float(labels.size)}]
    seen = []
    adapter.add_objects(NUCLEI, labels)
    executor = _executor(
        MEASURE_OBJECT_SIZE_SHAPE,
        (ArtifactSpec(NUCLEI_MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
        runtime_artifact_inputs=(
            ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),
        ),
    )

    @declared_processing_contract(ProcessingContract.PURE_2D)
    def measure(image_arg, *, labels):
        seen.append((image_arg.copy(), labels.copy()))
        return image_arg, rows

    executor.run(measure, image, cellprofiler_runtime=adapter)
    measurements = adapter.get_measurements(NUCLEI_MEASUREMENTS)

    assert len(seen) == 1
    measurement_image, measurement_labels = seen[0]
    assert measurement_image.shape == labels.shape
    assert measurement_image.dtype == image.dtype
    np.testing.assert_array_equal(measurement_image, np.zeros_like(labels, dtype=image.dtype))
    np.testing.assert_array_equal(measurement_labels, labels)
    assert _measurement_rows_for_assertion(measurements) == [
        {"object_id": 1, "area": float(labels.size), "slice_index": 0},
    ]
    assert measurements.object_name == NUCLEI


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

    @declared_processing_contract(ProcessingContract.PURE_2D)
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
    assert _measurement_rows_for_assertion(measurements) == [
        {
            "mean": 3.0,
            "label": 1,
            "object_name": NUCLEI,
            "source_image_name": DNA_IMAGE,
            "slice_index": 0,
        },
        {
            "mean": 9.0,
            "label": 1,
            "object_name": NUCLEI,
            "source_image_name": "PH3",
            "slice_index": 0,
        },
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

    @declared_processing_contract(ProcessingContract.PURE_2D)
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
    assert _measurement_rows_for_assertion(measurements) == [
        {"object_count": 1, "slice_index": 0}
    ]
    assert measurements.object_name == NUCLEI
    assert measurements.source_image_name == f"{DNA_IMAGE}__PH3"


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

    @declared_processing_contract(ProcessingContract.PURE_2D)
    def measure(image_arg, *, labels):
        if labels is nuclei:
            return image_arg, [{"object": NUCLEI}]
        if labels is cells:
            return image_arg, [{"object": CELLS}]
        raise AssertionError("unexpected labels")

    executor.run(measure, image, cellprofiler_runtime=adapter)
    measurements = adapter.get_measurements(MEASUREMENTS)

    assert _measurement_rows_for_assertion(measurements) == [
        {"object": NUCLEI, "object_name": NUCLEI, "slice_index": 0},
        {"object": CELLS, "object_name": CELLS, "slice_index": 0},
    ]
    assert measurements.object_name is None
    assert measurements.source_image_name == DNA_IMAGE
    assert adapter.measurement_tables_for_object(NUCLEI) == (measurements,)
    assert adapter.measurement_tables_for_object(CELLS) == (measurements,)


def test_measurement_lookup_filters_mixed_object_measurement_rows():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            CELLS: _plan(CELLS, ArtifactKind.OBJECT_LABELS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    adapter.add_objects(
        NUCLEI,
        np.array([[1, 2], [0, 0]], dtype=np.int32),
    )
    adapter.add_objects(
        CELLS,
        np.array([[1, 0], [0, 0]], dtype=np.int32),
    )
    adapter.add_measurements(
        MEASUREMENTS,
        [
            {"object_name": NUCLEI, "object_label": 1, "mean_intensity": 5.0},
            {"object_name": NUCLEI, "object_label": 2, "mean_intensity": 7.0},
            {"object_name": CELLS, "object_label": 1, "mean_intensity": 11.0},
        ],
    )

    values = measurement_values_for_feature(
        adapter.measurement_tables_for_object(NUCLEI),
        "Intensity_MeanIntensity_CropBlue",
        object_count=2,
        object_name=NUCLEI,
    )

    np.testing.assert_array_equal(values, np.array([5.0, 7.0]))


def test_measurement_lookup_reads_slotted_dataclass_rows():
    @dataclass(frozen=True, slots=True)
    class MeasurementRow:
        object_name: str
        object_label: int
        mean_intensity: float

    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=(
                    MeasurementRow(NUCLEI, 1, 5.0),
                    MeasurementRow(CELLS, 1, 11.0),
                ),
            ),
        ),
        "Intensity_MeanIntensity_CropBlue",
        object_count=1,
        object_name=NUCLEI,
    )

    np.testing.assert_array_equal(values, np.array([5.0]))


def test_measurement_lookup_reads_long_form_feature_value_rows():
    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=(
                    {
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "feature_name": "Math_Ratio",
                        "result_value": 0.5,
                    },
                    {
                        "object_name": CELLS,
                        "object_label": 1,
                        "feature_name": "Math_Ratio",
                        "result_value": 0.25,
                    },
                ),
            ),
        ),
        "Math_Ratio",
        object_count=1,
        object_name=NUCLEI,
    )

    np.testing.assert_array_equal(values, np.array([0.5]))


def test_measurement_lookup_aligns_values_to_label_slices():
    value_slices = measurement_values_for_label_slices(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=(
                    {"object_label": 10, "area": 100.0, "object_name": NUCLEI},
                    {"object_label": 20, "area": 200.0, "object_name": NUCLEI},
                    {"object_label": 30, "area": 300.0, "object_name": NUCLEI},
                ),
            ),
        ),
        "AreaShape_Area",
        np.array(
            [
                [[10, 20], [0, 0]],
                [[30, 0], [0, 0]],
            ],
            dtype=np.int32,
        ),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert len(value_slices) == 2
    np.testing.assert_array_equal(value_slices[0], np.array([100.0, 200.0]))
    np.testing.assert_array_equal(value_slices[1], np.array([300.0]))


def test_measurement_lookup_does_not_stop_on_broad_cellprofiler_alias():
    value_slices = measurement_values_for_label_slices(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=(
                    {
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "Location_MaxIntensity_X_OrigGreen": 477.0,
                    },
                    {
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "Intensity_MaxIntensity_OrigGreen": 0.0549019612,
                    },
                    {
                        "object_name": NUCLEI,
                        "object_label": 2,
                        "Intensity_MaxIntensity_OrigGreen": 0.9607843161,
                    },
                ),
            ),
        ),
        "Intensity_MaxIntensity_OrigGreen",
        np.array([[1, 2]], dtype=np.int32),
        object_name=NUCLEI,
    )

    np.testing.assert_allclose(value_slices[0], [0.0549019612, 0.9607843161])


def test_measurement_lookup_uses_canonical_runtime_identifier_for_numbered_features():
    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=(
                    {"Children_PH3_Count": 2.0},
                    {"Children_PH3_Count": 5.0},
                ),
            ),
        ),
        "Children_PH3_Count",
        object_count=2,
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_array_equal(values, np.array([2.0, 5.0]))


def test_cellprofiler_child_count_lookup_uses_parent_row_domain():
    lookup = CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT.feature_lookup(
        "Children_PH3_Count"
    )

    assert lookup.query_object_name(NUCLEI) is None


def test_adapter_batch_child_count_lookup_uses_parent_row_domain():
    adapter, _filemanager = _adapter(
        {
            "PH3": _plan("PH3", ArtifactKind.OBJECT_LABELS),
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    labels = np.array([[1, 2]], dtype=np.int32)
    adapter.add_objects("PH3", labels)
    adapter.add_objects(NUCLEI, labels)
    adapter.add_measurements(
        MEASUREMENTS,
        (
            {"object_name": NUCLEI, "object_label": 1, "Children_PH3_Count": 2.0},
            {"object_name": NUCLEI, "object_label": 2, "Children_PH3_Count": 5.0},
        ),
        fields=(
            FieldSpec("object_name", str),
            FieldSpec("object_label", int),
            FieldSpec("Children_PH3_Count", float),
        ),
        object_name=NUCLEI,
        object_id_field="object_label",
    )

    values_by_object = ObjectLabelMeasurementSliceBatchResolver(
        adapter=adapter,
        requests={
            "PH3": ObjectLabelMeasurementSliceRequest(
                feature_name="Children_PH3_Count",
                labels=labels,
            )
        },
        feature_name="Children_PH3_Count",
        group_key=None,
    ).resolve()

    np.testing.assert_allclose(values_by_object["PH3"][0], [2.0, 5.0])


def test_relationship_plane_records_use_declared_group_order_not_key_spelling():
    store = RuntimeValueStore()
    relationship_name = "Nuclei_PH3_relationships"
    group_paths = {
        "site_a": "/memory/relationship_site_a.pkl",
        "site_b": "/memory/relationship_site_b.pkl",
    }
    semantics = RelationshipSemantics.parent_child(NUCLEI, "PH3")
    records = []
    for group_key, parent_id in (("site_b", 2), ("site_a", 1)):
        relationship = ObjectRelationship(
            name=relationship_name,
            source=semantics.source,
            target=semantics.target,
            source_ids=(parent_id,),
            target_ids=(parent_id,),
            relationship_type=semantics.relationship_type,
        )
        value = normalize_artifact_value(
            ArtifactOutputPlan(
                name=relationship_name,
                path=group_paths[group_key],
                kind=ArtifactKind.RELATIONSHIPS,
                group_keys=(group_key,),
                paths_by_group={group_key: group_paths[group_key]},
            ),
            relationship,
            axis_id=AXIS_ID,
        )
        records.append(
            store.record(value, path=group_paths[group_key], backend="memory")
        )
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        artifact_inputs={
            relationship_name: ArtifactInputPlan(
                name=relationship_name,
                path="/memory/relationships.pkl",
                kind=ArtifactKind.RELATIONSHIPS,
                group_keys=("site_a", "site_b"),
                paths_by_group=group_paths,
            )
        },
    )

    plane_resolution = adapter._relationship_plane_projection_resolution(
        relationship_name,
        tuple(records),
        label_plane_count=2,
    )
    plane_records = plane_resolution.require_records()

    assert plane_resolution.is_available is True
    assert [record.plane_index for record in plane_records] == [0, 1]
    assert [tuple(record.relationship.source_ids) for record in plane_records] == [
        (1,),
        (2,),
    ]


def test_relationship_plane_projection_reports_missing_declared_group_order():
    adapter, _filemanager = _adapter({})

    resolution = adapter._relationship_plane_projection_resolution(
        "Nuclei_PH3_relationships",
        (),
        label_plane_count=2,
    )

    assert resolution.is_available is False
    assert resolution.unavailable_reason == "no_records"


def test_object_measurement_table_index_uses_unnamed_object_id_feature_tables():
    table = MeasurementTable(
        name="Shape",
        rows=(
            {"object_label": 1, "form_factor": 0.9},
            {"object_label": 2, "form_factor": 1.1},
        ),
        fields=(
            FieldSpec("object_label", int),
            FieldSpec("form_factor", float),
        ),
        object_id_field="object_label",
    )

    tables = ObjectMeasurementTableIndex.from_tables((table,)).for_object_feature(
        "Cells",
        "AreaShape_FormFactor",
    )

    assert tables == (table,)


def test_child_count_lookup_tolerates_heterogeneous_relationship_summary_rows():
    table = MeasurementTable(
        name=RELATE_OBJECTS,
        rows=(
            {"image_number": 1, "child_object_count": 2},
            {"object_name": NUCLEI, "object_label": 1, "Children_PH3_Count": 2.0},
            {"object_name": NUCLEI, "object_label": 2, "Children_PH3_Count": 0.0},
        ),
    )
    projected_table = MeasurementTableAxisQuery(
        MeasurementRowAxisField.IMAGE_NUMBER,
        1,
    ).tables((table,))[0]

    values = measurement_values_for_feature(
        (projected_table,),
        "Children_PH3_Count",
        object_count=2,
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [2.0, 0.0])


def test_adapter_feature_prefilter_scans_heterogeneous_wide_rows():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    labels = np.array([[1, 2]], dtype=np.int32)
    adapter.add_objects(NUCLEI, labels)
    adapter.add_measurements(
        MEASUREMENTS,
        (
            {
                "object_label": 1,
                "Location_MaxIntensity_X_OrigGreen": 477.0,
            },
            {
                "object_label": 1,
                "Intensity_MaxIntensity_OrigGreen": 0.0549019612,
            },
            {
                "object_label": 2,
                "Intensity_MaxIntensity_OrigGreen": 0.9607843161,
            },
        ),
        object_name=NUCLEI,
        fields=(FieldSpec("object_label", int),),
        object_id_field="object_label",
    )

    value_slices = adapter.measurement_values_for_label_slices(
        NUCLEI,
        "Intensity_MaxIntensity_OrigGreen",
        labels,
    )

    np.testing.assert_allclose(value_slices[0], [0.0549019612, 0.9607843161])


def test_adapter_measurement_vector_scope_uses_feature_bearing_axis_projection():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    labels = np.array([[1, 2]], dtype=np.int32)
    adapter.add_objects(NUCLEI, labels)
    adapter.add_measurements(
        MEASUREMENTS,
        (
            {
                "image_number": 1,
                "object_label": 1,
                "Intensity_MaxIntensity_OrigGreen": 0.25,
            },
            {
                "image_number": 1,
                "object_label": 2,
                "Intensity_MaxIntensity_OrigGreen": 0.75,
            },
        ),
        object_name=NUCLEI,
        object_id_field="object_label",
    )

    value_slices = adapter.measurement_values_for_label_slices(
        NUCLEI,
        "Intensity_MaxIntensity_OrigGreen",
        labels,
        image_number=99,
    )

    np.testing.assert_allclose(value_slices[0], [0.25, 0.75])


def test_adapter_multiplane_label_lookup_prefers_runtime_slice_axis():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[2, 0], [0, 0]],
        ],
        dtype=np.int32,
    )
    adapter.add_objects(NUCLEI, labels)
    adapter.add_measurements(
        MEASUREMENTS,
        (
            {
                "slice_index": 0,
                "image_number": 10,
                "object_label": 1,
                "Intensity_MaxIntensity_OrigGreen": 0.25,
            },
            {
                "slice_index": 1,
                "image_number": 11,
                "object_label": 2,
                "Intensity_MaxIntensity_OrigGreen": 0.75,
            },
        ),
        object_name=NUCLEI,
        object_id_field="object_label",
    )

    value_slices = adapter.measurement_values_for_label_slices(
        NUCLEI,
        "Intensity_MaxIntensity_OrigGreen",
        labels,
        image_number=99,
    )

    np.testing.assert_allclose(value_slices[0], [0.25])
    np.testing.assert_allclose(value_slices[1], [0.75])


def test_adapter_measurement_vector_scope_searches_axis_when_group_has_no_table():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    producer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="producer",
        artifact_outputs={
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        },
        filemanager=filemanager,
    )
    labels = np.array([[1, 2]], dtype=np.int32)
    producer.add_objects(NUCLEI, labels)
    producer.add_measurements(
        MEASUREMENTS,
        (
            {
                "object_label": 1,
                "Intensity_MaxIntensity_OrigGreen": 0.25,
            },
            {
                "object_label": 2,
                "Intensity_MaxIntensity_OrigGreen": 0.75,
            },
        ),
        object_name=NUCLEI,
        object_id_field="object_label",
    )
    consumer = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_id=AXIS_ID,
        group_key="consumer",
        filemanager=filemanager,
    )

    value_slices = consumer.measurement_values_for_label_slices(
        NUCLEI,
        "Intensity_MaxIntensity_OrigGreen",
        labels,
    )

    np.testing.assert_allclose(value_slices[0], [0.25, 0.75])


def test_measurement_lookup_filters_source_qualified_columnar_feature_rows():
    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASURE_OBJECT_INTENSITY,
                rows=SimpleColumnarRows(
                    {
                        "object_name": (NUCLEI, NUCLEI, NUCLEI, NUCLEI),
                        "object_label": (1, 2, 1, 2),
                        "source_image_name": (
                            DNA_IMAGE,
                            DNA_IMAGE,
                            "OrigGreen",
                            "OrigGreen",
                        ),
                        "max_intensity": (0.90, 0.95, 0.05, 0.80),
                    }
                ),
            ),
        ),
        "Intensity_MaxIntensity_OrigGreen",
        object_count=2,
        object_ids=(1, 2),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [0.05, 0.80])


def test_measurement_lookup_preserves_heterogeneous_columnar_batch_features():
    rows = ConcatenatedMeasurementColumnarRows(
        (
            SimpleColumnarRows(
                {
                    "object_name": (NUCLEI, NUCLEI),
                    "object_label": (1, 2),
                    "source_image_name": (DNA_IMAGE, DNA_IMAGE),
                    "mean_intensity": (0.90, 0.95),
                }
            ),
            SimpleColumnarRows(
                {
                    "object_name": (NUCLEI, NUCLEI),
                    "object_label": (1, 2),
                    "source_image_name": ("rawGFP", "rawGFP"),
                    "MeanIntensity_rawGFP": (0.05, 0.80),
                }
            ),
        )
    )

    values = measurement_values_for_feature(
        (MeasurementTable(name=MEASURE_OBJECT_INTENSITY, rows=rows),),
        "Intensity_MeanIntensity_rawGFP",
        object_count=2,
        object_ids=(1, 2),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [0.05, 0.80])


def test_measurement_image_number_projection_keeps_axis_invariant_columnar_rows():
    rows = ConcatenatedMeasurementColumnarRows(
        (
            SimpleColumnarRows(
                {
                    "object_name": (NUCLEI,),
                    "object_label": (1,),
                    "image_number": (23,),
                    "source_image_name": (DNA_IMAGE,),
                    "mean_intensity": (0.90,),
                }
            ),
            SimpleColumnarRows(
                {
                    "object_name": (NUCLEI,),
                    "object_label": (1,),
                    "source_image_name": ("rawGFP",),
                    "mean_intensity": (0.05,),
                }
            ),
        )
    )
    table = MeasurementTable(name=MEASURE_OBJECT_INTENSITY, rows=rows)

    values = measurement_values_for_feature(
        MeasurementTableAxisQuery(
            MeasurementRowAxisField.IMAGE_NUMBER,
            23,
        ).tables((table,)),
        "Intensity_MeanIntensity_rawGFP",
        object_count=1,
        object_ids=(1,),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [0.05])


def test_measurement_image_number_projection_treats_singleton_axis_as_invariant():
    rows = SimpleColumnarRows(
        {
            "object_name": (NUCLEI,),
            "object_label": (1,),
            "image_number": (1,),
            "source_image_name": ("rawGFP",),
            "mean_intensity": (0.05,),
        }
    )
    table = MeasurementTable(name=MEASURE_OBJECT_INTENSITY, rows=rows)

    values = measurement_values_for_feature(
        MeasurementTableAxisQuery(
            MeasurementRowAxisField.IMAGE_NUMBER,
            23,
        ).tables((table,)),
        "Intensity_MeanIntensity_rawGFP",
        object_count=1,
        object_ids=(1,),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [0.05])


def test_measurement_image_number_projection_does_not_treat_row_sequence_singleton_axis_as_invariant():
    table = MeasurementTable(
        name=MEASURE_OBJECT_INTENSITY,
        rows=[
            {
                "object_name": NUCLEI,
                "object_label": 1,
                "image_number": 1,
                "mean_intensity": 0.05,
            },
        ],
    )

    projected = MeasurementTableAxisQuery(
        MeasurementRowAxisField.IMAGE_NUMBER,
        23,
    ).tables((table,))

    assert measurement_rows(projected) == ()


def test_measurement_lookup_allows_empty_multiplane_label_planes():
    labels = np.array(
        [
            [[0, 0]],
            [[1, 2]],
        ],
        dtype=np.int32,
    )
    measurement_rows = SimpleColumnarRows(
        {
            "slice_index": (1, 1),
            "object_name": (NUCLEI, NUCLEI),
            "object_label": (1, 2),
            "source_image_name": ("rawGFP", "rawGFP"),
            "mean_intensity": (0.25, 0.75),
        }
    )

    value_slices = measurement_values_for_label_slices(
        (
            MeasurementTable(
                name=MEASURE_OBJECT_INTENSITY,
                rows=measurement_rows,
            ),
        ),
        "Intensity_MeanIntensity_rawGFP",
        labels,
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert value_slices[0].size == 0
    np.testing.assert_allclose(value_slices[1], [0.25, 0.75])


def test_measurement_lookup_broadcasts_singleton_columnar_slice_to_label_stack():
    value_slices = measurement_values_for_label_slices(
        (
            MeasurementTable(
                name=MEASURE_OBJECT_INTENSITY,
                rows=SimpleColumnarRows(
                    {
                        "slice_index": (0, 0),
                        "object_name": (NUCLEI, NUCLEI),
                        "object_label": (1, 2),
                        "source_image_name": ("rawGFP", "rawGFP"),
                        "mean_intensity": (0.25, 0.75),
                    }
                ),
            ),
        ),
        "Intensity_MeanIntensity_rawGFP",
        np.array(
            [
                [[1, 0], [0, 0]],
                [[2, 0], [0, 0]],
            ],
            dtype=np.int32,
        ),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(value_slices[0], [0.25])
    np.testing.assert_allclose(value_slices[1], [0.75])


def test_measurement_lookup_broadcasts_singleton_indexed_slice_to_label_stack():
    value_slices = measurement_values_for_label_slices(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=(
                    {
                        "slice_index": 0,
                        "object_label": 1,
                        "mean_intensity": 0.25,
                        "object_name": NUCLEI,
                    },
                    {
                        "slice_index": 0,
                        "object_label": 2,
                        "mean_intensity": 0.5,
                        "object_name": NUCLEI,
                    },
                ),
                source_image_name="rawGFP",
            ),
        ),
        "Intensity_MeanIntensity_rawGFP",
        np.array(
            [
                [[1, 0], [0, 0]],
                [[2, 0], [0, 0]],
            ],
            dtype=np.int32,
        ),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert len(value_slices) == 2
    np.testing.assert_array_equal(value_slices[0], np.array([0.25]))
    np.testing.assert_array_equal(value_slices[1], np.array([0.5]))


def test_measurement_lookup_projects_shifted_slice_domain_to_local_label_stack():
    value_slices = measurement_values_for_label_slices(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=(
                    {
                        "slice_index": 8,
                        "object_label": 1,
                        "mean_intensity": 0.25,
                        "object_name": NUCLEI,
                    },
                    {
                        "slice_index": 9,
                        "object_label": 2,
                        "mean_intensity": 0.5,
                        "object_name": NUCLEI,
                    },
                ),
                source_image_name="rawGFP",
            ),
        ),
        "Intensity_MeanIntensity_rawGFP",
        np.array(
            [
                [[1, 0], [0, 0]],
                [[2, 0], [0, 0]],
            ],
            dtype=np.int32,
        ),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_array_equal(value_slices[0], np.array([0.25]))
    np.testing.assert_array_equal(value_slices[1], np.array([0.5]))


def test_measurement_lookup_projects_image_number_domain_to_label_stack():
    value_slices = measurement_values_for_label_slices(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=SimpleColumnarRows(
                    {
                        "image_number": (8, 9),
                        "object_name": (NUCLEI, NUCLEI),
                        "object_label": (1, 2),
                        "source_image_name": ("rawGFP", "rawGFP"),
                        "mean_intensity": (0.25, 0.5),
                    }
                ),
            ),
        ),
        "Intensity_MeanIntensity_rawGFP",
        np.array(
            [
                [[1, 0], [0, 0]],
                [[2, 0], [0, 0]],
            ],
            dtype=np.int32,
        ),
        object_name=NUCLEI,
        row_axis=MeasurementRowAxisField.IMAGE_NUMBER,
        row_axis_start=8,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_array_equal(value_slices[0], np.array([0.25]))
    np.testing.assert_array_equal(value_slices[1], np.array([0.5]))


def test_measurement_lookup_repeats_smaller_axis_domain_across_label_stack():
    value_slices = measurement_values_for_label_slices(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=SimpleColumnarRows(
                    {
                        "slice_index": (0, 1),
                        "object_name": (NUCLEI, NUCLEI),
                        "object_label": (1, 1),
                        "source_image_name": ("rawGFP", "rawGFP"),
                        "mean_intensity": (0.25, 0.5),
                    }
                ),
            ),
        ),
        "Intensity_MeanIntensity_rawGFP",
        np.array(
            [
                [[1, 0], [0, 0]],
                [[1, 0], [0, 0]],
                [[1, 0], [0, 0]],
                [[1, 0], [0, 0]],
            ],
            dtype=np.int32,
        ),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_array_equal(value_slices[0], np.array([0.25]))
    np.testing.assert_array_equal(value_slices[1], np.array([0.5]))
    np.testing.assert_array_equal(value_slices[2], np.array([0.25]))
    np.testing.assert_array_equal(value_slices[3], np.array([0.5]))


def test_measurement_lookup_returns_empty_slices_for_empty_objects():
    value_slices = measurement_values_for_label_slices(
        (),
        "AreaShape_FormFactor",
        np.zeros((2, 3, 4), dtype=np.int32),
        object_name=NUCLEI,
    )

    assert len(value_slices) == 2
    assert all(value_slice.size == 0 for value_slice in value_slices)


def test_measurement_lookup_rejects_missing_feature_for_nonempty_objects():
    with pytest.raises(ValueError, match="AreaShape_FormFactor"):
        measurement_values_for_label_slices(
            (),
            "AreaShape_FormFactor",
            np.array([[1, 0], [0, 0]], dtype=np.int32),
            object_name=NUCLEI,
        )


def test_calculate_math_records_object_indexed_measurements():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            "PriorMeasurements": _plan("PriorMeasurements", ArtifactKind.MEASUREMENTS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    labels = np.array([[1, 2], [0, 0]], dtype=np.int32)
    adapter.add_objects(NUCLEI, labels)
    adapter.add_measurements(
        "PriorMeasurements",
        [
            {
                "object_name": NUCLEI,
                "object_label": 1,
                "mean_intensity": 10.0,
                "area": 20.0,
            },
            {
                "object_name": NUCLEI,
                "object_label": 2,
                "mean_intensity": 20.0,
                "area": 80.0,
            },
        ],
        object_name=NUCLEI,
    )
    executor = _executor(
        CALCULATE_MATH,
        (ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
        runtime_artifact_inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
    )

    result = executor.run(
        get_function(CALCULATE_MATH),
        np.zeros((2, 2), dtype=np.float32),
        cellprofiler_runtime=adapter,
        output_name="Ratio",
        operation="Divide",
        operand1_feature="Intensity_MeanIntensity_CropBlue",
        operand2_feature="AreaShape_Area",
        operand1_object_name=NUCLEI,
        operand2_object_name=NUCLEI,
        dtype_config=DtypeConfig(),
    )
    measurements = adapter.get_measurements(MEASUREMENTS)

    np.testing.assert_array_equal(result, np.zeros((2, 2), dtype=np.float32))
    assert measurements.object_name == NUCLEI
    assert [row["object_name"] for row in measurements.rows] == [NUCLEI, NUCLEI]
    assert [row["object_label"] for row in measurements.rows] == [1, 2]
    assert [row["feature_name"] for row in measurements.rows] == [
        "Math_Ratio",
        "Math_Ratio",
    ]
    np.testing.assert_allclose(
        [row["result_value"] for row in measurements.rows],
        [0.5, 0.25],
    )
    np.testing.assert_allclose(
        measurement_values_for_feature(
            adapter.measurement_tables_for_object(NUCLEI),
            "Math_Ratio",
            object_count=2,
            object_name=NUCLEI,
        ),
        np.array([0.5, 0.25]),
    )


def test_current_shape_vector_source_derives_area_shape_vectors_from_label_stack():
    labels = np.zeros((2, 3, 4), dtype=np.int32)
    labels[0, 0:2, 0:2] = 1
    labels[0, 2, 0:2] = 2
    labels[1, 0, 0:3] = 1
    labels[1, 1:3, 2:4] = 2

    result = CurrentObjectShapeFeatureVectorSourceStrategy().current_label_shape_vector(
        "AreaShape_Area",
        labels,
    )

    assert result.vector is not None
    assert len(result.vector.slices) == 2
    np.testing.assert_allclose(result.vector.slices[0], [4.0, 2.0])
    np.testing.assert_allclose(result.vector.slices[1], [3.0, 4.0])


def test_calculate_math_pads_missing_same_object_operand_values():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            "PriorMeasurements": _plan("PriorMeasurements", ArtifactKind.MEASUREMENTS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    labels = np.array([[1, 2], [0, 0]], dtype=np.int32)
    adapter.add_objects(NUCLEI, labels)
    adapter.add_measurements(
        "PriorMeasurements",
        [
            {
                "object_name": NUCLEI,
                "object_label": 1,
                "mean_intensity": 10.0,
                "area": 20.0,
            },
            {
                "object_name": NUCLEI,
                "object_label": 2,
                "mean_intensity": 20.0,
            },
        ],
        object_name=NUCLEI,
    )
    executor = _executor(
        CALCULATE_MATH,
        (ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
        runtime_artifact_inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
    )

    executor.run(
        get_function(CALCULATE_MATH),
        np.zeros((2, 2), dtype=np.float32),
        cellprofiler_runtime=adapter,
        output_name="Ratio",
        operation="Divide",
        operand1_feature="Intensity_MeanIntensity_CropBlue",
        operand2_feature="AreaShape_Area",
        operand1_object_name=NUCLEI,
        operand2_object_name=NUCLEI,
        dtype_config=DtypeConfig(),
    )

    measurements = adapter.get_measurements(MEASUREMENTS)
    assert [row["object_label"] for row in measurements.rows] == [1, 2]
    np.testing.assert_allclose(
        [row["result_value"] for row in measurements.rows],
        [0.5, np.nan],
        equal_nan=True,
    )


def test_calculate_math_resolves_image_scoped_measurements_via_core_query():
    adapter, _filemanager = _adapter(
        {
            "PriorMeasurements": _plan("PriorMeasurements", ArtifactKind.MEASUREMENTS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    adapter.add_measurements(
        "PriorMeasurements",
        [
            {
                "slice_index": 0,
                "area_occupied": 17809.0,
                "source_image_name": "ColocalizedRegion",
            },
            {
                "slice_index": 0,
                "area_occupied": 30324.0,
                "source_image_name": "Objects1",
            },
        ],
    )
    executor = _executor(
        CALCULATE_MATH,
        (ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        inputs=(),
    )

    executor.run(
        get_function(CALCULATE_MATH),
        np.zeros((2, 2), dtype=np.float32),
        cellprofiler_runtime=adapter,
        output_name="Stain1Colocalized",
        operation="Divide",
        operand1_feature="AreaOccupied_AreaOccupied_ColocalizedRegion",
        operand2_feature="AreaOccupied_AreaOccupied_Objects1",
        operand1_object_name=None,
        operand2_object_name=None,
        dtype_config=DtypeConfig(),
    )
    measurements = adapter.get_measurements(MEASUREMENTS)

    assert measurements.object_name is None
    assert len(measurements.rows) == 1
    row = measurements.rows[0]
    assert row["feature_name"] == "Math_Stain1Colocalized"
    assert row["operand1_value"] == 17809.0
    assert row["operand2_value"] == 30324.0
    assert row["result_value"] == pytest.approx(17809.0 / 30324.0)


def test_calculate_math_aligns_image_scoped_measurements_by_slice():
    adapter, _filemanager = _adapter(
        {
            "PriorMeasurements": _plan("PriorMeasurements", ArtifactKind.MEASUREMENTS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    adapter.add_measurements(
        "PriorMeasurements",
        [
            {
                "slice_index": 0,
                "area_occupied": 10.0,
                "source_image_name": "ColocalizedRegion",
            },
            {
                "slice_index": 1,
                "area_occupied": 15.0,
                "source_image_name": "ColocalizedRegion",
            },
            {
                "slice_index": 0,
                "area_occupied": 20.0,
                "source_image_name": "Objects1",
            },
            {
                "slice_index": 1,
                "area_occupied": 30.0,
                "source_image_name": "Objects1",
            },
        ],
    )
    executor = _executor(
        CALCULATE_MATH,
        (ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        inputs=(),
    )

    result = executor.run(
        get_function(CALCULATE_MATH),
        np.zeros((2, 2, 2), dtype=np.float32),
        cellprofiler_runtime=adapter,
        output_name="Stain1Colocalized",
        operation="Divide",
        operand1_feature="AreaOccupied_AreaOccupied_ColocalizedRegion",
        operand2_feature="AreaOccupied_AreaOccupied_Objects1",
        operand1_object_name=None,
        operand2_object_name=None,
        dtype_config=DtypeConfig(),
    )
    measurements = adapter.get_measurements(MEASUREMENTS)

    np.testing.assert_array_equal(result, np.zeros((2, 2, 2), dtype=np.float32))
    assert measurements.object_name is None
    assert [row["slice_index"] for row in measurements.rows] == [0, 1]
    assert [row["object_label"] for row in measurements.rows] == [None, None]
    np.testing.assert_allclose(
        [row["result_value"] for row in measurements.rows],
        [0.5, 0.5],
    )


def test_classify_objects_binds_runtime_measurement_values():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            "PriorMeasurements": _plan("PriorMeasurements", ArtifactKind.MEASUREMENTS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    labels = np.array([[1, 2], [0, 0]], dtype=np.int32)
    adapter.add_objects(NUCLEI, labels)
    adapter.add_measurements(
        "PriorMeasurements",
        [
            {
                "object_name": NUCLEI,
                "object_label": 1,
                "feature_name": "Math_Ratio",
                "result_value": 0.5,
            },
            {
                "object_name": NUCLEI,
                "object_label": 2,
                "feature_name": "Math_Ratio",
                "result_value": 0.8,
            },
        ],
        object_name=NUCLEI,
    )
    executor = _executor(
        "ClassifyObjectsSingleMeasurement",
        (ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
        runtime_artifact_inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
    )

    result = executor.run(
        get_function("ClassifyObjects"),
        np.zeros((3, 3), dtype=np.float32),
        cellprofiler_runtime=adapter,
        measurement_feature="Math_Ratio",
        bin_choice="even",
        bin_count=2,
        low_threshold=0.0,
        high_threshold=1.0,
        dtype_config=DtypeConfig(),
    )
    measurements = adapter.get_measurements(MEASUREMENTS)

    np.testing.assert_array_equal(result, np.zeros((3, 3), dtype=np.float32))
    assert measurements.object_name is None
    summary_rows = [row for row in measurements.rows if "slice_index" in row]
    object_rows = [row for row in measurements.rows if row.get("object_name") == NUCLEI]
    assert {
        (row["feature_name"], row["result_value"])
        for row in summary_rows
        if row["feature_name"].endswith("NumObjectsPerBin")
    } == {
        ("Classify_Bin_1_NumObjectsPerBin", 1),
        ("Classify_Bin_2_NumObjectsPerBin", 1),
    }
    assert {
        (row["object_label"], row["feature_name"], row["result_value"])
        for row in object_rows
    } == {
        (1, "Classify_Bin_1", 1),
        (1, "Classify_Bin_2", 0),
        (2, "Classify_Bin_1", 0),
        (2, "Classify_Bin_2", 1),
    }


def test_classify_objects_area_shape_uses_current_label_domain():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            "PriorMeasurements": _plan("PriorMeasurements", ArtifactKind.MEASUREMENTS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    labels = np.array([[1, 2], [0, 2]], dtype=np.int32)
    adapter.add_objects(NUCLEI, labels)
    adapter.add_measurements(
        "PriorMeasurements",
        [
            {
                "object_name": NUCLEI,
                "object_label": 1,
                "feature_name": "AreaShape_Area",
                "result_value": 99.0,
            },
            {
                "object_name": NUCLEI,
                "object_label": 2,
                "feature_name": "AreaShape_Area",
                "result_value": 99.0,
            },
        ],
        object_name=NUCLEI,
    )
    executor = _executor(
        "ClassifyObjectsSingleMeasurement",
        (ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
        runtime_artifact_inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
    )

    executor.run(
        get_function("ClassifyObjects"),
        np.zeros((2, 2), dtype=np.float32),
        cellprofiler_runtime=adapter,
        measurement_feature="AreaShape_Area",
        bin_choice="even",
        bin_count=2,
        low_threshold=0.0,
        high_threshold=2.0,
        dtype_config=DtypeConfig(),
    )

    assert {
        (row["feature_name"], row["result_value"])
        for row in adapter.get_measurements(MEASUREMENTS).rows
        if row.get("feature_name", "").endswith("NumObjectsPerBin")
    } == {
        ("Classify_Bin_1_NumObjectsPerBin", 1),
        ("Classify_Bin_2_NumObjectsPerBin", 1),
    }


def test_classify_objects_binds_custom_threshold_and_named_low_high_bins():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            "PriorMeasurements": _plan("PriorMeasurements", ArtifactKind.MEASUREMENTS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    labels = np.array([[1, 2], [3, 0]], dtype=np.int32)
    adapter.add_objects(NUCLEI, labels)
    adapter.add_measurements(
        "PriorMeasurements",
        [
            {
                "object_name": NUCLEI,
                "object_label": 1,
                "feature_name": "Intensity_MaxIntensity_OrigGreen",
                "result_value": 0.05,
            },
            {
                "object_name": NUCLEI,
                "object_label": 2,
                "feature_name": "Intensity_MaxIntensity_OrigGreen",
                "result_value": 0.15,
            },
            {
                "object_name": NUCLEI,
                "object_label": 3,
                "feature_name": "Intensity_MaxIntensity_OrigGreen",
                "result_value": 0.80,
            },
        ],
        object_name=NUCLEI,
    )
    executor = _executor(
        "ClassifyObjectsSingleMeasurement",
        (ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
        runtime_artifact_inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
    )

    executor.run(
        get_function("ClassifyObjects"),
        np.zeros((2, 2), dtype=np.float32),
        cellprofiler_runtime=adapter,
        measurement_feature="Intensity_MaxIntensity_OrigGreen",
        bin_choice="custom",
        bin_count=3,
        low_threshold=0.0,
        high_threshold=1.0,
        wants_low_bin=True,
        wants_high_bin=True,
        custom_thresholds="0.2",
        bin_names="PH3Neg,PH3Pos",
        dtype_config=DtypeConfig(),
    )
    rows = adapter.get_measurements(MEASUREMENTS).rows

    assert {
        (row["feature_name"], row["result_value"])
        for row in rows
        if row.get("feature_name", "").endswith("NumObjectsPerBin")
    } == {
        ("Classify_PH3Neg_NumObjectsPerBin", 2),
        ("Classify_PH3Pos_NumObjectsPerBin", 1),
    }
    assert {
        (row["object_label"], row["feature_name"], row["result_value"])
        for row in rows
        if row.get("object_name") == NUCLEI and row["result_value"] == 1
    } == {
        (1, "Classify_PH3Neg", 1),
        (2, "Classify_PH3Neg", 1),
        (3, "Classify_PH3Pos", 1),
    }


def test_classify_objects_binds_repeated_single_measurement_rules():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            "PriorMeasurements": _plan("PriorMeasurements", ArtifactKind.MEASUREMENTS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    labels = np.array([[1, 2], [0, 0]], dtype=np.int32)
    adapter.add_objects(NUCLEI, labels)
    adapter.add_measurements(
        "PriorMeasurements",
        [
            {
                "object_name": NUCLEI,
                "object_label": 1,
                "feature_name": "AreaShape_Area",
                "result_value": 4.0,
            },
            {
                "object_name": NUCLEI,
                "object_label": 2,
                "feature_name": "AreaShape_Area",
                "result_value": 12.0,
            },
            {
                "object_name": NUCLEI,
                "object_label": 1,
                "feature_name": "Intensity_MeanIntensity_DNA",
                "result_value": 0.02,
            },
            {
                "object_name": NUCLEI,
                "object_label": 2,
                "feature_name": "Intensity_MeanIntensity_DNA",
                "result_value": 0.2,
            },
        ],
        object_name=NUCLEI,
    )
    executor = _executor(
        "ClassifyObjectsSingleMeasurement",
        (ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
        runtime_artifact_inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
    )

    executor.run(
        get_function("ClassifyObjects"),
        np.zeros((3, 3), dtype=np.float32),
        cellprofiler_runtime=adapter,
        classification_rules=(
            {
                "measurement_feature": "AreaShape_Area",
                "bin_choice": "custom",
                "custom_thresholds": "0,5,20",
                "bin_names": "Small,Large",
            },
            {
                "measurement_feature": "Intensity_MeanIntensity_DNA",
                "bin_choice": "custom",
                "custom_thresholds": "0.05",
                "wants_low_bin": True,
                "wants_high_bin": True,
                "bin_names": "White,Red",
            },
        ),
        dtype_config=DtypeConfig(),
    )
    rows = adapter.get_measurements(MEASUREMENTS).rows

    assert {
        (row["feature_name"], row["result_value"])
        for row in rows
        if row.get("feature_name", "").endswith("NumObjectsPerBin")
    } == {
        ("Classify_Small_NumObjectsPerBin", 1),
        ("Classify_Large_NumObjectsPerBin", 1),
        ("Classify_White_NumObjectsPerBin", 1),
        ("Classify_Red_NumObjectsPerBin", 1),
    }
    assert {
        (row["object_label"], row["feature_name"], row["result_value"])
        for row in rows
        if row.get("object_name") == NUCLEI
    } == {
        (1, "Classify_Small", 1),
        (1, "Classify_Large", 0),
        (1, "Classify_White", 1),
        (1, "Classify_Red", 0),
        (2, "Classify_Small", 0),
        (2, "Classify_Large", 1),
        (2, "Classify_White", 0),
        (2, "Classify_Red", 1),
    }


def test_classify_objects_slices_runtime_measurements_with_label_stack():
    adapter, _filemanager = _adapter(
        {
            NUCLEI: _plan(NUCLEI, ArtifactKind.OBJECT_LABELS),
            "PriorMeasurements": _plan("PriorMeasurements", ArtifactKind.MEASUREMENTS),
            MEASUREMENTS: _plan(MEASUREMENTS, ArtifactKind.MEASUREMENTS),
        }
    )
    labels = np.array(
        [
            [[1, 2], [0, 0]],
            [[3, 4], [0, 0]],
        ],
        dtype=np.int32,
    )
    adapter.add_objects(NUCLEI, labels)
    adapter.add_measurements(
        "PriorMeasurements",
        [
            {
                "object_name": NUCLEI,
                "object_label": label,
                "area": float(label),
            }
            for label in (1, 2, 3, 4)
        ],
        object_name=NUCLEI,
    )
    executor = _executor(
        "ClassifyObjectsSingleMeasurement",
        (ArtifactSpec(MEASUREMENTS, ArtifactKind.MEASUREMENTS),),
        inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
        runtime_artifact_inputs=(ArtifactSpec(NUCLEI, ArtifactKind.OBJECT_LABELS),),
    )

    result = executor.run(
        get_function("ClassifyObjects"),
        np.zeros((2, 2), dtype=np.float32),
        cellprofiler_runtime=adapter,
        measurement_feature="AreaShape_Area",
        bin_choice="even",
        bin_count=2,
        low_threshold=0.0,
        high_threshold=4.0,
        dtype_config=DtypeConfig(),
    )
    measurements = adapter.get_measurements(MEASUREMENTS)

    assert result.shape == (2, 2)
    summary_rows = [
        row
        for row in measurements.rows
        if row.get("feature_name", "").endswith("NumObjectsPerBin")
    ]
    assert {
        (row["slice_index"], row["feature_name"], row["result_value"])
        for row in summary_rows
    } == {
        (0, "Classify_Bin_1_NumObjectsPerBin", 2),
        (0, "Classify_Bin_2_NumObjectsPerBin", 0),
        (1, "Classify_Bin_1_NumObjectsPerBin", 0),
        (1, "Classify_Bin_2_NumObjectsPerBin", 2),
    }
    assert {
        (row["object_label"], row["feature_name"], row["result_value"])
        for row in measurements.rows
        if row.get("object_name") == NUCLEI and row["result_value"] == 1
    } == {
        (1, "Classify_Bin_1", 1),
        (2, "Classify_Bin_1", 1),
        (3, "Classify_Bin_2", 1),
        (4, "Classify_Bin_2", 1),
    }


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

    @declared_processing_contract(ProcessingContract.PURE_2D)
    def measure(image_arg, *, labels):
        seen_images.append((image_arg.copy(), labels.copy()))
        return image_arg, [{"object_count": int(labels.max())}]

    result = executor.run(measure, image, cellprofiler_runtime=adapter)
    measurements = adapter.get_measurements(MEASUREMENTS)

    assert len(seen_images) == 4
    for measurement_image, measurement_labels in seen_images:
        assert measurement_image.shape == measurement_labels.shape == (4, 5)
    np.testing.assert_array_equal(result, image)
    assert _measurement_rows_for_assertion(measurements) == [
        {
            "object_count": 1,
            "slice_index": 0,
            "object_name": NUCLEI,
            "image_number": 1,
        },
        {
            "object_count": 1,
            "slice_index": 1,
            "object_name": NUCLEI,
            "image_number": 2,
        },
        {
            "object_count": 2,
            "slice_index": 0,
            "object_name": CELLS,
            "image_number": 1,
        },
        {
            "object_count": 2,
            "slice_index": 1,
            "object_name": CELLS,
            "image_number": 2,
        },
    ]


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

    @declared_processing_contract(ProcessingContract.FLEXIBLE)
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
    assert _measurement_rows_for_assertion(measurements) == [
        {
            "mean": 3.0,
            "label": 1,
            "object_name": NUCLEI,
            "source_image_name": DNA_IMAGE,
            "slice_index": 0,
        },
        {
            "mean": 3.0,
            "label": 2,
            "object_name": CELLS,
            "source_image_name": DNA_IMAGE,
            "slice_index": 0,
        },
        {
            "mean": 9.0,
            "label": 1,
            "object_name": NUCLEI,
            "source_image_name": "PH3",
            "slice_index": 0,
        },
        {
            "mean": 9.0,
            "label": 2,
            "object_name": CELLS,
            "source_image_name": "PH3",
            "slice_index": 0,
        },
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

    @declared_processing_contract(ProcessingContract.FLEXIBLE)
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
    assert measurements.object_name is None
    assert _measurement_rows_for_assertion(measurements) == [
        {"mean_children_per_parent": 2.0, "slice_index": 0},
        {
            "object_name": CELLS,
            "object_label": 1,
            "children_nuclei_count": 2,
            "slice_index": 0,
        },
        {
            "object_name": NUCLEI,
            "object_label": 1,
            "parent_cells": 1,
            "slice_index": 0,
        },
        {
            "object_name": NUCLEI,
            "object_label": 2,
            "parent_cells": 1,
            "slice_index": 0,
        },
    ]
    assert adapter.measurement_tables_for_object(CELLS) == (measurements,)
