from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from polystore.base import DataSink
from polystore.napari_stream import NapariStreamingBackend
from polystore.filemanager import FileManager
from polystore.streaming import (
    StreamingBatchMessageBuilder,
    StreamingBatchMessageRequest,
)
from polystore.streaming.viewer_transport import (
    ViewerMicroscopeHandlerABC,
    ViewerStreamKwarg,
)
from zmqruntime.viewer_protocol import ViewerTransportEndpoint

from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents
from openhcs.core.axis_filter import StepAxisFilterResolution, StepAxisFilterSet
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    GroupLineageSourceRelation,
    ImageArtifactType,
    MaterializationSourceIdentityRelation,
    MeasurementsArtifactType,
    MetadataArtifactType,
    ObjectLabelsArtifactType,
    SpecialArtifactType,
)
from openhcs.core.compiled_step_plan import (
    CompiledStepPlan,
    RuntimeArtifactMaterializationPlan,
)
from openhcs.core.config import WellFilterMode
from openhcs.core.callable_contract import CallableContract
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
)
from openhcs.core.runtime_artifact_values import (
    RuntimeValue,
)
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
    image_payload_metadata,
)
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectLabelVariantData,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
    RuntimePlaneProjection,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    CompiledFunctionPattern,
    FunctionInvocationKey,
)
from openhcs.core.source_image_provenance import (
    SourceImageProvenancePlanes,
)
from openhcs.core.source_metadata import (
    SOURCE_PLANE_COUNT_FIELD,
    SOURCE_PLANE_INDEX_FIELD,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.steps.function_artifact_materialization import (
    ArtifactMaterializationBackendPlan,
    PersistentArtifactMaterializationTargetPlan,
    StreamingOnlyArtifactMaterializationTargetPlan,
    actual_materialization_records,
    materialized_artifact_output_paths,
    materialize_artifact_outputs,
    observed_materialized_artifact_output_paths,
    planned_materialization_preview,
    runtime_artifact_materializations,
)
from openhcs.core.steps.function_output_identity import (
    IncompleteFunctionOutputFilenameIdentityError,
)
from openhcs.core.steps.function_runtime import FunctionOutputContextStrategy
from openhcs.core.streaming_config_factory import (
    StreamingViewerPresentation,
    StreamingViewerRuntimeConfig,
    StreamingViewerSurface,
)
from openhcs.processing.materialization import (
    CsvOptions,
    JsonOptions,
    ROIOptions,
    csv_only,
    json_only,
    roi_zip,
    tiff_stack,
)
from openhcs.processing.materialization.core import (
    MaterializationSpec,
    Output,
    materialization_outputs,
)
from openhcs.processing.materialization.options import (
    ImageFileOptions,
    MaterializedFilenameIdentity,
)
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.microscopes.imagexpress import ImageXpressFilenameParser
from polystore.streaming.viewer_transport import ViewerDisplayConfigABC


class StreamingConfigStub(ViewerDisplayConfigABC):
    backend = SimpleNamespace(value="napari_stream")
    COMPONENT_ORDER = AllComponents.ordered_names()
    host = "127.0.0.1"
    transport_mode = "tcp"
    colormap = SimpleNamespace(value="gray")
    variable_size_handling = SimpleNamespace(value="pad_to_max")

    def __init__(self, port):
        self.port = port

    def component_modes(self):
        return {component: "stack" for component in self.COMPONENT_ORDER}

    def display_payload_extra(self):
        return {
            "colormap": self.colormap.value,
            "variable_size_handling": self.variable_size_handling.value,
        }

    def streaming_viewer_surface(self, _context):
        return StreamingViewerSurface(
            runtime_config=StreamingViewerRuntimeConfig(
                transport_endpoint=ViewerTransportEndpoint(
                    host=self.host,
                    port=self.port,
                    transport_mode=self.transport_mode,
                ),
                persistent=False,
                presentation=StreamingViewerPresentation(title="Napari"),
            ),
            display_config=self,
            source=self.viewer_source(_context),
        )

    @staticmethod
    def viewer_source(_context):
        from polystore.streaming.viewer_transport import ViewerStreamSourceIdentity

        return ViewerStreamSourceIdentity(
            microscope_handler=_context.microscope_handler,
            plate_path=_context.plate_path,
        )


def streaming_config_stub(port=5555):
    return StreamingConfigStub(port)


def stream_request_from_backend_kwargs(backend_kwargs):
    return backend_kwargs["napari_stream"].values.to_kwargs()[
        ViewerStreamKwarg.STREAM_REQUEST.value
    ]


class MetadataHandlerStub:
    def __init__(self, values=None):
        self.values = values or {}

    def find_metadata_file(self, root):
        return Path(root) / "openhcs_metadata.json"

    def get_component_values(self, _root, component):
        return self.values.get(component, {})


class FilenameParserStub:
    def parse_filename(self, filename):
        import re

        positional_match = re.match(
            r"(?P<well>\d+)_(?P<site>POS\d+)_(?P<channel>[A-Za-z])"
            r"(?P<extension>\.[^.]+)?$",
            Path(filename).name,
        )
        if positional_match is not None:
            parsed = {
                key: value
                for key, value in positional_match.groupdict().items()
                if value is not None
            }
            parsed.setdefault("z_index", "1")
            parsed.setdefault("timepoint", "1")
            return parsed

        match = re.match(
            r"(?P<well>[A-Z]\d{2})_s(?P<site>\d+)_w(?P<channel>\d+)"
            r"(?:_z(?P<z_index>\d+))?(?:_t(?P<timepoint>\d+))?"
            r"(?P<extension>\.[^.]+)?$",
            Path(filename).name,
        )
        if match is None:
            return None
        parsed = {
            key: value for key, value in match.groupdict().items() if value is not None
        }
        parsed.setdefault("z_index", "1")
        parsed.setdefault("timepoint", "1")
        return parsed

    def construct_filename(
        self,
        *,
        well,
        site,
        channel,
        z_index=1,
        timepoint=1,
        extension=".tif",
    ):
        if str(site).startswith("POS"):
            return f"{well}_{site}_{channel}{extension}"
        return (
            f"{well}_s{int(site):03d}_w{int(channel)}"
            f"_z{int(z_index):03d}_t{int(timepoint):03d}{extension}"
        )


class MicroscopeHandlerStub(ViewerMicroscopeHandlerABC):
    def __init__(self, parser, metadata_handler):
        self.parser = parser
        self.metadata_handler = metadata_handler
        self.microscope_type = "test"


class FileManagerStub:
    def __init__(self):
        self.memory = {}
        self.directories = set()
        self.saved = []

    def _get_backend(self, backend):
        return BackendStub(backend)

    def exists(self, path, backend):
        return path in self.memory

    def ensure_directory(self, path, backend):
        self.directories.add((str(path), backend))

    def load(self, path, backend):
        return self.memory[path]

    def save(self, content, path, backend, **kwargs):
        self.saved.append((content, path, backend, kwargs))

    def save_batch(self, contents, paths, backend, **kwargs):
        self.saved.extend(
            (content, path, backend, kwargs)
            for content, path in zip(contents, paths, strict=True)
        )


class BackendStub(DataSink):
    requires_filesystem_validation = False

    def __init__(self, backend):
        self.backend = backend

    def supports_file_path(self, path):
        if self.backend != "napari_stream":
            return True
        name = Path(path).name.lower()
        return name.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".roi.zip"))

    def contextual_save_kwargs(self, *, images_dir):
        del images_dir
        return {}

    def save(self, data, identifier, **kwargs):
        raise AssertionError("FileManagerStub owns save interception")

    def save_batch(self, data_list, identifiers, **kwargs):
        raise AssertionError("FileManagerStub owns save interception")


def _plan(
    output_plan,
    *,
    streaming_configs=None,
    variable_components=(),
    group_by_value=None,
    execution_group_value=None,
    compiled_function_pattern=None,
) -> CompiledStepPlan:
    variable_components = tuple(variable_components)
    if execution_group_value is None:
        execution_group_value = group_by_value
    execution_group_scope = ComponentGroupScope.ungrouped()
    if execution_group_value is not None:
        execution_group_component = AllComponents.from_value(execution_group_value)
        if execution_group_component is None:
            raise ValueError(
                f"Unknown execution group component {execution_group_value!r}."
            )
        execution_group_scope = ComponentGroupScope.dynamic(execution_group_component)
    return CompiledStepPlan(
        step_index=6,
        step_name="measure",
        step_type="FunctionStep",
        axis_id="A01",
        artifact_outputs={plan.ref(): plan for plan in (output_plan,)},
        streaming_configs={} if streaming_configs is None else streaming_configs,
        analysis_results_dir="/analysis",
        pipeline_position=7,
        step_scope_id="measure-scope-7",
        output_dir=Path("/images"),
        input_dir=Path("/tmp/input"),
        read_backend="memory",
        write_backend="memory",
        group_by=(
            GroupBy(group_by_value) if group_by_value is not None else GroupBy.NONE
        ),
        execution_group_scope=execution_group_scope,
        variable_components=variable_components,
        compiled_function_pattern=(
            _artifact_only_compiled_pattern()
            if compiled_function_pattern is None
            else compiled_function_pattern
        ),
    )


def _artifact_only_compiled_pattern() -> CompiledFunctionPattern:
    def passthrough(image):
        return image

    contract = CallableContract.from_callable(passthrough)
    return CompiledFunctionPattern(
        groups=(
            CompiledFunctionGroup(
                group_key=DEFAULT_GROUP_KEY,
                invocations=(
                    CompiledFunctionInvocation(
                        key=FunctionInvocationKey.from_contract(
                            contract,
                            DEFAULT_GROUP_KEY,
                            0,
                        ),
                        contract=contract,
                    ),
                ),
            ),
        ),
        is_grouped=False,
    )


def _main_flow_output_compiled_pattern(
    output_plan: ArtifactOutputPlan,
) -> CompiledFunctionPattern:
    @artifact_outputs(ArtifactSpec.output(output_plan.name, output_plan.artifact_type))
    def publish(image):
        return image

    contract = CallableContract.from_callable(publish)
    return CompiledFunctionPattern(
        groups=(
            CompiledFunctionGroup(
                group_key=DEFAULT_GROUP_KEY,
                invocations=(
                    CompiledFunctionInvocation(
                        key=FunctionInvocationKey.from_contract(
                            contract,
                            DEFAULT_GROUP_KEY,
                            0,
                        ),
                        contract=contract,
                        artifact_output_plans=(output_plan,),
                    ),
                ),
            ),
        ),
        is_grouped=False,
    )


class ContextStub:
    pass


def _context(filemanager):
    context = ContextStub()
    context.filemanager = filemanager
    context.runtime_value_store = RuntimeValueStore()
    context.microscope_handler = MicroscopeHandlerStub(
        parser=FilenameParserStub(),
        metadata_handler=MetadataHandlerStub(
            {"channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"}}
        ),
    )
    context.plate_path = Path("/tmp/plate")
    context.input_dir = Path("/tmp/plate/images")
    context.owned_wells = ["A01"]
    context.axis_id = "A01"
    context.step_axis_filters = {}
    return context


def test_named_artifact_streaming_respects_compiled_streaming_filter():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
        materialization=roi_zip(),
    )
    config = streaming_config_stub()
    plan = _plan(output_plan, streaming_configs={"napari_stream": config})
    context = _context(FileManagerStub())
    context.step_axis_filters = {
        plan.step_index: StepAxisFilterSet(
            {
                type(config): StepAxisFilterResolution(
                    resolved_axis_values=frozenset({"B03"}),
                    filter_mode=WellFilterMode.INCLUDE,
                    original_filter="B03",
                )
            }
        )
    }
    materialization = SimpleNamespace(
        output_plan=output_plan,
        record=SimpleNamespace(
            key=SimpleNamespace(scope=SimpleNamespace(value_text=None))
        ),
    )
    target = StreamingOnlyArtifactMaterializationTargetPlan()

    excluded = target.backend_plan(plan, context, materialization)

    assert excluded.streaming_viewer_surfaces == {}

    context.axis_id = "B03"
    included = target.backend_plan(plan, context, materialization)

    assert tuple(included.streaming_viewer_surfaces) == ("napari_stream",)


def test_viewer_output_expectation_omits_empty_stream_payload() -> None:
    filemanager = FileManager({"napari_stream": NapariStreamingBackend()})
    context = _context(filemanager)
    viewer_surface = streaming_config_stub().streaming_viewer_surface(context)
    backend_plan = ArtifactMaterializationBackendPlan(
        persistent_backend_kwargs={},
        streaming_viewer_surfaces={"napari_stream": viewer_surface},
    )

    assert not backend_plan.supports_stream_output(
        filemanager,
        Output(path="/analysis/A01_neurites.graph.roi.zip", content=[]),
    )


def test_planned_materialization_preview_uses_declared_candidate_paths():
    output_plan = ArtifactOutputPlan(
        name="cell_counts",
        path="/memory/A01_cell_counts_step7.pkl",
        artifact_type=SpecialArtifactType,
        materialization=csv_only(),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)

    preview = planned_materialization_preview(
        context=context,
        plan=_plan(output_plan),
        output_key=output_plan.name,
        output_plan=output_plan,
    )

    assert preview is not None
    assert preview.runtime_metadata_can_refine_paths is True
    assert (
        preview.paths[0].shared_output_stem
        == "/analysis/A01_cell_counts_step7"
    )
    assert preview.paths[0].candidate_paths == (
        "/analysis/A01_cell_counts_step7_details.csv",
    )


def test_slice_aligned_object_label_arrays_preserve_source_slice_metadata():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    source = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A02_s001_w1_z001_t001.tif",
                "/input/A02_s002_w1_z001_t001.tif",
            ),
            component_metadata=(
                {"well": "A02", "site": 1, "channel": 1},
                {"well": "A02", "site": 2, "channel": 1},
            ),
        ),
    ).payload_with(np.zeros((2, 8, 8), dtype=np.float32), None)
    expected_labels = np.stack(
        (
            np.array([[0, 1], [0, 0]], dtype=np.int32),
            np.array([[0, 2], [0, 0]], dtype=np.int32),
        )
    )
    label_slices = RuntimeSliceAlignedValues(tuple(expected_labels))

    contextualized = FunctionOutputContextStrategy.for_output_plan(
        output_plan
    ).contextualize(
        source,
        label_slices,
        output_plan,
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )
    runtime_value = RuntimeValue.normalize(
        output_plan,
        contextualized,
        axis_id="A02",
    )

    assert isinstance(runtime_value.data, ObjectLabelSet)
    assert runtime_value.data.name == output_plan.name
    np.testing.assert_array_equal(runtime_value.data.labels, expected_labels)
    assert runtime_value.data.dtype == np.dtype(np.int32)
    assert runtime_value.data.representation is (
        contextualized.value_for_slice(0).representation
    )
    assert runtime_value.data.domain == ObjectLabelDomain(
        declared_object_id_domains=((1,), (2,)),
        scope=ObjectLabelDomainScope.PLANE,
    )
    assert runtime_value.data.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert runtime_value.data.source_spatial_domain == SourceSpatialDomain(
        origin_yx=(0, 0),
        source_shape_yx=(8, 8),
    )
    assert runtime_value.data.source_path is None
    assert dict(runtime_value.data.source_component_metadata or {}) == {
        "well": "A02",
        "channel": 1,
        "extension": ".tif",
    }
    assert runtime_value.data.source_image_names == ()
    assert runtime_value.data.source_image_provenance_planes == (
        image_payload_metadata(source).source_image_provenance_planes
    )
    assert runtime_value.data.dimensions == ()
    assert runtime_value.data.source_image_name is None
    assert runtime_value.materialization_payload() is runtime_value.data
    assert output_plan.materialization_payload(runtime_value) is runtime_value.data
    assert runtime_value.data.source_image_provenance_planes.paths == (
        "/input/A02_s001_w1_z001_t001.tif",
        "/input/A02_s002_w1_z001_t001.tif",
    )
    assert tuple(
        dict(item)
        for item in runtime_value.data.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A02", "site": 1, "channel": 1},
        {"well": "A02", "site": 2, "channel": 1},
    )


def test_image_outputs_merge_source_provenance_when_output_already_has_metadata():
    output_plan = ArtifactOutputPlan(
        name="Corrected",
        path="/memory/Corrected.pkl",
        artifact_type=ImageArtifactType,
    )
    source = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A02_s001_w1_z001_t001.tif",
                "/input/A02_s002_w1_z001_t001.tif",
            ),
            component_metadata=(
                {"well": "A02", "site": 1, "channel": 1},
                {"well": "A02", "site": 2, "channel": 1},
            ),
        )
    ).payload_with(np.zeros((2, 8, 8), dtype=np.float32), None)
    output = ImageMetadataPayload(
        data=np.ones((2, 8, 8), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )

    contextualized = FunctionOutputContextStrategy.for_output_plan(
        output_plan
    ).contextualize(
        source,
        output,
        output_plan,
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    metadata = image_payload_metadata(contextualized)
    assert metadata.source_dtype == "float32"
    assert metadata.source_image_provenance_planes.paths == (
        "/input/A02_s001_w1_z001_t001.tif",
        "/input/A02_s002_w1_z001_t001.tif",
    )
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A02", "site": 1, "channel": 1},
        {"well": "A02", "site": 2, "channel": 1},
    )


def test_object_label_payload_stack_preserves_source_slice_metadata():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    source = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A02_s001_w1_z001_t001.tif",
                "/input/A02_s002_w1_z001_t001.tif",
            ),
            component_metadata=(
                {"well": "A02", "site": 1, "channel": 1},
                {"well": "A02", "site": 2, "channel": 1},
            ),
        )
    ).payload_with(np.zeros((2, 8, 8), dtype=np.float32), None)
    expected_labels = np.stack(
        (
            np.array([[0, 1], [0, 0]], dtype=np.int32),
            np.array([[0, 2], [0, 0]], dtype=np.int32),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=expected_labels),
        domain=ObjectLabelDomain(
            declared_object_count=2,
        ),
    )

    contextualized = FunctionOutputContextStrategy.for_output_plan(
        output_plan
    ).contextualize(
        source,
        labels,
        output_plan,
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )
    runtime_value = RuntimeValue.normalize(
        output_plan,
        contextualized,
        axis_id="A02",
    )

    assert isinstance(runtime_value.data, ObjectLabelSet)
    assert runtime_value.data.name == output_plan.name
    np.testing.assert_array_equal(runtime_value.data.labels, expected_labels)
    assert runtime_value.data.dtype == np.dtype(np.int32)
    assert runtime_value.data.representation is labels.representation
    assert runtime_value.data.domain == labels.domain
    assert runtime_value.data.plane_axis is labels.plane_axis
    assert runtime_value.data.source_spatial_domain == SourceSpatialDomain(
        value_name="Object-label"
    )
    assert runtime_value.data.source_provenance == (
        image_payload_metadata(source).source_provenance
    )
    assert runtime_value.data.dimensions == ()
    assert runtime_value.data.source_image_name is None
    assert runtime_value.materialization_payload() is runtime_value.data
    assert output_plan.materialization_payload(runtime_value) is runtime_value.data
    assert runtime_value.data.source_image_provenance_planes.paths == (
        "/input/A02_s001_w1_z001_t001.tif",
        "/input/A02_s002_w1_z001_t001.tif",
    )
    assert tuple(
        dict(item)
        for item in runtime_value.data.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A02", "site": 1, "channel": 1},
        {"well": "A02", "site": 2, "channel": 1},
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
        RuntimeValue.normalize(output_plan, {"x": "from-runtime"}, axis_id="A01"),
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

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert materialized == [
        ({"x": "from-runtime"}, "/analysis/A01_positions_step7.roi.zip")
    ]


def test_materialize_artifact_outputs_attaches_image_schema_provenance(monkeypatch):
    output_plan = ArtifactOutputPlan(
        name="converted_image",
        path="/memory/converted_image.pkl",
        artifact_type=ImageArtifactType,
        materialization=tiff_stack(),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.from_output_plan(
            output_plan,
            ImagePayloadMetadata(
                source_component_metadata={
                    "well": "A01",
                    "site": "1",
                    "channel": "3",
                    "z_index": "1",
                    SOURCE_PLANE_INDEX_FIELD: "0",
                    SOURCE_PLANE_COUNT_FIELD: "3",
                }
            ).payload_with(np.zeros((3, 5, 7), dtype=np.float32)),
            execution_scope=RuntimeExecutionAxisScope(axis_id="A01"),
        ),
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

    materialize_artifact_outputs(
        filemanager,
        _plan(
            output_plan,
            variable_components=(
                VariableComponents.Z_INDEX,
                VariableComponents.CHANNEL,
            ),
        ),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert len(materialized) == 1
    data, path = materialized[0]
    assert path == "/images/A01_s001_w3_z001_t001.tif"
    assert isinstance(data, ImageMetadataPayload)
    assert dict(image_payload_metadata(data).source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "3",
        "z_index": "1",
        SOURCE_PLANE_INDEX_FIELD: "0",
        SOURCE_PLANE_COUNT_FIELD: "3",
    }


def test_materialize_artifact_outputs_uses_output_plan_axes_for_source_named_runtime_planes(
    monkeypatch,
):
    materialization = MaterializationSpec(
        ImageFileOptions(
            filename_suffix="_saved.tif",
            filename_identity=MaterializedFilenameIdentity.SOURCE_IDENTITY,
        )
    )
    output_plan = ArtifactOutputPlan(
        name="SavedImages",
        path="/memory/SavedImages.pkl",
        artifact_type=ImageArtifactType,
        materialization=materialization,
        variable_components=(VariableComponents.SITE,),
    )
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
            "z_index": "1",
            "timepoint": "1",
            "extension": ".tif",
        },
    ).payload_with(np.zeros((3, 4, 5), dtype=np.uint8), None)
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, payload, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    output_paths = []

    def fake_materialize(spec, data, path, manager, *_args, **kwargs):
        output_paths.extend(
            output.path
            for output in materialization_outputs(
                spec,
                data,
                path,
                manager,
                context=kwargs["context"],
                artifact_source_identity=kwargs["artifact_source_identity"],
                variable_components=kwargs["variable_components"],
            )
        )
        return output_paths[0]

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    plan = _plan(
        output_plan,
        variable_components=(VariableComponents.CHANNEL,),
    )
    plan.runtime_artifact_materialization = RuntimeArtifactMaterializationPlan(
        persistent_enabled=True,
        persistent_backend="disk",
    )
    expected_paths = tuple(
        Path(f"/images/A01_s{site:03d}_w1_z001_t001_saved.tif") for site in range(1, 4)
    )
    assert materialized_artifact_output_paths(plan, context) == expected_paths

    materialize_artifact_outputs(
        filemanager,
        plan,
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert output_paths == [str(path) for path in expected_paths]


def test_materialize_artifact_outputs_requires_runtime_store_record():
    output_plan = ArtifactOutputPlan(
        name="positions",
        path="/memory/positions.pkl",
        materialization=json_only(),
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = {"x": 1}
    context = _context(filemanager)

    with pytest.raises(RuntimeError, match="Missing RuntimeValueStore record"):
        materialize_artifact_outputs(
            filemanager,
            _plan(output_plan),
            PersistentArtifactMaterializationTargetPlan("disk"),
            context,
        )


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
        RuntimeValue.normalize(output_plan, {"x": 1}, axis_id="A01"),
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

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert materialized == [({"x": 1}, "/analysis/A01_positions_step7.roi.zip")]


def test_materialize_artifact_outputs_uses_runtime_record_identity_not_final_path(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/analysis/A01_measurements_step7.csv",
        artifact_type=MeasurementsArtifactType,
        materialization=csv_only(),
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/analysis/A01_measurements_step7.csv"},
    )
    runtime_output_plan = output_plan.for_group("1")
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(
            runtime_output_plan,
            MeasurementTable(
                name=runtime_output_plan.name,
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"object_id": 1, "area": 42},),
                    fields=(FieldSpec("object_id", int), FieldSpec("area", int)),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT,
                    runtime_output_plan.name,
                ),
            ),
            axis_id="A01",
        ),
        path="/memory/runtime_cache/measurements_1.pkl",
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

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert [(tuple(data), path) for data, path in materialized] == [
        (({"object_id": 1, "area": 42},), "/analysis/measurements_1.roi.zip")
    ]


def test_materialize_artifact_outputs_uses_declared_measurement_csv_spec(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        materialization=csv_only(),
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = [{"object_id": 1, "area": 42}]
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(
            output_plan,
            MeasurementTable(
                name=output_plan.name,
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"object_id": 1, "area": 42},),
                    fields=(FieldSpec("object_id", int), FieldSpec("area", int)),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT,
                    output_plan.name,
                ),
            ),
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

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    spec, data, path = materialized[0]
    assert isinstance(spec.outputs[0], CsvOptions)
    assert spec.outputs[0].filename_suffix == "_details.csv"
    assert tuple(data) == ({"object_id": 1, "area": 42},)
    assert path == "/analysis/A01_measurements_step7.roi.zip"


def test_multi_plane_measurement_materialization_uses_aggregate_artifact_name():
    output_plan = ArtifactOutputPlan(
        name="cell_counts",
        path="/memory/A01_s001_w1_cell_counts_step7.pkl",
        artifact_type=MeasurementsArtifactType,
        materialization=csv_only(),
        variable_components=(
            AllComponents.SITE,
            AllComponents.CHANNEL,
        ),
    )
    source_metadata = (
        {"well": "A01", "site": "1", "channel": "1"},
        {"well": "A01", "site": "1", "channel": "2"},
        {"well": "A01", "site": "2", "channel": "1"},
        {"well": "A01", "site": "2", "channel": "2"},
    )
    table = MeasurementTable(
        name=output_plan.name,
        rows=MeasurementSparseColumnarRows.from_rows(
            tuple(
                {
                    "slice_index": index,
                    "well": metadata["well"],
                    "site": metadata["site"],
                    "channel": metadata["channel"],
                    "cell_count": index + 1,
                }
                for index, metadata in enumerate(source_metadata)
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("well", str),
                FieldSpec("site", str),
                FieldSpec("channel", str),
                FieldSpec("cell_count", int),
            ),
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(
                f"/input/A01_s{metadata['site']}_w{metadata['channel']}.tif"
                for metadata in source_metadata
            ),
            component_metadata=source_metadata,
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT),
    )
    context = _context(FileManagerStub())
    context.runtime_value_store.record(
        RuntimeValue.normalize(
            output_plan,
            table,
            axis_id="A01",
        ),
        path=output_plan.path,
        backend="memory",
    )
    plan = _plan(
        output_plan,
        variable_components=(
            VariableComponents.SITE,
            VariableComponents.CHANNEL,
        ),
    )

    [materialization] = runtime_artifact_materializations(plan, context)

    assert str(materialization.base_path) == (
        "/analysis/A01_cell_counts_step7.roi.zip"
    )
    assert tuple(
        output.path for output in materialization.outputs(plan, context)
    ) == ("/analysis/A01_cell_counts_step7_details.csv",)


def test_multi_plane_special_output_uses_aggregate_artifact_name():
    output_plan = ArtifactOutputPlan(
        name="cell_counts",
        path="/memory/cell_counts.pkl",
        artifact_type=SpecialArtifactType,
        materialization=csv_only(),
    )
    context = _context(FileManagerStub())
    context.microscope_handler = MicroscopeHandlerStub(
        parser=ImageXpressFilenameParser(),
        metadata_handler=MetadataHandlerStub(),
    )
    context.runtime_value_store.record(
        RuntimeValue.normalize_for_execution_scope(
            output_plan,
            (
                {"slice_index": 0, "cell_count": 2},
                {"slice_index": 1, "cell_count": 3},
            ),
            execution_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=None,
                value=None,
                fixed_component_values=(
                    (AllComponents.Z_INDEX, "1"),
                    (AllComponents.TIMEPOINT, "1"),
                ),
            ),
        ),
        path=output_plan.path,
        backend="memory",
    )
    plan = _plan(
        output_plan,
        variable_components=(
            VariableComponents.SITE,
            VariableComponents.CHANNEL,
        ),
    )

    [materialization] = runtime_artifact_materializations(plan, context)

    assert str(materialization.base_path) == (
        "/analysis/A01_z_index-1_timepoint-1_cell_counts_step7.roi.zip"
    )
    assert tuple(
        output.path for output in materialization.outputs(plan, context)
    ) == (
        "/analysis/A01_z_index-1_timepoint-1_cell_counts_step7_details.csv",
    )


def test_scalar_special_output_preserves_complete_source_identity():
    output_plan = ArtifactOutputPlan(
        name="cell_counts",
        path="/memory/cell_counts.pkl",
        artifact_type=SpecialArtifactType,
        materialization=csv_only(),
    )
    context = _context(FileManagerStub())
    context.microscope_handler = MicroscopeHandlerStub(
        parser=ImageXpressFilenameParser(),
        metadata_handler=MetadataHandlerStub(),
    )
    context.runtime_value_store.record(
        RuntimeValue.normalize_for_execution_scope(
            output_plan,
            ({"slice_index": 0, "cell_count": 2},),
            execution_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=None,
                value=None,
                fixed_component_values=(
                    (AllComponents.SITE, "1"),
                    (AllComponents.CHANNEL, "2"),
                    (AllComponents.Z_INDEX, "1"),
                    (AllComponents.TIMEPOINT, "1"),
                ),
            ),
        ),
        path=output_plan.path,
        backend="memory",
    )
    plan = _plan(output_plan)

    [materialization] = runtime_artifact_materializations(plan, context)

    assert str(materialization.base_path) == (
        "/analysis/A01_s001_w2_z001_t001_cell_counts_step7.roi.zip"
    )
    assert tuple(
        output.path for output in materialization.outputs(plan, context)
    ) == (
        "/analysis/A01_s001_w2_z001_t001_cell_counts_step7_details.csv",
    )


def test_incomplete_scalar_special_output_keeps_strict_filename_failure():
    output_plan = ArtifactOutputPlan(
        name="cell_counts",
        path="/memory/cell_counts.pkl",
        artifact_type=SpecialArtifactType,
        materialization=csv_only(),
    )
    context = _context(FileManagerStub())
    context.microscope_handler = MicroscopeHandlerStub(
        parser=ImageXpressFilenameParser(),
        metadata_handler=MetadataHandlerStub(),
    )
    context.runtime_value_store.record(
        RuntimeValue.normalize_for_execution_scope(
            output_plan,
            ({"slice_index": 0, "cell_count": 2},),
            execution_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=None,
                value=None,
                fixed_component_values=(
                    (AllComponents.Z_INDEX, "1"),
                    (AllComponents.TIMEPOINT, "1"),
                ),
            ),
        ),
        path=output_plan.path,
        backend="memory",
    )

    with pytest.raises(
        IncompleteFunctionOutputFilenameIdentityError,
        match="Cannot construct FunctionStep output filename",
    ):
        runtime_artifact_materializations(_plan(output_plan), context)


def test_grouped_special_output_retains_group_coordinate_in_aggregate_name():
    output_plan = ArtifactOutputPlan(
        name="cell_counts",
        path="/memory/cell_counts.pkl",
        artifact_type=SpecialArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        materialization=csv_only(),
    )
    context = _context(FileManagerStub())
    context.microscope_handler = MicroscopeHandlerStub(
        parser=ImageXpressFilenameParser(),
        metadata_handler=MetadataHandlerStub(),
    )
    context.runtime_value_store.record(
        RuntimeValue.normalize_for_execution_scope(
            output_plan,
            (
                {"slice_index": 0, "cell_count": 2},
                {"slice_index": 1, "cell_count": 3},
            ),
            execution_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=AllComponents.CHANNEL,
                value="2",
                fixed_component_values=(
                    (AllComponents.Z_INDEX, "1"),
                    (AllComponents.TIMEPOINT, "1"),
                ),
            ),
        ),
        path=output_plan.path,
        backend="memory",
    )
    plan = _plan(
        output_plan,
        variable_components=(VariableComponents.SITE,),
    )

    [materialization] = runtime_artifact_materializations(plan, context)

    assert str(materialization.base_path) == (
        "/analysis/A01_channel-2_z_index-1_timepoint-1_cell_counts_step7.roi.zip"
    )
    assert tuple(
        output.path for output in materialization.outputs(plan, context)
    ) == (
        "/analysis/"
        "A01_channel-2_z_index-1_timepoint-1_cell_counts_step7_details.csv",
    )


def test_multi_plane_roi_aggregate_defers_source_filenames_to_plane_writer():
    output_plan = ArtifactOutputPlan(
        name="segmentation_masks",
        path="/memory/segmentation_masks.pkl",
        artifact_type=ObjectLabelsArtifactType,
        variable_components=(
            VariableComponents.SITE,
            VariableComponents.CHANNEL,
        ),
        materialization=roi_zip(),
    )
    source_metadata = tuple(
        {
            "well": "A01",
            "site": site,
            "channel": channel,
            "z_index": 1,
            "timepoint": 1,
            "extension": ".tif",
        }
        for site in (1, 2)
        for channel in (1, 2)
    )
    labels_array = np.zeros((4, 8, 8), dtype=np.int32)
    labels_array[:, 2:6, 3:7] = 1
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels_array),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,),) * 4,
            scope=ObjectLabelDomainScope.PLANE,
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(
                (
                    f"/input/A01_s{metadata['site']:03d}"
                    f"_w{metadata['channel']}_z001_t001.tif"
                )
                for metadata in source_metadata
            ),
            component_metadata=source_metadata,
        ),
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(8, 8)),
    )
    context = _context(FileManagerStub())
    context.microscope_handler = MicroscopeHandlerStub(
        parser=ImageXpressFilenameParser(),
        metadata_handler=MetadataHandlerStub(),
    )
    context.runtime_value_store.record(
        RuntimeValue.normalize_for_execution_scope(
            output_plan,
            labels,
            execution_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=None,
                value=None,
                fixed_component_values=(
                    (AllComponents.Z_INDEX, "1"),
                    (AllComponents.TIMEPOINT, "1"),
                ),
            ),
        ),
        path=output_plan.path,
        backend="memory",
    )
    plan = _plan(
        output_plan,
        variable_components=(
            VariableComponents.SITE,
            VariableComponents.CHANNEL,
        ),
    )

    [materialization] = runtime_artifact_materializations(plan, context)

    assert str(materialization.base_path) == (
        "/analysis/A01_z_index-1_timepoint-1_segmentation_masks_step7.roi.zip"
    )
    assert tuple(
        output.path
        for output in materialization.outputs(plan, context)
        if output.path.endswith(".roi.zip")
    ) == tuple(
        (
            f"/analysis/A01_s{site:03d}_w{channel}_z001_t001"
            "_segmentation_masks_step7_rois.roi.zip"
        )
        for site in (1, 2)
        for channel in (1, 2)
    )


def test_multi_plane_measurement_aggregate_names_retain_runtime_group_coordinate():
    output_plan = ArtifactOutputPlan(
        name="neurite_outgrowth_summary",
        path="/memory/neurite_outgrowth_summary.pkl",
        artifact_type=MeasurementsArtifactType,
        materialization=csv_only(),
        variable_components=(AllComponents.CHANNEL,),
    )
    context = _context(FileManagerStub())
    for site in ("1", "2"):
        source_metadata = tuple(
            {
                "well": "A01",
                "site": site,
                "channel": channel,
                "z_index": "1",
                "timepoint": "1",
            }
            for channel in ("1", "2")
        )
        table = MeasurementTable(
            name=output_plan.name,
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"number_of_cells": 1},),
                fields=(FieldSpec("number_of_cells", int),),
            ),
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=tuple(
                        f"/input/A01_s{site}_w{metadata['channel']}.tif"
                        for metadata in source_metadata
                    ),
                    component_metadata=source_metadata,
                )
            ),
            subject=MeasurementSubject(MeasurementScope.ARTIFACT),
        )
        context.runtime_value_store.record(
            RuntimeValue.normalize_for_execution_scope(
                output_plan,
                table,
                execution_scope=RuntimeExecutionAxisScope.from_raw(
                    "A01",
                    component=AllComponents.SITE,
                    value=site,
                    fixed_component_values=(
                        (AllComponents.Z_INDEX, "1"),
                        (AllComponents.TIMEPOINT, "1"),
                    ),
                ),
            ),
            path=output_plan.path,
            backend="memory",
        )

    materializations = runtime_artifact_materializations(
        _plan(
            output_plan,
            group_by_value="site",
            variable_components=(VariableComponents.CHANNEL,),
        ),
        context,
    )

    assert tuple(str(item.base_path) for item in materializations) == (
        "/analysis/A01_site-1_z_index-1_timepoint-1_"
        "neurite_outgrowth_summary_step7.roi.zip",
        "/analysis/A01_site-2_z_index-1_timepoint-1_"
        "neurite_outgrowth_summary_step7.roi.zip",
    )


def test_observed_materialized_paths_use_only_caller_owned_execution_records():
    output_plan = ArtifactOutputPlan(
        name="cell_counts",
        path="/memory/cell_counts.pkl",
        artifact_type=MeasurementsArtifactType,
        materialization=csv_only(),
        variable_components=(AllComponents.CHANNEL,),
    )
    context = _context(FileManagerStub())
    plan = _plan(
        output_plan,
        group_by_value="site",
        variable_components=(VariableComponents.CHANNEL,),
    )
    plan.runtime_artifact_materialization = RuntimeArtifactMaterializationPlan(
        persistent_enabled=True,
        persistent_backend="disk",
    )

    def record_for_site(site: str) -> None:
        source_metadata = tuple(
            {
                "well": "A01",
                "site": site,
                "channel": channel,
                "z_index": "1",
                "timepoint": "1",
            }
            for channel in ("1", "2")
        )
        context.runtime_value_store.record(
            RuntimeValue.normalize_for_execution_scope(
                output_plan,
                MeasurementTable(
                    name=output_plan.name,
                    rows=MeasurementSparseColumnarRows.from_rows(
                        ({"cell_count": int(site)},),
                        fields=(FieldSpec("cell_count", int),),
                    ),
                    source_image_provenance_planes=(
                        SourceImageProvenancePlanes.from_components(
                            paths=tuple(
                                f"/input/A01_s{site}_w{metadata['channel']}.tif"
                                for metadata in source_metadata
                            ),
                            component_metadata=source_metadata,
                        )
                    ),
                    subject=MeasurementSubject(MeasurementScope.ARTIFACT),
                ),
                execution_scope=RuntimeExecutionAxisScope.from_raw(
                    "A01",
                    component=AllComponents.SITE,
                    value=site,
                    fixed_component_values=(
                        (AllComponents.Z_INDEX, "1"),
                        (AllComponents.TIMEPOINT, "1"),
                    ),
                ),
            ),
            path=output_plan.path,
            backend="memory",
        )

    record_for_site("1")
    current_execution_cursor = context.runtime_value_store.observation_cursor()
    record_for_site("2")

    current_execution_records = context.runtime_value_store.observed_values_after(
        current_execution_cursor
    )
    assert observed_materialized_artifact_output_paths(
        plan,
        context,
        current_execution_records,
    ) == (
        Path(
            "/analysis/A01_site-2_z_index-1_timepoint-1_"
            "cell_counts_step7_details.csv"
        ),
    )


def test_materialize_artifact_outputs_unions_measurement_subject_records(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/A01_measurements_step7.pkl",
        artifact_type=MeasurementsArtifactType,
        materialization=csv_only(),
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/A01_w1_measurements_step7.pkl"},
    )
    runtime_output_plan = output_plan.for_group("1")
    filemanager = FileManagerStub()
    context = _context(filemanager)
    for table in (
        MeasurementTable(
            name="measurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"image_area": 100.0},),
                fields=(FieldSpec("image_area", float),),
            ),
            source_image_name="OrigBlue",
            subject=MeasurementSubject(MeasurementScope.IMAGE, "OrigBlue"),
        ),
        MeasurementTable(
            name="measurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"object_label": 1, "area": 42.0},),
                fields=(
                    FieldSpec("object_label", int),
                    FieldSpec("area", float),
                ),
            ),
            subject=MeasurementSubject(
                MeasurementScope.OBJECT, "Nuclei", "object_label"
            ),
        ),
    ):
        context.runtime_value_store.record(
            RuntimeValue.normalize(
                runtime_output_plan,
                table,
                axis_id="A01",
            ),
            path=runtime_output_plan.path,
            backend="memory",
        )
    materialized = []

    def fake_materialize(_spec, data, path, *_args, **_kwargs):
        materialized.append((tuple(data.iter_row_mappings()), path))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert materialized == [
        (
            (
                {"image_area": 100.0},
                {"object_label": 1, "area": 42.0},
            ),
            "/analysis/A01_w1_measurements_step7.roi.zip",
        )
    ]


def test_fixed_component_scopes_materialize_distinct_measurement_paths() -> None:
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"2": "/memory/measurements.pkl"},
        materialization=csv_only(),
    )
    group_plan = output_plan.for_group("2")
    context = _context(FileManagerStub())

    for z_index, subjects in (
        ("1", ("ParentObjects", "ChildObjects")),
        ("2", ("ParentObjects",)),
    ):
        execution_scope = RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=AllComponents.CHANNEL,
            value="2",
            fixed_component_values=((AllComponents.Z_INDEX, z_index),),
        )
        for subject_name in subjects:
            table = MeasurementTable(
                name="measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    (
                        {
                            "z_index": z_index,
                            "subject": subject_name,
                        },
                    ),
                    fields=(
                        FieldSpec("z_index", str),
                        FieldSpec("subject", str),
                    ),
                ),
                source_component_metadata={
                    "site": "1",
                    "channel": "2",
                    "extension": ".tif",
                },
                subject=MeasurementSubject(
                    MeasurementScope.OBJECT,
                    subject_name,
                ),
            )
            context.runtime_value_store.record(
                RuntimeValue.normalize_for_execution_scope(
                    group_plan,
                    table,
                    execution_scope=execution_scope,
                ),
                path=group_plan.path,
                backend="memory",
            )

    plan = _plan(
        output_plan,
        group_by_value="channel",
        variable_components=(VariableComponents.SITE,),
    )
    records = actual_materialization_records(
        store=context.runtime_value_store,
        plan=plan,
        output_plan=output_plan,
    )
    materializations = runtime_artifact_materializations(plan, context)

    assert len(records) == 2
    assert tuple(record.value.data.rows.row_count() for record in records) == (2, 1)
    assert tuple(
        record.key.scope.value_text_for_component(AllComponents.Z_INDEX)
        for record in records
    ) == ("1", "2")
    assert tuple(str(item.base_path) for item in materializations) == (
        "/analysis/A01_s001_w2_z001_t001_measurements_step7.roi.zip",
        "/analysis/A01_s001_w2_z002_t001_measurements_step7.roi.zip",
    )


def test_artifact_name_materialization_ignores_incomplete_source_identity() -> None:
    output_plan = ArtifactOutputPlan(
        name="SavedImage",
        path="/memory/SavedImage.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("3",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"3": "/memory/SavedImage.pkl"},
        materialization=MaterializationSpec(
            ImageFileOptions(
                filename_suffix=".tif",
                filename_identity=MaterializedFilenameIdentity.ARTIFACT_NAME,
            )
        ),
    )
    group_plan = output_plan.for_group("3")
    context = _context(FileManagerStub())
    context.runtime_value_store.record(
        RuntimeValue.normalize_for_execution_scope(
            group_plan,
            ImagePayloadMetadata(
                source_component_metadata={
                    "well": "A01",
                    "channel": "3",
                    "z_index": "1",
                    "timepoint": "1",
                },
            ).payload_with(np.ones((3, 4), dtype=np.uint8), None),
            execution_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=AllComponents.CHANNEL,
                value="3",
                fixed_component_values=((AllComponents.Z_INDEX, "1"),),
            ),
        ),
        path=group_plan.path,
        backend="memory",
    )

    materializations = runtime_artifact_materializations(
        _plan(output_plan, group_by_value="channel"),
        context,
    )

    assert tuple(str(item.base_path) for item in materializations) == (
        "/analysis/SavedImage.roi.zip",
    )


def test_fixed_scope_preserves_distinct_source_group_identity() -> None:
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"2": "/memory/measurements.pkl"},
        materialization=csv_only(),
    )
    group_plan = output_plan.for_group("2")
    context = _context(FileManagerStub())
    context.runtime_value_store.record(
        RuntimeValue.normalize_for_execution_scope(
            group_plan,
            MeasurementTable(
                name="measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"area": 42.0},),
                    fields=(FieldSpec("area", float),),
                ),
                source_component_metadata={
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "z_index": "1",
                    "timepoint": "1",
                    "extension": ".tif",
                },
                subject=MeasurementSubject(
                    MeasurementScope.OBJECT,
                    "Nuclei",
                ),
            ),
            execution_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=AllComponents.CHANNEL,
                value="2",
                fixed_component_values=((AllComponents.Z_INDEX, "1"),),
            ),
        ),
        path=group_plan.path,
        backend="memory",
    )

    materializations = runtime_artifact_materializations(
        _plan(output_plan, group_by_value="channel"),
        context,
    )

    assert tuple(str(item.base_path) for item in materializations) == (
        "/analysis/A01_s001_w1_z001_t001_measurements_step7.roi.zip",
    )


def test_fixed_component_scope_rejects_unresolved_filename_identity() -> None:
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"2": "/memory/measurements.pkl"},
        materialization=csv_only(),
    )
    group_plan = output_plan.for_group("2")
    context = _context(FileManagerStub())
    execution_scope = RuntimeExecutionAxisScope.from_raw(
        "A01",
        component=AllComponents.CHANNEL,
        value="2",
        fixed_component_values=((AllComponents.Z_INDEX, "1"),),
    )
    context.runtime_value_store.record(
        RuntimeValue.normalize_for_execution_scope(
            group_plan,
            MeasurementTable(
                name="measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"area": 42.0},),
                    fields=(FieldSpec("area", float),),
                ),
                source_path="/input/unparseable.tif",
                source_component_metadata={"extension": ".tif"},
                subject=MeasurementSubject(
                    MeasurementScope.OBJECT,
                    "Nuclei",
                ),
            ),
            execution_scope=execution_scope,
        ),
        path=group_plan.path,
        backend="memory",
    )

    with pytest.raises(
        ValueError,
        match="Cannot construct FunctionStep output filename",
    ):
        runtime_artifact_materializations(
            _plan(output_plan, group_by_value="channel"),
            context,
        )


def test_materialize_tabular_artifact_does_not_build_viewer_stream_kwargs(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        materialization=csv_only(),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(
            output_plan,
            MeasurementTable(
                name=output_plan.name,
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"object_id": 1, "area": 42},),
                    fields=(FieldSpec("object_id", int), FieldSpec("area", int)),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT,
                    output_plan.name,
                ),
            ),
            axis_id="A01",
        ),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(
            output_plan, streaming_configs={"napari_stream": streaming_config_stub()}
        ),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    spec, data, path, backends, backend_kwargs = materialized[0]
    assert isinstance(spec.outputs[0], CsvOptions)
    assert tuple(data) == ({"object_id": 1, "area": 42},)
    assert path == "/analysis/A01_measurements_step7.roi.zip"
    assert backends == ["disk"]
    assert dict(backend_kwargs["disk"]) == {}
    assert "napari_stream" not in backend_kwargs


def test_materialize_artifact_outputs_uses_actual_group_records(monkeypatch):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/A01_measurements_step7.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        paths_by_group={
            "1": "/memory/A01_w1_measurements_step7.pkl",
            "2": "/memory/A01_w2_measurements_step7.pkl",
        },
        materialization=csv_only(),
    )
    group_plan = output_plan.for_group("1")
    filemanager = FileManagerStub()
    filemanager.memory[group_plan.path] = [{"site": "1", "area": 42}]
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(
            group_plan,
            MeasurementTable(
                name=group_plan.name,
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"site": "1", "area": 42},),
                    fields=(FieldSpec("site", str), FieldSpec("area", int)),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT,
                    group_plan.name,
                ),
            ),
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

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert len(materialized) == 1
    spec, data, path = materialized[0]
    assert isinstance(spec.outputs[0], CsvOptions)
    assert tuple(data) == ({"site": "1", "area": 42},)
    assert path == "/analysis/A01_w1_measurements_step7.roi.zip"


def test_actual_materialization_records_uses_dynamic_runtime_groups():
    output_plan = ArtifactOutputPlan(
        name="segmentation_masks",
        path="/memory/A01_segmentation_masks_step7.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=(None,),
        group_component=AllComponents.CHANNEL,
        paths_by_group={None: "/memory/A01_segmentation_masks_step7.pkl"},
    )
    group_one = output_plan.for_group("1")
    group_two = output_plan.for_group("2")
    store = RuntimeValueStore()
    for group_plan, value in (
        (group_one, np.ones((3, 4, 5), dtype=np.int32)),
        (group_two, np.full((3, 4, 5), 2, dtype=np.int32)),
    ):
        labels = ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=value),
        )
        store.record(
            RuntimeValue.normalize(
                group_plan,
                labels,
                axis_id="A01",
            ),
            path=group_plan.path,
            backend="memory",
        )

    records = actual_materialization_records(
        store=store,
        plan=_plan(output_plan, group_by_value="channel"),
        output_plan=output_plan,
    )

    assert tuple(record.key.scope.value_text for record in records) == ("1", "2")
    assert tuple(record.path for record in records) == (
        "/memory/A01_w1_segmentation_masks_step7.pkl",
        "/memory/A01_w2_segmentation_masks_step7.pkl",
    )


def test_materialize_artifact_outputs_uses_group_measurement_artifact_identity(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/A01_measurements_step7.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.SITE,
        paths_by_group={
            "1": "/memory/A01_s001_measurements_step7.pkl",
            "2": "/memory/A01_s002_measurements_step7.pkl",
        },
        materialization=csv_only(),
    )
    group_one = output_plan.for_group("1")
    group_two = output_plan.for_group("2")
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(
            group_one,
            MeasurementTable(
                name="measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"site": "1", "object_id": 1, "area": 42},),
                    fields=(
                        FieldSpec("site", str),
                        FieldSpec("object_id", int),
                        FieldSpec("area", int),
                    ),
                ),
                source_path="/input/A01_s001_w5_z001_t001.TIF",
                source_component_metadata={
                    "well": "A01",
                    "site": "1",
                    "channel": "5",
                },
                subject=MeasurementSubject(MeasurementScope.ARTIFACT, "measurements"),
            ),
            axis_id="A01",
        ),
        path=group_one.path,
        backend="memory",
    )
    context.runtime_value_store.record(
        RuntimeValue.normalize(
            group_two,
            MeasurementTable(
                name="measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"site": "2", "object_id": 2, "area": 84},),
                    fields=(
                        FieldSpec("site", str),
                        FieldSpec("object_id", int),
                        FieldSpec("area", int),
                    ),
                ),
                source_path="/input/A01_s002_w5_z001_t001.TIF",
                source_component_metadata={
                    "well": "A01",
                    "site": "2",
                    "channel": "5",
                },
                subject=MeasurementSubject(MeasurementScope.ARTIFACT, "measurements"),
            ),
            axis_id="A01",
        ),
        path=group_two.path,
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

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert [path for _spec, _data, path in materialized] == [
        "/analysis/A01_s001_w5_z001_t001_measurements_step7.roi.zip",
        "/analysis/A01_s002_w5_z001_t001_measurements_step7.roi.zip",
    ]
    assert [tuple(data) for _spec, data, _path in materialized] == [
        ({"site": "1", "object_id": 1, "area": 42},),
        ({"site": "2", "object_id": 2, "area": 84},),
    ]


def test_materialize_artifact_outputs_keeps_grouped_artifact_record_path(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/A01_measurements_step7.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={
            "2": "/memory/A01_w2_measurements_step7.pkl",
        },
        materialization=csv_only(),
    )
    group_plan = output_plan.for_group("2")
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(
            group_plan,
            MeasurementTable(
                name=group_plan.name,
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"channel": "2", "area": 42},),
                    fields=(FieldSpec("channel", str), FieldSpec("area", int)),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT,
                    group_plan.name,
                ),
            ),
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

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan, group_by_value="channel"),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert [path for _spec, _data, path in materialized] == [
        "/analysis/A01_w2_measurements_step7.roi.zip",
    ]
    assert not context.runtime_value_store.values()[0].key.scope.has_fixed_components


def test_materialize_artifact_outputs_uses_null_component_group_identity_for_streaming(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={
            "2": "/memory/channel2_Nuclei.pkl",
        },
        materialization=roi_zip(),
    )
    group_plan = output_plan.for_group("2")
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32)),
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": None,
            "z_index": "1",
            "timepoint": "1",
            "extension": ".tif",
        },
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(100, 200)),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(group_plan, labels, axis_id="A01"),
        path=group_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(
            output_plan, streaming_configs={"napari_stream": streaming_config_stub()}
        ),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/analysis/A01_s001_w2_z001_t001_Nuclei_step7.roi.zip"
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": 1,
            "channel": 2,
            "z_index": 1,
            "timepoint": 1,
        },
    )


def test_materialize_artifact_outputs_streams_aggregate_artifact_with_incomplete_axis(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
        materialization=roi_zip(),
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32)),
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": None,
            "z_index": "1",
            "timepoint": "1",
            "extension": ".pkl",
        },
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(100, 200)),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, labels, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    streaming_config = streaming_config_stub()

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan, streaming_configs={"napari_stream": streaming_config}),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/analysis/A01_Nuclei_step7.roi.zip"
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.display_config.COMPONENT_ORDER == tuple(
        component
        for component in streaming_config.COMPONENT_ORDER
        if component != "channel"
    )
    assert stream_request.display_config.component_modes() == (
        streaming_config.component_modes()
    )
    assert stream_request.display_config.display_payload_extra() == {
        "colormap": "gray",
        "variable_size_handling": "pad_to_max",
    }
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": 1,
            "z_index": 1,
            "timepoint": 1,
        },
    )
    assert stream_request.message_extra["component_value_domain"] == {
        "well": ["A01"],
        "site": [1],
        "z_index": [1],
        "timepoint": [1],
    }


def test_materialize_artifact_outputs_uses_declared_metadata_json_spec(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="metadata",
        path="/memory/metadata.pkl",
        artifact_type=MetadataArtifactType,
        materialization=json_only(),
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = {"plate": "A"}
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, {"plate": "A"}, axis_id="A01"),
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

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    spec, data, _path = materialized[0]
    assert isinstance(spec.outputs[0], JsonOptions)
    assert data == {"plate": "A"}


def test_materialize_artifact_outputs_skips_special_without_explicit_spec(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="positions",
        path="/memory/positions.pkl",
        artifact_type=SpecialArtifactType,
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

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert materialized == []


def test_materialize_artifact_outputs_skips_explicitly_disabled_artifact_without_record(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="ER",
        path="/memory/ER.pkl",
        artifact_type=ImageArtifactType,
        materialization=None,
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    materialized = []

    def fake_materialize(*args, **kwargs):
        materialized.append((args, kwargs))

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert materialized == []


def test_materialize_artifact_outputs_uses_declared_object_labels_roi_spec(monkeypatch):
    output_plan = ArtifactOutputPlan(
        name="labels",
        path="/memory/labels.pkl",
        artifact_type=ObjectLabelsArtifactType,
        materialization=roi_zip(),
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.zeros((2, 2), dtype=np.int32),
        ),
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = labels
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, labels, axis_id="A01"),
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

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    spec, data, path = materialized[0]
    assert isinstance(spec.outputs[0], ROIOptions)
    assert isinstance(data, ObjectLabelSet)
    assert data.name == output_plan.name
    np.testing.assert_array_equal(data.labels, labels.labels)
    assert path == "/analysis/A01_labels_step7.roi.zip"


def test_materialize_artifact_outputs_can_target_streaming_without_persistent_backend(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="labels",
        path="/memory/labels.pkl",
        artifact_type=ObjectLabelsArtifactType,
        materialization=roi_zip(),
    )
    streaming_config = streaming_config_stub()
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32)),
        source_path="/input/A01_s001_w1.TIF",
        source_component_metadata={"well": "A01", "channel": 1},
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(100, 200)),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, labels, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan, streaming_configs={"napari_stream": streaming_config}),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    spec, data, path, backends, backend_kwargs = materialized[0]
    assert isinstance(spec.outputs[0], ROIOptions)
    assert isinstance(data, ObjectLabelSet)
    assert data.name == output_plan.name
    np.testing.assert_array_equal(data.labels, labels.labels)
    assert path == "/analysis/A01_s001_w1_z001_t001_labels_step7.roi.zip"
    assert backends == ["napari_stream"]
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.port == 5555
    assert stream_request.display_config is streaming_config
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": 1,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
        },
    )
    assert stream_request.message_extra["component_value_domain"] == {
        "well": ["A01"],
        "site": [1],
        "channel": [1, 2, 3],
        "z_index": [1],
        "timepoint": [1],
    }
    assert stream_request.producer.identities[0].to_payload() == {
        "origin": "pipeline",
            "output_kind": "artifact",
            "output_key": "labels",
            "projection_key": "labels",
            "step_name": "measure",
        "pipeline_position": 7,
        "step_scope_id": "measure-scope-7",
        "invocation_key": None,
        "artifact_kind": "object_labels",
    }


def test_main_flow_artifact_persists_without_duplicate_viewer_stream(monkeypatch):
    output_plan = ArtifactOutputPlan(
        name="CorrectedImage",
        path="/memory/CorrectedImage.pkl",
        artifact_type=ImageArtifactType,
        materialization=tiff_stack(),
    )
    payload = ImagePayloadMetadata(
        source_component_metadata={
            "well": "A01",
            "site": 1,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
        },
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(5, 7)),
    ).payload_with(np.ones((5, 7), dtype=np.float32))
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, payload, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        _spec,
        _data,
        _path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((backends, backend_kwargs))

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(
            output_plan,
            streaming_configs={"napari_stream": streaming_config_stub()},
            compiled_function_pattern=_main_flow_output_compiled_pattern(output_plan),
        ),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert len(materialized) == 1
    backends, backend_kwargs = materialized[0]
    assert backends == ["disk"]
    assert tuple(backend_kwargs) == ("disk",)


def test_materialize_artifact_outputs_uses_artifact_source_metadata_for_streaming(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="labels",
        path="/memory/labels.pkl",
        artifact_type=ObjectLabelsArtifactType,
        materialization=roi_zip(),
    )
    streaming_config = streaming_config_stub()
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32)),
        source_path="/input/A01_s002_w3_z001_t001.TIF",
        source_component_metadata={"channel": 3},
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(100, 200)),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.microscope_handler = MicroscopeHandlerStub(
        parser=FilenameParserStub(),
        metadata_handler=MetadataHandlerStub(
            {"channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"}}
        ),
    )
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, labels, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(
            output_plan,
            streaming_configs={"napari_stream": streaming_config},
        ),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/analysis/A01_s002_w3_z001_t001_labels_step7.roi.zip"
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": 2,
            "channel": 3,
            "z_index": 1,
            "timepoint": 1,
        },
    )
    assert stream_request.message_extra == {
        "component_value_domain": {
            "well": ["A01"],
            "site": [2],
            "channel": [1, 2, 3],
            "z_index": [1],
            "timepoint": [1],
        },
        "component_names_metadata": {
            "channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"},
            "well": {"A01": None},
            "site": {"2": None},
            "z_index": {"1": None},
            "timepoint": {"1": None},
        },
    }


def test_materialize_artifact_outputs_streams_payload_component_metadata(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
        materialization=roi_zip(),
    )
    streaming_config = streaming_config_stub()
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32)),
        source_path="/input/01_POS002_D.TIF",
        source_component_metadata={"well": "01", "site": "POS002", "channel": "D"},
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(100, 200)),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, labels, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan, streaming_configs={"napari_stream": streaming_config}),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/analysis/01_POS002_D_Nuclei_step7.roi.zip"
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.source.metadata.metadata_by_index == (
        {"well": "01", "site": "POS002", "channel": "D", "z_index": 1, "timepoint": 1},
    )


def test_materialize_artifact_outputs_uses_runtime_plane_group_identity(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="AdjacentImage",
        path="/memory/AdjacentImage.pkl",
        artifact_type=ImageArtifactType,
        materialization=csv_only(),
        group_keys=("11",),
        group_component=AllComponents.TIMEPOINT,
        paths_by_group={
            "11": "/memory/A01_t11_AdjacentImage_step7.pkl",
        },
    )
    group_plan = output_plan.for_group("11")
    payload = ImageMetadataPayload(
        data=np.ones((8, 12, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
                "z_index": "1",
                "extension": ".tif",
            },
            source_image_names=("OrigColor",),
        ),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    runtime_value = RuntimeValue.normalize(group_plan, payload, axis_id="A01")
    context.runtime_value_store.record(
        runtime_value,
        path=group_plan.path,
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

    materialize_artifact_outputs(
        filemanager,
        _plan(
            output_plan,
            group_by_value="timepoint",
            variable_components=(VariableComponents.TIMEPOINT,),
        ),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert materialized == [
        (
            runtime_value.data,
            "/images/A01_s001_w1_z001_t011.tif",
        )
    ]
    assert materialized[0][0] is runtime_value.data
    assert image_payload_metadata(materialized[0][0]).source_image_names == (
        "AdjacentImage",
    )


def test_materialize_artifact_outputs_merges_parser_axes_into_source_metadata(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
        materialization=roi_zip(),
    )
    streaming_config = streaming_config_stub()
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32)),
        source_path="/input/A01_s002_w3_z001_t001.TIF",
        source_component_metadata={"instrument": "test"},
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(100, 200)),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.microscope_handler = MicroscopeHandlerStub(
        parser=FilenameParserStub(),
        metadata_handler=MetadataHandlerStub(
            {"channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"}}
        ),
    )
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, labels, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan, streaming_configs={"napari_stream": streaming_config}),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/analysis/A01_s002_w3_z001_t001_Nuclei_step7.roi.zip"
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": 2,
            "channel": 3,
            "z_index": 1,
            "timepoint": 1,
        },
    )


@pytest.mark.parametrize(
    "domain_scope",
    (ObjectLabelDomainScope.PLANE, ObjectLabelDomainScope.PAYLOAD),
)
def test_materialize_artifact_outputs_uses_variable_components_for_streaming_identity(
    monkeypatch,
    domain_scope,
):
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
        materialization=roi_zip(),
    )
    streaming_config = streaming_config_stub()
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2, 2), dtype=np.int32)),
        plane_axis=(
            RuntimePlaneAxis.RUNTIME_SLICE
            if domain_scope is ObjectLabelDomainScope.PLANE
            else None
        ),
        domain=(
            ObjectLabelDomain(
                declared_object_id_domains=((), ()),
                scope=ObjectLabelDomainScope.PLANE,
            )
            if domain_scope is ObjectLabelDomainScope.PLANE
            else ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD)
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.TIF",
                "/input/A01_s001_w1_z002_t001.TIF",
            ),
            component_metadata=(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "z_index": "1",
                    "timepoint": "1",
                },
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "z_index": "2",
                    "timepoint": "1",
                },
            ),
        ),
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(100, 200)),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, labels, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(
            output_plan,
            streaming_configs={"napari_stream": streaming_config},
            variable_components=(VariableComponents.Z_INDEX,),
        ),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/analysis/A01_Nuclei_step7.roi.zip"
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert "z_index" in stream_request.display_config.COMPONENT_ORDER
    assert stream_request.message_extra["component_value_domain"]["z_index"] == [
        1,
        2,
    ]
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": 1,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
        },
        {
            "well": "A01",
            "site": 1,
            "channel": 1,
            "z_index": 2,
            "timepoint": 1,
        },
    )


def test_materialize_artifact_outputs_streams_singleton_roi_plane_from_output_plan():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
        variable_components=(VariableComponents.SITE,),
        materialization=roi_zip(),
    )
    labels_array = np.zeros((1, 8, 8), dtype=np.int32)
    labels_array[0, 2:6, 3:7] = 1
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels_array),
        source_path="/input/A01_s001_w1_z001_t001.TIF",
        source_component_metadata={
            "well": "A01",
            "site": 1,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
        },
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(8, 8)),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, labels, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(
            output_plan,
            streaming_configs={"napari_stream": streaming_config_stub()},
            variable_components=(VariableComponents.Z_INDEX,),
        ),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    roi_saves = [item for item in filemanager.saved if item[1].endswith(".roi.zip")]
    assert len(roi_saves) == 1
    roi_content, roi_path, _backend, stream_kwargs = roi_saves[0]
    stream_request = stream_kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.source.item_fields["plane_component_values"] == {
        "site": ["1"]
    }

    napari_backend = NapariStreamingBackend()
    streamed_item = StreamingBatchMessageBuilder.build(
        napari_backend,
        StreamingBatchMessageRequest(
            data_list=[roi_content],
            file_paths=[roi_path],
            stream_request=stream_request,
            component_names_request=napari_backend.component_names_request(
                stream_request
            ),
            display_payload_extra=napari_backend.display_payload_extra(stream_request),
        ),
    ).batch_images[0]
    assert streamed_item["plane_component_values"] == {"site": ["1"]}


def test_materialize_artifact_outputs_streams_source_binding_roi_plane_metadata(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="segmentation_masks",
        path="/memory/segmentation_masks.pkl",
        artifact_type=ObjectLabelsArtifactType,
        materialization=roi_zip(),
    )
    streaming_config = streaming_config_stub()
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2, 2), dtype=np.int32)),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.TIF",
                "/input/A01_s001_w2_z001_t001.TIF",
            ),
            component_metadata=(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "z_index": "1",
                    "timepoint": "1",
                },
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "2",
                    "z_index": "1",
                    "timepoint": "1",
                },
            ),
        ),
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(100, 200)),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, labels, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(
            output_plan,
            streaming_configs={"napari_stream": streaming_config},
            variable_components=(VariableComponents.CHANNEL,),
        ),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/analysis/A01_segmentation_masks_step7.roi.zip"
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": 1,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
        },
        {
            "well": "A01",
            "site": 1,
            "channel": 2,
            "z_index": 1,
            "timepoint": 1,
        },
    )


def test_materialize_rgb_artifact_streams_filename_channel_identity(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="RGBImage",
        path="/memory/RGBImage.pkl",
        artifact_type=ImageArtifactType,
        materialization=tiff_stack(),
    )
    streaming_config = streaming_config_stub()
    rgb_payload = ImageMetadataPayload(
        data=np.ones((5, 7, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w3_z001_t001.TIF",
                    "/input/A01_s001_w2_z001_t001.TIF",
                    "/input/A01_s001_w1_z001_t001.TIF",
                ),
                component_metadata=(
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "3",
                        "z_index": "1",
                        "timepoint": "1",
                    },
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "2",
                        "z_index": "1",
                        "timepoint": "1",
                    },
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "1",
                        "z_index": "1",
                        "timepoint": "1",
                    },
                ),
            ),
        ),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, rgb_payload, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(
            output_plan,
            streaming_configs={"napari_stream": streaming_config},
            variable_components=(VariableComponents.CHANNEL,),
        ),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/images/A01_s001_w3_z001_t001.TIF"
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": 1,
            "channel": 3,
            "z_index": 1,
            "timepoint": 1,
        },
    )


def test_materialize_image_uses_declared_filename_source_identity(monkeypatch):
    filename_source = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    selected_image = ArtifactSpec.input("RGBImage", ImageArtifactType)
    output_plan = ArtifactOutputPlan(
        name="SavedRGB",
        path="/memory/SavedRGB.pkl",
        artifact_type=ImageArtifactType,
        materialization=tiff_stack(),
        relations=(
            MaterializationSourceIdentityRelation(filename_source.ref()),
            GroupLineageSourceRelation(selected_image.ref()),
        ),
    )
    payload = ImageMetadataPayload(
        data=np.ones((5, 7, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w3_z001_t001.TIF",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "3",
                "z_index": "1",
                "timepoint": "1",
            },
            source_image_names=("RGBImage", "OrigRed"),
        ),
    )
    filename_source_metadata = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1_z001_t001.TIF",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
            "z_index": "1",
            "timepoint": "1",
        },
        source_image_names=("OrigBlue",),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(
            output_plan,
            payload,
            axis_id="A01",
            materialization_source_metadata=filename_source_metadata,
        ),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(_spec, _data, path, *_args, **_kwargs):
        materialized.append(path)
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan, variable_components=(VariableComponents.CHANNEL,)),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert materialized == ["/images/A01_s001_w1_z001_t001.TIF"]


def test_materialization_identity_replaces_provenance_not_payload_layout() -> None:
    filename_source = ArtifactSpec.input("OrigDNA", ImageArtifactType)
    selected_image = ArtifactSpec.input("NucleiImage", ImageArtifactType)
    options = ImageFileOptions(
        filename_suffix="_NucleiLabels.tif",
        filename_identity=MaterializedFilenameIdentity.SOURCE_IDENTITY,
    )
    output_plan = ArtifactOutputPlan(
        name="SavedNuclei",
        path="/memory/SavedNuclei.pkl",
        artifact_type=ImageArtifactType,
        materialization=MaterializationSpec(options),
        relations=(
            MaterializationSourceIdentityRelation(filename_source.ref()),
            GroupLineageSourceRelation(selected_image.ref()),
        ),
    )
    filename_source_metadata = ImagePayloadMetadata(
        source_path="/input/A01_s001_w2_z001_t001.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "2",
            "z_index": "1",
            "timepoint": "1",
        },
        source_image_names=("OrigDNA",),
        source_channel_axis=0,
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )
    value = RuntimeValue.normalize(
        output_plan,
        ImagePayloadMetadata(
            source_path="/derived/nuclei-rgb.tif",
            source_image_names=("NucleiImage",),
            source_channel_axis=3,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(
            np.ones((2, 5, 7, 3), dtype=np.uint16),
            None,
        ),
        axis_id="A01",
        materialization_source_metadata=filename_source_metadata,
    )

    metadata = output_plan.materialization_metadata(value)

    assert metadata.source_path == "/input/A01_s001_w2_z001_t001.tif"
    assert metadata.source_image_names == ("OrigDNA",)
    assert metadata.source_channel_axis == 3
    assert metadata.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_compiled_z_axis_reaches_source_named_image_materialization() -> None:
    filename_source = ArtifactSpec.input("OrigDNA", ImageArtifactType)
    selected_image = ArtifactSpec.input("NucleiImage", ImageArtifactType)
    output_plan = ArtifactOutputPlan(
        name="SavedNuclei",
        path="/memory/SavedNuclei.pkl",
        artifact_type=ImageArtifactType,
        materialization=MaterializationSpec(
            ImageFileOptions(
                filename_suffix="_NucleiLabels.tif",
                filename_identity=MaterializedFilenameIdentity.SOURCE_IDENTITY,
            )
        ),
        variable_components=(VariableComponents.Z_INDEX,),
        relations=(
            MaterializationSourceIdentityRelation(filename_source.ref()),
            GroupLineageSourceRelation(selected_image.ref()),
        ),
    )
    selected_payload = ImagePayloadMetadata(
        source_image_names=(selected_image.name,),
    ).payload_with(np.ones((2, 5, 7), dtype=np.uint16), None)
    saved_payload = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        selected_payload,
        ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(
            np.ones((2, 5, 7), dtype=np.uint16),
            None,
        ),
        output_plan,
        RuntimePlaneProjection.stack(2),
    )
    filename_source_metadata = ImagePayloadMetadata(
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "2",
            "timepoint": "1",
        },
        source_image_names=(filename_source.name,),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w2_z001_t001.tif",
                "/input/A01_s001_w2_z002_t001.tif",
            ),
            component_metadata=(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "2",
                    "z_index": "1",
                    "timepoint": "1",
                },
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "2",
                    "z_index": "2",
                    "timepoint": "1",
                },
            ),
        ),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(
            output_plan,
            saved_payload,
            axis_id="A01",
            materialization_source_metadata=filename_source_metadata,
        ),
        path=output_plan.path,
        backend="memory",
    )
    plan = _plan(
        output_plan,
        variable_components=(VariableComponents.Z_INDEX,),
    )
    plan.runtime_artifact_materialization = RuntimeArtifactMaterializationPlan(
        persistent_enabled=True,
        persistent_backend="disk",
    )

    assert image_payload_metadata(saved_payload).plane_axis is (
        RuntimePlaneAxis.RUNTIME_SLICE
    )
    assert materialized_artifact_output_paths(plan, context) == (
        Path("/images/A01_s001_w2_z001_t001_NucleiLabels.tif"),
        Path("/images/A01_s001_w2_z002_t001_NucleiLabels.tif"),
    )


def test_materialize_rgb_artifact_keeps_scalar_filename_identity_for_mixed_provenance(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="OrigOverlay",
        path="/memory/OrigOverlay.pkl",
        artifact_type=ImageArtifactType,
        materialization=tiff_stack(),
    )
    streaming_config = streaming_config_stub()
    rgb_payload = ImageMetadataPayload(
        data=np.ones((5, 7, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w2_z001_t001.TIF",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "2",
                "z_index": "1",
                "timepoint": "1",
            },
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w3_z001_t001.TIF",
                    "/input/A01_s001_w2_z001_t001.TIF",
                ),
                component_metadata=(
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "3",
                        "z_index": "1",
                        "timepoint": "1",
                    },
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "2",
                        "z_index": "1",
                        "timepoint": "1",
                    },
                ),
            ),
        ),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.normalize(output_plan, rgb_payload, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(
            output_plan,
            streaming_configs={"napari_stream": streaming_config},
            variable_components=(VariableComponents.SITE,),
        ),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/images/A01_s001_w2_z001_t001.TIF"
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": 1,
            "channel": 2,
            "z_index": 1,
            "timepoint": 1,
        },
    )
