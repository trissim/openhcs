import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from polystore.disk import DiskStorageBackend
from polystore.filemanager import FileManager
from polystore.streaming.viewer_transport import (
    ViewerDisplayConfigABC,
    ViewerStreamKwarg,
    ViewerStreamSourceIdentity,
)
from zmqruntime.viewer_protocol import ViewerTransportEndpoint

from openhcs.constants.constants import AllComponents, Backend, VariableComponents
from openhcs.core.aligned_image_payload import AlignedImageSliceContext
from openhcs.core.artifacts import (
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.axis_filter import StepAxisFilterResolution, StepAxisFilterSet
from openhcs.core.compiled_step_plan import CompiledStepPlan, MaterializedOutputPlan
from openhcs.core.config import WellFilterMode
from openhcs.core.function_patterns import compile_function_pattern
from openhcs.core.runtime_image_loading import ImagePayloadSourceMetadataContext
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.source_image_provenance import (
    SourceImageIdentity,
    SourceImageProvenancePlanes,
)
from openhcs.core.source_metadata import (
    SOURCE_PLANE_COUNT_FIELD,
    SOURCE_PLANE_INDEX_FIELD,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.step_dependencies import StepInputDependency
from openhcs.core.steps.function_output_identity import FunctionOutputIdentity
from openhcs.core.steps.function_output_manifest import (
    ProducedOutputSemantics,
    step_output_manifest,
)
from openhcs.core.steps.function_outputs import (
    MaterializedImageOutputWriter,
    MemoryOutputWriter,
    OpenHCSMetadataWriter,
    ProducedMemoryPathsAuthority,
    StreamOutputsAuthority,
)
from openhcs.core.streaming_config_factory import (
    StreamingViewerPresentation,
    StreamingViewerRuntimeConfig,
    StreamingViewerSurface,
)


@pytest.mark.parametrize("backend", [Backend.ZARR.value, "custom-array-store"])
def test_memory_output_writer_projects_runtime_image_payload_for_array_storage(
    backend,
):
    image = np.zeros((2, 3), dtype=np.uint16)
    payload = ImageMetadataPayload(
        data=image,
        metadata=ImagePayloadMetadata(source_dtype="uint16"),
    )

    prepared = MemoryOutputWriter.payloads(
        [payload],
        ["/virtual/A01_s1_w1.tif"],
        SimpleNamespace(write_backend=backend),
    )

    assert len(prepared) == 1
    assert prepared[0] is image


def test_memory_output_writer_rejects_payload_path_cardinality_mismatch():
    with pytest.raises(ValueError, match="1 payloads for 0 paths"):
        MemoryOutputWriter.payloads(
            [np.zeros((2, 3), dtype=np.uint16)],
            [],
            SimpleNamespace(write_backend=Backend.MEMORY.value),
        )


def test_memory_output_writer_delegates_disk_payload_preparation(monkeypatch):
    image = np.zeros((2, 3), dtype=np.uint16)
    prepared = object()

    def prepare_disk(payloads, paths):
        assert payloads == [image]
        assert paths == ["/virtual/A01_s1_w1.tif"]
        return [prepared]

    monkeypatch.setattr(
        "openhcs.core.steps.function_io.prepare_disk_image_payloads",
        prepare_disk,
    )

    assert MemoryOutputWriter.payloads(
        [image],
        ["/virtual/A01_s1_w1.tif"],
        SimpleNamespace(write_backend=Backend.DISK.value),
    ) == [prepared]


def complete_component_metadata(metadata):
    completed = {"z_index": "1", "timepoint": "1"}
    completed.update(metadata)
    return completed


def expected_viewer_metadata(metadata):
    projected = complete_component_metadata(metadata)
    return {
        component: (
            int(value)
            if component in {"site", "channel", "z_index", "timepoint"}
            else value
        )
        for component, value in projected.items()
        if component in StreamingConfigStub.COMPONENT_ORDER
    }


class FileManagerStub:
    def __init__(self, memory_payloads):
        self.memory_payloads = memory_payloads
        self.saved_batches = []

    def load_batch(self, paths, backend):
        assert backend == Backend.MEMORY.value
        return [self.memory_payloads[path] for path in paths]

    def save_batch(self, data, paths, backend, **kwargs):
        self.saved_batches.append((data, paths, backend, kwargs))

    def ensure_directory(self, path, backend):
        return None


class StreamingConfigStub(ViewerDisplayConfigABC):
    backend = SimpleNamespace(value="napari_stream")
    COMPONENT_ORDER = ("well", "site", "channel", "z_index", "timepoint")
    host = "127.0.0.1"
    port = 5555
    transport_mode = "tcp"

    def component_modes(self):
        return {component: "stack" for component in self.COMPONENT_ORDER}

    def display_payload_extra(self):
        return {}

    def streaming_viewer_surface(self, context):
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
            source=ViewerStreamSourceIdentity(
                microscope_handler=context.microscope_handler,
                plate_path=context.plate_path,
            ),
        )


class ParserStub:
    def parse_filename(self, name):
        stem = Path(name).stem
        well, site, channel = stem.split("_")
        return complete_component_metadata(
            {
                "well": well,
                "site": site.removeprefix("s"),
                "channel": channel.removeprefix("w"),
                "extension": "".join(Path(name).suffixes),
            }
        )

    def construct_filename(self, **metadata):
        extension = metadata.get("extension") or ".tif"
        return (
            f"{metadata['well']}_s{metadata['site']}_w{metadata['channel']}"
            f"{extension}"
        )

    def extract_component_coordinates(self, axis_id):
        assert axis_id == "A01"
        return 1, 1


class MetadataHandlerStub:
    def __init__(self, values=None):
        self.values = values or {}

    def find_metadata_file(self, root):
        return Path(root) / "openhcs_metadata.json"

    def get_component_values(self, _root, component):
        return self.values.get(component, {})

    def get_grid_dimensions(self, _root):
        return (1, 1)

    def get_pixel_size(self, _root):
        return 1.0


class ContextStub:
    pass


def context_stub(filemanager, parser=None):
    context = ContextStub()
    context.filemanager = filemanager
    context.microscope_handler = SimpleNamespace(
        parser=parser or ParserStub(),
        microscope_type="test",
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


def function_step_plan(
    step_name: str,
    variable_components: tuple[VariableComponents, ...] = (),
    pipeline_position: int = 3,
) -> CompiledStepPlan:
    return CompiledStepPlan(
        step_index=pipeline_position,
        step_name=step_name,
        step_type="FunctionStep",
        axis_id="A01",
        streaming_configs={"napari_stream": StreamingConfigStub()},
        artifact_outputs={},
        output_dir=Path("/tmp/output"),
        pipeline_position=pipeline_position,
        step_scope_id=f"step-scope-{pipeline_position}",
        main_input_dependency=StepInputDependency.no_main_flow(),
        variable_components=variable_components,
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
    )


def record_output_path(
    context,
    plan,
    path,
    output_context=None,
    image_metadata=None,
    identity=None,
):
    metadata = context.microscope_handler.parser.parse_filename(Path(path).name)
    assert metadata is not None
    output_identity = identity or FunctionOutputIdentity(
        component_values={
            str(key): value
            for key, value in metadata.items()
            if str(key) != "extension"
        },
        extension=metadata.get("extension"),
        source="test output path",
    )
    step_output_manifest(context).record_outputs(
        plan,
        [
            ProducedOutputSemantics.from_output(
                plan,
                path,
                output_identity,
                output_context=output_context,
                image_metadata=image_metadata,
            )
        ],
    )


def test_function_output_identity_preserves_non_axis_source_metadata() -> None:
    identity = FunctionOutputIdentity(
        component_values={"channel": "2", "site": "1"},
        extension=".tif",
        source="test",
    )

    metadata = identity.component_metadata(
        {
            "Run": "Sequence1",
            "Specimen": "DrosophilaEmbryo",
            "ChannelNumber": "1",
            "OpenHCSOriginalSourceMetadata": {"FrameNumber": "0007"},
        }
    )

    assert metadata == {
        "Run": "Sequence1",
        "Specimen": "DrosophilaEmbryo",
        "channel": "2",
        "site": "1",
        "extension": ".tif",
        "OpenHCSOriginalSourceMetadata": {"FrameNumber": "0007"},
    }


def test_step_output_manifest_prefers_main_dependency_over_auxiliary_artifact_inputs():
    context = context_stub(FileManagerStub({}))
    manifest = step_output_manifest(context)

    main_producer = function_step_plan("ErodeImage")
    main_producer.step_scope_id = "main-producer"
    main_producer.pipeline_position = 24
    seed_producer = function_step_plan("ConvertObjectsToImage")
    seed_producer.step_scope_id = "seed-producer"
    seed_producer.pipeline_position = 14

    record_output_path(
        context,
        main_producer,
        "/tmp/output/A01_s1_w1.tif",
        AlignedImageSliceContext.main_flow(
            "MembFinal",
            artifact_kind=ImageArtifactType.value,
        ),
    )
    record_output_path(
        context,
        seed_producer,
        "/tmp/output/A01_s1_w2.tif",
        AlignedImageSliceContext.main_flow(
            "cellSeeds",
            artifact_kind=ImageArtifactType.value,
        ),
    )

    consumer = function_step_plan("Watershed")
    consumer.main_input_dependency = StepInputDependency.step_output(
        source_step_index=24,
        source_step_scope_id="main-producer",
    )

    assert manifest.producer_paths_for(consumer) == ("A01_s1_w1.tif",)
    assert manifest.filter_to_producer_paths(
        consumer,
        ("A01_s1_w1.tif", "A01_s1_w2.tif"),
        context.microscope_handler.parser,
    ) == ["A01_s1_w1.tif"]


def image_payload_with_source_metadata(pixels, metadata, mask=None):
    completed_metadata = complete_component_metadata(metadata)
    return ImagePayloadMetadata(
        source_component_metadata=completed_metadata,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=(completed_metadata,)
        ),
    ).payload_with(pixels, mask)


def test_stream_outputs_unwraps_runtime_image_payloads_before_viewer_backend():
    path = "/tmp/output/A01_s1_w1.tif"
    pixels = np.ones((2, 3), dtype=np.uint16)
    payload = image_payload_with_source_metadata(
        pixels,
        {
            "well": "A01",
            "site": "1",
            "channel": "1",
            "extension": ".tif",
        },
        mask=np.ones_like(pixels, dtype=bool),
    )
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan("IdentifyPrimaryObjects")
    record_output_path(context, plan, path)

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == [path]
    assert backend == "napari_stream"
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.port == 5555
    assert stream_request.source.metadata.metadata_by_index == (
        expected_viewer_metadata(
            {
                "well": "A01",
                "site": "1",
                "channel": "1",
            }
        ),
    )
    assert stream_request.message_extra == {
        "component_value_domain": {
            "well": ["A01"],
            "site": [1],
            "channel": [1, 2, 3],
            "z_index": [1],
            "timepoint": [1],
        },
        "component_names_metadata": {
            "channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"},
            "well": {"A01": None},
            "site": {"1": None},
            "z_index": {"1": None},
            "timepoint": {"1": None},
        },
    }
    assert stream_request.producer.identities[0].to_payload() == {
        "origin": "pipeline",
        "output_kind": "main",
        "output_key": "main",
        "projection_key": "main",
        "step_name": "IdentifyPrimaryObjects",
        "pipeline_position": 3,
        "step_scope_id": "step-scope-3",
        "invocation_key": None,
        "artifact_kind": None,
    }
    assert streamed_data == [pixels]


def test_source_preserving_output_uses_manifest_path_for_materialization_and_streaming():
    source_path = "/tmp/source/A01_s1_w1.tif"
    materialized_path = "/tmp/materialized/A01_s1_w1.tif"
    pixels = np.ones((2, 3), dtype=np.uint16)
    payload = image_payload_with_source_metadata(
        pixels,
        {
            "well": "A01",
            "site": "1",
            "channel": "1",
            "extension": ".tif",
        },
    )
    filemanager = FileManagerStub({source_path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan("MeasureImageIntensity")
    plan.materialized_output = MaterializedOutputPlan(
        output_dir=Path("/tmp/materialized"),
        backend=Backend.ZARR.value,
        plate_root="/tmp/materialized",
        sub_dir=".",
        analysis_results_dir=None,
    )
    step_output_manifest(context).record_outputs(
        plan,
        [
            ProducedOutputSemantics.from_existing_main_flow_path(
                plan,
                source_path,
                context.microscope_handler.parser,
            )
        ],
    )

    MaterializedImageOutputWriter.write_if_needed(context, plan)
    StreamOutputsAuthority.stream_outputs(context, plan)

    materialized_batch, streamed_batch = filemanager.saved_batches
    assert materialized_batch[0][0] is pixels
    assert materialized_batch[1] == [materialized_path]
    assert materialized_batch[2] == Backend.ZARR.value
    assert streamed_batch[1] == [materialized_path]
    assert streamed_batch[2] == "napari_stream"


def test_stream_outputs_respects_compiled_filter_for_streaming_config():
    path = "/tmp/output/A01_s1_w1.tif"
    pixels = np.ones((2, 3), dtype=np.uint16)
    filemanager = FileManagerStub(
        {
            path: image_payload_with_source_metadata(
                pixels,
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "extension": ".tif",
                },
            )
        }
    )
    context = context_stub(filemanager)
    plan = function_step_plan("GaussianBlur")
    config = next(iter(plan.streaming_configs.values()))
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
    record_output_path(context, plan, path)

    StreamOutputsAuthority.stream_outputs(context, plan)

    assert filemanager.saved_batches == []

    context.axis_id = "B03"
    StreamOutputsAuthority.stream_outputs(context, plan)

    assert len(filemanager.saved_batches) == 1

    context.axis_id = "b03"
    StreamOutputsAuthority.stream_outputs(context, plan)

    assert len(filemanager.saved_batches) == 2


def test_stream_outputs_scope_viewer_layout_to_collapsed_source_components():
    path = "/tmp/output/A01_s1_w1.tif"
    stack_metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
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
                    "site": "2",
                    "channel": "1",
                    "z_index": "1",
                    "timepoint": "1",
                },
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    collapsed_metadata = stack_metadata.collapse_leading_plane_axis()
    payload = collapsed_metadata.payload_with(
        np.ones((2, 3), dtype=np.float64),
        None,
    )
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan("CorrectIlluminationCalculate")
    record_output_path(
        context,
        plan,
        path,
        image_metadata=collapsed_metadata,
        identity=FunctionOutputIdentity(
            component_values={
                "well": "A01",
                "channel": 1,
                "z_index": 1,
                "timepoint": 1,
            },
            filename_component_values={
                "well": "A01",
                "site": 1,
                "channel": 1,
                "z_index": 1,
                "timepoint": 1,
            },
            extension=".tif",
            source="collapsed source identity",
        ),
    )

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(_streamed_data, _streamed_paths, _backend, kwargs)] = filemanager.saved_batches
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.display_config.COMPONENT_ORDER == (
        "well",
        "channel",
        "z_index",
        "timepoint",
    )
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
        },
    )
    assert "site" not in stream_request.message_extra["component_value_domain"]


def test_stream_outputs_restore_manifest_image_metadata_after_memory_serialization():
    path = "/tmp/output/A01_s1_w1.tif"
    pixels = np.ones((2, 3), dtype=np.uint16)
    source_metadata = complete_component_metadata(
        {
            "well": "A01",
            "site": "1",
            "channel": "1",
            "extension": ".tif",
        }
    )
    image_metadata = ImagePayloadMetadata(
        source_component_metadata=source_metadata,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=(source_metadata,)
        ),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=(2, 3),
        ),
    )
    filemanager = FileManagerStub({path: pixels})
    context = context_stub(filemanager)
    plan = function_step_plan("IdentifyPrimaryObjects")
    record_output_path(
        context,
        plan,
        path,
        image_metadata=image_metadata,
    )

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(_streamed_data, _streamed_paths, _backend, kwargs)] = filemanager.saved_batches
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.source.item_fields == {
        "spatial_origin_yx": [0, 0],
        "source_spatial_shape_yx": [2, 3],
    }
    assert stream_request.source.metadata.metadata_by_index == (
        expected_viewer_metadata(source_metadata),
    )


def test_stream_outputs_keeps_scalar_records_from_variable_component_step():
    path = "/tmp/output/A01_s1_w1.tif"
    pixels = np.ones((2, 3), dtype=np.uint16)
    payload = image_payload_with_source_metadata(
        pixels,
        {
            "well": "A01",
            "site": "1",
            "channel": "1",
            "extension": ".tif",
        },
    )
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan(
        "Normalize",
        variable_components=(VariableComponents.SITE,),
    )
    record_output_path(context, plan, path)

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, _kwargs)] = filemanager.saved_batches
    assert streamed_data == [pixels]
    assert streamed_paths == [path]
    assert backend == "napari_stream"


def test_source_metadata_request_declares_volumetric_z_planes():
    pixels = np.ones((3, 4, 5), dtype=np.uint16)
    metadata = ImagePayloadSourceMetadataContext(
        SourceImageIdentity(
            "/tmp/plate/images/A01_s1_w3_z5.tif",
            {
                "well": "A01",
                "site": "1",
                "channel": "3",
                "z_index": "5",
                SOURCE_PLANE_INDEX_FIELD: "0",
                SOURCE_PLANE_COUNT_FIELD: "3",
            },
        )
    ).metadata(pixels)

    assert metadata.source_image_provenance_planes.paths == (
        "/tmp/plate/images/A01_s1_w3_z5.tif",
        "/tmp/plate/images/A01_s1_w3_z5.tif",
        "/tmp/plate/images/A01_s1_w3_z5.tif",
    )
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == (
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "5",
            SOURCE_PLANE_INDEX_FIELD: "0",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "6",
            SOURCE_PLANE_INDEX_FIELD: "1",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "7",
            SOURCE_PLANE_INDEX_FIELD: "2",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
    )


def test_source_metadata_request_does_not_infer_volumetric_planes_from_scalar_z():
    pixels = np.ones((3, 4, 5), dtype=np.uint16)
    metadata = ImagePayloadSourceMetadataContext(
        SourceImageIdentity(
            "/tmp/plate/images/A01_s1_w3_z5.tif",
            {
                "well": "A01",
                "site": "1",
                "channel": "3",
                "z_index": "5",
            },
        )
    ).metadata(pixels)

    assert not metadata.source_image_provenance_planes.has_values
    assert metadata.source_component_metadata == {
        "well": "A01",
        "site": "1",
        "channel": "3",
        "z_index": "5",
    }


def test_stream_outputs_projects_volumetric_source_stack_as_z_planes():
    class ZIndexParserStub(ParserStub):
        def parse_filename(self, name):
            stem = Path(name).stem
            well, site, channel, z_index = stem.split("_")
            return complete_component_metadata(
                {
                    "well": well,
                    "site": site.removeprefix("s"),
                    "channel": channel.removeprefix("w"),
                    "z_index": z_index.removeprefix("z"),
                    "extension": "".join(Path(name).suffixes),
                }
            )

        def construct_filename(self, **metadata):
            extension = metadata.get("extension") or ".tif"
            return (
                f"{metadata['well']}_s{metadata['site']}_w{metadata['channel']}"
                f"_z{metadata['z_index']}{extension}"
            )

    path = "/tmp/output/A01_s1_w1_z1.tif"
    pixels = np.ones((2, 5, 6), dtype=np.uint16)
    metadata = ImagePayloadSourceMetadataContext(
        SourceImageIdentity(
            path,
            complete_component_metadata(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "z_index": "1",
                    SOURCE_PLANE_INDEX_FIELD: "0",
                    SOURCE_PLANE_COUNT_FIELD: "2",
                }
            ),
        )
    ).metadata(pixels)
    payload = metadata.payload_with(pixels, None)
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager, parser=ZIndexParserStub())
    plan = function_step_plan("Resize")
    record_output_path(context, plan, path)

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == [
        "/tmp/output/A01_s1_w1_z1.tif",
        "/tmp/output/A01_s1_w1_z2.tif",
    ]
    assert backend == "napari_stream"
    assert [item.shape for item in streamed_data] == [(5, 6), (5, 6)]
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.source.metadata.metadata_by_index == (
        expected_viewer_metadata(
            {"well": "A01", "site": "1", "channel": "1", "z_index": "1"}
        ),
        expected_viewer_metadata(
            {"well": "A01", "site": "1", "channel": "1", "z_index": "2"}
        ),
    )


def test_stream_outputs_projects_declared_channel_stack_axis():
    path = "/tmp/output/A01_s1_w1.tif"
    pixels = np.ones((2, 5, 6), dtype=np.uint16)
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(path, path),
            component_metadata=(
                complete_component_metadata(
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "1",
                        "z_index": "1",
                    }
                ),
                complete_component_metadata(
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "2",
                        "z_index": "1",
                    }
                ),
            ),
        ),
    ).payload_with(pixels, None)
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan(
        "CalculateMath",
        variable_components=(VariableComponents.CHANNEL,),
    )
    record_output_path(context, plan, path)

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == ["/tmp/output/A01_s1_w1.tif", "/tmp/output/A01_s1_w2.tif"]
    assert backend == "napari_stream"
    assert [item.shape for item in streamed_data] == [(5, 6), (5, 6)]
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.source.metadata.metadata_by_index == (
        expected_viewer_metadata(
            {"well": "A01", "site": "1", "channel": "1", "z_index": "1"}
        ),
        expected_viewer_metadata(
            {"well": "A01", "site": "1", "channel": "2", "z_index": "1"}
        ),
    )


def test_stream_outputs_projects_stack_planes_with_item_source_paths():
    path = "/tmp/output/A01_s1_w1.tif"
    first_path = "/tmp/source/A01_s1_w1.tif"
    second_path = "/tmp/source/A01_s1_w2.tif"
    pixels = np.ones((2, 5, 6), dtype=np.uint16)
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(first_path, second_path),
            component_metadata=(
                complete_component_metadata(
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "1",
                        "z_index": "1",
                    }
                ),
                complete_component_metadata(
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "2",
                        "z_index": "1",
                    }
                ),
            ),
        ),
    ).payload_with(pixels, None)
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan(
        "MeasureColocalization",
        variable_components=(VariableComponents.CHANNEL,),
    )
    record_output_path(context, plan, path)

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == ["/tmp/output/A01_s1_w1.tif", "/tmp/output/A01_s1_w2.tif"]
    assert backend == "napari_stream"
    assert [item.shape for item in streamed_data] == [(5, 6), (5, 6)]
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.source.metadata.metadata_by_index == (
        expected_viewer_metadata(
            {"well": "A01", "site": "1", "channel": "1", "z_index": "1"}
        ),
        expected_viewer_metadata(
            {"well": "A01", "site": "1", "channel": "2", "z_index": "1"}
        ),
    )


def test_stream_outputs_rejects_unaddressed_stack_payload_metadata():
    path = "/tmp/output/A01_s1_w1.tif"
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_component_metadata=complete_component_metadata(
            {
                "well": "A01",
                "site": "1",
                "channel": "1",
            }
        ),
    ).payload_with(np.ones((2, 5, 6), dtype=np.uint16), None)
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan("Resize")
    record_output_path(context, plan, path)

    with pytest.raises(ValueError, match="per-slice component metadata"):
        StreamOutputsAuthority.stream_outputs(context, plan)


def test_stream_outputs_projects_semantic_image_stack_before_viewer_backend():
    path = "/tmp/output/A01_s1_w1.tif"
    pixels = np.ones((2, 3, 4, 3), dtype=np.uint8)
    first_metadata = complete_component_metadata(
        {"well": "A01", "site": "1", "channel": "1"}
    )
    second_metadata = complete_component_metadata(
        {"well": "A01", "site": "1", "channel": "2"}
    )
    payload = ImagePayloadMetadata(
        source_channel_axis=-1,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=(
            SourceImageProvenancePlanes.from_components(
                component_metadata=(first_metadata, second_metadata)
            )
        ),
    ).payload_with(pixels, None)
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan("OverlayObjects")
    record_output_path(context, plan, path)

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == ["/tmp/output/A01_s1_w1.tif", "/tmp/output/A01_s1_w2.tif"]
    assert backend == "napari_stream"
    assert [item.shape for item in streamed_data] == [(3, 4, 3), (3, 4, 3)]
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.source.item_fields == {"source_channel_axis": -1}
    assert stream_request.source.metadata.metadata_by_index == (
        expected_viewer_metadata(first_metadata),
        expected_viewer_metadata(second_metadata),
    )
    assert stream_request.message_extra["component_names_metadata"] == {
        "channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"},
        "well": {"A01": None},
        "site": {"1": None},
        "z_index": {"1": None},
        "timepoint": {"1": None},
    }
    assert stream_request.message_extra["component_value_domain"] == {
        "well": ["A01"],
        "site": [1],
        "channel": [1, 2, 3],
        "z_index": [1],
        "timepoint": [1],
    }
    assert stream_request.producer.identities[0].output_key == "main"


def test_stream_outputs_batches_named_main_outputs_by_projection():
    main_path = "/tmp/output/A01_s1_w1.tif"
    artifact_path = "/tmp/output/A01_s1_w2.tif"
    filemanager = FileManagerStub(
        {
            main_path: image_payload_with_source_metadata(
                np.ones((2, 3), dtype=np.uint16),
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "extension": ".tif",
                },
            ),
            artifact_path: image_payload_with_source_metadata(
                np.ones((2, 3), dtype=np.uint16) * 2,
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "2",
                    "extension": ".tif",
                },
            ),
        }
    )
    context = context_stub(filemanager)
    plan = function_step_plan("OverlayOutlines")
    record_output_path(context, plan, main_path)
    record_output_path(
        context,
        plan,
        artifact_path,
        output_context=AlignedImageSliceContext.main_flow(
            "OverlayImage",
            artifact_kind=ImageArtifactType.value,
        ),
    )

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == [main_path, artifact_path]
    assert backend == "napari_stream"
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert tuple(
        producer.output_key for producer in stream_request.producer.identities
    ) == ("main", "OverlayImage")
    assert [payload.shape for payload in streamed_data] == [(2, 3), (2, 3)]


def test_stream_outputs_partitions_one_producer_by_image_axis_fields():
    scalar_path = "/tmp/output/A01_s1_w1.tif"
    color_path = "/tmp/output/A01_s1_w2.tif"
    scalar_metadata = complete_component_metadata(
        {"well": "A01", "site": "1", "channel": "1"}
    )
    color_metadata = complete_component_metadata(
        {"well": "A01", "site": "1", "channel": "2"}
    )
    filemanager = FileManagerStub(
        {
            scalar_path: ImagePayloadMetadata(
                source_component_metadata=scalar_metadata,
                source_image_provenance_planes=(
                    SourceImageProvenancePlanes.from_components(
                        component_metadata=(scalar_metadata,)
                    )
                ),
            ).payload_with(np.ones((2, 3), dtype=np.uint16), None),
            color_path: ImagePayloadMetadata(
                source_channel_axis=-1,
                source_component_metadata=color_metadata,
                source_image_provenance_planes=(
                    SourceImageProvenancePlanes.from_components(
                        component_metadata=(color_metadata,)
                    )
                ),
            ).payload_with(np.ones((2, 3, 3), dtype=np.uint8), None),
        }
    )
    context = context_stub(filemanager)
    plan = function_step_plan("UntangleWorms")
    record_output_path(context, plan, scalar_path)
    record_output_path(
        context,
        plan,
        color_path,
        output_context=AlignedImageSliceContext.main_flow(
            "WormMask",
            artifact_kind=ImageArtifactType.value,
        ),
    )

    StreamOutputsAuthority.stream_outputs(context, plan)

    assert len(filemanager.saved_batches) == 2
    scalar_batch, color_batch = filemanager.saved_batches
    assert scalar_batch[1] == [scalar_path]
    assert color_batch[1] == [color_path]
    assert (
        scalar_batch[3][ViewerStreamKwarg.STREAM_REQUEST.value].source.item_fields == {}
    )
    assert color_batch[3][
        ViewerStreamKwarg.STREAM_REQUEST.value
    ].source.item_fields == {"source_channel_axis": -1}


def test_stream_outputs_rejects_unidentified_stack_without_per_slice_metadata():
    path = "/tmp/output/A01_s1_w1.tif"
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_component_metadata=complete_component_metadata(
            {"well": "A01", "site": "1", "channel": "1"}
        ),
    ).payload_with(np.ones((8, 520, 696), dtype=np.uint16), None)
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan("IdentifyPrimaryObjects")
    record_output_path(context, plan, path)

    with pytest.raises(ValueError, match="per-slice component metadata"):
        StreamOutputsAuthority.stream_outputs(context, plan)


def test_stream_outputs_keeps_recorded_main_stream():
    path = "/tmp/output/A01_s1_w1.tif"
    pixels = np.ones((2, 3), dtype=np.uint16)
    filemanager = FileManagerStub(
        {
            path: image_payload_with_source_metadata(
                pixels,
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "extension": ".tif",
                },
            )
        }
    )
    context = context_stub(filemanager)
    plan = function_step_plan("EnhanceOrSuppressFeatures", pipeline_position=4)
    record_output_path(context, plan, path)

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_data == [pixels]
    assert streamed_paths == [path]
    assert backend == "napari_stream"
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.producer.identities[0].output_kind == "main"


def test_stream_outputs_restores_exact_identity_for_reloaded_passthrough() -> None:
    path = "/tmp/output/A01_s1_w2.tif"
    filemanager = FileManagerStub({path: np.ones((2, 3), dtype=np.uint16)})
    context = context_stub(filemanager)
    plan = function_step_plan("MeasureObjectIntensity", pipeline_position=5)
    record_output_path(context, plan, path)

    StreamOutputsAuthority.stream_outputs(context, plan)

    stream_request = filemanager.saved_batches[0][3][
        ViewerStreamKwarg.STREAM_REQUEST.value
    ]
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": 1,
            "channel": 2,
            "z_index": 1,
            "timepoint": 1,
        },
    )


def test_stream_outputs_skips_object_label_main_flow_payloads():
    path = "/tmp/output/A01_s1_w1.tif"
    payload = image_payload_with_source_metadata(
        np.zeros((2, 3), dtype=np.uint16),
        {
            "well": "A01",
            "site": "1",
            "channel": "1",
            "extension": ".tif",
        },
    )
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan("ResizeObjects")
    record_output_path(
        context,
        plan,
        path,
        output_context=AlignedImageSliceContext.main_flow(
            output_key="Nuclei",
            artifact_kind=ObjectLabelsArtifactType.value,
        ),
    )

    StreamOutputsAuthority.stream_outputs(context, plan)

    assert filemanager.saved_batches == []


def test_image_persistence_skips_object_label_main_flow_payloads():
    path = "/tmp/output/A01_s1_w1.tif"
    filemanager = FileManagerStub({path: object()})
    context = context_stub(filemanager)
    plan = function_step_plan("RelateObjects")
    record_output_path(
        context,
        plan,
        path,
        output_context=AlignedImageSliceContext.main_flow(
            output_key="Children",
            artifact_kind=ObjectLabelsArtifactType.value,
        ),
    )

    assert ProducedMemoryPathsAuthority.paths(context, plan) == []


def test_metadata_writer_skips_owner_without_image_outputs():
    path = "/tmp/output/A01_s1_w1.tif"
    filemanager = FileManagerStub({path: object()})
    context = context_stub(filemanager)
    plan = function_step_plan("RelateObjects")
    plan.create_openhcs_metadata = True
    plan.write_backend = Backend.DISK.value
    record_output_path(
        context,
        plan,
        path,
        output_context=AlignedImageSliceContext.main_flow(
            output_key="Children",
            artifact_kind=ObjectLabelsArtifactType.value,
        ),
    )

    OpenHCSMetadataWriter.write(context, plan)


def test_completed_plate_metadata_includes_outputs_written_after_owner_axis(
    tmp_path,
):
    plate_root = tmp_path / "output_plate"
    images_dir = plate_root / "images"
    images_dir.mkdir(parents=True)
    first_image = images_dir / "A01_s1_w1.tif"
    later_image = images_dir / "B03_s1_w1.tif"
    first_image.write_bytes(b"first")

    filemanager = FileManager({Backend.DISK.value: DiskStorageBackend()})
    owner_context = context_stub(filemanager)
    owner_context.metadata_cache = {
        AllComponents.WELL: {"A01": None, "B03": None},
        AllComponents.SITE: {"1": None},
        AllComponents.CHANNEL: {"1": "DNA"},
        AllComponents.Z_INDEX: {"1": None},
        AllComponents.TIMEPOINT: {"1": None},
    }
    owner_plan = function_step_plan("final")
    owner_plan.output_dir = images_dir
    owner_plan.output_plate_root = str(plate_root)
    owner_plan.sub_dir = "images"
    owner_plan.analysis_results_dir = str(plate_root / "images_results")
    owner_plan.write_backend = Backend.DISK.value
    owner_plan.create_openhcs_metadata = True
    owner_context.step_plans = {owner_plan.step_index: owner_plan}
    record_output_path(owner_context, owner_plan, str(first_image))

    OpenHCSMetadataWriter.write(owner_context, owner_plan)
    initial_metadata = json.loads(
        (plate_root / "openhcs_metadata.json").read_text(encoding="utf-8")
    )["subdirectories"]["images"]
    assert initial_metadata["wells"] == {"A01": None}

    later_image.write_bytes(b"later")
    follower_context = context_stub(filemanager)
    follower_plan = function_step_plan("final")
    follower_plan.axis_id = "B03"
    follower_plan.output_dir = images_dir
    follower_plan.output_plate_root = str(plate_root)
    follower_plan.sub_dir = "images"
    follower_plan.analysis_results_dir = str(plate_root / "images_results")
    follower_plan.write_backend = Backend.DISK.value
    follower_context.step_plans = {follower_plan.step_index: follower_plan}

    OpenHCSMetadataWriter.finalize_completed_plate(
        {"A01": owner_context, "B03": follower_context}
    )

    metadata = json.loads(
        (plate_root / "openhcs_metadata.json").read_text(encoding="utf-8")
    )["subdirectories"]["images"]
    assert metadata["image_files"] == [
        "images/A01_s1_w1.tif",
        "images/B03_s1_w1.tif",
    ]
    assert metadata["wells"] == {"A01": None, "B03": None}
