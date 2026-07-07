from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from polystore.streaming.viewer_transport import (
    ViewerStreamBackendKwargs,
    ViewerStreamKwarg,
    ViewerStreamRequest,
    ViewerStreamSource,
)
from polystore.streaming.viewer_transport import ViewerStreamSourceIdentity
from zmqruntime.viewer_protocol import ViewerTransportEndpoint

from openhcs.constants.constants import AllComponents, Backend, VariableComponents
from openhcs.core.aligned_image_payload import AlignedImageSliceContext
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ImagePayloadSourceMetadataContext,
    RuntimeImagePayloadContext,
    SourceImageProvenancePlanes,
)
from openhcs.core.source_image_provenance import SourceImageIdentity
from openhcs.core.source_metadata import (
    SOURCE_PLANE_COUNT_FIELD,
    SOURCE_PLANE_INDEX_FIELD,
)
from openhcs.core.steps.function_outputs import StreamOutputsAuthority
from openhcs.core.steps.function_output_identity import FunctionOutputIdentity
from openhcs.core.steps.function_output_manifest import (
    ProducedOutputSemantics,
    step_output_manifest,
)
from openhcs.core.step_dependencies import StepInputDependency
from openhcs.utils.display_config_factory import ViewerDisplayConfigObject


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


class StreamingConfigStub(ViewerDisplayConfigObject):
    backend = SimpleNamespace(value="napari_stream")
    COMPONENT_ORDER = ("well", "site", "channel", "z_index", "timepoint")
    host = "127.0.0.1"
    port = 5555
    transport_mode = "tcp"

    def component_modes(self):
        return {component: "stack" for component in self.COMPONENT_ORDER}

    def streaming_viewer_surface(self, context):
        return StreamingViewerSurfaceStub(self, context)


class StreamingViewerSurfaceStub:
    def __init__(self, display_config, context):
        self.runtime_config = SimpleNamespace(
            transport_endpoint=ViewerTransportEndpoint(
                host=display_config.host,
                port=display_config.port,
                transport_mode=display_config.transport_mode,
            )
        )
        self.display_config = display_config
        self.source = ViewerStreamSourceIdentity(
            microscope_handler=context.microscope_handler,
            plate_path=context.plate_path,
        )

    def viewer_backend_kwargs(self, *, producer, source_metadata, message_context):
        return ViewerStreamBackendKwargs(
            ViewerStreamRequest.from_message_context(
                message_context=message_context,
                viewer_transport=self.runtime_config.transport_endpoint,
                display_config=self.display_config,
                source=ViewerStreamSource(
                    identity=self.source,
                    metadata=source_metadata,
                ),
                producer=producer,
            )
        )


class ParserStub:
    def parse_filename(self, name):
        stem = Path(name).stem
        well, site, channel = stem.split("_")
        return complete_component_metadata({
            "well": well,
            "site": site.removeprefix("s"),
            "channel": channel.removeprefix("w"),
            "extension": "".join(Path(name).suffixes),
        })

    def construct_filename(self, **metadata):
        extension = metadata.get("extension") or ".tif"
        return (
            f"{metadata['well']}_s{metadata['site']}_w{metadata['channel']}"
            f"{extension}"
        )


class MetadataHandlerStub:
    def __init__(self, values=None):
        self.values = values or {}

    def find_metadata_file(self, root):
        return Path(root) / "openhcs_metadata.json"

    def get_component_values(self, _root, component):
        return self.values.get(component, {})


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
    return context


def function_step_plan(
    path: str,
    step_name: str,
    variable_components: tuple[VariableComponents, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        streaming_configs=(StreamingConfigStub(),),
        artifact_outputs={},
        output_dir=Path("/tmp/output"),
        has_materialized_output=False,
        step_name=step_name,
        pipeline_position=3,
        step_scope_id="step-scope-3",
        axis_id="A01",
        get_paths_for_axis=lambda *_args: [path],
        main_input_dependency=SimpleNamespace(kind=None),
        variable_components=variable_components,
    )


def record_output_path(context, plan, path, output_context=None):
    metadata = context.microscope_handler.parser.parse_filename(Path(path).name)
    assert metadata is not None
    identity = FunctionOutputIdentity(
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
                identity,
                output_context=output_context,
            )
        ],
    )


def test_step_output_manifest_prefers_main_dependency_over_auxiliary_artifact_inputs():
    context = context_stub(FileManagerStub({}))
    manifest = step_output_manifest(context)

    main_producer = function_step_plan("A01_s1_w1.tif", "ErodeImage")
    main_producer.step_scope_id = "main-producer"
    main_producer.pipeline_position = 24
    seed_producer = function_step_plan("A01_s1_w2.tif", "ConvertObjectsToImage")
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

    consumer = function_step_plan("ignored.tif", "Watershed")
    consumer.main_input_dependency = StepInputDependency.step_output(
        source_step_index=24,
        source_step_scope_id="main-producer",
    )
    consumer.source_binding_plan = SimpleNamespace(has_primary_content=False)
    consumer.artifact_inputs = {
        "MembFinal": ArtifactInputPlan(
            name="MembFinal",
            path="memb",
            artifact_type=ImageArtifactType,
            source_step_id=24,
            source_step_scope_id="main-producer",
        ),
        "cellSeeds": ArtifactInputPlan(
            name="cellSeeds",
            path="seeds",
            artifact_type=ImageArtifactType,
            source_step_id=14,
            source_step_scope_id="seed-producer",
        ),
    }

    assert manifest.producer_paths_for(consumer) == ("A01_s1_w1.tif",)
    assert manifest.filter_to_producer_paths(
        consumer,
        ("A01_s1_w1.tif", "A01_s1_w2.tif"),
        context.microscope_handler.parser,
    ) == ["A01_s1_w1.tif"]


def image_payload_with_source_metadata(pixels, metadata, mask=None):
    completed_metadata = complete_component_metadata(metadata)
    return RuntimeImagePayloadContext(
        pixels,
        mask=mask,
        metadata=ImagePayloadMetadata(
            source_component_metadata=completed_metadata,
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                component_metadata=(completed_metadata,)
            ),
        ),
    ).payload()


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
    plan = function_step_plan(path, "IdentifyPrimaryObjects")
    record_output_path(context, plan, path)

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == [path]
    assert backend == "napari_stream"
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.port == 5555
    assert stream_request.source.metadata.metadata_by_index == (
        expected_viewer_metadata({
            "well": "A01",
            "site": "1",
            "channel": "1",
        }),
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
        }
    }
    assert stream_request.producer.identity.to_payload() == {
        "origin": "pipeline",
        "output_kind": "main",
        "output_key": "main",
        "step_name": "IdentifyPrimaryObjects",
        "pipeline_position": 3,
        "step_scope_id": "step-scope-3",
        "invocation_key": None,
        "artifact_kind": None,
    }
    assert streamed_data == [pixels]


def test_stream_outputs_rejects_unaddressed_payload_metadata():
    class ParserStub:
        def parse_filename(self, name):
            assert name == "A01_s1_w3.tif"
            return {"well": "A01", "site": "1", "channel": "3"}

    path = "/tmp/output/A01_s1_w3.tif"
    pixels = np.ones((2, 3), dtype=np.uint16)
    filemanager = FileManagerStub({path: pixels})
    context = context_stub(filemanager, ParserStub())
    plan = function_step_plan(path, "EnhanceOrSuppressFeatures")
    record_output_path(context, plan, path)

    with pytest.raises(ValueError, match="source component metadata"):
        StreamOutputsAuthority.stream_outputs(context, plan)


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
    ).metadata_request(pixels).metadata()

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
    ).metadata_request(pixels).metadata()

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
            return complete_component_metadata({
                "well": well,
                "site": site.removeprefix("s"),
                "channel": channel.removeprefix("w"),
                "z_index": z_index.removeprefix("z"),
                "extension": "".join(Path(name).suffixes),
            })

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
            complete_component_metadata({
                "well": "A01",
                "site": "1",
                "channel": "1",
                "z_index": "1",
                SOURCE_PLANE_INDEX_FIELD: "0",
                SOURCE_PLANE_COUNT_FIELD: "2",
            }),
        )
    ).metadata_request(pixels).metadata()
    payload = RuntimeImagePayloadContext(pixels, mask=None, metadata=metadata).payload()
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager, parser=ZIndexParserStub())
    plan = function_step_plan(path, "Resize")
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
    payload = RuntimeImagePayloadContext(
        pixels,
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata=complete_component_metadata({
                "well": "A01",
                "site": "1",
                "channel": "1",
                "z_index": "1",
            })
        ),
    ).payload()
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan(
        path,
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
    payload = RuntimeImagePayloadContext(
        pixels,
        mask=None,
        metadata=ImagePayloadMetadata(
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
            )
        ),
    ).payload()
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan(
        path,
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
    payload = RuntimeImagePayloadContext(
        np.ones((2, 5, 6), dtype=np.uint16),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
            }
        ),
    ).payload()
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan(path, "Resize")
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
    payload = RuntimeImagePayloadContext(
        pixels,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes = SourceImageProvenancePlanes.from_components(component_metadata = (first_metadata, second_metadata))),
    mask = None).payload()
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan(path, "OverlayObjects")
    record_output_path(context, plan, path)

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == ["/tmp/output/A01_s1_w1.tif", "/tmp/output/A01_s1_w2.tif"]
    assert backend == "napari_stream"
    assert [item.shape for item in streamed_data] == [(3, 4, 3), (3, 4, 3)]
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
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
    assert stream_request.producer.identity.output_key == "main"


def test_stream_outputs_partitions_mixed_producer_identities():
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
    plan = function_step_plan(main_path, "OverlayOutlines")
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

    assert len(filemanager.saved_batches) == 2
    first_data, first_paths, first_backend, first_kwargs = filemanager.saved_batches[0]
    second_data, second_paths, second_backend, second_kwargs = filemanager.saved_batches[1]
    assert first_paths == [main_path]
    assert second_paths == [artifact_path]
    assert first_backend == second_backend == "napari_stream"
    assert first_kwargs[
        ViewerStreamKwarg.STREAM_REQUEST.value
    ].producer.identity.output_key == "main"
    assert second_kwargs[
        ViewerStreamKwarg.STREAM_REQUEST.value
    ].producer.identity.output_key == "OverlayImage"
    assert first_data[0].shape == second_data[0].shape == (2, 3)


def test_stream_outputs_rejects_unidentified_stack_without_per_slice_metadata():
    class ParserStub:
        def parse_filename(self, name):
            assert name == "A01_s1_w1.tif"
            return {"well": "A01", "site": "1", "channel": "1"}

    path = "/tmp/output/A01_s1_w1.tif"
    payload = np.ones((8, 520, 696), dtype=np.uint16)
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager, ParserStub())
    plan = function_step_plan(path, "IdentifyPrimaryObjects")
    record_output_path(context, plan, path)

    with pytest.raises(ValueError, match="per-slice component metadata"):
        StreamOutputsAuthority.stream_outputs(context, plan)


def test_stream_outputs_keeps_main_stream_with_adapter_managed_artifact_outputs():
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
    invocation = SimpleNamespace(
        artifact_output_keys=("SemanticImage",),
        contract=SimpleNamespace(
            runtime_adapter=SimpleNamespace(manages_artifact_inputs=True)
        ),
    )
    plan = SimpleNamespace(
        streaming_configs=(StreamingConfigStub(),),
        artifact_outputs={"SemanticImage": object()},
        compiled_function_pattern=SimpleNamespace(
            iter_invocations=lambda: iter((invocation,))
        ),
        output_dir=Path("/tmp/output"),
        has_materialized_output=False,
        step_name="EnhanceOrSuppressFeatures",
        pipeline_position=4,
        step_scope_id="step-scope-4",
        axis_id="A01",
        get_paths_for_axis=lambda *_args: [path],
        main_input_dependency=SimpleNamespace(kind=None),
        variable_components=(),
    )
    record_output_path(context, plan, path)

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_data == [pixels]
    assert streamed_paths == [path]
    assert backend == "napari_stream"
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.producer.identity.output_kind == "main"


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
    plan = function_step_plan(path, "ResizeObjects")
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
