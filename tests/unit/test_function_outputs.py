from pathlib import Path
from types import SimpleNamespace

import numpy as np

from polystore.streaming.viewer_transport import ViewerStreamKwarg
from polystore.streaming.viewer_transport import ViewerStreamSourceIdentity
from zmqruntime.viewer_protocol import ViewerTransportEndpoint

from openhcs.constants.constants import Backend
from openhcs.core.runtime_values import ImagePayloadMetadata, RuntimeImagePayloadContext, SourceImageProvenancePlanes
from openhcs.core.steps.function_outputs import StreamOutputsAuthority
from openhcs.core.steps.function_output_identity import FunctionOutputIdentity
from openhcs.core.steps.function_output_manifest import (
    ProducedOutputSemantics,
    step_output_manifest,
)
from openhcs.utils.display_config_factory import ViewerDisplayConfigObject


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


class ParserStub:
    def parse_filename(self, name):
        stem = Path(name).stem
        well, site, channel = stem.split("_")
        return {
            "well": well,
            "site": site.removeprefix("s"),
            "channel": channel.removeprefix("w"),
            "extension": "".join(Path(name).suffixes),
        }


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


def function_step_plan(path: str, step_name: str) -> SimpleNamespace:
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
    )


def record_output_path(context, plan, path):
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
                ImagePayloadMetadata().source_provenance,
            )
        ],
    )


def test_stream_outputs_unwraps_runtime_image_payloads_before_viewer_backend():
    path = "/tmp/output/A01_s1_w1.tif"
    pixels = np.ones((2, 3), dtype=np.uint16)
    payload = RuntimeImagePayloadContext(
        pixels,
        mask=np.ones_like(pixels, dtype=bool),
    metadata = ImagePayloadMetadata()).payload()
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
    assert stream_request.source.metadata.component_metadata_by_path == (
        {
            "well": "A01",
            "site": "1",
            "channel": "1",
            "extension": ".tif",
        },
    )
    assert stream_request.message_extra == {
        "component_value_domain": {
            "well": ["A01"],
            "site": [1],
            "channel": [1, 2, 3],
        },
        "component_names_metadata": {
            "channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"},
            "well": {"A01": None},
            "site": {"1": None},
        }
    }
    assert stream_request.producer_identity.to_payload() == {
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


def test_stream_outputs_uses_path_metadata_when_payload_has_none():
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

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(_streamed_data, _streamed_paths, _backend, kwargs)] = filemanager.saved_batches
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.source.metadata.component_metadata_by_path == (
        {"well": "A01", "site": "1", "channel": "3"},
    )
    assert stream_request.message_extra["component_names_metadata"] == {
        "channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"},
        "well": {"A01": None},
        "site": {"1": None},
    }
    assert stream_request.message_extra["component_value_domain"] == {
        "well": ["A01"],
        "site": [1],
        "channel": [1, 2, 3],
    }


def test_stream_outputs_projects_semantic_image_stack_before_viewer_backend():
    path = "/tmp/output/A01_s1_w1.tif"
    pixels = np.ones((2, 3, 4, 3), dtype=np.uint8)
    first_metadata = {"well": "A01", "site": "1", "channel": "1"}
    second_metadata = {"well": "A01", "site": "1", "channel": "2"}
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
    assert streamed_paths == [path, path]
    assert backend == "napari_stream"
    assert [item.shape for item in streamed_data] == [(3, 4, 3), (3, 4, 3)]
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.source.metadata.component_metadata_by_path == (
        first_metadata,
        second_metadata,
    )
    assert stream_request.message_extra["component_names_metadata"] == {
        "channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"},
        "well": {"A01": None},
        "site": {"1": None},
    }
    assert stream_request.message_extra["component_value_domain"] == {
        "well": ["A01"],
        "site": [1],
        "channel": [1, 2, 3],
    }
    assert stream_request.producer_identity.output_key == "main"


def test_stream_outputs_skips_unidentified_stack_without_per_slice_metadata():
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

    StreamOutputsAuthority.stream_outputs(context, plan)

    assert filemanager.saved_batches == []


def test_stream_outputs_keeps_main_stream_with_adapter_managed_artifact_outputs():
    path = "/tmp/output/A01_s1_w1.tif"
    pixels = np.ones((2, 3), dtype=np.uint16)
    filemanager = FileManagerStub({path: pixels})
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
    )
    record_output_path(context, plan, path)

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_data == [pixels]
    assert streamed_paths == [path]
    assert backend == "napari_stream"
    stream_request = kwargs[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.producer_identity.output_kind == "main"
