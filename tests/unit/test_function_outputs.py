from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openhcs.constants.constants import Backend
from openhcs.core.runtime_values import ImagePayloadMetadata, image_payload_with_context
from openhcs.core.steps.function_outputs import StreamOutputsAuthority


class FileManagerStub:
    def __init__(self, memory_payloads):
        self.memory_payloads = memory_payloads
        self.saved_batches = []

    def load_batch(self, paths, backend):
        assert backend == Backend.MEMORY.value
        return [self.memory_payloads[path] for path in paths]

    def save_batch(self, data, paths, backend, **kwargs):
        self.saved_batches.append((data, paths, backend, kwargs))


class StreamingConfigStub:
    backend = SimpleNamespace(value="napari_stream")

    def get_streaming_kwargs(self, _context):
        return {"port": 5555}


class NullParserStub:
    def parse_filename(self, _name):
        return None


def context_stub(filemanager, parser=None):
    return SimpleNamespace(
        filemanager=filemanager,
        microscope_handler=SimpleNamespace(parser=parser or NullParserStub()),
    )


def function_step_plan(path: str, step_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        streaming_configs=(StreamingConfigStub(),),
        artifact_outputs={},
        output_dir=Path("/tmp/output"),
        has_materialized_output=False,
        step_name=step_name,
        pipeline_position=3,
        step_scope_id="step-scope-3",
        get_paths_for_axis=lambda *_args: [path],
    )


def test_stream_outputs_unwraps_runtime_image_payloads_before_viewer_backend():
    path = "/tmp/output/A01_s1_w1.tif"
    pixels = np.ones((2, 3), dtype=np.uint16)
    payload = image_payload_with_context(
        pixels,
        mask=np.ones_like(pixels, dtype=bool),
    )
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan(path, "IdentifyPrimaryObjects")

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == [path]
    assert backend == "napari_stream"
    producer_identity = kwargs.pop("producer_identity")
    assert kwargs == {"port": 5555, "component_metadata_by_path": (None,)}
    assert producer_identity.to_payload() == {
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

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(_streamed_data, _streamed_paths, _backend, kwargs)] = filemanager.saved_batches
    assert kwargs["component_metadata_by_path"] == (
        {"well": "A01", "site": "1", "channel": "3"},
    )


def test_stream_outputs_projects_semantic_image_stack_before_viewer_backend():
    path = "/tmp/output/A01_s1_w1.tif"
    pixels = np.ones((2, 3, 4, 3), dtype=np.uint8)
    first_metadata = {"well": "A01", "site": "1", "channel": "1"}
    second_metadata = {"well": "A01", "site": "1", "channel": "2"}
    payload = image_payload_with_context(
        pixels,
        metadata=ImagePayloadMetadata(
            channel_source_component_metadata=(first_metadata, second_metadata),
        ),
    )
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager)
    plan = function_step_plan(path, "OverlayObjects")

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == [path, path]
    assert backend == "napari_stream"
    assert [item.shape for item in streamed_data] == [(3, 4, 3), (3, 4, 3)]
    assert kwargs["component_metadata_by_path"] == (first_metadata, second_metadata)
    assert kwargs["producer_identity"].output_key == "main"


def test_stream_outputs_keeps_unidentified_stack_as_single_payload_with_path_metadata():
    class ParserStub:
        def parse_filename(self, name):
            assert name == "A01_s1_w1.tif"
            return {"well": "A01", "site": "1", "channel": "1"}

    path = "/tmp/output/A01_s1_w1.tif"
    payload = np.ones((8, 520, 696), dtype=np.uint16)
    filemanager = FileManagerStub({path: payload})
    context = context_stub(filemanager, ParserStub())
    plan = function_step_plan(path, "IdentifyPrimaryObjects")

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == [path]
    assert backend == "napari_stream"
    np.testing.assert_array_equal(streamed_data[0], payload)
    assert kwargs["component_metadata_by_path"] == (
        {"well": "A01", "site": "1", "channel": "1"},
    )


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
        get_paths_for_axis=lambda *_args: [path],
    )

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_data == [pixels]
    assert streamed_paths == [path]
    assert backend == "napari_stream"
    assert kwargs["producer_identity"].output_kind == "main"
