from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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


def function_step_plan(path: str, step_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        streaming_configs=(StreamingConfigStub(),),
        output_dir=Path("/tmp/output"),
        has_materialized_output=False,
        step_name=step_name,
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
    context = SimpleNamespace(filemanager=filemanager)
    plan = function_step_plan(path, "IdentifyPrimaryObjects")

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == [path]
    assert backend == "napari_stream"
    assert kwargs == {
        "port": 5555,
        "source": "IdentifyPrimaryObjects",
        "component_metadata_by_path": (None,),
    }
    assert streamed_data == [pixels]


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
    context = SimpleNamespace(filemanager=filemanager)
    plan = function_step_plan(path, "OverlayObjects")

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == [path, path]
    assert backend == "napari_stream"
    assert [item.shape for item in streamed_data] == [(3, 4, 3), (3, 4, 3)]
    assert kwargs["component_metadata_by_path"] == (first_metadata, second_metadata)


def test_stream_outputs_rejects_unidentified_image_stack():
    path = "/tmp/output/A01_s1_w1.tif"
    payload = np.ones((2, 3, 4, 3), dtype=np.uint8)
    filemanager = FileManagerStub({path: payload})
    context = SimpleNamespace(filemanager=filemanager)
    plan = function_step_plan(path, "OverlayObjects")

    with pytest.raises(ValueError, match="requires per-slice component metadata"):
        StreamOutputsAuthority.stream_outputs(context, plan)
