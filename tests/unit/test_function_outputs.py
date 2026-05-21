from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openhcs.constants.constants import Backend
from openhcs.core.runtime_values import image_payload_with_context
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


def test_stream_outputs_unwraps_runtime_image_payloads_before_viewer_backend():
    path = "/tmp/output/A01_s1_w1.tif"
    pixels = np.ones((2, 3), dtype=np.uint16)
    payload = image_payload_with_context(
        pixels,
        mask=np.ones_like(pixels, dtype=bool),
    )
    filemanager = FileManagerStub({path: payload})
    context = SimpleNamespace(filemanager=filemanager)
    plan = SimpleNamespace(
        streaming_configs=(StreamingConfigStub(),),
        output_dir=Path("/tmp/output"),
        has_materialized_output=False,
        step_name="IdentifyPrimaryObjects",
        get_paths_for_axis=lambda *_args: [path],
    )

    StreamOutputsAuthority.stream_outputs(context, plan)

    [(streamed_data, streamed_paths, backend, kwargs)] = filemanager.saved_batches
    assert streamed_paths == [path]
    assert backend == "napari_stream"
    assert kwargs == {"port": 5555, "source": "IdentifyPrimaryObjects"}
    assert streamed_data == [pixels]
