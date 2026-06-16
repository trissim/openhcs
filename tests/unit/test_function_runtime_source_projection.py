from pathlib import Path

import numpy as np

from openhcs.core.steps.function_runtime import (
    VirtualWorkspacePathLookup,
    VirtualWorkspaceSourceProjection,
    _stack_payload_context,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
    image_payload_with_context,
)


def test_virtual_workspace_pipeline_start_files_preserve_virtual_identity(tmp_path: Path) -> None:
    """Pipeline-start source resolution must not collapse per-well virtual files."""

    plate_path = tmp_path / "plate"
    real_path = tmp_path / "source" / "image.png"
    virtual_a = plate_path / "W001_s001_w1_z001_t001.png"
    virtual_b = plate_path / "W002_s001_w1_z001_t001.png"
    projection = VirtualWorkspaceSourceProjection(
        source_paths_by_virtual_path={
            virtual_a.name: str(real_path),
            str(virtual_a): str(real_path),
            virtual_b.name: str(real_path),
            str(virtual_b): str(real_path),
        },
        source_metadata_by_path={
            virtual_a.name: {"well": "W001"},
            virtual_b.name: {"well": "W002"},
        },
        workspace_root=str(plate_path),
    )

    assert projection.pipeline_start_files() == (str(virtual_a), str(virtual_b))
    assert projection.pipeline_start_files(axis_id="W001") == (str(virtual_a),)
    assert projection.pipeline_start_files(axis_id="W002") == (str(virtual_b),)
    assert projection.source_metadata_for(
        VirtualWorkspacePathLookup.from_paths(virtual_a.name, str(virtual_a))
    ) == {"well": "W001"}
    assert projection.source_metadata_for(
        VirtualWorkspacePathLookup.from_paths(virtual_b.name, str(virtual_b))
    ) == {"well": "W002"}


def test_stack_payload_context_promotes_single_channel_slice_metadata() -> None:
    first = image_payload_with_context(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            channel_source_paths=("/input/A01_s001_w1_z001_t001.tif",),
            channel_source_component_metadata=(
                {"well": "A01", "site": 1, "channel": 1},
            ),
        ),
    )
    second = image_payload_with_context(
        np.ones((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            channel_source_paths=("/input/A01_s002_w1_z001_t001.tif",),
            channel_source_component_metadata=(
                {"well": "A01", "site": 2, "channel": 1},
            ),
        ),
    )
    stack = np.stack(
        (
            image_payload_data(first),
            image_payload_data(second),
        )
    )

    payload = _stack_payload_context((first, second), stack)
    metadata = image_payload_metadata(payload)

    assert metadata.channel_source_paths == (
        "/input/A01_s001_w1_z001_t001.tif",
        "/input/A01_s002_w1_z001_t001.tif",
    )
    assert tuple(dict(item) for item in metadata.channel_source_component_metadata) == (
        {"well": "A01", "site": 1, "channel": 1},
        {"well": "A01", "site": 2, "channel": 1},
    )
