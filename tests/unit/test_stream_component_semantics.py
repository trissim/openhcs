from types import MappingProxyType

import pytest

from openhcs.core.source_metadata import ORIGINAL_SOURCE_METADATA_FIELD
from openhcs.core.steps.stream_component_semantics import (
    StreamSourceComponentMetadataItems,
    StreamViewerComponentMetadataProjector,
)


def test_stream_viewer_component_metadata_projector_keeps_only_declared_axes():
    projector = StreamViewerComponentMetadataProjector(
        ("well", "site", "channel", "timepoint")
    )

    projected = projector.project(
        {
            "well": "A01",
            "Site": "2",
            "ChannelNumber": "5",
            "extension": ".tif",
            "OpenHCSImageType": "Color image",
            ORIGINAL_SOURCE_METADATA_FIELD: MappingProxyType({"FrameNumber": "0011"}),
        }
    )

    assert projected == {
        "well": "A01",
        "site": "2",
        "channel": "5",
    }


def test_stream_source_component_metadata_items_project_viewer_metadata_by_index():
    source_metadata = StreamSourceComponentMetadataItems.from_values(
        (
            {
                "well": "A01",
                "site": "1",
                "channel": "2",
                ORIGINAL_SOURCE_METADATA_FIELD: MappingProxyType(
                    {"FrameNumber": "0011"}
                ),
            },
        )
    )

    viewer_metadata = source_metadata.viewer_source_metadata(("well", "channel"))

    assert viewer_metadata.metadata_by_index == (
        {
            "well": "A01",
            "channel": "2",
        },
    )


def test_stream_viewer_component_metadata_projector_requires_source_metadata():
    projector = StreamViewerComponentMetadataProjector(("well",))

    with pytest.raises(ValueError, match="requires source component metadata"):
        projector.project_required(index=3, metadata=None)
