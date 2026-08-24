"""Declaration-ownership tests for managed streaming viewer resolution."""

import pytest

from openhcs.core.config import StreamingConfig
from openhcs.core.streaming_config_declarations import (
    ViewerDeclarationABC,
    ViewerType,
)
from openhcs.runtime.viewer_protocol import ManagedViewerLifecycleMixin


@pytest.mark.parametrize("viewer_type", tuple(ViewerType), ids=lambda item: item.value)
def test_viewer_member_owns_nominal_leaf_and_runtime_type(viewer_type) -> None:
    declaration = viewer_type.declaration
    visualizer_type = declaration.visualizer_type()

    assert isinstance(declaration, ViewerDeclarationABC)
    assert visualizer_type.detached_server_entrypoint.viewer_type is viewer_type
    assert (
        ManagedViewerLifecycleMixin.__registry__[viewer_type.wire_value]
        is visualizer_type
    )


def test_viewer_identity_projects_config_and_presentation_names() -> None:
    assert ViewerType.NAPARI.config_key == "napari_streaming_config"
    assert ViewerType.NAPARI.step_plan_output_key == "napari_streaming_paths"
    assert ViewerType.NAPARI.display_name == "Napari"
    assert ViewerType.NAPARI.title == "OpenHCS Napari Visualization"
    assert set(StreamingConfig.__registry__) == set(ViewerType)


def test_viewer_identity_parses_only_at_the_wire_boundary() -> None:
    assert ViewerType.from_wire_value("fiji") is ViewerType.FIJI
    with pytest.raises(ValueError):
        ViewerType.from_wire_value("FijiViewerDeclaration")
