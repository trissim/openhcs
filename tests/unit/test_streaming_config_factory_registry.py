"""Registry-ownership tests for managed streaming viewer resolution."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from openhcs.core.streaming_config_factory import ManagedViewerTypeResolver
from openhcs.core.streaming_config_declarations import (
    FIJI_STREAMING_CONFIG_SPEC,
    NAPARI_STREAMING_CONFIG_SPEC,
    ViewerType,
)
from openhcs.runtime.viewer_protocol import ManagedViewerLifecycleMixin


def _spec(viewer_type: ViewerType) -> SimpleNamespace:
    return SimpleNamespace(
        viewer_type=viewer_type,
        visualizer_module="tests.synthetic_viewer",
    )


def test_managed_viewer_resolution_uses_the_owner_registry() -> None:
    class RegisteredViewer:
        viewer_type = "napari"

    class ViewerOwner:
        __registry__ = {"napari": RegisteredViewer}

    class UnregisteredShadow(ViewerOwner):
        viewer_type = "napari"

    assert (
        ManagedViewerTypeResolver.resolve(ViewerOwner, _spec(ViewerType.NAPARI))
        is RegisteredViewer
    )
    assert UnregisteredShadow not in ViewerOwner.__registry__.values()


def test_managed_viewer_resolution_rejects_an_unregistered_subclass() -> None:
    class ViewerOwner:
        __registry__: dict[str, type] = {}

    class UnregisteredViewer(ViewerOwner):
        viewer_type = "fiji"

    with pytest.raises(KeyError, match="viewer_type='fiji'"):
        ManagedViewerTypeResolver.resolve(ViewerOwner, _spec(ViewerType.FIJI))

    assert UnregisteredViewer not in ViewerOwner.__registry__.values()


@pytest.mark.parametrize(
    "spec",
    (NAPARI_STREAMING_CONFIG_SPEC, FIJI_STREAMING_CONFIG_SPEC),
    ids=lambda spec: spec.viewer_type.value,
)
def test_declared_viewer_modules_register_their_runtime_owner(spec) -> None:
    import_module(spec.visualizer_module)

    visualizer_type = ManagedViewerTypeResolver.resolve(
        ManagedViewerLifecycleMixin,
        spec,
    )

    assert (
        visualizer_type
        is ManagedViewerLifecycleMixin.__registry__[spec.viewer_type.value]
    )
    assert visualizer_type.viewer_type == spec.viewer_type.value
