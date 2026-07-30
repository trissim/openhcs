"""Lightweight declarations for OpenHCS streaming viewer config identities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from openhcs.constants.constants import Backend


class ViewerType(Enum):
    """Supported OpenHCS streaming viewer identities."""

    FIJI = "fiji"
    NAPARI = "napari"


@dataclass(frozen=True, slots=True)
class StreamingViewerPresentation:
    """Viewer-facing presentation identity."""

    title: str


@dataclass(frozen=True, slots=True)
class StreamingViewerConfigSpec:
    """Declarative identity for one OpenHCS viewer streaming config."""

    viewer_type: ViewerType
    registry_key: str
    display_name: str
    step_plan_output_key: str
    presentation: StreamingViewerPresentation
    backend: Backend
    visualizer_module: str


NAPARI_STREAMING_CONFIG_SPEC = StreamingViewerConfigSpec(
    viewer_type=ViewerType.NAPARI,
    registry_key="napari_streaming_config",
    display_name="Napari",
    step_plan_output_key="napari_streaming_paths",
    presentation=StreamingViewerPresentation("OpenHCS Napari Visualization"),
    backend=Backend.NAPARI_STREAM,
    visualizer_module="openhcs.runtime.napari_stream_visualizer",
)

FIJI_STREAMING_CONFIG_SPEC = StreamingViewerConfigSpec(
    viewer_type=ViewerType.FIJI,
    registry_key="fiji_streaming_config",
    display_name="Fiji",
    step_plan_output_key="fiji_streaming_paths",
    presentation=StreamingViewerPresentation("OpenHCS Fiji Visualization"),
    backend=Backend.FIJI_STREAM,
    visualizer_module="openhcs.runtime.fiji_stream_visualizer",
)
