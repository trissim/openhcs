"""Shared nominal viewer protocol values for streaming visualizers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from polystore.streaming_constants import StreamingDataType


class ViewerType(Enum):
    """Supported OpenHCS streaming viewer identities."""

    FIJI = "fiji"
    NAPARI = "napari"


class ViewerProtocolStatus(Enum):
    """Control/ack status values shared by viewer servers."""

    SUCCESS = "success"
    ERROR = "error"


class NapariLayerKind(Enum):
    """Napari layer creation families used by streaming display."""

    IMAGE = "image"
    SHAPES = "shapes"
    POINTS = "points"


class FijiPayloadKind(Enum):
    """Payload strings sent to the Fiji viewer process."""

    IMAGE = "image"
    ROIS = "rois"

    @property
    def streaming_data_type(self) -> StreamingDataType:
        if self is FijiPayloadKind.IMAGE:
            return StreamingDataType.IMAGE
        return StreamingDataType.ROIS

    @classmethod
    def from_payload(cls, payload: object) -> "FijiPayloadKind | None":
        try:
            return cls(str(payload))
        except ValueError:
            return None


@dataclass(frozen=True, slots=True)
class ViewerHeartbeatDescriptor:
    """Viewer-specific fields added to a streaming server pong response."""

    viewer_type: ViewerType
    server_name: str

    def apply_to(self, response: dict[str, Any]) -> dict[str, Any]:
        response["viewer"] = self.viewer_type.value
        response["openhcs"] = True
        response["server"] = self.server_name
        self._add_process_metrics(response)
        return response

    @staticmethod
    def _add_process_metrics(response: dict[str, Any]) -> None:
        try:
            import psutil

            process = psutil.Process(os.getpid())
            response["memory_mb"] = process.memory_info().rss / 1024 / 1024
            response["cpu_percent"] = process.cpu_percent(interval=0)
        except Exception:
            pass


NAPARI_HEARTBEAT = ViewerHeartbeatDescriptor(ViewerType.NAPARI, "NapariViewer")
FIJI_HEARTBEAT = ViewerHeartbeatDescriptor(ViewerType.FIJI, "FijiViewerServer")


@dataclass(frozen=True, slots=True)
class ChannelColormapPolicy:
    """Resolve channel-slice colors from component metadata."""

    colors_by_channel: Mapping[int, str] = field(
        default_factory=lambda: {1: "green", 2: "red"}
    )

    def colormap(self, channel_value: object) -> str | None:
        try:
            channel_number = int(channel_value)
        except (TypeError, ValueError):
            return None
        return self.colors_by_channel.get(channel_number)
