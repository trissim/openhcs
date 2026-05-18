"""Shared nominal viewer protocol values for streaming visualizers."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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


class QtPlatformName(Enum):
    """Qt platform plugin names used by detached viewer processes."""

    COCOA = "cocoa"
    XCB = "xcb"


class ViewerProcessPlatform(Enum):
    """Host platform family for detached viewer launch behavior."""

    WINDOWS = "win32"
    DARWIN = "Darwin"
    LINUX = "Linux"
    OTHER = "other"

    @classmethod
    def current(cls) -> "ViewerProcessPlatform":
        if sys.platform == cls.WINDOWS.value:
            return cls.WINDOWS
        system_name = platform.system()
        if system_name == cls.DARWIN.value:
            return cls.DARWIN
        if system_name == cls.LINUX.value:
            return cls.LINUX
        return cls.OTHER


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
class NapariViewerServerRequest:
    """Shared request fields for constructing a Napari viewer server process."""

    port: int
    viewer_title: str
    replace_layers: bool
    log_file_path: str | None
    transport_mode: object


@dataclass(frozen=True, slots=True)
class ViewerQtEnvironmentPolicy:
    """Apply viewer-safe Qt environment defaults for the current platform."""

    platform: ViewerProcessPlatform = field(
        default_factory=ViewerProcessPlatform.current
    )

    def apply_to(self, env: dict[str, str]) -> dict[str, str]:
        if "QT_QPA_PLATFORM" not in env:
            if self.platform is ViewerProcessPlatform.DARWIN:
                env["QT_QPA_PLATFORM"] = QtPlatformName.COCOA.value
            elif self.platform is ViewerProcessPlatform.LINUX:
                env["QT_QPA_PLATFORM"] = QtPlatformName.XCB.value
                env["QT_X11_NO_MITSHM"] = "1"
        elif self.platform is ViewerProcessPlatform.LINUX:
            env["QT_X11_NO_MITSHM"] = "1"
        return env


@dataclass(frozen=True, slots=True)
class DetachedViewerProcessRequest:
    """Launch request for a detached Python viewer process."""

    python_code: str
    log_file: Path
    cwd: Path = field(default_factory=lambda: Path.cwd())
    env: Mapping[str, str] | None = None
    platform: ViewerProcessPlatform = field(
        default_factory=ViewerProcessPlatform.current
    )

    def launch(self) -> subprocess.Popen:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        launch_env = dict(os.environ if self.env is None else self.env)
        ViewerQtEnvironmentPolicy(self.platform).apply_to(launch_env)
        log_handle = self.log_file.open("w")
        command = [sys.executable, "-c", self.python_code]
        if self.platform is ViewerProcessPlatform.WINDOWS:
            return subprocess.Popen(
                command,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS,
                env=launch_env,
                cwd=str(self.cwd),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        return subprocess.Popen(
            command,
            env=launch_env,
            cwd=str(self.cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


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


@dataclass(frozen=True, slots=True)
class ComponentDimensionLabelPolicy:
    """Build human-readable dimension labels for viewer component axes."""

    abbreviations: Mapping[str, str] = field(
        default_factory=lambda: {
            "channel": "Ch",
            "z_index": "Z",
            "timepoint": "T",
            "site": "Site",
            "well": "Well",
        }
    )
    metadata_formatters: Mapping[str, Callable[[object, object], str]] = field(
        default_factory=lambda: {
            "channel": lambda value, name: f"Ch{value}: {name}",
            "well": lambda _value, name: str(name),
        }
    )

    def labels_for(
        self,
        *,
        component: str,
        values: Iterable[object],
        metadata: Mapping[str, object],
    ) -> list[str]:
        return [
            self.label_for(component=component, value=value, metadata=metadata)
            for value in values
        ]

    def label_for(
        self,
        *,
        component: str,
        value: object,
        metadata: Mapping[str, object],
    ) -> str:
        metadata_name = metadata.get(str(value))
        if metadata_name and str(metadata_name).lower() != "none":
            return self._metadata_label(component, value, metadata_name)
        return f"{self.abbreviations.get(component, component)} {value}"

    def _metadata_label(
        self,
        component: str,
        value: object,
        metadata_name: object,
    ) -> str:
        formatter = self.metadata_formatters.get(component)
        if formatter is not None:
            return formatter(value, metadata_name)
        return f"{component.title()} {value}: {metadata_name}"
