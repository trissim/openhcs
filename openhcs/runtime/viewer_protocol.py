"""Shared nominal viewer protocol values for streaming visualizers."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.process import BaseProcess
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
class ViewerProcessHandle:
    """Nominal adapter over multiprocessing and subprocess viewer handles."""

    process: BaseProcess | subprocess.Popen

    @classmethod
    def from_process(cls, process: object) -> "ViewerProcessHandle":
        if isinstance(process, (BaseProcess, subprocess.Popen)):
            return cls(process)
        raise TypeError(f"Unsupported viewer process handle: {type(process)!r}")

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def pid_label(self) -> str:
        return str(self.pid) if self.pid is not None else "unknown"

    def is_alive(self) -> bool:
        if isinstance(self.process, BaseProcess):
            return self.process.is_alive()
        return self.process.poll() is None

    def terminate(self, *, timeout: float = 5.0, kill_timeout: float = 2.0) -> bool:
        if not self.is_alive():
            return False
        self.process.terminate()
        if isinstance(self.process, BaseProcess):
            self.process.join(timeout=timeout)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=kill_timeout)
                return True
            return False
        try:
            self.process.wait(timeout=timeout)
            return False
        except subprocess.TimeoutExpired:
            self.process.kill()
            return True


class ViewerControlPingMode(Enum):
    """Viewer control-port ping policy modes."""

    QUICK = "quick"
    EXISTING_VIEWER = "existing_viewer"


@dataclass(frozen=True, slots=True)
class ViewerControlPingPolicy:
    """Timeout/readiness coordinates for one control ping mode."""

    timeout_ms: int
    require_ready: bool


VIEWER_CONTROL_PING_POLICIES: Mapping[
    ViewerControlPingMode,
    ViewerControlPingPolicy,
] = {
    ViewerControlPingMode.QUICK: ViewerControlPingPolicy(
        timeout_ms=200,
        require_ready=False,
    ),
    ViewerControlPingMode.EXISTING_VIEWER: ViewerControlPingPolicy(
        timeout_ms=500,
        require_ready=True,
    ),
}


@dataclass(frozen=True, slots=True)
class ViewerControlPingRequest:
    """Typed control-port ping request for viewer readiness checks."""

    port: int
    transport_mode: object
    config: object
    host: str = "localhost"
    timeout_ms: int = 500
    require_ready: bool = True

    @classmethod
    def from_mode(
        cls,
        *,
        mode: ViewerControlPingMode,
        port: int,
        transport_mode: object,
        config: object,
    ) -> "ViewerControlPingRequest":
        policy = VIEWER_CONTROL_PING_POLICIES[mode]
        return cls(
            port=port,
            transport_mode=transport_mode,
            config=config,
            timeout_ms=policy.timeout_ms,
            require_ready=policy.require_ready,
        )

    def check(self) -> bool:
        from zmqruntime.transport import ping_control_port

        return ping_control_port(
            self.port,
            self.transport_mode,
            host=self.host,
            config=self.config,
            timeout_ms=self.timeout_ms,
            require_ready=self.require_ready,
        )


class ManagedViewerLifecycleMixin:
    """Shared liveness property for viewer process managers."""

    viewer_process_label = "viewer"

    @property
    def is_running(self) -> bool:
        if not self._is_running:
            return False

        if self._connected_to_existing:
            if not self._quick_ping_check():
                logging.getLogger(self.__class__.__module__).debug(
                    "%s viewer on port %s is no longer responsive",
                    self.viewer_process_label,
                    self.port,
                )
                self._is_running = False
                self._connected_to_existing = False
                return False
            return True

        if self.process is None:
            self._is_running = False
            return False

        try:
            alive = ViewerProcessHandle.from_process(self.process).is_alive()
            if not alive:
                logging.getLogger(self.__class__.__module__).debug(
                    "%s process on port %s is no longer alive",
                    self.viewer_process_label,
                    self.port,
                )
                self._is_running = False
            return alive
        except Exception as error:
            logging.getLogger(self.__class__.__module__).warning(
                "Error checking %s process status: %s",
                self.viewer_process_label,
                error,
            )
            self._is_running = False
            return False


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
