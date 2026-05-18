"""Shared nominal viewer protocol values for streaming visualizers."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
import textwrap
from abc import ABC, abstractmethod
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
        return VIEWER_PROCESS_PLATFORM_BY_SYSTEM_NAME.get(
            platform.system(),
            cls.OTHER,
        )


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


VIEWER_PROCESS_PLATFORM_BY_SYSTEM_NAME: Mapping[str, ViewerProcessPlatform] = {
    ViewerProcessPlatform.DARWIN.value: ViewerProcessPlatform.DARWIN,
    ViewerProcessPlatform.LINUX.value: ViewerProcessPlatform.LINUX,
}


@dataclass(frozen=True, slots=True)
class NapariViewerServerRequest:
    """Shared request fields for constructing a Napari viewer server process."""

    port: int
    viewer_title: str
    replace_layers: bool
    log_file_path: str | None
    transport_mode: object

    @classmethod
    def from_legacy_signature(
        cls,
        port: int,
        viewer_title: str,
        replace_layers: bool = False,
        log_file_path: str | None = None,
        transport_mode: object | None = None,
    ) -> "NapariViewerServerRequest":
        return cls(
            port=port,
            viewer_title=viewer_title,
            replace_layers=replace_layers,
            log_file_path=log_file_path,
            transport_mode=transport_mode,
        )


@dataclass(frozen=True, slots=True)
class NapariViewerProcessEntrypoint:
    """Generate the explicit Python entrypoint for a detached Napari process."""

    request: NapariViewerServerRequest
    python_path_root: Path
    entry_module: str = "openhcs.runtime.napari_viewer_server"
    entry_function: str = "run_napari_viewer_process_from_legacy_signature"

    def python_code(self) -> str:
        transport_name = self.request.transport_mode.name
        return textwrap.dedent(
            f"""
            import os
            import sys

            if hasattr(os, "setsid"):
                try:
                    os.setsid()
                except OSError:
                    pass

            sys.path.insert(0, {str(self.python_path_root)!r})

            try:
                from {self.entry_module} import {self.entry_function}
                from openhcs.core.config import TransportMode

                transport_mode = TransportMode.{transport_name}
                {self.entry_function}(
                    {self.request.port!r},
                    {self.request.viewer_title!r},
                    {self.request.replace_layers!r},
                    {self.request.log_file_path!r},
                    transport_mode,
                )
            except Exception as error:
                import logging
                import traceback

                logger = logging.getLogger("openhcs.runtime.napari_detached")
                logger.error("Detached napari error: %s", error)
                logger.error(traceback.format_exc())
                sys.exit(1)
            """
        ).strip()


@dataclass(frozen=True, slots=True)
class NapariDetachedProcessRequest:
    """Authoritative detached launch request for a Napari viewer process."""

    server_request: NapariViewerServerRequest
    log_file: Path
    cwd: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_legacy_signature(
        cls,
        port: int,
        viewer_title: str,
        replace_layers: bool = False,
        transport_mode: object | None = None,
        *,
        cwd: Path | None = None,
        log_dir: Path | None = None,
    ) -> "NapariDetachedProcessRequest":
        launch_cwd = Path.cwd() if cwd is None else cwd
        launch_log_dir = (
            Path.home() / ".local" / "share" / "openhcs" / "logs"
            if log_dir is None
            else log_dir
        )
        log_file = launch_log_dir / f"napari_detached_port_{port}.log"
        server_request = NapariViewerServerRequest.from_legacy_signature(
            port,
            viewer_title,
            replace_layers,
            str(log_file),
            transport_mode,
        )
        return cls(server_request=server_request, log_file=log_file, cwd=launch_cwd)

    def to_process_request(self) -> "DetachedViewerProcessRequest":
        entrypoint = NapariViewerProcessEntrypoint(
            request=self.server_request,
            python_path_root=self.cwd,
        )
        return DetachedViewerProcessRequest(
            python_code=entrypoint.python_code(),
            log_file=self.log_file,
            cwd=self.cwd,
        )

    def launch(self) -> subprocess.Popen:
        return self.to_process_request().launch()


@dataclass(frozen=True, slots=True)
class ViewerQtPlatformEnvironmentPolicy:
    """Environment mutations for one viewer platform."""

    qpa_platform: QtPlatformName | None = None
    always_set: Mapping[str, str] = field(default_factory=dict)

    def apply_to(self, env: dict[str, str]) -> dict[str, str]:
        if self.qpa_platform is not None and "QT_QPA_PLATFORM" not in env:
            env["QT_QPA_PLATFORM"] = self.qpa_platform.value
        env.update(self.always_set)
        return env


VIEWER_QT_ENVIRONMENT_POLICIES: Mapping[
    ViewerProcessPlatform,
    ViewerQtPlatformEnvironmentPolicy,
] = {
    ViewerProcessPlatform.WINDOWS: ViewerQtPlatformEnvironmentPolicy(),
    ViewerProcessPlatform.DARWIN: ViewerQtPlatformEnvironmentPolicy(
        qpa_platform=QtPlatformName.COCOA,
    ),
    ViewerProcessPlatform.LINUX: ViewerQtPlatformEnvironmentPolicy(
        qpa_platform=QtPlatformName.XCB,
        always_set={"QT_X11_NO_MITSHM": "1"},
    ),
    ViewerProcessPlatform.OTHER: ViewerQtPlatformEnvironmentPolicy(),
}


@dataclass(frozen=True, slots=True)
class ViewerQtEnvironmentPolicy:
    """Apply viewer-safe Qt environment defaults for the current platform."""

    platform: ViewerProcessPlatform = field(
        default_factory=ViewerProcessPlatform.current
    )

    def apply_to(self, env: dict[str, str]) -> dict[str, str]:
        return VIEWER_QT_ENVIRONMENT_POLICIES[self.platform].apply_to(env)


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


class ViewerLifecycleMode(Enum):
    """Runtime ownership state for a managed viewer."""

    STOPPED = "stopped"
    CONNECTED_EXTERNAL = "connected_external"
    OWNED_PROCESS = "owned_process"


@dataclass(slots=True)
class ViewerLifecycleState:
    """Nominal lifecycle state for viewer process managers."""

    mode: ViewerLifecycleMode = ViewerLifecycleMode.STOPPED

    @classmethod
    def stopped(cls) -> "ViewerLifecycleState":
        return cls()

    @property
    def is_active(self) -> bool:
        return self.mode is not ViewerLifecycleMode.STOPPED

    @property
    def is_connected_external(self) -> bool:
        return self.mode is ViewerLifecycleMode.CONNECTED_EXTERNAL

    def mark_connected_external(self) -> None:
        self.mode = ViewerLifecycleMode.CONNECTED_EXTERNAL

    def mark_owned_process(self) -> None:
        self.mode = ViewerLifecycleMode.OWNED_PROCESS

    def mark_stopped(self) -> None:
        self.mode = ViewerLifecycleMode.STOPPED


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


class ManagedViewerLifecycleMixin(ABC):
    """Shared liveness property for viewer process managers."""

    viewer_process_label = "viewer"

    @abstractmethod
    def check_connected_viewer(self) -> bool:
        """Return whether an externally-owned viewer is still responsive."""

    @property
    def is_running(self) -> bool:
        lifecycle_state = self.lifecycle_state
        if not lifecycle_state.is_active:
            return False

        if lifecycle_state.is_connected_external:
            if not self.check_connected_viewer():
                logging.getLogger(self.__class__.__module__).debug(
                    "%s viewer on port %s is no longer responsive",
                    self.viewer_process_label,
                    self.port,
                )
                lifecycle_state.mark_stopped()
                return False
            return True

        if self.process is None:
            lifecycle_state.mark_stopped()
            return False

        try:
            alive = ViewerProcessHandle.from_process(self.process).is_alive()
            if not alive:
                logging.getLogger(self.__class__.__module__).debug(
                    "%s process on port %s is no longer alive",
                    self.viewer_process_label,
                    self.port,
                )
                lifecycle_state.mark_stopped()
            return alive
        except Exception as error:
            logging.getLogger(self.__class__.__module__).warning(
                "Error checking %s process status: %s",
                self.viewer_process_label,
                error,
            )
            lifecycle_state.mark_stopped()
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
