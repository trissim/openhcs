"""Shared nominal viewer protocol values for streaming visualizers."""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import ClassVar, TypeAlias, cast

from openhcs.core.config import TransportMode as ViewerTransportMode
from openhcs.core.streaming_config_factory import (
    StreamingViewerRuntimeConfig,
)
from metaclass_registry import AutoRegisterMeta
from polystore.streaming_constants import StreamingDataType
from zmqruntime.config import TransportMode as ZMQTransportMode, ZMQConfig
from zmqruntime.streaming import VisualizerProcessManager
from zmqruntime.viewer_protocol import (
    ViewerBatchMessageType,
    ViewerBatchContextWireField,
    ViewerBatchWireField,
    ViewerControlResponseField,
    ViewerControlReplyHeader,
    ViewerControlReplyPayload,
    ViewerProtocolStatus,
    ViewerTransportEndpoint,
)


ViewerScalar: TypeAlias = str | int | float | bool | None
ViewerComponentValue: TypeAlias = ViewerScalar | tuple[ViewerScalar, ...]
NaturalTokenKey: TypeAlias = tuple[int, int | str]
NaturalTextKey: TypeAlias = tuple[NaturalTokenKey, ...]
ComponentValueSortKey: TypeAlias = tuple[int, int | float | NaturalTextKey, str, str]
ComponentTupleSortKey: TypeAlias = tuple[ComponentValueSortKey, ...]
ViewerHeartbeatValue: TypeAlias = str | bool | int | float | None
ViewerProcess: TypeAlias = BaseProcess | subprocess.Popen[bytes]
ViewerLaunchLiteral: TypeAlias = str | int | float | bool | None
ViewerControlWireValue: TypeAlias = (
    ViewerScalar
    | tuple["ViewerControlWireValue", ...]
    | list["ViewerControlWireValue"]
    | dict[str, "ViewerControlWireValue"]
)


class ViewerType(Enum):
    """Supported OpenHCS streaming viewer identities."""

    FIJI = "fiji"
    NAPARI = "napari"


class ViewerControlMessageType(Enum):
    """Shared control-message names consumed by viewer servers."""

    SCREENSHOT = "screenshot"
    SETTLE = "settle"
    STATE = "state"


@dataclass(frozen=True, slots=True)
class ViewerTypeIdentity:
    """Inherited viewer identity for runtime protocol records."""

    viewer_type: ViewerType


class ViewerPersistenceMode(Enum):
    """Viewer lifecycle ownership mode derived from streaming persistence."""

    PERSISTENT = "persistent"
    NON_PERSISTENT = "non-persistent"

    @classmethod
    def from_flag(cls, persistent: bool) -> "ViewerPersistenceMode":
        return VIEWER_PERSISTENCE_MODE_BY_FLAG[persistent]


VIEWER_PERSISTENCE_MODE_BY_FLAG: Mapping[bool, ViewerPersistenceMode] = {
    True: ViewerPersistenceMode.PERSISTENT,
    False: ViewerPersistenceMode.NON_PERSISTENT,
}


@dataclass(frozen=True, slots=True)
class ViewerControlResponse:
    """Typed view of a viewer control-message response."""

    payload: Mapping[str, ViewerControlWireValue]

    @property
    def status(self) -> ViewerProtocolStatus:
        status_value = self.payload.get(ViewerControlResponseField.STATUS.value)
        if status_value is None:
            raise ValueError("Viewer control response is missing a status field.")
        return ViewerProtocolStatus(str(status_value))

    def succeeded(self) -> bool:
        return self.status is ViewerProtocolStatus.SUCCESS


class ViewerComponentValueOrdering:
    """Canonical ordering for viewer component values and stack coordinates."""

    NATURAL_TOKEN_PATTERN = re.compile(r"(\d+)")
    INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
    FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")

    @classmethod
    def key(cls, value: ViewerComponentValue) -> ComponentValueSortKey:
        numeric_value = cls.numeric_value(value)
        if numeric_value is not None:
            return (0, numeric_value, type(value).__name__, str(value))

        text = str(value)
        return (1, cls.natural_text_key(text), type(value).__name__, text)

    @classmethod
    def tuple_key(cls, values: tuple[ViewerComponentValue, ...]) -> ComponentTupleSortKey:
        return tuple(cls.key(value) for value in values)

    @classmethod
    def numeric_value(cls, value: ViewerComponentValue) -> int | float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value
        if not isinstance(value, str):
            return None

        text = value.strip()
        if not text:
            return None
        if cls.INTEGER_PATTERN.fullmatch(text):
            return int(text)
        if cls.FLOAT_PATTERN.fullmatch(text):
            return float(text)
        return None

    @classmethod
    def natural_text_key(cls, text: str) -> NaturalTextKey:
        return tuple(
            (0, int(token)) if token.isdecimal() else (1, token.casefold())
            for token in cls.NATURAL_TOKEN_PATTERN.split(text)
            if token
        )


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
        if system_name in VIEWER_PROCESS_PLATFORM_BY_SYSTEM_NAME:
            return VIEWER_PROCESS_PLATFORM_BY_SYSTEM_NAME[system_name]
        return cls.OTHER


class NapariLayerKind(Enum):
    """Napari layer creation families used by streaming display."""

    IMAGE = "image"
    SHAPES = "shapes"
    POINTS = "points"
    LABELS = "labels"


class FijiPayloadKind(Enum):
    """Payload strings sent to the Fiji viewer process."""

    IMAGE = ("image", StreamingDataType.IMAGE, True)
    ROIS = ("rois", StreamingDataType.ROIS, False)

    def __init__(
        self,
        wire_value: str,
        streaming_data_type: StreamingDataType,
        uses_shared_memory: bool,
    ) -> None:
        self.wire_value = wire_value
        self.streaming_data_type = streaming_data_type
        self.uses_shared_memory = uses_shared_memory

    @classmethod
    def from_payload(cls, payload: str | None) -> "FijiPayloadKind | None":
        if payload is None:
            return None
        wire_value = str(payload)
        if wire_value in FIJI_PAYLOAD_KIND_BY_WIRE_VALUE:
            return FIJI_PAYLOAD_KIND_BY_WIRE_VALUE[wire_value]
        return None


FIJI_PAYLOAD_KIND_BY_WIRE_VALUE: Mapping[str, FijiPayloadKind] = {
    kind.wire_value: kind for kind in FijiPayloadKind
}


class ViewerHeartbeatField(Enum):
    """Fields owned by OpenHCS viewer heartbeat payloads."""

    VIEWER = "viewer"
    OPENHCS = "openhcs"
    SERVER = "server"
    MEMORY_MB = "memory_mb"
    CPU_PERCENT = "cpu_percent"


@dataclass(slots=True)
class ViewerHeartbeatPayload:
    """Nominal heartbeat payload builder around the ZMQ server pong mapping."""

    values: dict[str, ViewerHeartbeatValue] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        response: Mapping[str, ViewerHeartbeatValue],
    ) -> "ViewerHeartbeatPayload":
        return cls(dict(response))

    def set_field(
        self,
        field_name: ViewerHeartbeatField,
        value: ViewerHeartbeatValue,
    ) -> None:
        self.values[field_name.value] = value

    def mark_viewer(self, viewer_type: ViewerType, server_name: str) -> None:
        self.set_field(ViewerHeartbeatField.VIEWER, viewer_type.value)
        self.set_field(ViewerHeartbeatField.OPENHCS, True)
        self.set_field(ViewerHeartbeatField.SERVER, server_name)

    def add_process_metrics(self) -> None:
        import psutil

        process = psutil.Process(os.getpid())
        self.set_field(
            ViewerHeartbeatField.MEMORY_MB,
            process.memory_info().rss / 1024 / 1024,
        )
        self.set_field(
            ViewerHeartbeatField.CPU_PERCENT,
            process.cpu_percent(interval=0),
        )

    def to_dict(self) -> dict[str, ViewerHeartbeatValue]:
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class ViewerHeartbeatDescriptor(ViewerTypeIdentity):
    """Viewer-specific fields added to a streaming server pong response."""

    server_name: str

    def apply_to(
        self,
        response: Mapping[str, ViewerHeartbeatValue],
    ) -> dict[str, ViewerHeartbeatValue]:
        heartbeat = ViewerHeartbeatPayload.from_mapping(response)
        heartbeat.mark_viewer(self.viewer_type, self.server_name)
        try:
            heartbeat.add_process_metrics()
        except Exception:
            pass
        return heartbeat.to_dict()


NAPARI_HEARTBEAT = ViewerHeartbeatDescriptor(ViewerType.NAPARI, "NapariViewer")
FIJI_HEARTBEAT = ViewerHeartbeatDescriptor(ViewerType.FIJI, "FijiViewerServer")


def viewer_lifecycle_registry_key(
    name: str,
    cls: type,
) -> str:
    """Derive the lifecycle registry key from the declared detached entrypoint."""
    del name
    if "detached_server_entrypoint" not in cls.__dict__:
        raise TypeError(
            f"{cls.__name__} must declare detached_server_entrypoint to register "
            "as a managed viewer lifecycle."
        )
    entrypoint = cls.__dict__["detached_server_entrypoint"]
    return entrypoint.viewer_type.value


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerServerLaunchRequest:
    """Shared launch request fields for OpenHCS viewer server processes."""

    port: int
    log_file_path: str | None = None
    transport_mode: ViewerTransportMode = ViewerTransportMode.IPC


@dataclass(frozen=True, slots=True)
class NapariViewerServerRequest(ViewerServerLaunchRequest):
    """Nominal launch request consumed by the Napari viewer server."""

    viewer_title: str
    replace_layers: bool = False


VIEWER_PROCESS_PLATFORM_BY_SYSTEM_NAME: Mapping[str, ViewerProcessPlatform] = {
    ViewerProcessPlatform.DARWIN.value: ViewerProcessPlatform.DARWIN,
    ViewerProcessPlatform.LINUX.value: ViewerProcessPlatform.LINUX,
}


@dataclass(frozen=True, slots=True)
class ViewerRuntimeEndpoint:
    """OpenHCS viewer endpoint projected onto zmqruntime primitives."""

    transport: ViewerTransportEndpoint
    config: ZMQConfig

    @property
    def port(self) -> int:
        return self.transport.port

    @property
    def host(self) -> str:
        return self.transport.host

    @property
    def mode(self) -> ViewerTransportMode:
        return self.transport.transport_mode

    @property
    def zmq_transport_mode(self) -> ZMQTransportMode:
        from zmqruntime.transport import coerce_transport_mode

        zmq_mode = coerce_transport_mode(self.mode)
        if zmq_mode is None:
            raise ValueError(f"Unsupported viewer transport mode: {self.mode!r}")
        return zmq_mode

    @property
    def control_port(self) -> int:
        from zmqruntime.transport import get_control_port

        return get_control_port(self.port, self.config)

    def data_url(self) -> str:
        from zmqruntime.transport import get_zmq_transport_url

        return get_zmq_transport_url(
            self.port,
            host=self.host,
            mode=self.zmq_transport_mode,
            config=self.config,
        )

    def control_url(self) -> str:
        from zmqruntime.transport import get_control_url

        return get_control_url(
            self.port,
            self.zmq_transport_mode,
            host=self.host,
            config=self.config,
        )

    def in_use(self) -> bool:
        from zmqruntime.transport import is_port_in_use

        return is_port_in_use(
            self.port,
            self.zmq_transport_mode,
            host=self.host,
            config=self.config,
        )

    def wait_until_released(
        self,
        *,
        timeout: float,
        poll_interval: float = 0.1,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while self.in_use() and time.monotonic() < deadline:
            time.sleep(poll_interval)
        return not self.in_use()

    def ping(
        self,
        *,
        timeout_ms: int,
        require_ready: bool,
    ) -> bool:
        from zmqruntime.transport import ping_control_port

        return ping_control_port(
            self.port,
            self.zmq_transport_mode,
            host=self.host,
            config=self.config,
            timeout_ms=timeout_ms,
            require_ready=require_ready,
        )

    def wait_ready(self, *, timeout: float, require_ready: bool = True) -> bool:
        from zmqruntime.transport import wait_for_server_ready

        return wait_for_server_ready(
            self.port,
            self.zmq_transport_mode,
            host=self.host,
            config=self.config,
            timeout=timeout,
            require_ready=require_ready,
        )

    def release_bound_ports(self) -> None:
        if self.zmq_transport_mode is ZMQTransportMode.IPC:
            from zmqruntime.transport import remove_ipc_socket

            remove_ipc_socket(self.port, self.config)
            remove_ipc_socket(self.control_port, self.config)
            return

        from zmqruntime.server import ZMQServer

        ZMQServer.kill_processes_on_port(self.port)
        ZMQServer.kill_processes_on_port(self.control_port)


@dataclass(frozen=True, slots=True)
class DetachedViewerPythonExpression:
    """One expression allowed in generated detached-viewer Python."""

    source: str

    @classmethod
    def literal(cls, value: ViewerLaunchLiteral) -> "DetachedViewerPythonExpression":
        return cls(repr(value))

    @classmethod
    def symbol(cls, name: str) -> "DetachedViewerPythonExpression":
        if not name.isidentifier():
            raise ValueError(f"Detached viewer symbol is not a valid identifier: {name!r}")
        return cls(name)


@dataclass(frozen=True, slots=True)
class DetachedViewerPythonArguments:
    """Nominal argument list for detached-viewer entrypoint code generation."""

    expressions: tuple[DetachedViewerPythonExpression, ...] = ()

    @classmethod
    def from_literals(
        cls,
        *values: ViewerLaunchLiteral,
    ) -> "DetachedViewerPythonArguments":
        return cls(tuple(DetachedViewerPythonExpression.literal(value) for value in values))

    def append(
        self,
        *expressions: DetachedViewerPythonExpression,
    ) -> "DetachedViewerPythonArguments":
        return type(self)((*self.expressions, *expressions))

    def render(self) -> str:
        return ",\n".join(expression.source for expression in self.expressions)


@dataclass(frozen=True, slots=True)
class DetachedViewerLaunchRequest(ViewerTypeIdentity):
    """Authoritative detached launch request for a viewer process."""

    port: int
    python_code: str
    log_file: Path
    cwd: Path = field(default_factory=Path.cwd)
    env: Mapping[str, str] | None = None
    platform: ViewerProcessPlatform = field(
        default_factory=ViewerProcessPlatform.current
    )

    @classmethod
    def log_file_for(
        cls,
        *,
        viewer_type: ViewerType,
        port: int,
        log_dir: Path | None = None,
    ) -> Path:
        launch_log_dir = (
            Path.home() / ".local" / "share" / "openhcs" / "logs"
            if log_dir is None
            else log_dir
        )
        return launch_log_dir / f"{viewer_type.value}_detached_port_{port}.log"

    def command(self) -> list[str]:
        return [sys.executable, "-c", self.python_code]

    def launch(self) -> subprocess.Popen[bytes]:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        launch_env = dict(os.environ if self.env is None else self.env)
        ViewerQtEnvironmentPolicy(self.platform).apply_to(launch_env)
        log_handle = self.log_file.open("w")
        if self.platform is ViewerProcessPlatform.WINDOWS:
            return subprocess.Popen(
                self.command(),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS,
                env=launch_env,
                cwd=str(self.cwd),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        return subprocess.Popen(
            self.command(),
            env=launch_env,
            cwd=str(self.cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


@dataclass(frozen=True, slots=True)
class DetachedViewerServerEntrypointSpec(ViewerTypeIdentity):
    """Declared server function used to launch one detached viewer family."""

    module_name: str
    function_name: str
    extra_imports: tuple[str, ...] = ()

    def log_file_for(self, port: int) -> Path:
        return DetachedViewerLaunchRequest.log_file_for(
            viewer_type=self.viewer_type,
            port=port,
        )

    def python_code(
        self,
        python_path_root: Path,
        *,
        transport_mode: ViewerTransportMode,
        arguments: DetachedViewerPythonArguments,
    ) -> str:
        transport_name = transport_mode.name
        rendered_arguments = arguments.render()
        call_arguments = "\n".join(
            f"    {line}" for line in rendered_arguments.splitlines()
        )
        lines = [
            "import os",
            "import sys",
            "",
            'if os.name == "posix":',
            "    try:",
            "        os.setsid()",
            "    except OSError:",
            "        pass",
            "",
            f"sys.path.insert(0, {str(python_path_root)!r})",
            "",
            "try:",
            f"    from {self.module_name} import {self.function_name}",
            "    from openhcs.core.config import TransportMode",
        ]
        lines.extend(f"    {extra_import}" for extra_import in self.extra_imports)
        lines.extend(
            [
                "",
                f"    transport_mode = TransportMode.{transport_name}",
                f"    {self.function_name}(",
                call_arguments,
                "    )",
                "except Exception as error:",
                "    import logging",
                "    import traceback",
                "",
                '    logger = logging.getLogger("openhcs.runtime.detached_viewer")',
                '    logger.error("Detached viewer error: %s", error)',
                "    logger.error(traceback.format_exc())",
                "    sys.exit(1)",
            ]
        )
        return "\n".join(lines)

    def launch_request(
        self,
        *,
        port: int,
        transport_mode: ViewerTransportMode,
        arguments: DetachedViewerPythonArguments,
        log_file: Path,
        cwd: Path | None = None,
    ) -> DetachedViewerLaunchRequest:
        if cwd is None:
            cwd = Path.cwd()
        return DetachedViewerLaunchRequest(
            viewer_type=self.viewer_type,
            port=port,
            python_code=self.python_code(
                cwd,
                transport_mode=transport_mode,
                arguments=arguments,
            ),
            log_file=log_file,
            cwd=cwd,
        )


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
class ViewerProcessHandle:
    """Nominal adapter over multiprocessing and subprocess viewer handles."""

    process: ViewerProcess

    @classmethod
    def from_process(cls, process: ViewerProcess) -> "ViewerProcessHandle":
        if isinstance(process, (BaseProcess, subprocess.Popen)):
            return cls(process)
        raise TypeError(f"Unsupported viewer process handle: {type(process)!r}")

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def pid_label(self) -> str:
        if self.pid is None:
            return "unknown"
        return str(self.pid)

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
            self.process.wait(timeout=kill_timeout)
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

    endpoint: ViewerRuntimeEndpoint
    timeout_ms: int = 500
    require_ready: bool = True

    @classmethod
    def from_mode(
        cls,
        *,
        mode: ViewerControlPingMode,
        endpoint: ViewerRuntimeEndpoint,
    ) -> "ViewerControlPingRequest":
        policy = VIEWER_CONTROL_PING_POLICIES[mode]
        return cls(
            endpoint=endpoint,
            timeout_ms=policy.timeout_ms,
            require_ready=policy.require_ready,
        )

@dataclass(frozen=True, slots=True)
class ViewerControlMessageRequest:
    """Typed REQ/REP control-message request shared by viewer visualizers."""

    endpoint: ViewerRuntimeEndpoint
    message_type: str
    timeout: float = 2.0

    def send(self) -> ViewerControlResponse:
        import pickle

        import zmq

        context = None
        socket = None
        try:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
            socket.connect(self.endpoint.control_url())
            socket.send(
                pickle.dumps(
                    {ViewerControlResponseField.TYPE.value: self.message_type}
                )
            )
            payload = pickle.loads(socket.recv())
            if not isinstance(payload, Mapping):
                raise TypeError(
                    "Viewer control response must be a mapping, "
                    f"got {type(payload).__name__}."
                )
            return ViewerControlResponse(
                cast(Mapping[str, ViewerControlWireValue], payload)
            )
        finally:
            if socket is not None:
                socket.close()
            if context is not None:
                context.term()


class ManagedViewerLifecycleMixin(
    VisualizerProcessManager,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Shared liveness property for viewer process managers."""

    __registry_key__ = "viewer_type"
    __key_extractor__ = staticmethod(viewer_lifecycle_registry_key)
    __skip_if_no_key__ = True

    __registry__: ClassVar[dict[str, type["ManagedViewerLifecycleMixin"]]]
    viewer_type: ClassVar[str | None] = None
    viewer_process_label: ClassVar[str] = "viewer"
    detached_server_entrypoint: ClassVar[DetachedViewerServerEntrypointSpec]

    def __init__(
        self,
        *,
        runtime_config: StreamingViewerRuntimeConfig,
        transport_config: ZMQConfig,
    ) -> None:
        super().__init__(port=runtime_config.transport_endpoint.port)
        self.persistent: bool = runtime_config.persistent
        self.lifecycle_presentation = runtime_config.presentation
        self.runtime_endpoint = ViewerRuntimeEndpoint(
            transport=runtime_config.transport_endpoint,
            config=transport_config,
        )
        self.lifecycle_state: ViewerLifecycleState = ViewerLifecycleState.stopped()

    @property
    def required_port(self) -> int:
        port = self.port
        if port is None:
            raise RuntimeError("OpenHCS streaming viewers require a configured port.")
        return port

    @property
    def persistence_mode(self) -> ViewerPersistenceMode:
        return ViewerPersistenceMode.from_flag(self.persistent)

    @property
    def persistence_label(self) -> str:
        return self.persistence_mode.value

    @property
    def viewer_title(self) -> str:
        return self.lifecycle_presentation.title

    @abstractmethod
    def start_viewer(self, async_mode: bool = False) -> None:
        """Start the concrete viewer server process."""

    @abstractmethod
    def detached_server_arguments(
        self,
        *,
        log_file: Path,
    ) -> DetachedViewerPythonArguments:
        """Return entrypoint arguments for this concrete viewer server."""

    def check_connected_viewer(self) -> bool:
        """Return whether an externally-owned viewer is still responsive."""
        request = ViewerControlPingRequest.from_mode(
            mode=ViewerControlPingMode.QUICK,
            endpoint=self.runtime_endpoint,
        )
        return request.endpoint.ping(
            timeout_ms=request.timeout_ms,
            require_ready=request.require_ready,
        )

    def request_bound_viewer_shutdown(self, timeout: float = 1.0) -> bool:
        """Ask the viewer currently bound to this endpoint to terminate."""
        response = ViewerControlMessageRequest(
            endpoint=self.runtime_endpoint,
            message_type="force_shutdown",
            timeout=timeout,
        ).send()
        return response.succeeded()

    def prepare_fresh_viewer_start(self) -> None:
        """Ensure this viewer endpoint is not backed by a previous run."""
        if not self.runtime_endpoint.in_use():
            return

        if self.check_connected_viewer():
            if not self.request_bound_viewer_shutdown():
                raise RuntimeError(
                    f"{self.viewer_process_label} viewer on port {self.required_port} "
                    "did not acknowledge shutdown before a fresh start."
                )
            if self.runtime_endpoint.wait_until_released(timeout=3.0):
                return

        self.runtime_endpoint.release_bound_ports()
        if not self.runtime_endpoint.wait_until_released(timeout=2.0):
            raise RuntimeError(
                f"{self.viewer_process_label} viewer on port {self.required_port} "
                "remained bound after forced endpoint release."
            )

    def existing_viewer_is_ready(self) -> bool:
        request = ViewerControlPingRequest.from_mode(
            mode=ViewerControlPingMode.EXISTING_VIEWER,
            endpoint=self.runtime_endpoint,
        )
        return request.endpoint.ping(
            timeout_ms=request.timeout_ms,
            require_ready=request.require_ready,
        )

    def wait_for_ready(self, timeout: float = 10.0) -> bool:
        """Satisfy zmqruntime's process-manager readiness contract."""
        return self.runtime_endpoint.wait_ready(
            timeout=timeout,
            require_ready=True,
        )

    def detached_launch_request(self) -> DetachedViewerLaunchRequest:
        port = self.required_port
        log_file = self.detached_server_entrypoint.log_file_for(port)
        return self.detached_server_entrypoint.launch_request(
            port=port,
            transport_mode=self.runtime_endpoint.mode,
            arguments=self.detached_server_arguments(log_file=log_file),
            log_file=log_file,
        )

    def launch_detached_viewer(self) -> subprocess.Popen[bytes]:
        launch_request = self.detached_launch_request()
        process = launch_request.launch()
        logging.getLogger(type(self).__module__).info(
            "%s detached process started (PID: %s), logging to %s",
            self.viewer_process_label,
            process.pid,
            launch_request.log_file,
        )
        return process

    def get_launch_command(self) -> list[str]:
        return self.detached_launch_request().command()

    def get_launch_env(self) -> dict[str, str]:
        return ViewerQtEnvironmentPolicy().apply_to(dict(os.environ))

    def cleanup_viewer_client(self) -> None:
        """Release client-side resources before forced viewer termination."""

    def force_stop(self, timeout: float = 5.0) -> None:
        """Terminate the viewer process regardless of persistence policy."""
        with self._lock:
            self.cleanup_viewer_client()
            if self.process is not None:
                killed = ViewerProcessHandle.from_process(self.process).terminate(
                    timeout=timeout,
                    kill_timeout=2.0,
                )
                if killed:
                    logging.getLogger(type(self).__module__).warning(
                        "%s viewer required force kill during shutdown",
                        self.viewer_process_label,
                    )
                self.process = None
            self.runtime_endpoint.release_bound_ports()
            self.lifecycle_state.mark_stopped()

    def start(self, detached: bool = True) -> subprocess.Popen[bytes]:
        self.start_viewer(async_mode=False)
        if self.process is None:
            raise RuntimeError(f"{self.viewer_process_label} viewer process failed to start.")
        return self.process

    @property
    def process_pid_label(self) -> str:
        process = self.process
        if process is None:
            return "unknown"
        return ViewerProcessHandle.from_process(process).pid_label

    def send_control_message(self, message_type: str, timeout: float = 2.0) -> bool:
        if not self.is_running:
            logging.getLogger(type(self).__module__).warning(
                "%s viewer cannot send %s - viewer not running",
                self.viewer_process_label,
                message_type,
            )
            return False

        try:
            response = ViewerControlMessageRequest(
                endpoint=self.runtime_endpoint,
                message_type=message_type,
                timeout=timeout,
            ).send()
            if response.succeeded():
                logging.getLogger(type(self).__module__).info(
                    "%s viewer acknowledged %s",
                    self.viewer_process_label,
                    message_type,
                )
                return True
            logging.getLogger(type(self).__module__).warning(
                "%s viewer failed %s: %s",
                self.viewer_process_label,
                message_type,
                response.payload,
            )
            return False
        except Exception as error:
            logging.getLogger(type(self).__module__).warning(
                "%s viewer failed to send %s: %s",
                self.viewer_process_label,
                message_type,
                error,
            )
            return False

    def clear_viewer_state(self) -> bool:
        """Clear accumulated viewer state for a new pipeline run."""

        return self.send_control_message("clear_state")

    def settle_viewer_state(self, timeout: float = 30.0) -> bool:
        """Wait for queued viewer layer updates before state/screenshot reads."""

        return self.send_control_message(
            ViewerControlMessageType.SETTLE.value,
            timeout=timeout,
        )

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

    def colormap(self, channel_value: ViewerComponentValue) -> str | None:
        channel_number = self._channel_number(channel_value)
        if channel_number is None:
            return None
        return self.colors_by_channel.get(channel_number)

    @staticmethod
    def _channel_number(channel_value: ViewerComponentValue) -> int | None:
        if (
            channel_value is None
            or isinstance(channel_value, bool)
            or isinstance(channel_value, tuple)
        ):
            return None
        if isinstance(channel_value, int):
            return channel_value
        if isinstance(channel_value, float):
            if channel_value.is_integer():
                return int(channel_value)
            return None
        stripped = channel_value.strip()
        if stripped and stripped.lstrip("+-").isdigit():
            return int(stripped)
        return None
