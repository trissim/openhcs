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
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import ClassVar, TypeAlias, cast

from metaclass_registry import AutoRegisterMeta
from polystore.streaming_constants import StreamingDataType
from pyqt_reactive.process_launch import BackgroundProcessLaunchPolicy
from zmqruntime.config import TransportMode, ZMQConfig
from zmqruntime.messages import ControlMessageType
from zmqruntime.streaming import VisualizerProcessManager
from zmqruntime.transport import resolve_transport_mode
from zmqruntime.viewer_protocol import (
    ViewerBatchContextWireField as ViewerBatchContextWireField,
)
from zmqruntime.viewer_protocol import (
    ViewerBatchMessageType as ViewerBatchMessageType,
)
from zmqruntime.viewer_protocol import (
    ViewerBatchWireField as ViewerBatchWireField,
)
from zmqruntime.viewer_protocol import (
    ViewerControlReplyHeader as ViewerControlReplyHeader,
)
from zmqruntime.viewer_protocol import (
    ViewerControlReplyPayload as ViewerControlReplyPayload,
)
from zmqruntime.viewer_protocol import (
    ViewerControlResponseField,
    ViewerProtocolStatus,
    ViewerTransportEndpoint,
)

from openhcs.core.streaming_config_declarations import ViewerType
from openhcs.core.streaming_config_factory import (
    StreamingViewerRuntimeConfig,
)
from openhcs.runtime.viewer_controls import (
    ViewerNavigationControlOptions as ViewerNavigationControlOptions,
)
from openhcs.runtime.viewer_controls import (
    ViewerPayloadControlOptions as ViewerPayloadControlOptions,
)
from openhcs.runtime.viewer_controls import (
    ViewerScalar,
    ViewerStateControlOptions,
)

ViewerComponentValue: TypeAlias = ViewerScalar | tuple[ViewerScalar, ...]
NaturalTokenKey: TypeAlias = tuple[int, int | str]
NaturalTextKey: TypeAlias = tuple[NaturalTokenKey, ...]
ComponentValueSortKey: TypeAlias = tuple[int, int | float | NaturalTextKey, str, str]
ComponentTupleSortKey: TypeAlias = tuple[ComponentValueSortKey, ...]
ViewerProcess: TypeAlias = BaseProcess | subprocess.Popen[bytes]
ViewerLaunchLiteral: TypeAlias = str | int | float | bool | None


class ViewerControlMessageType(Enum):
    """Shared control-message names consumed by viewer servers."""

    SCREENSHOT = "screenshot"
    CLEAR_STATE = "clear_state"
    SETTLE = "settle"
    STATE = "state"
    PAYLOADS = "payloads"
    NAVIGATE = "navigate"


class ViewerSettlePhase(str, Enum):
    """Lifecycle phase for incremental viewer settlement."""

    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class ViewerSettleField(str, Enum):
    """Wire fields owned by viewer-settlement progress replies."""

    PHASE = "settle_phase"
    COMPLETED_UPDATE_COUNT = "completed_update_count"
    TOTAL_UPDATE_COUNT = "total_update_count"
    ACTIVE_ROUTE = "active_route"
    ACTIVE_ROUTE_WORK_UNIT_COUNT = "active_route_work_unit_count"
    ACTIVE_ROUTE_WORK_UNIT_ACTIVE = "active_route_work_unit_active"


class ViewerPayloadSummaryField(str, Enum):
    """Viewer payload-summary response fields."""

    SHAPE = "shape"
    NONZERO_COUNT = "nonzero_count"


class ViewerControlField(str, Enum):
    """Application-specific viewer control response payload fields."""

    SNAPSHOT = "snapshot"
    VIEWER = "viewer"
    RESOURCE = "resource"
    WIDTH = "width"
    HEIGHT = "height"
    LAYERS = "layers"
    LAYER_COUNT = "layer_count"
    ACTIVE_DIMENSION_LABEL_ROUTE = "active_dimension_label_route"
    VIEWER_NDIM = "viewer_ndim"
    CURRENT_STEP = "current_step"
    AXIS_LABELS = "axis_labels"
    COMPONENT_GROUP_COUNT = "component_group_count"
    COMPONENT_ITEM_COUNT = "component_item_count"


class ViewerLayerField(str, Enum):
    """Viewer layer-state response payload fields."""

    ROUTE_KEY = "route_key"
    PRODUCER_IDENTITIES = "producer_identities"
    TITLE = "title"
    MOUNTED = "mounted"
    ITEM_COUNT = "item_count"
    DATA_TYPES = "data_types"
    COMPONENT_VALUES = "component_values"
    PAYLOAD_SUMMARIES = "payload_summaries"
    AXIS_LABELS = "axis_labels"
    STACK_AXES = "stack_axes"
    AXIS_OFFSETS = "axis_offsets"
    SCALAR_LABELS = "scalar_labels"
    LABELS = "labels"
    COMPONENT_VALUE_COUNT = "component_value_count"
    COMPONENT_VALUES_TRUNCATED = "component_values_truncated"
    PAYLOAD_SUMMARY_COUNT = "payload_summary_count"
    PAYLOAD_SUMMARIES_TRUNCATED = "payload_summaries_truncated"
    AXIS_COMPONENT_VALUES = "axis_component_values"
    ROUTED_COMPONENT_VALUES = "routed_component_values"
    DATA_SHAPE = "data_shape"
    TRANSLATE = "translate"
    VISIBLE = "visible"
    SELECTED = "selected"
    FEATURE_ROW_COUNT = "feature_row_count"
    SELECTED_DATA_INDICES = "selected_data_indices"
    PENDING_UPDATE = "pending_update"
    PAYLOADS = "payloads"


class ViewerPayloadField(str, Enum):
    """Viewer layer payload-record response fields."""

    ROUTE_KEY = "route_key"
    DATA_TYPE = "data_type"
    PATH = "path"
    COMPONENTS = "components"
    AXIS_INDICES = "axis_indices"
    AGGREGATE_AXIS_INDICES = "aggregate_axis_indices"
    SUMMARY = "summary"
    ARRAY_VALUES = "array_values"
    ARRAY_VALUE_SUMMARY = "array_value_summary"
    SHAPE_PAYLOADS = "shape_payloads"


class ViewerDescriptorField(str, Enum):
    """Viewer descriptor payload fields."""

    TYPE = "type"
    TITLE = "title"


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
        if persistent:
            return cls.PERSISTENT
        return cls.NON_PERSISTENT


@dataclass(frozen=True, slots=True)
class ViewerControlResponse:
    """Typed view of a viewer control-message response."""

    payload: Mapping[str, object]

    @property
    def status(self) -> ViewerProtocolStatus:
        status_value = self.payload.get(ViewerControlResponseField.STATUS.value)
        if status_value is None:
            raise ValueError("Viewer control response is missing a status field.")
        return ViewerProtocolStatus(str(status_value))

    def succeeded(self) -> bool:
        return self.status is ViewerProtocolStatus.SUCCESS


@dataclass(frozen=True, slots=True)
class ViewerSettleProgress:
    """Typed progress for one viewer's incremental settlement cycle."""

    phase: ViewerSettlePhase
    completed_update_count: int
    total_update_count: int
    active_route: str | None = None
    active_route_work_unit_count: int = 0
    active_route_work_unit_active: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.completed_update_count) is not int
            or type(self.total_update_count) is not int
            or self.completed_update_count < 0
            or self.total_update_count < 0
            or self.completed_update_count > self.total_update_count
        ):
            raise ValueError(
                "Viewer settle progress counts must satisfy 0 <= completed <= total."
            )
        if self.phase is ViewerSettlePhase.COMPLETE and (
            self.completed_update_count != self.total_update_count
            or self.active_route is not None
        ):
            raise ValueError(
                "Completed viewer settlement must have no active route and all "
                "updates completed."
            )
        if self.active_route is not None and not isinstance(self.active_route, str):
            raise TypeError("Viewer settle active_route must be a string or None.")
        if (
            type(self.active_route_work_unit_count) is not int
            or self.active_route_work_unit_count < 0
        ):
            raise ValueError(
                "Viewer settle active-route work-unit count must be non-negative."
            )
        if self.active_route is None and self.active_route_work_unit_count:
            raise ValueError(
                "Viewer settle work-unit progress requires an active route."
            )
        if type(self.active_route_work_unit_active) is not bool:
            raise TypeError("Viewer settle active work-unit state must be boolean.")
        if self.active_route is None and self.active_route_work_unit_active:
            raise ValueError(
                "Viewer settle active work-unit state requires an active route."
            )

    @classmethod
    def complete(cls, total_update_count: int = 0) -> "ViewerSettleProgress":
        """Return terminal successful settlement progress."""

        return cls(
            phase=ViewerSettlePhase.COMPLETE,
            completed_update_count=total_update_count,
            total_update_count=total_update_count,
        )

    @classmethod
    def from_response(
        cls,
        response: ViewerControlResponse,
    ) -> "ViewerSettleProgress":
        """Parse exact settlement progress from a control response."""

        payload = response.payload
        active_route_value = payload[ViewerSettleField.ACTIVE_ROUTE.value]
        active_work_unit_value = payload[
            ViewerSettleField.ACTIVE_ROUTE_WORK_UNIT_ACTIVE.value
        ]
        if type(active_work_unit_value) is not bool:
            raise TypeError("Viewer settle active work-unit state must be boolean.")
        return cls(
            phase=ViewerSettlePhase(str(payload[ViewerSettleField.PHASE.value])),
            completed_update_count=int(
                payload[ViewerSettleField.COMPLETED_UPDATE_COUNT.value]
            ),
            total_update_count=int(payload[ViewerSettleField.TOTAL_UPDATE_COUNT.value]),
            active_route=(
                None if active_route_value is None else str(active_route_value)
            ),
            active_route_work_unit_count=int(
                payload[ViewerSettleField.ACTIVE_ROUTE_WORK_UNIT_COUNT.value]
            ),
            active_route_work_unit_active=active_work_unit_value,
        )

    def to_wire_mapping(self) -> dict[str, object]:
        """Return exact fields for a viewer control reply."""

        return {
            ViewerSettleField.PHASE.value: self.phase.value,
            ViewerSettleField.COMPLETED_UPDATE_COUNT.value: (
                self.completed_update_count
            ),
            ViewerSettleField.TOTAL_UPDATE_COUNT.value: self.total_update_count,
            ViewerSettleField.ACTIVE_ROUTE.value: self.active_route,
            ViewerSettleField.ACTIVE_ROUTE_WORK_UNIT_COUNT.value: (
                self.active_route_work_unit_count
            ),
            ViewerSettleField.ACTIVE_ROUTE_WORK_UNIT_ACTIVE.value: (
                self.active_route_work_unit_active
            ),
        }


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
    def tuple_key(
        cls, values: tuple[ViewerComponentValue, ...]
    ) -> ComponentTupleSortKey:
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

    WINDOWS = ("win32", None, None, {})
    DARWIN = ("Darwin", "Darwin", QtPlatformName.COCOA, {})
    LINUX = (
        "Linux",
        "Linux",
        QtPlatformName.XCB,
        {
            "QT_X11_NO_MITSHM": "1",
            "vblank_mode": "0",
        },
    )
    OTHER = ("other", None, None, {})

    def __new__(
        cls,
        value: str,
        system_name: str | None,
        qpa_platform: QtPlatformName | None,
        always_set: Mapping[str, str],
    ) -> "ViewerProcessPlatform":
        member = object.__new__(cls)
        member._value_ = value
        member.system_name = system_name
        member.qpa_platform = qpa_platform
        member.always_set = dict(always_set)
        return member

    @classmethod
    def current(cls) -> "ViewerProcessPlatform":
        if sys.platform == cls.WINDOWS.value:
            return cls.WINDOWS
        system_name = platform.system()
        for platform_family in cls:
            if platform_family.system_name == system_name:
                return platform_family
        return cls.OTHER

    def qt_environment_policy(self) -> "ViewerQtPlatformEnvironmentPolicy":
        return ViewerQtPlatformEnvironmentPolicy(
            qpa_platform=self.qpa_platform,
            always_set=self.always_set,
        )


class ViewerLaunchContextMode(Enum):
    """Provenance and availability of one detached viewer launch session."""

    INHERITED_GRAPHICAL_SESSION = "inherited_graphical_session"
    PROJECTED_GRAPHICAL_SESSION = "projected_graphical_session"
    HEADLESS = "headless"


@dataclass(frozen=True, slots=True)
class ViewerLaunchContext:
    """Typed graphical-session context consumed by detached viewer launch."""

    mode: ViewerLaunchContextMode
    environment_overlay: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment_overlay",
            dict(self.environment_overlay),
        )

    @classmethod
    def inherited_graphical_session(cls) -> "ViewerLaunchContext":
        """Declare that the launching process already owns a GUI session."""
        return cls(ViewerLaunchContextMode.INHERITED_GRAPHICAL_SESSION)

    @classmethod
    def projected_graphical_session(
        cls,
        environment: Mapping[str, str],
    ) -> "ViewerLaunchContext":
        """Carry a graphical environment already validated by its owner."""
        return cls(
            ViewerLaunchContextMode.PROJECTED_GRAPHICAL_SESSION,
            environment,
        )

    @classmethod
    def headless(cls) -> "ViewerLaunchContext":
        """Declare that no authoritative graphical session is available."""
        return cls(ViewerLaunchContextMode.HEADLESS)

    @property
    def graphical_session_available(self) -> bool:
        return self.mode is not ViewerLaunchContextMode.HEADLESS

    def child_environment(
        self,
        base_environment: Mapping[str, str],
    ) -> dict[str, str]:
        """Overlay projected GUI values onto the launching process environment."""
        environment = dict(base_environment)
        environment.update(self.environment_overlay)
        return environment


@dataclass(frozen=True, slots=True)
class ViewerGraphicalSessionUnavailableError(RuntimeError):
    """Raised before spawn when a detached interactive viewer has no GUI session."""

    viewer_type: ViewerType
    port: int

    def __str__(self) -> str:
        return (
            f"Cannot launch detached {self.viewer_type.value} viewer on port "
            f"{self.port}: no authoritative graphical session is available."
        )


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
        for kind in cls:
            if kind.wire_value == wire_value:
                return kind
        return None


def viewer_lifecycle_registry_key(
    name: str,
    cls: type,
) -> str:
    """Derive the lifecycle registry key from the declared detached entrypoint."""
    del name
    try:
        entrypoint = cast(
            DetachedViewerServerEntrypointSpec,
            cls.detached_server_entrypoint,
        )
    except AttributeError as exc:
        raise TypeError(
            f"{cls.__name__} must declare detached_server_entrypoint to register "
            "as a managed viewer lifecycle."
        ) from exc
    return entrypoint.viewer_type.value


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerServerLaunchRequest:
    """Shared launch request fields for OpenHCS viewer server processes."""

    port: int
    log_file_path: str | None = None
    transport_mode: TransportMode = TransportMode.IPC


@dataclass(frozen=True, slots=True)
class NapariViewerServerRequest(ViewerServerLaunchRequest):
    """Nominal launch request consumed by the Napari viewer server."""

    viewer_title: str
    replace_layers: bool = False


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
    def mode(self) -> TransportMode:
        return resolve_transport_mode(self.transport.transport_mode)

    @property
    def control_port(self) -> int:
        from zmqruntime.transport import get_control_port

        return get_control_port(self.port, self.config)

    def data_url(self) -> str:
        from zmqruntime.transport import get_zmq_transport_url

        return get_zmq_transport_url(
            self.port,
            host=self.host,
            mode=self.mode,
            config=self.config,
        )

    def control_url(self) -> str:
        from zmqruntime.transport import get_control_url

        return get_control_url(
            self.port,
            self.mode,
            host=self.host,
            config=self.config,
        )

    def in_use(self) -> bool:
        """Return whether either endpoint in this data/control pair is bound."""

        from zmqruntime.transport import is_port_in_use

        return any(
            is_port_in_use(
                port,
                self.mode,
                host=self.host,
                config=self.config,
            )
            for port in (self.port, self.control_port)
        )

    def remove_stale_ipc_sockets(self) -> tuple[int, ...]:
        """Remove only unowned IPC data/control paths for this endpoint."""

        from zmqruntime.transport import ipc_socket_is_stale, remove_ipc_socket

        if self.mode is not TransportMode.IPC:
            return ()
        removed: list[int] = []
        for port in (self.port, self.control_port):
            if ipc_socket_is_stale(port, self.config) and remove_ipc_socket(
                port,
                self.config,
            ):
                removed.append(port)
        return tuple(removed)

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
            self.mode,
            host=self.host,
            config=self.config,
            timeout_ms=timeout_ms,
            require_ready=require_ready,
        )

    def wait_ready(self, *, timeout: float, require_ready: bool = True) -> bool:
        from zmqruntime.transport import wait_for_server_ready

        return wait_for_server_ready(
            self.port,
            self.mode,
            host=self.host,
            config=self.config,
            timeout=timeout,
            require_ready=require_ready,
        )

    def release_bound_ports(self) -> None:
        if self.mode is TransportMode.IPC:
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
            raise ValueError(
                f"Detached viewer symbol is not a valid identifier: {name!r}"
            )
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
        return cls(
            tuple(DetachedViewerPythonExpression.literal(value) for value in values)
        )

    def append(
        self,
        *expressions: DetachedViewerPythonExpression,
    ) -> "DetachedViewerPythonArguments":
        return type(self)((*self.expressions, *expressions))

    def render(self) -> str:
        return ",\n".join(expression.source for expression in self.expressions)


@dataclass(frozen=True, slots=True)
class DetachedViewerLaunchLog:
    """Bounded diagnostics reader for one authoritative detached-viewer log."""

    path: Path
    max_bytes: int = 8192
    max_lines: int = 40

    def tail(self) -> str:
        try:
            with self.path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                offset = max(0, size - self.max_bytes)
                handle.seek(offset)
                payload = handle.read(self.max_bytes)
        except (FileNotFoundError, PermissionError, OSError):
            return ""

        if offset:
            _partial_line, separator, payload = payload.partition(b"\n")
            if not separator:
                return ""
        lines = payload.decode(errors="replace").splitlines()
        return "\n".join(lines[-self.max_lines :])


class DetachedViewerLaunchFailure(RuntimeError):
    """Detached viewer startup error plus its durable bounded diagnostics."""

    def __init__(
        self,
        *,
        viewer_type: ViewerType,
        port: int,
        cause: Exception,
        log_file: Path,
        log_tail: str,
    ) -> None:
        self.viewer_type = viewer_type
        self.port = port
        self.cause = cause
        self.log_file = log_file
        self.log_tail = log_tail
        details = (
            f"\nLast bounded launch-log output:\n{log_tail}"
            if log_tail
            else "\nThe launch log contained no readable output."
        )
        super().__init__(f"{cause}\nDetached viewer log: {log_file}{details}")


@dataclass(frozen=True, slots=True)
class DetachedViewerLaunchRequest(ViewerTypeIdentity):
    """Authoritative detached launch request for a viewer process."""

    port: int
    python_code: str
    log_file: Path
    cwd: Path = field(default_factory=Path.cwd)
    launch_context: ViewerLaunchContext = field(
        default_factory=ViewerLaunchContext.inherited_graphical_session
    )
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

    def failure(self, cause: Exception) -> DetachedViewerLaunchFailure:
        """Project one startup exception through this request's log authority."""
        return DetachedViewerLaunchFailure(
            viewer_type=self.viewer_type,
            port=self.port,
            cause=cause,
            log_file=self.log_file,
            log_tail=DetachedViewerLaunchLog(self.log_file).tail(),
        )

    def launch(self) -> subprocess.Popen[bytes]:
        if not self.launch_context.graphical_session_available:
            raise ViewerGraphicalSessionUnavailableError(
                viewer_type=self.viewer_type,
                port=self.port,
            )
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        launch_env = self.launch_context.child_environment(os.environ)
        ViewerQtEnvironmentPolicy(self.platform).apply_to(launch_env)
        log_handle = self.log_file.open("w")
        return subprocess.Popen(
            self.command(),
            env=launch_env,
            cwd=str(self.cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            **BackgroundProcessLaunchPolicy.current(
                detached=True
            ).popen_arguments(),
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
        transport_mode: TransportMode,
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
            "    from zmqruntime.config import TransportMode",
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
        transport_mode: TransportMode,
        arguments: DetachedViewerPythonArguments,
        log_file: Path,
        cwd: Path | None = None,
        launch_context: ViewerLaunchContext | None = None,
    ) -> DetachedViewerLaunchRequest:
        if cwd is None:
            cwd = Path.cwd()
        if launch_context is None:
            launch_context = ViewerLaunchContext.inherited_graphical_session()
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
            launch_context=launch_context,
        )


@dataclass(frozen=True, slots=True)
class ViewerQtPlatformEnvironmentPolicy:
    """Environment mutations for one viewer platform."""

    qpa_platform: QtPlatformName | None = None
    always_set: Mapping[str, str] = field(default_factory=dict)

    def apply_to(
        self,
        env: MutableMapping[str, str],
    ) -> MutableMapping[str, str]:
        if self.qpa_platform is not None and "QT_QPA_PLATFORM" not in env:
            env["QT_QPA_PLATFORM"] = self.qpa_platform.value
        env.update(self.always_set)
        return env


@dataclass(frozen=True, slots=True)
class ViewerQtEnvironmentPolicy:
    """Apply viewer-safe Qt environment defaults for the current platform."""

    platform: ViewerProcessPlatform = field(
        default_factory=ViewerProcessPlatform.current
    )

    @staticmethod
    def active_qt_plugin_path() -> str | None:
        """Return the active Qt binding's authoritative plugin directory."""

        try:
            from qtpy.QtCore import QLibraryInfo
        except ImportError:
            return None

        plugin_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
        return plugin_path or None

    def apply_to(
        self,
        env: MutableMapping[str, str],
    ) -> MutableMapping[str, str]:
        self.platform.qt_environment_policy().apply_to(env)
        plugin_path = self.active_qt_plugin_path()
        if plugin_path is not None:
            # Private plugin trees exported by dependencies such as OpenCV are
            # not interchangeable with the active Qt binding. Qt itself owns
            # the plugin directory used by the detached viewer.
            env.pop("QT_PLUGIN_PATH", None)
            env["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
        return env


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

    QUICK = ("quick", 200, False)
    EXISTING_VIEWER = ("existing_viewer", 500, True)

    def __new__(
        cls,
        value: str,
        timeout_ms: int,
        require_ready: bool,
    ) -> "ViewerControlPingMode":
        member = object.__new__(cls)
        member._value_ = value
        member.timeout_ms = timeout_ms
        member.require_ready = require_ready
        return member

    def policy(self) -> "ViewerControlPingPolicy":
        return ViewerControlPingPolicy(
            timeout_ms=self.timeout_ms,
            require_ready=self.require_ready,
        )


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
        policy = mode.policy()
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
    payload: object | None = None
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
            request: dict[str, object] = {
                ViewerControlResponseField.TYPE.value: self.message_type
            }
            if self.payload is not None:
                request[ViewerControlResponseField.PAYLOAD.value] = self.payload
            socket.send(pickle.dumps(request))
            payload = pickle.loads(socket.recv())
            if not isinstance(payload, Mapping):
                raise TypeError(
                    "Viewer control response must be a mapping, "
                    f"got {type(payload).__name__}."
                )
            return ViewerControlResponse(cast(Mapping[str, object], payload))
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
    ) -> None:
        super().__init__(port=runtime_config.transport_endpoint.port)
        self.persistent: bool = runtime_config.persistent
        self.display_enabled: bool = runtime_config.display_enabled
        self.scope_accent_color = runtime_config.scope_accent_color
        self.lifecycle_presentation = runtime_config.presentation
        self._launch_context = ViewerLaunchContext.inherited_graphical_session()
        self.runtime_endpoint = ViewerRuntimeEndpoint(
            transport=runtime_config.transport_endpoint,
            config=runtime_config.transport_config,
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
            message_type=ControlMessageType.FORCE_SHUTDOWN.value,
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
        """Wait for the viewer endpoint to bind and report ready."""
        return self.runtime_endpoint.wait_ready(
            timeout=timeout,
            require_ready=True,
        )

    def configure_launch_context(
        self,
        launch_context: ViewerLaunchContext,
    ) -> None:
        """Set the typed graphical context before this lifecycle launches."""
        self._launch_context = launch_context

    def detached_launch_request(self) -> DetachedViewerLaunchRequest:
        port = self.required_port
        log_file = self.detached_server_entrypoint.log_file_for(port)
        return self.detached_server_entrypoint.launch_request(
            port=port,
            transport_mode=self.runtime_endpoint.mode,
            arguments=self.detached_server_arguments(log_file=log_file),
            log_file=log_file,
            launch_context=self._launch_context,
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
            raise RuntimeError(
                f"{self.viewer_process_label} viewer process failed to start."
            )
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

        return self.send_control_message(ViewerControlMessageType.CLEAR_STATE.value)

    def settle_viewer_state(
        self,
        timeout: float = 30.0,
        *,
        progress_callback: Callable[[ViewerSettleProgress], None] | None = None,
    ) -> bool:
        """Wait while the viewer reports forward settlement progress."""

        if timeout <= 0:
            raise ValueError("Viewer settlement no-progress timeout must be positive.")

        logger = logging.getLogger(type(self).__module__)
        last_progress_marker = (-1, -1)
        no_progress_deadline = time.monotonic() + timeout
        while True:
            try:
                response = ViewerControlMessageRequest(
                    endpoint=self.runtime_endpoint,
                    message_type=ViewerControlMessageType.SETTLE.value,
                    timeout=timeout,
                ).send()
                progress = ViewerSettleProgress.from_response(response)
                if progress_callback is not None:
                    progress_callback(progress)
            except Exception as error:
                logger.warning(
                    "%s viewer settlement progress request failed: %s",
                    self.viewer_process_label,
                    error,
                )
                return False

            if not response.succeeded() or progress.phase is ViewerSettlePhase.FAILED:
                logger.warning(
                    "%s viewer settlement failed: %s",
                    self.viewer_process_label,
                    response.payload,
                )
                return False
            if progress.phase is ViewerSettlePhase.COMPLETE:
                logger.info(
                    "%s viewer settled %d/%d layer updates",
                    self.viewer_process_label,
                    progress.completed_update_count,
                    progress.total_update_count,
                )
                return True

            now = time.monotonic()
            progress_marker = (
                progress.completed_update_count,
                progress.active_route_work_unit_count,
            )
            if progress_marker > last_progress_marker:
                last_progress_marker = progress_marker
                no_progress_deadline = now + timeout
                logger.info(
                    "%s viewer settling layer updates: %d/%d; active route "
                    "completed %d bounded work unit(s)",
                    self.viewer_process_label,
                    progress.completed_update_count,
                    progress.total_update_count,
                    progress.active_route_work_unit_count,
                )
            elif progress.active_route_work_unit_active:
                no_progress_deadline = now + timeout
            elif now >= no_progress_deadline:
                logger.warning(
                    "%s viewer settlement made no progress for %.1f seconds "
                    "at %d/%d updates and %d active-route work unit(s)",
                    self.viewer_process_label,
                    timeout,
                    progress.completed_update_count,
                    progress.total_update_count,
                    progress.active_route_work_unit_count,
                )
                return False
            time.sleep(min(0.05, max(0.0, no_progress_deadline - now)))

    def read_viewer_state(self, timeout: float = 30.0) -> ViewerControlResponse:
        """Return typed state from the settled viewer control endpoint."""

        if not self.is_running:
            raise RuntimeError(
                f"{self.viewer_process_label} viewer on port {self.required_port} "
                "is not running."
            )
        response = ViewerControlMessageRequest(
            endpoint=self.runtime_endpoint,
            message_type=ViewerControlMessageType.STATE.value,
            payload=ViewerStateControlOptions(),
            timeout=timeout,
        ).send()
        if not response.succeeded():
            raise RuntimeError(
                f"{self.viewer_process_label} viewer state request failed: "
                f"{response.payload!r}."
            )
        return response

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
