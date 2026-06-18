"""Process boundary server for the OpenHCS UI agent bridge."""

from __future__ import annotations

import json
import os
import pickle
import secrets
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Callable, ClassVar

from metaclass_registry import AutoRegisterMeta
from zmqruntime.transport import coerce_transport_mode, get_zmq_transport_url

from openhcs.constants.constants import CONTROL_PORT_OFFSET
from openhcs.agent.dto.common import AgentError, JsonObject, SCHEMA_VERSION
from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.dto.ui_bridge import (
    UiActionCatalog,
    UiActionInvokeRequest,
    UiActionInvokeResult,
    UiBranchCatalog,
    UiBranchSwitchRequest,
    UiBridgeConnectionFields,
    UiBridgeConnectionSpec,
    UiBridgeDescriptorFile,
    UiBridgeDescriptorWirePayload,
    UiBridgeOperationRef,
    UiBridgeOperationStatusRequest,
    UiBridgeRequestEnvelope,
    UiBridgeResponseEnvelope,
    UiBridgeStatus,
    UiCodeDocument,
    UiCodeDocumentApplyResult,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentCatalog,
    UiCodeDocumentRequest,
    UiCodeDocumentValidationResult,
    UiCodeDocumentValidationRequest,
    UiObjectStateScopeCatalog,
    UiObjectStateScopeListRequest,
    UiStateSurfaceCatalog,
    UiStateSurfaceDocument,
    UiStateSurfaceRequest,
    UiSnapshotCatalog,
    UiSnapshotListRequest,
    UiSnapshotRestoreResult,
    UiSnapshotRestoreRequest,
    UiTimeTravelHeadRequest,
    UiWindowCatalog,
    UiWindowFocusRequest,
    UiWindowFocusResult,
    UiWindowNavigateRequest,
    UiWindowNavigateResult,
    UiWindowSnapshotRequest,
    UiWindowSnapshotResult,
)
from openhcs.agent.serialization import to_jsonable
from openhcs.agent.services.ui_bridge_service import (
    UI_BRIDGE_PROTOCOL_VERSION,
    UiBridgeDescriptorDirectoryAuthority,
)
from openhcs.agent.services.ui_bridge_transport import (
    AgentDtoJsonCodec,
    UiBridgeOperationName,
)
from openhcs.pyqt_gui.config import AgentUiBridgeConfig, AgentUiBridgeDescriptorPaths
from openhcs.pyqt_gui.services.ui_agent_bridge import UiAgentBridgeService
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG


DEFAULT_UI_BRIDGE_HOST = "127.0.0.1"
DEFAULT_UI_BRIDGE_TRANSPORT = "tcp"
DEFAULT_UI_BRIDGE_START_TIMEOUT_SECONDS = 5.0
UI_BRIDGE_BROWSER_SERVER_NAME = "OpenHCSUiBridgeServer"
UI_BRIDGE_BROWSER_PONG_TYPE = "pong"
UI_BRIDGE_BROWSER_ERROR_TYPE = "error"


class UiBridgeBrowserControlMessageType(str, Enum):
    PING = "ping"


@dataclass(frozen=True, slots=True)
class UiBridgeBrowserControlRequest:
    message_type: UiBridgeBrowserControlMessageType

    @classmethod
    def from_wire_payload(cls, wire_payload: bytes) -> "UiBridgeBrowserControlRequest":
        payload = pickle.loads(wire_payload)
        if not isinstance(payload, dict):
            raise ValueError("UI bridge browser control request must be a dictionary.")
        raw_message_type = payload["type"]
        if not isinstance(raw_message_type, str):
            raise ValueError("UI bridge browser control request is missing a type.")
        return cls(message_type=UiBridgeBrowserControlMessageType(raw_message_type))

UiBridgeOperationDispatchResult = (
    UiBridgeStatus
    | UiCodeDocumentCatalog
    | UiCodeDocument
    | UiCodeDocumentValidationResult
    | UiCodeDocumentApplyResult
    | UiStateSurfaceCatalog
    | UiStateSurfaceDocument
    | UiActionCatalog
    | UiActionInvokeResult
    | UiWindowCatalog
    | UiWindowFocusResult
    | UiWindowNavigateResult
    | UiObjectStateScopeCatalog
    | UiSnapshotCatalog
    | UiSnapshotRestoreResult
    | UiBranchCatalog
    | UiBridgeOperationRef
)


@dataclass(frozen=True, slots=True)
class UiBridgeServerIdentitySeed:
    value: str | None = None

    def resolve(self) -> str:
        if self.value is not None:
            return self.value
        return f"ui-{uuid.uuid4()}"


@dataclass(frozen=True, slots=True)
class UiBridgeServerAuthSeed:
    value: str | None = None

    def resolve(self) -> str:
        if self.value is not None:
            return self.value
        return secrets.token_urlsafe(32)


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorPathRequest:
    directory_path: Path | None = None
    explicit_file_path: Path | None = None

    @classmethod
    def from_agent_paths(
        cls,
        paths: AgentUiBridgeDescriptorPaths,
    ) -> "UiBridgeDescriptorPathRequest":
        return cls(
            directory_path=(
                Path(paths.directory_path)
                if paths.directory_path is not None
                else None
            ),
            explicit_file_path=(
                Path(paths.explicit_file_path)
                if paths.explicit_file_path is not None
                else None
            ),
        )

    def path_for(self, bridge_instance_id: str) -> Path:
        if self.explicit_file_path is not None:
            return self.explicit_file_path.expanduser()
        return self.directory_or_default() / f"ui_bridge_{bridge_instance_id}.json"

    def directory_or_default(self) -> Path:
        if self.directory_path is not None:
            return self.directory_path.expanduser()
        return UiBridgeDescriptorDirectoryAuthority.default_descriptor_dir()


@dataclass(frozen=True, slots=True)
class UiBridgeServerConfig:
    connection: ExecutionConnectionSpec = field(
        default_factory=lambda: ExecutionConnectionSpec(
            host=DEFAULT_UI_BRIDGE_HOST,
            port=0,
            transport_mode=DEFAULT_UI_BRIDGE_TRANSPORT,
        )
    )
    descriptor_path_request: UiBridgeDescriptorPathRequest = field(
        default_factory=UiBridgeDescriptorPathRequest
    )
    identity_seed: UiBridgeServerIdentitySeed = field(
        default_factory=UiBridgeServerIdentitySeed
    )
    auth_seed: UiBridgeServerAuthSeed = field(default_factory=UiBridgeServerAuthSeed)
    poll_timeout_ms: int = 100
    shutdown_timeout_seconds: float = 2.0

    @property
    def resolved_bridge_instance_id(self) -> str:
        return self.identity_seed.resolve()

    @property
    def resolved_auth_token(self) -> str:
        return self.auth_seed.resolve()

    @classmethod
    def from_agent_config(
        cls,
        config: AgentUiBridgeConfig,
    ) -> "UiBridgeServerConfig":
        return cls(
            connection=config.connection,
            descriptor_path_request=UiBridgeDescriptorPathRequest.from_agent_paths(
                config.descriptor_paths
            ),
        )


@dataclass(frozen=True, slots=True)
class UiBridgeServerBinding:
    fields: UiBridgeConnectionFields

    @classmethod
    def from_runtime(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        bridge_instance_id: str,
        descriptor_file_path: Path,
        auth_token: str,
    ) -> "UiBridgeServerBinding":
        return cls(
            fields=UiBridgeConnectionFields.from_values(
                host=connection.host,
                port=connection.port,
                transport_mode=connection.transport_mode,
                persistent=connection.persistent,
                auth_token=auth_token,
                descriptor_file_path=str(descriptor_file_path),
                bridge_instance_id=bridge_instance_id,
            )
        )

    @property
    def connection(self) -> ExecutionConnectionSpec:
        if self.fields.connection is None:
            raise RuntimeError("UI bridge binding is missing connection fields.")
        return self.fields.connection

    @property
    def bridge_instance_id(self) -> str:
        if self.fields.bridge_instance_id is None:
            raise RuntimeError("UI bridge binding is missing an instance id.")
        return self.fields.bridge_instance_id

    @property
    def descriptor_file_path(self) -> Path:
        if self.fields.descriptor_file_path is None:
            raise RuntimeError("UI bridge binding is missing a descriptor path.")
        return Path(self.fields.descriptor_file_path)

    @property
    def auth_token(self) -> str:
        if self.fields.auth_token is None:
            raise RuntimeError("UI bridge binding is missing an auth token.")
        return self.fields.auth_token

    def token_bearing_connection(self) -> UiBridgeConnectionSpec:
        return UiBridgeConnectionSpec.from_fields(
            self.fields,
        )


class UiBridgeDescriptorWriter:
    """Own the token-bearing descriptor file for one running UI bridge."""

    def __init__(self, config: UiBridgeServerConfig) -> None:
        self._config = config

    def path_for(self, bridge_instance_id: str) -> Path:
        return self._config.descriptor_path_request.path_for(bridge_instance_id)

    def write(
        self,
        descriptor: UiBridgeDescriptorFile,
    ) -> Path:
        path = Path(descriptor.descriptor_file_path).expanduser()
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        path.write_text(
            json.dumps(
                to_jsonable(UiBridgeDescriptorWirePayload.from_descriptor(descriptor)),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        path.chmod(0o600)
        return path

    def remove(self, path: Path | None) -> None:
        if path is None:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            return


@dataclass(frozen=True, slots=True)
class UiBridgeUnsupportedOperationError(LookupError):
    """Raised when an agent requests an operation this bridge does not expose."""

    operation_name: str

    def __str__(self) -> str:
        return f"Unsupported UI bridge operation: {self.operation_name}"


class UiBridgeRequestOperation(ABC, metaclass=AutoRegisterMeta):
    """Registered handler for one UI bridge request operation."""

    __registry_key__ = "operation"
    __skip_if_no_key__ = True

    operation: ClassVar[UiBridgeOperationName | None] = None

    @classmethod
    def for_name(cls, operation: UiBridgeOperationName) -> "UiBridgeRequestOperation":
        if operation not in cls.__registry__:
            raise UiBridgeUnsupportedOperationError(operation.value)
        return cls.__registry__[operation]()

    @classmethod
    def supported_operation_names(cls) -> tuple[str, ...]:
        return tuple(
            operation.value
            for operation in UiBridgeOperationName
            if operation in cls.__registry__
        )

    @abstractmethod
    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiBridgeOperationDispatchResult:
        raise NotImplementedError


class UiBridgeStatusOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.STATUS

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiBridgeStatus:
        del request
        return dispatcher.status_result()


class UiBridgeListDocumentsOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.LIST_DOCUMENTS

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiCodeDocumentCatalog:
        del request
        return dispatcher.bridge.list_documents()


class UiBridgeListStateSurfacesOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.LIST_STATE_SURFACES

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiStateSurfaceCatalog:
        del request
        return dispatcher.bridge.list_state_surfaces()


class UiBridgeGetDocumentOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.GET_DOCUMENT

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiCodeDocument:
        return dispatcher.bridge.get_document(
            dispatcher.request_payload(UiCodeDocumentRequest, request)
        )


class UiBridgeGetStateSurfaceOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.GET_STATE_SURFACE

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiStateSurfaceDocument:
        return dispatcher.bridge.get_state_surface(
            dispatcher.request_payload(UiStateSurfaceRequest, request)
        )


class UiBridgeListActionsOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.LIST_ACTIONS

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiActionCatalog:
        del request
        return dispatcher.bridge.list_actions()


class UiBridgeInvokeActionOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.INVOKE_ACTION

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiActionInvokeResult:
        return dispatcher.bridge.invoke_action(
            dispatcher.request_payload(UiActionInvokeRequest, request)
        )


class UiBridgeListWindowsOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.LIST_WINDOWS

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiWindowCatalog:
        del request
        return dispatcher.bridge.list_windows()


class UiBridgeFocusWindowOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.FOCUS_WINDOW

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiWindowFocusResult:
        return dispatcher.bridge.focus_window(
            dispatcher.request_payload(UiWindowFocusRequest, request)
        )


class UiBridgeNavigateWindowOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.NAVIGATE_WINDOW

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiWindowNavigateResult:
        return dispatcher.bridge.navigate_window(
            dispatcher.request_payload(UiWindowNavigateRequest, request)
        )


class UiBridgeSnapshotWindowOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.SNAPSHOT_WINDOW

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiWindowSnapshotResult:
        return dispatcher.bridge.snapshot_window(
            dispatcher.request_payload(UiWindowSnapshotRequest, request)
        )


class UiBridgeListObjectStateScopesOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.LIST_OBJECT_STATE_SCOPES

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiObjectStateScopeCatalog:
        return dispatcher.bridge.list_object_state_scopes(
            dispatcher.request_payload(UiObjectStateScopeListRequest, request)
        )


class UiBridgeValidateDocumentOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.VALIDATE_DOCUMENT

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiCodeDocumentValidationResult:
        return dispatcher.bridge.validate_document(
            dispatcher.request_payload(UiCodeDocumentValidationRequest, request)
        )


class UiBridgeApplyDocumentOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.APPLY_DOCUMENT

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiCodeDocumentApplyResult:
        return dispatcher.bridge.apply_document(
            dispatcher.request_payload(UiCodeDocumentApplyRequest, request)
        )


class UiBridgeListSnapshotsOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.LIST_SNAPSHOTS

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiSnapshotCatalog:
        return dispatcher.bridge.list_snapshots(
            dispatcher.request_payload(UiSnapshotListRequest, request)
        )


class UiBridgeRestoreSnapshotOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.RESTORE_SNAPSHOT

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiSnapshotRestoreResult:
        return dispatcher.bridge.restore_snapshot(
            dispatcher.request_payload(UiSnapshotRestoreRequest, request)
        )


class UiBridgeTimeTravelHeadOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.TIME_TRAVEL_HEAD

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiSnapshotRestoreResult:
        return dispatcher.bridge.time_travel_head(
            dispatcher.request_payload(UiTimeTravelHeadRequest, request)
        )


class UiBridgeListBranchesOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.LIST_BRANCHES

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiBranchCatalog:
        del request
        return dispatcher.bridge.list_branches()


class UiBridgeSwitchBranchOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.SWITCH_BRANCH

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiSnapshotRestoreResult:
        return dispatcher.bridge.switch_branch(
            dispatcher.request_payload(UiBranchSwitchRequest, request)
        )


class UiBridgeGetOperationStatusOperation(UiBridgeRequestOperation):
    operation = UiBridgeOperationName.GET_OPERATION_STATUS

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiBridgeOperationRef:
        status_request = dispatcher.request_payload(UiBridgeOperationStatusRequest, request)
        return dispatcher.bridge.get_operation_status(status_request.operation_id)


class UiBridgeRequestErrorAuthority:
    """Classify request-dispatch exceptions into agent-facing error codes."""

    _ERROR_CODE_BY_TYPE = {
        PermissionError: "ui_bridge_auth_failed",
        UiBridgeUnsupportedOperationError: "unsupported_ui_bridge_operation",
        ValueError: "invalid_ui_bridge_request",
    }
    DEFAULT_ERROR_CODE = "ui_bridge_request_failed"

    @classmethod
    def agent_error(cls, exception: Exception) -> AgentError:
        exception_type = cls._classified_type(exception)
        if exception_type in cls._ERROR_CODE_BY_TYPE:
            return AgentError.from_exception(
                cls._ERROR_CODE_BY_TYPE[exception_type],
                exception,
            )
        return AgentError.from_exception(cls.DEFAULT_ERROR_CODE, exception)

    @classmethod
    def _classified_type(cls, exception: Exception) -> type[BaseException]:
        for exception_type in type(exception).__mro__:
            if exception_type in cls._ERROR_CODE_BY_TYPE:
                return exception_type
        return BaseException


class UiBridgeRequestDispatcher:
    """Route validated UI bridge envelopes to the in-process bridge service."""

    _AUTH_FREE_OPERATIONS = frozenset((UiBridgeOperationName.STATUS.value,))

    def __init__(
        self,
        bridge: UiAgentBridgeService,
        *,
        auth_token: str,
        binding_supplier: Callable[[], UiBridgeServerBinding],
    ) -> None:
        self._bridge = bridge
        self._auth_token = auth_token
        self._binding_supplier = binding_supplier

    @property
    def bridge(self) -> UiAgentBridgeService:
        return self._bridge

    def dispatch(self, payload: JsonObject) -> JsonObject:
        try:
            request = AgentDtoJsonCodec.dataclass_from_json(
                UiBridgeRequestEnvelope,
                payload,
            )
            self._validate_request(request)
            result = self._operation_result(request)
            response = UiBridgeResponseEnvelope(
                schema_version=SCHEMA_VERSION,
                bridge_protocol_version=UI_BRIDGE_PROTOCOL_VERSION,
                request_id=request.request_id,
                ok=True,
                payload=self._result_payload(result),
            )
        except Exception as exc:
            response = self._error_response(payload, exc)
        return self._response_payload(response)

    def _validate_request(self, request: UiBridgeRequestEnvelope) -> None:
        if request.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported agent schema version: {request.schema_version}")
        if request.bridge_protocol_version != UI_BRIDGE_PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported UI bridge protocol version: {request.bridge_protocol_version}"
            )
        if (
            request.operation not in self._AUTH_FREE_OPERATIONS
            and request.auth_token != self._auth_token
        ):
            raise PermissionError("UI bridge auth token is missing or invalid.")

    def _operation_result(
        self,
        request: UiBridgeRequestEnvelope,
    ) -> UiBridgeOperationDispatchResult:
        try:
            operation = UiBridgeOperationName(request.operation)
        except ValueError as exc:
            raise UiBridgeUnsupportedOperationError(request.operation) from exc
        return UiBridgeRequestOperation.for_name(operation).execute(self, request)

    def status_result(self) -> UiBridgeStatus:
        binding = self._binding_supplier()
        return replace(
            self._bridge.status(),
            auth_required=True,
            bridge_instance_id=binding.bridge_instance_id,
            connection=binding.connection,
            descriptor_file_path=str(binding.descriptor_file_path),
            supported_operations=UiBridgeRequestOperation.supported_operation_names(),
            provider_catalog_schema_versions=(SCHEMA_VERSION,),
            bridge_features=(
                "ui_code_documents",
                "ui_state_surfaces",
                "ui_actions",
                "ui_windows",
                "ui_window_navigation",
                "ui_window_snapshots",
                "objectstate_scopes",
                "objectstate_snapshots",
                "objectstate_branches",
                "operation_status",
            ),
        )

    @staticmethod
    def request_payload(target_type, request: UiBridgeRequestEnvelope):
        return AgentDtoJsonCodec.dataclass_from_json(target_type, request.payload)

    @staticmethod
    def _result_payload(result) -> JsonObject:
        payload = to_jsonable(result)
        if not isinstance(payload, dict):
            raise TypeError(
                f"UI bridge operation result must serialize to a JSON object, "
                f"got {type(payload).__name__}"
            )
        return payload

    @staticmethod
    def _response_payload(response: UiBridgeResponseEnvelope) -> JsonObject:
        payload = to_jsonable(response)
        if not isinstance(payload, dict):
            raise TypeError("UI bridge response envelope did not serialize to a JSON object.")
        return payload

    def _error_response(
        self,
        payload: JsonObject,
        exception: Exception,
    ) -> UiBridgeResponseEnvelope:
        return UiBridgeResponseEnvelope(
            schema_version=SCHEMA_VERSION,
            bridge_protocol_version=UI_BRIDGE_PROTOCOL_VERSION,
            request_id=self._request_id_from_payload(payload),
            ok=False,
            payload={},
            errors=(self._agent_error(exception),),
        )

    @staticmethod
    def _request_id_from_payload(payload: JsonObject) -> str:
        request_id = payload.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id
        return "unknown"

    @staticmethod
    def _agent_error(exception: Exception) -> AgentError:
        return UiBridgeRequestErrorAuthority.agent_error(exception)


class UiBridgeControlServer:
    """Background ZMQ REP server exposing one UiAgentBridgeService."""

    def __init__(
        self,
        bridge: UiAgentBridgeService,
        config: UiBridgeServerConfig | None = None,
    ) -> None:
        self._bridge = bridge
        self._config = config or UiBridgeServerConfig()
        self._bridge_instance_id = self._config.resolved_bridge_instance_id
        self._auth_token = self._config.resolved_auth_token
        self._descriptor_writer = UiBridgeDescriptorWriter(self._config)
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._binding: UiBridgeServerBinding | None = None
        self._startup_error: Exception | None = None

    @property
    def binding(self) -> UiBridgeServerBinding:
        if self._binding is None:
            raise RuntimeError("UI bridge server is not running.")
        return self._binding

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive() and self._binding is not None

    def start(
        self,
        *,
        timeout_seconds: float = DEFAULT_UI_BRIDGE_START_TIMEOUT_SECONDS,
    ) -> UiBridgeServerBinding:
        if self.is_running:
            return self.binding
        self._stop_event.clear()
        self._ready_event.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._serve,
            name=f"OpenHCSUiBridge-{self._bridge_instance_id}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout_seconds):
            self.stop()
            raise TimeoutError("Timed out waiting for UI bridge server to start.")
        if self._startup_error is not None:
            raise RuntimeError("Failed to start UI bridge server.") from self._startup_error
        return self.binding

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(self._config.shutdown_timeout_seconds)
        self._descriptor_writer.remove(
            self._binding.descriptor_file_path if self._binding is not None else None
        )
        self._thread = None
        self._binding = None

    def _serve(self) -> None:
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        browser_control_socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        browser_control_socket.setsockopt(zmq.LINGER, 0)
        descriptor_file_path: Path | None = None
        try:
            connection = self._bind(socket)
            self._bind_browser_control_socket(browser_control_socket, connection)
            descriptor_file_path = self._descriptor_writer.path_for(self._bridge_instance_id)
            self._binding = UiBridgeServerBinding.from_runtime(
                connection=connection,
                bridge_instance_id=self._bridge_instance_id,
                descriptor_file_path=descriptor_file_path,
                auth_token=self._auth_token,
            )
            self._descriptor_writer.write(
                UiBridgeDescriptorFile(
                    schema_version=SCHEMA_VERSION,
                    bridge_protocol_version=UI_BRIDGE_PROTOCOL_VERSION,
                    bridge_instance_id=self._bridge_instance_id,
                    pid=os.getpid(),
                    started_at_unix=time.time(),
                    connection=connection,
                    auth_token=self._auth_token,
                    descriptor_file_path=str(descriptor_file_path),
                )
            )
            dispatcher = UiBridgeRequestDispatcher(
                self._bridge,
                auth_token=self._auth_token,
                binding_supplier=lambda: self.binding,
            )
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            poller.register(browser_control_socket, zmq.POLLIN)
            self._ready_event.set()
            while not self._stop_event.is_set():
                events = dict(poller.poll(self._config.poll_timeout_ms))
                if socket in events:
                    request_payload = socket.recv_json()
                    if not isinstance(request_payload, dict):
                        request_payload = {}
                    socket.send_json(dispatcher.dispatch(request_payload))
                if browser_control_socket in events:
                    browser_control_socket.send(
                        self._browser_control_response_payload(
                            browser_control_socket.recv(),
                            connection,
                        )
                    )
        except Exception as exc:
            self._startup_error = exc
            self._ready_event.set()
        finally:
            self._descriptor_writer.remove(descriptor_file_path)
            browser_control_socket.close(linger=0)
            socket.close(linger=0)
            context.term()

    def _bind(self, socket) -> ExecutionConnectionSpec:
        requested_connection = self._config.connection
        mode = coerce_transport_mode(requested_connection.transport_mode)
        if mode is None or mode.value == DEFAULT_UI_BRIDGE_TRANSPORT:
            return self._bind_tcp(socket)
        if requested_connection.port is None:
            raise ValueError("Non-TCP UI bridge transport requires an explicit port.")
        socket.bind(
            get_zmq_transport_url(
                requested_connection.port,
                host=requested_connection.host,
                mode=mode,
                config=OPENHCS_ZMQ_CONFIG,
            )
        )
        return ExecutionConnectionSpec(
            host=requested_connection.host,
            port=requested_connection.port,
            transport_mode=mode.value,
        )

    def _bind_tcp(self, socket) -> ExecutionConnectionSpec:
        requested_connection = self._config.connection
        if requested_connection.port in (None, 0):
            port = socket.bind_to_random_port(f"tcp://{requested_connection.host}")
        else:
            port = requested_connection.port
            socket.bind(f"tcp://{requested_connection.host}:{port}")
        return ExecutionConnectionSpec(
            host=requested_connection.host,
            port=port,
            transport_mode=DEFAULT_UI_BRIDGE_TRANSPORT,
        )

    def _bind_browser_control_socket(
        self,
        socket,
        connection: ExecutionConnectionSpec,
    ) -> None:
        if connection.port is None:
            raise ValueError("UI bridge browser control socket requires a data port.")
        control_port = connection.port + CONTROL_PORT_OFFSET
        mode = coerce_transport_mode(connection.transport_mode)
        if mode is None or mode.value == DEFAULT_UI_BRIDGE_TRANSPORT:
            socket.bind(f"tcp://{connection.host}:{control_port}")
            return
        socket.bind(
            get_zmq_transport_url(
                control_port,
                host=connection.host,
                mode=mode,
                config=OPENHCS_ZMQ_CONFIG,
            )
        )

    def _browser_control_response_payload(
        self,
        request_payload: bytes,
        connection: ExecutionConnectionSpec,
    ) -> bytes:
        try:
            request = UiBridgeBrowserControlRequest.from_wire_payload(request_payload)
            return pickle.dumps(
                self._browser_control_handlers(connection)[request.message_type]()
            )
        except Exception as exc:
            return pickle.dumps(
                {
                    "type": UI_BRIDGE_BROWSER_ERROR_TYPE,
                    "status": UI_BRIDGE_BROWSER_ERROR_TYPE,
                    "message": str(exc),
                }
            )

    def _browser_control_handlers(
        self,
        connection: ExecutionConnectionSpec,
    ) -> dict[UiBridgeBrowserControlMessageType, Callable[[], JsonObject]]:
        return {
            UiBridgeBrowserControlMessageType.PING: lambda: self._browser_pong(
                connection
            ),
        }

    def _browser_pong(self, connection: ExecutionConnectionSpec) -> JsonObject:
        if connection.port is None:
            raise ValueError("UI bridge browser pong requires a data port.")
        return {
            "type": UI_BRIDGE_BROWSER_PONG_TYPE,
            "port": connection.port,
            "control_port": connection.port + CONTROL_PORT_OFFSET,
            "server": UI_BRIDGE_BROWSER_SERVER_NAME,
            "ready": True,
            "log_file_path": self._current_log_file_path(),
            "schema_version": SCHEMA_VERSION,
            "bridge_protocol_version": UI_BRIDGE_PROTOCOL_VERSION,
            "bridge_instance_id": self._bridge_instance_id,
        }

    @staticmethod
    def _current_log_file_path() -> str | None:
        try:
            from openhcs.core.log_utils import get_current_log_file_path

            return get_current_log_file_path()
        except Exception:
            return None

    def __enter__(self) -> "UiBridgeControlServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.stop()
