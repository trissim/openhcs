"""Process boundary server for the OpenHCS UI agent bridge."""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
from python_introspect import project_dataclass
from zmqruntime.messages import (
    ControlErrorResponse,
    ControlMessageType,
    ControlRequestHeader,
    PongResponse,
    ServerRole,
)
from zmqruntime.transport import resolve_transport_mode

from openhcs.agent.dto.common import SCHEMA_VERSION, AgentError, JsonObject
from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.dto.ui_bridge import (
    UiActionCatalog,
    UiActionInvokeResult,
    UiBranchCatalog,
    UiBridgeConnectionSpec,
    UiBridgeDescriptorFile,
    UiBridgeDescriptorWirePayload,
    UiBridgeOperationRef,
    UiBridgeRequestEnvelope,
    UiBridgeResponseEnvelope,
    UiBridgeStatus,
    UiCodeDocument,
    UiCodeDocumentApplyResult,
    UiCodeDocumentCatalog,
    UiCodeDocumentValidationResult,
    UiObjectStateFieldHelpResult,
    UiObjectStateScopeCatalog,
    UiSelectedPlateWorkflowResult,
    UiSnapshotCatalog,
    UiSnapshotRestoreResult,
    UiStateSurfaceCatalog,
    UiStateSurfaceDocument,
    UiWidgetActionInvokeResult,
    UiWidgetTreeResult,
    UiWindowCatalog,
    UiWindowCloseResult,
    UiWindowFocusResult,
    UiWindowNavigateResult,
    UiWindowSnapshotResult,
)
from openhcs.agent.runtime_platform import AgentRuntimePlatformAuthority
from openhcs.agent.services.ui_bridge_service import (
    UI_BRIDGE_PROTOCOL_VERSION,
    UiBridgeOperationContract,
    UiBridgeOperationContractABC,
)
from openhcs.agent.services.ui_bridge_transport import (
    AgentDtoJsonCodec,
)
from openhcs.pyqt_gui.config import AgentUiBridgeConfig
from openhcs.pyqt_gui.services.ui_agent_bridge import (
    InProcessUiBridgeGateway,
    UiAgentBridgeService,
)
from openhcs.runtime.zmq_application import OPENHCS_ENDPOINT_APPLICATION
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG, OpenHCSZMQConfig
from openhcs.serialization.json import to_jsonable

DEFAULT_UI_BRIDGE_START_TIMEOUT_SECONDS = 5.0
UI_BRIDGE_BROWSER_SERVER_NAME = "OpenHCSUiBridgeServer"


@dataclass(frozen=True, slots=True, kw_only=True)
class UiBridgeBrowserPong(PongResponse):
    """UI-bridge specialization of the canonical server heartbeat."""

    schema_version: str
    bridge_protocol_version: str
    bridge_instance_id: str

    def to_dict(self) -> JsonObject:
        payload = PongResponse.to_dict(self)
        payload.update(
            {
                "schema_version": self.schema_version,
                "bridge_protocol_version": self.bridge_protocol_version,
                "bridge_instance_id": self.bridge_instance_id,
            }
        )
        return payload


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
    | UiSelectedPlateWorkflowResult
    | UiWindowCatalog
    | UiWindowCloseResult
    | UiWindowFocusResult
    | UiWindowNavigateResult
    | UiWindowSnapshotResult
    | UiWidgetTreeResult
    | UiWidgetActionInvokeResult
    | UiObjectStateFieldHelpResult
    | UiObjectStateScopeCatalog
    | UiSnapshotCatalog
    | UiSnapshotRestoreResult
    | UiBranchCatalog
    | UiBridgeOperationRef
)


@dataclass(frozen=True, slots=True)
class UiBridgeServerBinding(UiBridgeConnectionSpec):
    def __post_init__(self) -> None:
        missing_fields = tuple(
            field_name
            for field_name, value in (
                ("port", self.port),
                ("descriptor_file_path", self.descriptor_file_path),
                ("bridge_instance_id", self.bridge_instance_id),
                ("auth_token", self.auth_token),
            )
            if value is None
        )
        if missing_fields:
            raise ValueError(
                "UI bridge binding requires resolved fields: "
                + ", ".join(missing_fields)
            )

    @classmethod
    def from_runtime(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        bridge_instance_id: str,
        descriptor_file_path: Path,
        auth_token: str,
    ) -> "UiBridgeServerBinding":
        if connection.port is None:
            raise ValueError("UI bridge binding requires a resolved data port.")
        return project_dataclass(
            cls,
            connection,
            auth_token=auth_token,
            descriptor_file_path=str(descriptor_file_path),
            bridge_instance_id=bridge_instance_id,
        )

    @property
    def connection(self) -> ExecutionConnectionSpec:
        return self.public_connection()

    def token_bearing_connection(self) -> UiBridgeConnectionSpec:
        return self


@dataclass(slots=True)
class UiBridgeUnsupportedOperationError(LookupError):
    """Raised when an agent requests an operation this bridge does not expose."""

    operation_name: str

    def __str__(self) -> str:
        return f"Unsupported UI bridge operation: {self.operation_name}"


class UiBridgeServerInProcessGateway(InProcessUiBridgeGateway):
    """In-process gateway with server descriptor details in status responses."""

    def __init__(
        self,
        bridge: UiAgentBridgeService,
        *,
        binding_supplier: Callable[[], UiBridgeServerBinding],
    ) -> None:
        super().__init__(bridge)
        self._binding_supplier = binding_supplier

    def status(self, connection: UiBridgeConnectionSpec) -> UiBridgeStatus:
        binding = self._binding_supplier()
        return replace(
            super().status(connection),
            auth_required=True,
            bridge_instance_id=binding.bridge_instance_id,
            connection=binding.connection,
            descriptor_file_path=str(binding.descriptor_file_path),
            supported_operations=UiBridgeOperationContractABC.supported_operation_names(),
            provider_catalog_schema_versions=(SCHEMA_VERSION,),
            bridge_features=UiBridgeOperationContractABC.supported_bridge_features(),
        )


@dataclass(frozen=True, slots=True)
class UiBridgeRequestOperation:
    """Generic server adapter for a typed UI bridge operation contract."""

    contract: UiBridgeOperationContract

    @classmethod
    def for_name(cls, operation_name: str) -> "UiBridgeRequestOperation":
        try:
            return cls(UiBridgeOperationContractABC.for_name(operation_name))
        except KeyError as exc:
            raise UiBridgeUnsupportedOperationError(operation_name) from exc

    @classmethod
    def supported_operation_names(cls) -> tuple[str, ...]:
        return UiBridgeOperationContractABC.supported_operation_names()

    @property
    def requires_auth(self) -> bool:
        return self.contract.requires_auth

    def execute(
        self,
        dispatcher: "UiBridgeRequestDispatcher",
        request: UiBridgeRequestEnvelope,
    ) -> UiBridgeOperationDispatchResult:
        return self.contract.invoke_with_payload(
            dispatcher.gateway,
            dispatcher.bridge_connection,
            dispatcher.contract_payload(self.contract, request),
        )


class UiBridgeRequestExceptionClassifier(metaclass=AutoRegisterMeta):
    """Registered classifier for request-dispatch exception codes."""

    __registry_key__ = "exception_key"
    __skip_if_no_key__ = True

    exception_type: ClassVar[type[BaseException] | None] = None
    exception_key: ClassVar[str | None] = None
    error_code: ClassVar[str]

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.exception_type is not None and cls.exception_key is None:
            cls.exception_key = cls.exception_type_key(cls.exception_type)

    @staticmethod
    def exception_type_key(exception_type: type[BaseException]) -> str:
        return f"{exception_type.__module__}.{exception_type.__qualname__}"

    @classmethod
    def classifier_for_exception(
        cls,
        exception: Exception,
    ) -> type["UiBridgeRequestExceptionClassifier"] | None:
        for exception_type in type(exception).__mro__:
            classifier = cls.__registry__.get(cls.exception_type_key(exception_type))
            if classifier is not None:
                return classifier
        return None


class UiBridgePermissionErrorClassifier(UiBridgeRequestExceptionClassifier):
    exception_type = PermissionError
    error_code = "ui_bridge_auth_failed"


class UiBridgeUnsupportedOperationErrorClassifier(UiBridgeRequestExceptionClassifier):
    exception_type = UiBridgeUnsupportedOperationError
    error_code = "unsupported_ui_bridge_operation"


class UiBridgeValueErrorClassifier(UiBridgeRequestExceptionClassifier):
    exception_type = ValueError
    error_code = "invalid_ui_bridge_request"


class UiBridgeRequestErrorAuthority:
    """Classify request-dispatch exceptions into agent-facing error codes."""

    DEFAULT_ERROR_CODE = "ui_bridge_request_failed"

    @classmethod
    def agent_error(cls, exception: Exception) -> AgentError:
        classifier = UiBridgeRequestExceptionClassifier.classifier_for_exception(
            exception
        )
        if classifier is not None:
            return AgentError.from_exception(classifier.error_code, exception)
        return AgentError.from_exception(cls.DEFAULT_ERROR_CODE, exception)


class UiBridgeRequestDispatcher:
    """Route validated UI bridge envelopes to the in-process bridge service."""

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

    @property
    def gateway(self) -> UiBridgeServerInProcessGateway:
        return UiBridgeServerInProcessGateway(
            self._bridge,
            binding_supplier=self._binding_supplier,
        )

    @property
    def bridge_connection(self) -> UiBridgeConnectionSpec:
        return self._binding_supplier().connection

    def dispatch(self, payload: JsonObject) -> JsonObject:
        try:
            request = AgentDtoJsonCodec.dataclass_from_json(
                UiBridgeRequestEnvelope,
                payload,
            )
            self._validate_request_contract(request)
            operation = UiBridgeRequestOperation.for_name(request.operation)
            self._validate_auth(request, operation)
            result = operation.execute(self, request)
            response = UiBridgeResponseEnvelope(
                schema_version=SCHEMA_VERSION,
                bridge_protocol_version=UI_BRIDGE_PROTOCOL_VERSION,
                application=OPENHCS_ENDPOINT_APPLICATION,
                request_id=request.request_id,
                ok=True,
                payload=self._result_payload(result),
            )
        except Exception as exc:
            response = self._error_response(payload, exc)
        return self._response_payload(response)

    def _validate_request_contract(self, request: UiBridgeRequestEnvelope) -> None:
        if request.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported agent schema version: {request.schema_version}"
            )
        if request.bridge_protocol_version != UI_BRIDGE_PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported UI bridge protocol version: {request.bridge_protocol_version}"
            )
        OPENHCS_ENDPOINT_APPLICATION.compatibility_with(
            request.application
        ).require_match()

    def _validate_auth(
        self,
        request: UiBridgeRequestEnvelope,
        operation: UiBridgeRequestOperation,
    ) -> None:
        if operation.requires_auth and request.auth_token != self._auth_token:
            raise PermissionError("UI bridge auth token is missing or invalid.")

    @staticmethod
    def request_payload(target_type, request: UiBridgeRequestEnvelope):
        return AgentDtoJsonCodec.dataclass_from_json(target_type, request.payload)

    def contract_payload(
        self,
        contract: UiBridgeOperationContract,
        request: UiBridgeRequestEnvelope,
    ):
        return contract.decode_request_payload(
            request.payload,
            AgentDtoJsonCodec.dataclass_from_json,
        )

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
            raise TypeError(
                "UI bridge response envelope did not serialize to a JSON object."
            )
        return payload

    def _error_response(
        self,
        payload: JsonObject,
        exception: Exception,
    ) -> UiBridgeResponseEnvelope:
        return UiBridgeResponseEnvelope(
            schema_version=SCHEMA_VERSION,
            bridge_protocol_version=UI_BRIDGE_PROTOCOL_VERSION,
            application=OPENHCS_ENDPOINT_APPLICATION,
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
        config: AgentUiBridgeConfig | None = None,
        transport_config: OpenHCSZMQConfig = OPENHCS_ZMQ_CONFIG,
    ) -> None:
        self._bridge = bridge
        self._config = config or AgentUiBridgeConfig()
        self._transport_config = transport_config
        self._bridge_instance_id = self._config.resolve_bridge_instance_id()
        self._auth_token = self._config.resolve_auth_token()
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
    def config(self) -> AgentUiBridgeConfig:
        """Return the exact immutable configuration owned by this server."""

        return self._config

    @property
    def transport_config(self) -> OpenHCSZMQConfig:
        """Return the exact immutable transport configuration owned by this server."""

        return self._transport_config

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return (
            not self._stop_event.is_set()
            and thread is not None
            and thread.is_alive()
            and self._binding is not None
        )

    def matches_configuration(
        self,
        config: AgentUiBridgeConfig,
        transport_config: OpenHCSZMQConfig,
    ) -> bool:
        """Return whether this live server owns the exact declarations."""

        return (
            self.is_running
            and self._config == config
            and self._transport_config == transport_config
        )

    def start(
        self,
        *,
        timeout_seconds: float = DEFAULT_UI_BRIDGE_START_TIMEOUT_SECONDS,
    ) -> UiBridgeServerBinding:
        if self.is_running:
            return self.binding
        if self._stop_event.is_set():
            raise RuntimeError("A stopped UI bridge server cannot be restarted.")
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
            error = self._startup_error
            self.stop()
            raise RuntimeError("Failed to start UI bridge server.") from error
        return self.binding

    def stop(self) -> None:
        self._stop_event.set()
        self._bridge.close()
        thread = self._thread
        if thread is not None:
            thread.join(self._config.shutdown_timeout_seconds)
            if thread.is_alive():
                raise TimeoutError("Timed out waiting for UI bridge server to stop.")
        self._remove_descriptor_file(
            Path(self._binding.descriptor_file_path)
            if self._binding is not None
            else None
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
        connection: ExecutionConnectionSpec | None = None
        try:
            connection = self._bind(socket)
            self._bind_browser_control_socket(browser_control_socket, connection)
            descriptor_file_path = self._config.descriptor_path_for(
                self._bridge_instance_id
            )
            self._binding = UiBridgeServerBinding.from_runtime(
                connection=connection,
                bridge_instance_id=self._bridge_instance_id,
                descriptor_file_path=descriptor_file_path,
                auth_token=self._auth_token,
            )
            self._write_descriptor_file(
                UiBridgeDescriptorFile(
                    schema_version=SCHEMA_VERSION,
                    bridge_protocol_version=UI_BRIDGE_PROTOCOL_VERSION,
                    application=OPENHCS_ENDPOINT_APPLICATION,
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
            self._remove_descriptor_file(descriptor_file_path)
            browser_control_socket.close(linger=0)
            socket.close(linger=0)
            context.term()
            if connection is not None:
                connection.transport_endpoint().cleanup(self._transport_config)

    def _write_descriptor_file(self, descriptor: UiBridgeDescriptorFile) -> Path:
        path = AgentRuntimePlatformAuthority.resolved_path(
            descriptor.descriptor_file_path
        )
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        path.write_text(
            json.dumps(
                to_jsonable(
                    project_dataclass(UiBridgeDescriptorWirePayload, descriptor)
                ),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if AgentRuntimePlatformAuthority.current().supports_posix_permissions():
            path.chmod(0o600)
        return path

    @staticmethod
    def _remove_descriptor_file(path: Path | None) -> None:
        if path is None:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def _bind(self, socket) -> ExecutionConnectionSpec:
        requested_connection = self._config
        mode = resolve_transport_mode(requested_connection.transport_mode)
        port = mode.declaration.bind_socket(
            socket,
            requested_connection.host,
            requested_connection.port,
            self._transport_config,
        )
        return ExecutionConnectionSpec(
            host=requested_connection.host,
            port=port,
            transport_mode=mode,
            persistent=requested_connection.persistent,
        )

    def _bind_browser_control_socket(
        self,
        socket,
        connection: ExecutionConnectionSpec,
    ) -> None:
        socket.bind(connection.zmq_control_url(self._transport_config))

    def _browser_control_response_payload(
        self,
        request_payload: bytes,
        connection: ExecutionConnectionSpec,
    ) -> bytes:
        try:
            request = ControlRequestHeader.from_wire_payload(request_payload)
            if request.message_type is ControlMessageType.PING:
                return pickle.dumps(self._browser_pong(connection).to_dict())
            raise ValueError(
                f"Unsupported UI bridge browser control message: {request.message_type}"
            )
        except Exception as exc:
            return pickle.dumps(ControlErrorResponse.from_exception(exc).to_dict())

    def _browser_pong(
        self,
        connection: ExecutionConnectionSpec,
    ) -> UiBridgeBrowserPong:
        control_port = connection.zmq_control_port(self._transport_config)
        return UiBridgeBrowserPong(
            port=connection.require_port("UI bridge browser heartbeat"),
            control_port=control_port,
            ready=True,
            server=UI_BRIDGE_BROWSER_SERVER_NAME,
            server_role=ServerRole.GENERIC,
            log_file_path=self._current_log_file_path(),
            schema_version=SCHEMA_VERSION,
            bridge_protocol_version=UI_BRIDGE_PROTOCOL_VERSION,
            bridge_instance_id=self._bridge_instance_id,
            application=OPENHCS_ENDPOINT_APPLICATION,
        )

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
