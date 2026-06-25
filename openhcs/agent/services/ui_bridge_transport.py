"""ZMQ transport for the OpenHCS running-UI bridge."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from types import NoneType, UnionType
from typing import TypeVar, get_args, get_origin, get_type_hints

from typing_extensions import TypeForm

from openhcs.agent.dto.common import JsonObject, JsonValue, SCHEMA_VERSION
from openhcs.agent.dto.ui_bridge import (
    UiActionCatalog,
    UiActionInvokeRequest,
    UiActionInvokeResult,
    UiBranchCatalog,
    UiBranchSwitchRequest,
    UiBridgeConnectionSpec,
    UiBridgeOperationRef,
    UiBridgeOperationStatusRequest,
    UiBridgeRequestEnvelope,
    UiBridgeResponseEnvelope,
    UiBridgeStatus,
    UiCodeDocument,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentApplyResult,
    UiCodeDocumentCatalog,
    UiCodeDocumentRequest,
    UiCodeDocumentValidationRequest,
    UiCodeDocumentValidationResult,
    UiObjectStateScopeCatalog,
    UiObjectStateScopeListRequest,
    UiSelectedPlateWorkflowRequest,
    UiSelectedPlateWorkflowResult,
    UiStateSurfaceCatalog,
    UiStateSurfaceDocument,
    UiStateSurfaceRequest,
    UiSnapshotCatalog,
    UiSnapshotListRequest,
    UiSnapshotRestoreRequest,
    UiSnapshotRestoreResult,
    UiTimeTravelHeadRequest,
    UiWindowCatalog,
    UiWindowCloseRequest,
    UiWindowCloseResult,
    UiWindowFocusRequest,
    UiWindowFocusResult,
    UiWindowNavigateRequest,
    UiWindowNavigateResult,
    UiWindowSnapshotRequest,
    UiWindowSnapshotResult,
    UiWidgetTreeRequest,
    UiWidgetTreeResult,
)
from openhcs.agent.serialization import to_jsonable
from openhcs.agent.services.ui_bridge_service import (
    DEFAULT_UI_BRIDGE_TIMEOUT_MS,
    UI_BRIDGE_PROTOCOL_VERSION,
    UiBridgeGatewayABC,
    UiBridgeGatewayResponseError,
    UiBridgeGatewayTimeoutError,
    UiBridgeGatewayUnavailableError,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG


DtoT = TypeVar("DtoT")
UI_BRIDGE_CONTROL_TIMEOUT_MAX_MS = DEFAULT_UI_BRIDGE_TIMEOUT_MS
UI_BRIDGE_CONTROL_TIMEOUT_MIN_MS = 1


class UiBridgeOperationName(str, Enum):
    STATUS = "status"
    LIST_DOCUMENTS = "list_documents"
    GET_DOCUMENT = "get_document"
    LIST_STATE_SURFACES = "list_state_surfaces"
    GET_STATE_SURFACE = "get_state_surface"
    LIST_ACTIONS = "list_actions"
    INVOKE_ACTION = "invoke_action"
    LIST_WINDOWS = "list_windows"
    FOCUS_WINDOW = "focus_window"
    NAVIGATE_WINDOW = "navigate_window"
    CLOSE_WINDOW = "close_window"
    SNAPSHOT_WINDOW = "snapshot_window"
    WIDGET_TREE = "widget_tree"
    LIST_OBJECT_STATE_SCOPES = "list_object_state_scopes"
    VALIDATE_DOCUMENT = "validate_document"
    APPLY_DOCUMENT = "apply_document"
    LIST_SNAPSHOTS = "list_snapshots"
    RESTORE_SNAPSHOT = "restore_snapshot"
    TIME_TRAVEL_HEAD = "time_travel_head"
    LIST_BRANCHES = "list_branches"
    SWITCH_BRANCH = "switch_branch"
    GET_OPERATION_STATUS = "get_operation_status"
    SELECTED_PLATE_WORKFLOW = "selected_plate_workflow"


UiBridgeOperationRequestPayload = (
    UiCodeDocumentRequest
    | UiStateSurfaceRequest
    | UiActionInvokeRequest
    | UiWindowFocusRequest
    | UiWindowNavigateRequest
    | UiWindowCloseRequest
    | UiWindowSnapshotRequest
    | UiWidgetTreeRequest
    | UiObjectStateScopeListRequest
    | UiCodeDocumentValidationRequest
    | UiCodeDocumentApplyRequest
    | UiSnapshotListRequest
    | UiSnapshotRestoreRequest
    | UiTimeTravelHeadRequest
    | UiBranchSwitchRequest
    | UiBridgeOperationStatusRequest
    | UiSelectedPlateWorkflowRequest
    | None
)


class AgentDtoJsonCodec:
    """Typed JSON hydration for OpenHCS agent dataclasses."""

    @classmethod
    def dataclass_from_json(
        cls,
        target_type: type[DtoT],
        payload: JsonObject,
    ) -> DtoT:
        if not is_dataclass(target_type):
            raise TypeError(f"Target is not a dataclass: {target_type!r}")
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"Expected JSON object for {target_type.__name__}, "
                f"got {type(payload).__name__}"
            )
        type_hints = get_type_hints(target_type)
        kwargs = {}
        for field in fields(target_type):
            if field.name in payload:
                if field.name in type_hints:
                    annotation = type_hints[field.name]
                else:
                    annotation = field.type
                kwargs[field.name] = cls.coerce(
                    annotation,
                    payload[field.name],
                )
                continue
            if field.default is not MISSING or field.default_factory is not MISSING:
                continue
            raise KeyError(f"Missing required field {field.name!r} for {target_type.__name__}")
        return target_type(**kwargs)

    @classmethod
    def coerce(cls, annotation: TypeForm, value: JsonValue):
        if value is None:
            return None

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin in (UnionType,):
            return cls._coerce_union(args, value)
        if origin is None and isinstance(annotation, UnionType):
            return cls._coerce_union(args, value)
        if str(origin) == "typing.Union":
            return cls._coerce_union(args, value)
        if origin is tuple:
            return cls._coerce_tuple(args, value)
        if origin in (list, Sequence):
            element_type = args[0] if args else JsonValue
            return [cls.coerce(element_type, item) for item in cls._sequence(value)]
        if origin in (dict, Mapping):
            if not isinstance(value, Mapping):
                raise TypeError(f"Expected JSON object, got {type(value).__name__}")
            return dict(value)
        if is_dataclass(annotation):
            return cls.dataclass_from_json(annotation, value)
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return annotation(value)
        return value

    @classmethod
    def _coerce_union(cls, args: tuple[TypeForm, ...], value: JsonValue):
        errors: list[Exception] = []
        for candidate in args:
            if candidate is NoneType:
                continue
            try:
                return cls.coerce(candidate, value)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[-1]
        return value

    @classmethod
    def _coerce_tuple(cls, args: tuple[TypeForm, ...], value: JsonValue):
        sequence = cls._sequence(value)
        if not args:
            return tuple(sequence)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(cls.coerce(args[0], item) for item in sequence)
        return tuple(
            cls.coerce(item_type, item)
            for item_type, item in zip(args, sequence, strict=False)
        )

    @staticmethod
    def _sequence(value: JsonValue) -> Sequence[JsonValue]:
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise TypeError(f"Expected JSON array, got {type(value).__name__}")
        return value

class UiBridgeControlClient:
    """Small JSON/ZMQ client for the UI bridge control socket."""

    def request(
        self,
        connection: UiBridgeConnectionSpec,
        operation: UiBridgeOperationName,
        payload: UiBridgeOperationRequestPayload = None,
    ) -> JsonObject:
        if connection.port is None:
            raise UiBridgeGatewayUnavailableError

        request = UiBridgeRequestEnvelope(
            schema_version=SCHEMA_VERSION,
            bridge_protocol_version=UI_BRIDGE_PROTOCOL_VERSION,
            request_id=str(uuid.uuid4()),
            operation=operation.value,
            auth_token=connection.auth_token,
            payload=self._payload_object(payload),
        )
        response_payload = self._send(connection, to_jsonable(request))
        response = AgentDtoJsonCodec.dataclass_from_json(
            UiBridgeResponseEnvelope,
            response_payload,
        )
        self._validate_response(request, response)
        if not response.ok:
            raise UiBridgeGatewayResponseError(response.errors)
        return response.payload

    def _send(
        self,
        connection: UiBridgeConnectionSpec,
        request_payload: JsonObject,
    ) -> JsonObject:
        import zmq

        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        timeout_ms = self._socket_timeout_ms(connection)
        request_operation = self._request_operation(request_payload)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        try:
            socket.connect(connection.zmq_data_url(OPENHCS_ZMQ_CONFIG))
            socket.send_json(request_payload)
            response = socket.recv_json()
        except zmq.Again as exc:
            raise UiBridgeGatewayTimeoutError(
                operation=request_operation,
                timeout_ms=timeout_ms,
            ) from exc
        finally:
            socket.close(linger=0)
        if not isinstance(response, Mapping):
            raise TypeError(f"UI bridge response must be a JSON object, got {type(response).__name__}")
        return dict(response)

    @staticmethod
    def _socket_timeout_ms(connection: UiBridgeConnectionSpec) -> int:
        return min(
            max(connection.timeout_ms, UI_BRIDGE_CONTROL_TIMEOUT_MIN_MS),
            UI_BRIDGE_CONTROL_TIMEOUT_MAX_MS,
        )

    @staticmethod
    def _request_operation(request_payload: JsonObject) -> str:
        if "operation" not in request_payload:
            raise ValueError("UI bridge request payload missing required field 'operation'.")
        operation = request_payload["operation"]
        if not isinstance(operation, str):
            raise TypeError("UI bridge request payload field 'operation' must be a string.")
        return operation

    @staticmethod
    def _payload_object(payload: UiBridgeOperationRequestPayload) -> JsonObject:
        if payload is None:
            return {}
        json_payload = to_jsonable(payload)
        if not isinstance(json_payload, Mapping):
            raise TypeError(
                f"UI bridge request payload must serialize to a JSON object, "
                f"got {type(json_payload).__name__}"
            )
        return json_payload

    @staticmethod
    def _validate_response(
        request: UiBridgeRequestEnvelope,
        response: UiBridgeResponseEnvelope,
    ) -> None:
        if response.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported agent schema version: {response.schema_version}")
        if response.bridge_protocol_version != UI_BRIDGE_PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported UI bridge protocol version: {response.bridge_protocol_version}"
            )
        if response.request_id != request.request_id:
            raise ValueError("UI bridge response request_id does not match request.")


class ZMQUiBridgeGateway(UiBridgeGatewayABC):
    """Gateway that connects MCP/agent services to a running PyQt UI bridge."""

    registry_key = "zmq"

    def __init__(self, client: UiBridgeControlClient | None = None) -> None:
        if client is None:
            client = UiBridgeControlClient()
        self._client = client

    def status(self, connection: UiBridgeConnectionSpec) -> UiBridgeStatus:
        payload = self._client.request(connection, UiBridgeOperationName.STATUS)
        return AgentDtoJsonCodec.dataclass_from_json(UiBridgeStatus, payload)

    def list_documents(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiCodeDocumentCatalog:
        payload = self._client.request(connection, UiBridgeOperationName.LIST_DOCUMENTS)
        return AgentDtoJsonCodec.dataclass_from_json(UiCodeDocumentCatalog, payload)

    def list_state_surfaces(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiStateSurfaceCatalog:
        payload = self._client.request(connection, UiBridgeOperationName.LIST_STATE_SURFACES)
        return AgentDtoJsonCodec.dataclass_from_json(UiStateSurfaceCatalog, payload)

    def list_actions(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiActionCatalog:
        payload = self._client.request(connection, UiBridgeOperationName.LIST_ACTIONS)
        return AgentDtoJsonCodec.dataclass_from_json(UiActionCatalog, payload)

    def list_windows(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiWindowCatalog:
        payload = self._client.request(connection, UiBridgeOperationName.LIST_WINDOWS)
        return AgentDtoJsonCodec.dataclass_from_json(UiWindowCatalog, payload)

    def list_object_state_scopes(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiObjectStateScopeListRequest,
    ) -> UiObjectStateScopeCatalog:
        payload = self._client.request(
            connection,
            UiBridgeOperationName.LIST_OBJECT_STATE_SCOPES,
            request,
        )
        return AgentDtoJsonCodec.dataclass_from_json(UiObjectStateScopeCatalog, payload)

    def get_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentRequest,
    ) -> UiCodeDocument:
        payload = self._client.request(connection, UiBridgeOperationName.GET_DOCUMENT, request)
        return AgentDtoJsonCodec.dataclass_from_json(UiCodeDocument, payload)

    def get_state_surface(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiStateSurfaceRequest,
    ) -> UiStateSurfaceDocument:
        payload = self._client.request(
            connection,
            UiBridgeOperationName.GET_STATE_SURFACE,
            request,
        )
        return AgentDtoJsonCodec.dataclass_from_json(UiStateSurfaceDocument, payload)

    def invoke_action(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiActionInvokeRequest,
    ) -> UiActionInvokeResult:
        payload = self._client.request(
            connection,
            UiBridgeOperationName.INVOKE_ACTION,
            request,
        )
        return AgentDtoJsonCodec.dataclass_from_json(UiActionInvokeResult, payload)

    def selected_plate_workflow(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSelectedPlateWorkflowRequest,
    ) -> UiSelectedPlateWorkflowResult:
        payload = self._client.request(
            connection,
            UiBridgeOperationName.SELECTED_PLATE_WORKFLOW,
            request,
        )
        return AgentDtoJsonCodec.dataclass_from_json(
            UiSelectedPlateWorkflowResult,
            payload,
        )

    def focus_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowFocusRequest,
    ) -> UiWindowFocusResult:
        payload = self._client.request(
            connection,
            UiBridgeOperationName.FOCUS_WINDOW,
            request,
        )
        return AgentDtoJsonCodec.dataclass_from_json(UiWindowFocusResult, payload)

    def navigate_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowNavigateRequest,
    ) -> UiWindowNavigateResult:
        payload = self._client.request(
            connection,
            UiBridgeOperationName.NAVIGATE_WINDOW,
            request,
        )
        return AgentDtoJsonCodec.dataclass_from_json(UiWindowNavigateResult, payload)

    def close_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowCloseRequest,
    ) -> UiWindowCloseResult:
        payload = self._client.request(
            connection,
            UiBridgeOperationName.CLOSE_WINDOW,
            request,
        )
        return AgentDtoJsonCodec.dataclass_from_json(UiWindowCloseResult, payload)

    def snapshot_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowSnapshotRequest,
    ) -> UiWindowSnapshotResult:
        payload = self._client.request(
            connection,
            UiBridgeOperationName.SNAPSHOT_WINDOW,
            request,
        )
        return AgentDtoJsonCodec.dataclass_from_json(UiWindowSnapshotResult, payload)

    def widget_tree(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWidgetTreeRequest,
    ) -> UiWidgetTreeResult:
        payload = self._client.request(
            connection,
            UiBridgeOperationName.WIDGET_TREE,
            request,
        )
        return AgentDtoJsonCodec.dataclass_from_json(UiWidgetTreeResult, payload)

    def validate_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentValidationRequest,
    ) -> UiCodeDocumentValidationResult:
        payload = self._client.request(
            connection,
            UiBridgeOperationName.VALIDATE_DOCUMENT,
            request,
        )
        return AgentDtoJsonCodec.dataclass_from_json(
            UiCodeDocumentValidationResult,
            payload,
        )

    def apply_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentApplyRequest,
    ) -> UiCodeDocumentApplyResult:
        payload = self._client.request(connection, UiBridgeOperationName.APPLY_DOCUMENT, request)
        return AgentDtoJsonCodec.dataclass_from_json(UiCodeDocumentApplyResult, payload)

    def list_snapshots(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSnapshotListRequest,
    ) -> UiSnapshotCatalog:
        payload = self._client.request(connection, UiBridgeOperationName.LIST_SNAPSHOTS, request)
        return AgentDtoJsonCodec.dataclass_from_json(UiSnapshotCatalog, payload)

    def restore_snapshot(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSnapshotRestoreRequest,
    ) -> UiSnapshotRestoreResult:
        payload = self._client.request(
            connection,
            UiBridgeOperationName.RESTORE_SNAPSHOT,
            request,
        )
        return AgentDtoJsonCodec.dataclass_from_json(UiSnapshotRestoreResult, payload)

    def time_travel_head(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiTimeTravelHeadRequest,
    ) -> UiSnapshotRestoreResult:
        payload = self._client.request(
            connection,
            UiBridgeOperationName.TIME_TRAVEL_HEAD,
            request,
        )
        return AgentDtoJsonCodec.dataclass_from_json(UiSnapshotRestoreResult, payload)

    def list_branches(self, connection: UiBridgeConnectionSpec) -> UiBranchCatalog:
        payload = self._client.request(connection, UiBridgeOperationName.LIST_BRANCHES)
        return AgentDtoJsonCodec.dataclass_from_json(UiBranchCatalog, payload)

    def switch_branch(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiBranchSwitchRequest,
    ) -> UiSnapshotRestoreResult:
        payload = self._client.request(connection, UiBridgeOperationName.SWITCH_BRANCH, request)
        return AgentDtoJsonCodec.dataclass_from_json(UiSnapshotRestoreResult, payload)

    def get_operation_status(
        self,
        connection: UiBridgeConnectionSpec,
        operation_id: str,
    ) -> UiBridgeOperationRef:
        payload = self._client.request(
            connection,
            UiBridgeOperationName.GET_OPERATION_STATUS,
            UiBridgeOperationStatusRequest(operation_id=operation_id),
        )
        return AgentDtoJsonCodec.dataclass_from_json(UiBridgeOperationRef, payload)
