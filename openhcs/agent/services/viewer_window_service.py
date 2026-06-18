"""Agent service for running viewer window interactions."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypeVar

import zmq
from zmqruntime.transport import get_control_url

from openhcs.agent.dto.common import (
    AgentError,
    AgentResourceRef,
    JsonObject,
    JsonValue,
    SCHEMA_VERSION,
)
from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.dto.viewer import (
    ViewerWindowDescriptor,
    ViewerWindowSnapshotRequest,
    ViewerWindowSnapshotResult,
    viewer_window_snapshot_error,
)
from openhcs.runtime.window_snapshot import (
    WindowSnapshotCaptureSpec,
    WindowSnapshotWirePayload,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG


OptionalViewerFieldT = TypeVar("OptionalViewerFieldT")


class ViewerControlMessageType:
    """Viewer control message names consumed by streaming viewer servers."""

    SCREENSHOT = "screenshot"


class ViewerControlField:
    """Viewer control payload fields."""

    TYPE = "type"
    SNAPSHOT = "snapshot"
    STATUS = "status"
    MESSAGE = "message"
    VIEWER = "viewer"
    RESOURCE = "resource"
    WIDTH = "width"
    HEIGHT = "height"


class ViewerDescriptorField:
    """Viewer descriptor payload fields."""

    TYPE = "type"
    TITLE = "title"


class ViewerWindowGatewayABC(ABC):
    """Transport boundary for interacting with running viewer windows."""

    @abstractmethod
    def snapshot_window(self, request: ViewerWindowSnapshotRequest) -> JsonObject:
        raise NotImplementedError


class ZMQViewerWindowGateway(ViewerWindowGatewayABC):
    """Viewer gateway backed by the existing ZMQ control socket."""

    def snapshot_window(self, request: ViewerWindowSnapshotRequest) -> JsonObject:
        message = {
            ViewerControlField.TYPE: ViewerControlMessageType.SCREENSHOT,
            ViewerControlField.SNAPSHOT: request.snapshot.to_wire_payload().as_dict(),
        }
        return self._send_control_message(request, message)

    def _send_control_message(
        self,
        request: ViewerWindowSnapshotRequest,
        message: JsonObject,
    ) -> JsonObject:
        connection = request.connection
        if connection.port is None:
            raise ValueError("Viewer control request requires an explicit viewer data port.")
        control_url = get_control_url(
            connection.port,
            connection.transport_mode,
            host=connection.host,
            config=OPENHCS_ZMQ_CONFIG,
        )
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, request.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, request.timeout_ms)
        try:
            socket.connect(control_url)
            socket.send(pickle.dumps(message))
            response = pickle.loads(socket.recv())
        finally:
            socket.close(linger=0)
            context.term()
        if not isinstance(response, Mapping):
            raise TypeError(
                f"Viewer control response must be a mapping, got {type(response).__name__}."
            )
        return dict(response)


class ViewerWindowService:
    """Expose running viewer screenshots as bounded agent resources."""

    SUCCESS_STATUS = "success"

    def __init__(self, gateway: ViewerWindowGatewayABC | None = None) -> None:
        if gateway is None:
            self._gateway = ZMQViewerWindowGateway()
        else:
            self._gateway = gateway

    def snapshot_window(
        self,
        *,
        port: int,
        snapshot: WindowSnapshotCaptureSpec,
        host: str = "localhost",
        transport_mode: str | None = None,
        timeout_ms: int = 5000,
    ) -> ViewerWindowSnapshotResult:
        connection = ExecutionConnectionSpec(
            host=host,
            port=port,
            transport_mode=transport_mode,
        )
        request = ViewerWindowSnapshotRequest(
            connection=connection,
            snapshot=snapshot,
            timeout_ms=timeout_ms,
        )
        try:
            response = self._gateway.snapshot_window(request)
        except Exception as exc:
            return viewer_window_snapshot_error(
                connection=connection,
                error=AgentError.from_exception("viewer_window_snapshot_failed", exc),
            )

        try:
            return self._snapshot_result_from_response(
                connection=connection,
                request=request,
                response=response,
            )
        except Exception as exc:
            return viewer_window_snapshot_error(
                connection=connection,
                error=AgentError.from_exception(
                    "viewer_window_snapshot_response_invalid",
                    exc,
                ),
            )

    def _snapshot_result_from_response(
        self,
        *,
        connection: ExecutionConnectionSpec,
        request: ViewerWindowSnapshotRequest,
        response: JsonObject,
    ) -> ViewerWindowSnapshotResult:
        status = self._required_str(response, ViewerControlField.STATUS)
        if status != self.SUCCESS_STATUS:
            message = self._required_str(response, ViewerControlField.MESSAGE)
            return viewer_window_snapshot_error(
                connection=connection,
                error=AgentError(
                    code="viewer_window_snapshot_failed",
                    message=message,
                ),
            )
        response_snapshot = WindowSnapshotCaptureSpec.from_wire_payload(
            WindowSnapshotWirePayload(
                self._required_str_mapping(response, ViewerControlField.SNAPSHOT)
            )
        )
        if response_snapshot != request.snapshot:
            return viewer_window_snapshot_error(
                connection=connection,
                error=AgentError(
                    code="viewer_window_snapshot_contract_mismatch",
                    message=(
                        "Viewer screenshot response snapshot contract did not match "
                        "the request snapshot contract."
                    ),
                ),
            )

        viewer_payload = self._required_mapping(response, ViewerControlField.VIEWER)
        resource_payload = self._required_mapping(response, ViewerControlField.RESOURCE)
        return ViewerWindowSnapshotResult(
            schema_version=SCHEMA_VERSION,
            connection=connection,
            captured=True,
            resource=AgentResourceRef(
                uri=self._required_str(resource_payload, "uri"),
                title=self._required_str(resource_payload, "title"),
                mime_type=self._required_str(resource_payload, "mime_type"),
                path=self._optional_typed(resource_payload, "path", str),
                size_bytes=self._optional_typed(resource_payload, "size_bytes", int),
                sha256=self._optional_typed(resource_payload, "sha256", str),
            ),
            viewer=ViewerWindowDescriptor(
                viewer_type=self._required_str(
                    viewer_payload,
                    ViewerDescriptorField.TYPE,
                ),
                title=self._required_str(
                    viewer_payload,
                    ViewerDescriptorField.TITLE,
                ),
            ),
            width=self._optional_typed(response, ViewerControlField.WIDTH, int),
            height=self._optional_typed(response, ViewerControlField.HEIGHT, int),
            snapshot=request.snapshot,
            response=response,
        )

    @staticmethod
    def _required_mapping(payload: Mapping[str, JsonValue], field_name: str) -> JsonObject:
        if field_name not in payload:
            raise KeyError(f"Viewer response missing required field {field_name!r}.")
        value = payload[field_name]
        if not isinstance(value, Mapping):
            raise TypeError(f"Viewer response field {field_name!r} must be a mapping.")
        return dict(value)

    @staticmethod
    def _required_str_mapping(
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> Mapping[str, str]:
        mapping = ViewerWindowService._required_mapping(payload, field_name)
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Viewer response field {field_name!r} must use string keys."
                )
            if not isinstance(value, str):
                raise TypeError(
                    f"Viewer response field {field_name!r} values must be strings."
                )
        return mapping

    @staticmethod
    def _required_str(payload: Mapping[str, JsonValue], field_name: str) -> str:
        if field_name not in payload:
            raise KeyError(f"Viewer response missing required field {field_name!r}.")
        value = payload[field_name]
        if not isinstance(value, str):
            raise TypeError(f"Viewer response field {field_name!r} must be a string.")
        return value

    @staticmethod
    def _optional_typed(
        payload: Mapping[str, JsonValue],
        field_name: str,
        expected_type: type[OptionalViewerFieldT],
    ) -> OptionalViewerFieldT | None:
        if field_name not in payload:
            return None
        value = payload[field_name]
        if value is None:
            return None
        if not isinstance(value, expected_type):
            type_name = expected_type.__name__
            raise TypeError(
                f"Viewer response field {field_name!r} must be a {type_name}."
            )
        return value
