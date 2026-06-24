"""Agent service for running viewer window interactions."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TypeVar

import zmq

from openhcs.agent.dto.common import (
    AgentError,
    AgentResourceRef,
    AgentWarning,
    JsonObject,
    JsonValue,
    SCHEMA_VERSION,
)
from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.dto.viewer import (
    ViewerWindowLayerValidationSummary,
    ViewerWindowLayerState,
    ViewerWindowDescriptor,
    ViewerWindowProbeResult,
    ViewerWindowSnapshotRequest,
    ViewerWindowSnapshotResult,
    ViewerWindowStateRequest,
    ViewerWindowStateResult,
    ViewerWindowValidationRequest,
    ViewerWindowValidationSummaryResult,
    viewer_window_snapshot_error,
    viewer_window_probe_from_state,
    viewer_window_state_error,
    viewer_window_validation_error,
)
from openhcs.runtime.window_snapshot import (
    WindowSnapshotCaptureSpec,
    WindowSnapshotWirePayload,
)
from openhcs.runtime.viewer_protocol import ViewerControlMessageType
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG


OptionalViewerFieldT = TypeVar("OptionalViewerFieldT")


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
    LAYERS = "layers"
    LAYER_COUNT = "layer_count"
    ACTIVE_DIMENSION_LABEL_ROUTE = "active_dimension_label_route"
    VIEWER_NDIM = "viewer_ndim"
    CURRENT_STEP = "current_step"
    AXIS_LABELS = "axis_labels"
    COMPONENT_GROUP_COUNT = "component_group_count"
    COMPONENT_ITEM_COUNT = "component_item_count"


class ViewerLayerField:
    """Viewer layer-state payload fields."""

    ROUTE_KEY = "route_key"
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
    AXIS_COMPONENT_VALUES = "axis_component_values"
    ROUTED_COMPONENT_VALUES = "routed_component_values"
    DATA_SHAPE = "data_shape"
    TRANSLATE = "translate"
    VISIBLE = "visible"
    SELECTED = "selected"
    PENDING_UPDATE = "pending_update"


class ViewerValidationWarningCode:
    """Warning codes emitted by viewer state validation."""

    LAYER_COUNT_MISMATCH = "viewer_layer_count_mismatch"
    LAYER_UNMOUNTED = "viewer_layer_unmounted"
    LAYER_PENDING_UPDATE = "viewer_layer_pending_update"
    REQUIRED_AXIS_LABELS_MISSING = "viewer_required_axis_labels_missing"
    PAYLOAD_NONZERO_METADATA_MISSING = "viewer_payload_nonzero_metadata_missing"
    PAYLOADS_ZERO = "viewer_payloads_zero"
    COORDINATE_GAPS = "viewer_layer_coordinate_gaps"


class ViewerDescriptorField:
    """Viewer descriptor payload fields."""

    TYPE = "type"
    TITLE = "title"


class ViewerWindowGatewayABC(ABC):
    """Transport boundary for interacting with running viewer windows."""

    @abstractmethod
    def snapshot_window(self, request: ViewerWindowSnapshotRequest) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def window_state(self, request: ViewerWindowStateRequest) -> JsonObject:
        raise NotImplementedError


class ZMQViewerWindowGateway(ViewerWindowGatewayABC):
    """Viewer gateway backed by the existing ZMQ control socket."""

    def __init__(self, context_factory: Callable[[], zmq.Context] = zmq.Context) -> None:
        self._context_factory = context_factory

    def snapshot_window(self, request: ViewerWindowSnapshotRequest) -> JsonObject:
        message = {
            ViewerControlField.TYPE: ViewerControlMessageType.SCREENSHOT.value,
            ViewerControlField.SNAPSHOT: request.snapshot.to_wire_payload().as_dict(),
        }
        return self._send_control_message(request, message)

    def window_state(self, request: ViewerWindowStateRequest) -> JsonObject:
        message = {
            ViewerControlField.TYPE: ViewerControlMessageType.STATE.value,
        }
        return self._send_control_message(request, message)

    def _send_control_message(
        self,
        request: ViewerWindowSnapshotRequest | ViewerWindowStateRequest,
        message: JsonObject,
    ) -> JsonObject:
        connection = request.connection
        control_url = connection.zmq_control_url(OPENHCS_ZMQ_CONFIG)
        context = self._context_factory()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, request.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, request.timeout_ms)
        poller = zmq.Poller()
        try:
            socket.connect(control_url)
            socket.send(pickle.dumps(message), flags=zmq.DONTWAIT)
            poller.register(socket, zmq.POLLIN)
            events = dict(poller.poll(request.timeout_ms))
            if events.get(socket) != zmq.POLLIN:
                raise TimeoutError(
                    "Viewer control request timed out after "
                    f"{request.timeout_ms}ms waiting for {control_url}."
                )
            response = pickle.loads(socket.recv(flags=zmq.DONTWAIT))
        finally:
            socket.close(linger=0)
            context.destroy(linger=0)
        if not isinstance(response, Mapping):
            raise TypeError(
                f"Viewer control response must be a mapping, got {type(response).__name__}."
            )
        return dict(response)


class ViewerWindowService:
    """Expose running viewer windows through bounded agent resources."""

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

    def window_state(
        self,
        *,
        port: int,
        host: str = "localhost",
        transport_mode: str | None = None,
        timeout_ms: int = 5000,
    ) -> ViewerWindowStateResult:
        connection = ExecutionConnectionSpec(
            host=host,
            port=port,
            transport_mode=transport_mode,
        )
        request = ViewerWindowStateRequest(
            connection=connection,
            timeout_ms=timeout_ms,
        )
        try:
            response = self._gateway.window_state(request)
        except Exception as exc:
            return viewer_window_state_error(
                connection=connection,
                error=AgentError.from_exception("viewer_window_state_failed", exc),
            )

        try:
            return self._state_result_from_response(
                connection=connection,
                response=response,
            )
        except Exception as exc:
            return viewer_window_state_error(
                connection=connection,
                error=AgentError.from_exception(
                    "viewer_window_state_response_invalid",
                    exc,
                ),
        )

    def probe_window(
        self,
        *,
        port: int,
        host: str = "localhost",
        transport_mode: str | None = None,
        timeout_ms: int = 5000,
    ) -> ViewerWindowProbeResult:
        state = self.window_state(
            port=port,
            host=host,
            transport_mode=transport_mode,
            timeout_ms=timeout_ms,
        )
        return viewer_window_probe_from_state(state)

    def validation_summary(
        self,
        *,
        port: int,
        host: str = "localhost",
        transport_mode: str | None = None,
        timeout_ms: int = 5000,
        expected_layer_count: int | None = None,
        required_axis_labels: tuple[str, ...] = (),
        require_nonzero_payloads: bool = True,
    ) -> ViewerWindowValidationSummaryResult:
        connection = ExecutionConnectionSpec(
            host=host,
            port=port,
            transport_mode=transport_mode,
        )
        request = ViewerWindowValidationRequest(
            connection=connection,
            timeout_ms=timeout_ms,
            expected_layer_count=expected_layer_count,
            required_axis_labels=required_axis_labels,
            require_nonzero_payloads=require_nonzero_payloads,
        )
        state = self.window_state(
            port=port,
            host=host,
            transport_mode=transport_mode,
            timeout_ms=timeout_ms,
        )
        if not state.observed:
            if state.errors:
                error = state.errors[0]
            else:
                error = AgentError(
                    code="viewer_window_state_unobserved",
                    message="Viewer window state was not observed.",
                )
            return viewer_window_validation_error(
                connection=connection,
                error=error,
                expected_layer_count=request.expected_layer_count,
                required_axis_labels=request.required_axis_labels,
                require_nonzero_payloads=request.require_nonzero_payloads,
            )

        layer_summaries = tuple(
            self._layer_validation_summary(
                layer,
                required_axis_labels=request.required_axis_labels,
                require_nonzero_payloads=request.require_nonzero_payloads,
            )
            for layer in state.layers
        )
        payload_count = sum(layer.payload_count for layer in layer_summaries)
        nonzero_payload_count = sum(
            layer.nonzero_payload_count for layer in layer_summaries
        )
        zero_payload_count = sum(layer.zero_payload_count for layer in layer_summaries)
        missing_nonzero_count = sum(
            layer.missing_nonzero_count for layer in layer_summaries
        )
        layer_count_matches = (
            request.expected_layer_count is None
            or state.layer_count == request.expected_layer_count
        )
        valid = layer_count_matches and all(layer.valid for layer in layer_summaries)
        warnings = self._validation_warnings(
            expected_layer_count=request.expected_layer_count,
            observed_layer_count=state.layer_count,
            layer_summaries=layer_summaries,
            require_nonzero_payloads=request.require_nonzero_payloads,
        )
        return ViewerWindowValidationSummaryResult(
            schema_version=SCHEMA_VERSION,
            connection=connection,
            valid=valid,
            warnings=warnings,
            observed=True,
            viewer=state.viewer,
            layer_count=state.layer_count,
            mounted_layer_count=sum(1 for layer in state.layers if layer.mounted),
            pending_update_count=sum(
                1 for layer in state.layers if layer.pending_update
            ),
            payload_count=payload_count,
            nonzero_payload_count=nonzero_payload_count,
            zero_payload_count=zero_payload_count,
            missing_nonzero_count=missing_nonzero_count,
            expected_layer_count=request.expected_layer_count,
            required_axis_labels=request.required_axis_labels,
            require_nonzero_payloads=request.require_nonzero_payloads,
            layer_summaries=layer_summaries,
            state=state,
        )

    @staticmethod
    def _validation_warnings(
        *,
        expected_layer_count: int | None,
        observed_layer_count: int,
        layer_summaries: tuple[ViewerWindowLayerValidationSummary, ...],
        require_nonzero_payloads: bool,
    ) -> tuple[AgentWarning, ...]:
        warnings: list[AgentWarning] = []
        if (
            expected_layer_count is not None
            and expected_layer_count != observed_layer_count
        ):
            warnings.append(
                AgentWarning(
                    code=ViewerValidationWarningCode.LAYER_COUNT_MISMATCH,
                    message=(
                        "Viewer layer count did not match the expected layer count: "
                        f"expected {expected_layer_count}, observed {observed_layer_count}."
                    ),
                )
            )

        for layer in layer_summaries:
            layer_name = layer.title or layer.route_key
            if not layer.mounted:
                warnings.append(
                    AgentWarning(
                        code=ViewerValidationWarningCode.LAYER_UNMOUNTED,
                        message=f"Viewer layer {layer_name!r} is not mounted.",
                    )
                )
            if layer.pending_update:
                warnings.append(
                    AgentWarning(
                        code=ViewerValidationWarningCode.LAYER_PENDING_UPDATE,
                        message=(
                            f"Viewer layer {layer_name!r} still has a pending update."
                        ),
                    )
                )
            if layer.missing_required_axis_labels:
                missing = ", ".join(layer.missing_required_axis_labels)
                warnings.append(
                    AgentWarning(
                        code=ViewerValidationWarningCode.REQUIRED_AXIS_LABELS_MISSING,
                        message=(
                            f"Viewer layer {layer_name!r} is missing required axis "
                            f"labels: {missing}."
                        ),
                    )
                )
            if require_nonzero_payloads and layer.missing_nonzero_count:
                warnings.append(
                    AgentWarning(
                        code=ViewerValidationWarningCode.PAYLOAD_NONZERO_METADATA_MISSING,
                        message=(
                            f"Viewer layer {layer_name!r} has "
                            f"{layer.missing_nonzero_count} payload summaries without "
                            "nonzero_count metadata."
                        ),
                    )
                )
            if require_nonzero_payloads and layer.zero_payload_count:
                warnings.append(
                    AgentWarning(
                        code=ViewerValidationWarningCode.PAYLOADS_ZERO,
                        message=(
                            f"Viewer layer {layer_name!r} has "
                            f"{layer.zero_payload_count} empty payload summaries."
                        ),
                    )
                )
            if layer.coordinate_gap_count:
                warnings.append(
                    AgentWarning(
                        code=ViewerValidationWarningCode.COORDINATE_GAPS,
                        message=(
                            f"Viewer layer {layer_name!r} has "
                            f"{layer.coordinate_gap_count} viewer coordinate "
                            "position(s) with no routed payload."
                        ),
                    )
                )
        return tuple(warnings)

    @staticmethod
    def _layer_validation_summary(
        layer: ViewerWindowLayerState,
        *,
        required_axis_labels: tuple[str, ...],
        require_nonzero_payloads: bool,
    ) -> ViewerWindowLayerValidationSummary:
        payload_count = len(layer.payload_summaries)
        nonzero_payload_count = 0
        zero_payload_count = 0
        missing_nonzero_count = 0
        for payload_summary in layer.payload_summaries:
            nonzero_count = ViewerWindowService._payload_nonzero_count(payload_summary)
            if nonzero_count is None:
                missing_nonzero_count += 1
            elif nonzero_count > 0:
                nonzero_payload_count += 1
            else:
                zero_payload_count += 1

        missing_required_axis_labels = tuple(
            axis_label
            for axis_label in required_axis_labels
            if axis_label not in layer.axis_labels
        )
        coordinate_gap_count = ViewerWindowService._coordinate_gap_count(layer)
        nonzero_valid = (
            not require_nonzero_payloads
            or (
                payload_count > 0
                and zero_payload_count == 0
                and missing_nonzero_count == 0
            )
        )
        valid = (
            layer.mounted
            and not layer.pending_update
            and not missing_required_axis_labels
            and nonzero_valid
        )
        return ViewerWindowLayerValidationSummary(
            route_key=layer.route_key,
            title=layer.title,
            mounted=layer.mounted,
            item_count=layer.item_count,
            payload_count=payload_count,
            nonzero_payload_count=nonzero_payload_count,
            zero_payload_count=zero_payload_count,
            missing_nonzero_count=missing_nonzero_count,
            coordinate_gap_count=coordinate_gap_count,
            axis_labels=layer.axis_labels,
            stack_axes=layer.stack_axes,
            missing_required_axis_labels=missing_required_axis_labels,
            pending_update=layer.pending_update,
            valid=valid,
        )

    @staticmethod
    def _coordinate_gap_count(layer: ViewerWindowLayerState) -> int:
        if not layer.stack_axes:
            return 0
        axis_position_count = ViewerWindowService._component_position_count(
            layer.axis_component_values,
            layer.stack_axes,
        )
        routed_position_count = ViewerWindowService._component_position_count(
            layer.routed_component_values,
            layer.stack_axes,
        )
        if axis_position_count <= routed_position_count:
            return 0
        return axis_position_count - routed_position_count

    @staticmethod
    def _component_position_count(
        component_values: Mapping[str, JsonValue],
        stack_axes: tuple[str, ...],
    ) -> int:
        position_count = 1
        for component in stack_axes:
            if component not in component_values:
                return 0
            values = component_values[component]
            if not isinstance(values, (list, tuple)):
                return 0
            position_count *= len(values)
        return position_count

    @staticmethod
    def _payload_nonzero_count(payload_summary: JsonObject) -> int | None:
        field_name = "nonzero_count"
        if field_name not in payload_summary:
            return None
        value = payload_summary[field_name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Viewer payload nonzero_count must be an integer.")
        return value

    def _state_result_from_response(
        self,
        *,
        connection: ExecutionConnectionSpec,
        response: JsonObject,
    ) -> ViewerWindowStateResult:
        status = self._required_str(response, ViewerControlField.STATUS)
        if status != self.SUCCESS_STATUS:
            message = self._required_str(response, ViewerControlField.MESSAGE)
            return viewer_window_state_error(
                connection=connection,
                error=AgentError(
                    code="viewer_window_state_failed",
                    message=message,
                ),
            )

        viewer_payload = self._required_mapping(response, ViewerControlField.VIEWER)
        layer_payloads = self._required_sequence(
            response,
            ViewerControlField.LAYERS,
        )
        return ViewerWindowStateResult(
            schema_version=SCHEMA_VERSION,
            connection=connection,
            observed=True,
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
            layer_count=self._required_int(response, ViewerControlField.LAYER_COUNT),
            layers=tuple(
                self._layer_state_from_payload(layer_payload)
                for layer_payload in layer_payloads
            ),
            active_dimension_label_route=self._optional_typed(
                response,
                ViewerControlField.ACTIVE_DIMENSION_LABEL_ROUTE,
                str,
            ),
            viewer_ndim=self._required_int(response, ViewerControlField.VIEWER_NDIM),
            current_step=self._required_typed_tuple(
                response,
                ViewerControlField.CURRENT_STEP,
                int,
            ),
            axis_labels=self._required_typed_tuple(
                response,
                ViewerControlField.AXIS_LABELS,
                str,
            ),
            component_group_count=self._required_int(
                response,
                ViewerControlField.COMPONENT_GROUP_COUNT,
            ),
            component_item_count=self._required_int(
                response,
                ViewerControlField.COMPONENT_ITEM_COUNT,
            ),
            response=response,
        )

    def _layer_state_from_payload(
        self,
        payload: JsonValue,
    ) -> ViewerWindowLayerState:
        if not isinstance(payload, Mapping):
            raise TypeError("Viewer layer state entries must be mappings.")
        return ViewerWindowLayerState(
            route_key=self._required_str(payload, ViewerLayerField.ROUTE_KEY),
            title=self._optional_typed(payload, ViewerLayerField.TITLE, str),
            mounted=self._required_bool(payload, ViewerLayerField.MOUNTED),
            item_count=self._required_int(payload, ViewerLayerField.ITEM_COUNT),
            data_types=self._required_typed_tuple(
                payload,
                ViewerLayerField.DATA_TYPES,
                str,
            ),
            component_values=self._required_mapping_tuple(
                payload,
                ViewerLayerField.COMPONENT_VALUES,
            ),
            payload_summaries=self._required_mapping_tuple(
                payload,
                ViewerLayerField.PAYLOAD_SUMMARIES,
            ),
            axis_labels=self._required_typed_tuple(
                payload,
                ViewerLayerField.AXIS_LABELS,
                str,
            ),
            stack_axes=self._required_typed_tuple(
                payload,
                ViewerLayerField.STACK_AXES,
                str,
            ),
            axis_offsets=self._required_typed_tuple(
                payload,
                ViewerLayerField.AXIS_OFFSETS,
                int,
            ),
            scalar_labels=self._required_typed_tuple(
                payload,
                ViewerLayerField.SCALAR_LABELS,
                str,
            ),
            labels=self._required_mapping(payload, ViewerLayerField.LABELS),
            axis_component_values=self._required_mapping(
                payload,
                ViewerLayerField.AXIS_COMPONENT_VALUES,
            ),
            routed_component_values=self._required_mapping(
                payload,
                ViewerLayerField.ROUTED_COMPONENT_VALUES,
            ),
            data_shape=self._required_typed_tuple(
                payload,
                ViewerLayerField.DATA_SHAPE,
                int,
            ),
            translate=self._required_typed_tuple(
                payload,
                ViewerLayerField.TRANSLATE,
                float,
            ),
            visible=self._required_bool(payload, ViewerLayerField.VISIBLE),
            selected=self._required_bool(payload, ViewerLayerField.SELECTED),
            pending_update=self._required_bool(payload, ViewerLayerField.PENDING_UPDATE),
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
    def _required_sequence(
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> tuple[JsonValue, ...]:
        if field_name not in payload:
            raise KeyError(f"Viewer response missing required field {field_name!r}.")
        value = payload[field_name]
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Viewer response field {field_name!r} must be a sequence.")
        return tuple(value)

    @staticmethod
    def _required_typed_tuple(
        payload: Mapping[str, JsonValue],
        field_name: str,
        expected_type: type[OptionalViewerFieldT],
    ) -> tuple[OptionalViewerFieldT, ...]:
        values = ViewerWindowService._required_sequence(payload, field_name)
        for value in values:
            if not isinstance(value, expected_type):
                type_name = expected_type.__name__
                raise TypeError(
                    f"Viewer response field {field_name!r} values must be {type_name}."
                )
        return values

    @staticmethod
    def _required_mapping_tuple(
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> tuple[JsonObject, ...]:
        values = ViewerWindowService._required_sequence(payload, field_name)
        mappings: list[JsonObject] = []
        for value in values:
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"Viewer response field {field_name!r} values must be mappings."
                )
            mappings.append(dict(value))
        return tuple(mappings)

    @staticmethod
    def _required_str(payload: Mapping[str, JsonValue], field_name: str) -> str:
        if field_name not in payload:
            raise KeyError(f"Viewer response missing required field {field_name!r}.")
        value = payload[field_name]
        if not isinstance(value, str):
            raise TypeError(f"Viewer response field {field_name!r} must be a string.")
        return value

    @staticmethod
    def _required_int(payload: Mapping[str, JsonValue], field_name: str) -> int:
        if field_name not in payload:
            raise KeyError(f"Viewer response missing required field {field_name!r}.")
        value = payload[field_name]
        if not isinstance(value, int):
            raise TypeError(f"Viewer response field {field_name!r} must be an integer.")
        return value

    @staticmethod
    def _required_bool(payload: Mapping[str, JsonValue], field_name: str) -> bool:
        if field_name not in payload:
            raise KeyError(f"Viewer response missing required field {field_name!r}.")
        value = payload[field_name]
        if not isinstance(value, bool):
            raise TypeError(f"Viewer response field {field_name!r} must be a boolean.")
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
