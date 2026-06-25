"""Agent service for running viewer window interactions."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
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
    ViewerWindowValidationPolicy,
    ViewerWindowValidationRequest,
    ViewerWindowValidationSummaryResult,
    viewer_window_snapshot_error,
    viewer_window_probe_from_state,
    viewer_window_state_error,
    viewer_window_validation_error,
)
from openhcs.core.image_shapes import ArrayShape
from openhcs.runtime.window_snapshot import (
    WindowSnapshotCaptureSpec,
    WindowSnapshotWirePayload,
)
from openhcs.runtime.viewer_protocol import ViewerControlMessageType
from openhcs.runtime.viewer_component_system import (
    ComponentValue,
    ComponentValues,
    ViewerComponentMetadataPayload,
    ViewerComponentValueParser,
    ViewerLayerAxisProjection,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG


OptionalViewerFieldT = TypeVar("OptionalViewerFieldT")


ComponentIndex = tuple[int, ...]
ComponentIndexSet = set[ComponentIndex]


@dataclass(frozen=True, slots=True)
class ViewerLayerPayloadCoordinateSet:
    """Payload coordinates projected through the shared viewer-axis projection."""

    indices: tuple[ComponentIndex, ...]
    invalid_payload_count: int

    @property
    def unique_indices(self) -> ComponentIndexSet:
        return set(self.indices)

    @property
    def duplicate_count(self) -> int:
        return len(self.indices) - len(self.unique_indices)

    @classmethod
    def from_layer_projection(
        cls,
        *,
        projection: ViewerLayerAxisProjection,
        payload_summaries: Sequence[JsonObject],
    ) -> "ViewerLayerPayloadCoordinateSet":
        indices: list[ComponentIndex] = []
        invalid_payload_count = 0
        for payload_summary in payload_summaries:
            try:
                indices.extend(cls._payload_indices(projection, payload_summary))
            except (TypeError, ValueError):
                invalid_payload_count += 1
        return cls(
            indices=tuple(indices),
            invalid_payload_count=invalid_payload_count,
        )

    @classmethod
    def _payload_indices(
        cls,
        projection: ViewerLayerAxisProjection,
        payload_summary: JsonObject,
    ) -> tuple[ComponentIndex, ...]:
        components = cls._payload_components(payload_summary)
        projected_values = tuple(
            cls._projected_payload_values(
                components=components,
                aggregate_values=cls._aggregate_component_values(payload_summary),
                component=component,
            )
            for component in projection.projected_axis_components
        )
        return tuple(
            projection.coordinate_index(
                {
                    **components,
                    **dict(
                        zip(
                            projection.projected_axis_components,
                            coordinate_values,
                            strict=True,
                        )
                    ),
                },
                context="viewer payload summary",
            )
            for coordinate_values in product(*projected_values)
        )

    @staticmethod
    def _payload_components(payload_summary: JsonObject) -> dict[str, ComponentValue]:
        components_payload = payload_summary.get("components")
        if not isinstance(components_payload, Mapping):
            raise ValueError("Viewer payload summary missing components.")
        return ViewerComponentMetadataPayload.component_map(
            components_payload,
            context="viewer payload summary",
        )

    @staticmethod
    def _projected_payload_values(
        *,
        components: Mapping[str, ComponentValue],
        aggregate_values: Mapping[str, tuple[ComponentValue, ...]],
        component: str,
    ) -> tuple[ComponentValue, ...]:
        if component in aggregate_values:
            return aggregate_values[component]
        if component not in components:
            raise ValueError(
                f"Viewer payload summary missing stack component {component!r}."
            )
        return (components[component],)

    @classmethod
    def _aggregate_component_values(
        cls,
        payload_summary: JsonObject,
    ) -> dict[str, tuple[ComponentValue, ...]]:
        aggregate_values_payload = payload_summary.get("aggregate_component_values")
        if aggregate_values_payload is None:
            return {}
        if not isinstance(aggregate_values_payload, Mapping):
            raise TypeError(
                "Viewer aggregate component values must be a component mapping."
            )
        return {
            str(component): cls._component_value_sequence(
                values,
                context=f"viewer aggregate component {component!r}",
            )
            for component, values in aggregate_values_payload.items()
        }

    @staticmethod
    def _component_value_sequence(
        values: JsonValue,
        *,
        context: str,
    ) -> tuple[ComponentValue, ...]:
        if isinstance(values, str) or not isinstance(values, Sequence):
            raise TypeError(f"{context} must be a sequence.")
        if not values:
            raise ValueError(f"{context} must not be empty.")
        return tuple(
            ViewerComponentValueParser.parse(value, context=context)
            for value in values
        )


@dataclass(frozen=True, slots=True)
class ViewerLayerValidationProjection:
    """Viewer-state layer axes converted into the common projection contract."""

    projection: ViewerLayerAxisProjection

    @property
    def viewer_coordinate_count(self) -> int:
        if not self.projection.projected_axis_components:
            return 1
        count = 1
        for axis_size in self.projection.axis_shape():
            count *= axis_size
        return count

    @property
    def expected_indices(self) -> ComponentIndexSet:
        return self.projection.expected_indices()

    @classmethod
    def from_layer(
        cls,
        layer: ViewerWindowLayerState,
    ) -> "ViewerLayerValidationProjection":
        return cls(
            projection=ViewerLayerAxisProjection(
                projected_axis_components=layer.stack_axes,
                component_values=cls._component_values(
                    layer.axis_component_values,
                    layer.stack_axes,
                    context="viewer layer axis domains",
                ),
                routed_component_values=cls._component_values(
                    layer.routed_component_values,
                    layer.stack_axes,
                    context="viewer layer routed domains",
                ),
                axis_offsets=layer.axis_offsets,
            )
        )

    @staticmethod
    def _component_values(
        payload: Mapping[str, JsonValue],
        components: Sequence[str],
        *,
        context: str,
    ) -> ComponentValues:
        component_values: ComponentValues = {}
        for component in components:
            if component not in payload:
                raise ValueError(f"{context} missing component {component!r}.")
            values = payload[component]
            if isinstance(values, str) or not isinstance(values, Sequence):
                raise TypeError(
                    f"{context} component {component!r} must be a sequence."
                )
            if not values:
                raise ValueError(f"{context} component {component!r} is empty.")
            component_values[component] = [
                ViewerComponentValueParser.parse(
                    value,
                    context=f"{context} component {component!r}",
                )
                for value in values
            ]
        return component_values


@dataclass(frozen=True, slots=True)
class ViewerLayerCoordinateCoverage:
    """Concrete stack-coordinate coverage for one viewer layer."""

    viewer_coordinate_count: int
    expected_coordinate_count: int
    payload_coordinate_count: int
    missing_payload_coordinate_count: int
    duplicate_payload_coordinate_count: int
    payload_without_coordinate_count: int

    @property
    def coordinate_gap_count(self) -> int:
        if self.viewer_coordinate_count <= self.expected_coordinate_count:
            return 0
        return self.viewer_coordinate_count - self.expected_coordinate_count

    @property
    def valid(self) -> bool:
        return (
            self.coordinate_gap_count == 0
            and self.missing_payload_coordinate_count == 0
            and self.duplicate_payload_coordinate_count == 0
            and self.payload_without_coordinate_count == 0
        )

    @classmethod
    def from_layer(
        cls,
        layer: ViewerWindowLayerState,
    ) -> "ViewerLayerCoordinateCoverage":
        projection = ViewerLayerValidationProjection.from_layer(layer)
        expected_indices = projection.expected_indices
        payload_coordinates = ViewerLayerPayloadCoordinateSet.from_layer_projection(
            projection=projection.projection,
            payload_summaries=layer.payload_summaries,
        )
        return cls(
            viewer_coordinate_count=projection.viewer_coordinate_count,
            expected_coordinate_count=len(expected_indices),
            payload_coordinate_count=len(payload_coordinates.unique_indices),
            missing_payload_coordinate_count=len(
                expected_indices - payload_coordinates.unique_indices
            ),
            duplicate_payload_coordinate_count=payload_coordinates.duplicate_count,
            payload_without_coordinate_count=(
                payload_coordinates.invalid_payload_count
            ),
        )


@dataclass(frozen=True, slots=True)
class ViewerWindowValidationTotals:
    """Aggregate validation counters for a viewer state."""

    mounted_layer_count: int
    pending_update_count: int
    payload_count: int
    nonzero_payload_count: int
    zero_payload_count: int
    missing_nonzero_count: int
    missing_payload_coordinate_count: int
    duplicate_payload_coordinate_count: int
    payload_without_coordinate_count: int
    spatial_mismatch_count: int

    @classmethod
    def from_state_and_layers(
        cls,
        *,
        state: ViewerWindowStateResult,
        layer_summaries: Sequence[ViewerWindowLayerValidationSummary],
    ) -> "ViewerWindowValidationTotals":
        return cls(
            mounted_layer_count=sum(1 for layer in state.layers if layer.mounted),
            pending_update_count=sum(
                1 for layer in state.layers if layer.pending_update
            ),
            payload_count=sum(layer.payload_count for layer in layer_summaries),
            nonzero_payload_count=sum(
                layer.nonzero_payload_count for layer in layer_summaries
            ),
            zero_payload_count=sum(
                layer.zero_payload_count for layer in layer_summaries
            ),
            missing_nonzero_count=sum(
                layer.missing_nonzero_count for layer in layer_summaries
            ),
            missing_payload_coordinate_count=sum(
                layer.missing_payload_coordinate_count
                for layer in layer_summaries
            ),
            duplicate_payload_coordinate_count=sum(
                layer.duplicate_payload_coordinate_count
                for layer in layer_summaries
            ),
            payload_without_coordinate_count=sum(
                layer.payload_without_coordinate_count for layer in layer_summaries
            ),
            spatial_mismatch_count=sum(
                layer.spatial_mismatch_count for layer in layer_summaries
            ),
        )


class ViewerWindowValidationAuthority:
    """Build validation DTOs from observed viewer state."""

    @classmethod
    def validation_summary(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        request: ViewerWindowValidationRequest,
        state: ViewerWindowStateResult,
    ) -> ViewerWindowValidationSummaryResult:
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
                validation_policy=request.validation_policy,
            )

        layer_summaries = tuple(
            cls.layer_validation_summary(
                layer,
                validation_policy=request.validation_policy,
            )
            for layer in state.layers
        )
        totals = ViewerWindowValidationTotals.from_state_and_layers(
            state=state,
            layer_summaries=layer_summaries,
        )
        layer_count_matches = (
            request.expected_layer_count is None
            or state.layer_count == request.expected_layer_count
        )
        return ViewerWindowValidationSummaryResult(
            schema_version=SCHEMA_VERSION,
            connection=connection,
            valid=layer_count_matches and all(
                layer.valid for layer in layer_summaries
            ),
            warnings=cls.validation_warnings(
                validation_policy=request.validation_policy,
                observed_layer_count=state.layer_count,
                layer_summaries=layer_summaries,
            ),
            observed=True,
            viewer=state.viewer,
            layer_count=state.layer_count,
            mounted_layer_count=totals.mounted_layer_count,
            pending_update_count=totals.pending_update_count,
            payload_count=totals.payload_count,
            nonzero_payload_count=totals.nonzero_payload_count,
            zero_payload_count=totals.zero_payload_count,
            missing_nonzero_count=totals.missing_nonzero_count,
            missing_payload_coordinate_count=(
                totals.missing_payload_coordinate_count
            ),
            duplicate_payload_coordinate_count=(
                totals.duplicate_payload_coordinate_count
            ),
            payload_without_coordinate_count=totals.payload_without_coordinate_count,
            spatial_mismatch_count=totals.spatial_mismatch_count,
            validation_policy=request.validation_policy,
            layer_summaries=layer_summaries,
            state=state,
        )

    @classmethod
    def validation_warnings(
        cls,
        *,
        validation_policy: ViewerWindowValidationPolicy,
        observed_layer_count: int,
        layer_summaries: Sequence[ViewerWindowLayerValidationSummary],
    ) -> tuple[AgentWarning, ...]:
        warnings: list[AgentWarning] = []
        if (
            validation_policy.expected_layer_count is not None
            and validation_policy.expected_layer_count != observed_layer_count
        ):
            warnings.append(
                AgentWarning(
                    code=ViewerValidationWarningCode.LAYER_COUNT_MISMATCH,
                    message=(
                        "Viewer layer count did not match the expected layer count: "
                        f"expected {validation_policy.expected_layer_count}, "
                        f"observed {observed_layer_count}."
                    ),
                )
            )

        for layer in layer_summaries:
            warnings.extend(
                cls.layer_validation_warnings(
                    layer=layer,
                    validation_policy=validation_policy,
                )
            )
        return tuple(warnings)

    @classmethod
    def layer_validation_warnings(
        cls,
        *,
        layer: ViewerWindowLayerValidationSummary,
        validation_policy: ViewerWindowValidationPolicy,
    ) -> tuple[AgentWarning, ...]:
        warnings: list[AgentWarning] = []
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
        if validation_policy.require_nonzero_payloads and layer.missing_nonzero_count:
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
        if validation_policy.require_nonzero_payloads and layer.zero_payload_count:
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
        if layer.missing_payload_coordinate_count:
            warnings.append(
                AgentWarning(
                    code=ViewerValidationWarningCode.PAYLOAD_COORDINATES_MISSING,
                    message=(
                        f"Viewer layer {layer_name!r} is missing "
                        f"{layer.missing_payload_coordinate_count} routed "
                        "payload coordinate(s)."
                    ),
                )
            )
        if layer.duplicate_payload_coordinate_count:
            warnings.append(
                AgentWarning(
                    code=ViewerValidationWarningCode.PAYLOAD_COORDINATES_DUPLICATE,
                    message=(
                        f"Viewer layer {layer_name!r} has "
                        f"{layer.duplicate_payload_coordinate_count} duplicate "
                        "payload coordinate(s)."
                    ),
                )
            )
        if layer.payload_without_coordinate_count:
            warnings.append(
                AgentWarning(
                    code=ViewerValidationWarningCode.PAYLOADS_WITHOUT_COORDINATES,
                    message=(
                        f"Viewer layer {layer_name!r} has "
                        f"{layer.payload_without_coordinate_count} payload(s) "
                        "without a valid routed coordinate."
                    ),
                )
            )
        if layer.spatial_mismatch_count:
            warnings.append(
                AgentWarning(
                    code=ViewerValidationWarningCode.SPATIAL_MISMATCH,
                    message=(
                        f"Viewer layer {layer_name!r} has "
                        f"{layer.spatial_mismatch_count} payload spatial "
                        "shape mismatch(es)."
                    ),
                )
            )
        return tuple(warnings)

    @classmethod
    def layer_validation_summary(
        cls,
        layer: ViewerWindowLayerState,
        *,
        validation_policy: ViewerWindowValidationPolicy,
    ) -> ViewerWindowLayerValidationSummary:
        payload_count = len(layer.payload_summaries)
        nonzero_payload_count = 0
        zero_payload_count = 0
        missing_nonzero_count = 0
        for payload_summary in layer.payload_summaries:
            nonzero_count = cls.payload_nonzero_count(payload_summary)
            if nonzero_count is None:
                missing_nonzero_count += 1
            elif nonzero_count > 0:
                nonzero_payload_count += 1
            else:
                zero_payload_count += 1

        missing_required_axis_labels = tuple(
            axis_label
            for axis_label in validation_policy.required_axis_labels
            if axis_label not in layer.axis_labels
        )
        coordinate_coverage = ViewerLayerCoordinateCoverage.from_layer(layer)
        spatial_mismatch_count = cls.spatial_mismatch_count(layer)
        nonzero_valid = (
            not validation_policy.require_nonzero_payloads
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
            and coordinate_coverage.valid
            and spatial_mismatch_count == 0
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
            coordinate_gap_count=coordinate_coverage.coordinate_gap_count,
            expected_coordinate_count=coordinate_coverage.expected_coordinate_count,
            routed_coordinate_count=coordinate_coverage.expected_coordinate_count,
            payload_coordinate_count=coordinate_coverage.payload_coordinate_count,
            missing_payload_coordinate_count=(
                coordinate_coverage.missing_payload_coordinate_count
            ),
            duplicate_payload_coordinate_count=(
                coordinate_coverage.duplicate_payload_coordinate_count
            ),
            payload_without_coordinate_count=(
                coordinate_coverage.payload_without_coordinate_count
            ),
            spatial_mismatch_count=spatial_mismatch_count,
            axis_labels=layer.axis_labels,
            stack_axes=layer.stack_axes,
            missing_required_axis_labels=missing_required_axis_labels,
            pending_update=layer.pending_update,
            valid=valid,
        )

    @classmethod
    def spatial_mismatch_count(cls, layer: ViewerWindowLayerState) -> int:
        layer_spatial_shape = cls.layer_spatial_shape(layer)
        if layer_spatial_shape is None:
            return 0
        mismatch_count = 0
        for payload_summary in layer.payload_summaries:
            candidate = cls.payload_spatial_shape(payload_summary)
            if candidate is None:
                continue
            if candidate != layer_spatial_shape:
                mismatch_count += 1
        return mismatch_count

    @classmethod
    def layer_spatial_shape(
        cls,
        layer: ViewerWindowLayerState,
    ) -> tuple[int, int] | None:
        local_shape = tuple(
            int(value) for value in layer.data_shape[len(layer.stack_axes):]
        )
        return cls.image_spatial_shape(local_shape)

    @classmethod
    def payload_spatial_shape(
        cls,
        payload_summary: JsonObject,
    ) -> tuple[int, int] | None:
        shape = payload_summary.get("shape")
        if (
            isinstance(shape, Sequence)
            and not isinstance(shape, str)
            and len(shape) >= 2
        ):
            return cls.image_spatial_shape(tuple(int(value) for value in shape))
        source_shapes = payload_summary.get("source_spatial_shapes_yx")
        if (
            not isinstance(source_shapes, Sequence)
            or isinstance(source_shapes, str)
            or len(source_shapes) != 1
        ):
            return None
        source_shape = source_shapes[0]
        if (
            not isinstance(source_shape, Sequence)
            or isinstance(source_shape, str)
            or len(source_shape) != 2
        ):
            return None
        return (int(source_shape[0]), int(source_shape[1]))

    @staticmethod
    def image_spatial_shape(shape: tuple[int, ...]) -> tuple[int, int] | None:
        if len(shape) < 2:
            return None
        if (
            len(shape) >= 3
            and ArrayShape(ndim=len(shape), shape=shape).has_channel_last()
        ):
            return tuple(int(value) for value in shape[-3:-1])
        return tuple(int(value) for value in shape[-2:])

    @staticmethod
    def payload_nonzero_count(payload_summary: JsonObject) -> int | None:
        field_name = "nonzero_count"
        if field_name not in payload_summary:
            return None
        value = payload_summary[field_name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Viewer payload nonzero_count must be an integer.")
        return value


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
    PAYLOAD_COORDINATES_MISSING = "viewer_payload_coordinates_missing"
    PAYLOAD_COORDINATES_DUPLICATE = "viewer_payload_coordinates_duplicate"
    PAYLOADS_WITHOUT_COORDINATES = "viewer_payloads_without_coordinates"
    SPATIAL_MISMATCH = "viewer_layer_spatial_mismatch"


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
            validation_policy=ViewerWindowValidationPolicy(
                expected_layer_count=expected_layer_count,
                required_axis_labels=required_axis_labels,
                require_nonzero_payloads=require_nonzero_payloads,
            ),
            timeout_ms=timeout_ms,
        )
        state = self.window_state(
            port=port,
            host=host,
            transport_mode=transport_mode,
            timeout_ms=timeout_ms,
        )
        return ViewerWindowValidationAuthority.validation_summary(
            connection=connection,
            request=request,
            state=state,
        )

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
