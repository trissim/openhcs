"""DTOs for agent-facing viewer window interactions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import ClassVar, Self, cast

from metaclass_registry import AutoRegisterMeta
from polystore.streaming.identity import StreamProducerIdentity

from openhcs.agent.dto.common import (
    AgentError,
    AgentResourceRef,
    AgentResultEnvelope,
    JsonObject,
    JsonValue,
    SCHEMA_VERSION,
)
from openhcs.agent.dto.execution import (
    ExecutionConnectionProjection,
    ExecutionConnectionSpec,
)
from openhcs.agent.path_policy import DEFAULT_AGENT_WINDOW_SNAPSHOT_DIR
from openhcs.serialization.json import to_jsonable
from openhcs.runtime.viewer_controls import (
    ViewerNavigationControlOptions,
    ViewerPayloadControlOptions,
    ViewerStateControlOptions,
)
from openhcs.runtime.window_snapshot import (
    WindowSnapshotCaptureScope,
    WindowSnapshotCaptureSpec,
)

VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT = 5000


@dataclass(frozen=True, slots=True)
class ViewerWindowDescriptor:
    """Agent-facing identity for the viewer window that produced a resource."""

    viewer_type: str
    title: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowControlRequest(ExecutionConnectionProjection):
    """Shared control-message request fields for running viewer windows."""

    timeout_ms: int = VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT
    include_response: bool = True

    @classmethod
    def factory_injected_field_names(cls) -> frozenset[str]:
        connection_field_names = frozenset(
            field.name for field in dataclass_fields(ExecutionConnectionProjection)
        )
        timeout_field_names = frozenset(
            field.name
            for field in dataclass_fields(cls)
            if field.default == VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT
        )
        return connection_field_names | timeout_field_names

    @staticmethod
    def axis_indices_from_fields(
        value: tuple[int, ...] | list[int] | dict[str, int] | None,
    ) -> tuple[int, ...] | dict[str, int] | None:
        if value is None or isinstance(value, dict):
            return value
        if isinstance(value, tuple):
            return value
        return tuple(value)

    @staticmethod
    def axis_indices_to_tool_value(
        value: tuple[int, ...] | dict[str, int] | None,
    ) -> list[int] | dict[str, int] | None:
        if isinstance(value, tuple):
            return list(value)
        return value

    @staticmethod
    def labels_from_fields(
        value: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        return tuple(value)

    def connection_tool_arguments(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {"host": self.connection.host}
        if self.connection.port is not None:
            payload["port"] = self.connection.port
        if self.connection.transport_mode is not None:
            payload["transport_mode"] = self.connection.transport_mode
        if self.timeout_ms != VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT:
            payload["timeout_ms"] = self.timeout_ms
        return payload


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowSnapshotRequest(
    WindowSnapshotCaptureSpec, ViewerWindowControlRequest
):
    @classmethod
    def from_connection(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        timeout_ms: int = VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT,
        output_dir_path: str | None = None,
        capture_scope: str = WindowSnapshotCaptureScope.WIDGET.value,
    ) -> "ViewerWindowSnapshotRequest":
        if output_dir_path is None:
            output_dir_path = str(DEFAULT_AGENT_WINDOW_SNAPSHOT_DIR)
        return cls(
            connection=connection,
            timeout_ms=timeout_ms,
            output_dir_path=output_dir_path,
            capture_scope=WindowSnapshotCaptureScope(capture_scope),
        )

    @classmethod
    def from_fields(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        timeout_ms: int = VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT,
        output_dir_path: str | None = None,
        capture_scope: str = WindowSnapshotCaptureScope.WIDGET.value,
    ) -> "ViewerWindowSnapshotRequest":
        return cls.from_connection(
            connection=connection,
            timeout_ms=timeout_ms,
            output_dir_path=output_dir_path,
            capture_scope=capture_scope,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        payload = self.connection_tool_arguments()
        payload.update(
            {
                "output_dir_path": self.output_dir_path,
                "capture_scope": self.capture_scope.value,
            }
        )
        return payload


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowStateRequest(ViewerWindowControlRequest):
    state_controls: ViewerStateControlOptions = field(
        default_factory=ViewerStateControlOptions
    )

    @classmethod
    def from_fields(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        timeout_ms: int = VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT,
        route_key: str | None = None,
        include_component_values: bool = False,
        max_component_values_per_layer: int | None = None,
        include_payload_summaries: bool = False,
        max_payload_summaries_per_layer: int | None = None,
        include_response: bool = False,
    ) -> Self:
        return cls(
            connection=connection,
            timeout_ms=timeout_ms,
            include_response=include_response,
            state_controls=ViewerStateControlOptions.from_overrides(
                route_key=route_key,
                include_component_values=include_component_values,
                max_component_values_per_layer=max_component_values_per_layer,
                include_payload_summaries=include_payload_summaries,
                max_payload_summaries_per_layer=max_payload_summaries_per_layer,
            ),
        )

    @classmethod
    def from_cli_fields(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        timeout_ms: int = VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT,
        route_key: str | None = None,
        include_component_values: bool = True,
        max_component_values_per_layer: int | None = 8,
        include_payload_summaries: bool = True,
        max_payload_summaries_per_layer: int | None = 8,
        include_response: bool = False,
    ) -> Self:
        return cls.from_fields(
            connection=connection,
            timeout_ms=timeout_ms,
            route_key=route_key,
            include_component_values=include_component_values,
            max_component_values_per_layer=max_component_values_per_layer,
            include_payload_summaries=include_payload_summaries,
            max_payload_summaries_per_layer=max_payload_summaries_per_layer,
            include_response=include_response,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        payload = self.connection_tool_arguments()
        payload.update(
            cast(
                dict[str, JsonValue], to_jsonable(self.state_controls)
            )
        )
        payload["include_response"] = self.include_response
        return payload


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowPayloadRequest(ViewerWindowControlRequest):
    payload_controls: ViewerPayloadControlOptions = field(
        default_factory=ViewerPayloadControlOptions
    )

    @staticmethod
    def array_slices_from_fields(
        value: (
            tuple[tuple[int, int], ...] | list[tuple[int, int]] | list[list[int]] | None
        ),
    ) -> tuple[tuple[int, int], ...] | None:
        if value is None:
            return None
        return tuple(tuple(slice_pair) for slice_pair in value)

    @classmethod
    def from_fields(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        timeout_ms: int = VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT,
        route_key: str | None = None,
        axis_indices: tuple[int, ...] | list[int] | dict[str, int] | None = None,
        include_array_values: bool | None = None,
        max_array_elements: int | None = None,
        array_slices: (
            tuple[tuple[int, int], ...] | list[tuple[int, int]] | list[list[int]] | None
        ) = None,
        include_shape_payloads: bool | None = None,
        max_shape_payloads: int | None = None,
        include_response: bool = False,
    ) -> Self:
        return cls(
            connection=connection,
            timeout_ms=timeout_ms,
            include_response=include_response,
            payload_controls=ViewerPayloadControlOptions.from_overrides(
                route_key=route_key,
                axis_indices=cls.axis_indices_from_fields(axis_indices),
                include_array_values=include_array_values,
                max_array_elements=max_array_elements,
                array_slices=cls.array_slices_from_fields(array_slices),
                include_shape_payloads=include_shape_payloads,
                max_shape_payloads=max_shape_payloads,
            ),
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        payload = self.connection_tool_arguments()
        payload.update(
            cast(
                dict[str, JsonValue],
                to_jsonable(self.payload_controls),
            )
        )
        payload["include_response"] = self.include_response
        return payload


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowNavigationRequest(ViewerWindowControlRequest):
    navigation: ViewerNavigationControlOptions

    @classmethod
    def from_fields(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        timeout_ms: int = VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT,
        route_key: str,
        axis_indices: dict[str, int] | None = None,
        visible: bool | None = True,
        selected: bool | None = True,
        data_index: int | None = None,
    ) -> Self:
        return cls(
            connection=connection,
            timeout_ms=timeout_ms,
            navigation=ViewerNavigationControlOptions.from_overrides(
                route_key=route_key,
                axis_indices=axis_indices,
                visible=visible,
                selected=selected,
                data_index=data_index,
            ),
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        payload = self.connection_tool_arguments()
        payload.update(
            cast(dict[str, JsonValue], to_jsonable(self.navigation))
        )
        return payload


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowLayerIsolationRequest(ViewerWindowControlRequest):
    visible_route_keys: tuple[str, ...]
    selected_route_key: str | None = None
    axis_indices: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if not self.visible_route_keys:
            raise ValueError("visible_route_keys must contain at least one route key.")

    @classmethod
    def from_fields(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        timeout_ms: int = VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT,
        visible_route_keys: tuple[str, ...] | list[str],
        selected_route_key: str | None = None,
        axis_indices: dict[str, int] | None = None,
    ) -> Self:
        return cls(
            connection=connection,
            timeout_ms=timeout_ms,
            visible_route_keys=tuple(visible_route_keys),
            selected_route_key=selected_route_key,
            axis_indices=axis_indices,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        payload = self.connection_tool_arguments()
        payload.update(
            {
                "visible_route_keys": list(self.visible_route_keys),
                "selected_route_key": self.selected_route_key,
                "axis_indices": self.axis_indices,
            }
        )
        return payload

    @property
    def requested_visible_route_keys(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.visible_route_keys))

    @property
    def selected_route(self) -> str:
        return self.selected_route_key or self.requested_visible_route_keys[-1]

    @property
    def visible_routes(self) -> frozenset[str]:
        return frozenset((*self.requested_visible_route_keys, self.selected_route))

    def state_request(self) -> ViewerWindowStateRequest:
        return ViewerWindowStateRequest(
            connection=self.connection,
            timeout_ms=self.timeout_ms,
            include_response=False,
            state_controls=ViewerStateControlOptions.from_overrides(
                include_component_values=False,
                include_payload_summaries=False,
            ),
        )

    def navigation_request(
        self,
        *,
        route_key: str,
        visible: bool,
        selected: bool,
    ) -> ViewerWindowNavigationRequest:
        return ViewerWindowNavigationRequest(
            connection=self.connection,
            timeout_ms=self.timeout_ms,
            navigation=ViewerNavigationControlOptions.from_overrides(
                route_key=route_key,
                axis_indices=self.axis_indices if selected else None,
                visible=visible,
                selected=selected,
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowImageSampleRequest(ViewerWindowControlRequest):
    route_key: str | None = None
    axis_indices: tuple[int, ...] | dict[str, int] | None = None
    y: int = 0
    x: int = 0
    height: int = 32
    width: int = 32
    include_array_values: bool = False
    max_array_elements: int = 4096
    max_records: int = 3
    """Maximum matched image records returned in the MCP result."""

    def __post_init__(self) -> None:
        if self.y < 0 or self.x < 0:
            raise ValueError("Sample origin y/x must be nonnegative.")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Sample height/width must be positive.")
        if self.max_records < 0:
            raise ValueError("max_records must be nonnegative.")

    @classmethod
    def from_fields(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        timeout_ms: int = VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT,
        route_key: str | None = None,
        axis_indices: tuple[int, ...] | list[int] | dict[str, int] | None = None,
        y: int = 0,
        x: int = 0,
        height: int = 32,
        width: int = 32,
        include_array_values: bool = False,
        max_array_elements: int = 4096,
        max_records: int = 3,
    ) -> Self:
        return cls(
            connection=connection,
            timeout_ms=timeout_ms,
            route_key=route_key,
            axis_indices=cls.axis_indices_from_fields(axis_indices),
            y=y,
            x=x,
            height=height,
            width=width,
            include_array_values=include_array_values,
            max_array_elements=max_array_elements,
            max_records=max_records,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        payload = self.connection_tool_arguments()
        payload.update(
            {
                "route_key": self.route_key,
                "axis_indices": self.axis_indices_to_tool_value(self.axis_indices),
                "y": self.y,
                "x": self.x,
                "height": self.height,
                "width": self.width,
                "include_array_values": self.include_array_values,
                "max_array_elements": self.max_array_elements,
                "max_records": self.max_records,
            }
        )
        return payload

    @property
    def array_slices(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((self.y, self.y + self.height), (self.x, self.x + self.width))

    @property
    def protocol_max_array_elements(self) -> int:
        return self.max_array_elements if self.include_array_values else 0

    def payload_request(self) -> ViewerWindowPayloadRequest:
        return ViewerWindowPayloadRequest(
            connection=self.connection,
            timeout_ms=self.timeout_ms,
            include_response=False,
            payload_controls=ViewerPayloadControlOptions.from_overrides(
                route_key=self.route_key,
                axis_indices=self.axis_indices,
                include_array_values=True,
                max_array_elements=self.protocol_max_array_elements,
                array_slices=self.array_slices,
                include_shape_payloads=False,
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowRoiSummaryRequest(ViewerWindowControlRequest):
    route_key: str | None = None
    axis_indices: tuple[int, ...] | dict[str, int] | None = None
    max_rois: int = 512
    max_examples: int = 5

    def __post_init__(self) -> None:
        if self.max_rois < 0:
            raise ValueError("max_rois must be nonnegative.")
        if self.max_examples < 0:
            raise ValueError("max_examples must be nonnegative.")

    @classmethod
    def from_fields(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        timeout_ms: int = VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT,
        route_key: str | None = None,
        axis_indices: tuple[int, ...] | list[int] | dict[str, int] | None = None,
        max_rois: int = 512,
        max_examples: int = 5,
    ) -> Self:
        return cls(
            connection=connection,
            timeout_ms=timeout_ms,
            route_key=route_key,
            axis_indices=cls.axis_indices_from_fields(axis_indices),
            max_rois=max_rois,
            max_examples=max_examples,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        payload = self.connection_tool_arguments()
        payload.update(
            {
                "route_key": self.route_key,
                "axis_indices": self.axis_indices_to_tool_value(self.axis_indices),
                "max_rois": self.max_rois,
                "max_examples": self.max_examples,
            }
        )
        return payload

    def payload_request(self) -> ViewerWindowPayloadRequest:
        return ViewerWindowPayloadRequest(
            connection=self.connection,
            timeout_ms=self.timeout_ms,
            include_response=False,
            payload_controls=ViewerPayloadControlOptions.from_overrides(
                route_key=self.route_key,
                axis_indices=self.axis_indices,
                include_array_values=False,
                include_shape_payloads=True,
                max_shape_payloads=self.max_rois,
            ),
        )


@dataclass(frozen=True, slots=True)
class ViewerWindowValidationPolicy:
    """Validation contract applied to a viewer state observation."""

    expected_layer_count: int | None = None
    required_axis_labels: tuple[str, ...] = ()
    required_component_labels: tuple[str, ...] = ()
    require_nonzero_payloads: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowValidationRequest(ViewerWindowControlRequest):
    validation_policy: ViewerWindowValidationPolicy = field(
        default_factory=ViewerWindowValidationPolicy
    )
    state_controls: ViewerStateControlOptions = field(
        default_factory=ViewerStateControlOptions
    )
    include_state: bool = False

    @classmethod
    def from_fields(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        timeout_ms: int = VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT,
        route_key: str | None = None,
        expected_layer_count: int | None = None,
        required_axis_labels: tuple[str, ...] | list[str] = (),
        required_component_labels: tuple[str, ...] | list[str] = (),
        require_nonzero_payloads: bool = True,
        include_state: bool = False,
    ) -> Self:
        return cls(
            connection=connection,
            timeout_ms=timeout_ms,
            validation_policy=ViewerWindowValidationPolicy(
                expected_layer_count=expected_layer_count,
                required_axis_labels=cls.labels_from_fields(required_axis_labels),
                required_component_labels=cls.labels_from_fields(
                    required_component_labels
                ),
                require_nonzero_payloads=require_nonzero_payloads,
            ),
            state_controls=ViewerStateControlOptions.from_overrides(
                route_key=route_key,
                include_component_values=False,
                include_payload_summaries=True,
            ),
            include_state=include_state,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        payload = self.connection_tool_arguments()
        payload.update(
            {
                "route_key": self.state_controls.route_key,
                "expected_layer_count": self.expected_layer_count,
                "required_axis_labels": list(self.required_axis_labels),
                "required_component_labels": list(self.required_component_labels),
                "require_nonzero_payloads": self.require_nonzero_payloads,
                "include_state": self.include_state,
            }
        )
        return payload

    @property
    def expected_layer_count(self) -> int | None:
        return self.validation_policy.expected_layer_count

    @property
    def required_axis_labels(self) -> tuple[str, ...]:
        return self.validation_policy.required_axis_labels

    @property
    def required_component_labels(self) -> tuple[str, ...]:
        return self.validation_policy.required_component_labels

    @property
    def require_nonzero_payloads(self) -> bool:
        return self.validation_policy.require_nonzero_payloads


@dataclass(frozen=True, slots=True)
class ViewerWindowProbeResult(AgentResultEnvelope, ExecutionConnectionProjection):
    reachable: bool
    observed: bool = False
    viewer: ViewerWindowDescriptor | None = None
    layer_count: int = 0
    component_group_count: int = 0
    component_item_count: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowLayerDescriptor:
    """Shared identity and semantic-axis state for one viewer layer."""

    route_key: str
    title: str | None
    mounted: bool
    item_count: int
    producer_identities: tuple[StreamProducerIdentity, ...] = ()
    axis_labels: tuple[str, ...] = ()
    stack_axes: tuple[str, ...] = ()
    pending_update: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowLayerState(ViewerWindowLayerDescriptor):
    data_types: tuple[str, ...] = ()
    component_values: tuple[JsonObject, ...] = ()
    component_value_count: int = 0
    component_values_truncated: bool = False
    payload_summaries: tuple[JsonObject, ...] = ()
    payload_summary_count: int = 0
    payload_summaries_truncated: bool = False
    axis_offsets: tuple[int, ...] = ()
    scalar_labels: tuple[str, ...] = ()
    labels: JsonObject = field(default_factory=dict)
    axis_component_values: JsonObject = field(default_factory=dict)
    routed_component_values: JsonObject = field(default_factory=dict)
    data_shape: tuple[int, ...] = ()
    translate: tuple[float, ...] = ()
    visible: bool = False
    selected: bool = False
    feature_row_count: int = 0
    selected_data_indices: tuple[int, ...] = ()


class ViewerWindowErrorResultFactory(ABC, metaclass=AutoRegisterMeta):
    """Nominal constructor contract for viewer results with standard errors."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    __registry__: ClassVar[dict[str, type["ViewerWindowErrorResultFactory"]]] = {}
    registry_key: ClassVar[str | None] = None

    @classmethod
    def registered_result_types(
        cls,
    ) -> tuple[type["ViewerWindowErrorResultFactory"], ...]:
        return tuple(cls.__registry__.values())

    @classmethod
    def from_error(
        cls: type[Self],
        *,
        connection: ExecutionConnectionSpec,
        error: AgentError,
    ) -> Self:
        return cls.from_error_context(
            ViewerWindowErrorContext.from_error(
                connection=connection,
                error=error,
            )
        )

    @classmethod
    @abstractmethod
    def from_error_context(
        cls: type[Self],
        context: "ViewerWindowErrorContext",
    ) -> Self:
        """Build an error result from one viewer-window error context."""


class ViewerWindowSnapshotErrorResultMixin(ViewerWindowErrorResultFactory):
    """Error constructor for viewer snapshot result DTOs."""

    @classmethod
    def from_error_context(
        cls: type[Self],
        context: "ViewerWindowSnapshotErrorContext",
    ) -> Self:
        return cls(
            schema_version=SCHEMA_VERSION,
            connection=context.connection,
            output_dir_path=context.output_dir_path,
            capture_scope=context.capture_scope,
            captured=False,
            errors=(context.error,),
        )

    @classmethod
    def from_request_error(
        cls: type[Self],
        *,
        request: "ViewerWindowSnapshotRequest",
        error: AgentError,
    ) -> Self:
        return cls.from_error_context(
            ViewerWindowSnapshotErrorContext.from_request_error(
                request=request,
                error=error,
            )
        )


class ViewerWindowObservedErrorResultMixin(ViewerWindowErrorResultFactory):
    """Error constructor for viewer result DTOs with an observed flag."""

    @classmethod
    def from_error_context(
        cls: type[Self],
        context: "ViewerWindowErrorContext",
    ) -> Self:
        return cls(
            schema_version=SCHEMA_VERSION,
            connection=context.connection,
            observed=False,
            errors=(context.error,),
        )


class ViewerWindowValidationErrorResultMixin:
    """Error constructor for viewer validation result DTOs."""

    @classmethod
    def from_error_context(
        cls: type[Self],
        context: "ViewerWindowValidationErrorContext",
    ) -> Self:
        return cls(
            schema_version=SCHEMA_VERSION,
            connection=context.connection,
            valid=False,
            errors=(context.error,),
            validation_policy=context.validation_policy,
        )


@dataclass(frozen=True, slots=True)
class ViewerWindowSnapshotResult(
    WindowSnapshotCaptureSpec,
    ViewerWindowSnapshotErrorResultMixin,
    AgentResultEnvelope,
    ExecutionConnectionProjection,
):
    registry_key: ClassVar[str] = "snapshot"

    captured: bool
    resource: AgentResourceRef | None = None
    viewer: ViewerWindowDescriptor | None = None
    width: int | None = None
    height: int | None = None
    response: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ViewerWindowStateResult(
    ViewerWindowObservedErrorResultMixin,
    AgentResultEnvelope,
    ExecutionConnectionProjection,
):
    registry_key: ClassVar[str] = "state"

    observed: bool
    viewer: ViewerWindowDescriptor | None = None
    layer_count: int = 0
    layers: tuple[ViewerWindowLayerState, ...] = ()
    active_dimension_label_route: str | None = None
    viewer_ndim: int = 0
    current_step: tuple[int, ...] = ()
    axis_labels: tuple[str, ...] = ()
    component_group_count: int = 0
    component_item_count: int = 0
    response: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ViewerWindowPayloadRecord:
    route_key: str
    data_type: str
    path: str
    components: JsonObject
    axis_indices: tuple[int, ...] = ()
    aggregate_axis_indices: tuple[int, ...] = ()
    summary: JsonObject = field(default_factory=dict)
    array_values: tuple[JsonValue, ...] = ()
    array_value_summary: JsonObject = field(default_factory=dict)
    shape_payloads: tuple[JsonObject, ...] = ()


@dataclass(frozen=True, slots=True)
class ViewerWindowLayerPayloads(ViewerWindowLayerDescriptor):
    payloads: tuple[ViewerWindowPayloadRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class ViewerWindowPayloadResult(
    ViewerWindowObservedErrorResultMixin,
    AgentResultEnvelope,
    ExecutionConnectionProjection,
):
    registry_key: ClassVar[str] = "payload"

    observed: bool
    viewer: ViewerWindowDescriptor | None = None
    layer_count: int = 0
    layers: tuple[ViewerWindowLayerPayloads, ...] = ()
    response: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ViewerWindowNavigationResult(
    ViewerWindowObservedErrorResultMixin,
    AgentResultEnvelope,
    ExecutionConnectionProjection,
):
    registry_key: ClassVar[str] = "navigation"

    observed: bool
    viewer: ViewerWindowDescriptor | None = None
    route_key: str | None = None
    visible: bool | None = None
    selected: bool | None = None
    data_index: int | None = None
    feature_row_count: int = 0
    selected_data_indices: tuple[int, ...] = ()
    active_dimension_label_route: str | None = None
    current_step: tuple[int, ...] = ()
    axis_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ViewerWindowLayerVisibilityRecord:
    route_key: str
    title: str | None
    visible: bool
    selected: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowLayerIsolationResult(AgentResultEnvelope):
    observed: bool
    applied: bool
    selected_route_key: str | None = None
    visible_route_keys: tuple[str, ...] = ()
    hidden_route_keys: tuple[str, ...] = ()
    missing_route_keys: tuple[str, ...] = ()
    available_layers: tuple[ViewerWindowLayerVisibilityRecord, ...] = ()
    changed_route_count: int = 0
    layer_count: int = 0
    active_dimension_label_route: str | None = None
    current_step: tuple[int, ...] = ()
    axis_labels: tuple[str, ...] = ()
    visible_layers: tuple[ViewerWindowLayerVisibilityRecord, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowImageSampleResult(AgentResultEnvelope):
    observed: bool
    route_key: str | None = None
    requested_route_key: str | None = None
    auto_selected_route_key: str | None = None
    candidate_image_route_keys: tuple[str, ...] = ()
    axis_indices: tuple[int, ...] | dict[str, int] | None = None
    array_slices: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))
    record_count: int = 0
    returned_record_count: int = 0
    records_truncated_count: int = 0
    total_payload_record_count: int = 0
    raw_image_record_count: int = 0
    filtered_out_image_record_count: int = 0
    non_image_record_count: int = 0
    axis_filter_applied_by_viewer: bool = True
    client_side_axis_filter_applied: bool = False
    sample_protocol_supported: bool = False
    sample_included_count: int = 0
    sample_omitted_count: int = 0
    records: tuple[JsonObject, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowRoiSummaryResult(AgentResultEnvelope):
    observed: bool
    route_key: str | None = None
    axis_indices: tuple[int, ...] | dict[str, int] | None = None
    layer_count: int = 0
    payload_record_count: int = 0
    payload_type_counts: dict[str, int] = field(default_factory=dict)
    roi_payload_count: int = 0
    total_roi_count: int = 0
    returned_roi_count: int = 0
    roi_count_exact: bool = True
    total_roi_member_count: int = 0
    returned_roi_member_count: int = 0
    roi_payloads_truncated: bool = False
    payloads: tuple[JsonObject, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ViewerWindowValidationCounters:
    """Shared validation counters for per-layer and whole-window summaries."""

    payload_count: int = 0
    nonzero_payload_count: int = 0
    zero_payload_count: int = 0
    missing_nonzero_count: int = 0
    missing_payload_coordinate_count: int = 0
    duplicate_payload_coordinate_count: int = 0
    payload_without_coordinate_count: int = 0
    valid: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowLayerValidationSummary(
    ViewerWindowValidationCounters,
    ViewerWindowLayerDescriptor,
):
    coordinate_gap_count: int = 0
    expected_coordinate_count: int = 0
    routed_coordinate_count: int = 0
    payload_coordinate_count: int = 0
    missing_required_axis_labels: tuple[str, ...] = ()
    component_labels: tuple[str, ...] = ()
    missing_required_component_labels: tuple[str, ...] = ()
    axis_labels_present_as_components: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ViewerWindowValidationSummaryResult(
    ViewerWindowValidationErrorResultMixin,
    ViewerWindowValidationCounters,
    AgentResultEnvelope,
    ExecutionConnectionProjection,
):
    observed: bool = False
    viewer: ViewerWindowDescriptor | None = None
    layer_count: int = 0
    mounted_layer_count: int = 0
    pending_update_count: int = 0
    active_dimension_label_route: str | None = None
    active_dimension_label_route_valid: bool = False
    validation_policy: ViewerWindowValidationPolicy = field(
        default_factory=ViewerWindowValidationPolicy
    )
    layer_summaries: tuple[ViewerWindowLayerValidationSummary, ...] = ()
    state: ViewerWindowStateResult | None = None

    @property
    def expected_layer_count(self) -> int | None:
        return self.validation_policy.expected_layer_count

    @property
    def required_axis_labels(self) -> tuple[str, ...]:
        return self.validation_policy.required_axis_labels

    @property
    def required_component_labels(self) -> tuple[str, ...]:
        return self.validation_policy.required_component_labels

    @property
    def require_nonzero_payloads(self) -> bool:
        return self.validation_policy.require_nonzero_payloads


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowErrorContext(ExecutionConnectionProjection):
    """Nominal source context for viewer result errors."""

    error: AgentError

    @classmethod
    def from_error(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        error: AgentError,
    ) -> Self:
        return cls(
            connection=connection,
            error=error,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowValidationErrorContext(ViewerWindowErrorContext):
    """Viewer validation error context plus the applied validation policy."""

    validation_policy: ViewerWindowValidationPolicy


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowSnapshotErrorContext(
    WindowSnapshotCaptureSpec,
    ViewerWindowErrorContext,
):
    """Viewer snapshot error context plus the requested capture contract."""

    @classmethod
    def from_request_error(
        cls,
        *,
        request: ViewerWindowSnapshotRequest,
        error: AgentError,
    ) -> Self:
        return cls(
            connection=request.connection,
            output_dir_path=request.output_dir_path,
            capture_scope=request.capture_scope,
            error=error,
        )


def viewer_window_probe_from_state(
    state: ViewerWindowStateResult,
) -> ViewerWindowProbeResult:
    return ViewerWindowProbeResult(
        schema_version=SCHEMA_VERSION,
        connection=state.connection,
        reachable=state.observed,
        observed=state.observed,
        viewer=state.viewer,
        layer_count=state.layer_count,
        component_group_count=state.component_group_count,
        component_item_count=state.component_item_count,
        errors=state.errors,
        warnings=state.warnings,
    )
