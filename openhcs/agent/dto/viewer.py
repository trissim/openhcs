"""DTOs for agent-facing viewer window interactions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Self

from metaclass_registry import AutoRegisterMeta

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
from openhcs.runtime.viewer_protocol import (
    ViewerControlWireValue,
    ViewerPayloadControlOptions,
)
from openhcs.runtime.window_snapshot import WindowSnapshotCaptureSpec


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


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowSnapshotRequest(WindowSnapshotCaptureSpec, ViewerWindowControlRequest):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowStateRequest(ViewerWindowControlRequest):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowPayloadRequest(ViewerWindowControlRequest):
    payload_controls: ViewerPayloadControlOptions = field(
        default_factory=ViewerPayloadControlOptions
    )

    def to_wire_payload(self) -> dict[str, ViewerControlWireValue]:
        return self.payload_controls.to_wire_payload()


@dataclass(frozen=True, slots=True)
class ViewerWindowValidationPolicy:
    """Validation contract applied to a viewer state observation."""

    expected_layer_count: int | None = None
    required_axis_labels: tuple[str, ...] = ()
    require_nonzero_payloads: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowValidationRequest(ViewerWindowControlRequest):
    validation_policy: ViewerWindowValidationPolicy = field(
        default_factory=ViewerWindowValidationPolicy
    )

    @property
    def expected_layer_count(self) -> int | None:
        return self.validation_policy.expected_layer_count

    @property
    def required_axis_labels(self) -> tuple[str, ...]:
        return self.validation_policy.required_axis_labels

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
    axis_labels: tuple[str, ...] = ()
    stack_axes: tuple[str, ...] = ()
    pending_update: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowLayerState(ViewerWindowLayerDescriptor):
    data_types: tuple[str, ...] = ()
    component_values: tuple[JsonObject, ...] = ()
    payload_summaries: tuple[JsonObject, ...] = ()
    axis_offsets: tuple[int, ...] = ()
    scalar_labels: tuple[str, ...] = ()
    labels: JsonObject = field(default_factory=dict)
    axis_component_values: JsonObject = field(default_factory=dict)
    routed_component_values: JsonObject = field(default_factory=dict)
    data_shape: tuple[int, ...] = ()
    translate: tuple[float, ...] = ()
    visible: bool = False
    selected: bool = False


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
    spatial_mismatch_count: int = 0
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
