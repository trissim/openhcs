"""DTOs for agent-facing viewer window interactions."""

from __future__ import annotations

from dataclasses import dataclass, field

from openhcs.agent.dto.common import (
    AgentError,
    AgentResourceRef,
    AgentResultEnvelope,
    JsonObject,
    SCHEMA_VERSION,
)
from openhcs.agent.dto.execution import ExecutionConnectionProjection, ExecutionConnectionSpec
from openhcs.runtime.window_snapshot import WindowSnapshotCaptureSpec


@dataclass(frozen=True, slots=True)
class ViewerWindowDescriptor:
    """Agent-facing identity for the viewer window that produced a resource."""

    viewer_type: str
    title: str


@dataclass(frozen=True, slots=True)
class ViewerWindowSnapshotRequest(ExecutionConnectionProjection):
    snapshot: WindowSnapshotCaptureSpec
    timeout_ms: int = 5000


@dataclass(frozen=True, slots=True)
class ViewerWindowStateRequest(ExecutionConnectionProjection):
    timeout_ms: int = 5000


@dataclass(frozen=True, slots=True)
class ViewerWindowValidationPolicy:
    """Validation contract applied to a viewer state observation."""

    expected_layer_count: int | None = None
    required_axis_labels: tuple[str, ...] = ()
    require_nonzero_payloads: bool = True


@dataclass(frozen=True, slots=True)
class ViewerWindowValidationRequest(ExecutionConnectionProjection):
    validation_policy: ViewerWindowValidationPolicy = field(
        default_factory=ViewerWindowValidationPolicy
    )
    timeout_ms: int = 5000

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


@dataclass(frozen=True, slots=True)
class ViewerWindowSnapshotResult(AgentResultEnvelope, ExecutionConnectionProjection):
    captured: bool
    resource: AgentResourceRef | None = None
    viewer: ViewerWindowDescriptor | None = None
    width: int | None = None
    height: int | None = None
    snapshot: WindowSnapshotCaptureSpec | None = None
    response: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ViewerWindowStateResult(AgentResultEnvelope, ExecutionConnectionProjection):
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


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowLayerValidationSummary(ViewerWindowLayerDescriptor):
    payload_count: int
    nonzero_payload_count: int
    zero_payload_count: int
    missing_nonzero_count: int
    coordinate_gap_count: int = 0
    expected_coordinate_count: int = 0
    routed_coordinate_count: int = 0
    payload_coordinate_count: int = 0
    missing_payload_coordinate_count: int = 0
    duplicate_payload_coordinate_count: int = 0
    payload_without_coordinate_count: int = 0
    spatial_mismatch_count: int = 0
    missing_required_axis_labels: tuple[str, ...] = ()
    valid: bool = False


@dataclass(frozen=True, slots=True)
class ViewerWindowValidationSummaryResult(
    AgentResultEnvelope,
    ExecutionConnectionProjection,
):
    valid: bool
    observed: bool = False
    viewer: ViewerWindowDescriptor | None = None
    layer_count: int = 0
    mounted_layer_count: int = 0
    pending_update_count: int = 0
    payload_count: int = 0
    nonzero_payload_count: int = 0
    zero_payload_count: int = 0
    missing_nonzero_count: int = 0
    missing_payload_coordinate_count: int = 0
    duplicate_payload_coordinate_count: int = 0
    payload_without_coordinate_count: int = 0
    spatial_mismatch_count: int = 0
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


def viewer_window_snapshot_error(
    *,
    connection: ExecutionConnectionSpec,
    error: AgentError,
) -> ViewerWindowSnapshotResult:
    return ViewerWindowSnapshotResult(
        schema_version=SCHEMA_VERSION,
        connection=connection,
        captured=False,
        errors=(error,),
    )


def viewer_window_state_error(
    *,
    connection: ExecutionConnectionSpec,
    error: AgentError,
) -> ViewerWindowStateResult:
    return ViewerWindowStateResult(
        schema_version=SCHEMA_VERSION,
        connection=connection,
        observed=False,
        errors=(error,),
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


def viewer_window_validation_error(
    *,
    connection: ExecutionConnectionSpec,
    error: AgentError,
    validation_policy: ViewerWindowValidationPolicy,
) -> ViewerWindowValidationSummaryResult:
    return ViewerWindowValidationSummaryResult(
        schema_version=SCHEMA_VERSION,
        connection=connection,
        valid=False,
        errors=(error,),
        validation_policy=validation_policy,
    )
