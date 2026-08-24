"""Agent service for running viewer window interactions."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import product
from typing import ClassVar, Generic, TypeVar

import zmq
from metaclass_registry import AutoRegisterMeta
from polystore.streaming.identity import StreamProducerIdentity
from pyqt_reactive.services.window_snapshot import WindowSnapshotCaptureSpec

import openhcs.core.plate_image_inventory as core_plate_image_inventory
from openhcs.agent.dto.common import (
    SCHEMA_VERSION,
    AgentError,
    AgentResourceRef,
    AgentWarning,
    JsonObject,
    JsonValue,
)
from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.dto.viewer import (
    ViewerWindowDescriptor,
    ViewerWindowImageSampleRequest,
    ViewerWindowImageSampleResult,
    ViewerWindowLayerIsolationRequest,
    ViewerWindowLayerIsolationResult,
    ViewerWindowLayerPayloads,
    ViewerWindowLayerState,
    ViewerWindowLayerValidationSummary,
    ViewerWindowLayerVisibilityRecord,
    ViewerWindowNavigationRequest,
    ViewerWindowNavigationResult,
    ViewerWindowPayloadRecord,
    ViewerWindowPayloadRequest,
    ViewerWindowPayloadResult,
    ViewerWindowProbeResult,
    ViewerWindowRoiSummaryRequest,
    ViewerWindowRoiSummaryResult,
    ViewerWindowSnapshotRequest,
    ViewerWindowSnapshotResult,
    ViewerWindowStateRequest,
    ViewerWindowStateResult,
    ViewerWindowValidationCounters,
    ViewerWindowValidationErrorContext,
    ViewerWindowValidationPolicy,
    ViewerWindowValidationRequest,
    ViewerWindowValidationSummaryResult,
    viewer_window_probe_from_state,
)
from openhcs.agent.path_policy import AgentPathPolicy, AgentPathPolicyError
from openhcs.runtime.viewer_component_system import (
    ComponentValue,
    ComponentValues,
    ViewerComponentMetadataPayload,
    ViewerComponentValueParser,
    ViewerLayerAxisProjection,
)
from openhcs.runtime.viewer_protocol import (
    ViewerControlField,
    ViewerControlMessageType,
    ViewerControlResponseField,
    ViewerDescriptorField,
    ViewerLayerField,
    ViewerPayloadField,
    ViewerPayloadSummaryField,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG

OptionalViewerFieldT = TypeVar("OptionalViewerFieldT")
ValidationWarningContextT = TypeVar("ValidationWarningContextT")


ComponentIndex = tuple[int, ...]
ComponentIndexSet = set[ComponentIndex]


@dataclass(frozen=True, slots=True)
class ViewerPayloadComponentProjection:
    """Viewer payload component values prepared for axis-coordinate projection."""

    components: Mapping[str, ComponentValue]
    aggregate_values: Mapping[str, tuple[ComponentValue, ...]]

    @classmethod
    def from_summary(
        cls,
        payload_summary: JsonObject,
    ) -> "ViewerPayloadComponentProjection":
        return cls(
            components=cls._payload_components(payload_summary),
            aggregate_values=cls._aggregate_component_values(payload_summary),
        )

    def projected_values(self, component: str) -> tuple[ComponentValue, ...]:
        if component in self.aggregate_values:
            return self.aggregate_values[component]
        if component not in self.components:
            raise ValueError(
                f"Viewer payload summary missing stack component {component!r}."
            )
        return (self.components[component],)

    def coordinate_components(
        self,
        *,
        projected_axis_components: Sequence[str],
        coordinate_values: Sequence[ComponentValue],
    ) -> dict[str, ComponentValue]:
        return {
            **self.components,
            **dict(
                zip(
                    projected_axis_components,
                    coordinate_values,
                    strict=True,
                )
            ),
        }

    @staticmethod
    def _payload_components(payload_summary: JsonObject) -> dict[str, ComponentValue]:
        components_payload = payload_summary.get("components")
        if not isinstance(components_payload, Mapping):
            raise ValueError("Viewer payload summary missing components.")
        return ViewerComponentMetadataPayload.component_map(
            components_payload,
            context="viewer payload summary",
        )

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
            ViewerComponentValueParser.parse(value, context=context) for value in values
        )


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
        payload_projection = ViewerPayloadComponentProjection.from_summary(
            payload_summary
        )
        projected_values = tuple(
            payload_projection.projected_values(component)
            for component in projection.projected_axis_components
        )
        return tuple(
            projection.coordinate_index(
                payload_projection.coordinate_components(
                    projected_axis_components=projection.projected_axis_components,
                    coordinate_values=coordinate_values,
                ),
                context="viewer payload summary",
            )
            for coordinate_values in product(*projected_values)
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
            self.missing_payload_coordinate_count == 0
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


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerWindowValidationTotals(ViewerWindowValidationCounters):
    """Aggregate validation counters for a viewer state."""

    mounted_layer_count: int = 0
    pending_update_count: int = 0

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
                layer.missing_payload_coordinate_count for layer in layer_summaries
            ),
            duplicate_payload_coordinate_count=sum(
                layer.duplicate_payload_coordinate_count for layer in layer_summaries
            ),
            payload_without_coordinate_count=sum(
                layer.payload_without_coordinate_count for layer in layer_summaries
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
            return ViewerWindowValidationSummaryResult.from_error_context(
                ViewerWindowValidationErrorContext(
                    connection=connection,
                    error=error,
                    validation_policy=request.validation_policy,
                )
            )

        validation_context = ViewerWindowValidationWarningContext(
            validation_policy=request.validation_policy,
            observed_layer_count=state.layer_count,
            active_dimension_label_route=state.active_dimension_label_route,
            layer_route_keys=tuple(layer.route_key for layer in state.layers),
        )
        layer_summaries = tuple(
            cls.layer_validation_summary(
                layer,
                validation_context=validation_context,
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
            valid=(
                layer_count_matches
                and validation_context.active_dimension_label_route_valid
                and all(layer.valid for layer in layer_summaries)
            ),
            warnings=cls.validation_warnings(
                validation_context=validation_context,
                layer_summaries=layer_summaries,
            ),
            observed=True,
            viewer=state.viewer,
            layer_count=state.layer_count,
            mounted_layer_count=totals.mounted_layer_count,
            pending_update_count=totals.pending_update_count,
            active_dimension_label_route=state.active_dimension_label_route,
            active_dimension_label_route_valid=(
                validation_context.active_dimension_label_route_valid
            ),
            payload_count=totals.payload_count,
            nonzero_payload_count=totals.nonzero_payload_count,
            zero_payload_count=totals.zero_payload_count,
            missing_nonzero_count=totals.missing_nonzero_count,
            missing_payload_coordinate_count=(totals.missing_payload_coordinate_count),
            duplicate_payload_coordinate_count=(
                totals.duplicate_payload_coordinate_count
            ),
            payload_without_coordinate_count=totals.payload_without_coordinate_count,
            validation_policy=request.validation_policy,
            layer_summaries=layer_summaries,
            state=state if request.include_state else None,
        )

    @classmethod
    def validation_warnings(
        cls,
        *,
        validation_context: ViewerWindowValidationWarningContext,
        layer_summaries: Sequence[ViewerWindowLayerValidationSummary],
    ) -> tuple[AgentWarning, ...]:
        warnings: list[AgentWarning] = []
        warnings.extend(ViewerWindowValidationWarningRule.warnings(validation_context))
        for layer in layer_summaries:
            warnings.extend(
                cls.layer_validation_warnings(
                    layer=layer,
                    validation_context=validation_context,
                )
            )
        return tuple(warnings)

    @classmethod
    def layer_validation_warnings(
        cls,
        *,
        layer: ViewerWindowLayerValidationSummary,
        validation_context: ViewerValidationPolicyCarrier,
    ) -> tuple[AgentWarning, ...]:
        return ViewerLayerValidationWarningRule.warnings(
            ViewerLayerValidationWarningContext(
                validation_policy=validation_context.validation_policy,
                layer=layer,
            )
        )

    @classmethod
    def layer_validation_summary(
        cls,
        layer: ViewerWindowLayerState,
        *,
        validation_context: ViewerValidationPolicyCarrier,
    ) -> ViewerWindowLayerValidationSummary:
        validation_policy = validation_context.validation_policy
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
        component_labels = cls.component_labels(layer)
        missing_required_component_labels = tuple(
            component_label
            for component_label in validation_policy.required_component_labels
            if component_label not in component_labels
        )
        axis_labels_present_as_components = tuple(
            axis_label
            for axis_label in missing_required_axis_labels
            if axis_label in component_labels
        )
        coordinate_coverage = ViewerLayerCoordinateCoverage.from_layer(layer)
        nonzero_valid = not validation_policy.require_nonzero_payloads or (
            payload_count > 0 and zero_payload_count == 0 and missing_nonzero_count == 0
        )
        valid = (
            layer.mounted
            and not layer.pending_update
            and not missing_required_axis_labels
            and not missing_required_component_labels
            and nonzero_valid
            and coordinate_coverage.valid
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
            axis_labels=layer.axis_labels,
            stack_axes=layer.stack_axes,
            missing_required_axis_labels=missing_required_axis_labels,
            component_labels=component_labels,
            missing_required_component_labels=missing_required_component_labels,
            axis_labels_present_as_components=axis_labels_present_as_components,
            pending_update=layer.pending_update,
            valid=valid,
        )

    @staticmethod
    def component_labels(layer: ViewerWindowLayerState) -> tuple[str, ...]:
        component_labels: set[str] = set(layer.axis_component_values.keys())
        component_labels.update(layer.routed_component_values.keys())
        for component_values in layer.component_values:
            component_labels.update(
                str(component) for component in component_values.keys()
            )
        for payload_summary in layer.payload_summaries:
            components = payload_summary.get("components")
            if isinstance(components, Mapping):
                component_labels.update(
                    str(component) for component in components.keys()
                )
            aggregate_values = payload_summary.get("aggregate_component_values")
            if isinstance(aggregate_values, Mapping):
                component_labels.update(
                    str(component) for component in aggregate_values.keys()
                )
        return tuple(sorted(component_labels))

    @staticmethod
    def payload_nonzero_count(payload_summary: JsonObject) -> int | None:
        field_name = ViewerPayloadSummaryField.NONZERO_COUNT
        if field_name not in payload_summary:
            return None
        value = payload_summary[field_name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Viewer payload nonzero_count must be an integer.")
        return value


class ViewerValidationWarningCode:
    """Warning codes emitted by viewer state validation."""

    LAYER_COUNT_MISMATCH = "viewer_layer_count_mismatch"
    ACTIVE_DIMENSION_LABEL_ROUTE_MISSING = "viewer_active_dimension_label_route_missing"
    LAYER_UNMOUNTED = "viewer_layer_unmounted"
    LAYER_PENDING_UPDATE = "viewer_layer_pending_update"
    REQUIRED_AXIS_LABELS_MISSING = "viewer_required_axis_labels_missing"
    REQUIRED_COMPONENT_LABELS_MISSING = "viewer_required_component_labels_missing"
    PAYLOAD_NONZERO_METADATA_MISSING = "viewer_payload_nonzero_metadata_missing"
    PAYLOADS_ZERO = "viewer_payloads_zero"
    COORDINATE_GAPS = "viewer_layer_coordinate_gaps"
    PAYLOAD_COORDINATES_MISSING = "viewer_payload_coordinates_missing"
    PAYLOAD_COORDINATES_DUPLICATE = "viewer_payload_coordinates_duplicate"
    PAYLOADS_WITHOUT_COORDINATES = "viewer_payloads_without_coordinates"


@dataclass(frozen=True, kw_only=True)
class ViewerValidationPolicyCarrier:
    """Shared validation policy carrier for viewer validation projections."""

    validation_policy: ViewerWindowValidationPolicy


@dataclass(frozen=True, slots=True)
class ViewerExpectedLayerCountEvidence:
    """Expected/observed layer-count comparison owned by one nominal record."""

    expected_layer_count: int | None
    observed_layer_count: int

    @classmethod
    def from_context(
        cls,
        context: "ViewerWindowValidationWarningContext",
    ) -> "ViewerExpectedLayerCountEvidence":
        return cls(
            expected_layer_count=context.validation_policy.expected_layer_count,
            observed_layer_count=context.observed_layer_count,
        )

    @property
    def matches(self) -> bool:
        return (
            self.expected_layer_count is None
            or self.expected_layer_count == self.observed_layer_count
        )


@dataclass(frozen=True, slots=True)
class ViewerWindowValidationWarningContext(ViewerValidationPolicyCarrier):
    """Validation warning facts that apply to the whole viewer window."""

    observed_layer_count: int
    active_dimension_label_route: str | None
    layer_route_keys: tuple[str, ...]

    @property
    def active_dimension_label_route_valid(self) -> bool:
        return (
            self.active_dimension_label_route is None
            or self.active_dimension_label_route in self.layer_route_keys
        )


@dataclass(frozen=True, slots=True)
class ViewerLayerValidationWarningContext(ViewerValidationPolicyCarrier):
    """Validation warning facts that apply to one projected viewer layer."""

    layer: ViewerWindowLayerValidationSummary

    @property
    def layer_name(self) -> str:
        if self.layer.title is not None:
            return self.layer.title
        return self.layer.route_key


class ViewerValidationWarningRuleBase(ABC, Generic[ValidationWarningContextT]):
    """Shared registry projection for nominal validation warning rules."""

    @classmethod
    def warnings(
        cls: type["ViewerValidationWarningRuleBase[ValidationWarningContextT]"],
        context: ValidationWarningContextT,
    ) -> tuple[AgentWarning, ...]:
        warnings: list[AgentWarning] = []
        for rule_type in cls.__registry__.values():
            warning = rule_type().warning(context)
            if warning is not None:
                warnings.append(warning)
        return tuple(warnings)

    @abstractmethod
    def warning(
        self,
        context: ValidationWarningContextT,
    ) -> AgentWarning | None:
        """Return one warning when this rule applies."""


class ViewerWarningCodeBase(ABC):
    """Shared registry-key declaration for viewer validation warning families."""

    __registry_key__ = "warning_code"
    __skip_if_no_key__ = True
    warning_code: ClassVar[str | None] = None


class ViewerWindowValidationWarningRule(
    ViewerWarningCodeBase,
    ViewerValidationWarningRuleBase[ViewerWindowValidationWarningContext],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal warning rule family for whole-window validation facts."""


class ViewerLayerValidationWarningRule(
    ViewerWarningCodeBase,
    ViewerValidationWarningRuleBase[ViewerLayerValidationWarningContext],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal warning rule family for per-layer validation facts."""


class ViewerLayerCountMismatchWarningRule(ViewerWindowValidationWarningRule):
    """Warn when the observed layer count violates the requested policy."""

    warning_code = ViewerValidationWarningCode.LAYER_COUNT_MISMATCH

    def warning(
        self,
        context: ViewerWindowValidationWarningContext,
    ) -> AgentWarning | None:
        evidence = ViewerExpectedLayerCountEvidence.from_context(context)
        if evidence.matches:
            return None
        return AgentWarning(
            code=self.warning_code,
            message=(
                "Viewer layer count did not match the expected layer count: "
                f"expected {evidence.expected_layer_count}, "
                f"observed {evidence.observed_layer_count}."
            ),
        )


class ViewerActiveDimensionLabelRouteMissingWarningRule(
    ViewerWindowValidationWarningRule
):
    """Warn when a declared active dimension-label route is not observed."""

    warning_code = ViewerValidationWarningCode.ACTIVE_DIMENSION_LABEL_ROUTE_MISSING

    def warning(
        self,
        context: ViewerWindowValidationWarningContext,
    ) -> AgentWarning | None:
        if context.active_dimension_label_route_valid:
            return None
        return AgentWarning(
            code=self.warning_code,
            message=(
                "Viewer active dimension-label route "
                f"{context.active_dimension_label_route!r} did not name an "
                "observed layer route."
            ),
        )


class ViewerLayerUnmountedWarningRule(ViewerLayerValidationWarningRule):
    """Warn when a layer route has no mounted viewer layer."""

    warning_code = ViewerValidationWarningCode.LAYER_UNMOUNTED

    def warning(
        self,
        context: ViewerLayerValidationWarningContext,
    ) -> AgentWarning | None:
        if context.layer.mounted:
            return None
        return AgentWarning(
            code=self.warning_code,
            message=f"Viewer layer {context.layer_name!r} is not mounted.",
        )


class ViewerLayerPendingUpdateWarningRule(ViewerLayerValidationWarningRule):
    """Warn when a layer still has unapplied viewer updates."""

    warning_code = ViewerValidationWarningCode.LAYER_PENDING_UPDATE

    def warning(
        self,
        context: ViewerLayerValidationWarningContext,
    ) -> AgentWarning | None:
        if not context.layer.pending_update:
            return None
        return AgentWarning(
            code=self.warning_code,
            message=f"Viewer layer {context.layer_name!r} still has a pending update.",
        )


class ViewerRequiredAxisLabelsMissingWarningRule(ViewerLayerValidationWarningRule):
    """Warn when a layer is missing policy-required axis labels."""

    warning_code = ViewerValidationWarningCode.REQUIRED_AXIS_LABELS_MISSING

    def warning(
        self,
        context: ViewerLayerValidationWarningContext,
    ) -> AgentWarning | None:
        if not context.layer.missing_required_axis_labels:
            return None
        missing = ", ".join(context.layer.missing_required_axis_labels)
        component_present = ", ".join(context.layer.axis_labels_present_as_components)
        hint = None
        if component_present:
            hint = (
                "These labels are present as component metadata rather than "
                f"mounted axes: {component_present}. Use required_component_labels "
                "when singleton biological components are acceptable."
            )
        return AgentWarning(
            code=self.warning_code,
            message=(
                f"Viewer layer {context.layer_name!r} is missing required axis "
                f"labels: {missing}."
            ),
            hint=hint,
        )


class ViewerRequiredComponentLabelsMissingWarningRule(ViewerLayerValidationWarningRule):
    """Warn when a layer is missing policy-required component labels."""

    warning_code = ViewerValidationWarningCode.REQUIRED_COMPONENT_LABELS_MISSING

    def warning(
        self,
        context: ViewerLayerValidationWarningContext,
    ) -> AgentWarning | None:
        if not context.layer.missing_required_component_labels:
            return None
        missing = ", ".join(context.layer.missing_required_component_labels)
        return AgentWarning(
            code=self.warning_code,
            message=(
                f"Viewer layer {context.layer_name!r} is missing required "
                f"component labels: {missing}."
            ),
        )


class ViewerPayloadCountWarningRule(ViewerLayerValidationWarningRule, ABC):
    """Template for nonzero-payload validation warnings driven by layer counts."""

    payload_count_description: ClassVar[str]

    def warning(
        self,
        context: ViewerLayerValidationWarningContext,
    ) -> AgentWarning | None:
        if not context.validation_policy.require_nonzero_payloads:
            return None
        count = self.count(context.layer)
        if not count:
            return None
        return AgentWarning(
            code=self.warning_code,
            message=self.message(context, count),
        )

    @abstractmethod
    def count(self, layer: ViewerWindowLayerValidationSummary) -> int:
        """Return the count that activates this warning."""

    @abstractmethod
    def message(
        self,
        context: ViewerLayerValidationWarningContext,
        count: int,
    ) -> str:
        return (
            f"Viewer layer {context.layer_name!r} has "
            f"{count} {self.payload_count_description}."
        )


class ViewerPayloadNonzeroMetadataMissingWarningRule(ViewerPayloadCountWarningRule):
    """Warn when nonzero validation is enabled but metadata is absent."""

    warning_code = ViewerValidationWarningCode.PAYLOAD_NONZERO_METADATA_MISSING
    payload_count_description = "payload summaries without nonzero_count metadata"

    def count(self, layer: ViewerWindowLayerValidationSummary) -> int:
        return layer.missing_nonzero_count


class ViewerPayloadsZeroWarningRule(ViewerPayloadCountWarningRule):
    """Warn when nonzero validation is enabled and payload summaries are empty."""

    warning_code = ViewerValidationWarningCode.PAYLOADS_ZERO
    payload_count_description = "empty payload summaries"

    def count(self, layer: ViewerWindowLayerValidationSummary) -> int:
        return layer.zero_payload_count


class ViewerCoordinateGapsWarningRule(ViewerLayerValidationWarningRule):
    """Warn when the viewer coordinate domain has unrouted positions."""

    warning_code = ViewerValidationWarningCode.COORDINATE_GAPS

    def warning(
        self,
        context: ViewerLayerValidationWarningContext,
    ) -> AgentWarning | None:
        if not context.layer.coordinate_gap_count:
            return None
        return AgentWarning(
            code=self.warning_code,
            message=(
                f"Viewer layer {context.layer_name!r} has "
                f"{context.layer.coordinate_gap_count} viewer coordinate "
                "position(s) with no routed payload."
            ),
        )


class ViewerPayloadCoordinatesMissingWarningRule(ViewerLayerValidationWarningRule):
    """Warn when routed payload coordinates are absent from a layer."""

    warning_code = ViewerValidationWarningCode.PAYLOAD_COORDINATES_MISSING

    def warning(
        self,
        context: ViewerLayerValidationWarningContext,
    ) -> AgentWarning | None:
        if not context.layer.missing_payload_coordinate_count:
            return None
        return AgentWarning(
            code=self.warning_code,
            message=(
                f"Viewer layer {context.layer_name!r} is missing "
                f"{context.layer.missing_payload_coordinate_count} routed "
                "payload coordinate(s)."
            ),
        )


class ViewerPayloadCoordinatesDuplicateWarningRule(ViewerLayerValidationWarningRule):
    """Warn when multiple payloads claim the same routed coordinate."""

    warning_code = ViewerValidationWarningCode.PAYLOAD_COORDINATES_DUPLICATE

    def warning(
        self,
        context: ViewerLayerValidationWarningContext,
    ) -> AgentWarning | None:
        if not context.layer.duplicate_payload_coordinate_count:
            return None
        return AgentWarning(
            code=self.warning_code,
            message=(
                f"Viewer layer {context.layer_name!r} has "
                f"{context.layer.duplicate_payload_coordinate_count} duplicate "
                "payload coordinate(s)."
            ),
        )


class ViewerPayloadsWithoutCoordinatesWarningRule(ViewerLayerValidationWarningRule):
    """Warn when payload records cannot be projected into viewer coordinates."""

    warning_code = ViewerValidationWarningCode.PAYLOADS_WITHOUT_COORDINATES

    def warning(
        self,
        context: ViewerLayerValidationWarningContext,
    ) -> AgentWarning | None:
        if not context.layer.payload_without_coordinate_count:
            return None
        return AgentWarning(
            code=self.warning_code,
            message=(
                f"Viewer layer {context.layer_name!r} has "
                f"{context.layer.payload_without_coordinate_count} payload(s) "
                "without a valid routed coordinate."
            ),
        )


class ViewerWindowGatewayABC(ABC):
    """Transport boundary for interacting with running viewer windows."""

    @abstractmethod
    def snapshot_window(self, request: ViewerWindowSnapshotRequest) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def window_state(self, request: ViewerWindowStateRequest) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def window_payloads(self, request: ViewerWindowPayloadRequest) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def navigate_window(self, request: ViewerWindowNavigationRequest) -> JsonObject:
        raise NotImplementedError


class ZMQViewerWindowGateway(ViewerWindowGatewayABC):
    """Viewer gateway backed by the existing ZMQ control socket."""

    def __init__(
        self, context_factory: Callable[[], zmq.Context] = zmq.Context
    ) -> None:
        self._context_factory = context_factory

    def snapshot_window(self, request: ViewerWindowSnapshotRequest) -> JsonObject:
        message = {
            ViewerControlResponseField.TYPE: ViewerControlMessageType.SCREENSHOT.value,
            ViewerControlResponseField.PAYLOAD.value: request,
        }
        return self._send_control_message(request, message)

    def window_state(self, request: ViewerWindowStateRequest) -> JsonObject:
        message: dict[str, object] = {
            ViewerControlResponseField.TYPE: ViewerControlMessageType.STATE.value,
            ViewerControlResponseField.PAYLOAD.value: request.state_controls,
        }
        return self._send_control_message(request, message)

    def window_payloads(self, request: ViewerWindowPayloadRequest) -> JsonObject:
        message: dict[str, object] = {
            ViewerControlResponseField.TYPE: ViewerControlMessageType.PAYLOADS.value,
            ViewerControlResponseField.PAYLOAD.value: request.payload_controls,
        }
        return self._send_control_message(request, message)

    def navigate_window(self, request: ViewerWindowNavigationRequest) -> JsonObject:
        message: dict[str, object] = {
            ViewerControlResponseField.TYPE: ViewerControlMessageType.NAVIGATE.value,
            ViewerControlResponseField.PAYLOAD.value: request.navigation,
        }
        return self._send_control_message(request, message)

    def _send_control_message(
        self,
        request: (
            ViewerWindowSnapshotRequest
            | ViewerWindowStateRequest
            | ViewerWindowPayloadRequest
            | ViewerWindowNavigationRequest
        ),
        message: Mapping[str, object],
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

    def __init__(
        self,
        gateway: ViewerWindowGatewayABC | None = None,
        path_policy: AgentPathPolicy | None = None,
    ) -> None:
        if gateway is None:
            self._gateway = ZMQViewerWindowGateway()
        else:
            self._gateway = gateway
        self._path_policy = path_policy or AgentPathPolicy.from_environment()

    def snapshot_window(
        self,
        request: ViewerWindowSnapshotRequest,
    ) -> ViewerWindowSnapshotResult:
        try:
            request = self._writable_snapshot_request(request)
        except AgentPathPolicyError as exc:
            return ViewerWindowSnapshotResult.from_request_error(
                request=request,
                error=exc.to_agent_error(),
            )
        connection = request.connection
        try:
            response = self._gateway.snapshot_window(request)
        except Exception as exc:
            return ViewerWindowSnapshotResult.from_request_error(
                request=request,
                error=AgentError.from_exception("viewer_window_snapshot_failed", exc),
            )

        try:
            return self._snapshot_result_from_response(
                connection=connection,
                request=request,
                response=response,
            )
        except Exception as exc:
            return ViewerWindowSnapshotResult.from_request_error(
                request=request,
                error=AgentError.from_exception(
                    "viewer_window_snapshot_response_invalid", exc
                ),
            )

    def _writable_snapshot_request(
        self,
        request: ViewerWindowSnapshotRequest,
    ) -> ViewerWindowSnapshotRequest:
        return replace(
            request,
            output_dir_path=str(
                self._path_policy.assert_writable(request.output_dir_path)
            ),
        )

    def _snapshot_result_from_response(
        self,
        *,
        connection: ExecutionConnectionSpec,
        request: ViewerWindowSnapshotRequest,
        response: JsonObject,
    ) -> ViewerWindowSnapshotResult:
        status = self._required_scalar(
            response, ViewerControlResponseField.STATUS, str, "a string"
        )
        if status != self.SUCCESS_STATUS:
            message = self._required_scalar(
                response, ViewerControlResponseField.MESSAGE, str, "a string"
            )
            return ViewerWindowSnapshotResult.from_request_error(
                request=request,
                error=AgentError(code="viewer_window_snapshot_failed", message=message),
            )
        response_snapshot = response.get(ViewerControlField.SNAPSHOT.value)
        if not isinstance(response_snapshot, WindowSnapshotCaptureSpec):
            raise TypeError(
                "Viewer screenshot response must carry its nominal snapshot contract."
            )
        if not response_snapshot.same_capture_contract(request):
            return ViewerWindowSnapshotResult.from_request_error(
                request=request,
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
            output_dir_path=request.output_dir_path,
            capture_scope=request.capture_scope,
            captured=True,
            resource=AgentResourceRef(
                uri=self._required_scalar(resource_payload, "uri", str, "a string"),
                title=self._required_scalar(resource_payload, "title", str, "a string"),
                mime_type=self._required_scalar(
                    resource_payload, "mime_type", str, "a string"
                ),
                path=self._optional_typed(resource_payload, "path", str),
                size_bytes=self._optional_typed(resource_payload, "size_bytes", int),
                sha256=self._optional_typed(resource_payload, "sha256", str),
            ),
            viewer=ViewerWindowDescriptor.from_wire_fields(
                viewer_wire_value=self._required_scalar(
                    viewer_payload,
                    ViewerDescriptorField.TYPE,
                    str,
                    "a string",
                ),
                title=self._required_scalar(
                    viewer_payload,
                    ViewerDescriptorField.TITLE,
                    str,
                    "a string",
                ),
            ),
            width=self._optional_typed(response, ViewerControlField.WIDTH, int),
            height=self._optional_typed(response, ViewerControlField.HEIGHT, int),
            response=response,
        )

    def window_state(
        self,
        request: ViewerWindowStateRequest,
    ) -> ViewerWindowStateResult:
        connection = request.connection
        try:
            response = self._gateway.window_state(request)
        except Exception as exc:
            return ViewerWindowStateResult.from_error(
                connection=connection,
                error=AgentError.from_exception("viewer_window_state_failed", exc),
            )

        try:
            return self._state_result_from_response(
                connection=connection,
                response=response,
                include_response=request.include_response,
            )
        except Exception as exc:
            return ViewerWindowStateResult.from_error(
                connection=connection,
                error=AgentError.from_exception(
                    "viewer_window_state_response_invalid", exc
                ),
            )

    def probe_window(
        self,
        request: ViewerWindowStateRequest,
    ) -> ViewerWindowProbeResult:
        state = self.window_state(request)
        return viewer_window_probe_from_state(state)

    def window_payloads(
        self,
        request: ViewerWindowPayloadRequest,
    ) -> ViewerWindowPayloadResult:
        connection = request.connection
        try:
            response = self._gateway.window_payloads(request)
        except Exception as exc:
            return ViewerWindowPayloadResult.from_error(
                connection=connection,
                error=AgentError.from_exception("viewer_window_payloads_failed", exc),
            )

        try:
            return self._payload_result_from_response(
                connection=connection,
                response=response,
                include_response=request.include_response,
            )
        except Exception as exc:
            return ViewerWindowPayloadResult.from_error(
                connection=connection,
                error=AgentError.from_exception(
                    "viewer_window_payloads_response_invalid", exc
                ),
            )

    def navigate_window(
        self,
        request: ViewerWindowNavigationRequest,
    ) -> ViewerWindowNavigationResult:
        connection = request.connection
        try:
            response = self._gateway.navigate_window(request)
        except Exception as exc:
            return ViewerWindowNavigationResult.from_error(
                connection=connection,
                error=AgentError.from_exception("viewer_window_navigation_failed", exc),
            )

        try:
            return self._navigation_result_from_response(
                connection=connection,
                request=request,
                response=response,
            )
        except Exception as exc:
            return ViewerWindowNavigationResult.from_error(
                connection=connection,
                error=AgentError.from_exception(
                    "viewer_window_navigation_response_invalid", exc
                ),
            )

    def isolate_layers(
        self,
        request: ViewerWindowLayerIsolationRequest,
    ) -> ViewerWindowLayerIsolationResult:
        state = self.window_state(request.state_request())
        if state.errors:
            return ViewerWindowLayerIsolationResult(
                schema_version=SCHEMA_VERSION,
                observed=state.observed,
                applied=False,
                errors=state.errors,
                warnings=state.warnings,
            )

        mounted_layers = tuple(layer for layer in state.layers if layer.mounted)
        layer_routes = {layer.route_key for layer in mounted_layers}
        missing_routes = tuple(
            route_key
            for route_key in request.visible_routes
            if route_key not in layer_routes
        )
        if missing_routes:
            return ViewerWindowLayerIsolationResult(
                schema_version=SCHEMA_VERSION,
                observed=state.observed,
                applied=False,
                missing_route_keys=missing_routes,
                available_layers=self._layer_visibility_records(mounted_layers),
                errors=(
                    AgentError(
                        code="viewer_layer_route_missing",
                        message=(
                            "Viewer layer route key(s) are not mounted: "
                            + ", ".join(missing_routes)
                        ),
                    ),
                ),
                warnings=state.warnings,
            )

        navigation_errors: list[AgentError] = []
        changed_routes: list[str] = []
        for layer in mounted_layers:
            route_visible = layer.route_key in request.visible_routes
            route_selected = layer.route_key == request.selected_route
            navigation = self.navigate_window(
                request.navigation_request(
                    route_key=layer.route_key,
                    visible=route_visible,
                    selected=route_selected,
                )
            )
            if navigation.errors:
                navigation_errors.extend(navigation.errors)
            else:
                changed_routes.append(layer.route_key)

        final_state = self.window_state(request.state_request())
        if final_state.errors:
            return ViewerWindowLayerIsolationResult(
                schema_version=SCHEMA_VERSION,
                observed=final_state.observed,
                applied=not navigation_errors,
                selected_route_key=request.selected_route,
                visible_route_keys=request.requested_visible_route_keys,
                hidden_route_keys=tuple(
                    layer.route_key
                    for layer in mounted_layers
                    if layer.route_key not in request.visible_routes
                ),
                changed_route_count=len(changed_routes),
                layer_count=len(mounted_layers),
                active_dimension_label_route=None,
                current_step=(),
                axis_labels=(),
                available_layers=self._layer_visibility_records(mounted_layers),
                visible_layers=tuple(
                    ViewerWindowLayerVisibilityRecord(
                        route_key=layer.route_key,
                        title=layer.title,
                        visible=layer.route_key in request.visible_routes,
                        selected=layer.route_key == request.selected_route,
                    )
                    for layer in mounted_layers
                    if layer.route_key in request.visible_routes
                ),
                errors=tuple((*navigation_errors, *final_state.errors)),
                warnings=final_state.warnings,
            )

        return ViewerWindowLayerIsolationResult(
            schema_version=SCHEMA_VERSION,
            observed=final_state.observed,
            applied=not navigation_errors and not final_state.errors,
            selected_route_key=request.selected_route,
            visible_route_keys=tuple(
                layer.route_key
                for layer in final_state.layers
                if layer.mounted and layer.visible
            ),
            hidden_route_keys=tuple(
                layer.route_key
                for layer in final_state.layers
                if layer.mounted and not layer.visible
            ),
            changed_route_count=len(changed_routes),
            layer_count=sum(1 for layer in final_state.layers if layer.mounted),
            active_dimension_label_route=final_state.active_dimension_label_route,
            current_step=final_state.current_step,
            axis_labels=final_state.axis_labels,
            available_layers=self._layer_visibility_records(final_state.layers),
            visible_layers=tuple(
                ViewerWindowLayerVisibilityRecord(
                    route_key=layer.route_key,
                    title=layer.title,
                    visible=layer.visible,
                    selected=layer.selected,
                )
                for layer in final_state.layers
                if layer.mounted and layer.visible
            ),
            errors=tuple((*navigation_errors, *final_state.errors)),
            warnings=final_state.warnings,
        )

    def sample_image(
        self,
        request: ViewerWindowImageSampleRequest,
    ) -> ViewerWindowImageSampleResult:
        result = self.window_payloads(request.payload_request())
        raw_image_records = tuple(
            {
                "layer_route_key": layer.route_key,
                "layer_title": layer.title,
                "payload_route_key": payload.route_key,
                "data_type": payload.data_type,
                "path": payload.path,
                "components": payload.components,
                "axis_indices": payload.axis_indices,
                "aggregate_axis_indices": payload.aggregate_axis_indices,
                "summary": payload.summary,
                "array_value_summary": payload.array_value_summary,
                "array_values": payload.array_values,
            }
            for layer in result.layers
            for payload in layer.payloads
            if payload.data_type == "image"
        )
        image_layer_route_keys = tuple(
            dict.fromkeys(
                record["layer_route_key"]
                for record in raw_image_records
                if isinstance(record["layer_route_key"], str)
            )
        )
        requested_route_key = request.route_key
        resolved_route_key = request.route_key
        auto_selected_route_key = None
        route_selection_failed = False
        local_errors: list[AgentError] = []
        local_warnings: list[AgentWarning] = []

        if request.route_key is None:
            if len(image_layer_route_keys) == 1:
                auto_selected_route_key = image_layer_route_keys[0]
                resolved_route_key = auto_selected_route_key
                local_warnings.append(
                    AgentWarning(
                        code="viewer_image_route_auto_selected",
                        message=(
                            "No route_key was supplied; MCP selected the only "
                            "viewer layer containing image payloads."
                        ),
                        hint=(
                            f"Pass route_key={auto_selected_route_key!r} to make "
                            "future sampling explicit."
                        ),
                    )
                )
            elif len(image_layer_route_keys) > 1:
                local_errors.append(
                    AgentError(
                        code="viewer_image_route_ambiguous",
                        message=(
                            "No route_key was supplied and the viewer contains "
                            "multiple layers with image payloads."
                        ),
                        hint="Pass one of candidate_image_route_keys as route_key.",
                    )
                )
                route_selection_failed = True
            elif not result.errors:
                local_errors.append(
                    AgentError(
                        code="viewer_image_layer_unavailable",
                        message="Viewer payload query returned no image payload records.",
                        hint=(
                            "Use openhcs_get_viewer_window_state or "
                            "openhcs_get_viewer_window_payloads to inspect available "
                            "layers, or stream an image layer first."
                        ),
                    )
                )
                route_selection_failed = True

        if route_selection_failed:
            image_records = ()
        elif resolved_route_key is None:
            image_records = raw_image_records
        else:
            image_records = tuple(
                record
                for record in raw_image_records
                if record["layer_route_key"] == resolved_route_key
            )
        axis_filter_applied_by_viewer = True
        client_side_axis_filter_applied = False
        if isinstance(request.axis_indices, tuple):
            route_filtered_image_records = image_records
            image_records = tuple(
                record
                for record in route_filtered_image_records
                if tuple(record["axis_indices"]) == request.axis_indices
            )
            axis_filter_applied_by_viewer = len(image_records) == len(
                route_filtered_image_records
            )
            client_side_axis_filter_applied = not axis_filter_applied_by_viewer
            if client_side_axis_filter_applied:
                local_warnings.append(
                    AgentWarning(
                        code="viewer_payload_axis_filter_not_applied",
                        message=(
                            "Viewer returned image payload records outside the "
                            "requested axis_indices; MCP filtered them client-side."
                        ),
                        hint=(
                            "The viewer process may be stale. Restart and re-stream "
                            "the viewer if bounded sampling should happen in-process."
                        ),
                    )
                )
        sample_protocol_supported = any(
            "requested" in record["array_value_summary"] for record in image_records
        )
        if image_records and not sample_protocol_supported:
            local_warnings.append(
                AgentWarning(
                    code="viewer_payload_array_sampling_unavailable",
                    message=(
                        "Viewer payload records did not include array_value_summary; "
                        "bounded array sampling was not applied by the viewer."
                    ),
                    hint=(
                        "Restart the viewer process and re-stream the layer so it "
                        "loads the current viewer payload protocol."
                    ),
                )
            )
        matched_image_record_count = len(image_records)
        returned_image_records = image_records[: request.max_records]
        sample_included_count = sum(
            1
            for record in returned_image_records
            if record["array_value_summary"].get("included") is True
        )
        total_record_count = sum(len(layer.payloads) for layer in result.layers)
        raw_image_record_count = len(raw_image_records)
        return ViewerWindowImageSampleResult(
            schema_version=SCHEMA_VERSION,
            observed=result.observed,
            route_key=resolved_route_key,
            requested_route_key=requested_route_key,
            auto_selected_route_key=auto_selected_route_key,
            candidate_image_route_keys=image_layer_route_keys,
            axis_indices=request.axis_indices,
            array_slices=request.array_slices,
            record_count=matched_image_record_count,
            returned_record_count=len(returned_image_records),
            records_truncated_count=(
                matched_image_record_count - len(returned_image_records)
            ),
            total_payload_record_count=total_record_count,
            raw_image_record_count=raw_image_record_count,
            filtered_out_image_record_count=(
                raw_image_record_count - matched_image_record_count
            ),
            non_image_record_count=total_record_count - raw_image_record_count,
            axis_filter_applied_by_viewer=axis_filter_applied_by_viewer,
            client_side_axis_filter_applied=client_side_axis_filter_applied,
            sample_protocol_supported=sample_protocol_supported,
            sample_included_count=sample_included_count,
            sample_omitted_count=(len(returned_image_records) - sample_included_count),
            records=returned_image_records,
            errors=(*result.errors, *local_errors),
            warnings=(*result.warnings, *local_warnings),
        )

    def summarize_rois(
        self,
        request: ViewerWindowRoiSummaryRequest,
    ) -> ViewerWindowRoiSummaryResult:
        result = self.window_payloads(request.payload_request())

        payload_summaries = []
        payload_type_counts: dict[str, int] = {}
        payload_record_count = 0
        total_roi_count = 0
        returned_roi_count = 0
        total_roi_member_count = 0
        returned_roi_member_count = 0
        roi_count_exact = True
        for layer in result.layers:
            for payload in layer.payloads:
                payload_record_count += 1
                payload_type_counts[payload.data_type] = (
                    payload_type_counts.get(payload.data_type, 0) + 1
                )
                if payload.data_type != "shapes":
                    continue
                shape_payload_count = int(
                    payload.summary.get(
                        "shape_payload_count",
                        len(payload.shape_payloads),
                    )
                )
                returned_shape_payload_count = len(payload.shape_payloads)
                payload_truncated = returned_shape_payload_count < shape_payload_count
                semantic_payloads = self._semantic_shape_payloads(
                    payload.shape_payloads
                )
                total_roi_count += len(semantic_payloads)
                returned_roi_count += len(semantic_payloads)
                total_roi_member_count += shape_payload_count
                returned_roi_member_count += returned_shape_payload_count
                roi_count_exact = roi_count_exact and not payload_truncated
                duplicate_member_count = (
                    max(0, shape_payload_count - len(semantic_payloads))
                    if not payload_truncated
                    else max(0, returned_shape_payload_count - len(semantic_payloads))
                )
                areas = self._numeric_metadata(semantic_payloads, "area")
                perimeters = self._numeric_metadata(semantic_payloads, "perimeter")
                payload_summaries.append(
                    {
                        "layer_route_key": layer.route_key,
                        "layer_title": layer.title,
                        "payload_route_key": payload.route_key,
                        "path": payload.path,
                        "components": payload.components,
                        "axis_indices": payload.axis_indices,
                        "roi_count": len(semantic_payloads),
                        "returned_roi_count": len(semantic_payloads),
                        "roi_count_exact": not payload_truncated,
                        "roi_member_count": shape_payload_count,
                        "returned_roi_member_count": returned_shape_payload_count,
                        "roi_duplicate_member_count": duplicate_member_count,
                        "roi_payloads_truncated": payload_truncated,
                        "area": self._numeric_stats(areas),
                        "perimeter": self._numeric_stats(perimeters),
                        "bounds_yx": payload.summary.get("shape_coordinate_bounds_yx"),
                        "coordinate_count": payload.summary.get(
                            "shape_coordinate_count"
                        ),
                        "spatial_origin_yx": payload.summary.get("spatial_origin_yx"),
                        "source_spatial_shape_yx": payload.summary.get(
                            "source_spatial_shape_yx"
                        ),
                        "out_of_source_bounds_count": payload.summary.get(
                            "shape_out_of_source_bounds_count"
                        ),
                        "example_rois": tuple(
                            self._example_roi(shape_payload)
                            for shape_payload in semantic_payloads[
                                : request.max_examples
                            ]
                        ),
                    }
                )

        return ViewerWindowRoiSummaryResult(
            schema_version=SCHEMA_VERSION,
            observed=result.observed,
            route_key=request.route_key,
            axis_indices=request.axis_indices,
            layer_count=result.layer_count,
            payload_record_count=payload_record_count,
            payload_type_counts=payload_type_counts,
            roi_payload_count=len(payload_summaries),
            total_roi_count=total_roi_count,
            returned_roi_count=returned_roi_count,
            roi_count_exact=roi_count_exact,
            total_roi_member_count=total_roi_member_count,
            returned_roi_member_count=returned_roi_member_count,
            roi_payloads_truncated=not roi_count_exact,
            payloads=tuple(payload_summaries),
            errors=result.errors,
            warnings=result.warnings,
        )

    @staticmethod
    def _numeric_metadata(
        shape_payloads: tuple[Mapping[str, JsonValue], ...],
        field_name: str,
    ) -> tuple[float, ...]:
        values: list[float] = []
        for shape_payload in shape_payloads:
            metadata = shape_payload.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            value = metadata.get(field_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        return tuple(values)

    @staticmethod
    def _semantic_shape_payloads(
        shape_payloads: tuple[Mapping[str, JsonValue], ...],
    ) -> tuple[Mapping[str, JsonValue], ...]:
        unique = {}
        for shape_payload in shape_payloads:
            metadata = shape_payload.get("metadata")
            if isinstance(metadata, Mapping) and metadata:
                identity = (
                    core_plate_image_inventory.semantic_roi_identity_from_metadata(
                        metadata
                    )
                )
            else:
                identity = repr(sorted(shape_payload.items(), key=repr))
            unique.setdefault(identity, shape_payload)
        return tuple(unique.values())

    @staticmethod
    def _numeric_stats(values: tuple[float, ...]) -> dict[str, float] | None:
        if not values:
            return None
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "mean": sum(ordered) / len(ordered),
            "max": ordered[-1],
        }

    @staticmethod
    def _example_roi(shape_payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        metadata = shape_payload.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        return {
            "type": shape_payload.get("type"),
            "label": metadata.get("label"),
            "area": metadata.get("area"),
            "centroid_yx": metadata.get("centroid"),
            "bbox_yxyx": metadata.get("bbox"),
            "perimeter": metadata.get("perimeter"),
            "source_spatial_shape_yx": metadata.get("source_spatial_shape_yx"),
        }

    def validation_summary(
        self,
        request: ViewerWindowValidationRequest,
    ) -> ViewerWindowValidationSummaryResult:
        connection = request.connection
        state = self.window_state(
            ViewerWindowStateRequest(
                connection=connection,
                timeout_ms=request.timeout_ms,
                state_controls=request.state_controls,
                include_response=False,
            )
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
        include_response: bool = True,
    ) -> ViewerWindowStateResult:
        status = self._required_scalar(
            response, ViewerControlResponseField.STATUS, str, "a string"
        )
        if status != self.SUCCESS_STATUS:
            message = self._required_scalar(
                response, ViewerControlResponseField.MESSAGE, str, "a string"
            )
            return ViewerWindowStateResult.from_error(
                connection=connection,
                error=AgentError(code="viewer_window_state_failed", message=message),
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
            viewer=ViewerWindowDescriptor.from_wire_fields(
                viewer_wire_value=self._required_scalar(
                    viewer_payload,
                    ViewerDescriptorField.TYPE,
                    str,
                    "a string",
                ),
                title=self._required_scalar(
                    viewer_payload,
                    ViewerDescriptorField.TITLE,
                    str,
                    "a string",
                ),
            ),
            layer_count=self._required_scalar(
                response, ViewerControlField.LAYER_COUNT, int, "an integer"
            ),
            layers=tuple(
                self._layer_state_from_payload(layer_payload)
                for layer_payload in layer_payloads
            ),
            active_dimension_label_route=self._optional_typed(
                response,
                ViewerControlField.ACTIVE_DIMENSION_LABEL_ROUTE,
                str,
            ),
            viewer_ndim=self._required_scalar(
                response, ViewerControlField.VIEWER_NDIM, int, "an integer"
            ),
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
            component_group_count=self._required_scalar(
                response,
                ViewerControlField.COMPONENT_GROUP_COUNT,
                int,
                "an integer",
            ),
            component_item_count=self._required_scalar(
                response,
                ViewerControlField.COMPONENT_ITEM_COUNT,
                int,
                "an integer",
            ),
            response=response if include_response else {},
        )

    def _navigation_result_from_response(
        self,
        *,
        connection: ExecutionConnectionSpec,
        request: ViewerWindowNavigationRequest,
        response: JsonObject,
    ) -> ViewerWindowNavigationResult:
        status = self._required_scalar(
            response, ViewerControlResponseField.STATUS, str, "a string"
        )
        if status != self.SUCCESS_STATUS:
            message = self._required_scalar(
                response, ViewerControlResponseField.MESSAGE, str, "a string"
            )
            return ViewerWindowNavigationResult.from_error(
                connection=connection,
                error=AgentError(
                    code="viewer_window_navigation_failed", message=message
                ),
            )

        state = self._state_result_from_response(
            connection=connection,
            response=response,
        )
        target_layer = next(
            (
                layer
                for layer in state.layers
                if layer.route_key == request.navigation.route_key
            ),
            None,
        )
        return ViewerWindowNavigationResult(
            schema_version=SCHEMA_VERSION,
            connection=connection,
            observed=state.observed,
            viewer=state.viewer,
            route_key=request.navigation.route_key,
            visible=target_layer.visible if target_layer is not None else None,
            selected=target_layer.selected if target_layer is not None else None,
            data_index=request.navigation.data_index,
            feature_row_count=(
                target_layer.feature_row_count if target_layer is not None else 0
            ),
            selected_data_indices=(
                target_layer.selected_data_indices if target_layer is not None else ()
            ),
            active_dimension_label_route=state.active_dimension_label_route,
            current_step=state.current_step,
            axis_labels=state.axis_labels,
            available_layers=self._layer_visibility_records(state.layers),
            errors=state.errors,
            warnings=state.warnings,
        )

    @staticmethod
    def _layer_visibility_records(
        layers: Sequence[ViewerWindowLayerState],
    ) -> tuple[ViewerWindowLayerVisibilityRecord, ...]:
        """Project the viewer-owned mounted layer graph for mutation results."""

        return tuple(
            ViewerWindowLayerVisibilityRecord(
                route_key=layer.route_key,
                title=layer.title,
                visible=layer.visible,
                selected=layer.selected,
            )
            for layer in layers
            if layer.mounted
        )

    def _payload_result_from_response(
        self,
        *,
        connection: ExecutionConnectionSpec,
        response: JsonObject,
        include_response: bool = True,
    ) -> ViewerWindowPayloadResult:
        status = self._required_scalar(
            response, ViewerControlResponseField.STATUS, str, "a string"
        )
        if status != self.SUCCESS_STATUS:
            message = self._required_scalar(
                response, ViewerControlResponseField.MESSAGE, str, "a string"
            )
            return ViewerWindowPayloadResult.from_error(
                connection=connection,
                error=AgentError(code="viewer_window_payloads_failed", message=message),
            )

        viewer_payload = self._required_mapping(response, ViewerControlField.VIEWER)
        layer_payloads = self._required_sequence(
            response,
            ViewerControlField.LAYERS,
        )
        return ViewerWindowPayloadResult(
            schema_version=SCHEMA_VERSION,
            connection=connection,
            observed=True,
            viewer=ViewerWindowDescriptor.from_wire_fields(
                viewer_wire_value=self._required_scalar(
                    viewer_payload,
                    ViewerDescriptorField.TYPE,
                    str,
                    "a string",
                ),
                title=self._required_scalar(
                    viewer_payload,
                    ViewerDescriptorField.TITLE,
                    str,
                    "a string",
                ),
            ),
            layer_count=self._required_scalar(
                response, ViewerControlField.LAYER_COUNT, int, "an integer"
            ),
            layers=tuple(
                self._layer_payloads_from_payload(layer_payload)
                for layer_payload in layer_payloads
            ),
            response=response if include_response else {},
        )

    def _layer_payloads_from_payload(
        self,
        payload: JsonValue,
    ) -> ViewerWindowLayerPayloads:
        if not isinstance(payload, Mapping):
            raise TypeError("Viewer layer payload entries must be mappings.")
        payload_records = self._required_sequence(payload, ViewerLayerField.PAYLOADS)
        producer_payloads = self._required_mapping_tuple(
            payload,
            ViewerLayerField.PRODUCER_IDENTITIES,
        )
        return ViewerWindowLayerPayloads(
            route_key=self._required_scalar(
                payload, ViewerLayerField.ROUTE_KEY, str, "a string"
            ),
            title=self._optional_typed(payload, ViewerLayerField.TITLE, str),
            mounted=self._required_scalar(
                payload, ViewerLayerField.MOUNTED, bool, "a boolean"
            ),
            item_count=self._required_scalar(
                payload, ViewerLayerField.ITEM_COUNT, int, "an integer"
            ),
            producer_identities=tuple(
                StreamProducerIdentity.from_payload(producer_payload)
                for producer_payload in producer_payloads
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
            pending_update=self._required_scalar(
                payload, ViewerLayerField.PENDING_UPDATE, bool, "a boolean"
            ),
            payloads=tuple(
                self._payload_record_from_payload(payload_record)
                for payload_record in payload_records
            ),
        )

    def _payload_record_from_payload(
        self,
        payload: JsonValue,
    ) -> ViewerWindowPayloadRecord:
        if not isinstance(payload, Mapping):
            raise TypeError("Viewer payload records must be mappings.")
        return ViewerWindowPayloadRecord(
            route_key=self._required_scalar(
                payload, ViewerPayloadField.ROUTE_KEY, str, "a string"
            ),
            data_type=self._required_scalar(
                payload, ViewerPayloadField.DATA_TYPE, str, "a string"
            ),
            path=self._required_scalar(
                payload, ViewerPayloadField.PATH, str, "a string"
            ),
            components=self._required_mapping(payload, ViewerPayloadField.COMPONENTS),
            axis_indices=self._required_typed_tuple(
                payload,
                ViewerPayloadField.AXIS_INDICES,
                int,
            ),
            aggregate_axis_indices=self._required_typed_tuple(
                payload,
                ViewerPayloadField.AGGREGATE_AXIS_INDICES,
                int,
            ),
            summary=self._required_mapping(payload, ViewerPayloadField.SUMMARY),
            array_values=self._required_sequence(
                payload,
                ViewerPayloadField.ARRAY_VALUES,
            ),
            array_value_summary=self._optional_mapping(
                payload,
                ViewerPayloadField.ARRAY_VALUE_SUMMARY,
            ),
            shape_payloads=self._required_mapping_tuple(
                payload,
                ViewerPayloadField.SHAPE_PAYLOADS,
            ),
        )

    def _layer_state_from_payload(
        self,
        payload: JsonValue,
    ) -> ViewerWindowLayerState:
        if not isinstance(payload, Mapping):
            raise TypeError("Viewer layer state entries must be mappings.")
        producer_payloads = self._required_mapping_tuple(
            payload,
            ViewerLayerField.PRODUCER_IDENTITIES,
        )
        return ViewerWindowLayerState(
            route_key=self._required_scalar(
                payload, ViewerLayerField.ROUTE_KEY, str, "a string"
            ),
            title=self._optional_typed(payload, ViewerLayerField.TITLE, str),
            mounted=self._required_scalar(
                payload, ViewerLayerField.MOUNTED, bool, "a boolean"
            ),
            item_count=self._required_scalar(
                payload, ViewerLayerField.ITEM_COUNT, int, "an integer"
            ),
            producer_identities=tuple(
                StreamProducerIdentity.from_payload(producer_payload)
                for producer_payload in producer_payloads
            ),
            data_types=self._required_typed_tuple(
                payload,
                ViewerLayerField.DATA_TYPES,
                str,
            ),
            component_values=self._required_mapping_tuple(
                payload,
                ViewerLayerField.COMPONENT_VALUES,
            ),
            component_value_count=self._optional_typed(
                payload,
                ViewerLayerField.COMPONENT_VALUE_COUNT,
                int,
            )
            or self._sequence_length(payload, ViewerLayerField.COMPONENT_VALUES),
            component_values_truncated=self._optional_typed(
                payload,
                ViewerLayerField.COMPONENT_VALUES_TRUNCATED,
                bool,
            )
            or False,
            payload_summaries=self._required_mapping_tuple(
                payload,
                ViewerLayerField.PAYLOAD_SUMMARIES,
            ),
            payload_summary_count=self._optional_typed(
                payload,
                ViewerLayerField.PAYLOAD_SUMMARY_COUNT,
                int,
            )
            or self._sequence_length(payload, ViewerLayerField.PAYLOAD_SUMMARIES),
            payload_summaries_truncated=self._optional_typed(
                payload,
                ViewerLayerField.PAYLOAD_SUMMARIES_TRUNCATED,
                bool,
            )
            or False,
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
            visible=self._required_scalar(
                payload, ViewerLayerField.VISIBLE, bool, "a boolean"
            ),
            selected=self._required_scalar(
                payload, ViewerLayerField.SELECTED, bool, "a boolean"
            ),
            feature_row_count=self._optional_typed(
                payload,
                ViewerLayerField.FEATURE_ROW_COUNT,
                int,
            )
            or 0,
            selected_data_indices=self._optional_typed_tuple(
                payload,
                ViewerLayerField.SELECTED_DATA_INDICES,
                int,
            ),
            pending_update=self._required_scalar(
                payload, ViewerLayerField.PENDING_UPDATE, bool, "a boolean"
            ),
        )

    @staticmethod
    def _required_mapping(
        payload: Mapping[str, JsonValue], field_name: str
    ) -> JsonObject:
        if field_name not in payload:
            raise KeyError(f"Viewer response missing required field {field_name!r}.")
        value = payload[field_name]
        if not isinstance(value, Mapping):
            raise TypeError(f"Viewer response field {field_name!r} must be a mapping.")
        return dict(value)

    @staticmethod
    def _optional_mapping(
        payload: Mapping[str, JsonValue], field_name: str
    ) -> JsonObject:
        if field_name not in payload:
            return {}
        value = payload[field_name]
        if value is None:
            return {}
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
    def _sequence_length(
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> int:
        if field_name not in payload:
            return 0
        value = payload[field_name]
        if not isinstance(value, (list, tuple)):
            return 0
        return len(value)

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
    def _optional_typed_tuple(
        payload: Mapping[str, JsonValue],
        field_name: str,
        expected_type: type[OptionalViewerFieldT],
    ) -> tuple[OptionalViewerFieldT, ...]:
        if field_name not in payload:
            return ()
        return ViewerWindowService._required_typed_tuple(
            payload,
            field_name,
            expected_type,
        )

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
    def _required_scalar(
        payload: Mapping[str, JsonValue],
        field_name: str,
        expected_type: type[OptionalViewerFieldT],
        expected_description: str,
    ) -> OptionalViewerFieldT:
        if field_name not in payload:
            raise KeyError(f"Viewer response missing required field {field_name!r}.")
        value = payload[field_name]
        if expected_type is int and isinstance(value, bool):
            raise TypeError(
                f"Viewer response field {field_name!r} must be {expected_description}."
            )
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Viewer response field {field_name!r} must be {expected_description}."
            )
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
