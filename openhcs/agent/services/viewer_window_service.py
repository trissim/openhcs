"""Agent service for running viewer window interactions."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import ClassVar, Generic, TypeVar

from metaclass_registry import AutoRegisterMeta
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
    ViewerWindowLayerPayloads,
    ViewerWindowProbeResult,
    ViewerWindowPayloadRecord,
    ViewerWindowPayloadRequest,
    ViewerWindowPayloadResult,
    ViewerWindowValidationCounters,
    ViewerWindowSnapshotRequest,
    ViewerWindowSnapshotResult,
    ViewerWindowStateRequest,
    ViewerWindowStateResult,
    ViewerWindowValidationErrorContext,
    ViewerWindowValidationPolicy,
    ViewerWindowValidationRequest,
    ViewerWindowValidationSummaryResult,
    viewer_window_probe_from_state,
)
from openhcs.core.image_shapes import ArrayShape
from openhcs.runtime.window_snapshot import (
    WindowSnapshotCaptureSpec,
    WindowSnapshotWirePayload,
)
from openhcs.runtime.viewer_protocol import (
    ViewerControlMessageType,
)
from openhcs.runtime.viewer_component_system import (
    ComponentValue,
    ComponentValues,
    ViewerComponentMetadataPayload,
    ViewerComponentValueParser,
    ViewerLayerAxisProjection,
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
            spatial_mismatch_count=sum(
                layer.spatial_mismatch_count for layer in layer_summaries
            ),
        )


class ViewerImageSpatialShapeAuthority:
    """Resolve image-like array shapes to y/x spatial shape for validation."""

    @staticmethod
    def spatial_shape(shape: tuple[int, ...]) -> tuple[int, int] | None:
        if len(shape) < 2:
            return None
        if (
            len(shape) >= 3
            and ArrayShape(ndim=len(shape), shape=shape).has_channel_last()
        ):
            return tuple(int(value) for value in shape[-3:-1])
        return tuple(int(value) for value in shape[-2:])


class ViewerPayloadSummaryField:
    """Semantic fields carried by viewer payload summaries."""

    SHAPE = "shape"
    SOURCE_SPATIAL_SHAPES_YX = "source_spatial_shapes_yx"
    NONZERO_COUNT = "nonzero_count"


@dataclass(frozen=True, slots=True)
class ViewerPayloadSpatialShapeEvidence:
    """Payload summary fields that can prove y/x spatial shape."""

    shape: tuple[int, ...] | None
    source_spatial_shapes_yx: tuple[tuple[int, int], ...]

    @classmethod
    def from_summary(
        cls,
        payload_summary: JsonObject,
    ) -> "ViewerPayloadSpatialShapeEvidence":
        return cls(
            shape=cls._optional_int_tuple(
                payload_summary,
                ViewerPayloadSummaryField.SHAPE,
            ),
            source_spatial_shapes_yx=cls._spatial_shape_tuple(
                payload_summary,
                ViewerPayloadSummaryField.SOURCE_SPATIAL_SHAPES_YX,
            ),
        )

    def spatial_shape(self) -> tuple[int, int] | None:
        if self.shape is not None:
            return ViewerImageSpatialShapeAuthority.spatial_shape(self.shape)
        if len(self.source_spatial_shapes_yx) != 1:
            return None
        return self.source_spatial_shapes_yx[0]

    @classmethod
    def _optional_int_tuple(
        cls,
        payload_summary: JsonObject,
        field_name: str,
    ) -> tuple[int, ...] | None:
        if field_name not in payload_summary:
            return None
        value = payload_summary[field_name]
        if value is None:
            return None
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise TypeError(f"Viewer payload summary {field_name!r} must be a sequence.")
        return tuple(
            cls._int_value(item, field_name=field_name)
            for item in value
        )

    @classmethod
    def _spatial_shape_tuple(
        cls,
        payload_summary: JsonObject,
        field_name: str,
    ) -> tuple[tuple[int, int], ...]:
        if field_name not in payload_summary:
            return ()
        value = payload_summary[field_name]
        if value is None:
            return ()
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise TypeError(f"Viewer payload summary {field_name!r} must be a sequence.")
        return tuple(
            cls._spatial_shape_value(item, field_name=field_name)
            for item in value
        )

    @classmethod
    def _spatial_shape_value(
        cls,
        value: JsonValue,
        *,
        field_name: str,
    ) -> tuple[int, int]:
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise TypeError(
                f"Viewer payload summary {field_name!r} entries must be sequences."
            )
        if len(value) != 2:
            raise ValueError(
                f"Viewer payload summary {field_name!r} entries must have length 2."
            )
        return (
            cls._int_value(value[0], field_name=field_name),
            cls._int_value(value[1], field_name=field_name),
        )

    @staticmethod
    def _int_value(value: JsonValue, *, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"Viewer payload summary {field_name!r} values must be integers."
            )
        return value


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
            valid=layer_count_matches and all(layer.valid for layer in layer_summaries),
            warnings=cls.validation_warnings(
                validation_context=validation_context,
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
            missing_payload_coordinate_count=(totals.missing_payload_coordinate_count),
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
        validation_context: ViewerWindowValidationWarningContext,
        layer_summaries: Sequence[ViewerWindowLayerValidationSummary],
    ) -> tuple[AgentWarning, ...]:
        warnings: list[AgentWarning] = []
        warnings.extend(
            ViewerWindowValidationWarningRule.warnings(validation_context)
        )
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
        coordinate_coverage = ViewerLayerCoordinateCoverage.from_layer(layer)
        spatial_mismatch_count = cls.spatial_mismatch_count(layer)
        nonzero_valid = not validation_policy.require_nonzero_payloads or (
            payload_count > 0 and zero_payload_count == 0 and missing_nonzero_count == 0
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
            candidate = ViewerPayloadSpatialShapeEvidence.from_summary(
                payload_summary
            ).spatial_shape()
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
            int(value) for value in layer.data_shape[len(layer.stack_axes) :]
        )
        return ViewerImageSpatialShapeAuthority.spatial_shape(local_shape)

    @staticmethod
    def payload_nonzero_count(payload_summary: JsonObject) -> int | None:
        field_name = ViewerPayloadSummaryField.NONZERO_COUNT
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
    PAYLOADS = "payloads"


class ViewerPayloadField:
    """Viewer layer payload-record fields."""

    ROUTE_KEY = "route_key"
    DATA_TYPE = "data_type"
    PATH = "path"
    COMPONENTS = "components"
    AXIS_INDICES = "axis_indices"
    AGGREGATE_AXIS_INDICES = "aggregate_axis_indices"
    SUMMARY = "summary"
    ARRAY_VALUES = "array_values"
    SHAPE_PAYLOADS = "shape_payloads"


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
        return AgentWarning(
            code=self.warning_code,
            message=(
                f"Viewer layer {context.layer_name!r} is missing required axis "
                f"labels: {missing}."
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


class ViewerSpatialMismatchWarningRule(ViewerLayerValidationWarningRule):
    """Warn when payload spatial shapes do not match the displayed layer."""

    warning_code = ViewerValidationWarningCode.SPATIAL_MISMATCH

    def warning(
        self,
        context: ViewerLayerValidationWarningContext,
    ) -> AgentWarning | None:
        if not context.layer.spatial_mismatch_count:
            return None
        return AgentWarning(
            code=self.warning_code,
            message=(
                f"Viewer layer {context.layer_name!r} has "
                f"{context.layer.spatial_mismatch_count} payload spatial "
                "shape mismatch(es)."
            ),
        )


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

    @abstractmethod
    def window_payloads(self, request: ViewerWindowPayloadRequest) -> JsonObject:
        raise NotImplementedError


class ZMQViewerWindowGateway(ViewerWindowGatewayABC):
    """Viewer gateway backed by the existing ZMQ control socket."""

    def __init__(
        self, context_factory: Callable[[], zmq.Context] = zmq.Context
    ) -> None:
        self._context_factory = context_factory

    def snapshot_window(self, request: ViewerWindowSnapshotRequest) -> JsonObject:
        message = {
            ViewerControlField.TYPE: ViewerControlMessageType.SCREENSHOT.value,
            ViewerControlField.SNAPSHOT: request.to_wire_payload().as_dict(),
        }
        return self._send_control_message(request, message)

    def window_state(self, request: ViewerWindowStateRequest) -> JsonObject:
        message = {
            ViewerControlField.TYPE: ViewerControlMessageType.STATE.value,
        }
        return self._send_control_message(request, message)

    def window_payloads(self, request: ViewerWindowPayloadRequest) -> JsonObject:
        message: dict[str, JsonValue] = {
            ViewerControlField.TYPE: ViewerControlMessageType.PAYLOADS.value,
        }
        message.update(request.to_wire_payload())
        return self._send_control_message(request, message)

    def _send_control_message(
        self,
        request: (
            ViewerWindowSnapshotRequest
            | ViewerWindowStateRequest
            | ViewerWindowPayloadRequest
        ),
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
        request: ViewerWindowSnapshotRequest,
    ) -> ViewerWindowSnapshotResult:
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

    def _snapshot_result_from_response(
        self,
        *,
        connection: ExecutionConnectionSpec,
        request: ViewerWindowSnapshotRequest,
        response: JsonObject,
    ) -> ViewerWindowSnapshotResult:
        status = self._required_scalar(
            response, ViewerControlField.STATUS, str, "a string"
        )
        if status != self.SUCCESS_STATUS:
            message = self._required_scalar(
                response, ViewerControlField.MESSAGE, str, "a string"
            )
            return ViewerWindowSnapshotResult.from_request_error(
                request=request,
                error=AgentError(code="viewer_window_snapshot_failed", message=message),
            )
        response_snapshot = WindowSnapshotCaptureSpec.from_wire_payload(
            WindowSnapshotWirePayload(
                self._required_str_mapping(response, ViewerControlField.SNAPSHOT)
            )
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
            viewer=ViewerWindowDescriptor(
                viewer_type=self._required_scalar(
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
            )
        except Exception as exc:
            return ViewerWindowPayloadResult.from_error(
                connection=connection,
                error=AgentError.from_exception(
                    "viewer_window_payloads_response_invalid", exc
                ),
            )

    def validation_summary(
        self,
        request: ViewerWindowValidationRequest,
    ) -> ViewerWindowValidationSummaryResult:
        connection = request.connection
        state = self.window_state(
            ViewerWindowStateRequest(
                connection=connection,
                timeout_ms=request.timeout_ms,
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
    ) -> ViewerWindowStateResult:
        status = self._required_scalar(
            response, ViewerControlField.STATUS, str, "a string"
        )
        if status != self.SUCCESS_STATUS:
            message = self._required_scalar(
                response, ViewerControlField.MESSAGE, str, "a string"
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
            viewer=ViewerWindowDescriptor(
                viewer_type=self._required_scalar(
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
            response=response,
        )

    def _payload_result_from_response(
        self,
        *,
        connection: ExecutionConnectionSpec,
        response: JsonObject,
    ) -> ViewerWindowPayloadResult:
        status = self._required_scalar(
            response, ViewerControlField.STATUS, str, "a string"
        )
        if status != self.SUCCESS_STATUS:
            message = self._required_scalar(
                response, ViewerControlField.MESSAGE, str, "a string"
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
            viewer=ViewerWindowDescriptor(
                viewer_type=self._required_scalar(
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
            response=response,
        )

    def _layer_payloads_from_payload(
        self,
        payload: JsonValue,
    ) -> ViewerWindowLayerPayloads:
        if not isinstance(payload, Mapping):
            raise TypeError("Viewer layer payload entries must be mappings.")
        payload_records = self._required_sequence(payload, ViewerLayerField.PAYLOADS)
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
            visible=self._required_scalar(
                payload, ViewerLayerField.VISIBLE, bool, "a boolean"
            ),
            selected=self._required_scalar(
                payload, ViewerLayerField.SELECTED, bool, "a boolean"
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
