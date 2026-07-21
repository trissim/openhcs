"""Tracking backends for CellProfiler-compatible TrackObjects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, ClassVar, NamedTuple, TYPE_CHECKING

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType, VariableComponents
from openhcs.core.artifacts import (
    ArtifactType,
    ImageArtifactType,
    ArtifactSpec,
    ArtifactSpecCollection,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    RelationshipsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    DataclassMeasurementColumnarRows,
    MeasurementSparseColumnarRows,
    measurement_row_has_object_identity,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    object_label_input_execution_mode,
    required_variable_components,
    special_inputs,
    )
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.core.runtime_output_matching import RuntimeOutputBundle
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
    measurement_row_mapping,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
    MeasurementScope,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.interop.cellprofiler.runtime.primary_image_input_policies import (
    ObjectLabelDrivenPrimaryImageInputPolicy,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    CellProfilerObjectMeasurementRowPolicy,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    CurrentPayloadMeasurementRecordMixin,
    FieldDerivedMeasurementFeatureModule,
    MeasurementFeatureRecord,
    NoObjectNameMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ModuleOwnedResultMeasurementRows,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    required_setting_value,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendAuthority,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
)

if TYPE_CHECKING:
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
    from openhcs.interop.cellprofiler.parser import ModuleBlock
    from openhcs.interop.cellprofiler.runtime.output_record_request import (
        CellProfilerOutputRecordRequest,
    )
    from openhcs.core.steps.function_runtime import RuntimeCallableArgument


class TrackObjectsObjectMeasurementRowPolicy(CellProfilerObjectMeasurementRowPolicy):
    """TrackObjects emits object rows and image-level tracking counts together."""

    explicit_row_ownership_required = True

    def row_is_object_scoped(self, row: 'RuntimeCallableArgument') -> bool:
        row_mapping = measurement_row_mapping(row)
        return measurement_row_has_object_identity(row_mapping)

    def image_row_source_image_name(self, source_image_name: str | None) -> str:
        """Own image-level counts without qualifying them by an input channel."""

        del source_image_name
        return MeasurementScope.IMAGE.value


class TrackObjectsModule(
    TrackObjectsObjectMeasurementRowPolicy,
    FieldDerivedMeasurementFeatureModule,
    NoObjectNameMeasurementRecordMixin,
    ObjectLabelDrivenPrimaryImageInputPolicy,
    ObjectArtifactInputModule,
    CurrentPayloadMeasurementRecordMixin,
    MeasurementArtifactOutputModule,
):
    module_name = "TrackObjects"
    function_name = "track_objects"
    validated = True
    confidence = 1.0
    parent_relationship_type = "Parent"

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        """Project typed tracking records with contract-owned qualifiers."""

        object_name: str

        @classmethod
        def for_request(cls, module_type, request):
            object_name = module_type.runtime_object_measurement_row_policy().table_object_owner(
                request.callable_contract.artifact_inputs.specs
            )
            if object_name is None:
                raise ValueError(
                    "TrackObjects measurement projection requires one tracked "
                    "object input."
                )
            return cls(
                request.output_value,
                module_type=module_type,
                object_name=object_name,
            )

        def source_records(self, record_type: type[object]) -> tuple[object, ...]:
            source_rows = self.source_rows()
            batches = (
                source_rows.row_batches
                if isinstance(source_rows, ConcatenatedColumnarRows)
                else (source_rows,)
            )
            matching = tuple(
                batch
                for batch in batches
                if isinstance(batch, DataclassMeasurementColumnarRows)
                and batch.row_type is record_type
            )
            if len(matching) != 1:
                raise TypeError(
                    "TrackObjects measurement projection requires exactly one "
                    f"{record_type.__name__} batch, got {matching!r}."
                )
            return tuple(matching[0].rows)

        def project_records(
            self,
            records: tuple[object, ...],
            *,
            qualified_parts: tuple[object, ...],
        ) -> list[dict[str, object]]:
            """Embed the declared scale axis once in final feature identities."""

            projected_rows: list[dict[str, object]] = []
            for record in records:
                axis_values = self.module_type.measurement_record_axis_values(record)
                axis_values.pop(MeasurementRowAxisField.SCALE.value)
                projected_rows.extend(
                    self.module_type.measurement_feature_rows(
                        axis_values=axis_values,
                        feature_values=(
                            self.module_type.measurement_record_field_values(record)
                        ),
                        qualified_parts=qualified_parts,
                        value_field=type(record).measurement_value_field,
                    )
                )
            return projected_rows

        def rows(self) -> MeasurementSparseColumnarRows:
            object_records = self.source_records(TrackingObjectMeasurement)
            image_records = self.source_records(TrackingImageMeasurement)
            scale_values = {
                self.module_type.measurement_record_axis_values(record)[
                    MeasurementRowAxisField.SCALE.value
                ]
                for record in (*object_records, *image_records)
            }
            if len(scale_values) != 1:
                raise ValueError(
                    "TrackObjects typed measurement rows must declare exactly one "
                    f"scale, got {scale_values!r}."
                )
            scale_qualifier = (scale_values.pop(),)
            projected_rows = self.project_records(
                object_records,
                qualified_parts=scale_qualifier,
            )
            projected_rows.extend(
                self.project_records(
                    image_records,
                    qualified_parts=(self.object_name, *scale_qualifier),
                )
            )
            object_records_by_slice: dict[int, list[MeasurementFeatureRecord]] = {}
            for record in object_records:
                axis_values = self.module_type.measurement_record_axis_values(record)
                slice_index = int(
                    axis_values[MeasurementRowAxisField.SLICE_INDEX.value]
                )
                object_records_by_slice.setdefault(slice_index, []).append(record)
            for slice_index, slice_records in object_records_by_slice.items():
                projected_rows.extend(
                    self.module_type.mean_measurement_feature_rows_from_records(
                        tuple(slice_records),
                        axis_values={
                            MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                        },
                        object_name=self.object_name,
                        qualified_parts=scale_qualifier,
                    )
                )
            fields = FieldSpec.merge_exact(
                (
                    tuple(
                        field
                        for field in self.module_type.measurement_feature_row_fields(
                            TrackingObjectMeasurement
                        )
                        if field.name != MeasurementRowAxisField.SCALE.value
                    ),
                    tuple(
                        field
                        for field in self.module_type.measurement_feature_row_fields(
                            TrackingImageMeasurement
                        )
                        if field.name != MeasurementRowAxisField.SCALE.value
                    ),
                ),
                context="TrackObjects projected measurement fields",
            )
            rows = MeasurementSparseColumnarRows.from_rows(
                projected_rows,
                fields=fields,
                object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
            )
            row_policy = self.module_type.runtime_object_measurement_row_policy()
            return row_policy.annotate_record_rows(
                rows,
                object_name=self.object_name,
                source_image_name=MeasurementScope.IMAGE.value,
            )

    @classmethod
    def measurement_record_source_image_name(
        cls,
        request: "CellProfilerOutputRecordRequest",
        rows: ColumnarRows,
    ) -> None:
        """Keep mixed tracking rows independent of the current image channel."""

        del cls, request, rows
        return None

    @classmethod
    def complete_table_measurement_rows(
        cls,
        request: "CellProfilerOutputRecordRequest",
        rows: ColumnarRows,
    ) -> ColumnarRows:
        """Leave the empty raw-row branch to the nominal typed projector."""

        del cls, request
        return rows

    tracking_method_setting = "Choose a tracking method"
    tracked_objects_setting = "Select the objects to track"
    pixel_radius_setting = "Maximum pixel distance to consider matches"
    movement_model_setting = "Select the movement model"
    radius_standard_deviation_setting = (
        "Number of standard deviations for search radius"
    )
    search_radius_limit_setting = SettingNameFamily(
        "Search radius limit, in pixel units",
        aliases=("Search radius limit, in pixel units (Min,Max)",),
    )
    second_phase_setting = "Run the second phase of the LAP algorithm?"
    gap_cost_setting = "Gap closing cost"
    split_cost_setting = "Split alternative cost"
    merge_cost_setting = "Merge alternative cost"
    mitosis_cost_setting = "Mitosis alternative cost"
    maximum_gap_displacement_setting = "Maximum gap displacement in pixel units"
    maximum_split_score_setting = "Maximum split score"
    maximum_merge_score_setting = "Maximum merge score"
    maximum_frame_distance_setting = "Maximum temporal gap in frames"
    maximum_mitosis_distance_setting = "Maximum mitosis distance in pixel units"
    filter_by_lifetime_setting = "Filter objects by lifetime?"
    use_minimum_lifetime_setting = "Filter using a minimum lifetime?"
    minimum_lifetime_setting = "Minimum lifetime"
    use_maximum_lifetime_setting = "Filter using a maximum lifetime?"
    maximum_lifetime_setting = "Maximum lifetime"
    retain_image_setting = "Save color-coded image?"
    output_image_setting = "Name the output image"
    ignored_settings = (
        "Average cell diameter in pixels",
        "Cost of cell to empty matching",
        "Select display option",
        "Select object measurement to use for tracking",
        "Use advanced configuration parameters",
        "Weight of area difference in function matching cost",
    )
    default_output_image_name = "TrackedCells"
    tracked_objects_binding = SettingToKeywordBinding.input(
        tracked_objects_setting, ObjectLabelsArtifactType, runtime_parameter_name="labels"
    )

    @classmethod
    def primary_image_domain_input_binding(cls) -> SettingToKeywordBinding:
        """Use the tracked-object labels as the invocation image domain."""

        return cls.tracked_objects_binding

    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting, ImageArtifactType
    )

    @classmethod
    def canonical_output_artifact_name(
        cls,
        *,
        artifact_type: type[ArtifactType],
        output_position: int,
        block_position: int,
        step_context: "ArtifactDeclarationStepContext",
    ) -> str:
        """Preserve the callable-owned default identity for the retained image."""

        expected_type = cls.output_image_binding.require_artifact_type()
        if artifact_type is not expected_type or output_position != 0:
            raise ValueError(
                f"{cls.__name__} declares one retained-image output slot, got "
                f"{artifact_type.__name__} at position {output_position}."
            )
        del block_position, step_context
        return cls.default_output_image_name

    setting_bindings = (
        tracked_objects_binding,
        output_image_binding,
        SettingToKeywordBinding(
            tracking_method_setting,
            "tracking_method",
            normalize_cellprofiler_setting_name,
        ),
        SettingToKeywordBinding(
            pixel_radius_setting,
            "pixel_radius",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            movement_model_setting,
            "movement_model",
            normalize_cellprofiler_setting_name,
        ),
        SettingToKeywordBinding(
            radius_standard_deviation_setting,
            "radius_std",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            second_phase_setting,
            "run_second_phase",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(gap_cost_setting, "gap_cost", parse_cellprofiler_int),
        SettingToKeywordBinding(
            split_cost_setting,
            "split_cost",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            merge_cost_setting,
            "merge_cost",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            mitosis_cost_setting,
            "mitosis_cost",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            maximum_gap_displacement_setting,
            "max_gap_displacement",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            maximum_split_score_setting,
            "max_split_score",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            maximum_merge_score_setting,
            "max_merge_score",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            maximum_frame_distance_setting,
            "max_frame_distance",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            maximum_mitosis_distance_setting,
            "mitosis_max_distance",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            filter_by_lifetime_setting,
            "filter_by_lifetime",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            use_minimum_lifetime_setting,
            "use_minimum_lifetime",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            minimum_lifetime_setting,
            "minimum_lifetime",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            use_maximum_lifetime_setting,
            "use_maximum_lifetime",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            maximum_lifetime_setting,
            "maximum_lifetime",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            retain_image_setting,
            "save_color_coded_image",
            parse=parse_cellprofiler_bool,
        ),
    )

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        retain_image = parse_cellprofiler_bool(
            required_setting_value(module, cls.retain_image_setting)
        )
        return tuple(
            binding
            for binding in bindings
            if retain_image or binding is not cls.output_image_binding
        )

    @classmethod
    def tracking_method(cls, module: "ModuleBlock") -> "TrackingMethod":
        """Return the nominal tracking method declared by one module."""

        return coerce_cellprofiler_enum(
            TrackingMethod,
            required_setting_value(module, cls.tracking_method_setting),
        )

    @classmethod
    def require_supported_tracking_method(
        cls,
        module: "ModuleBlock",
    ) -> "TrackingMethod":
        """Fail before contract emission for methods absent from the runtime."""

        method = cls.tracking_method(module)
        if method not in {TrackingMethod.OVERLAP, TrackingMethod.DISTANCE}:
            raise NotImplementedError(
                "TrackObjects tracking method is not supported by the converter: "
                f"{method.value!r}"
            )
        return method

    @classmethod
    def tracked_object_input(
        cls,
        module: "ModuleBlock",
        artifact_inputs: ArtifactSpecCollection,
    ) -> ArtifactSpec:
        """Return the tracked object artifact from its setting-owned role."""

        return artifact_inputs.require_by_name_and_artifact_type(
            required_setting_value(module, cls.tracked_objects_setting),
            ObjectLabelsArtifactType,
        )

    @classmethod
    def artifact_contract_outputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        """Declare CP's temporal Parent relationship before measurement rows."""

        cls.require_supported_tracking_method(module)
        tracked_objects = cls.tracked_object_input(module, artifact_inputs)
        declaration = ObjectRelationshipDeclaration(
            source=tracked_objects.ref(),
            target=tracked_objects.ref(),
            relationship_type=cls.parent_relationship_type,
            source_role="parent",
            target_role="child",
            source_id_field="parent_id",
            target_id_field="child_id",
            producer_module_number=module.module_num,
            source_runtime_slice_offset=-1,
        )
        relationship = ArtifactSpec.output(
            declaration.artifact_name(),
            RelationshipsArtifactType,
            relations=(
                SourceStackLineageSourceRelation(source=tracked_objects.ref()),
                declaration,
            ),
        )
        inherited = list(
            super().artifact_contract_outputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            )
        )
        measurement_positions = tuple(
            index
            for index, spec in enumerate(inherited)
            if spec.artifact_type is MeasurementsArtifactType
        )
        if len(measurement_positions) != 1:
            raise ValueError(
                "TrackObjects requires exactly one measurement output, got "
                f"{tuple(spec.ref() for spec in inherited)!r}."
            )
        inherited.insert(measurement_positions[0], relationship)
        return tuple(inherited)

    @classmethod
    def postprocess_bound_settings(
        cls,
        module: "ModuleBlock",
        bound: "BoundModuleSettings",
    ) -> "BoundModuleSettings":
        kwargs = dict(bound.kwargs)
        tracking_method = cls.require_supported_tracking_method(module)
        kwargs["tracking_method"] = tracking_method.value
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        search_radius = optional_setting_value(
            module,
            cls.search_radius_limit_setting,
        )
        if search_radius is not None:
            limits = tuple(part.strip() for part in search_radius.split(","))
            if len(limits) != 2 or not all(limits):
                raise ValueError(
                    "TrackObjects search radius limit must contain two values, got "
                    f"{search_radius!r}."
                )
            kwargs["radius_limit_min"], kwargs["radius_limit_max"] = tuple(
                parse_cellprofiler_float(value) for value in limits
            )
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(
                    cls.search_radius_limit_setting.canonical
                ),
                None,
            )
        return BoundModuleSettings(
            kwargs,
            unmapped_kwargs,
            bound.setting_coverage,
        )

class TrackingMethod(Enum):
    """CellProfiler TrackObjects tracking method."""

    OVERLAP = "overlap"
    DISTANCE = "distance"
    MEASUREMENTS = "measurements"
    LAP = "lap"


class MovementModel(Enum):
    """CellProfiler TrackObjects movement model for LAP-style settings."""

    RANDOM = "random"
    VELOCITY = "velocity"
    BOTH = "both"


@dataclass
class TrackingResult:
    """Tracking measurements for objects in current frame."""

    slice_index: int
    object_count: int
    new_object_count: int
    lost_object_count: int
    split_count: int
    merge_count: int


@dataclass
class ObjectTrackingData:
    """Per-object tracking data."""

    label: np.ndarray
    parent_object_number: np.ndarray
    parent_image_number: np.ndarray
    trajectory_x: np.ndarray
    trajectory_y: np.ndarray
    distance_traveled: np.ndarray
    displacement: np.ndarray
    integrated_distance: np.ndarray
    linearity: np.ndarray
    lifetime: np.ndarray


TrackingObjectFrameKey = tuple[int, int]


@dataclass(slots=True)
class TrackingObjectMeasurement(MeasurementFeatureRecord):
    """Raw TrackObjects measurements before CP row projection."""

    measurement_value_field: ClassVar[MeasurementRowValueField] = (
        MeasurementRowValueField.MEASUREMENT_VALUE
    )
    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    object_label: Annotated[int, MeasurementRowAxisField.OBJECT_LABEL]
    scale: Annotated[int, MeasurementRowAxisField.SCALE]
    displacement: float
    distance_traveled: float
    final_age: int
    integrated_distance: float
    label: int
    lifetime: int
    linearity: float
    parent_image_number: int
    parent_object_number: int
    trajectory_x: float
    trajectory_y: float


@dataclass(frozen=True, slots=True)
class TrackingImageMeasurement(MeasurementFeatureRecord):
    """Raw TrackObjects image-level count measurements."""

    measurement_value_field: ClassVar[MeasurementRowValueField] = (
        MeasurementRowValueField.MEASUREMENT_VALUE
    )
    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    scale: Annotated[int, MeasurementRowAxisField.SCALE]
    new_object_count: int
    lost_object_count: int
    split_object_count: int
    merged_object_count: int


TrackingFrameResult = tuple[int, list[TrackingObjectMeasurement], int, int, int, int]
TrackingFrameResults = list[TrackingFrameResult]


@dataclass(frozen=True, slots=True)
class TrackObjectsResult(RuntimeOutputBundle):
    """Nominal TrackObjects result with its temporal CP relationship."""

    output_image: np.ndarray
    parent_relationship: DirectedObjectRelationshipPayload
    tracking_measurements: ConcatenatedColumnarRows

    def as_runtime_tuple(
        self,
    ) -> tuple[
        np.ndarray,
        DirectedObjectRelationshipPayload,
        ConcatenatedColumnarRows,
    ]:
        """Lower to the canonical image, relationship, and measurement ABI."""

        return (
            self.output_image,
            self.parent_relationship,
            self.tracking_measurements,
        )


@dataclass(frozen=True, slots=True)
class TrackingFrameRequest:
    """Shared per-frame TrackObjects inputs."""

    current_labels: np.ndarray
    old_labels: np.ndarray | None
    old_object_numbers: np.ndarray
    max_object_number: int
    pixel_radius: int


class TrackingKernelFrame(NamedTuple):
    """Numba-compatible dense-label state shared by tracking kernels."""

    current_labels: np.ndarray
    old_labels: np.ndarray
    old_object_numbers: np.ndarray
    max_object_number: int
    current_count: int
    old_count: int


class ObjectTrackingBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """TrackObjects primitives keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def label_centers(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return y/x centers for dense positive labels."""

    @abstractmethod
    def track_by_overlap(
        self, request: TrackingFrameRequest
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Assign track labels using maximum object overlap."""

    @abstractmethod
    def track_by_distance(
        self, request: TrackingFrameRequest
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Assign track labels using nearest centroid within a radius."""

    @abstractmethod
    def overlap_transition_counts(
        self,
        previous_labels: np.ndarray,
        current_labels: np.ndarray,
        previous_track_labels: np.ndarray,
        current_track_labels: np.ndarray,
    ) -> tuple[int, int]:
        """Return lost and merged counts from inter-frame label overlap."""


class NumbaNumpyObjectTrackingBackendStrategy(ObjectTrackingBackendStrategy):
    """Numba implementation of TrackObjects dense-label primitives."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        current = np.array([[0, 1], [2, 2]], dtype=np.int32)
        previous = np.array([[0, 1], [1, 0]], dtype=np.int32)
        old_numbers = np.array([0, 1], dtype=np.int32)
        self.label_centers(current)
        request = TrackingFrameRequest(current, previous, old_numbers, 1, 5)
        self.track_by_overlap(request)
        self.track_by_distance(request)
        self.overlap_transition_counts(
            previous,
            current,
            old_numbers,
            np.array([1, 2], dtype=np.int32),
        )

    def label_centers(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        labels_array = np.asarray(labels)
        label_count = int(labels_array.max()) if labels_array.size else 0
        if label_count == 0:
            return (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
        centers = _label_centers_numba(np.ascontiguousarray(labels_array), label_count)
        return (centers[1:, 0], centers[1:, 1])

    def track_by_overlap(
        self, request: TrackingFrameRequest
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        current = np.asarray(request.current_labels)
        current_count = int(current.max()) if current.size else 0
        if request.old_labels is None or current_count == 0:
            return self._new_track_labels(current_count, request.max_object_number)
        old = np.asarray(request.old_labels)
        old_count = int(old.max()) if old.size else 0
        if old_count == 0:
            return self._new_track_labels(current_count, request.max_object_number)
        return _track_by_overlap_numba(
            TrackingKernelFrame(
                current_labels=np.ascontiguousarray(current),
                old_labels=np.ascontiguousarray(old),
                old_object_numbers=np.asarray(
                    request.old_object_numbers, dtype=np.int64
                ),
                max_object_number=int(request.max_object_number),
                current_count=current_count,
                old_count=old_count,
            )
        )

    def track_by_distance(
        self, request: TrackingFrameRequest
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        current = np.asarray(request.current_labels)
        current_count = int(current.max()) if current.size else 0
        if request.old_labels is None or current_count == 0:
            return self._new_track_labels(current_count, request.max_object_number)
        old = np.asarray(request.old_labels)
        old_count = int(old.max()) if old.size else 0
        if old_count == 0:
            return self._new_track_labels(current_count, request.max_object_number)
        return _track_by_distance_numba(
            TrackingKernelFrame(
                current_labels=np.ascontiguousarray(current),
                old_labels=np.ascontiguousarray(old),
                old_object_numbers=np.asarray(
                    request.old_object_numbers, dtype=np.int64
                ),
                max_object_number=int(request.max_object_number),
                current_count=current_count,
                old_count=old_count,
            ),
            int(request.pixel_radius),
        )

    def overlap_transition_counts(
        self,
        previous_labels: np.ndarray,
        current_labels: np.ndarray,
        previous_track_labels: np.ndarray,
        current_track_labels: np.ndarray,
    ) -> tuple[int, int]:
        """Return exact CP-style lost/merged counts without a Python pixel scan."""

        previous = np.asarray(previous_labels, dtype=int).ravel()
        current = np.asarray(current_labels, dtype=int).ravel()
        previous_object_ids = np.unique(previous[previous > 0])
        current_object_ids = np.unique(current[current > 0])
        if previous_object_ids.size == 0:
            return (0, 0)
        if current_object_ids.size == 0:
            return (int(previous_object_ids.size), 0)

        paired_size = min(previous.size, current.size)
        paired_previous = previous[:paired_size]
        paired_current = current[:paired_size]
        overlap_mask = (paired_previous > 0) & (paired_current > 0)
        overlap_pairs = np.unique(
            np.column_stack(
                (paired_previous[overlap_mask], paired_current[overlap_mask])
            ),
            axis=0,
        )

        previous_tracks = np.asarray(previous_track_labels, dtype=int).ravel()
        current_tracks = np.asarray(current_track_labels, dtype=int).ravel()
        current_tracks = np.unique(current_tracks[current_tracks > 0])
        previous_object_tracks = np.zeros(previous_object_ids.size, dtype=int)
        tracked_previous_mask = previous_object_ids <= previous_tracks.size
        previous_object_tracks[tracked_previous_mask] = previous_tracks[
            previous_object_ids[tracked_previous_mask] - 1
        ]
        overlapping_previous_ids = (
            overlap_pairs[:, 0]
            if overlap_pairs.size
            else np.zeros(0, dtype=previous_object_ids.dtype)
        )
        lost_count = np.count_nonzero(
            ~np.isin(previous_object_ids, overlapping_previous_ids)
            & ~np.isin(previous_object_tracks, current_tracks)
        )

        if overlap_pairs.size == 0 or previous_tracks.size == 0:
            return (int(lost_count), 0)
        tracked_pair_mask = overlap_pairs[:, 0] <= previous_tracks.size
        tracked_pairs = overlap_pairs[tracked_pair_mask]
        pair_track_ids = previous_tracks[tracked_pairs[:, 0] - 1]
        positive_track_mask = pair_track_ids > 0
        current_track_pairs = np.unique(
            np.column_stack(
                (
                    tracked_pairs[positive_track_mask, 1],
                    pair_track_ids[positive_track_mask],
                )
            ),
            axis=0,
        )
        if current_track_pairs.size == 0:
            return (int(lost_count), 0)
        _current_ids, distinct_track_counts = np.unique(
            current_track_pairs[:, 0],
            return_counts=True,
        )
        merge_count = np.count_nonzero(distinct_track_counts > 1)
        return (int(lost_count), int(merge_count))

    def _new_track_labels(
        self, object_count: int, max_object_number: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if object_count == 0:
            return (
                np.array([], dtype=int),
                np.zeros(0, dtype=int),
                np.zeros(0, dtype=int),
                max_object_number,
            )
        new_labels = np.arange(1, object_count + 1, dtype=int) + max_object_number
        return (
            new_labels,
            np.zeros(object_count, dtype=int),
            np.zeros(object_count, dtype=int),
            max_object_number + object_count,
        )


class TrackObjectsMethodStrategy(
    EnumKeyedStrategyMixin[TrackingMethod], ABC, metaclass=AutoRegisterMeta
):
    """Registered CellProfiler TrackObjects method behavior."""

    __registry_key__ = "method_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"
    __enum_label_attr__ = "method_label"
    method: ClassVar[TrackingMethod | None] = None
    method_label: ClassVar[str | None] = None

    @classmethod
    def for_method(cls, method: str | TrackingMethod) -> "TrackObjectsMethodStrategy":
        resolved = coerce_cellprofiler_enum(TrackingMethod, method)
        return cls.for_enum_member(resolved)

    def track(
        self,
        request: TrackingFrameRequest,
        backend_provider: CellProfilerBackendProvider | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Assign stable object identities for the current frame."""
        return self._track_with_backend(
            ObjectTrackingBackendStrategy.for_memory_type(
                backend_provider=backend_provider
            ),
            request,
        )

    def relationship_pairs(
        self,
        request: TrackingFrameRequest,
        parent_object_numbers: np.ndarray,
        backend_provider: CellProfilerBackendProvider | None,
    ) -> tuple[tuple[int, int], ...]:
        """Return CP's union of forward and reverse frame correspondences."""

        forward_pairs = tuple(
            (int(parent_id), child_id)
            for child_id, parent_id in enumerate(parent_object_numbers, start=1)
            if int(parent_id) > 0
        )
        reverse_pairs = self._reverse_relationship_pairs(
            ObjectTrackingBackendStrategy.for_memory_type(
                backend_provider=backend_provider
            ),
            request,
        )
        return tuple(sorted(set((*forward_pairs, *reverse_pairs))))

    @abstractmethod
    def _track_with_backend(
        self, backend: ObjectTrackingBackendStrategy, request: TrackingFrameRequest
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Delegate one frame to the concrete tracking primitive."""

    @abstractmethod
    def _reverse_relationship_pairs(
        self,
        backend: ObjectTrackingBackendStrategy,
        request: TrackingFrameRequest,
    ) -> tuple[tuple[int, int], ...]:
        """Return previous-to-current matches for relationship completeness."""


class OverlapTrackObjectsMethodStrategy(TrackObjectsMethodStrategy):
    """Track objects by maximum overlap between frames."""

    method = TrackingMethod.OVERLAP

    def _track_with_backend(
        self, backend: ObjectTrackingBackendStrategy, request: TrackingFrameRequest
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        return backend.track_by_overlap(request)

    def _reverse_relationship_pairs(
        self,
        backend: ObjectTrackingBackendStrategy,
        request: TrackingFrameRequest,
    ) -> tuple[tuple[int, int], ...]:
        del backend
        if request.old_labels is None:
            return ()
        old_labels = np.asarray(request.old_labels, dtype=np.int32)
        current_labels = np.asarray(request.current_labels, dtype=np.int32)
        old_count = int(old_labels.max()) if old_labels.size else 0
        current_count = int(current_labels.max()) if current_labels.size else 0
        if old_count == 0 or current_count == 0:
            return ()
        overlap = np.zeros((old_count + 1, current_count + 1), dtype=np.int64)
        mask = (old_labels > 0) & (current_labels > 0)
        np.add.at(
            overlap,
            (old_labels[mask], current_labels[mask]),
            1,
        )
        pairs: list[tuple[int, int]] = []
        for old_id in range(1, old_count + 1):
            current_id = int(np.argmax(overlap[old_id]))
            if current_id > 0 and overlap[old_id, current_id] > 0:
                pairs.append((old_id, current_id))
        return tuple(pairs)


class DistanceTrackObjectsMethodStrategy(TrackObjectsMethodStrategy):
    """Track objects by minimum distance between centroids."""

    method = TrackingMethod.DISTANCE

    def _track_with_backend(
        self, backend: ObjectTrackingBackendStrategy, request: TrackingFrameRequest
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        return backend.track_by_distance(request)

    def _reverse_relationship_pairs(
        self,
        backend: ObjectTrackingBackendStrategy,
        request: TrackingFrameRequest,
    ) -> tuple[tuple[int, int], ...]:
        if request.old_labels is None:
            return ()
        old_y, old_x = backend.label_centers(np.asarray(request.old_labels))
        current_y, current_x = backend.label_centers(np.asarray(request.current_labels))
        if len(old_y) == 0 or len(current_y) == 0:
            return ()
        radius_squared = float(request.pixel_radius * request.pixel_radius)
        pairs: list[tuple[int, int]] = []
        for old_index, (old_row, old_column) in enumerate(
            zip(old_y, old_x, strict=True)
        ):
            distances_squared = (current_y - old_row) ** 2 + (
                current_x - old_column
            ) ** 2
            current_index = int(np.argmin(distances_squared))
            if float(distances_squared[current_index]) <= radius_squared:
                pairs.append((old_index + 1, current_index + 1))
        return tuple(pairs)


@required_variable_components(VariableComponents.TIMEPOINT)
@numpy(contract=ProcessingContract.PURE_3D)
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.FULL_STACK)
@special_inputs("labels")
def track_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    tracking_method: str = "overlap",
    pixel_radius: int = 50,
    movement_model: str = "both",
    radius_std: float = 3.0,
    radius_limit_min: float = 2.0,
    radius_limit_max: float = 10.0,
    run_second_phase: bool = True,
    gap_cost: int = 40,
    split_cost: int = 40,
    merge_cost: int = 40,
    mitosis_cost: int = 80,
    max_gap_displacement: int = 5,
    max_split_score: int = 50,
    max_merge_score: int = 50,
    max_frame_distance: int = 5,
    mitosis_max_distance: int = 40,
    filter_by_lifetime: bool = False,
    use_minimum_lifetime: bool = True,
    minimum_lifetime: int = 1,
    use_maximum_lifetime: bool = False,
    maximum_lifetime: int = 100,
    save_color_coded_image: bool = False,
    name_the_output_image: str = TrackObjectsModule.default_output_image_name,
    tracking_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[
    np.ndarray,
    DirectedObjectRelationshipPayload,
    ConcatenatedColumnarRows,
]:
    """Track objects across sequential frames.

    Args:
        labels: Object-label stack whose per-frame regions are linked into tracks.
        radius_limit_min: Minimum object-radius estimate accepted by the movement
            model, in pixels.
        radius_limit_max: Maximum object-radius estimate accepted by the movement
            model, in pixels.
    """
    del (
        movement_model,
        radius_std,
        radius_limit_min,
        radius_limit_max,
        run_second_phase,
        gap_cost,
        split_cost,
        merge_cost,
        mitosis_cost,
        max_gap_displacement,
        max_split_score,
        max_merge_score,
        max_frame_distance,
        mitosis_max_distance,
        filter_by_lifetime,
        use_minimum_lifetime,
        minimum_lifetime,
        use_maximum_lifetime,
        maximum_lifetime,
        save_color_coded_image,
        name_the_output_image,
    )
    label_frames = _label_frames(labels)
    tracking_state = _initial_tracking_state()
    tracking_strategy = TrackObjectsMethodStrategy.for_method(tracking_method)
    frame_results: TrackingFrameResults = []
    relationship_parent_ids: list[int] = []
    relationship_child_ids: list[int] = []
    relationship_slice_indices: list[int] = []
    for frame_index, current_labels in enumerate(label_frames):
        image_number = frame_index + 1
        old_labels = tracking_state.get("old_labels")
        old_object_numbers = tracking_state.get("old_object_numbers", np.array([], int))
        max_object_number = int(tracking_state.get("max_object_number", 0))
        frame_request = TrackingFrameRequest(
            current_labels=current_labels,
            old_labels=old_labels,
            old_object_numbers=old_object_numbers,
            max_object_number=max_object_number,
            pixel_radius=pixel_radius,
        )
        new_labels, parent_obj_nums, parent_img_nums, max_object_number = (
            tracking_strategy.track(
                frame_request,
                tracking_backend_provider,
            )
        )
        parent_img_nums = np.where(parent_obj_nums > 0, image_number - 1, 0)
        relationship_pairs = tracking_strategy.relationship_pairs(
            frame_request,
            parent_obj_nums,
            tracking_backend_provider,
        )
        relationship_parent_ids.extend(
            parent_id for parent_id, _child_id in relationship_pairs
        )
        relationship_child_ids.extend(
            child_id for _parent_id, child_id in relationship_pairs
        )
        relationship_slice_indices.extend(
            frame_index for _parent_id, _child_id in relationship_pairs
        )
        object_rows = _tracking_object_rows(
            current_labels,
            slice_index=frame_index,
            measurement_scale=int(pixel_radius),
            track_labels=new_labels,
            parent_object_numbers=parent_obj_nums,
            parent_image_numbers=parent_img_nums,
            previous_object_states=tracking_state["track_states"],
            tracking_backend_provider=tracking_backend_provider,
        )
        new_object_count = int(np.sum(parent_obj_nums == 0))
        lost_object_count, split_count, merge_count = _tracking_transition_counts(
            old_labels,
            current_labels,
            old_object_numbers,
            new_labels,
            tracking_backend_provider,
        )
        frame_results.append(
            (
                frame_index,
                object_rows,
                new_object_count,
                lost_object_count,
                split_count,
                merge_count,
            )
        )
        tracking_state["old_labels"] = current_labels.copy()
        tracking_state["old_object_numbers"] = new_labels.copy()
        tracking_state["max_object_number"] = max_object_number
    _apply_final_age_measurements(frame_results)
    object_measurements: list[TrackingObjectMeasurement] = []
    image_measurements: list[TrackingImageMeasurement] = []
    for (
        slice_index,
        object_rows,
        new_object_count,
        lost_object_count,
        split_count,
        merge_count,
    ) in frame_results:
        object_measurements.extend(object_rows)
        image_measurements.append(
            TrackingImageMeasurement(
                slice_index=slice_index,
                scale=int(pixel_radius),
                new_object_count=new_object_count,
                lost_object_count=lost_object_count,
                split_object_count=split_count,
                merged_object_count=merge_count,
            )
        )
    if np.asarray(image).ndim != 3:
        raise ValueError("TrackObjects requires a declared 3-D timepoint stack.")
    return TrackObjectsResult(
        output_image=image,
        parent_relationship=DirectedObjectRelationshipPayload(
            source_ids=tuple(relationship_parent_ids),
            target_ids=tuple(relationship_child_ids),
            slice_indices=tuple(relationship_slice_indices),
            slice_count=len(label_frames),
        ),
        tracking_measurements=ConcatenatedColumnarRows(
            (
                DataclassMeasurementColumnarRows(
                    tuple(object_measurements),
                    row_type=TrackingObjectMeasurement,
                ),
                DataclassMeasurementColumnarRows(
                    tuple(image_measurements),
                    row_type=TrackingImageMeasurement,
                ),
            )
        ),
    )


def _initial_tracking_state() -> dict[str, Any]:
    return {
        "old_labels": None,
        "old_object_numbers": np.array([], int),
        "max_object_number": 0,
        "track_states": {},
    }


def _label_frames(labels: ObjectLabelValue) -> np.ndarray:
    if not isinstance(labels, ObjectLabelValue):
        raise TypeError("TrackObjects requires a nominal ObjectLabelValue input.")
    plane_count = labels.declared_plane_count()
    if plane_count is None:
        raise ValueError(
            "TrackObjects requires object labels with a declared plane domain."
        )
    label_array = object_label_dense_array(labels, dtype=np.int32)
    if label_array.ndim != 3 or label_array.shape[0] != plane_count:
        raise ValueError(
            "TrackObjects requires one dense 2-D label plane per declared timepoint; "
            f"got shape {label_array.shape!r} for {plane_count} planes."
        )
    return label_array


def _tracking_object_rows(
    labels: np.ndarray,
    *,
    slice_index: int,
    measurement_scale: int,
    track_labels: np.ndarray,
    parent_object_numbers: np.ndarray,
    parent_image_numbers: np.ndarray,
    previous_object_states: dict[int, dict[str, Any]],
    tracking_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> list[TrackingObjectMeasurement]:
    y_centers, x_centers = ObjectTrackingBackendStrategy.for_memory_type(
        backend_provider=tracking_backend_provider
    ).label_centers(labels)
    next_object_states: dict[int, dict[str, Any]] = {}
    rows: list[TrackingObjectMeasurement] = []
    for object_index, track_label in enumerate(track_labels):
        object_number = object_index + 1
        track_id = int(track_label)
        y = float(y_centers[object_index])
        x = float(x_centers[object_index])
        parent_object_number = int(parent_object_numbers[object_index])
        previous_state = (
            previous_object_states.get(parent_object_number)
            if parent_object_number > 0
            else None
        )
        if previous_state is None:
            origin = (y, x)
            previous = (y, x)
            integrated_distance = 0.0
            lifetime = 1
        else:
            origin = previous_state["origin"]
            previous = previous_state["previous"]
            integrated_distance = float(previous_state["integrated_distance"])
            lifetime = int(previous_state["lifetime"]) + 1
        trajectory_y = y - float(previous[0])
        trajectory_x = x - float(previous[1])
        distance_traveled = float(np.hypot(trajectory_y, trajectory_x))
        integrated_distance += distance_traveled
        displacement = float(np.hypot(y - float(origin[0]), x - float(origin[1])))
        linearity = (
            displacement / integrated_distance
            if integrated_distance > 0.0
            else float("nan")
        )
        next_object_states[object_number] = {
            "origin": origin,
            "previous": (y, x),
            "integrated_distance": integrated_distance,
            "lifetime": lifetime,
        }
        rows.append(
            TrackingObjectMeasurement(
                slice_index=slice_index,
                object_label=object_number,
                scale=measurement_scale,
                displacement=displacement,
                distance_traveled=distance_traveled,
                final_age=float("nan"),
                integrated_distance=integrated_distance,
                label=track_id,
                lifetime=lifetime,
                linearity=linearity,
                parent_image_number=int(parent_image_numbers[object_index]),
                parent_object_number=int(parent_object_numbers[object_index]),
                trajectory_x=trajectory_x,
                trajectory_y=trajectory_y,
            )
        )
    previous_object_states.clear()
    previous_object_states.update(next_object_states)
    return rows


def _tracking_transition_counts(
    previous_labels: np.ndarray | None,
    current_labels: np.ndarray,
    previous_track_labels: np.ndarray,
    current_track_labels: np.ndarray,
    tracking_backend_provider: BackendProviderInput,
) -> tuple[int, int, int]:
    previous_counts = _positive_value_counts(previous_track_labels)
    current_counts = _positive_value_counts(current_track_labels)
    split_count = sum((1 for count in current_counts.values() if count > 1))
    if previous_labels is None:
        return (0, int(split_count), 0)
    lost_count, overlap_merge_count = (
        ObjectTrackingBackendStrategy.for_memory_type(
            backend_provider=tracking_backend_provider
        ).overlap_transition_counts(
            previous_labels,
            current_labels,
            previous_track_labels,
            current_track_labels,
        )
    )
    track_merge_count = sum(
        (
            previous_counts[track_label] - current_counts[track_label]
            for track_label in set(previous_counts) | set(current_counts)
            if 0
            < current_counts.get(track_label, 0)
            < previous_counts.get(track_label, 0)
        )
    )
    merge_count = max(overlap_merge_count, track_merge_count)
    return (int(lost_count), int(split_count), int(merge_count))

def _positive_value_counts(values: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for value in np.asarray(values, dtype=int).ravel():
        if value <= 0:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _apply_final_age_measurements(frame_results: TrackingFrameResults) -> None:
    labels_by_frame: dict[int, set[int]] = {}
    object_values: dict[TrackingObjectFrameKey, tuple[int, int]] = {}
    final_age_records: dict[TrackingObjectFrameKey, TrackingObjectMeasurement] = {}
    for slice_index, object_rows, *_counts in frame_results:
        for measurement in object_rows:
            object_label = measurement.object_label
            key = (slice_index, object_label)
            track_label = int(float(measurement.label))
            labels_by_frame.setdefault(slice_index, set()).add(track_label)
            object_values[key] = (track_label, int(measurement.lifetime))
            final_age_records[key] = measurement
    last_slice_index = frame_results[-1][0] if frame_results else 0
    for (slice_index, object_label), (track_label, lifetime) in object_values.items():
        next_labels = labels_by_frame.get(slice_index + 1, set())
        if slice_index != last_slice_index and track_label in next_labels:
            continue
        final_age_records[slice_index, object_label].final_age = int(lifetime)


@njit(cache=True)
def _label_centers_numba(labels: np.ndarray, label_count: int) -> np.ndarray:
    sums = np.zeros((label_count + 1, 2), dtype=np.float64)
    counts = np.zeros(label_count + 1, dtype=np.int64)
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label_id = int(labels[y, x])
            if label_id > 0 and label_id <= label_count:
                sums[label_id, 0] += y
                sums[label_id, 1] += x
                counts[label_id] += 1
    centers = np.empty((label_count + 1, 2), dtype=np.float64)
    for label_id in range(label_count + 1):
        if counts[label_id] == 0:
            centers[label_id, 0] = np.nan
            centers[label_id, 1] = np.nan
        else:
            centers[label_id, 0] = sums[label_id, 0] / counts[label_id]
            centers[label_id, 1] = sums[label_id, 1] / counts[label_id]
    return centers


@njit(cache=True)
def _track_by_overlap_numba(
    frame: TrackingKernelFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    overlap = np.zeros((frame.current_count + 1, frame.old_count + 1), dtype=np.int64)
    height, width = frame.current_labels.shape
    for y in range(height):
        for x in range(width):
            current_label = int(frame.current_labels[y, x])
            old_label = int(frame.old_labels[y, x])
            if (
                current_label > 0
                and current_label <= frame.current_count
                and (old_label > 0)
                and (old_label <= frame.old_count)
            ):
                overlap[current_label, old_label] += 1
    new_labels = np.zeros(frame.current_count, dtype=np.int64)
    parent_object_numbers = np.zeros(frame.current_count, dtype=np.int64)
    parent_image_numbers = np.zeros(frame.current_count, dtype=np.int64)
    max_object_number = frame.max_object_number
    for current_index in range(frame.current_count):
        current_label = current_index + 1
        best_old = 0
        best_overlap = 0
        for old_label in range(1, frame.old_count + 1):
            current_overlap = overlap[current_label, old_label]
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_old = old_label
        if best_old > 0 and best_overlap > 0:
            new_labels[current_index] = frame.old_object_numbers[best_old - 1]
            parent_object_numbers[current_index] = best_old
            parent_image_numbers[current_index] = 1
        else:
            max_object_number += 1
            new_labels[current_index] = max_object_number
    return (new_labels, parent_object_numbers, parent_image_numbers, max_object_number)


@njit(cache=True)
def _track_by_distance_numba(
    frame: TrackingKernelFrame, pixel_radius: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    current_centers = _label_centers_numba(frame.current_labels, frame.current_count)
    old_centers = _label_centers_numba(frame.old_labels, frame.old_count)
    new_labels = np.zeros(frame.current_count, dtype=np.int64)
    parent_object_numbers = np.zeros(frame.current_count, dtype=np.int64)
    parent_image_numbers = np.zeros(frame.current_count, dtype=np.int64)
    radius_squared = float(pixel_radius * pixel_radius)
    max_object_number = frame.max_object_number
    for current_index in range(frame.current_count):
        current_label = current_index + 1
        current_y = current_centers[current_label, 0]
        current_x = current_centers[current_label, 1]
        best_old = -1
        best_distance_squared = float((pixel_radius + 1) * (pixel_radius + 1))
        for old_index in range(frame.old_count):
            old_label = old_index + 1
            old_y = old_centers[old_label, 0]
            old_x = old_centers[old_label, 1]
            dy = current_y - old_y
            dx = current_x - old_x
            distance_squared = dy * dy + dx * dx
            if distance_squared < best_distance_squared:
                best_distance_squared = distance_squared
                best_old = old_index
        if best_old >= 0 and best_distance_squared <= radius_squared:
            new_labels[current_index] = frame.old_object_numbers[best_old]
            parent_object_numbers[current_index] = best_old + 1
            parent_image_numbers[current_index] = 1
        else:
            max_object_number += 1
            new_labels[current_index] = max_object_number
    return (new_labels, parent_object_numbers, parent_image_numbers, max_object_number)


__all__ = public_names_from_objects(
    DistanceTrackObjectsMethodStrategy,
    MovementModel,
    NumbaNumpyObjectTrackingBackendStrategy,
    ObjectTrackingData,
    ObjectTrackingBackendStrategy,
    OverlapTrackObjectsMethodStrategy,
    TrackObjectsMethodStrategy,
    TrackingFrameRequest,
    TrackingKernelFrame,
    TrackingMethod,
    TrackingResult,
    track_objects,
)
