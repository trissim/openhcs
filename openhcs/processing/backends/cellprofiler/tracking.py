"""Tracking backends for CellProfiler-compatible TrackObjects."""

from __future__ import annotations
from openhcs.constants.constants import VariableComponents
from openhcs.core.measurement_row_materialization import (
    measurement_row_has_object_identity,
)
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
    MeasurementScope,
    measurement_row_mapping,
)
from openhcs.core.runtime_values import image_payload_metadata
from openhcs.interop.cellprofiler.runtime.bound_parameters import (
    RuntimeBoundParameterName,
)
from openhcs.interop.cellprofiler.runtime.primary_image_input_policies import (
    ObjectLabelDrivenPrimaryImageInputPolicy,
)
from openhcs.interop.cellprofiler.settings_binder import (
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_int,
)
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
    MeasurementArtifactOutputModule,
    ModuleSettingsSourceModule,
    ObjectArtifactInputModule,
    ObjectMeasurementRowsModule,
    ScopedMeasurementModule,
    StructuringElementSettingsModule,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    CellProfilerObjectMeasurementRowPolicy,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    FieldDerivedMeasurementFeatureModule,
    FieldsFromRowsMeasurementRecordMixin,
    MeasurementFeatureRecord,
    NoObjectNameMeasurementRecordMixin,
    NoSourceMeasurementRecordMixin,
    TrackingMeasurementRecordRowsMixin,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)


class TrackObjectsObjectMeasurementRowPolicy(CellProfilerObjectMeasurementRowPolicy):
    """TrackObjects emits object rows and image-level tracking counts together."""

    explicit_row_ownership_required = True

    def row_is_object_scoped(self, row: "CellProfilerRuntimeValue") -> bool:
        row_mapping = measurement_row_mapping(row)
        return measurement_row_has_object_identity(row_mapping)

    def image_row_source_image_name(self, source_image_name: str | None) -> str | None:
        del source_image_name
        return MeasurementScope.IMAGE.value


class TrackObjectsModule(
    TrackingMeasurementRecordRowsMixin,
    FieldDerivedMeasurementFeatureModule,
    NoObjectNameMeasurementRecordMixin,
    NoSourceMeasurementRecordMixin,
    FieldsFromRowsMeasurementRecordMixin,
    ObjectLabelDrivenPrimaryImageInputPolicy,
    ObjectArtifactInputModule,
    MeasurementArtifactOutputModule,
    ObjectMeasurementRowsModule,
    TrackObjectsObjectMeasurementRowPolicy,
):
    module_name = "TrackObjects"
    function_name = "track_objects"
    validated = True
    contract = ProcessingContract.PURE_3D
    confidence = 1.0
    ignored_settings = (
        "Average cell diameter in pixels",
        "Cost of cell to empty matching",
        "Filter objects by lifetime?",
        "Filter using a maximum lifetime?",
        "Filter using a minimum lifetime?",
        "Gap closing cost",
        "Maximum gap displacement in pixel units",
        "Maximum lifetime",
        "Maximum merge score",
        "Maximum mitosis distance in pixel units",
        "Maximum split score",
        "Maximum temporal gap in frames",
        "Merge alternative cost",
        "Minimum lifetime",
        "Mitosis alternative cost",
        "Number of standard deviations for search radius",
        "Run the second phase of the LAP algorithm?",
        "Save color-coded image?",
        "Name the output image",
        "Search radius limit, in pixel units",
        "Select display option",
        "Select object measurement to use for tracking",
        "Select the movement model",
        "Split alternative cost",
        "Use advanced configuration parameters",
        "Weight of area difference in function matching cost",
    )
    required_variable_components = (VariableComponents.TIMEPOINT,)
    tracked_objects_setting = "Select the objects to track"
    image_number_start_kwarg: ClassVar[RuntimeBoundParameterName] = (
        RuntimeBoundParameterName("image_number_start")
    )

    @classmethod
    def object_input_setting_names(cls):
        return (cls.tracked_objects_setting,)

    def invocation_runtime_kwargs(
        self,
        *,
        module_name: str,
        plan: "CellProfilerModuleRuntimePlan",
        image_request: "CellProfilerImageRequest",
        adapter: "CellProfilerRuntimeAdapter",
        current_image: "CellProfilerRuntimeValue",
        runtime_kwargs: "CellProfilerKwargs",
        object_input_source_image_name: Callable[[], str | None],
    ) -> "CellProfilerKwargs":
        """Bind the first CellProfiler image number for tracked source paths."""
        del module_name, plan
        source_paths = image_payload_metadata(current_image).source_image_paths
        if not source_paths:
            source_paths = adapter.cellprofiler_source_paths_for_image_name(
                image_request.source_image_name or object_input_source_image_name()
            )
        return {
            **runtime_kwargs,
            type(
                self
            ).image_number_start_kwarg: adapter.cellprofiler_image_number_start_for_source_paths(
                source_paths
            ),
        }

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        generic = cls._bind_generic_settings(
            module, binder=binder, param_mapping=param_mapping, use_declaration=False
        )
        method = module.get_setting("Choose a tracking method", "Overlap").strip()
        tracking_method = normalize_cellprofiler_setting_name(method)
        if tracking_method not in {"overlap", "distance"}:
            raise NotImplementedError(
                f"TrackObjects tracking method is not supported by the converter: {method!r}"
            )
        object_name = module.get_setting(cls.tracked_objects_setting, "Objects").strip()
        if not object_name:
            raise ValueError("TrackObjects requires a non-empty tracked object name.")
        pixel_radius = parse_cellprofiler_int(
            module.get_setting("Maximum pixel distance to consider matches", "50")
        )
        unmapped_kwargs = dict(generic.unmapped_kwargs)
        for setting_name in (
            "Choose a tracking method",
            "Select the objects to track",
            "Maximum pixel distance to consider matches",
        ):
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=BoundModuleSettings(
                {
                    **dict(generic.kwargs),
                    "object_name": object_name,
                    "tracking_method": tracking_method,
                    "pixel_radius": pixel_radius,
                },
                unmapped_kwargs,
            ),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )


from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, ClassVar, NamedTuple
import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.constants.constants import MemoryType
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
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
    image_number: Annotated[int, MeasurementRowAxisField.IMAGE_NUMBER]
    object_label: Annotated[int, MeasurementRowAxisField.OBJECT_LABEL]
    displacement: float
    distance_traveled: float
    final_age: float
    integrated_distance: float
    label: int
    lifetime: int
    linearity: float
    parent_image_number: float
    parent_object_number: int
    trajectory_x: float
    trajectory_y: float


@dataclass(frozen=True, slots=True)
class TrackingImageMeasurement(MeasurementFeatureRecord):
    """Raw TrackObjects image-level count measurements."""

    measurement_value_field: ClassVar[MeasurementRowValueField] = (
        MeasurementRowValueField.MEASUREMENT_VALUE
    )
    image_number: Annotated[int, MeasurementRowAxisField.IMAGE_NUMBER]
    new_object_count: int
    lost_object_count: int
    split_object_count: int
    merged_object_count: int


TrackingFrameResult = tuple[int, list[TrackingObjectMeasurement], int, int, int, int]
TrackingFrameResults = list[TrackingFrameResult]


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

    @abstractmethod
    def _track_with_backend(
        self, backend: ObjectTrackingBackendStrategy, request: TrackingFrameRequest
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Delegate one frame to the concrete tracking primitive."""


class OverlapTrackObjectsMethodStrategy(TrackObjectsMethodStrategy):
    """Track objects by maximum overlap between frames."""

    method = TrackingMethod.OVERLAP

    def _track_with_backend(
        self, backend: ObjectTrackingBackendStrategy, request: TrackingFrameRequest
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        return backend.track_by_overlap(request)


class DistanceTrackObjectsMethodStrategy(TrackObjectsMethodStrategy):
    """Track objects by minimum distance between centroids."""

    method = TrackingMethod.DISTANCE

    def _track_with_backend(
        self, backend: ObjectTrackingBackendStrategy, request: TrackingFrameRequest
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        return backend.track_by_distance(request)


@numpy(contract=ProcessingContract.PURE_3D)
@special_inputs("labels")
@special_outputs(
    (
        "tracking_results",
        csv_materializer(
            fields=TrackObjectsModule.measurement_feature_row_fields(
                TrackingObjectMeasurement,
            ),
            analysis_type="tracking",
        ),
    )
)
def track_objects(
    image: np.ndarray,
    labels: np.ndarray,
    object_name: str = "Objects",
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
    image_number_start: int = 1,
    tracking_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    _tracking_state: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Track objects across sequential frames."""
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
    )
    label_frames = _label_frames(labels)
    if _tracking_state is None:
        _tracking_state = _initial_tracking_state()
    tracking_strategy = TrackObjectsMethodStrategy.for_method(tracking_method)
    frame_results: TrackingFrameResults = []
    for frame_index, current_labels in enumerate(label_frames):
        image_number = int(image_number_start) + frame_index
        old_labels = _tracking_state.get("old_labels")
        old_object_numbers = _tracking_state.get(
            "old_object_numbers", np.array([], int)
        )
        max_object_number = int(_tracking_state.get("max_object_number", 0))
        new_labels, parent_obj_nums, parent_img_nums, max_object_number = (
            tracking_strategy.track(
                TrackingFrameRequest(
                    current_labels=current_labels,
                    old_labels=old_labels,
                    old_object_numbers=old_object_numbers,
                    max_object_number=max_object_number,
                    pixel_radius=pixel_radius,
                ),
                tracking_backend_provider,
            )
        )
        parent_img_nums = np.where(parent_obj_nums > 0, image_number - 1, 0)
        object_rows = _tracking_object_rows(
            current_labels,
            image_number=image_number,
            track_labels=new_labels,
            parent_object_numbers=parent_obj_nums,
            parent_image_numbers=parent_img_nums,
            previous_object_states=_tracking_state["track_states"],
            tracking_backend_provider=tracking_backend_provider,
        )
        new_object_count = int(np.sum(parent_obj_nums == 0))
        lost_object_count, split_count, merge_count = _tracking_transition_counts(
            old_labels, current_labels, old_object_numbers, new_labels
        )
        frame_results.append(
            (
                image_number,
                object_rows,
                new_object_count,
                lost_object_count,
                split_count,
                merge_count,
            )
        )
        _tracking_state["old_labels"] = current_labels.copy()
        _tracking_state["old_object_numbers"] = new_labels.copy()
        _tracking_state["max_object_number"] = max_object_number
    _apply_final_age_measurements(frame_results)
    rows: list[dict[str, Any]] = []
    for (
        image_number,
        object_rows,
        new_object_count,
        lost_object_count,
        split_count,
        merge_count,
    ) in frame_results:
        rows.extend(
            _tracking_object_row_mappings(
                object_rows,
                feature_suffix=str(int(pixel_radius)),
            )
        )
        rows.extend(
            _tracking_image_rows(
                image_number=image_number,
                object_rows=object_rows,
                new_object_count=new_object_count,
                lost_object_count=lost_object_count,
                split_count=split_count,
                merge_count=merge_count,
                feature_suffix=str(int(pixel_radius)),
                object_name=object_name,
            )
        )
    if image.ndim == 2:
        return (image[np.newaxis, ...], rows)
    return (image, rows)


def _initial_tracking_state() -> dict[str, Any]:
    return {
        "old_labels": None,
        "old_object_numbers": np.array([], int),
        "max_object_number": 0,
        "track_states": {},
    }


def _label_frames(labels: np.ndarray) -> np.ndarray:
    label_array = object_label_dense_array(labels, dtype=np.int32)
    if label_array.ndim == 2:
        return label_array[np.newaxis, ...]
    if label_array.ndim == 3:
        return label_array
    if label_array.ndim == 4 and label_array.shape[1] == 1:
        return label_array[:, 0]
    raise NotImplementedError(
        f"TrackObjects expects 2-D labels or a site stack of 2-D labels, got shape {label_array.shape!r}."
    )


def _tracking_object_rows(
    labels: np.ndarray,
    *,
    image_number: int,
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
                image_number=image_number,
                object_label=object_number,
                displacement=displacement,
                distance_traveled=distance_traveled,
                final_age=float("nan"),
                integrated_distance=integrated_distance,
                label=track_id,
                lifetime=lifetime,
                linearity=linearity,
                parent_image_number=float(parent_image_numbers[object_index]),
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
) -> tuple[int, int, int]:
    previous_counts = _positive_value_counts(previous_track_labels)
    current_counts = _positive_value_counts(current_track_labels)
    split_count = sum((1 for count in current_counts.values() if count > 1))
    if previous_labels is None:
        return (0, int(split_count), 0)
    lost_count, overlap_merge_count = _tracking_overlap_transition_counts(
        previous_labels, current_labels, previous_track_labels, current_track_labels
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


def _tracking_overlap_transition_counts(
    previous_labels: np.ndarray,
    current_labels: np.ndarray,
    previous_track_labels: np.ndarray,
    current_track_labels: np.ndarray,
) -> tuple[int, int]:
    """Return CP-style lost/merged object counts from inter-frame label overlap."""
    previous = np.asarray(previous_labels, dtype=int)
    current = np.asarray(current_labels, dtype=int)
    previous_object_ids = {int(value) for value in np.unique(previous) if value > 0}
    current_object_ids = {int(value) for value in np.unique(current) if value > 0}
    if not previous_object_ids:
        return (0, 0)
    if not current_object_ids:
        return (len(previous_object_ids), 0)
    previous_ids_by_current: dict[int, set[int]] = {
        label_id: set() for label_id in current_object_ids
    }
    current_ids_by_previous: dict[int, set[int]] = {
        label_id: set() for label_id in previous_object_ids
    }
    for previous_id, current_id in zip(previous.ravel(), current.ravel(), strict=False):
        previous_label = int(previous_id)
        current_label = int(current_id)
        if previous_label <= 0 or current_label <= 0:
            continue
        current_ids_by_previous.setdefault(previous_label, set()).add(current_label)
        previous_ids_by_current.setdefault(current_label, set()).add(previous_label)
    current_tracks = {
        int(value) for value in np.asarray(current_track_labels).ravel() if value > 0
    }
    lost_count = 0
    for previous_id in previous_object_ids:
        previous_track = (
            int(previous_track_labels[previous_id - 1])
            if previous_id <= len(previous_track_labels)
            else 0
        )
        if current_ids_by_previous[previous_id] or previous_track in current_tracks:
            continue
        lost_count += 1
    previous_tracks_by_current = {
        current_id: {
            int(previous_track_labels[previous_id - 1])
            for previous_id in previous_ids_by_current[current_id]
            if previous_id <= len(previous_track_labels)
            and int(previous_track_labels[previous_id - 1]) > 0
        }
        for current_id in current_object_ids
    }
    merge_count = sum(
        (1 for tracks in previous_tracks_by_current.values() if len(tracks) > 1)
    )
    return (int(lost_count), int(merge_count))


def _positive_value_counts(values: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for value in np.asarray(values, dtype=int).ravel():
        if value <= 0:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _apply_final_age_measurements(frame_results: TrackingFrameResults) -> None:
    labels_by_frame: dict[int, set[int]] = {}
    object_values: dict[TrackingObjectFrameKey, tuple[int, float]] = {}
    final_age_records: dict[TrackingObjectFrameKey, TrackingObjectMeasurement] = {}
    for image_number, object_rows, *_counts in frame_results:
        for measurement in object_rows:
            object_label = measurement.object_label
            key = (image_number, object_label)
            track_label = int(float(measurement.label))
            labels_by_frame.setdefault(image_number, set()).add(track_label)
            object_values[key] = (track_label, float(measurement.lifetime))
            final_age_records[key] = measurement
    last_image_number = frame_results[-1][0] if frame_results else 0
    for (image_number, object_label), (track_label, lifetime) in object_values.items():
        next_labels = labels_by_frame.get(image_number + 1, set())
        if image_number != last_image_number and track_label in next_labels:
            continue
        final_age_records[image_number, object_label].final_age = lifetime


def _tracking_object_row_mappings(
    object_rows: list[TrackingObjectMeasurement],
    *,
    feature_suffix: str,
) -> list[dict[str, Any]]:
    return TrackObjectsModule.measurement_feature_rows_from_records(
        tuple(object_rows),
        qualified_parts=(feature_suffix,),
    )


def _tracking_image_rows(
    *,
    image_number: int,
    object_rows: list[TrackingObjectMeasurement],
    new_object_count: int,
    lost_object_count: int,
    split_count: int,
    merge_count: int,
    feature_suffix: str,
    object_name: str,
) -> list[dict[str, Any]]:
    axis_values = {MeasurementRowAxisField.IMAGE_NUMBER.value: image_number}
    rows = TrackObjectsModule.measurement_feature_rows_from_records(
        (
            TrackingImageMeasurement(
                image_number=image_number,
                new_object_count=new_object_count,
                lost_object_count=lost_object_count,
                split_object_count=split_count,
                merged_object_count=merge_count,
            ),
        ),
        qualified_parts=(object_name, feature_suffix),
    )
    rows.extend(
        TrackObjectsModule.mean_measurement_feature_rows_from_records(
            tuple(object_rows),
            axis_values=axis_values,
            object_name=object_name,
            qualified_parts=(feature_suffix,),
        )
    )
    return rows


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
