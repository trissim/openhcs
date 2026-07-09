"""Neighbor-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from inspect import Parameter, signature
import logging
import os
import time
from typing import ClassVar
import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.constants.constants import MemoryType
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import runtime_bound_parameters
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_invocation import SliceIndexRuntimeParameter
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    required_setting_value,
    setting_values,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    ColumnarFieldsMeasurementRecordMixin,
    NoSourceMeasurementRecordMixin,
    TableMeasurementRecordRowsMixin,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    LABEL_PAYLOAD_FINAL,
    _label_payload_small_removed,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    ObjectInputBindingRequest,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    CellProfilerObjectInputPolicyMixin,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargDict,
)
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    BoundModuleSettings,
    CellProfilerModule,
    ImageArtifactOutputCapability,
    ImageArtifactOutputModule,
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    ObjectLabelArtifactInputCapability,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def _runtime_profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_runtime_profile(label: str, seconds: float, **fields: object) -> None:
    if not _runtime_profile_enabled():
        return
    field_text = " ".join((f"{key}={value}" for key, value in fields.items()))
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


def _profile_elapsed(label: str, start: float, **fields: object) -> float:
    now = time.perf_counter()
    _log_runtime_profile(label, now - start, **fields)
    return now


class DistanceMethod(Enum):
    """CellProfiler MeasureObjectNeighbors distance method."""

    def __new__(
        cls, absorbed_value: str, *cellprofiler_literals: str
    ) -> "DistanceMethod":
        obj = object.__new__(cls)
        obj._value_ = absorbed_value
        obj.cellprofiler_literals = (absorbed_value, *cellprofiler_literals)
        return obj

    ADJACENT = ("adjacent",)
    EXPAND = ("expand", "Expand until adjacent")
    WITHIN = ("within", "Within a specified distance")


@dataclass(frozen=True, slots=True)
class NeighborMeasurements:
    """Per-object neighbor measurements."""

    slice_index: int
    object_id: int
    scale: int | str
    number_of_neighbors: int
    percent_touching: float
    first_closest_object_number: int
    first_closest_distance: float
    second_closest_object_number: int
    second_closest_distance: float
    angle_between_neighbors: float


@dataclass(frozen=True, slots=True)
class NeighborDistancePlan:
    """Prepared label state for one neighbor distance method."""

    working_labels: np.ndarray
    distance: int
    measurement_scale: int | str


class NeighborDistancePlanner(
    EnumKeyedStrategyMixin[DistanceMethod], ABC, metaclass=AutoRegisterMeta
):
    """Prepare neighbor-distance state for one closed distance method."""

    __registry_key__ = "method_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"
    __enum_label_attr__ = "method_label"
    method_label: ClassVar[str | None] = None
    method: ClassVar[DistanceMethod | None] = None

    @classmethod
    def for_method(cls, method: DistanceMethod | str) -> "NeighborDistancePlanner":
        return cls.for_enum_member(coerce_cellprofiler_enum(DistanceMethod, method))

    @abstractmethod
    def plan(self, labels: np.ndarray, neighbor_distance: int) -> NeighborDistancePlan:
        """Return working labels and neighborhood distance."""


class AdjacentNeighborDistancePlanner(NeighborDistancePlanner):
    """Adjacent-neighbor topology without label expansion."""

    method = DistanceMethod.ADJACENT

    def plan(self, labels: np.ndarray, neighbor_distance: int) -> NeighborDistancePlan:
        del neighbor_distance
        return NeighborDistancePlan(labels.copy(), 1, "Adjacent")


class ExpandedNeighborDistancePlanner(NeighborDistancePlanner):
    """Expand labels until adjacent before measuring neighbors."""

    method = DistanceMethod.EXPAND

    def plan(self, labels: np.ndarray, neighbor_distance: int) -> NeighborDistancePlan:
        del neighbor_distance
        from scipy.ndimage import distance_transform_edt

        i, j = distance_transform_edt(
            labels == 0, return_distances=False, return_indices=True
        )
        return NeighborDistancePlan(labels[i, j], 1, "Expanded")


class WithinNeighborDistancePlanner(NeighborDistancePlanner):
    """Measure neighbors within a fixed distance."""

    method = DistanceMethod.WITHIN

    def plan(self, labels: np.ndarray, neighbor_distance: int) -> NeighborDistancePlan:
        return NeighborDistancePlan(
            labels.copy(), neighbor_distance, int(neighbor_distance)
        )


@dataclass(frozen=True, slots=True)
class MeasureObjectNeighborsObjectInputParameters:
    """Callable-signature authority for MeasureObjectNeighbors object inputs."""

    measured_labels: str
    small_removed_labels: str
    neighbor_labels: str
    small_removed_neighbor_labels: str
    neighbors_are_same_objects: str
    primary_image_parameter_count: ClassVar[int] = 1
    object_binding_parameter_count: ClassVar[int] = 5
    supported_parameter_kinds: ClassVar[frozenset] = frozenset(
        (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    )

    @classmethod
    def from_callable(
        cls, func: CellProfilerFunction
    ) -> "MeasureObjectNeighborsObjectInputParameters":
        parameters = tuple(signature(func).parameters.values())
        start = cls.primary_image_parameter_count
        stop = start + cls.object_binding_parameter_count
        object_parameters = parameters[start:stop]
        if len(object_parameters) != cls.object_binding_parameter_count:
            raise TypeError(
                "MeasureObjectNeighbors callable must declare contiguous object-binding parameters after its primary image parameter."
            )
        unsupported = tuple(
            (
                parameter
                for parameter in object_parameters
                if parameter.kind not in cls.supported_parameter_kinds
            )
        )
        if unsupported:
            raise TypeError(
                "MeasureObjectNeighbors object-binding parameters must be positional-or-keyword or keyword-only parameters."
            )
        return cls(*(parameter.name for parameter in object_parameters))

    @property
    def bound_parameter_names(self) -> tuple[str, ...]:
        return (
            self.measured_labels,
            self.small_removed_labels,
            self.neighbor_labels,
            self.small_removed_neighbor_labels,
            self.neighbors_are_same_objects,
        )


class MeasureObjectNeighborsInputPolicy(CellProfilerObjectInputPolicyMixin):
    """Bind MeasureObjectNeighbors object-label inputs."""

    def bound_parameter_names(
        self, plan: "CellProfilerModuleRuntimePlan"
    ) -> tuple[str, ...]:
        if not plan.object_inputs:
            return ()
        return MeasureObjectNeighborsObjectInputParameters.from_callable(
            plan.func
        ).bound_parameter_names

    def bind(self, request: ObjectInputBindingRequest) -> CellProfilerKwargDict:
        if len(request.object_inputs) not in (1, 2):
            raise NotImplementedError(
                f"MeasureObjectNeighbors requires one or two object runtime inputs, got {[spec.name for spec in request.object_inputs]}."
            )
        parameters = MeasureObjectNeighborsObjectInputParameters.from_callable(
            request.func
        )
        measured = request.object_inputs[0]
        neighbor = request.object_inputs[-1]
        measured_payload = request.label_payload_for(measured)
        neighbor_payload = (
            measured_payload
            if measured == neighbor
            else request.label_payload_for(neighbor)
        )
        same_objects = measured == neighbor
        neighbor_labels = None
        small_removed_neighbor_labels = None
        if not same_objects:
            neighbor_labels = LABEL_PAYLOAD_FINAL.value(neighbor_payload)
            small_removed_neighbor_labels = _label_payload_small_removed(
                neighbor_payload
            )
        return {
            parameters.measured_labels: LABEL_PAYLOAD_FINAL.value(measured_payload),
            parameters.small_removed_labels: _label_payload_small_removed(
                measured_payload
            ),
            parameters.neighbor_labels: neighbor_labels,
            parameters.small_removed_neighbor_labels: small_removed_neighbor_labels,
            parameters.neighbors_are_same_objects: same_objects,
        }


@dataclass(frozen=True, slots=True)
class NeighborRetainedImageRequest:
    """Own sidecar image materialization and return ordering for neighbor metrics."""

    labels: np.ndarray
    retain_neighbor_count_image: bool
    neighbor_count_colormap: str
    retain_percent_touching_image: bool
    percent_touching_colormap: str

    def empty_metric_image(self) -> np.ndarray:
        return np.zeros_like(self.labels, dtype=float)

    def output(
        self,
        image: np.ndarray,
        measurements: list,
        *,
        neighbor_count_image: np.ndarray,
        percent_touching_image: np.ndarray,
    ) -> tuple:
        retained = self.retained_images(
            neighbor_count_image=neighbor_count_image,
            percent_touching_image=percent_touching_image,
        )
        if retained:
            return (*retained, measurements)
        return (image, measurements)

    def retained_images(
        self, *, neighbor_count_image: np.ndarray, percent_touching_image: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        retained: list[np.ndarray] = []
        if self.retain_neighbor_count_image:
            retained.append(
                self.colored_metric_image(
                    neighbor_count_image, self.neighbor_count_colormap
                )
            )
        if self.retain_percent_touching_image:
            retained.append(
                self.colored_metric_image(
                    percent_touching_image, self.percent_touching_colormap
                )
            )
        return tuple(retained)

    def colored_metric_image(
        self, metric_image: np.ndarray, colormap_name: str
    ) -> np.ndarray:
        """Color one object metric image using CellProfiler-style masked RGB output."""
        import matplotlib.cm

        cmap_name = str(colormap_name).strip() or "Default"
        if cmap_name.lower() == "default":
            cmap_name = "viridis"
        scalar_mappable = matplotlib.cm.ScalarMappable(
            cmap=matplotlib.cm.get_cmap(cmap_name)
        )
        rgb = scalar_mappable.to_rgba(metric_image)[:, :, :3]
        rgb[self.labels <= 0] = 0
        return rgb


def variant_numbers_for_final_labels(
    final_labels: np.ndarray,
    variant_labels: np.ndarray,
    *,
    neighbor_backend: "NeighborTopologyBackendStrategy | None" = None,
) -> np.ndarray:
    """Map each final object ID to the corresponding variant object ID."""
    backend = neighbor_backend or neighbor_topology_backend()
    return backend.variant_numbers_for_final_labels(final_labels, variant_labels)


def labels_or_default(labels: np.ndarray | None, default: np.ndarray) -> np.ndarray:
    """Return a semantic label variant or the final labels when absent."""
    return default if labels is None else labels


def _labels_aligned_to_image_plane(
    labels: np.ndarray, image: np.ndarray, slice_index: int
) -> np.ndarray:
    """Project leading label-stack axes when a pure-2D image is being measured."""
    image_array = np.asarray(image)
    if (
        image_array.ndim == 2
        and labels.ndim > 2
        and (labels.shape[-2:] == image_array.shape)
    ):
        if 0 <= slice_index < labels.shape[0]:
            labels = labels[slice_index]
        if labels.ndim > 2:
            labels = np.max(labels, axis=tuple(range(labels.ndim - 2)))
        return labels.astype(labels.dtype, copy=False)
    return labels


def require_matching_shape(
    labels: np.ndarray, variant: np.ndarray, variant_name: str
) -> None:
    """Validate that one neighbor label variant shares the final label shape."""
    if labels.shape != variant.shape:
        raise ValueError(
            f"{variant_name} shape {variant.shape!r} does not match final labels shape={labels.shape!r}."
        )


@dataclass(frozen=True, slots=True)
class NeighborTopologyArrays:
    """Dense per-variant-object neighbor topology measurements."""

    neighbor_count: np.ndarray
    touching_pixel_count: np.ndarray


@dataclass(frozen=True, slots=True)
class NeighborClosestArrays:
    """Dense nearest-neighbor vectors and final object IDs."""

    first_x_vector: np.ndarray
    first_y_vector: np.ndarray
    second_x_vector: np.ndarray
    second_y_vector: np.ndarray
    angle_between_neighbors: np.ndarray
    final_first_object_number: np.ndarray
    final_second_object_number: np.ndarray


class NeighborTopologyBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Neighbor topology operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def measure_topology(
        self,
        working_labels: np.ndarray,
        neighbor_working_labels: np.ndarray,
        perimeter_outlines: np.ndarray,
        object_numbers: np.ndarray,
        *,
        distance: int,
        neighbors_are_same_objects: bool,
        footprint: np.ndarray,
        touching_footprint: np.ndarray,
        variant_object_count: int,
        variant_neighbor_count: int,
    ) -> NeighborTopologyArrays:
        """Return neighbor counts and touching-pixel counts."""

    @abstractmethod
    def variant_numbers_for_final_labels(
        self, final_labels: np.ndarray, variant_labels: np.ndarray
    ) -> np.ndarray:
        """Map final object IDs to their dominant variant object ID."""

    @abstractmethod
    def perimeter_counts(
        self, perimeter_outlines: np.ndarray, *, variant_object_count: int
    ) -> np.ndarray:
        """Return per-object perimeter pixel counts."""

    @abstractmethod
    def closest_neighbors(
        self,
        object_centers: np.ndarray,
        neighbor_centers: np.ndarray,
        object_numbers: np.ndarray,
        neighbor_numbers: np.ndarray,
        final_has_pixels: np.ndarray,
        neighbor_has_pixels: np.ndarray,
        *,
        neighbors_are_same_objects: bool,
        variant_object_count: int,
        variant_neighbor_count: int,
        final_object_count: int,
    ) -> NeighborClosestArrays:
        """Return nearest-neighbor vectors and final object numbering."""


class NumbaNumpyNeighborTopologyBackendStrategy(NeighborTopologyBackendStrategy):
    """Numba-accelerated NumPy backend for neighbor topology."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        working = np.array([[0, 1, 1], [0, 0, 2], [3, 0, 0]], dtype=np.int32)
        neighbor = np.array([[0, 1, 0], [2, 2, 0], [0, 3, 3]], dtype=np.int32)
        outline = (working > 0).astype(np.int32)
        object_numbers = np.array([1, 2, 3], dtype=np.int32)
        footprint = np.ones((3, 3), dtype=np.bool_)
        self.measure_topology(
            working,
            neighbor,
            outline,
            object_numbers,
            distance=1,
            neighbors_are_same_objects=False,
            footprint=footprint,
            touching_footprint=footprint,
            variant_object_count=3,
            variant_neighbor_count=3,
        )
        self.perimeter_counts(outline, variant_object_count=3)
        self.variant_numbers_for_final_labels(working, neighbor)
        self.closest_neighbors(
            np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
            np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 2.0]]),
            object_numbers,
            object_numbers,
            np.ones(4, dtype=np.bool_),
            np.ones(4, dtype=np.bool_),
            neighbors_are_same_objects=False,
            variant_object_count=3,
            variant_neighbor_count=3,
            final_object_count=3,
        )

    def measure_topology(
        self,
        working_labels: np.ndarray,
        neighbor_working_labels: np.ndarray,
        perimeter_outlines: np.ndarray,
        object_numbers: np.ndarray,
        *,
        distance: int,
        neighbors_are_same_objects: bool,
        footprint: np.ndarray,
        touching_footprint: np.ndarray,
        variant_object_count: int,
        variant_neighbor_count: int,
    ) -> NeighborTopologyArrays:
        working_array = np.ascontiguousarray(working_labels, dtype=np.int32)
        neighbor_array = np.ascontiguousarray(neighbor_working_labels, dtype=np.int32)
        outline_array = np.ascontiguousarray(perimeter_outlines, dtype=np.int32)
        object_numbers_array = np.ascontiguousarray(object_numbers, dtype=np.int32)
        if working_array.ndim > 2:
            working_array = np.max(
                working_array, axis=tuple(range(working_array.ndim - 2))
            )
        if neighbor_array.ndim > 2:
            neighbor_array = np.max(
                neighbor_array, axis=tuple(range(neighbor_array.ndim - 2))
            )
        if outline_array.ndim > 2:
            outline_array = np.max(
                outline_array, axis=tuple(range(outline_array.ndim - 2))
            )
        if working_array.ndim != 2:
            raise NotImplementedError(
                "CellProfiler neighbor topology currently supports 2-D labels."
            )
        if neighbor_array.shape != working_array.shape:
            raise ValueError(
                f"Neighbor topology labels must share a shape; got {working_array.shape!r} and {neighbor_array.shape!r}."
            )
        measured_object_mask = np.zeros(int(variant_object_count) + 1, dtype=np.bool_)
        for object_number in object_numbers_array:
            if 0 < object_number <= int(variant_object_count):
                measured_object_mask[int(object_number)] = True
        offset_y, offset_x = _footprint_offsets(footprint)
        touching_offset_y, touching_offset_x = _footprint_offsets(touching_footprint)
        neighbor_count, touching_pixel_count = _measure_neighbor_topology_numba(
            working_array,
            neighbor_array,
            outline_array,
            measured_object_mask,
            offset_y,
            offset_x,
            touching_offset_y,
            touching_offset_x,
            bool(neighbors_are_same_objects),
            int(variant_object_count),
            int(variant_neighbor_count),
        )
        return NeighborTopologyArrays(
            neighbor_count=neighbor_count, touching_pixel_count=touching_pixel_count
        )

    def variant_numbers_for_final_labels(
        self, final_labels: np.ndarray, variant_labels: np.ndarray
    ) -> np.ndarray:
        final_array = np.ascontiguousarray(final_labels, dtype=np.int32)
        variant_array = np.ascontiguousarray(variant_labels, dtype=np.int32)
        if final_array.shape != variant_array.shape:
            raise ValueError(
                f"Final and variant labels must share a shape; got {final_array.shape!r} and {variant_array.shape!r}."
            )
        final_count = int(final_array.max()) if final_array.size else 0
        variant_count = int(variant_array.max()) if variant_array.size else 0
        return _variant_numbers_for_final_labels_numba(
            final_array.ravel(), variant_array.ravel(), final_count, variant_count
        )

    def perimeter_counts(
        self, perimeter_outlines: np.ndarray, *, variant_object_count: int
    ) -> np.ndarray:
        outline_array = np.ascontiguousarray(perimeter_outlines, dtype=np.int32)
        return np.maximum(
            _perimeter_counts_numba(outline_array.ravel(), int(variant_object_count)), 1
        )

    def closest_neighbors(
        self,
        object_centers: np.ndarray,
        neighbor_centers: np.ndarray,
        object_numbers: np.ndarray,
        neighbor_numbers: np.ndarray,
        final_has_pixels: np.ndarray,
        neighbor_has_pixels: np.ndarray,
        *,
        neighbors_are_same_objects: bool,
        variant_object_count: int,
        variant_neighbor_count: int,
        final_object_count: int,
    ) -> NeighborClosestArrays:
        result = _closest_neighbors_numba(
            np.ascontiguousarray(object_centers, dtype=np.float64),
            np.ascontiguousarray(neighbor_centers, dtype=np.float64),
            np.ascontiguousarray(object_numbers, dtype=np.int32),
            np.ascontiguousarray(neighbor_numbers, dtype=np.int32),
            np.ascontiguousarray(final_has_pixels, dtype=np.bool_),
            np.ascontiguousarray(neighbor_has_pixels, dtype=np.bool_),
            bool(neighbors_are_same_objects),
            int(variant_object_count),
            int(variant_neighbor_count),
            int(final_object_count),
        )
        return NeighborClosestArrays(*result)


def neighbor_topology_backend(
    *, backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION
) -> NeighborTopologyBackendStrategy:
    """Return the selected neighbor topology backend."""
    return NeighborTopologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY, backend_provider=backend_provider
    )


@numpy(contract=ProcessingContract.PURE_2D)
@runtime_bound_parameters(SliceIndexRuntimeParameter)
def measure_object_neighbors(
    image: np.ndarray,
    labels: np.ndarray,
    small_removed_labels: np.ndarray | None = None,
    neighbor_labels: np.ndarray | None = None,
    small_removed_neighbor_labels: np.ndarray | None = None,
    neighbors_are_same_objects: bool = True,
    distance_method: DistanceMethod | str = DistanceMethod.EXPAND,
    neighbor_distance: int = 5,
    consider_discarded_objects: bool = True,
    retain_neighbor_count_image: bool = False,
    neighbor_count_colormap: str = "Default",
    retain_percent_touching_image: bool = False,
    percent_touching_colormap: str = "Default",
    neighbor_topology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    outline_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    relationship_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    slice_index: int | None = None,
) -> tuple[np.ndarray, list]:
    """Measure neighbor relationships between objects."""
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.outlines import (
        ObjectOutlineBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.relationships import (
        ObjectRelationshipBackendStrategy,
    )

    profile_start = time.perf_counter()
    profile_mark = profile_start
    label_array = object_label_dense_array(labels, dtype=np.int32)
    if (
        slice_index is None
        and np.asarray(image).ndim == 2
        and (label_array.ndim == 3)
        and (label_array.shape[-2:] == np.asarray(image).shape)
    ):
        rows: list = []
        for plane_index in range(label_array.shape[0]):
            _image, plane_rows = measure_object_neighbors(
                image,
                label_array[plane_index],
                neighbor_labels=neighbor_labels,
                small_removed_labels=small_removed_labels,
                small_removed_neighbor_labels=small_removed_neighbor_labels,
                distance_method=distance_method,
                neighbor_distance=neighbor_distance,
                neighbors_are_same_objects=neighbors_are_same_objects,
                consider_discarded_objects=consider_discarded_objects,
                retain_neighbor_count_image=retain_neighbor_count_image,
                neighbor_count_colormap=neighbor_count_colormap,
                retain_percent_touching_image=retain_percent_touching_image,
                percent_touching_colormap=percent_touching_colormap,
                neighbor_topology_backend_provider=neighbor_topology_backend_provider,
                outline_backend_provider=outline_backend_provider,
                morphology_backend_provider=morphology_backend_provider,
                relationship_backend_provider=relationship_backend_provider,
                slice_index=plane_index,
            )
            rows.extend(plane_rows)
        return (image, rows)
    slice_index = 0 if slice_index is None else int(slice_index)
    labels = _labels_aligned_to_image_plane(label_array, image, slice_index)
    final_labels = labels
    retained_image_request = NeighborRetainedImageRequest(
        labels=final_labels,
        retain_neighbor_count_image=retain_neighbor_count_image,
        neighbor_count_colormap=neighbor_count_colormap,
        retain_percent_touching_image=retain_percent_touching_image,
        percent_touching_colormap=percent_touching_colormap,
    )
    neighbor_final_labels = (
        final_labels
        if neighbor_labels is None
        else _labels_aligned_to_image_plane(
            object_label_dense_array(neighbor_labels, dtype=np.int32),
            image,
            slice_index,
        )
    )
    measured_variant_labels = labels_or_default(
        (
            None
            if small_removed_labels is None
            else _labels_aligned_to_image_plane(
                object_label_dense_array(small_removed_labels, dtype=np.int32),
                image,
                slice_index,
            )
        ),
        final_labels,
    )
    neighbor_variant_labels = (
        measured_variant_labels
        if neighbors_are_same_objects and small_removed_neighbor_labels is None
        else labels_or_default(
            (
                None
                if small_removed_neighbor_labels is None
                else _labels_aligned_to_image_plane(
                    object_label_dense_array(
                        small_removed_neighbor_labels, dtype=np.int32
                    ),
                    image,
                    slice_index,
                )
            ),
            neighbor_final_labels,
        )
    )
    require_matching_shape(
        final_labels, measured_variant_labels, "small_removed_labels"
    )
    require_matching_shape(
        neighbor_final_labels, neighbor_variant_labels, "small_removed_neighbor_labels"
    )
    profile_mark = _profile_elapsed(
        "measure_object_neighbors.normalize_inputs",
        profile_mark,
        shape=final_labels.shape,
    )
    final_object_count = int(final_labels.max()) if final_labels.size else 0
    if final_object_count == 0:
        empty_metric_image = retained_image_request.empty_metric_image()
        return retained_image_request.output(
            image,
            [],
            neighbor_count_image=empty_metric_image,
            percent_touching_image=empty_metric_image,
        )
    measured_topology_labels = measured_variant_labels
    neighbor_topology_labels = neighbor_variant_labels.copy()
    if not consider_discarded_objects:
        neighbor_topology_labels[neighbor_final_labels <= 0] = 0
    neighbor_backend = neighbor_topology_backend(
        backend_provider=neighbor_topology_backend_provider
    )
    object_numbers = variant_numbers_for_final_labels(
        final_labels, measured_variant_labels, neighbor_backend=neighbor_backend
    )
    neighbor_numbers = (
        object_numbers
        if neighbors_are_same_objects
        else variant_numbers_for_final_labels(
            neighbor_final_labels,
            neighbor_topology_labels,
            neighbor_backend=neighbor_backend,
        )
    )
    profile_mark = _profile_elapsed(
        "measure_object_neighbors.variant_mapping",
        profile_mark,
        final_object_count=final_object_count,
        neighbors_are_same_objects=neighbors_are_same_objects,
    )
    final_has_pixels = (
        np.bincount(final_labels.ravel(), minlength=final_object_count + 1)[1:] > 0
    )
    neighbor_final_count = (
        final_object_count
        if neighbors_are_same_objects
        else int(neighbor_final_labels.max()) if neighbor_final_labels.size else 0
    )
    neighbor_has_pixels = (
        final_has_pixels
        if neighbors_are_same_objects
        else np.bincount(
            neighbor_final_labels.ravel(), minlength=neighbor_final_count + 1
        )[1:]
        > 0
    )
    variant_object_count = (
        int(measured_topology_labels.max()) if measured_topology_labels.size else 0
    )
    variant_neighbor_count = (
        int(neighbor_topology_labels.max()) if neighbor_topology_labels.size else 0
    )
    if variant_object_count == 0 or variant_neighbor_count == 0:
        measurements = [
            NeighborMeasurements(
                slice_index=slice_index,
                object_id=i + 1,
                scale=coerce_cellprofiler_enum(DistanceMethod, distance_method).value,
                number_of_neighbors=0,
                percent_touching=0.0,
                first_closest_object_number=0,
                first_closest_distance=0.0,
                second_closest_object_number=0,
                second_closest_distance=0.0,
                angle_between_neighbors=0.0,
            )
            for i in range(final_object_count)
        ]
        empty_metric_image = retained_image_request.empty_metric_image()
        return retained_image_request.output(
            image,
            measurements,
            neighbor_count_image=empty_metric_image,
            percent_touching_image=empty_metric_image,
        )
    neighbor_count = np.zeros(variant_object_count)
    first_x_vector = np.zeros(variant_object_count)
    second_x_vector = np.zeros(variant_object_count)
    first_y_vector = np.zeros(variant_object_count)
    second_y_vector = np.zeros(variant_object_count)
    angle = np.zeros(variant_object_count)
    percent_touching = np.zeros(variant_object_count)
    final_first_object_number = np.zeros(final_object_count, dtype=int)
    final_second_object_number = np.zeros(final_object_count, dtype=int)
    normalized_distance_method = coerce_cellprofiler_enum(
        DistanceMethod, distance_method
    )
    distance_plan = NeighborDistancePlanner.for_method(normalized_distance_method).plan(
        measured_topology_labels, neighbor_distance
    )
    profile_mark = _profile_elapsed(
        "measure_object_neighbors.distance_plan",
        profile_mark,
        method=normalized_distance_method.value,
        variant_object_count=variant_object_count,
        variant_neighbor_count=variant_neighbor_count,
    )
    working_labels = distance_plan.working_labels
    distance = distance_plan.distance
    measurement_scale = distance_plan.measurement_scale
    neighbor_working_labels = (
        working_labels.copy()
        if neighbors_are_same_objects
        and normalized_distance_method is DistanceMethod.EXPAND
        else neighbor_topology_labels
    )
    if variant_neighbor_count > (1 if neighbors_are_same_objects else 0):
        relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type(
            backend_provider=relationship_backend_provider
        )
        ocenters = relationship_backend.label_centers(measured_variant_labels)
        ncenters = relationship_backend.label_centers(neighbor_variant_labels)
        profile_mark = _profile_elapsed(
            "measure_object_neighbors.centers",
            profile_mark,
            variant_object_count=variant_object_count,
            variant_neighbor_count=variant_neighbor_count,
        )
        perimeter_outlines = ObjectOutlineBackendStrategy.for_memory_type(
            backend_provider=outline_backend_provider
        ).outline(working_labels)
        perimeters = neighbor_backend.perimeter_counts(
            perimeter_outlines, variant_object_count=variant_object_count
        )
        profile_mark = _profile_elapsed(
            "measure_object_neighbors.outline_perimeters",
            profile_mark,
            shape=working_labels.shape,
        )
        if variant_neighbor_count >= (2 if neighbors_are_same_objects else 1):
            closest = neighbor_backend.closest_neighbors(
                ocenters,
                ncenters,
                object_numbers,
                neighbor_numbers,
                final_has_pixels,
                neighbor_has_pixels,
                neighbors_are_same_objects=neighbors_are_same_objects,
                variant_object_count=variant_object_count,
                variant_neighbor_count=variant_neighbor_count,
                final_object_count=final_object_count,
            )
            first_x_vector = closest.first_x_vector
            first_y_vector = closest.first_y_vector
            second_x_vector = closest.second_x_vector
            second_y_vector = closest.second_y_vector
            angle = closest.angle_between_neighbors
            final_first_object_number = closest.final_first_object_number
            final_second_object_number = closest.final_second_object_number
            profile_mark = _profile_elapsed(
                "measure_object_neighbors.closest",
                profile_mark,
                variant_object_count=variant_object_count,
                variant_neighbor_count=variant_neighbor_count,
            )
        morphology_backend = MorphologyBackendStrategy.for_memory_type(
            backend_provider=morphology_backend_provider
        )
        topology = neighbor_backend.measure_topology(
            working_labels,
            neighbor_working_labels,
            perimeter_outlines,
            object_numbers,
            distance=distance,
            neighbors_are_same_objects=neighbors_are_same_objects,
            footprint=morphology_backend.disk_footprint(distance),
            touching_footprint=morphology_backend.disk_footprint(distance + 0.5),
            variant_object_count=variant_object_count,
            variant_neighbor_count=variant_neighbor_count,
        )
        neighbor_count = topology.neighbor_count
        percent_touching = topology.touching_pixel_count * 100 / perimeters
        profile_mark = _profile_elapsed(
            "measure_object_neighbors.topology", profile_mark, distance=distance
        )
    neighbor_count_image = np.zeros(final_labels.shape, dtype=float)
    percent_touching_image = np.zeros(final_labels.shape, dtype=float)
    object_mask = final_labels > 0
    if np.any(object_mask):
        final_indexes = final_labels[object_mask] - 1
        variant_numbers = object_numbers[final_indexes]
        valid_variant_mask = variant_numbers > 0
        neighbor_count_values = np.zeros(final_indexes.shape, dtype=float)
        percent_touching_values = np.zeros(final_indexes.shape, dtype=float)
        variant_indexes = variant_numbers[valid_variant_mask] - 1
        neighbor_count_values[valid_variant_mask] = neighbor_count[variant_indexes]
        percent_touching_values[valid_variant_mask] = percent_touching[variant_indexes]
        neighbor_count_image[object_mask] = neighbor_count_values
        percent_touching_image[object_mask] = percent_touching_values
    profile_mark = _profile_elapsed(
        "measure_object_neighbors.metric_images",
        profile_mark,
        final_object_count=final_object_count,
    )
    measurements = []
    for i in range(final_object_count):
        object_number = object_numbers[i]
        object_index = object_number - 1
        if object_number <= 0:
            measurements.append(
                NeighborMeasurements(
                    slice_index=slice_index,
                    object_id=i + 1,
                    scale=measurement_scale,
                    number_of_neighbors=0,
                    percent_touching=0.0,
                    first_closest_object_number=0,
                    first_closest_distance=0.0,
                    second_closest_object_number=0,
                    second_closest_distance=0.0,
                    angle_between_neighbors=0.0,
                )
            )
            continue
        first_dist = np.sqrt(
            first_x_vector[object_index] ** 2 + first_y_vector[object_index] ** 2
        )
        second_dist = np.sqrt(
            second_x_vector[object_index] ** 2 + second_y_vector[object_index] ** 2
        )
        measurements.append(
            NeighborMeasurements(
                slice_index=slice_index,
                object_id=i + 1,
                scale=measurement_scale,
                number_of_neighbors=int(neighbor_count[object_index]),
                percent_touching=float(percent_touching[object_index]),
                first_closest_object_number=int(final_first_object_number[i]),
                first_closest_distance=float(first_dist),
                second_closest_object_number=int(final_second_object_number[i]),
                second_closest_distance=float(second_dist),
                angle_between_neighbors=float(angle[object_index]),
            )
        )
    profile_mark = _profile_elapsed(
        "measure_object_neighbors.rows", profile_mark, row_count=len(measurements)
    )
    del profile_mark
    _log_runtime_profile(
        "measure_object_neighbors.total",
        time.perf_counter() - profile_start,
        final_object_count=final_object_count,
        variant_object_count=variant_object_count,
        variant_neighbor_count=variant_neighbor_count,
    )
    return retained_image_request.output(
        image,
        measurements,
        neighbor_count_image=neighbor_count_image,
        percent_touching_image=percent_touching_image,
    )


def _prepare_measure_object_neighbors() -> None:
    """Compile neighbor topology kernels before benchmark execution."""
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[4:8, 4:8] = 1
    labels[4:8, 10:14] = 2
    measure_object_neighbors.__wrapped__(
        np.zeros_like(labels, dtype=np.float32),
        labels,
        distance_method=DistanceMethod.WITHIN,
        neighbor_distance=3,
    )


measure_object_neighbors.__openhcs_prepare__ = _prepare_measure_object_neighbors


def _footprint_offsets(footprint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    footprint_array = np.asarray(footprint, dtype=bool)
    if footprint_array.ndim != 2:
        raise NotImplementedError(
            "CellProfiler neighbor topology currently supports 2-D footprints."
        )
    center_y = footprint_array.shape[0] // 2
    center_x = footprint_array.shape[1] // 2
    coords = np.argwhere(footprint_array)
    offsets = np.ascontiguousarray(
        np.column_stack((coords[:, 0] - center_y, coords[:, 1] - center_x)),
        dtype=np.int64,
    )
    return (offsets[:, 0], offsets[:, 1])


@njit(cache=True)
def _measure_neighbor_topology_numba(
    working_labels: np.ndarray,
    neighbor_working_labels: np.ndarray,
    perimeter_outlines: np.ndarray,
    measured_object_mask: np.ndarray,
    offset_y: np.ndarray,
    offset_x: np.ndarray,
    touching_offset_y: np.ndarray,
    touching_offset_x: np.ndarray,
    neighbors_are_same_objects: bool,
    variant_object_count: int,
    variant_neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = working_labels.shape
    adjacency = np.zeros(
        (variant_object_count, variant_neighbor_count + 1), dtype=np.bool_
    )
    touching_pixel_count = np.zeros(variant_object_count, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            object_number = working_labels[y, x]
            if (
                object_number <= 0
                or object_number > variant_object_count
                or (not measured_object_mask[object_number])
            ):
                continue
            object_index = object_number - 1
            if perimeter_outlines[y, x] != object_number:
                continue
            for offset_index in range(offset_y.size):
                neighbor_y = y + offset_y[offset_index]
                neighbor_x = x + offset_x[offset_index]
                if (
                    neighbor_y < 0
                    or neighbor_y >= height
                    or neighbor_x < 0
                    or (neighbor_x >= width)
                ):
                    continue
                neighbor_number = neighbor_working_labels[neighbor_y, neighbor_x]
                if neighbor_number <= 0 or neighbor_number > variant_neighbor_count:
                    continue
                if neighbors_are_same_objects and neighbor_number == object_number:
                    continue
                adjacency[object_index, neighbor_number] = True
            for offset_index in range(touching_offset_y.size):
                neighbor_y = y + touching_offset_y[offset_index]
                neighbor_x = x + touching_offset_x[offset_index]
                if (
                    neighbor_y < 0
                    or neighbor_y >= height
                    or neighbor_x < 0
                    or (neighbor_x >= width)
                ):
                    continue
                if neighbors_are_same_objects:
                    touches = (
                        working_labels[neighbor_y, neighbor_x] != 0
                        and working_labels[neighbor_y, neighbor_x] != object_number
                    )
                else:
                    touches = neighbor_working_labels[neighbor_y, neighbor_x] != 0
                if touches:
                    touching_pixel_count[object_index] += 1.0
                    break
    neighbor_count = np.zeros(variant_object_count, dtype=np.float64)
    for object_index in range(variant_object_count):
        count = 0.0
        for neighbor_number in range(1, variant_neighbor_count + 1):
            if adjacency[object_index, neighbor_number]:
                count += 1.0
        neighbor_count[object_index] = count
    return (neighbor_count, touching_pixel_count)


@njit(cache=True)
def _variant_numbers_for_final_labels_numba(
    final_labels_flat: np.ndarray,
    variant_labels_flat: np.ndarray,
    final_count: int,
    variant_count: int,
) -> np.ndarray:
    numbers = np.zeros(final_count, dtype=np.int32)
    if final_count == 0 or variant_count == 0:
        return numbers
    overlaps = np.zeros((final_count + 1, variant_count + 1), dtype=np.int32)
    for index in range(final_labels_flat.size):
        final_number = final_labels_flat[index]
        variant_number = variant_labels_flat[index]
        if (
            final_number > 0
            and final_number <= final_count
            and (variant_number > 0)
            and (variant_number <= variant_count)
        ):
            overlaps[final_number, variant_number] += 1
    for final_number in range(1, final_count + 1):
        best_variant = 0
        best_count = 0
        for variant_number in range(1, variant_count + 1):
            count = overlaps[final_number, variant_number]
            if count > best_count:
                best_count = count
                best_variant = variant_number
        numbers[final_number - 1] = best_variant
    return numbers


@njit(cache=True)
def _perimeter_counts_numba(
    perimeter_outlines_flat: np.ndarray, variant_object_count: int
) -> np.ndarray:
    counts = np.zeros(variant_object_count, dtype=np.float64)
    for index in range(perimeter_outlines_flat.size):
        object_number = perimeter_outlines_flat[index]
        if object_number > 0 and object_number <= variant_object_count:
            counts[object_number - 1] += 1.0
    return counts


@njit(cache=True)
def _closest_neighbors_numba(
    object_centers: np.ndarray,
    neighbor_centers: np.ndarray,
    object_numbers: np.ndarray,
    neighbor_numbers: np.ndarray,
    final_has_pixels: np.ndarray,
    neighbor_has_pixels: np.ndarray,
    neighbors_are_same_objects: bool,
    variant_object_count: int,
    variant_neighbor_count: int,
    final_object_count: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    first_x_vector = np.zeros(variant_object_count, dtype=np.float64)
    first_y_vector = np.zeros(variant_object_count, dtype=np.float64)
    second_x_vector = np.zeros(variant_object_count, dtype=np.float64)
    second_y_vector = np.zeros(variant_object_count, dtype=np.float64)
    angle = np.zeros(variant_object_count, dtype=np.float64)
    final_first_object_number = np.zeros(final_object_count, dtype=np.int64)
    final_second_object_number = np.zeros(final_object_count, dtype=np.int64)
    for object_index in range(variant_object_count):
        if object_index >= object_centers.shape[0]:
            continue
        object_y = object_centers[object_index, 0]
        object_x = object_centers[object_index, 1]
        if not (np.isfinite(object_y) and np.isfinite(object_x)):
            continue
        first_distance = np.inf
        second_distance = np.inf
        first_neighbor = -1
        second_neighbor = -1
        for neighbor_index in range(variant_neighbor_count):
            if neighbor_index >= neighbor_centers.shape[0]:
                continue
            if neighbors_are_same_objects and neighbor_index == object_index:
                continue
            neighbor_y = neighbor_centers[neighbor_index, 0]
            neighbor_x = neighbor_centers[neighbor_index, 1]
            if not (np.isfinite(neighbor_y) and np.isfinite(neighbor_x)):
                continue
            dy = object_y - neighbor_y
            dx = object_x - neighbor_x
            distance = dy * dy + dx * dx
            if distance < first_distance:
                second_distance = first_distance
                second_neighbor = first_neighbor
                first_distance = distance
                first_neighbor = neighbor_index
            elif distance < second_distance:
                second_distance = distance
                second_neighbor = neighbor_index
        if first_neighbor >= 0:
            first_x_vector[object_index] = (
                neighbor_centers[first_neighbor, 1] - object_x
            )
            first_y_vector[object_index] = (
                neighbor_centers[first_neighbor, 0] - object_y
            )
        if second_neighbor >= 0:
            second_x_vector[object_index] = (
                neighbor_centers[second_neighbor, 1] - object_x
            )
            second_y_vector[object_index] = (
                neighbor_centers[second_neighbor, 0] - object_y
            )
        norm1 = np.sqrt(
            first_x_vector[object_index] * first_x_vector[object_index]
            + first_y_vector[object_index] * first_y_vector[object_index]
        )
        norm2 = np.sqrt(
            second_x_vector[object_index] * second_x_vector[object_index]
            + second_y_vector[object_index] * second_y_vector[object_index]
        )
        if norm1 > 0.0 and norm2 > 0.0:
            dot = (
                first_x_vector[object_index] * second_x_vector[object_index]
                + first_y_vector[object_index] * second_y_vector[object_index]
            ) / (norm1 * norm2)
            if dot < -1.0:
                dot = -1.0
            elif dot > 1.0:
                dot = 1.0
            angle[object_index] = np.arccos(dot) * 180.0 / np.pi
    for final_object_index in range(final_object_count):
        if (
            final_object_index >= final_has_pixels.size
            or not final_has_pixels[final_object_index]
        ):
            continue
        object_number = object_numbers[final_object_index]
        object_index = object_number - 1
        if (
            object_index < 0
            or object_index >= variant_object_count
            or object_index >= object_centers.shape[0]
        ):
            continue
        object_y = object_centers[object_index, 0]
        object_x = object_centers[object_index, 1]
        if not (np.isfinite(object_y) and np.isfinite(object_x)):
            continue
        first_distance = np.inf
        second_distance = np.inf
        first_final_neighbor = 0
        second_final_neighbor = 0
        for final_neighbor_index in range(neighbor_numbers.size):
            if (
                final_neighbor_index >= neighbor_has_pixels.size
                or not neighbor_has_pixels[final_neighbor_index]
            ):
                continue
            if (
                neighbors_are_same_objects
                and final_neighbor_index == final_object_index
            ):
                continue
            neighbor_number = neighbor_numbers[final_neighbor_index]
            neighbor_index = neighbor_number - 1
            if (
                neighbor_index < 0
                or neighbor_index >= variant_neighbor_count
                or neighbor_index >= neighbor_centers.shape[0]
            ):
                continue
            neighbor_y = neighbor_centers[neighbor_index, 0]
            neighbor_x = neighbor_centers[neighbor_index, 1]
            if not (np.isfinite(neighbor_y) and np.isfinite(neighbor_x)):
                continue
            dy = object_y - neighbor_y
            dx = object_x - neighbor_x
            distance = dy * dy + dx * dx
            if distance < first_distance:
                second_distance = first_distance
                second_final_neighbor = first_final_neighbor
                first_distance = distance
                first_final_neighbor = final_neighbor_index + 1
            elif distance < second_distance:
                second_distance = distance
                second_final_neighbor = final_neighbor_index + 1
        final_first_object_number[final_object_index] = first_final_neighbor
        final_second_object_number[final_object_index] = second_final_neighbor
    return (
        first_x_vector,
        first_y_vector,
        second_x_vector,
        second_y_vector,
        angle,
        final_first_object_number,
        final_second_object_number,
    )


class MeasureObjectNeighborsModule(
    MeasureObjectNeighborsInputPolicy,
    TableMeasurementRecordRowsMixin,
    NoSourceMeasurementRecordMixin,
    ColumnarFieldsMeasurementRecordMixin,
    ObjectArtifactInputModule,
    ImageArtifactOutputModule,
    MeasurementArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "MeasureObjectNeighbors"
    function_name = "measure_object_neighbors"
    validated = True
    confidence = 1.0
    measurement_category_prefixes = (("neighbors",),)
    measured_objects_setting = SettingNameFamily("Select objects to measure")
    neighbor_objects_setting = SettingNameFamily(
        "Select neighboring objects to measure"
    )
    retain_count_image_setting = (
        "Retain the image of objects colored by numbers of neighbors?"
    )
    retain_percent_image_setting = (
        "Retain the image of objects colored by percent of touching pixels?"
    )
    output_image_setting = SettingNameFamily("Name the output image")
    object_input_settings = (measured_objects_setting, neighbor_objects_setting)
    image_output_settings = (output_image_setting,)
    ignored_settings = (
        measured_objects_setting,
        neighbor_objects_setting,
        retain_count_image_setting,
        retain_percent_image_setting,
        output_image_setting,
        "Select colormap",
    )
    setting_bindings = (
        SettingToKeywordBinding("Method to determine neighbors", "distance_method"),
        SettingToKeywordBinding(
            "Neighbor distance", "neighbor_distance", parse_cellprofiler_int
        ),
        SettingToKeywordBinding(
            "Consider objects discarded for touching image border?",
            "consider_discarded_objects",
            parse_cellprofiler_bool,
        ),
    )

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: "BoundModuleSettings"
    ) -> "BoundModuleSettings":
        colormaps = module.get_setting_values("Select colormap")
        kwargs = {
            **dict(bound.kwargs),
            "retain_neighbor_count_image": parse_cellprofiler_bool(
                module.get_setting(
                    "Retain the image of objects colored by numbers of neighbors?", "No"
                )
            ),
            "neighbor_count_colormap": colormaps[0] if colormaps else "Default",
            "retain_percent_touching_image": parse_cellprofiler_bool(
                module.get_setting(
                    "Retain the image of objects colored by percent of touching pixels?",
                    "No",
                )
            ),
            "percent_touching_colormap": (
                colormaps[1]
                if len(colormaps) > 1
                else colormaps[0] if colormaps else "Default"
            ),
        }
        return BoundModuleSettings(
            kwargs,
            bound.unmapped_kwargs,
            bound.invocation_options,
            bound.setting_coverage,
        )

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        measured = ObjectLabelArtifactInputCapability.bind_artifact(cls, builder, module, ObjectLabelArtifactInputCapability.spec(required_setting_value(module, cls.measured_objects_setting)))
        neighbors = ObjectLabelArtifactInputCapability.bind_artifact(cls, builder, module, ObjectLabelArtifactInputCapability.spec(required_setting_value(module, cls.neighbor_objects_setting)))
        output_names = setting_values(module, cls.output_image_setting)
        outputs = []
        if optional_setting_value(
            module, cls.retain_count_image_setting
        ) in {"Yes", "yes", "True", "true"}:
            outputs.append(
                cls.image_output_artifact(
                    builder,
                    module,
                    output_names[0],
                    setting=cls.output_image_setting,
                )
            )
        if optional_setting_value(
            module, cls.retain_percent_image_setting
        ) in {"Yes", "yes", "True", "true"}:
            outputs.append(
                cls.image_output_artifact(
                    builder,
                    module,
                    output_names[1],
                    setting=cls.output_image_setting,
                )
            )
        outputs.append(cls.measurement_output_artifact(builder, module))
        return assembler.assemble_contract(
            module, builder, inputs=[measured, neighbors], outputs=outputs
        )

    @classmethod
    def compile_time_public_setting_names(cls):
        return (
            *super().compile_time_public_setting_names(),
            cls.retain_count_image_setting,
            cls.retain_percent_image_setting,
        )


__all__ = [
    "AdjacentNeighborDistancePlanner",
    "DistanceMethod",
    "MeasureObjectNeighborsModule",
    "ExpandedNeighborDistancePlanner",
    "NeighborDistancePlan",
    "NeighborDistancePlanner",
    "NeighborMeasurements",
    "NeighborRetainedImageRequest",
    "NeighborTopologyArrays",
    "NeighborClosestArrays",
    "NeighborTopologyBackendStrategy",
    "NumbaNumpyNeighborTopologyBackendStrategy",
    "WithinNeighborDistancePlanner",
    "labels_or_default",
    "measure_object_neighbors",
    "neighbor_topology_backend",
    "require_matching_shape",
    "variant_numbers_for_final_labels",
]
