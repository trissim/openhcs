"""Neighbor-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import os
import time
from typing import TYPE_CHECKING, Annotated, ClassVar
import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.constants.constants import MemoryType
from openhcs.core.aligned_image_payload import AlignedImageStack
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    RelationshipsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.callable_contract import KeywordRuntimeParameter
from openhcs.core.memory import numpy
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
    MeasurementSparseColumnarRows,
)
from openhcs.core.pipeline.function_contracts import (
    runtime_bound_parameters,
    special_inputs,
    )
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_batch_contracts import SliceIndexRuntimeParameter
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
)
from openhcs.core.runtime_image_values import (
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
)
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.core.runtime_object_labels import (
    ObjectLabelVariantData,
    ObjectLabelPayload,
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.interop.cellprofiler.setting_names import (
    RepeatedSettingSequence,
    SettingNameFamily,
    optional_setting_value,
    setting_name_matches,
    setting_names,
    setting_values,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_setting_parser,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    FieldDerivedMeasurementFeatureModule,
    MeasurementFeatureRecord,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ModuleOwnedResultMeasurementRows,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument
from openhcs.interop.cellprofiler.runtime.primary_image_input_policies import (
    ObjectLabelDrivenPrimaryImageInputPolicy,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.parser import ModuleBlock

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
class NeighborMeasurements(MeasurementFeatureRecord):
    """Per-object neighbor measurements."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    object_id: Annotated[int, MeasurementRowAxisField.OBJECT_ID]
    scale: Annotated[int | str, MeasurementRowAxisField.SCALE]
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
    def for_method(cls, method: DistanceMethod) -> "NeighborDistancePlanner":
        if not isinstance(method, DistanceMethod):
            raise TypeError(
                "Neighbor distance planning requires a DistanceMethod, "
                f"got {type(method).__name__}."
            )
        return cls.for_enum_member(method)

    @abstractmethod
    def plan(self, labels: np.ndarray, neighbor_distance: int) -> NeighborDistancePlan:
        """Return working labels and neighborhood distance."""

    @abstractmethod
    def measurement_scale(self, neighbor_distance: int) -> int | str:
        """Return the exact feature qualifier for this distance policy."""


class AdjacentNeighborDistancePlanner(NeighborDistancePlanner):
    """Adjacent-neighbor topology without label expansion."""

    method = DistanceMethod.ADJACENT

    def plan(self, labels: np.ndarray, neighbor_distance: int) -> NeighborDistancePlan:
        return NeighborDistancePlan(
            labels.copy(),
            1,
            self.measurement_scale(neighbor_distance),
        )

    def measurement_scale(self, neighbor_distance: int) -> int | str:
        del neighbor_distance
        return "Adjacent"


class ExpandedNeighborDistancePlanner(NeighborDistancePlanner):
    """Expand labels until adjacent before measuring neighbors."""

    method = DistanceMethod.EXPAND

    def plan(self, labels: np.ndarray, neighbor_distance: int) -> NeighborDistancePlan:
        from scipy.ndimage import distance_transform_edt

        i, j = distance_transform_edt(
            labels == 0, return_distances=False, return_indices=True
        )
        return NeighborDistancePlan(
            labels[i, j],
            1,
            self.measurement_scale(neighbor_distance),
        )

    def measurement_scale(self, neighbor_distance: int) -> int | str:
        del neighbor_distance
        return "Expanded"


class WithinNeighborDistancePlanner(NeighborDistancePlanner):
    """Measure neighbors within a fixed distance."""

    method = DistanceMethod.WITHIN

    def plan(self, labels: np.ndarray, neighbor_distance: int) -> NeighborDistancePlan:
        return NeighborDistancePlan(
            labels.copy(),
            neighbor_distance,
            self.measurement_scale(neighbor_distance),
        )

    def measurement_scale(self, neighbor_distance: int) -> int | str:
        return int(neighbor_distance)


class _NeighborsAreSameObjectsRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound topology derived from the object input contract."""

    parameter_name = "neighbors_are_same_objects"
    annotation_type = bool
    parameter_default = True


class MeasureObjectNeighborsInputPolicy:
    """Bind MeasureObjectNeighbors object-label inputs."""

    @classmethod
    def primary_image_domain_input_binding(cls) -> SettingToKeywordBinding:
        """Use measured objects, not neighboring objects, as the image domain."""

        return cls.measured_objects_binding

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> dict[str, RuntimeCallableArgument]:
        same_objects = len(ArtifactSpecCollection(request.object_inputs).ref_set()) == 1
        bound = super().bind_runtime_inputs(request)
        bound[_NeighborsAreSameObjectsRuntimeParameter.require_parameter_name()] = (
            same_objects
        )
        return bound


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
        relationship: DirectedObjectRelationshipPayload,
        measurements: list[NeighborMeasurements],
        *,
        neighbor_count_image: np.ndarray,
        percent_touching_image: np.ndarray,
    ) -> tuple:
        retained = self.retained_images(
            image=image,
            neighbor_count_image=neighbor_count_image,
            percent_touching_image=percent_touching_image,
        )
        measurement_rows = DataclassMeasurementColumnarRows(
            tuple(measurements),
            row_type=NeighborMeasurements,
        )
        if len(retained) == 1:
            return (retained[0], relationship, measurement_rows)
        if len(retained) > 1:
            return (AlignedImageStack(retained), relationship, measurement_rows)
        return (image, relationship, measurement_rows)

    def retained_images(
        self,
        *,
        image: np.ndarray,
        neighbor_count_image: np.ndarray,
        percent_touching_image: np.ndarray,
    ) -> tuple[RuntimeArrayData, ...]:
        retained: list[RuntimeArrayData] = []
        output_metadata = image_payload_metadata(image).replace_fields(
            source_channel_axis=-1
        )
        if self.retain_neighbor_count_image:
            retained.append(
                with_image_payload_data(
                    image,
                    self.colored_metric_image(
                        neighbor_count_image, self.neighbor_count_colormap
                    ),
                    metadata=output_metadata,
                )
            )
        if self.retain_percent_touching_image:
            retained.append(
                with_image_payload_data(
                    image,
                    self.colored_metric_image(
                        percent_touching_image, self.percent_touching_colormap
                    ),
                    metadata=output_metadata,
                )
            )
        return tuple(retained)

    def colored_metric_image(
        self, metric_image: np.ndarray, colormap_name: str
    ) -> np.ndarray:
        """Color one object metric image using CellProfiler-style masked RGB output."""
        import matplotlib
        from matplotlib.cm import ScalarMappable

        cmap_name = str(colormap_name).strip() or "Default"
        if cmap_name.lower() == "default":
            cmap_name = "viridis"
        scalar_mappable = ScalarMappable(cmap=matplotlib.colormaps[cmap_name])
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
    """Per-variant-object neighbor topology measurements and directed edges."""

    neighbor_count: np.ndarray
    touching_pixel_count: np.ndarray
    source_variant_numbers: np.ndarray
    target_variant_numbers: np.ndarray

    def relationship_payload(
        self,
        object_numbers: np.ndarray,
        neighbor_numbers: np.ndarray,
        *,
        slice_index: int,
    ) -> DirectedObjectRelationshipPayload:
        """Map variant-label adjacency back to final object numbering."""

        source_variant_count = max(
            int(self.source_variant_numbers.max())
            if self.source_variant_numbers.size
            else 0,
            int(object_numbers.max()) if object_numbers.size else 0,
        )
        target_variant_count = max(
            int(self.target_variant_numbers.max())
            if self.target_variant_numbers.size
            else 0,
            int(neighbor_numbers.max()) if neighbor_numbers.size else 0,
        )
        source_by_variant = np.zeros(source_variant_count + 1, dtype=np.int64)
        target_by_variant = np.zeros(target_variant_count + 1, dtype=np.int64)
        for final_number, variant_number in enumerate(object_numbers, start=1):
            if 0 < variant_number < source_by_variant.size:
                source_by_variant[int(variant_number)] = final_number
        for final_number, variant_number in enumerate(neighbor_numbers, start=1):
            if 0 < variant_number < target_by_variant.size:
                target_by_variant[int(variant_number)] = final_number
        source_ids = source_by_variant[self.source_variant_numbers]
        target_ids = target_by_variant[self.target_variant_numbers]
        valid_pairs = (source_ids > 0) & (target_ids > 0)
        return DirectedObjectRelationshipPayload(
            source_ids=tuple(int(source_id) for source_id in source_ids[valid_pairs]),
            target_ids=tuple(int(target_id) for target_id in target_ids[valid_pairs]),
            slice_indices=(int(slice_index),) * int(np.count_nonzero(valid_pairs)),
            slice_count=int(slice_index) + 1,
        )


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
        """Return neighbor counts, touching-pixel counts, and directed edges."""

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
        if any(
            array.ndim != 2 for array in (working_array, neighbor_array, outline_array)
        ):
            raise NotImplementedError(
                "CellProfiler neighbor topology requires projected 2-D label planes."
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
        (
            neighbor_count,
            touching_pixel_count,
            source_variant_numbers,
            target_variant_numbers,
        ) = (
            _measure_neighbor_topology_numba(
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
        )
        return NeighborTopologyArrays(
            neighbor_count=neighbor_count,
            touching_pixel_count=touching_pixel_count,
            source_variant_numbers=source_variant_numbers,
            target_variant_numbers=target_variant_numbers,
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
@special_inputs("labels", "neighbor_labels")
@runtime_bound_parameters(
    _NeighborsAreSameObjectsRuntimeParameter,
    SliceIndexRuntimeParameter,
)
def measure_object_neighbors(
    image: np.ndarray,
    labels: ObjectLabelValue,
    neighbor_labels: ObjectLabelValue | None = None,
    neighbors_are_same_objects: bool = True,
    *,
    distance_method: DistanceMethod,
    neighbor_distance: int,
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
) -> tuple[
    np.ndarray | AlignedImageStack,
    DirectedObjectRelationshipPayload,
    DataclassMeasurementColumnarRows,
]:
    """Measure neighbor relationships between objects.

    Args:
        labels: Primary object-label plane whose objects receive neighbor
            measurements.
        neighbor_labels: Optional second object-label plane used as the neighbor
            population; leave unset when objects should be compared with their
            own set.
        neighbor_count_colormap: Colormap for the retained neighbor-count image;
            used only when ``retain_neighbor_count_image`` is enabled.
        percent_touching_colormap: Colormap for the retained percent-touching
            image; used only when ``retain_percent_touching_image`` is enabled.
    """
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
    distance_planner = NeighborDistancePlanner.for_method(distance_method)
    slice_index = 0 if slice_index is None else int(slice_index)
    relationship = DirectedObjectRelationshipPayload(
        source_ids=(),
        target_ids=(),
        slice_count=slice_index + 1,
    )
    image_array = np.asarray(image)
    final_labels = object_label_dense_array(labels, dtype=np.int32)
    if final_labels.ndim != 2 or image_array.ndim != 2:
        raise ValueError(
            "MeasureObjectNeighbors requires image and object labels already "
            f"projected to one 2-D plane, got image {image_array.shape!r} and "
            f"labels {final_labels.shape!r}."
        )
    if final_labels.shape != image_array.shape:
        raise ValueError(
            "MeasureObjectNeighbors image and projected labels must share a "
            f"shape; got image {image_array.shape!r} and labels "
            f"{final_labels.shape!r}."
        )
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
        else object_label_dense_array(neighbor_labels, dtype=np.int32)
    )
    measured_variant_labels = object_label_dense_array(
        labels.small_removed_labels
        if labels.small_removed_labels is not None
        else labels,
        dtype=np.int32,
    )
    neighbor_payload = labels if neighbor_labels is None else neighbor_labels
    neighbor_variant_labels = object_label_dense_array(
        neighbor_payload.small_removed_labels
        if neighbor_payload.small_removed_labels is not None
        else neighbor_payload,
        dtype=np.int32,
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
            relationship,
            [],
            neighbor_count_image=empty_metric_image,
            percent_touching_image=empty_metric_image,
        )
    measured_topology_labels = measured_variant_labels
    neighbor_topology_labels = neighbor_variant_labels
    if not consider_discarded_objects:
        neighbor_topology_labels = neighbor_variant_labels.copy()
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
        else int(neighbor_final_labels.max())
        if neighbor_final_labels.size
        else 0
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
                scale=distance_planner.measurement_scale(neighbor_distance),
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
            relationship,
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
    distance_plan = distance_planner.plan(measured_topology_labels, neighbor_distance)
    profile_mark = _profile_elapsed(
        "measure_object_neighbors.distance_plan",
        profile_mark,
        method=distance_method.value,
        variant_object_count=variant_object_count,
        variant_neighbor_count=variant_neighbor_count,
    )
    working_labels = distance_plan.working_labels
    distance = distance_plan.distance
    measurement_scale = distance_plan.measurement_scale
    neighbor_working_labels = (
        working_labels
        if neighbors_are_same_objects and distance_method is DistanceMethod.EXPAND
        else neighbor_topology_labels
    )
    if variant_neighbor_count > (1 if neighbors_are_same_objects else 0):
        relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type(
            backend_provider=relationship_backend_provider
        )
        ocenters = relationship_backend.label_centers(measured_variant_labels)
        ncenters = (
            ocenters
            if neighbors_are_same_objects
            else relationship_backend.label_centers(neighbor_variant_labels)
        )
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
        relationship = topology.relationship_payload(
            object_numbers,
            neighbor_numbers,
            slice_index=slice_index,
        )
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
        relationship,
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
        ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = working_labels.shape
    adjacency_words = np.zeros(
        (variant_object_count, (variant_neighbor_count >> 6) + 1),
        dtype=np.uint64,
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
                word_index = neighbor_number >> 6
                bit_index = neighbor_number & 63
                adjacency_words[object_index, word_index] |= np.uint64(1) << np.uint64(
                    bit_index
                )
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
    relationship_count = 0
    for object_index in range(variant_object_count):
        count = 0.0
        for word_index in range(adjacency_words.shape[1]):
            remaining = adjacency_words[object_index, word_index]
            while remaining != 0:
                count += 1.0
                remaining &= remaining - np.uint64(1)
        neighbor_count[object_index] = count
        relationship_count += int(count)
    source_variant_numbers = np.empty(relationship_count, dtype=np.int32)
    target_variant_numbers = np.empty(relationship_count, dtype=np.int32)
    relationship_index = 0
    for object_index in range(variant_object_count):
        for word_index in range(adjacency_words.shape[1]):
            remaining = adjacency_words[object_index, word_index]
            bit_index = 0
            while remaining != 0:
                if remaining & np.uint64(1):
                    target_number = (word_index << 6) + bit_index
                    if 0 < target_number <= variant_neighbor_count:
                        source_variant_numbers[relationship_index] = object_index + 1
                        target_variant_numbers[relationship_index] = target_number
                        relationship_index += 1
                remaining >>= np.uint64(1)
                bit_index += 1
    return (
        neighbor_count,
        touching_pixel_count,
        source_variant_numbers,
        target_variant_numbers,
    )


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
    identity_final_mapping = (
        neighbors_are_same_objects
        and final_object_count == variant_object_count
        and final_object_count == variant_neighbor_count
        and object_numbers.size == final_object_count
        and neighbor_numbers.size == final_object_count
        and final_has_pixels.size >= final_object_count
        and neighbor_has_pixels.size >= final_object_count
    )
    if identity_final_mapping:
        for final_object_index in range(final_object_count):
            if (
                object_numbers[final_object_index] != final_object_index + 1
                or neighbor_numbers[final_object_index] != final_object_index + 1
                or not final_has_pixels[final_object_index]
                or not neighbor_has_pixels[final_object_index]
            ):
                identity_final_mapping = False
                break
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
        if identity_final_mapping and object_index < final_object_count:
            if first_neighbor >= 0:
                final_first_object_number[object_index] = first_neighbor + 1
            if second_neighbor >= 0:
                final_second_object_number[object_index] = second_neighbor + 1
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
        if identity_final_mapping:
            continue
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


class MeasureObjectNeighborsMeasurementRecordRowsMixin(
    FieldDerivedMeasurementFeatureModule
):
    """Project nominal neighbor records to CellProfiler feature identities."""

    measurement_feature_family = "Neighbors"

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        measurement_scale: int | str

        @classmethod
        def for_request(cls, module_type, request):
            planner = NeighborDistancePlanner.for_method(
                request.call_kwargs["distance_method"]
            )
            return cls(
                request.output_value,
                module_type=module_type,
                measurement_scale=planner.measurement_scale(
                    int(request.call_kwargs["neighbor_distance"])
                ),
            )

        def rows(self) -> MeasurementSparseColumnarRows:
            source_rows = self.source_rows()
            source_fields = {
                field_spec.name: field_spec for field_spec in source_rows.fields
            }
            axis_names = MeasurementRowAxisField.field_names()
            feature_fields = tuple(
                field_spec
                for field_spec in source_rows.fields
                if field_spec.name not in axis_names
            )
            projected_feature_names = {
                field_spec.name: self.module_type.measurement_feature_name(
                    field_spec.name,
                    self.measurement_scale,
                )
                for field_spec in feature_fields
            }
            projected_rows: list[dict[str, object]] = []
            projected_fields = (
                source_fields[MeasurementRowAxisField.SLICE_INDEX.value],
                source_fields[MeasurementRowAxisField.OBJECT_ID.value],
                *(
                    FieldSpec(
                        projected_feature_names[field_spec.name],
                        field_spec.dtype,
                        required=False,
                    )
                    for field_spec in feature_fields
                ),
            )
            for source_row in source_rows.iter_row_mappings():
                scale = source_row[MeasurementRowAxisField.SCALE.value]
                if scale != self.measurement_scale:
                    raise ValueError(
                        "MeasureObjectNeighbors row scale does not match its "
                        f"invocation contract: {scale!r} != "
                        f"{self.measurement_scale!r}."
                    )
                projected_row = {
                    MeasurementRowAxisField.SLICE_INDEX.value: source_row[
                        MeasurementRowAxisField.SLICE_INDEX.value
                    ],
                    MeasurementRowAxisField.OBJECT_ID.value: source_row[
                        MeasurementRowAxisField.OBJECT_ID.value
                    ],
                }
                for field_spec in feature_fields:
                    feature_name = projected_feature_names[field_spec.name]
                    projected_row[feature_name] = source_row[field_spec.name]
                projected_rows.append(projected_row)
            return MeasurementSparseColumnarRows.from_rows(
                projected_rows,
                fields=projected_fields,
                declared_object_measurement_domain_covered=True,
                object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
            )


class MeasureObjectNeighborsModule(
    ObjectLabelDrivenPrimaryImageInputPolicy,
    MeasureObjectNeighborsInputPolicy,
    MeasureObjectNeighborsMeasurementRecordRowsMixin,
    ObjectArtifactInputModule,
    MeasurementArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "MeasureObjectNeighbors"
    function_name = "measure_object_neighbors"
    validated = True
    confidence = 1.0
    measurement_category_prefixes = (("neighbors",),)
    relationship_type = "Neighbors"
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
    distance_method_setting = "Method to determine neighbors"
    neighbor_distance_setting = "Neighbor distance"
    consider_discarded_objects_setting = (
        "Consider objects discarded for touching image border?"
    )
    output_image_setting = SettingNameFamily("Name the output image")
    colormap_setting = "Select colormap"
    measured_objects_binding = SettingToKeywordBinding.input(
        measured_objects_setting, ObjectLabelsArtifactType, runtime_parameter_name="labels"
    )
    neighbor_objects_binding = SettingToKeywordBinding.input(
        neighbor_objects_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name="neighbor_labels",
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting, ImageArtifactType
    )
    retain_count_image_binding = SettingToKeywordBinding(
        retain_count_image_setting,
        "retain_neighbor_count_image",
        parse_cellprofiler_bool,
    )
    retain_percent_image_binding = SettingToKeywordBinding(
        retain_percent_image_setting,
        "retain_percent_touching_image",
        parse_cellprofiler_bool,
    )
    retained_image_bindings = (
        retain_count_image_binding,
        retain_percent_image_binding,
    )
    setting_bindings = (
        measured_objects_binding,
        neighbor_objects_binding,
        output_image_binding,
        SettingToKeywordBinding(
            distance_method_setting,
            "distance_method",
            cellprofiler_enum_setting_parser(DistanceMethod),
        ),
        SettingToKeywordBinding(
            neighbor_distance_setting,
            "neighbor_distance",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            consider_discarded_objects_setting,
            "consider_discarded_objects",
            parse_cellprofiler_bool,
        ),
        *retained_image_bindings,
    )

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        retains_image = any(
            cls._module_flag(module, binding.setting_name)
            for binding in cls.retained_image_bindings
        )
        return tuple(
            binding
            for binding in bindings
            if retains_image or binding is not cls.output_image_binding
        )

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: "BoundModuleSettings"
    ) -> "BoundModuleSettings":
        colormaps = module.get_setting_values(cls.colormap_setting)
        colormap_values = RepeatedSettingSequence(colormaps, default="Default")
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        for concrete_name in setting_names(cls.colormap_setting):
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(concrete_name), None
            )
        output_names = setting_values(module, cls.output_image_setting)
        kwargs = {
            **dict(bound.kwargs),
            "neighbor_count_colormap": colormap_values.at(0),
            "percent_touching_colormap": colormap_values.at(1),
        }
        if output_names:
            kwargs[cls.output_image_binding.require_parameter_name()] = (
                output_names[0] if len(output_names) == 1 else output_names
            )
        return BoundModuleSettings(
            kwargs,
            unmapped_kwargs,
            bound.setting_coverage,
        )

    @classmethod
    def _artifact_input_record_groups(
        cls,
        *,
        module,
        invocation_key,
        step_context,
    ):
        """Default an omitted neighbor role to the measured-object identity."""

        if cls.artifact_names_for_binding(module, cls.neighbor_objects_binding):
            return super()._artifact_input_record_groups(
                module=module,
                invocation_key=invocation_key,
                step_context=step_context,
            )

        from openhcs.interop.cellprofiler.parser import ModuleSetting

        measured_names = cls.artifact_names_for_binding(
            module,
            cls.measured_objects_binding,
        )
        if measured_names:
            measured_groups = ((),)
        else:
            measured_groups = cls._artifact_input_record_groups_for_bindings(
                module=module,
                invocation_key=invocation_key,
                bindings=(cls.measured_objects_binding,),
                step_context=step_context,
            )
        result = []
        for records in measured_groups:
            names = measured_names or tuple(
                record.value
                for record in records
                if setting_name_matches(
                    record.name,
                    cls.measured_objects_binding.setting_name,
                )
            )
            if len(names) != 1:
                raise ValueError(
                    "MeasureObjectNeighbors requires one measured-object identity "
                    f"before deriving its neighboring-object role, got {names!r}."
                )
            result.append(
                (
                    *records,
                    ModuleSetting(
                        setting_names(cls.neighbor_objects_binding.setting_name)[0],
                        names[0],
                    ),
                )
            )
        return tuple(result)

    @classmethod
    def _derived_identity_setting_records(
        cls,
        *,
        invocation,
        block_position,
        existing_records,
        step_context,
    ):
        """Derive both fixed retained-image slots when either is active."""

        from openhcs.interop.cellprofiler.parser import ModuleSetting

        del invocation
        records: list[ModuleSetting] = []
        if not cls._record_values(existing_records, cls.neighbor_objects_setting):
            measured_name = cls._record_value(
                existing_records,
                cls.measured_objects_setting,
            )
            if measured_name is None:
                raise ValueError(
                    "MeasureObjectNeighbors cannot derive its neighboring-object "
                    "identity without a measured-object input."
                )
            records.append(
                ModuleSetting(
                    cls.neighbor_objects_setting.canonical,
                    measured_name,
                )
            )
        if cls._record_values(existing_records, cls.output_image_setting):
            return tuple(records)
        retained = tuple(
            cls._record_flag(existing_records, binding.setting_name)
            for binding in cls.retained_image_bindings
        )
        if not any(retained):
            return tuple(records)
        records.extend(
            ModuleSetting(
                cls.output_image_setting.canonical,
                cls.canonical_output_artifact_name(
                    artifact_type=ImageArtifactType,
                    output_position=output_position,
                    block_position=block_position,
                    step_context=step_context,
                ),
            )
            for output_position in range(len(cls.retained_image_bindings))
        )
        return tuple(records)

    @classmethod
    def _record_values(cls, records, setting_name) -> tuple[str, ...]:
        from openhcs.interop.cellprofiler.setting_names import setting_name_matches

        return tuple(
            record.value
            for record in records
            if setting_name_matches(record.name, setting_name)
        )

    @classmethod
    def _record_flag(cls, records, setting_name) -> bool:
        value = cls._record_value(records, setting_name)
        return False if value is None else parse_cellprofiler_bool(value)

    @classmethod
    def _record_value(cls, records, setting_name) -> str | None:
        values = cls._record_values(records, setting_name)
        if len(values) > 1:
            raise ValueError(
                f"Expected one {cls.module_name} setting row for "
                f"{setting_name!r}, got {values!r}."
            )
        return values[0] if values else None

    @classmethod
    def _module_flag(cls, module: "ModuleBlock", setting_name) -> bool:
        value = optional_setting_value(module, setting_name)
        return False if value is None else parse_cellprofiler_bool(value)

    @classmethod
    def artifact_contract_outputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        output_names = setting_values(module, cls.output_image_setting)
        retained = tuple(
            cls._module_flag(module, binding.setting_name)
            for binding in cls.retained_image_bindings
        )
        if any(retained) and len(output_names) != len(cls.retained_image_bindings):
            raise ValueError(
                f"MeasureObjectNeighbors({module.module_num}) requires "
                f"{len(cls.retained_image_bindings)} fixed "
                "retained-image names whenever a retained image is enabled, got "
                f"{output_names!r}."
            )
        measured_names = setting_values(module, cls.measured_objects_setting)
        if len(measured_names) != 1:
            raise ValueError(
                f"MeasureObjectNeighbors({module.module_num}) requires exactly one "
                f"measured object set, got {measured_names!r}."
            )
        measured_objects = artifact_inputs.require_by_name_and_artifact_type(
            measured_names[0],
            ObjectLabelsArtifactType,
        )
        neighboring_names = setting_values(module, cls.neighbor_objects_setting)
        neighboring_name = (
            measured_names[0] if not neighboring_names else neighboring_names[0]
        )
        neighboring_objects = artifact_inputs.require_by_name_and_artifact_type(
            neighboring_name,
            ObjectLabelsArtifactType,
        )
        outputs = [
            ArtifactSpec.output_preserving_source_stack_scope(
                output_names[output_position],
                ImageArtifactType,
                measured_objects,
            )
            for output_position, is_retained in enumerate(retained)
            if is_retained
        ]
        relationship_declaration = ObjectRelationshipDeclaration(
            source=measured_objects.ref(),
            target=neighboring_objects.ref(),
            relationship_type=cls.relationship_type,
            source_role="neighbor_source",
            target_role="neighbor_target",
            source_id_field="object_number1",
            target_id_field="object_number2",
            producer_module_number=module.module_num,
        )
        outputs.append(
            ArtifactSpec.output(
                relationship_declaration.artifact_name(),
                RelationshipsArtifactType,
                relations=(
                    SourceStackLineageSourceRelation(source=measured_objects.ref()),
                    relationship_declaration,
                ),
            )
        )
        outputs.append(
            cls.measurement_output_artifact(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            )
        )
        return tuple(outputs)


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
    "measure_object_neighbors",
    "neighbor_topology_backend",
    "require_matching_shape",
    "variant_numbers_for_final_labels",
]
