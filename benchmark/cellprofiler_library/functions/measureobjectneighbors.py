"""
Converted from CellProfiler: MeasureObjectNeighbors
Original: MeasureObjectNeighbors.run

Measures neighbor relationships between objects including:
- Number of neighbors
- Percent of boundary touching neighbors
- First and second closest object distances
- Angle between neighbors
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import ClassVar, Tuple
from dataclasses import dataclass
from enum import Enum
from metaclass_registry import AutoRegisterMeta
from openhcs.core.memory import numpy
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.backends.cellprofiler.morphology import MorphologyBackendStrategy
from openhcs.processing.backends.cellprofiler.neighbors import (
    neighbor_topology_backend,
)
from openhcs.processing.backends.cellprofiler.outlines import object_outline_backend
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)


class DistanceMethod(Enum):
    ADJACENT = "adjacent"
    EXPAND = "expand"
    WITHIN = "within"


_CELLPROFILER_DISTANCE_METHODS = {
    "adjacent": DistanceMethod.ADJACENT,
    "expand": DistanceMethod.EXPAND,
    "expand_until_adjacent": DistanceMethod.EXPAND,
    "within": DistanceMethod.WITHIN,
    "within_a_specified_distance": DistanceMethod.WITHIN,
}


def _coerce_distance_method(value: DistanceMethod | str) -> DistanceMethod:
    if isinstance(value, DistanceMethod):
        return value
    normalized = "_".join(str(value).strip().lower().replace("-", " ").split())
    try:
        return _CELLPROFILER_DISTANCE_METHODS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported neighbor distance method {value!r}.") from exc


@dataclass
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


@dataclass(frozen=True)
class NeighborDistancePlan:
    working_labels: np.ndarray
    distance: int
    measurement_scale: int | str


class NeighborDistancePlanner(ABC, metaclass=AutoRegisterMeta):
    """Prepare neighbor-distance state for one closed distance method."""

    __registry_key__ = "method_label"
    method_label: ClassVar[str | None] = None
    method: ClassVar[DistanceMethod | None] = None

    @classmethod
    def for_method(cls, method: DistanceMethod) -> "NeighborDistancePlanner":
        return cls.__registry__[method.value]()

    @abstractmethod
    def plan(
        self,
        labels: np.ndarray,
        neighbor_distance: int,
    ) -> NeighborDistancePlan:
        """Return working labels and neighborhood distance."""


class AdjacentNeighborDistancePlanner(NeighborDistancePlanner):
    method = DistanceMethod.ADJACENT
    method_label = method.value

    def plan(
        self,
        labels: np.ndarray,
        neighbor_distance: int,
    ) -> NeighborDistancePlan:
        del neighbor_distance
        return NeighborDistancePlan(labels.copy(), 1, "Adjacent")


class ExpandedNeighborDistancePlanner(NeighborDistancePlanner):
    method = DistanceMethod.EXPAND
    method_label = method.value

    def plan(
        self,
        labels: np.ndarray,
        neighbor_distance: int,
    ) -> NeighborDistancePlan:
        del neighbor_distance
        from scipy.ndimage import distance_transform_edt

        i, j = distance_transform_edt(
            labels == 0,
            return_distances=False,
            return_indices=True,
        )
        return NeighborDistancePlan(labels[i, j], 1, "Expanded")


class WithinNeighborDistancePlanner(NeighborDistancePlanner):
    method = DistanceMethod.WITHIN
    method_label = method.value

    def plan(
        self,
        labels: np.ndarray,
        neighbor_distance: int,
    ) -> NeighborDistancePlan:
        return NeighborDistancePlan(
            labels.copy(),
            neighbor_distance,
            int(neighbor_distance),
        )


def _strel_disk(
    radius: float,
    *,
    morphology_backend_provider: CellProfilerBackendProvider | None,
) -> np.ndarray:
    """Create the CellProfiler disk-shaped structuring element."""
    return MorphologyBackendStrategy.for_memory_type(
        backend_provider=morphology_backend_provider,
    ).disk_footprint(radius)


def _centers_of_labels(
    labels: np.ndarray,
    *,
    relationship_backend_provider: CellProfilerBackendProvider | None,
) -> np.ndarray:
    """Calculate centers of mass for each labeled object."""
    return ObjectRelationshipBackendStrategy.for_memory_type(
        backend_provider=relationship_backend_provider,
    ).label_centers(labels)


def _variant_numbers_for_final_labels(
    final_labels: np.ndarray,
    variant_labels: np.ndarray,
) -> np.ndarray:
    """Map each final object ID to the corresponding variant object ID."""
    final_count = int(np.max(final_labels)) if final_labels.size else 0
    numbers = np.zeros(final_count, dtype=np.int32)
    for final_id in range(1, final_count + 1):
        overlap = variant_labels[final_labels == final_id]
        overlap = overlap[overlap > 0]
        if overlap.size == 0:
            continue
        counts = np.bincount(overlap.astype(np.int32, copy=False))
        numbers[final_id - 1] = int(np.argmax(counts))
    return numbers


def _labels_or_default(
    labels: np.ndarray | None,
    default: np.ndarray,
) -> np.ndarray:
    """Return a semantic label variant or the final labels when absent."""
    return default if labels is None else labels


def _require_matching_shape(
    labels: np.ndarray,
    variant: np.ndarray,
    variant_name: str,
) -> None:
    if labels.shape != variant.shape:
        raise ValueError(
            f"{variant_name} shape {variant.shape!r} does not match final "
            f"labels shape {labels.shape!r}."
        )


def _outline(
    labels: np.ndarray,
    *,
    outline_backend_provider: CellProfilerBackendProvider | None,
) -> np.ndarray:
    """Create CellProfiler-style labeled object outlines."""
    return object_outline_backend(
        backend_provider=outline_backend_provider,
    ).outline(labels)


def _neighbor_output(
    image: np.ndarray,
    measurements: list,
    *,
    retained_images: tuple[np.ndarray, ...],
) -> tuple:
    """Return retained images first so artifact contracts map them by kind."""
    if retained_images:
        return (*retained_images, measurements)
    return image, measurements


def _retained_neighbor_images(
    labels: np.ndarray,
    neighbor_count_image: np.ndarray,
    percent_touching_image: np.ndarray,
    *,
    retain_neighbor_count_image: bool,
    neighbor_count_colormap: str,
    retain_percent_touching_image: bool,
    percent_touching_colormap: str,
) -> tuple[np.ndarray, ...]:
    retained: list[np.ndarray] = []
    if retain_neighbor_count_image:
        retained.append(
            _colored_object_metric_image(
                labels,
                neighbor_count_image,
                neighbor_count_colormap,
            )
        )
    if retain_percent_touching_image:
        retained.append(
            _colored_object_metric_image(
                labels,
                percent_touching_image,
                percent_touching_colormap,
            )
        )
    return tuple(retained)


def _colored_object_metric_image(
    labels: np.ndarray,
    metric_image: np.ndarray,
    colormap_name: str,
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
    rgb[labels <= 0] = 0
    return rgb


@numpy
def measure_object_neighbors(
    image: np.ndarray,
    labels: np.ndarray,
    neighbor_labels: np.ndarray | None = None,
    small_removed_labels: np.ndarray | None = None,
    small_removed_neighbor_labels: np.ndarray | None = None,
    distance_method: DistanceMethod | str = DistanceMethod.EXPAND,
    neighbor_distance: int = 5,
    neighbors_are_same_objects: bool = True,
    consider_discarded_objects: bool = True,
    retain_neighbor_count_image: bool = False,
    neighbor_count_colormap: str = "Default",
    retain_percent_touching_image: bool = False,
    percent_touching_colormap: str = "Default",
    neighbor_topology_backend_provider: CellProfilerBackendProvider | None = None,
    outline_backend_provider: CellProfilerBackendProvider | None = None,
    morphology_backend_provider: CellProfilerBackendProvider | None = None,
    relationship_backend_provider: CellProfilerBackendProvider | None = None,
) -> Tuple[np.ndarray, list]:
    """
    Measure neighbor relationships between objects.

    CellProfiler Parameter Mapping:
    'Select objects to measure' -> (pipeline-handled)
    'Select neighboring objects to measure' -> (pipeline-handled)
    'Method to determine neighbors' -> distance_method
    'Neighbor distance' -> neighbor_distance
    'Consider objects discarded for touching image border?' -> consider_discarded_objects
    'Retain the image of objects colored by numbers of neighbors?' -> retain_neighbor_count_image
    'Retain the image of objects colored by percent of touching pixels?' -> retain_percent_touching_image
    'Name the output image' -> (pipeline-handled)
    'Select colormap' -> [neighbor_count_colormap, percent_touching_colormap]
    
    Args:
        image: Input image (H, W)
        labels: Final label image with segmented measured objects (H, W)
        neighbor_labels: Final labels for the neighboring object set. Defaults
            to labels when measuring neighbors within the same object set.
        small_removed_labels: Optional measured-object label variant retaining
            objects discarded from the final label set.
        small_removed_neighbor_labels: Optional neighboring-object label variant
            retaining objects discarded from the final neighbor label set.
        distance_method: Method to determine neighbors:
            - ADJACENT: Objects must have adjacent boundary pixels
            - EXPAND: Expand objects until all boundaries touch
            - WITHIN: Expand by specified distance
        neighbor_distance: Distance for WITHIN method
        neighbors_are_same_objects: If True, measure neighbors within same object set
        consider_discarded_objects: If True, allow the small-removed variants to
            contribute to neighbor topology.
    
    Returns:
        Tuple of (image, list of NeighborMeasurements)
    """
    labels = labels.astype(np.int32, copy=False)
    final_labels = labels
    neighbor_final_labels = (
        final_labels
        if neighbor_labels is None
        else neighbor_labels.astype(np.int32, copy=False)
    )
    measured_variant_labels = _labels_or_default(
        small_removed_labels,
        final_labels,
    ).astype(np.int32, copy=False)
    neighbor_variant_labels = (
        measured_variant_labels
        if neighbors_are_same_objects and small_removed_neighbor_labels is None
        else _labels_or_default(
            small_removed_neighbor_labels,
            neighbor_final_labels,
        ).astype(np.int32, copy=False)
    )

    _require_matching_shape(final_labels, measured_variant_labels, "small_removed_labels")
    _require_matching_shape(
        neighbor_final_labels,
        neighbor_variant_labels,
        "small_removed_neighbor_labels",
    )

    final_object_count = int(final_labels.max()) if final_labels.size else 0
    
    if final_object_count == 0:
        return _neighbor_output(
            image,
            [],
            retained_images=_retained_neighbor_images(
                final_labels,
                np.zeros_like(final_labels, dtype=float),
                np.zeros_like(final_labels, dtype=float),
                retain_neighbor_count_image=retain_neighbor_count_image,
                neighbor_count_colormap=neighbor_count_colormap,
                retain_percent_touching_image=retain_percent_touching_image,
                percent_touching_colormap=percent_touching_colormap,
            ),
        )

    measured_topology_labels = measured_variant_labels
    neighbor_topology_labels = neighbor_variant_labels.copy()
    if not consider_discarded_objects:
        neighbor_topology_labels[neighbor_final_labels <= 0] = 0

    object_numbers = _variant_numbers_for_final_labels(
        final_labels,
        measured_variant_labels,
    )
    neighbor_numbers = (
        object_numbers
        if neighbors_are_same_objects
        else _variant_numbers_for_final_labels(
            neighbor_final_labels,
            neighbor_topology_labels,
        )
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
            neighbor_final_labels.ravel(),
            minlength=neighbor_final_count + 1,
        )[1:] > 0
    )

    variant_object_count = (
        int(measured_topology_labels.max()) if measured_topology_labels.size else 0
    )
    variant_neighbor_count = (
        int(neighbor_topology_labels.max()) if neighbor_topology_labels.size else 0
    )
    if variant_object_count == 0 or variant_neighbor_count == 0:
        measurements = []
        for i in range(final_object_count):
            measurements.append(NeighborMeasurements(
                slice_index=0,
                object_id=i + 1,
                scale=_coerce_distance_method(distance_method).value,
                number_of_neighbors=0,
                percent_touching=0.0,
                first_closest_object_number=0,
                first_closest_distance=0.0,
                second_closest_object_number=0,
                second_closest_distance=0.0,
                angle_between_neighbors=0.0,
            ))
        return _neighbor_output(
            image,
            measurements,
            retained_images=_retained_neighbor_images(
                final_labels,
                np.zeros_like(final_labels, dtype=float),
                np.zeros_like(final_labels, dtype=float),
                retain_neighbor_count_image=retain_neighbor_count_image,
                neighbor_count_colormap=neighbor_count_colormap,
                retain_percent_touching_image=retain_percent_touching_image,
                percent_touching_colormap=percent_touching_colormap,
            ),
        )
    
    # Initialize measurement arrays
    neighbor_count = np.zeros(variant_object_count)
    pixel_count = np.zeros(variant_object_count)
    first_x_vector = np.zeros(variant_object_count)
    second_x_vector = np.zeros(variant_object_count)
    first_y_vector = np.zeros(variant_object_count)
    second_y_vector = np.zeros(variant_object_count)
    angle = np.zeros(variant_object_count)
    percent_touching = np.zeros(variant_object_count)
    final_first_object_number = np.zeros(final_object_count, dtype=int)
    final_second_object_number = np.zeros(final_object_count, dtype=int)
    
    normalized_distance_method = _coerce_distance_method(distance_method)
    distance_plan = NeighborDistancePlanner.for_method(
        normalized_distance_method
    ).plan(
        measured_topology_labels,
        neighbor_distance,
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
        # Calculate object centers
        ocenters = _centers_of_labels(
            measured_variant_labels,
            relationship_backend_provider=relationship_backend_provider,
        )
        ncenters = _centers_of_labels(
            neighbor_variant_labels,
            relationship_backend_provider=relationship_backend_provider,
        )
        
        # Calculate perimeters
        object_indexes = np.arange(variant_object_count) + 1
        perimeter_outlines = _outline(
            working_labels,
            outline_backend_provider=outline_backend_provider,
        )
        perimeters = np.array([np.sum(perimeter_outlines == i) for i in object_indexes])
        perimeters = np.maximum(perimeters, 1)  # Avoid division by zero
        
        # Find nearest neighbors using variant-label center distances.
        if variant_neighbor_count >= (2 if neighbors_are_same_objects else 1):
            for i in range(variant_object_count):
                if i >= len(ocenters) or not np.all(np.isfinite(ocenters[i])):
                    continue
                usable_ncenters = ncenters[:variant_neighbor_count]
                distances = np.sqrt(
                    (ocenters[i, 0] - usable_ncenters[:, 0])**2 +
                    (ocenters[i, 1] - usable_ncenters[:, 1])**2
                )
                if neighbors_are_same_objects and i < len(distances):
                    distances[i] = np.inf
                
                sorted_idx = np.argsort(distances)
                sorted_idx = sorted_idx[np.isfinite(distances[sorted_idx])]

                if len(sorted_idx) > 0:
                    first_idx = sorted_idx[0]
                    first_x_vector[i] = usable_ncenters[first_idx, 1] - ocenters[i, 1]
                    first_y_vector[i] = usable_ncenters[first_idx, 0] - ocenters[i, 0]
                
                if len(sorted_idx) > 1:
                    second_idx = sorted_idx[1]
                    second_x_vector[i] = usable_ncenters[second_idx, 1] - ocenters[i, 1]
                    second_y_vector[i] = usable_ncenters[second_idx, 0] - ocenters[i, 0]
        
        # Calculate angles between neighbors
        for i in range(variant_object_count):
            v1 = np.array([first_x_vector[i], first_y_vector[i]])
            v2 = np.array([second_x_vector[i], second_y_vector[i]])
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                dot = np.dot(v1, v2) / (norm1 * norm2)
                dot = np.clip(dot, -1, 1)
                angle[i] = np.arccos(dot) * 180.0 / np.pi
        
        strel = _strel_disk(
            distance,
            morphology_backend_provider=morphology_backend_provider,
        )
        strel_touching = _strel_disk(
            distance + 0.5,
            morphology_backend_provider=morphology_backend_provider,
        )

        topology = neighbor_topology_backend(
            backend_provider=neighbor_topology_backend_provider,
        ).measure_topology(
            working_labels,
            neighbor_working_labels,
            perimeter_outlines,
            object_numbers,
            distance=distance,
            neighbors_are_same_objects=neighbors_are_same_objects,
            footprint=strel,
            touching_footprint=strel_touching,
            variant_object_count=variant_object_count,
            variant_neighbor_count=variant_neighbor_count,
        )
        neighbor_count = topology.neighbor_count
        pixel_count = topology.touching_pixel_count
        percent_touching = pixel_count * 100 / perimeters

        object_variant_indexes = object_numbers - 1
        neighbor_variant_indexes = neighbor_numbers - 1
        valid_object_indexes = object_variant_indexes >= 0
        valid_neighbor_indexes = neighbor_variant_indexes >= 0
        if np.any(valid_object_indexes) and np.any(valid_neighbor_indexes):
            object_rows = object_variant_indexes[valid_object_indexes]
            neighbor_rows = neighbor_variant_indexes[valid_neighbor_indexes]
            distance_matrix = np.sqrt(
                (
                    ocenters[object_rows[:, np.newaxis], 0]
                    - ncenters[neighbor_rows[np.newaxis, :], 0]
                ) ** 2
                + (
                    ocenters[object_rows[:, np.newaxis], 1]
                    - ncenters[neighbor_rows[np.newaxis, :], 1]
                ) ** 2
            )
            distance_matrix[~final_has_pixels[valid_object_indexes], :] = np.inf
            distance_matrix[:, ~neighbor_has_pixels[valid_neighbor_indexes]] = np.inf
            if neighbors_are_same_objects:
                same_count = min(distance_matrix.shape)
                distance_matrix[np.arange(same_count), np.arange(same_count)] = np.inf

            sorted_neighbors = np.argsort(distance_matrix, axis=1)
            valid_final_ids = np.flatnonzero(valid_object_indexes)
            valid_neighbor_ids = np.flatnonzero(valid_neighbor_indexes)
            for row_index, final_id in enumerate(valid_final_ids):
                ordered = sorted_neighbors[row_index]
                ordered = ordered[np.isfinite(distance_matrix[row_index, ordered])]
                if ordered.size > 0:
                    final_first_object_number[final_id] = valid_neighbor_ids[ordered[0]] + 1
                if ordered.size > 1:
                    final_second_object_number[final_id] = valid_neighbor_ids[ordered[1]] + 1
    
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

    # Build measurement results
    measurements = []
    for i in range(final_object_count):
        object_number = object_numbers[i]
        object_index = object_number - 1
        if object_number <= 0:
            measurements.append(NeighborMeasurements(
                slice_index=0,
                object_id=i + 1,
                scale=measurement_scale,
                number_of_neighbors=0,
                percent_touching=0.0,
                first_closest_object_number=0,
                first_closest_distance=0.0,
                second_closest_object_number=0,
                second_closest_distance=0.0,
                angle_between_neighbors=0.0,
            ))
            continue
        first_dist = np.sqrt(
            first_x_vector[object_index] ** 2
            + first_y_vector[object_index] ** 2
        )
        second_dist = np.sqrt(
            second_x_vector[object_index] ** 2
            + second_y_vector[object_index] ** 2
        )
        
        measurements.append(NeighborMeasurements(
            slice_index=0,
            object_id=i + 1,
            scale=measurement_scale,
            number_of_neighbors=int(neighbor_count[object_index]),
            percent_touching=float(percent_touching[object_index]),
            first_closest_object_number=int(final_first_object_number[i]),
            first_closest_distance=float(first_dist),
            second_closest_object_number=int(final_second_object_number[i]),
            second_closest_distance=float(second_dist),
            angle_between_neighbors=float(angle[object_index])
        ))
    
    return _neighbor_output(
        image,
        measurements,
        retained_images=_retained_neighbor_images(
            final_labels,
            neighbor_count_image,
            percent_touching_image,
            retain_neighbor_count_image=retain_neighbor_count_image,
            neighbor_count_colormap=neighbor_count_colormap,
            retain_percent_touching_image=retain_percent_touching_image,
            percent_touching_colormap=percent_touching_colormap,
        ),
    )
