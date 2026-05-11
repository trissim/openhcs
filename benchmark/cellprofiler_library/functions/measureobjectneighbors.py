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
import logging
import os
import time
from typing import Tuple
from openhcs.core.memory import numpy
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
)
from openhcs.processing.backends.cellprofiler.morphology import MorphologyBackendStrategy
from openhcs.processing.backends.cellprofiler.neighbors import (
    DistanceMethod,
    NeighborDistancePlanner,
    NeighborMeasurements,
    NeighborRetainedImageRequest,
    labels_or_default,
    neighbor_topology_backend,
    require_matching_shape,
    variant_numbers_for_final_labels,
)
from openhcs.processing.backends.cellprofiler.outlines import object_outline_backend
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def _runtime_profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_runtime_profile(label: str, seconds: float, **fields: object) -> None:
    if not _runtime_profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


def _profile_elapsed(label: str, start: float, **fields: object) -> float:
    now = time.perf_counter()
    _log_runtime_profile(label, now - start, **fields)
    return now


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
    neighbor_topology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    outline_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    relationship_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
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
    profile_start = time.perf_counter()
    profile_mark = profile_start
    labels = object_label_dense_array(labels, dtype=np.int32)
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
        else object_label_dense_array(neighbor_labels, dtype=np.int32)
    )
    measured_variant_labels = labels_or_default(
        None
        if small_removed_labels is None
        else object_label_dense_array(small_removed_labels, dtype=np.int32),
        final_labels,
    )
    neighbor_variant_labels = (
        measured_variant_labels
        if neighbors_are_same_objects and small_removed_neighbor_labels is None
        else labels_or_default(
            None
            if small_removed_neighbor_labels is None
            else object_label_dense_array(small_removed_neighbor_labels, dtype=np.int32),
            neighbor_final_labels,
        )
    )

    require_matching_shape(final_labels, measured_variant_labels, "small_removed_labels")
    require_matching_shape(
        neighbor_final_labels,
        neighbor_variant_labels,
        "small_removed_neighbor_labels",
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
        backend_provider=neighbor_topology_backend_provider,
    )

    object_numbers = variant_numbers_for_final_labels(
        final_labels,
        measured_variant_labels,
        neighbor_backend=neighbor_backend,
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
                scale=coerce_cellprofiler_enum(DistanceMethod, distance_method).value,
                number_of_neighbors=0,
                percent_touching=0.0,
                first_closest_object_number=0,
                first_closest_distance=0.0,
                second_closest_object_number=0,
                second_closest_distance=0.0,
                angle_between_neighbors=0.0,
            ))
        empty_metric_image = retained_image_request.empty_metric_image()
        return retained_image_request.output(
            image,
            measurements,
            neighbor_count_image=empty_metric_image,
            percent_touching_image=empty_metric_image,
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
    
    normalized_distance_method = coerce_cellprofiler_enum(
        DistanceMethod,
        distance_method,
    )
    distance_plan = NeighborDistancePlanner.for_method(
        normalized_distance_method
    ).plan(
        measured_topology_labels,
        neighbor_distance,
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
        # Calculate object centers
        relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type(
            backend_provider=relationship_backend_provider,
        )
        ocenters = relationship_backend.label_centers(measured_variant_labels)
        ncenters = relationship_backend.label_centers(neighbor_variant_labels)
        profile_mark = _profile_elapsed(
            "measure_object_neighbors.centers",
            profile_mark,
            variant_object_count=variant_object_count,
            variant_neighbor_count=variant_neighbor_count,
        )
        
        # Calculate perimeters
        perimeter_outlines = object_outline_backend(
            backend_provider=outline_backend_provider,
        ).outline(working_labels)
        perimeters = neighbor_backend.perimeter_counts(
            perimeter_outlines,
            variant_object_count=variant_object_count,
        )
        profile_mark = _profile_elapsed(
            "measure_object_neighbors.outline_perimeters",
            profile_mark,
            shape=working_labels.shape,
        )
        
        # Find nearest neighbors using variant-label center distances.
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
            backend_provider=morphology_backend_provider,
        )
        strel = morphology_backend.disk_footprint(distance)
        strel_touching = morphology_backend.disk_footprint(distance + 0.5)

        topology = neighbor_backend.measure_topology(
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
        profile_mark = _profile_elapsed(
            "measure_object_neighbors.topology",
            profile_mark,
            distance=distance,
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
    profile_mark = _profile_elapsed(
        "measure_object_neighbors.rows",
        profile_mark,
        row_count=len(measurements),
    )
    
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
