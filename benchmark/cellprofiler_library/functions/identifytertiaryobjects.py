"""Converted from CellProfiler: IdentifyTertiaryObjects

Identifies tertiary objects (e.g., cytoplasm) by removing smaller primary
objects (e.g., nuclei) from larger secondary objects (e.g., cells),
leaving a ring shape.
"""

import numpy as np
from typing import Any, Tuple
from dataclasses import dataclass
from numba import njit
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import (
    RuntimePure2DSliceBatchRequest,
    pure_2d_batch_executor,
)
from openhcs.core.runtime_semantics import (
    ExplicitObjectLabelDomainDeclaration,
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    ParentChildRelationshipPayload,
    aligned_dense_object_label_arrays,
    aligned_dense_object_label_stack_alignment,
    dense_object_label_plane_id_domains,
    object_label_parent_child_payload,
)
from openhcs.core.runtime_values import (
    ObjectLabelPayload,
    ObjectLabelSet,
    object_label_dense_array,
    object_label_payload_with_dense_labels,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.cellprofiler.outlines import ObjectOutlineBackendStrategy


@dataclass
class TertiaryObjectStats:
    slice_index: int
    object_count: int
    mean_area: float
    primary_parent_count: int
    secondary_parent_count: int


@dataclass(frozen=True, slots=True)
class TertiaryObjectLabelOutput:
    """Typed tertiary labels preserving the secondary object-label domain."""

    source: object
    labels: np.ndarray

    def value(self) -> object:
        if not isinstance(self.source, (ObjectLabelPayload, ObjectLabelSet)):
            return self.labels
        return object_label_payload_with_dense_labels(
            self.source,
            self.labels,
            domain_declaration=ExplicitObjectLabelDomainDeclaration(
                ObjectLabelDomain(
                    declared_object_id_domains=dense_object_label_plane_id_domains(
                        self.labels,
                        domain_scope=ObjectLabelDomainScope.PLANE,
                    ),
                    scope=ObjectLabelDomainScope.PLANE,
                )
            ),
        )


def _positive_label_count(labels: np.ndarray) -> int:
    return int(np.count_nonzero(np.bincount(np.asarray(labels).ravel())[1:]))


def _positive_label_mean_area(labels: np.ndarray) -> tuple[int, float]:
    areas = np.bincount(np.asarray(labels).ravel())[1:]
    positive_areas = areas[areas > 0]
    if positive_areas.size == 0:
        return 0, 0.0
    return int(positive_areas.size), float(np.mean(positive_areas))


def _identify_tertiary_objects_batch(
    request: RuntimePure2DSliceBatchRequest,
) -> list[Any]:
    kwargs = request.kwargs
    slice_count = request.slice_count
    alignment = aligned_dense_object_label_stack_alignment(
        kwargs["primary_labels"],
        kwargs["secondary_labels"],
        slice_count=slice_count,
    )
    if alignment is None:
        return [
            request.execute_one(slice_index)
            for slice_index in range(request.slice_count)
        ]
    primary_stack = alignment.first_stack
    secondary_stack = alignment.second_stack

    tertiary_stack, object_counts, mean_areas, primary_counts, secondary_counts = (
        _tertiary_stack_numba(
            primary_stack,
            secondary_stack,
            bool(kwargs.get("shrink_primary", True)),
        )
    )
    output_tertiary_stack = alignment.restore_second_stack(tertiary_stack)

    return [
        (
                request.slices_2d[slice_index],
            object_label_parent_child_payload(
                secondary_stack[slice_index],
                tertiary_stack[slice_index],
            ),
            object_label_parent_child_payload(
                primary_stack[slice_index],
                tertiary_stack[slice_index],
                child_region_labels=secondary_stack[slice_index],
            ),
            TertiaryObjectStats(
                slice_index=slice_index,
                object_count=int(object_counts[slice_index]),
                mean_area=float(mean_areas[slice_index]),
                primary_parent_count=int(primary_counts[slice_index]),
                secondary_parent_count=int(secondary_counts[slice_index]),
            ),
            TertiaryObjectLabelOutput(
                kwargs["secondary_labels"],
                output_tertiary_stack[slice_index],
            ).value(),
        )
        for slice_index in range(slice_count)
    ]


@njit(cache=True)
def _tertiary_stack_numba(
    primary_stack: np.ndarray,
    secondary_stack: np.ndarray,
    shrink_primary: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    slice_count, height, width = secondary_stack.shape
    max_primary = 0
    max_secondary = 0
    for z in range(slice_count):
        for y in range(height):
            for x in range(width):
                primary_label = primary_stack[z, y, x]
                secondary_label = secondary_stack[z, y, x]
                if primary_label > max_primary:
                    max_primary = primary_label
                if secondary_label > max_secondary:
                    max_secondary = secondary_label

    tertiary_stack = np.zeros_like(secondary_stack)
    primary_present = np.zeros((slice_count, max_primary + 1), dtype=np.uint8)
    secondary_present = np.zeros((slice_count, max_secondary + 1), dtype=np.uint8)
    tertiary_present = np.zeros((slice_count, max_secondary + 1), dtype=np.uint8)
    tertiary_areas = np.zeros((slice_count, max_secondary + 1), dtype=np.int64)
    first_y = np.full((slice_count, max_secondary + 1), -1, dtype=np.int64)
    first_x = np.full((slice_count, max_secondary + 1), -1, dtype=np.int64)

    for z in range(slice_count):
        for y in range(height):
            for x in range(width):
                primary_label = primary_stack[z, y, x]
                secondary_label = secondary_stack[z, y, x]
                if primary_label > 0:
                    primary_present[z, primary_label] = 1
                if secondary_label > 0:
                    secondary_present[z, secondary_label] = 1
                    if first_y[z, secondary_label] < 0:
                        first_y[z, secondary_label] = y
                        first_x[z, secondary_label] = x

                keep_pixel = primary_label <= 0
                if shrink_primary and primary_label > 0:
                    for dy in range(-1, 2):
                        ny = y + dy
                        for dx in range(-1, 2):
                            nx = x + dx
                            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                                keep_pixel = True
                            elif primary_stack[z, ny, nx] != primary_label:
                                keep_pixel = True

                if keep_pixel and secondary_label > 0:
                    tertiary_stack[z, y, x] = secondary_label
                    tertiary_present[z, secondary_label] = 1
                    tertiary_areas[z, secondary_label] += 1

    for z in range(slice_count):
        for label in range(1, max_secondary + 1):
            if secondary_present[z, label] == 0 or tertiary_present[z, label] != 0:
                continue
            y = first_y[z, label]
            x = first_x[z, label]
            if y >= 0:
                tertiary_stack[z, y, x] = label
                tertiary_present[z, label] = 1
                tertiary_areas[z, label] += 1

    object_counts = np.zeros(slice_count, dtype=np.int64)
    mean_areas = np.zeros(slice_count, dtype=np.float64)
    primary_counts = np.zeros(slice_count, dtype=np.int64)
    secondary_counts = np.zeros(slice_count, dtype=np.int64)
    for z in range(slice_count):
        total_area = 0
        for label in range(1, max_primary + 1):
            if primary_present[z, label] != 0:
                primary_counts[z] += 1
        for label in range(1, max_secondary + 1):
            if secondary_present[z, label] != 0:
                secondary_counts[z] += 1
            if tertiary_present[z, label] != 0:
                object_counts[z] += 1
                total_area += tertiary_areas[z, label]
        if object_counts[z] > 0:
            mean_areas[z] = total_area / object_counts[z]
    return tertiary_stack, object_counts, mean_areas, primary_counts, secondary_counts


@numpy
def identify_tertiary_objects(
    image: np.ndarray,
    primary_labels: np.ndarray | ObjectLabelPayload,
    secondary_labels: np.ndarray | ObjectLabelPayload,
    shrink_primary: bool = True,
    outline_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[
    np.ndarray,
    ParentChildRelationshipPayload,
    ParentChildRelationshipPayload,
    TertiaryObjectStats,
    np.ndarray,
]:
    """
    Identify tertiary objects by subtracting primary objects from secondary objects.
    
    Creates ring-shaped objects (e.g., cytoplasm) by removing smaller objects
    (e.g., nuclei) from larger objects (e.g., cells).
    
    Args:
        image: Input image, shape (D, H, W) - used as reference, passed through
        primary_labels: Label image of smaller objects (e.g., nuclei), shape (H, W)
        secondary_labels: Label image of larger objects (e.g., cells), shape (H, W)
        shrink_primary: If True, shrink primary objects by 1 pixel before subtraction
                       to ensure tertiary objects always have some area
    
    Returns:
        Tuple of:
        - Original image (passed through)
        - Secondary-parent to tertiary-child relationships
        - Primary-parent to tertiary-child relationships
        - TertiaryObjectStats dataclass with measurements
        - Tertiary label image (ring-shaped objects)

    CellProfiler Parameter Mapping:
    (CellProfiler setting -> Python parameter)
        'Select the larger identified objects' -> (pipeline-handled)
        'Select the smaller identified objects' -> (pipeline-handled)
        'Name the tertiary objects to be identified' -> (pipeline-handled)
        'Shrink smaller object prior to subtraction?' -> shrink_primary
    """
    # Handle 3D input - process slice by slice or take first slice
    if image.ndim == 3:
        # For FLEXIBLE contract, we process the first slice as reference
        ref_image = image[0]
    else:
        ref_image = image
    
    # Ensure labels are 2D
    primary_payload = primary_labels
    secondary_payload = secondary_labels
    primary_array = object_label_dense_array(primary_labels, dtype=np.int32)
    secondary_array = object_label_dense_array(secondary_labels, dtype=np.int32)
    if primary_array.ndim == 3:
        primary_array = primary_array[0]
    if secondary_array.ndim == 3:
        secondary_array = secondary_array[0]
    primary_array, secondary_array = aligned_dense_object_label_arrays(
        primary_array,
        secondary_array,
    )
    
    # Ensure shapes match
    if primary_array.shape != secondary_array.shape:
        raise ValueError(
            f"Primary and secondary label shapes must match. "
            f"Got {primary_array.shape} vs {secondary_array.shape}"
        )
    
    # Find outlines of primary objects
        primary_outline = ObjectOutlineBackendStrategy.for_memory_type(
            backend_provider=outline_backend_provider,
        ).outline(primary_array)
    
    # Create tertiary labels by subtracting primary from secondary
    tertiary_labels = secondary_array.copy()
    
    if shrink_primary:
        # Keep pixels that are either background OR on the outline of primary
        # This shrinks primary objects by 1 pixel
        primary_mask = np.logical_or(primary_array == 0, primary_outline > 0)
    else:
        # Only keep pixels where primary is background
        primary_mask = primary_array == 0
    
    # Remove primary object pixels from tertiary
    tertiary_labels[~primary_mask] = 0
    
    object_count, mean_area = _positive_label_mean_area(tertiary_labels)
    
    # Count unique parent objects
    primary_parent_count = _positive_label_count(primary_array)
    secondary_parent_count = _positive_label_count(secondary_array)
    
    stats = TertiaryObjectStats(
        slice_index=0,
        object_count=object_count,
        mean_area=float(mean_area),
        primary_parent_count=int(primary_parent_count),
        secondary_parent_count=int(secondary_parent_count)
    )
    
    # Ensure output has correct shape (D, H, W)
    if image.ndim == 3:
        tertiary_labels_out = np.expand_dims(tertiary_labels, axis=0)
    else:
        tertiary_labels_out = tertiary_labels
    
    return (
        image,
        object_label_parent_child_payload(secondary_payload, tertiary_labels),
        object_label_parent_child_payload(
            primary_payload,
            tertiary_labels,
            child_region_labels=secondary_array,
        ),
        stats,
        TertiaryObjectLabelOutput(secondary_payload, tertiary_labels_out).value(),
    )


def _prepare_identify_tertiary_objects() -> None:
    """Compile tertiary-object kernels before benchmarked execution."""
    image = np.zeros((32, 32), dtype=np.float32)
    primary = np.zeros((32, 32), dtype=np.int32)
    secondary = np.zeros((32, 32), dtype=np.int32)
    primary[10:20, 10:20] = 1
    secondary[6:24, 6:24] = 1
    identify_tertiary_objects.__wrapped__(
        image,
        primary,
        secondary,
        shrink_primary=True,
    )
    _tertiary_stack_numba(
        np.expand_dims(primary, axis=0),
        np.expand_dims(secondary, axis=0),
        True,
    )


identify_tertiary_objects.__openhcs_prepare__ = _prepare_identify_tertiary_objects
pure_2d_batch_executor(_identify_tertiary_objects_batch)(identify_tertiary_objects)
