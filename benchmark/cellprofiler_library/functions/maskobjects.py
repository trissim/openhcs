"""Converted from CellProfiler: MaskObjects

Removes objects outside of a specified region or regions.
"""

import numpy as np
from typing import Tuple
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.runtime_semantics import (
    ExplicitObjectLabelDomainDeclaration,
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    ParentChildRelationshipPayload,
    aligned_dense_object_label_mask_stack_alignment,
    dense_object_label_plane_id_domains,
    project_dense_object_label_stack,
)
from openhcs.core.runtime_values import (
    ObjectLabelRuntimeSliceStackContract,
    object_label_dense_array,
    object_label_payload_with_dense_labels,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    MaskObjectsOutputLabels,
    MaskObjectsPlaneOperation,
    MaskObjectsStats,
)
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois

from openhcs.interop.cellprofiler.mask_objects_settings import (
    MaskObjectsNumberingChoice as NumberingChoice,
    MaskObjectsOverlapHandling as OverlapHandling,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum


class MaskChoice(Enum):
    OBJECTS = "objects"
    IMAGE = "image"


@numpy
@special_inputs("labels", "mask")
@special_outputs(
    ("mask_stats", csv_materializer(
        fields=["slice_index", "original_object_count", "remaining_object_count", "objects_removed"],
        analysis_type="mask_objects"
    )),
    "object_relationships",
    ("masked_labels", segmentation_mask_rois())
)
def mask_objects(
    image: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    overlap_handling: OverlapHandling = OverlapHandling.MASK,
    overlap_fraction: float = 0.5,
    numbering: NumberingChoice = NumberingChoice.RENUMBER,
    invert_mask: bool = False,
    relationship_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[np.ndarray, MaskObjectsStats, ParentChildRelationshipPayload, np.ndarray]:
    """
    Mask objects based on a binary mask or masking objects.
    
    Args:
        image: Input image, shape (D, H, W) - passed through unchanged
        labels: Label image of objects to mask, shape (H, W)
        mask: Binary mask or label image defining masking region, shape (H, W)
        overlap_handling: How to handle partially masked objects
            - MASK: Keep only the overlapping portion
            - KEEP: Keep whole object if any part overlaps
            - REMOVE: Remove object if any part is outside mask
            - REMOVE_PERCENTAGE: Remove based on overlap fraction
        overlap_fraction: Minimum fraction of object that must overlap (for REMOVE_PERCENTAGE)
        numbering: Whether to renumber objects consecutively or retain original labels
        invert_mask: If True, use the inverse of the mask
    
    Returns:
        Tuple of (image, stats, parent-child relationship, masked_labels)
    """
    overlap_handling = coerce_cellprofiler_enum(OverlapHandling, overlap_handling)
    numbering = coerce_cellprofiler_enum(NumberingChoice, numbering)
    label_array = object_label_dense_array(labels, dtype=np.int32)
    relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type(
        backend_provider=relationship_backend_provider,
    )
    operation = MaskObjectsPlaneOperation(
        overlap_handling=overlap_handling,
        overlap_fraction=overlap_fraction,
        numbering=numbering,
        invert_mask=invert_mask,
        relationship_backend=relationship_backend,
    )
    stack_slice_count = ObjectLabelRuntimeSliceStackContract.runtime_slice_count(labels)
    if stack_slice_count is None and label_array.ndim == 3:
        stack_slice_count = int(label_array.shape[0])
    if stack_slice_count is not None and stack_slice_count > 1:
        stack_alignment = aligned_dense_object_label_mask_stack_alignment(
            label_array,
            mask,
            slice_count=stack_slice_count,
        )
        if stack_alignment is not None:
            plane_results = tuple(
                operation.apply(
                    stack_alignment.label_stack[slice_index],
                    stack_alignment.mask_stack[slice_index],
                    slice_index=slice_index,
                )
                for slice_index in range(stack_slice_count)
            )
            masked_stack = stack_alignment.restore_label_stack(
                np.stack([result.labels for result in plane_results], axis=0)
            )
            plane_domains = dense_object_label_plane_id_domains(
                masked_stack,
                domain_scope=ObjectLabelDomainScope.PLANE,
            )
            masked_payload = object_label_payload_with_dense_labels(
                labels,
                masked_stack,
                domain_declaration=ExplicitObjectLabelDomainDeclaration(
                    ObjectLabelDomain(
                        declared_object_id_domains=plane_domains,
                        scope=ObjectLabelDomainScope.PLANE,
                    )
                ),
            )
            relationships = ParentChildRelationshipPayload(
                parent_ids=tuple(
                    parent_id
                    for result in plane_results
                    for parent_id in result.relationships.parent_ids
                ),
                child_ids=tuple(
                    child_id
                    for result in plane_results
                    for child_id in result.relationships.child_ids
                ),
                slice_indices=tuple(
                    slice_index
                    for slice_index, result in enumerate(plane_results)
                    for _child_id in result.relationships.child_ids
                ),
                slice_count=stack_slice_count,
            )
            return image, list(result.stats for result in plane_results), relationships, masked_payload

    try:
        label_image = project_dense_object_label_stack(
            label_array,
        ).astype(np.int32, copy=False)
    except ValueError as exc:
        raise ValueError(
            "MaskObjects could not project object labels; "
            f"labels shape={label_array.shape!r}, "
            f"mask shape={mask.shape!r}."
        ) from exc
    result = operation.apply(label_image, mask)

    masked_labels = MaskObjectsOutputLabels(labels, result.labels).value()
    return image, result.stats, result.relationships, masked_labels
