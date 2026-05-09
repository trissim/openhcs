"""Converted from CellProfiler: MaskObjects

Removes objects outside of a specified region or regions.
This module allows you to delete the objects or portions of objects that
are outside of a region (mask) you specify.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.runtime_semantics import (
    ExplicitObjectLabelDomainDeclaration,
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    ParentChildRelationshipPayload,
    aligned_dense_object_label_mask_stack_alignment,
    aligned_dense_object_labels_and_mask,
    dense_object_label_plane_id_domains,
    project_dense_object_label_stack,
)
from openhcs.core.runtime_values import (
    ObjectLabelPayload,
    ObjectLabelRuntimeSliceStackContract,
    ObjectLabelSet,
    object_label_dense_array,
    object_label_payload_with_dense_labels,
)
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois

from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum


class MaskChoice(Enum):
    OBJECTS = "objects"
    IMAGE = "image"


class OverlapHandling(Enum):
    MASK = "keep_overlapping_region"  # Keep only overlapping portion
    KEEP = "keep"  # Keep whole object if any overlap
    REMOVE = "remove"  # Remove if any part outside
    REMOVE_PERCENTAGE = "remove_depending_on_overlap"  # Remove based on fraction


class NumberingChoice(Enum):
    RENUMBER = "renumber"  # Consecutive numbering
    RETAIN = "retain"  # Keep original labels


@dataclass
class MaskObjectsStats:
    slice_index: int
    original_object_count: int
    remaining_object_count: int
    objects_removed: int


@dataclass(frozen=True, slots=True)
class MaskObjectsPlaneResult:
    """MaskObjects result for one runtime plane."""

    labels: np.ndarray
    stats: MaskObjectsStats
    relationships: ParentChildRelationshipPayload


@dataclass(frozen=True, slots=True)
class MaskObjectsPlaneOperation:
    """CellProfiler MaskObjects semantics for one aligned object-label plane."""

    overlap_handling: OverlapHandling
    overlap_fraction: float
    numbering: NumberingChoice
    invert_mask: bool
    relationship_backend: ObjectRelationshipBackendStrategy

    def apply(
        self,
        label_image: np.ndarray,
        mask: np.ndarray,
        *,
        slice_index: int = 0,
    ) -> MaskObjectsPlaneResult:
        import scipy.ndimage as ndi

        label_image = np.asarray(label_image, dtype=np.int32)
        _aligned_labels, mask = aligned_dense_object_labels_and_mask(label_image, mask)
        label_image = _aligned_labels.astype(np.int32, copy=False)

        binary_mask = mask > 0 if mask.max() > 1 else mask.astype(bool)
        if self.invert_mask:
            binary_mask = ~binary_mask

        masked_labels = label_image.copy()
        nobjects = int(np.max(label_image))
        if nobjects == 0:
            return MaskObjectsPlaneResult(
                labels=masked_labels,
                stats=MaskObjectsStats(
                    slice_index=slice_index,
                    original_object_count=0,
                    remaining_object_count=0,
                    objects_removed=0,
                ),
                relationships=ParentChildRelationshipPayload(
                    parent_ids=(),
                    child_ids=(),
                ),
            )

        binary_mask = _size_binary_mask_like_labels(label_image, binary_mask)
        if self.overlap_handling == OverlapHandling.MASK:
            masked_labels = masked_labels * binary_mask.astype(masked_labels.dtype)
        else:
            object_indices = np.arange(1, nobjects + 1, dtype=np.int32)
            pixel_counts = np.atleast_1d(
                ndi.sum(binary_mask.astype(np.float64), label_image, object_indices)
            )

            if self.overlap_handling == OverlapHandling.KEEP:
                keep = pixel_counts > 0
            else:
                total_pixels = np.atleast_1d(
                    ndi.sum(
                        np.ones(label_image.shape, dtype=np.float64),
                        label_image,
                        object_indices,
                    )
                )

                if self.overlap_handling == OverlapHandling.REMOVE:
                    keep = pixel_counts == total_pixels
                elif self.overlap_handling == OverlapHandling.REMOVE_PERCENTAGE:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        fractions = np.where(
                            total_pixels > 0,
                            pixel_counts / total_pixels,
                            0,
                        )
                    keep = fractions >= self.overlap_fraction
                else:
                    keep = pixel_counts > 0

            keep_lookup = np.concatenate([[False], keep])
            masked_labels[~keep_lookup[label_image]] = 0

        if self.numbering == NumberingChoice.RENUMBER:
            unique_labels = np.unique(masked_labels[masked_labels != 0])
            if len(unique_labels) > 0:
                indexer = np.zeros(nobjects + 1, dtype=np.int32)
                indexer[unique_labels] = np.arange(
                    1,
                    len(unique_labels) + 1,
                    dtype=np.int32,
                )
                masked_labels = indexer[masked_labels]
                remaining_count = len(unique_labels)
            else:
                remaining_count = 0
        else:
            remaining_count = len(np.unique(masked_labels[masked_labels != 0]))

        return MaskObjectsPlaneResult(
            labels=masked_labels,
            stats=MaskObjectsStats(
                slice_index=slice_index,
                original_object_count=nobjects,
                remaining_object_count=remaining_count,
                objects_removed=nobjects - remaining_count,
            ),
            relationships=self.relationship_backend.parent_child_payload_from_labels(
                label_image,
                masked_labels,
            ),
        )


@dataclass(frozen=True, slots=True)
class MaskObjectsOutputLabels:
    """Typed MaskObjects label output preserving input object-label semantics."""

    source: object
    labels: np.ndarray

    def value(self) -> object:
        if not isinstance(self.source, (ObjectLabelPayload, ObjectLabelSet)):
            return self.labels
        plane_domains = dense_object_label_plane_id_domains(
            self.labels,
            domain_scope=ObjectLabelDomainScope.PLANE,
        )
        return object_label_payload_with_dense_labels(
            self.source,
            self.labels,
            domain_declaration=ExplicitObjectLabelDomainDeclaration(
                ObjectLabelDomain(
                    declared_object_id_domains=plane_domains,
                    scope=ObjectLabelDomainScope.PLANE,
                )
            ),
        )


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
    relationship_backend_provider: CellProfilerBackendProvider | None = None,
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
    overlap_handling = _coerce_function_enum(OverlapHandling, overlap_handling)
    numbering = _coerce_function_enum(NumberingChoice, numbering)
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


def _size_binary_mask_like_labels(
    labels: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """Return a binary mask sized like CP size_similarly(labels, mask)."""
    if binary_mask.shape == labels.shape:
        return binary_mask
    result = np.zeros(labels.shape, dtype=bool)
    common_slices = tuple(
        slice(0, min(label_extent, mask_extent))
        for label_extent, mask_extent in zip(labels.shape, binary_mask.shape, strict=False)
    )
    if not common_slices:
        return result
    result[common_slices] = binary_mask[common_slices]
    return result
