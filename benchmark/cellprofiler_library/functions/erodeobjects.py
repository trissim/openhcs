"""
Converted from CellProfiler: ErodeObjects
Original: erode_objects
"""

import numpy as np
from typing import Tuple
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_semantics import (
    ParentChildRelationshipPayload,
    object_label_lineage_payload,
)
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois
from dataclasses import dataclass

from benchmark.cellprofiler_library.functions.structuring_elements import (
    StructuringElement,
    adapt_structuring_element_rank,
    build_structuring_element,
)


@dataclass
class ErosionStats:
    slice_index: int
    input_object_count: int
    output_object_count: int
    objects_removed: int


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("erosion_stats", csv_materializer(
        fields=["slice_index", "input_object_count", "output_object_count", "objects_removed"],
        analysis_type="erosion"
    )),
    "parent_child_relationship",
    ("eroded_labels", segmentation_mask_rois())
)
def erode_objects(
    image: np.ndarray,
    labels: np.ndarray,
    structuring_element: StructuringElement | str = StructuringElement.DISK,
    size: int = 1,
    preserve_midpoints: bool = True,
    relabel_objects: bool = False,
) -> Tuple[np.ndarray, ErosionStats, ParentChildRelationshipPayload, np.ndarray]:
    """Erode objects based on the structuring element provided.
    
    This function erodes labeled objects using morphological erosion.
    Objects smaller than the structuring element will be removed entirely
    unless preserve_midpoints is enabled.
    
    Args:
        image: Input intensity image (passed through unchanged)
        labels: Input labeled objects array
        structuring_element: Shape of structuring element
        size: Size/radius of structuring element
        preserve_midpoints: If True, central pixels for each object will not be eroded
        relabel_objects: If True, resulting objects will be relabeled sequentially
        
    Returns:
        Tuple of (image, erosion_stats, eroded_labels)
    """
    from skimage.measure import label as relabel
    labels = object_label_dense_array(labels, dtype=np.int32)

    footprint = adapt_structuring_element_rank(
        build_structuring_element(structuring_element, size),
        labels.ndim,
    )

    input_labels = np.unique(labels)
    input_labels = input_labels[input_labels != 0]
    input_count = len(input_labels)

    contours = _morphological_gradient(labels, footprint)
    eroded = labels * (contours == 0)

    if preserve_midpoints:
        missing_labels = np.setxor1d(labels, eroded)
        preservation = MidpointPreservationPolicy.for_footprint(footprint)
        eroded = preservation.preserve_missing_labels(
            labels,
            eroded,
            missing_labels,
        )

    if relabel_objects:
        eroded = relabel(eroded > 0).astype(labels.dtype)

    output_labels = np.unique(eroded)
    output_labels = output_labels[output_labels != 0]
    output_count = len(output_labels)

    stats = ErosionStats(
        slice_index=0,
        input_object_count=input_count,
        output_object_count=output_count,
        objects_removed=input_count - output_count
    )
    
    relationship = object_label_lineage_payload(labels, eroded)
    return image, stats, relationship, eroded


def _morphological_gradient(labels: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    import scipy.ndimage

    if footprint.ndim == 2 and labels.ndim > 2:
        output = np.zeros_like(labels)
        for index, plane in enumerate(labels):
            output[index] = scipy.ndimage.morphological_gradient(
                plane,
                footprint=footprint,
            )
        return output
    if footprint.ndim > 2 and labels.ndim == 2:
        raise NotImplementedError(
            "A 3D structuring element cannot be applied to a 2D object set."
        )
    return scipy.ndimage.morphological_gradient(labels, footprint=footprint)


class MidpointPreservationPolicy:
    """CellProfiler midpoint preservation for labels lost during erosion."""

    def preserve_missing_labels(
        self,
        labels: np.ndarray,
        eroded: np.ndarray,
        missing_labels: np.ndarray,
    ) -> np.ndarray:
        for label_id in missing_labels:
            binary = labels == label_id
            midpoint = self.midpoint_distance(binary)
            eroded[midpoint == np.max(midpoint)] = label_id
        return eroded

    def midpoint_distance(self, binary: np.ndarray) -> np.ndarray:
        import scipy.ndimage

        return scipy.ndimage.distance_transform_edt(binary)

    @classmethod
    def for_footprint(cls, footprint: np.ndarray) -> "MidpointPreservationPolicy":
        if SimpleDiskMidpointPreservationPolicy.matches(footprint):
            return SimpleDiskMidpointPreservationPolicy()
        return cls()


class SimpleDiskMidpointPreservationPolicy(MidpointPreservationPolicy):
    """CellProfiler's optimized disk-1 behavior restores entire missing labels."""

    @classmethod
    def matches(cls, footprint: np.ndarray) -> bool:
        import skimage.morphology

        return (
            footprint.ndim == 2
            and footprint.shape == (3, 3)
            and np.array_equal(footprint, skimage.morphology.disk(1))
        )

    def preserve_missing_labels(
        self,
        labels: np.ndarray,
        eroded: np.ndarray,
        missing_labels: np.ndarray,
    ) -> np.ndarray:
        return eroded + labels * np.isin(labels, missing_labels)
