"""
Converted from CellProfiler: ErodeObjects
Original: erode_objects
"""

import numpy as np
import logging
import os
import time
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
from openhcs.processing.backends.cellprofiler.morphology import MorphologyBackendStrategy

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


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
    total_started_at = time.perf_counter()
    labels = object_label_dense_array(labels, dtype=np.int32)

    footprint = adapt_structuring_element_rank(
        build_structuring_element(structuring_element, size),
        labels.ndim,
    )

    phase_started_at = time.perf_counter()
    input_labels = np.unique(labels)
    input_labels = input_labels[input_labels != 0]
    input_count = len(input_labels)
    _log_profile("erode_objects_input_labels", time.perf_counter() - phase_started_at)

    phase_started_at = time.perf_counter()
    eroded = MorphologyBackendStrategy.for_memory_type().erode_labeled_objects(
        labels,
        footprint,
    )
    _log_profile("erode_objects_backend", time.perf_counter() - phase_started_at)

    if preserve_midpoints:
        phase_started_at = time.perf_counter()
        missing_labels = np.setxor1d(labels, eroded)
        preservation = MidpointPreservationPolicy.for_footprint(footprint)
        eroded = preservation.preserve_missing_labels(
            labels,
            eroded,
            missing_labels,
        )
        _log_profile(
            "erode_objects_preserve_midpoints",
            time.perf_counter() - phase_started_at,
            missing=len(missing_labels),
            policy=type(preservation).__name__,
        )

    if relabel_objects:
        phase_started_at = time.perf_counter()
        eroded = relabel(eroded > 0).astype(labels.dtype)
        _log_profile("erode_objects_relabel", time.perf_counter() - phase_started_at)

    phase_started_at = time.perf_counter()
    output_labels = np.unique(eroded)
    output_labels = output_labels[output_labels != 0]
    output_count = len(output_labels)
    _log_profile("erode_objects_output_labels", time.perf_counter() - phase_started_at)

    stats = ErosionStats(
        slice_index=0,
        input_object_count=input_count,
        output_object_count=output_count,
        objects_removed=input_count - output_count
    )
    
    phase_started_at = time.perf_counter()
    relationship = object_label_lineage_payload(labels, eroded)
    _log_profile("erode_objects_lineage", time.perf_counter() - phase_started_at)
    _log_profile("erode_objects_total", time.perf_counter() - total_started_at)
    return image, stats, relationship, eroded


class MidpointPreservationPolicy:
    """CellProfiler midpoint preservation for labels lost during erosion."""

    def preserve_missing_labels(
        self,
        labels: np.ndarray,
        eroded: np.ndarray,
        missing_labels: np.ndarray,
    ) -> np.ndarray:
        for label_id in missing_labels:
            label_positions = np.argwhere(labels == label_id)
            if label_positions.size == 0:
                continue
            lower = label_positions.min(axis=0)
            upper = label_positions.max(axis=0) + 1
            expanded_lower = np.maximum(lower - 1, 0)
            expanded_upper = np.minimum(upper + 1, labels.shape)
            expanded_slices = tuple(
                slice(int(start), int(stop))
                for start, stop in zip(expanded_lower, expanded_upper, strict=True)
            )
            inner_slices = tuple(
                slice(int(start - expanded_start), int(stop - expanded_start))
                for start, stop, expanded_start in zip(
                    lower,
                    upper,
                    expanded_lower,
                    strict=True,
                )
            )
            output_slices = tuple(
                slice(int(start), int(stop))
                for start, stop in zip(lower, upper, strict=True)
            )
            binary = labels[expanded_slices] == label_id
            midpoint = self.midpoint_distance(binary)[inner_slices]
            eroded_region = eroded[output_slices]
            eroded_region[midpoint == np.max(midpoint)] = label_id
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
