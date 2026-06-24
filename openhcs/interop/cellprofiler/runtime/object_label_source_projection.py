"""Object-label source projection for CellProfiler current-image alignment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_values import (
    ObjectLabelSet,
    image_payload_data,
    image_payload_metadata,
    object_label_dense_array,
)
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.current_image_context import (
    CellProfilerRequiredCurrentImageContext,
)
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerRuntimeValue
from openhcs.interop.cellprofiler.runtime.projection import (
    CurrentSourcePayloadPlaneSelector,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)
from openhcs.interop.cellprofiler.runtime.source_identity import (
    SourcePlaneIdentitySequenceAlignment,
)


class ObjectLabelPlaneAlignmentAbsenceReason(str, Enum):
    """Reasons current-image/object-label plane alignment has no value."""

    MISSING_PLANE_IDENTITY = "missing_plane_identity"
    SINGLE_LABEL_IDENTITY = "single_label_identity"
    LABELS_NOT_PLANE_STACK = "labels_not_plane_stack"
    NO_SINGLE_PLANE_MATCH = "no_single_plane_match"
    NO_STACK_PLANE_MATCH = "no_stack_plane_match"


@dataclass(frozen=True, slots=True)
class ObjectLabelPlaneAlignmentResult:
    """Typed result for current-image/object-label plane alignment."""

    value: CellProfilerRuntimeValue | None
    absence_reason: ObjectLabelPlaneAlignmentAbsenceReason | None = None


@dataclass(frozen=True, slots=True)
class CurrentImageObjectLabelPlaneAlignment(CellProfilerRequiredCurrentImageContext):
    """Order object-label planes by the current image stack's source identities."""

    adapter: CellProfilerRuntimeAdapter
    labels: ObjectLabelSet

    def aligned_dense_value(self) -> CellProfilerRuntimeValue | None:
        return self.alignment_result().value

    def alignment_result(self) -> ObjectLabelPlaneAlignmentResult:
        current_image = self.current_image
        selector = CurrentSourcePayloadPlaneSelector(
            self.adapter,
            current_image,
        )
        image_identities = selector.payload_image_set_identities(current_image)
        label_identities = selector.payload_image_set_identities(
            self.labels.runtime_payload()
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "mask_image_object_label_plane_alignment",
            0.0,
            image_identity_count=len(image_identities),
            label_identity_count=len(label_identities),
            image_has_identity=any(image_identities),
            label_has_identity=any(label_identities),
            current_image_metadata=image_payload_metadata(
                current_image
            ).source_component_metadata,
            current_image_shape=tuple(np.shape(image_payload_data(current_image))),
            current_image_source_path=image_payload_metadata(current_image).source_path,
            current_image_source_image_provenance_paths=image_payload_metadata(
                current_image
            ).source_image_provenance_planes.paths,
            label_shape=tuple(np.shape(object_label_dense_array(self.labels))),
            label_plane_counts=self.label_plane_counts(),
        )
        if not image_identities or not label_identities:
            return ObjectLabelPlaneAlignmentResult(
                None,
                ObjectLabelPlaneAlignmentAbsenceReason.MISSING_PLANE_IDENTITY,
            )
        if len(label_identities) <= 1:
            return ObjectLabelPlaneAlignmentResult(
                None,
                ObjectLabelPlaneAlignmentAbsenceReason.SINGLE_LABEL_IDENTITY,
            )
        label_stack = object_label_dense_array(self.labels, dtype=np.int32)
        if not isinstance(label_stack, np.ndarray) or label_stack.ndim < 3:
            return ObjectLabelPlaneAlignmentResult(
                None,
                ObjectLabelPlaneAlignmentAbsenceReason.LABELS_NOT_PLANE_STACK,
            )
        if len(image_identities) == 1:
            identity_alignment = SourcePlaneIdentitySequenceAlignment(
                image_identities,
                label_identities,
            )
            label_index = identity_alignment.target_index_for_image_plane(
                image_identities[0]
            )
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "mask_image_object_label_plane_alignment_single",
                0.0,
                matched_label_index=label_index,
                label_shape=tuple(np.shape(label_stack)),
            )
            if label_index is None:
                return ObjectLabelPlaneAlignmentResult(
                    None,
                    ObjectLabelPlaneAlignmentAbsenceReason.NO_SINGLE_PLANE_MATCH,
                )
            return ObjectLabelPlaneAlignmentResult(label_stack[label_index])
        identity_alignment = SourcePlaneIdentitySequenceAlignment(
            image_identities,
            label_identities,
        )
        label_indexes = identity_alignment.target_indexes_for_image_planes()
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "mask_image_object_label_plane_alignment_match",
            0.0,
            image_identity_count=len(image_identities),
            label_identity_count=len(label_identities),
            matched_label_indexes=label_indexes,
            label_shape=tuple(np.shape(object_label_dense_array(self.labels))),
        )
        if label_indexes is None:
            return ObjectLabelPlaneAlignmentResult(
                None,
                ObjectLabelPlaneAlignmentAbsenceReason.NO_STACK_PLANE_MATCH,
            )
        return ObjectLabelPlaneAlignmentResult(
            RuntimeSliceAlignedValues(
                tuple(label_stack[index] for index in label_indexes)
            )
        )

    def label_plane_counts(self) -> tuple[int, ...] | None:
        labels = object_label_dense_array(self.labels)
        if not isinstance(labels, np.ndarray) or labels.ndim < 3:
            return None
        counts: list[int] = []
        for plane in labels:
            count = 0
            if plane.size:
                count = int(np.max(plane))
            counts.append(count)
        return tuple(counts)
