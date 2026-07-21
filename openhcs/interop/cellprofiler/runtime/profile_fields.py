"""CellProfiler runtime profile field projection."""

from __future__ import annotations

import numpy as np

from openhcs.core.image_shapes import ArrayShape
from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
)
from openhcs.core.runtime_image_values import (
    image_payload_data,
)
from openhcs.interop.cellprofiler.runtime.invocation import CellProfilerMeasurementImage
from openhcs.core.steps.function_runtime import RuntimeCallableArgument, RuntimeProfileFieldValue


def object_label_stage_profile_fields(
    stage: str,
    measurement_image: CellProfilerMeasurementImage,
    object_spec: ArtifactSpec,
    value: ObjectLabelValue,
) -> tuple[tuple[str, RuntimeProfileFieldValue], ...]:
    """Return structured profile fields for object-label values."""
    label_data = value.labels
    domain = value.domain
    return (
        ("stage", stage),
        ("object", object_spec.name),
        ("source_image_name", measurement_image.source_image_name),
        ("reference_domain", measurement_image.reference_domain.value),
        ("value_type", type(value).__name__),
        ("data_type", type(label_data).__name__),
        ("data_shape", ArrayShape.shape_for(label_data)),
        ("domain_scope", domain.scope),
        ("plane_axis", value.plane_axis),
        ("representation", value.representation),
        ("spatial_origin_yx", value.spatial_origin_yx),
        ("source_spatial_shape_yx", value.source_spatial_shape_yx),
        ("source_image_names", value.source_image_names),
        (
            "declared_domain_lengths",
            tuple(len(object_ids) for object_ids in domain.declared_object_id_domains),
        ),
        ("declared_object_count", domain.declared_object_count),
        ("declared_object_ids_count", len(domain.declared_object_ids)),
    )


def dense_label_argument_stage_profile_fields(
    stage: str,
    measurement_image: CellProfilerMeasurementImage,
    object_spec: ArtifactSpec,
    value: RuntimeCallableArgument,
) -> tuple[tuple[str, RuntimeProfileFieldValue], ...]:
    """Return structured profile fields for dense label arrays."""
    data = image_payload_data(value)
    return (
        ("stage", stage),
        ("object", object_spec.name),
        ("source_image_name", measurement_image.source_image_name),
        ("reference_domain", measurement_image.reference_domain.value),
        ("value_type", type(value).__name__),
        ("data_type", type(data).__name__),
        ("data_shape", ArrayShape.shape_for(data)),
    )


def cellprofiler_profile_payload_fields(
    prefix: str,
    value: RuntimeCallableArgument,
) -> dict[str, RuntimeCallableArgument]:
    """Return cheap payload shape/size fields for CellProfiler runtime profiling."""
    data = image_payload_data(value)
    data_array = data if isinstance(data, np.ndarray) else None
    return {
        f"{prefix}_type": type(data).__name__,
        f"{prefix}_shape": None if data_array is None else data_array.shape,
        f"{prefix}_nbytes": None if data_array is None else int(data_array.nbytes),
    }


def object_label_artifact_profile_fields(
    value: ObjectLabelValue,
) -> dict[str, RuntimeCallableArgument]:
    """Return object-label artifact fields for runtime adapter profiling."""
    source_component_metadata = None
    if value.source_component_metadata is not None:
        source_component_metadata = dict(value.source_component_metadata)
    domain = value.domain
    return {
        "label_shape": ArrayShape.shape_for(value.labels),
        "declared_object_count": domain.declared_object_count,
        "declared_object_ids": len(domain.declared_object_ids),
        "declared_object_id_domains": len(domain.declared_object_id_domains),
        "domain_scope": domain.scope.value,
        "plane_axis": None if value.plane_axis is None else value.plane_axis.value,
        "source_path": value.source_path,
        "source_component_metadata": source_component_metadata,
        "source_image_provenance_plane_count": (
            value.source_image_provenance_planes.count
        ),
    }
