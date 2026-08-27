"""
Converted from CellProfiler: IdentifyObjectsManually
Original: IdentifyObjectsManually.run

Note: This module in CellProfiler requires interactive user input via a GUI dialog.
OpenHCS requires an explicit interactive label payload because headless execution
cannot infer a user's manual segmentation.

For actual manual annotation, use external tools (napari, Fiji, etc.) and import
the resulting label images.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactOutputModule,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.interop.cellprofiler.setting_names import SettingNameFamily
from openhcs.interop.cellprofiler.settings_binder import SettingToKeywordBinding
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.artifacts import (
    ArtifactSpec,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.pipeline.function_contracts import artifact_inputs


@dataclass(frozen=True, slots=True)
class ManualObjectStats:
    """Statistics for manually identified objects."""

    slice_index: int
    object_count: int
    mean_area: float
    mean_centroid_x: float
    mean_centroid_y: float


@artifact_inputs(
    ArtifactSpec.input(
        "labels_input",
        ObjectLabelsArtifactType,
        parameter_name="labels_input",
    )
)
@numpy(contract=ProcessingContract.PURE_2D)
def identify_objects_manually(
    image: np.ndarray,
    labels_input: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    DataclassMeasurementColumnarRows,
    ObjectLabelValue,
]:
    """
    Placeholder for manual object identification.

    In CellProfiler, this module displays an interactive UI where users can
    manually outline objects using mouse tools (outline, zoom, erase).

    In OpenHCS batch processing context, this function requires a pre-annotated
    label payload and fails explicitly when no interactive result is supplied.

    For actual manual annotation workflows:
    - Use napari, Fiji, or other annotation tools to create label images
    - Import the label images as a separate channel/input
    - Pass them via labels_input parameter

    Args:
        image: Input image to display for annotation, shape (H, W)
        labels_input: Optional pre-annotated label image, shape (H, W).
    Returns:
        Tuple of:
        - Original image (unchanged)
        - ManualObjectStats dataclass with object measurements
        - Label image where each object has a unique integer ID

    Note:
        This module cannot be used in batch mode in CellProfiler.
        The OpenHCS version provides a passthrough for pre-annotated labels
        or returns empty results for pipeline compatibility.
    """
    from skimage.measure import regionprops

    image_array = np.asarray(image)
    if image_array.ndim != 2:
        raise ValueError(
            "IdentifyObjectsManually requires one 2-D guiding image plane, got "
            f"shape {image_array.shape!r}."
        )
    h, w = image_array.shape
    if labels_input is None:
        raise NotImplementedError(
            "IdentifyObjectsManually requires an explicit interactive label payload; "
            "headless execution cannot silently synthesize empty objects."
        )
    labels = object_label_dense_array(labels_input, dtype=np.int32)
    if labels.shape != (h, w):
        raise ValueError(
            "IdentifyObjectsManually labels must match the guiding image plane; "
            f"got labels {labels.shape!r} and image {(h, w)!r}."
        )
    # Calculate object statistics
    object_ids = tuple(
        int(label_id) for label_id in np.unique(labels) if int(label_id) > 0
    )
    object_count = len(object_ids)

    if object_count > 0:
        props = regionprops(labels)
        areas = [p.area for p in props]
        centroids_y = [p.centroid[0] for p in props]
        centroids_x = [p.centroid[1] for p in props]

        mean_area = float(np.mean(areas))
        mean_centroid_x = float(np.mean(centroids_x))
        mean_centroid_y = float(np.mean(centroids_y))
    else:
        mean_area = 0.0
        mean_centroid_x = 0.0
        mean_centroid_y = 0.0

    stats = ManualObjectStats(
        slice_index=0,
        object_count=object_count,
        mean_area=mean_area,
        mean_centroid_x=mean_centroid_x,
        mean_centroid_y=mean_centroid_y,
    )

    # Return image unchanged, stats, and labels
    return (
        image,
        DataclassMeasurementColumnarRows((stats,), row_type=ManualObjectStats),
        SourceImageObjectLabelBuildRequest(
            image=image,
            labels=labels,
            declared_object_count=object_count,
            declared_object_ids=object_ids,
        ).payload(),
    )


class IdentifyObjectsManuallyModule(
    MeasurementArtifactOutputModule,
    ObjectArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "IdentifyObjectsManually"
    function_name = "identify_objects_manually"
    validated = True
    confidence = 1.0
    image_input_setting = SettingNameFamily("Select the input image")
    object_output_setting = SettingNameFamily("Name the objects to be identified")
    setting_bindings = (SettingToKeywordBinding.input(image_input_setting, ImageArtifactType),SettingToKeywordBinding.output(object_output_setting, ObjectLabelsArtifactType),)
