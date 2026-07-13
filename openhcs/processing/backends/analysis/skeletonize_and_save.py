"""Per-slice skeletonization with tabular measurements and ROI output."""

from dataclasses import asdict, dataclass, fields
from typing import List, Optional, Tuple

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects, skeletonize

from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import (
    CsvOptions,
    MaterializationSpec,
    ROIOptions,
)


@dataclass(frozen=True)
class SkeletonizationResult:
    """Canonical per-slice measurements for skeleton analysis."""

    slice_index: int
    skeleton_count: int
    skeleton_length_pixels: int
    foreground_area_pixels: int
    threshold: float

    @classmethod
    def csv_fields(cls) -> List[str]:
        """Return CSV field order from the dataclass declaration."""

        return [field.name for field in fields(cls)]


@numpy
@special_outputs(
    (
        "skeleton_measurements",
        MaterializationSpec(CsvOptions(fields=SkeletonizationResult.csv_fields())),
    ),
    ("skeleton_rois", MaterializationSpec(ROIOptions())),
)
def skeletonize_and_save(
    image,
    threshold: Optional[float] = None,
    min_component_size: int = 1,
) -> Tuple[np.ndarray, List[dict], List[np.ndarray]]:
    """Skeletonize each image plane and emit CSV measurements and labeled ROIs.

    Each plane along the first axis is thresholded independently. When
    ``threshold`` is ``None``, Otsu's method is used for that plane and falls
    back to its mean intensity if Otsu cannot determine a threshold. Connected
    skeleton components smaller than ``min_component_size`` pixels are removed.

    The ``skeleton_measurements`` special output is materialized as CSV and is
    automatically eligible for OpenHCS plate-level MetaXpress-style analysis
    consolidation. The ``skeleton_rois`` output is materialized as labeled ROI
    masks using the standard ROI writer.

    Args:
        image: A 3D array with shape ``(Z, Y, X)`` or ``(C, Y, X)``. The first
            axis is treated as independent image planes. The original object is
            returned unchanged as the primary output.
        threshold: Optional global intensity threshold applied to every plane.
            When omitted, a threshold is calculated independently per plane.
        min_component_size: Minimum connected skeleton length, in pixels, to
            retain. Must be at least ``1``.

    Returns:
        A tuple containing the unchanged input image, per-slice measurement
        dictionaries, and one sequentially labeled ``int32`` skeleton mask per
        input plane.

    Raises:
        ValueError: If ``image`` is not three-dimensional, has empty spatial
            dimensions, or ``min_component_size`` is less than ``1``.
    """

    image_array = np.asarray(image)
    if image_array.ndim != 3:
        raise ValueError(
            "skeletonize_and_save expects a 3D array with shape (Z, Y, X) "
            "or (C, Y, X)"
        )
    if image_array.shape[1] == 0 or image_array.shape[2] == 0:
        raise ValueError("skeletonize_and_save requires non-empty image planes")
    if min_component_size < 1:
        raise ValueError("min_component_size must be at least 1")

    results: List[dict] = []
    masks: List[np.ndarray] = []

    for slice_index, slice_2d in enumerate(image_array):
        if threshold is None:
            try:
                slice_threshold = float(threshold_otsu(slice_2d))
            except (TypeError, ValueError):
                slice_threshold = float(np.mean(slice_2d))
        else:
            slice_threshold = float(threshold)

        binary = slice_2d > slice_threshold
        skeleton = skeletonize(binary)
        if min_component_size > 1:
            skeleton = remove_small_objects(
                skeleton.astype(bool, copy=False),
                min_size=min_component_size,
            )

        labeled_skeleton = label(skeleton).astype(np.int32, copy=False)
        result = SkeletonizationResult(
            slice_index=slice_index,
            skeleton_count=int(labeled_skeleton.max()),
            skeleton_length_pixels=int(np.count_nonzero(labeled_skeleton)),
            foreground_area_pixels=int(np.count_nonzero(binary)),
            threshold=slice_threshold,
        )
        results.append(asdict(result))
        masks.append(labeled_skeleton)

    return image, results, masks
