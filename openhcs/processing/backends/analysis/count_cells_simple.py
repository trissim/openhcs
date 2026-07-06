"""
Simple cell counting using thresholding and connected component labeling.

This module provides a compact NumPy/scikit-image cell counter intended for
basic bright- or dark-object workflows. The implementation deliberately keeps
the segmentation model simple: threshold each 2D slice, label connected
components, optionally split oversized components, then apply geometric filters
to the final candidate objects.
"""

from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import MaterializationSpec, CsvOptions, ROIOptions

from enum import Enum
from typing import Tuple, List

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, threshold_li, threshold_yen
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential, watershed


class ThresholdMethod(str, Enum):
    """Thresholding methods for cell detection."""
    OTSU = "otsu"
    LI = "li"
    YEN = "yen"
    PERCENTILE = "percentile"
    MANUAL = "manual"


class Foreground(str, Enum):
    """Foreground type for thresholding."""
    BRIGHT = "bright"
    DARK = "dark"


# Make the Enums importable/stable for multiprocessing/ZMQ pickling
import openhcs.processing.backends.analysis.count_cells_simple as _count_cells_simple

ThresholdMethod.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "ThresholdMethod", ThresholdMethod)

Foreground.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "Foreground", Foreground)


@numpy
@special_outputs(
    (
        "cell_counts",
        MaterializationSpec(CsvOptions(fields=["slice_index", "cell_count"]))
    ),
    ("segmentation_masks", MaterializationSpec(ROIOptions()))
)
def count_cells_simple(
    image,
    threshold_method: ThresholdMethod = ThresholdMethod.OTSU,
    threshold: float = 0.5,
    threshold_percentile: float = 99.0,
    foreground: Foreground = Foreground.BRIGHT,
    min_size: int = 20,
    max_size: int = 100000,
    max_eccentricity: float = 1.0,
    watershed_large_objects: bool = False,
    watershed_min_distance: int = 5,
    watershed_footprint_size: int = 3
) -> Tuple[np.ndarray, List[dict], List[np.ndarray]]:
    """
    Count thresholded objects in a 3D image stack with optional shape cleanup.

    The function processes each 2D plane of ``image`` independently. For every
    slice it computes a threshold, builds a binary foreground mask, labels
    connected components, optionally applies distance-transform watershed only
    to components larger than ``max_size``, and then applies the final object
    acceptance filters. The reported count and ROI mask therefore describe the
    final accepted connected components after all splitting and filtering.

    Filtering order is intentional:

    1. ``min_size`` is used only in the final acceptance pass.
    2. If ``watershed_large_objects`` is enabled, only raw connected components
       whose area is greater than ``max_size`` are split.
    3. ``min_size``, ``max_size``, and ``max_eccentricity`` are then applied to
       the resulting candidate regions.

    This means a large merged component can be split into multiple accepted
    cells, as long as each watershed fragment lands within the size and shape
    limits. If watershed cannot find at least two seeds for a large object, that
    object remains a single candidate and is usually rejected by ``max_size``.

    Args:
        image: Input 3D image stack with shape ``(Z, Y, X)``. OpenHCS also uses
            this shape for logical single-plane data, so a single 2D image
            should be supplied as ``image[None, :, :]``. The input image is
            returned unchanged as the primary output.
        threshold_method: Thresholding strategy used to create the foreground
            mask for each slice. ``OTSU``, ``LI``, and ``YEN`` compute an
            automatic threshold from the slice. ``PERCENTILE`` uses
            ``threshold_percentile``. ``MANUAL`` uses ``threshold`` directly,
            with fractional values in ``[0, 1]`` interpreted as a fraction of
            the slice maximum when the image intensity range is larger than
            ``[0, 1]``.
        threshold: Manual threshold value used only when
            ``threshold_method=ThresholdMethod.MANUAL``. Values greater than
            ``1`` are treated as native image-intensity values. Values in
            ``[0, 1]`` are treated as normalized fractions for scaled integer or
            high-dynamic-range arrays.
        threshold_percentile: Percentile in ``[0, 100]`` used only when
            ``threshold_method=ThresholdMethod.PERCENTILE``. For bright
            foregrounds, pixels above this percentile become foreground; for
            dark foregrounds, pixels below it become foreground.
        foreground: Polarity of the objects to count. ``BRIGHT`` counts pixels
            above the computed threshold. ``DARK`` counts pixels below the
            computed threshold.
        min_size: Minimum accepted object area in pixels after optional
            watershed splitting. Objects smaller than this are removed from the
            output mask and not counted.
        max_size: Maximum accepted object area in pixels after optional
            watershed splitting. Objects larger than this are treated as
            artifacts or unresolved merged objects and are removed from the
            output mask. When ``watershed_large_objects=True``, this same value
            also defines which raw connected components are considered large
            enough to attempt splitting.
        max_eccentricity: Maximum accepted object eccentricity after optional
            watershed splitting and size filtering. Eccentricity is the
            scikit-image region shape measure where ``0.0`` is circular and
            ``1.0`` is nearly line-like. The default ``1.0`` preserves the
            previous behavior and effectively disables shape filtering. Lower
            values reject elongated debris, scratches, neurites, or merged
            streak-like objects that otherwise satisfy the size filter.
        watershed_large_objects: If ``True``, apply distance-transform watershed
            only to connected components whose raw area is greater than
            ``max_size``. This is useful when touching cells merge into one
            oversized component that would otherwise be rejected by the size
            filter. It is disabled by default to preserve the original simple
            connected-component behavior.
        watershed_min_distance: Minimum pixel spacing between local maxima used
            as watershed seeds for oversized objects. Larger values produce
            fewer, more conservative splits; smaller values can split dense or
            noisy objects more aggressively.
        watershed_footprint_size: Side length, in pixels, of the square
            footprint used when finding local maxima for watershed seeds.
            Larger footprints suppress nearby competing maxima and reduce
            over-splitting; smaller footprints allow more seed points.

    Returns:
        Tuple:
          - ``image`` unchanged, preserving OpenHCS primary-output semantics.
          - A list of per-slice dictionaries with ``slice_index`` and
            ``cell_count`` fields. The count is the number of accepted labels
            after optional splitting and all filters.
          - A list of labeled ``int32`` segmentation masks, one per input
            slice. Background is ``0``. Accepted objects are relabeled
            sequentially from ``1`` within each slice.

    Raises:
        ValueError: If size limits, eccentricity limits, or watershed seed
            parameters are outside their supported ranges.
    """
    if min_size < 0:
        raise ValueError("min_size must be >= 0")
    if max_size < min_size:
        raise ValueError("max_size must be >= min_size")
    if not 0.0 <= max_eccentricity <= 1.0:
        raise ValueError("max_eccentricity must be in [0.0, 1.0]")
    if watershed_min_distance < 1:
        raise ValueError("watershed_min_distance must be >= 1")
    if watershed_footprint_size < 1:
        raise ValueError("watershed_footprint_size must be >= 1")

    results = []
    masks = []

    for i, slice_data in enumerate(image):
        # Convert enum strings to enum objects if needed
        if isinstance(threshold_method, str):
            threshold_method = ThresholdMethod(threshold_method)
        if isinstance(foreground, str):
            foreground = Foreground(foreground)
        
        # Compute threshold in the native intensity scale of slice_data
        if threshold_method == ThresholdMethod.OTSU:
            thr = threshold_otsu(slice_data)
        elif threshold_method == ThresholdMethod.LI:
            thr = threshold_li(slice_data)
        elif threshold_method == ThresholdMethod.YEN:
            thr = threshold_yen(slice_data)
        elif threshold_method == ThresholdMethod.PERCENTILE:
            thr = float(np.percentile(slice_data, threshold_percentile))
        elif threshold_method == ThresholdMethod.MANUAL:
            # If user supplies a fractional threshold in [0,1] but the data is scaled
            # (e.g. uint16 or normalized to ~[0,65535]), interpret it as fraction-of-max.
            thr = float(threshold)
            if 0.0 <= thr <= 1.0:
                max_val = float(np.max(slice_data))
                if max_val > 1.0:
                    thr *= max_val
        else:
            raise ValueError(f"Unknown threshold_method: {threshold_method!r}")

        # Apply threshold
        if foreground == Foreground.BRIGHT:
            binary = slice_data > thr
        elif foreground == Foreground.DARK:
            binary = slice_data < thr
        else:
            raise ValueError(f"Unknown foreground: {foreground!r} (expected 'bright' or 'dark')")

        # Label connected components
        labeled, num_objects = ndi.label(binary)

        if watershed_large_objects and num_objects > 0:
            labeled = _watershed_large_objects(
                labeled,
                max_size=max_size,
                min_distance=watershed_min_distance,
                footprint_size=watershed_footprint_size,
            )

        # Filter objects by size and eccentricity.
        if num_objects > 0:
            keep_labels = []
            for region in regionprops(labeled):
                if (
                    min_size <= region.area <= max_size
                    and region.eccentricity <= max_eccentricity
                ):
                    keep_labels.append(region.label)

            labeled_filtered = np.where(np.isin(labeled, keep_labels), labeled, 0)
            labeled_filtered, _, _ = relabel_sequential(labeled_filtered)
            final_count = len(keep_labels)
        else:
            labeled_filtered = np.zeros_like(labeled)
            final_count = 0

        results.append({
            "slice_index": i,
            "cell_count": int(final_count)
        })

        masks.append(labeled_filtered.astype(np.int32, copy=False))

    return image, results, masks


def _watershed_large_objects(
    labeled: np.ndarray,
    max_size: int,
    min_distance: int,
    footprint_size: int,
) -> np.ndarray:
    """Split connected components above max_size using distance watershed."""
    output = np.zeros_like(labeled, dtype=np.int32)
    next_label = 1
    footprint = np.ones((footprint_size, footprint_size), dtype=bool)

    for region in regionprops(labeled):
        component = labeled == region.label

        if region.area <= max_size:
            output[component] = next_label
            next_label += 1
            continue

        distance = ndi.distance_transform_edt(component)
        seeds = peak_local_max(
            distance,
            min_distance=min_distance,
            footprint=footprint,
            labels=component,
        )

        if len(seeds) <= 1:
            output[component] = next_label
            next_label += 1
            continue

        markers = np.zeros_like(labeled, dtype=np.int32)
        markers[seeds[:, 0], seeds[:, 1]] = np.arange(1, len(seeds) + 1)
        split_labels = watershed(-distance, markers, mask=component)

        for split_region in regionprops(split_labels):
            output[split_labels == split_region.label] = next_label
            next_label += 1

    return output
