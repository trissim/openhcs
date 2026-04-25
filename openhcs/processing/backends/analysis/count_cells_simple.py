"""
Simple cell counting using thresholding and connected component labeling.

This module provides a straightforward cell counting function with basic
thresholding methods and size filtering.
"""

from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import MaterializationSpec, CsvOptions, ROIOptions

from enum import Enum
from typing import Tuple, List

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_li, threshold_yen


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
@artifact_outputs(
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
    max_size: int = 100000
) -> Tuple[np.ndarray, List[dict], List[np.ndarray]]:
    """
    Count cells in a 3D image using simple thresholding and connected component labeling.

    Args:
        image: Input image as 3D array (C, Y, X)
        threshold_method: One of {"otsu", "li", "yen", "percentile", "manual"}
        threshold: Manual threshold value (used when threshold_method="manual")
        threshold_percentile: Percentile in [0,100] (used when threshold_method="percentile")
        foreground: "bright" (cells brighter than background) or "dark"
        min_size: Minimum pixel size to keep an object
        max_size: Maximum pixel size to keep an object (filters out large artifacts)

    Returns:
        Tuple:
          - image (unchanged)
          - list of dicts with cell count per slice
          - list of segmentation masks (one per slice)
    """
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

        # Filter objects by size (min and max)
        if num_objects > 0:
            sizes = ndi.sum(binary, labeled, index=range(1, num_objects + 1))
            keep_labels = [
                idx + 1 for idx, size in enumerate(sizes)
                if min_size <= size <= max_size
            ]

            filtered = np.isin(labeled, keep_labels)
            labeled_filtered, final_count = ndi.label(filtered)
        else:
            labeled_filtered = np.zeros_like(labeled)
            final_count = 0

        results.append({
            "slice_index": i,
            "cell_count": int(final_count)
        })

        masks.append(labeled_filtered.astype(np.int32, copy=False))

    return image, results, masks

