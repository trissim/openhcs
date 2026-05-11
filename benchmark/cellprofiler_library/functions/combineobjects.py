"""
Converted from CellProfiler: CombineObjects
Original: combineobjects
"""

from typing import Tuple

import numpy as np

from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.interop.cellprofiler.image_module_settings import (
    CombineObjectsMethod as CombineMethod,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    CombineObjectsStats,
    CombineObjectsStrategy,
)
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois


@numpy
@special_outputs(
    (
        "combine_stats",
        csv_materializer(
            fields=[
                "slice_index",
                "method",
                "input_objects_x",
                "input_objects_y",
                "output_objects",
            ],
            analysis_type="combine_objects",
        ),
    ),
    ("labels", segmentation_mask_rois()),
)
def combineobjects(
    image: np.ndarray,
    method: CombineMethod | str = CombineMethod.MERGE,
) -> Tuple[np.ndarray, CombineObjectsStats, np.ndarray]:
    """Combine objects from two label images using CellProfiler policies."""
    labels_x = object_label_dense_array(image[0], dtype=np.int32)
    labels_y = object_label_dense_array(image[1], dtype=np.int32)
    stats, combined_labels = CombineObjectsStrategy.for_method(method).result(
        labels_x,
        labels_y,
    )
    return labels_x.astype(np.float32), stats, combined_labels


__all__ = [
    "CombineMethod",
    "CombineObjectsStats",
    "CombineObjectsStrategy",
    "combineobjects",
]
