"""
Converted from CellProfiler: ExpandOrShrinkObjects
Original: expand_or_shrink_objects
"""

import numpy as np

from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.runtime_semantics import ExplicitObjectLabelDomainDeclaration
from openhcs.core.runtime_values import (
    ObjectLabelPayload,
    object_label_dense_array,
    object_label_payload_with_dense_labels,
)
from openhcs.interop.cellprofiler.expand_or_shrink_settings import ExpandShrinkMode
from openhcs.processing.backends.cellprofiler.morphology import (
    ExpandShrinkOperationStrategy,
    prepare_expand_or_shrink_objects,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import segmentation_mask_rois


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("labels", segmentation_mask_rois()))
def expand_or_shrink_objects(
    image: np.ndarray,
    labels: np.ndarray | ObjectLabelPayload,
    mode: ExpandShrinkMode | str = ExpandShrinkMode.EXPAND_DEFINED_PIXELS,
    iterations: int = 1,
    fill_holes: bool = True,
) -> tuple:
    """
    Expand or shrink labeled objects using CellProfiler-compatible semantics.

    The benchmark layer owns only the OpenHCS function contract; the operation
    registry and label-domain semantics live in the CellProfiler morphology
    backend.
    """
    labels_int = object_label_dense_array(labels, dtype=np.int32)

    operation = ExpandShrinkOperationStrategy.for_mode(mode)
    result_labels = operation.apply(
        labels_int,
        iterations=iterations,
        fill_holes=fill_holes,
    )

    return image, object_label_payload_with_dense_labels(
        labels,
        result_labels.astype(np.int32, copy=False),
        domain_declaration=ExplicitObjectLabelDomainDeclaration(
            operation.output_domain(result_labels)
        ),
    )


expand_or_shrink_objects.__openhcs_prepare__ = prepare_expand_or_shrink_objects
