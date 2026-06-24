"""CellProfiler image-payload collapse policies."""

from __future__ import annotations

import numpy as np

from openhcs.core.image_shapes import is_color_image_stack
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerRuntimeValue,
)

class SingletonStackOutputCollapsePolicy:
    """Collapse single-plane stack outputs while preserving payload context."""

    def collapse(self, value: CellProfilerRuntimeValue) -> CellProfilerRuntimeValue:
        metadata = image_payload_metadata(value)
        mask = image_payload_mask(value)
        if mask is not None or metadata.has_values:
            data = image_payload_data(value)
            collapsed_data = self.collapse(data)
            collapsed = collapsed_data is not data
            return (metadata.for_source_plane(0) if collapsed else metadata).payload_with(
                collapsed_data,
                mask=None if mask is None else self.collapse_mask(mask),
            )
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[0] == 1:
            return value[0]
        if isinstance(value, np.ndarray) and value.ndim >= 4 and value.shape[0] == 1:
            return value[0]
        if is_color_image_stack(value) and value.shape[0] == 1:
            return value[0]
        if isinstance(value, tuple):
            return tuple(self.collapse(item) for item in value)
        return value

    def collapse_mask(self, mask: CellProfilerRuntimeValue) -> CellProfilerRuntimeValue:
        """Collapse a singleton mask stack in the same domain as image data."""
        if isinstance(mask, np.ndarray) and mask.ndim == 3 and mask.shape[0] == 1:
            return mask[0]
        if isinstance(mask, np.ndarray) and mask.ndim == 4 and mask.shape[0] == 1:
            return mask[0]
        return mask


SINGLETON_STACK_OUTPUT_COLLAPSE = SingletonStackOutputCollapsePolicy()
