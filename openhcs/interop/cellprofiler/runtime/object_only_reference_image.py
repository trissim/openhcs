"""Reference-image projection for object-only CellProfiler modules."""

from __future__ import annotations

import numpy as np

from openhcs.core.aligned_image_payload import AlignedImageStack
from openhcs.core.image_shapes import is_color_image_slice, is_color_image_stack
from openhcs.core.runtime_values import image_payload_data
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerRuntimeValue


class ObjectOnlyReferenceImagePolicy:
    """Choose the single image plane used to carry object-only CP modules."""

    def reference_image(self, image: CellProfilerRuntimeValue) -> CellProfilerRuntimeValue:
        image_data = image_payload_data(image)
        if isinstance(image_data, AlignedImageStack):
            return self.reference_image(image_data.slices[0])
        if is_color_image_stack(image_data):
            return image_data[0, :, :, 0]
        if is_color_image_slice(image_data):
            return image_data[:, :, 0]
        while isinstance(image_data, np.ndarray) and image_data.ndim > 2:
            if image_data.shape[0] < 1:
                break
            image_data = image_data[0]
            if is_color_image_slice(image_data):
                return image_data[:, :, 0]
        return image_data


OBJECT_ONLY_REFERENCE_IMAGE = ObjectOnlyReferenceImagePolicy()
