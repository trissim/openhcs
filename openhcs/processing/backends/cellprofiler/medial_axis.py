"""
Converted from CellProfiler: Medialaxis
Original: medialaxis
"""

import numpy as np
from openhcs.core.artifacts import ImageArtifactType
from openhcs.core.memory.decorators import numpy as numpy_backend
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.settings_binder import SettingToKeywordBinding
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@numpy_backend(contract=ProcessingContract.PURE_2D)
def medialaxis(image: np.ndarray) -> np.ndarray:
    """
    Compute the medial axis (skeleton) of a binary image.

    The medial axis is the set of all points having more than one closest
    point on the object's boundary. It provides a thin representation of
    the shape that preserves topology.

    Args:
        image: Input binary image of shape (H, W). Non-zero values are
               treated as foreground.

    Returns:
        Binary image of shape (H, W) containing the medial axis skeleton.
    """
    from skimage.morphology import medial_axis as skimage_medial_axis

    binary = image > 0
    skeleton = skimage_medial_axis(binary)
    return skeleton.astype(np.float32)


class MedialaxisModule(CellProfilerModule):
    module_name = "Medialaxis"
    function_name = "medialaxis"
    validated = True
    confidence = 1.0
    setting_bindings = (
        SettingToKeywordBinding.input(
            "Select the input image",
            ImageArtifactType,
        ),
        SettingToKeywordBinding.output(
            "Name the output image",
            ImageArtifactType,
        ),
    )
