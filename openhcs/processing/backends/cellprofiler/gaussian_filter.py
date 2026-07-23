"""
Converted from CellProfiler: GaussianFilter
Original: gaussianfilter
"""

from typing import ClassVar

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.runtime_image_values import (
    image_payload_data,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    parse_cellprofiler_float,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.core.artifacts import ImageArtifactType


class GaussianFilterModule(
    CellProfilerModule
):
    module_name = "GaussianFilter"
    function_name = "gaussian_filter"
    validated = True
    confidence = 1.0
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (SettingToKeywordBinding.input("Select the input image", ImageArtifactType),SettingToKeywordBinding.output("Name the output image", ImageArtifactType),SettingToKeywordBinding("Sigma", "sigma", parse_cellprofiler_float),)


import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.FLEXIBLE)
def gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply CellProfiler-compatible Gaussian smoothing to an image.

    CellProfiler divides the user sigma by image voxel spacing before invoking
    the library filter, so volumetric source metadata must stay on the payload."""
    from skimage.filters import gaussian as skimage_gaussian

    pixel_data = np.asarray(image_payload_data(image))
    spacing = image_payload_metadata(image).source_voxel_spacing.spacing_for_ndim(
        pixel_data.ndim
    )
    effective_sigma = np.divide(float(sigma), np.asarray(spacing, dtype=np.float64))
    filtered = skimage_gaussian(pixel_data, sigma=effective_sigma)
    return with_image_payload_data(image, filtered)
