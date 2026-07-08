"""
Converted from CellProfiler: GaussianFilter
Original: gaussianfilter
"""

from collections.abc import Callable
from typing import Any

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.interop.cellprofiler.semantic_defaults import (
    SourceVolumetricPixelDataExecutionContract,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    parse_cellprofiler_float,
)
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
    ModuleSettingsSourceModule,
    ScopedMeasurementModule,
    StructuringElementSettingsModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)


class GaussianFilterExecutionDomainContract(SourceVolumetricPixelDataExecutionContract):
    contract_key = "GaussianFilter.execution_domain"
    source_filename = "gaussianfilter.py"
    callable_name = "gaussianfilter"

    @property
    def absorbed_callable(self) -> Callable[..., Any]:
        return gaussian_filter


class GaussianFilterModule(
    ImageArtifactInputModule, ImageArtifactOutputModule, CellProfilerModule
):
    module_name = "GaussianFilter"
    function_name = "gaussian_filter"
    validated = True
    confidence = 1.0
    image_input_settings = ("Select the input image",)
    image_output_settings = ("Name the output image",)
    semantic_default_contract_types = (GaussianFilterExecutionDomainContract,)
    semantic_default_contract_module_name = "GaussianFilter"
    setting_bindings = (
        SettingToKeywordBinding("Sigma", "sigma", parse_cellprofiler_float),
    )


import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
)


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.FLEXIBLE)
def gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply CellProfiler-compatible Gaussian smoothing to an image.

    CellProfiler divides the user sigma by image voxel spacing before invoking
    the library filter, so volumetric source metadata must stay on the payload.
    """
    from skimage.filters import gaussian as skimage_gaussian

    pixel_data = np.asarray(image_payload_data(image))
    spacing = image_payload_metadata(image).source_voxel_spacing.spacing_for_ndim(
        pixel_data.ndim
    )
    effective_sigma = np.divide(float(sigma), np.asarray(spacing, dtype=np.float64))
    filtered = skimage_gaussian(pixel_data, sigma=effective_sigma)
    return with_image_payload_data(image, filtered)
