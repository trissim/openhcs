"""
Converted from CellProfiler: GaussianFilter
Original: gaussianfilter
"""

from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    parse_cellprofiler_float,
)

from openhcs.processing.backends.cellprofiler.module_classes import (
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
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
from openhcs.interop.cellprofiler.cellprofiler_literals import cellprofiler_enum_from_literal

class GaussianFilterModule(CellProfilerModule):
    module_name = 'GaussianFilter'
    function_name = 'gaussian_filter'
    validated = True
    contract = 'unknown'
    confidence = 1.0
    setting_bindings = (
        SettingToKeywordBinding("Sigma", "sigma", parse_cellprofiler_float),
    )



import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
)


@numpy(contract=ProcessingContract.PURE_2D)
def gaussian_filter(
    image: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply Gaussian smoothing filter to an image.
    
    Args:
        image: Input image array with shape (H, W)
        sigma: Standard deviation for Gaussian kernel. Higher values produce
               more smoothing. Default is 1.0.
    
    Returns:
        Smoothed image with same shape as input.
    """
    from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    
    return scipy_gaussian_filter(image, sigma=sigma)
