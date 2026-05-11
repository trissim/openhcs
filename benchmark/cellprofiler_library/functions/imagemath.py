"""
Converted from CellProfiler: ImageMath
Original: ImageMath module

Performs simple mathematical operations on image intensities.
Supports addition, subtraction, multiplication, division, averaging,
min/max, standard deviation, inversion, log transform, and logical operations.
"""

import numpy as np
from typing import Tuple

from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_metadata,
    image_payload_with_context,
)
from openhcs.interop.cellprofiler.image_math_settings import (
    ImageMathOperation as MathOperation,
)
from openhcs.processing.backends.cellprofiler.image_math import (
    ImageMathMaskPolicy,
    ImageMathOperationStrategy,
)


@numpy
def image_math(
    image: np.ndarray,
    operation: MathOperation = MathOperation.ADD,
    factors: Tuple[float, ...] = (1.0, 1.0),
    exponent: float = 1.0,
    after_factor: float = 1.0,
    addend: float = 0.0,
    truncate_low: bool = True,
    truncate_high: bool = True,
    replace_nan: bool = True,
    ignore_masks: bool = False,
) -> np.ndarray:
    """
    Perform mathematical operations on image intensities.
    
    Args:
        image: Input array of shape (N, H, W) where N images are stacked along dim 0.
               For single-image operations (INVERT, LOG_TRANSFORM, NOT, NONE),
               only the first slice is used.
               For multi-image operations, all N slices are combined.
        operation: The mathematical operation to perform.
        factors: Tuple of multiplication factors for each input image (applied before operation).
        exponent: Raise the result to this power (after operation).
        after_factor: Multiply the result by this value (after operation).
        addend: Add this value to the result (after operation).
        truncate_low: Set values less than 0 to 0.
        truncate_high: Set values greater than 1 to 1.
        replace_nan: Replace NaN values with 0.
        ignore_masks: Drop any input image mask instead of preserving it.
    
    Returns:
        Processed image of shape (1, H, W).
    """
    operation_strategy = ImageMathOperationStrategy.coerce(operation)
    mask_policy = ImageMathMaskPolicy(ignore_masks=ignore_masks)
    source_payload = image
    image = image_payload_data(image)
    
    # Handle input dimensions
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    
    n_images = image.shape[0]
    
    # Extend factors if needed
    if len(factors) < n_images:
        factors = tuple(factors) + (1.0,) * (n_images - len(factors))
    
    # Apply factors to each image (except for binary output operations)
    pixel_data = []
    operand_count = 1 if operation_strategy.single_image else n_images
    operand_masks = mask_policy.operand_masks(source_payload, operand_count)
    for i in range(operand_count):
        pd = image[i].astype(np.float64)
        if not operation_strategy.binary_output and factors[i] != 1.0:
            pd = pd * factors[i]
        pixel_data.append(pd)
    output_pixel_data = operation_strategy.apply(
        operation_strategy.prepare_initial_output(image, pixel_data, factors),
        pixel_data,
        factors,
    )
    
    # Post-processing (not for binary output operations)
    if not operation_strategy.binary_output:
        if exponent != 1.0:
            output_pixel_data = output_pixel_data ** exponent
        if after_factor != 1.0:
            output_pixel_data = output_pixel_data * after_factor
        if addend != 0.0:
            output_pixel_data = output_pixel_data + addend
        
        # Truncation
        if truncate_low:
            output_pixel_data[output_pixel_data < 0] = 0
        if truncate_high:
            output_pixel_data[output_pixel_data > 1] = 1
        if replace_nan:
            output_pixel_data[np.isnan(output_pixel_data)] = 0
    
    # Ensure output is (1, H, W)
    if output_pixel_data.ndim == 2:
        output_pixel_data = output_pixel_data[np.newaxis, :, :]

    output_mask = mask_policy.output_mask(operand_masks)
    output_pixel_data = mask_policy.apply_output_mask(output_pixel_data, output_mask)

    output = output_pixel_data.astype(np.float32)
    if ignore_masks:
        return output
    return image_payload_with_context(
        output,
        mask=output_mask,
        metadata=image_payload_metadata(source_payload).without_unit_interval_intensity_scale(),
    )
