"""
Converted from CellProfiler: CorrectIlluminationApply
Original: correct_illumination_apply
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.memory.decorators import numpy

from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum


class IlluminationCorrectionMethod(Enum):
    DIVIDE = "divide"
    SUBTRACT = "subtract"


@dataclass(frozen=True, slots=True)
class IlluminationCorrectionRequest:
    image_pixels: np.ndarray
    illumination_function: np.ndarray


class IlluminationCorrectionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal correction implementation for one CellProfiler method."""

    __registry_key__ = "method"
    __skip_if_no_key__ = True
    method: ClassVar[IlluminationCorrectionMethod | None] = None

    @classmethod
    def for_method(
        cls,
        method: IlluminationCorrectionMethod,
    ) -> "IlluminationCorrectionStrategy":
        return cls.__registry__[method]()

    @abstractmethod
    def apply(self, request: IlluminationCorrectionRequest) -> np.ndarray:
        """Apply the correction method."""


class DivideIlluminationCorrectionStrategy(IlluminationCorrectionStrategy):
    method = IlluminationCorrectionMethod.DIVIDE

    def apply(self, request: IlluminationCorrectionRequest) -> np.ndarray:
        safe_illumination = np.where(
            request.illumination_function == 0,
            1e-10,
            request.illumination_function,
        )
        return request.image_pixels / safe_illumination


class SubtractIlluminationCorrectionStrategy(IlluminationCorrectionStrategy):
    method = IlluminationCorrectionMethod.SUBTRACT

    def apply(self, request: IlluminationCorrectionRequest) -> np.ndarray:
        return request.image_pixels - request.illumination_function


@numpy
def correct_illumination_apply(
    image: np.ndarray,
    method: IlluminationCorrectionMethod | str = IlluminationCorrectionMethod.DIVIDE,
    truncate_low: bool = True,
    truncate_high: bool = True,
) -> np.ndarray:
    """
    Apply illumination correction to an image using a provided illumination function.
    
    This function corrects uneven illumination by either dividing or subtracting
    an illumination function from the input image.
    
    Args:
        image: Shape (2, H, W) - two images stacked:
               image[0] = image to correct
               image[1] = illumination function
        method: Method to apply correction - DIVIDE or SUBTRACT
        truncate_low: Set output values less than 0 equal to 0
        truncate_high: Set output values greater than 1 equal to 1
    
    Returns:
        Corrected image with shape (1, H, W)
    """
    method = _coerce_function_enum(IlluminationCorrectionMethod, method)

    image_pixels = image[0]
    illumination_function = image[1]
    assert image_pixels.shape == illumination_function.shape, (
        f"Input image shape {image_pixels.shape} and illumination function shape "
        f"{illumination_function.shape} must be equal"
    )

    output_pixels = IlluminationCorrectionStrategy.for_method(method).apply(
        IlluminationCorrectionRequest(
            image_pixels=image_pixels,
            illumination_function=illumination_function,
        )
    )

    # Optionally clip values
    if truncate_low:
        output_pixels = np.maximum(output_pixels, 0.0)
    if truncate_high:
        output_pixels = np.minimum(output_pixels, 1.0)
    
    # Return with shape (1, H, W) to maintain 3D convention
    return output_pixels[np.newaxis, ...].astype(np.float32)
