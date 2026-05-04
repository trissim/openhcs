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
from openhcs.core.runtime_values import (
    image_payload_metadata,
    image_payload_data,
    image_payload_mask,
    image_payload_with_context,
)

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

    __registry_key__ = "method_label"
    __skip_if_no_key__ = True
    method_label: ClassVar[str | None] = None
    method: ClassVar[IlluminationCorrectionMethod | None] = None

    @classmethod
    def for_method(
        cls,
        method: IlluminationCorrectionMethod,
    ) -> "IlluminationCorrectionStrategy":
        return cls.__registry__[method.value]()

    @abstractmethod
    def apply(self, request: IlluminationCorrectionRequest) -> np.ndarray:
        """Apply the correction method."""


class DivideIlluminationCorrectionStrategy(IlluminationCorrectionStrategy):
    method = IlluminationCorrectionMethod.DIVIDE
    method_label = method.value

    def apply(self, request: IlluminationCorrectionRequest) -> np.ndarray:
        safe_illumination = np.where(
            request.illumination_function == 0,
            1e-10,
            request.illumination_function,
        )
        return request.image_pixels / safe_illumination


class SubtractIlluminationCorrectionStrategy(IlluminationCorrectionStrategy):
    method = IlluminationCorrectionMethod.SUBTRACT
    method_label = method.value

    def apply(self, request: IlluminationCorrectionRequest) -> np.ndarray:
        return request.image_pixels - request.illumination_function


@numpy
def correct_illumination_apply(
    image: np.ndarray,
    method: (
        IlluminationCorrectionMethod
        | str
        | tuple[IlluminationCorrectionMethod | str, ...]
    ) = IlluminationCorrectionMethod.DIVIDE,
    truncate_low: bool | tuple[bool, ...] = True,
    truncate_high: bool | tuple[bool, ...] = True,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """
    Apply illumination correction to an image using a provided illumination function.
    
    This function corrects uneven illumination by either dividing or subtracting
    an illumination function from the input image.
    
    Args:
        image: Shape (2*N, H, W) - image/function pairs stacked:
               image[2*i] = image to correct
               image[2*i+1] = illumination function
        method: Method to apply correction - DIVIDE or SUBTRACT
        truncate_low: Set output values less than 0 equal to 0
        truncate_high: Set output values greater than 1 equal to 1
    
    Returns:
        Corrected image with shape (1, H, W)
    """
    pixel_stack = np.asarray(image_payload_data(image))
    if pixel_stack.ndim < 3 or pixel_stack.shape[0] % 2 != 0:
        raise ValueError(
            "CorrectIlluminationApply requires stacked image/function pairs "
            f"with shape (2*N, ...), got {pixel_stack.shape!r}."
        )
    pair_count = int(pixel_stack.shape[0] // 2)
    methods = _repeat_illumination_methods(method, pair_count)
    truncate_low_values = _repeat_bool_setting(
        truncate_low,
        pair_count,
        parameter_name="truncate_low",
    )
    truncate_high_values = _repeat_bool_setting(
        truncate_high,
        pair_count,
        parameter_name="truncate_high",
    )
    outputs = tuple(
        _correct_illumination_pair(
            image,
            pixel_stack,
            pair_index,
            method=methods[pair_index],
            truncate_low=truncate_low_values[pair_index],
            truncate_high=truncate_high_values[pair_index],
        )
        for pair_index in range(pair_count)
    )
    if pair_count == 1:
        return outputs[0]
    return outputs


def _correct_illumination_pair(
    image: object,
    pixel_stack: np.ndarray,
    pair_index: int,
    *,
    method: IlluminationCorrectionMethod,
    truncate_low: bool,
    truncate_high: bool,
) -> np.ndarray:
    input_index = pair_index * 2
    image_pixels = pixel_stack[input_index]
    illumination_function = pixel_stack[input_index + 1]
    if image_pixels.shape != illumination_function.shape:
        raise ValueError(
            f"Input image shape {image_pixels.shape} and illumination function "
            f"shape {illumination_function.shape} must be equal."
        )

    output_pixels = IlluminationCorrectionStrategy.for_method(method).apply(
        IlluminationCorrectionRequest(
            image_pixels=image_pixels,
            illumination_function=illumination_function,
        )
    )
    if truncate_low:
        output_pixels = np.maximum(output_pixels, 0.0)
    if truncate_high:
        output_pixels = np.minimum(output_pixels, 1.0)
    return image_payload_with_context(
        output_pixels[np.newaxis, ...].astype(np.float32),
        mask=_input_mask(image, input_index),
        metadata=image_payload_metadata(image).for_channel(input_index),
    )


def _repeat_illumination_methods(
    value: (
        IlluminationCorrectionMethod
        | str
        | tuple[IlluminationCorrectionMethod | str, ...]
    ),
    pair_count: int,
) -> tuple[IlluminationCorrectionMethod, ...]:
    if isinstance(value, tuple):
        if len(value) != pair_count:
            raise ValueError(
                "CorrectIlluminationApply method count must match image/function "
                f"pair count; got {len(value)} methods for {pair_count} pairs."
            )
        return tuple(
            _coerce_function_enum(IlluminationCorrectionMethod, method)
            for method in value
        )
    method = _coerce_function_enum(IlluminationCorrectionMethod, value)
    return (method,) * pair_count


def _repeat_bool_setting(
    value: bool | tuple[bool, ...],
    pair_count: int,
    *,
    parameter_name: str,
) -> tuple[bool, ...]:
    if isinstance(value, tuple):
        if len(value) != pair_count:
            raise ValueError(
                f"CorrectIlluminationApply {parameter_name} count must match "
                f"image/function pair count; got {len(value)} values for "
                f"{pair_count} pairs."
            )
        return tuple(bool(item) for item in value)
    return (bool(value),) * pair_count


def _input_mask(image: object, input_index: int) -> object | None:
    """Return the mask associated with one source image being corrected."""
    mask = image_payload_mask(image)
    if mask is None:
        return None
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim == 3 and mask_array.shape[0] > 0:
        return mask_array[input_index : input_index + 1]
    return mask_array
