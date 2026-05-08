"""Converted from CellProfiler: Resize."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

from ._enum import _coerce_function_enum


class ResizeMethod(Enum):
    BY_FACTOR = "by_factor"
    TO_SIZE = "to_size"


class InterpolationMethod(Enum):
    NEAREST_NEIGHBOR = "nearest_neighbor"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


@dataclass(frozen=True, slots=True)
class ResizeGeometry:
    """CellProfiler resize geometry for pixels and per-pixel validity masks."""

    output_shape: tuple[int, ...]
    interpolation_order: int

    @classmethod
    def from_parameters(
        cls,
        input_shape: tuple[int, ...],
        *,
        resize_method: ResizeMethod,
        resizing_factors: tuple[float, ...],
        specific_shape: tuple[int, ...],
        interpolation: InterpolationMethod,
    ) -> "ResizeGeometry":
        if resize_method is ResizeMethod.BY_FACTOR:
            output_shape = tuple(
                int(np.round(axis_size * factor))
                for axis_size, factor in zip(input_shape, resizing_factors, strict=True)
            )
        else:
            output_shape = specific_shape
        return cls(
            output_shape=output_shape,
            interpolation_order=cls.resolve_interpolation_order(interpolation),
        )

    @staticmethod
    def resolve_interpolation_order(interpolation: InterpolationMethod) -> int:
        if interpolation is InterpolationMethod.NEAREST_NEIGHBOR:
            return 0
        if interpolation is InterpolationMethod.BILINEAR:
            return 1
        if interpolation is InterpolationMethod.BICUBIC:
            return 3
        raise TypeError(f"Unsupported Resize interpolation {interpolation!r}.")

    def resize_pixels(self, pixels: Any) -> np.ndarray:
        import skimage.transform

        return skimage.transform.resize(
            pixels,
            self.output_shape,
            order=self.interpolation_order,
            mode="symmetric",
            preserve_range=True,
        ).astype(np.asarray(pixels).dtype, copy=False)

    def resize_mask(self, mask: Any | None) -> np.ndarray | None:
        import scipy.ndimage as ndi

        if mask is None:
            return None
        mask_array = np.asarray(mask, dtype=bool)
        zoom = tuple(
            output_size / input_size
            for output_size, input_size in zip(
                self.output_shape,
                mask_array.shape,
                strict=True,
            )
        )
        return ndi.zoom(
            mask_array.astype(np.float32),
            zoom,
            order=0,
            mode="constant",
            grid_mode=True,
        ).astype(bool, copy=False)

    def resize_payload(self, image: Any) -> Any:
        pixels = image_payload_data(image)
        output_pixels = self.resize_pixels(pixels)
        return image_payload_with_context(
            output_pixels,
            mask=self.resize_mask(image_payload_mask(image)),
            metadata=image_payload_metadata(image).without_spatial_domain(),
        )


@numpy(contract=ProcessingContract.PURE_2D)
def resize(
    image: np.ndarray,
    resize_method: ResizeMethod = ResizeMethod.BY_FACTOR,
    resizing_factor_x: float = 0.25,
    resizing_factor_y: float = 0.25,
    specific_width: int = 100,
    specific_height: int = 100,
    interpolation: InterpolationMethod = InterpolationMethod.NEAREST_NEIGHBOR,
) -> np.ndarray:
    """
    Resize an image by a factor or to specific dimensions.
    
    Args:
        image: Input image with shape (H, W)
        resize_method: Whether to resize by factor or to specific size
        resizing_factor_x: X scaling factor (used if resize_method is BY_FACTOR)
        resizing_factor_y: Y scaling factor (used if resize_method is BY_FACTOR)
        specific_width: Target width in pixels (used if resize_method is TO_SIZE)
        specific_height: Target height in pixels (used if resize_method is TO_SIZE)
        interpolation: Interpolation method to use
        
    Returns:
        Resized image with shape (new_H, new_W)
    """
    resize_method = _coerce_function_enum(ResizeMethod, resize_method)
    interpolation = _coerce_function_enum(InterpolationMethod, interpolation)

    pixels = image_payload_data(image)
    geometry = ResizeGeometry.from_parameters(
        tuple(np.asarray(pixels).shape[:2]),
        resize_method=resize_method,
        resizing_factors=(resizing_factor_y, resizing_factor_x),
        specific_shape=(specific_height, specific_width),
        interpolation=interpolation,
    )
    return geometry.resize_payload(image)


@numpy(contract=ProcessingContract.PURE_3D)
def resize_volumetric(
    image: np.ndarray,
    resize_method: ResizeMethod = ResizeMethod.BY_FACTOR,
    resizing_factor_x: float = 0.25,
    resizing_factor_y: float = 0.25,
    resizing_factor_z: float = 0.25,
    specific_width: int = 100,
    specific_height: int = 100,
    specific_planes: int = 10,
    interpolation: InterpolationMethod = InterpolationMethod.NEAREST_NEIGHBOR,
) -> np.ndarray:
    """
    Resize a 3D volumetric image by a factor or to specific dimensions.
    
    Args:
        image: Input volumetric image with shape (D, H, W)
        resize_method: Whether to resize by factor or to specific size
        resizing_factor_x: X scaling factor (used if resize_method is BY_FACTOR)
        resizing_factor_y: Y scaling factor (used if resize_method is BY_FACTOR)
        resizing_factor_z: Z scaling factor (used if resize_method is BY_FACTOR)
        specific_width: Target width in pixels (used if resize_method is TO_SIZE)
        specific_height: Target height in pixels (used if resize_method is TO_SIZE)
        specific_planes: Target number of planes (used if resize_method is TO_SIZE)
        interpolation: Interpolation method to use
        
    Returns:
        Resized volumetric image with shape (new_D, new_H, new_W)
    """
    resize_method = _coerce_function_enum(ResizeMethod, resize_method)
    interpolation = _coerce_function_enum(InterpolationMethod, interpolation)

    pixels = image_payload_data(image)
    geometry = ResizeGeometry.from_parameters(
        tuple(np.asarray(pixels).shape[:3]),
        resize_method=resize_method,
        resizing_factors=(resizing_factor_z, resizing_factor_y, resizing_factor_x),
        specific_shape=(specific_planes, specific_height, specific_width),
        interpolation=interpolation,
    )
    return geometry.resize_payload(image)
