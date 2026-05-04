"""
Converted from CellProfiler: ColorToGray
Original: color_to_gray, split_colortogray
"""

from enum import Enum
from typing import Any

import numpy as np

from openhcs.core.image_shapes import is_color_image_slice, is_color_image_stack
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import image_payload_data, with_image_payload_data


class ImageChannelType(Enum):
    RGB = "rgb"
    HSV = "hsv"
    CHANNELS = "channels"


class ColorToGrayMode(Enum):
    COMBINE = "combine"
    SPLIT = "split"


@numpy
def color_to_gray(
    image: np.ndarray,
    mode: ColorToGrayMode | str = ColorToGrayMode.SPLIT,
    image_type: ImageChannelType | str = ImageChannelType.RGB,
    channel_indices: tuple[int, ...] = (0, 1, 2),
    contributions: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> np.ndarray | tuple[np.ndarray, ...]:
    """
    Convert an OpenHCS color payload into one or more grayscale images.

    CellProfiler ColorToGray consumes one HWC color image per image set. OpenHCS
    may carry a singleton or multi-file stack as NHWC; this function preserves
    that outer stack axis while applying CellProfiler's channel semantics.
    """

    resolved_mode = _coerce_enum(ColorToGrayMode, mode, "mode")
    resolved_image_type = _coerce_enum(ImageChannelType, image_type, "image_type")
    if resolved_mode is ColorToGrayMode.COMBINE:
        output = _combine_colortogray(image, channel_indices, contributions)
        return with_image_payload_data(image, output)
    return tuple(
        with_image_payload_data(image, output)
        for output in _split_colortogray(image, resolved_image_type, channel_indices)
    )


def _combine_colortogray(
    image: np.ndarray,
    channel_indices: tuple[int, ...],
    contributions: tuple[float, ...],
) -> np.ndarray:
    image_data = image_payload_data(image)
    if len(channel_indices) != len(contributions):
        raise ValueError("channel_indices and contributions must have same length.")
    weights = _normalized_weights(contributions)
    color_stack = _as_nhwc_color_stack(image_data)
    result = np.zeros(color_stack.shape[:3], dtype=np.float32)
    for channel_index, weight in zip(channel_indices, weights, strict=True):
        if channel_index >= color_stack.shape[-1]:
            raise ValueError(
                f"ColorToGray channel index {channel_index} is outside payload "
                f"with {color_stack.shape[-1]} channels."
            )
        result += color_stack[..., channel_index].astype(np.float32) * weight
    return _restore_singleton_slice_shape(image_data, result)


def _split_colortogray(
    image: np.ndarray,
    image_type: ImageChannelType,
    channel_indices: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    image_data = image_payload_data(image)
    color_stack = _as_nhwc_color_stack(image_data).astype(np.float32)
    source_stack = (
        _rgb_to_hsv(color_stack)
        if image_type is ImageChannelType.HSV
        else color_stack
    )
    return tuple(
        _restore_singleton_slice_shape(image_data, _channel(source_stack, index))
        for index in channel_indices
    )


def _channel(color_stack: np.ndarray, channel_index: int) -> np.ndarray:
    if channel_index >= color_stack.shape[-1]:
        raise ValueError(
            f"ColorToGray channel index {channel_index} is outside payload "
            f"with {color_stack.shape[-1]} channels."
        )
    return color_stack[..., channel_index]


def _as_nhwc_color_stack(image: np.ndarray) -> np.ndarray:
    if is_color_image_stack(image):
        return image
    if is_color_image_slice(image):
        return image[np.newaxis, ...]
    raise ValueError(
        "ColorToGray requires an OpenHCS color image shaped (H, W, C) or "
        f"(N, H, W, C), got {getattr(image, 'shape', 'unknown')}."
    )


def _restore_singleton_slice_shape(
    original: np.ndarray,
    stack: np.ndarray,
) -> np.ndarray:
    if is_color_image_slice(original):
        return stack[0]
    return stack


def _normalized_weights(contributions: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(contributions)
    if total == 0:
        raise ValueError("Contributions cannot all be zero.")
    return tuple(float(contribution) / total for contribution in contributions)


def _rgb_to_hsv(rgb_stack: np.ndarray) -> np.ndarray:
    if rgb_stack.shape[-1] < 3:
        raise ValueError("HSV conversion requires at least three RGB channels.")
    rgb = rgb_stack[..., :3]
    if rgb.size and np.nanmax(rgb) > 1.0:
        rgb = rgb / 255.0
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    max_channel = np.maximum(np.maximum(red, green), blue)
    min_channel = np.minimum(np.minimum(red, green), blue)
    delta = max_channel - min_channel
    value = max_channel
    saturation = np.divide(
        delta,
        max_channel,
        out=np.zeros_like(delta),
        where=max_channel != 0,
    )
    hue = np.zeros_like(red)
    nonzero_delta = delta != 0
    red_is_max = (max_channel == red) & nonzero_delta
    green_is_max = (max_channel == green) & nonzero_delta
    blue_is_max = (max_channel == blue) & nonzero_delta
    hue[red_is_max] = ((green[red_is_max] - blue[red_is_max]) / delta[red_is_max]) % 6
    hue[green_is_max] = (
        (blue[green_is_max] - red[green_is_max]) / delta[green_is_max]
    ) + 2
    hue[blue_is_max] = (
        (red[blue_is_max] - green[blue_is_max]) / delta[blue_is_max]
    ) + 4
    hue = hue / 6.0
    return np.stack((hue, saturation, value), axis=-1).astype(np.float32)


def _coerce_enum[T: Enum](
    enum_type: type[T],
    value: T | str,
    parameter_name: str,
) -> T:
    if isinstance(value, enum_type):
        return value
    normalized = str(value).strip().lower()
    for option in enum_type:
        if normalized in {option.name.lower(), str(option.value).lower()}:
            return option
    raise ValueError(f"Unsupported ColorToGray {parameter_name}: {value!r}")
