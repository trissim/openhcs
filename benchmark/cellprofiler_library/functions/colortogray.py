"""Benchmark-library facade for CellProfiler ColorToGray."""

from openhcs.processing.backends.cellprofiler.color import (
    ColorToGrayMode,
    ImageChannelType,
    color_to_gray,
    color_to_gray_channel,
    combine_color_to_gray,
    nhwc_color_stack,
    normalized_color_to_gray_weights,
    restore_color_to_gray_shape,
    rgb_to_hsv_stack,
    split_color_to_gray,
)

__all__ = [
    "ColorToGrayMode",
    "ImageChannelType",
    "color_to_gray",
    "color_to_gray_channel",
    "combine_color_to_gray",
    "nhwc_color_stack",
    "normalized_color_to_gray_weights",
    "restore_color_to_gray_shape",
    "rgb_to_hsv_stack",
    "split_color_to_gray",
]
