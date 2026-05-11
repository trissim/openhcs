"""Benchmark-library facade for OpenHCS trailing spatial-axis helpers."""

from openhcs.core.image_shapes import (
    apply_over_trailing_spatial_axes,
    trailing_spatial_factors,
    trailing_spatial_target_shape,
)

__all__ = [
    "apply_over_trailing_spatial_axes",
    "trailing_spatial_factors",
    "trailing_spatial_target_shape",
]
