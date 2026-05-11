"""Benchmark-library facade for CellProfiler MaskImage."""

from openhcs.processing.backends.cellprofiler.image_geometry import (
    MaskSource,
    mask_image,
    mask_image_stacked,
    mask_image_with_binary,
    masked_image_plane,
)

__all__ = [
    "MaskSource",
    "mask_image",
    "mask_image_stacked",
    "mask_image_with_binary",
    "masked_image_plane",
]
