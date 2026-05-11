"""
Converted from CellProfiler: Crop
Original: crop, measure_area_retained_after_cropping, measure_original_image_area, get_measurements
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.interop.cellprofiler.crop_settings import (
    CropShape,
    CroppingMethod,
    RemovalMethod,
)
from openhcs.processing.backends.cellprofiler.crop import (
    CropBoundaryPair,
    CropImageRequest,
    CropMaskRequest,
    CropMeasurement,
    CropRequest,
    CropShapeMaskStrategy,
    CropSpatialBounds,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


@numpy
@special_inputs("mask_plane")
@special_outputs(
    (
        "crop_measurements",
        csv_materializer(
            fields=[
                "slice_index",
                "original_area",
                "area_retained",
                "fraction_retained",
            ],
            analysis_type="crop",
        ),
    )
)
def crop(
    image: np.ndarray,
    mask_plane: np.ndarray | None = None,
    crop_shape: CropShape | str = CropShape.RECTANGLE,
    cropping_method: CroppingMethod | str = CroppingMethod.COORDINATES,
    removal_method: RemovalMethod | str = RemovalMethod.NO,
    left_right_rectangle_positions: CropBoundaryPair = None,
    top_bottom_rectangle_positions: CropBoundaryPair = None,
    ellipse_center: tuple[float, float] | None = None,
    ellipse_x_radius: float | None = None,
    ellipse_y_radius: float | None = None,
    cropping_labels: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, CropMeasurement]:
    """Crop an image and return its CellProfiler crop-mask sidecar."""
    return CropRequest(
        image=image,
        mask_plane=mask_plane,
        crop_shape=crop_shape,
        cropping_method=cropping_method,
        removal_method=removal_method,
        left_right_rectangle_positions=left_right_rectangle_positions,
        top_bottom_rectangle_positions=top_bottom_rectangle_positions,
        ellipse_center=ellipse_center,
        ellipse_x_radius=ellipse_x_radius,
        ellipse_y_radius=ellipse_y_radius,
        cropping_labels=cropping_labels,
    ).execute()


@numpy(contract=ProcessingContract.PURE_2D)
def crop_simple(
    image: np.ndarray,
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
) -> np.ndarray:
    """Simple rectangular crop by specifying pixel amounts to remove from each edge."""
    h, w = image.shape

    y_start = crop_top
    y_end = h - crop_bottom if crop_bottom > 0 else h
    x_start = crop_left
    x_end = w - crop_right if crop_right > 0 else w

    y_start = max(0, min(y_start, h - 1))
    y_end = max(y_start + 1, min(y_end, h))
    x_start = max(0, min(x_start, w - 1))
    x_end = max(x_start + 1, min(x_end, w))

    return image[y_start:y_end, x_start:x_end].copy()


__all__ = [
    "CropBoundaryPair",
    "CropImageRequest",
    "CropMaskRequest",
    "CropMeasurement",
    "CropRequest",
    "CropShapeMaskStrategy",
    "CropSpatialBounds",
    "crop",
    "crop_simple",
]
