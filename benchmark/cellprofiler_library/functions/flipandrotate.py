"""
Converted from CellProfiler: FlipAndRotate
Original: FlipAndRotate module

Flips (mirror image) and/or rotates an image.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer


class FlipMethod(Enum):
    NONE = "none"
    LEFT_TO_RIGHT = "left_to_right"
    TOP_TO_BOTTOM = "top_to_bottom"
    BOTH = "both"


class RotateMethod(Enum):
    NONE = "none"
    ANGLE = "angle"
    COORDINATES = "coordinates"


class AlignmentDirection(Enum):
    HORIZONTALLY = "horizontally"
    VERTICALLY = "vertically"


@dataclass
class RotationResult:
    slice_index: int
    rotation_angle: float


def _affine_offset(shape: Tuple[int, int], transform: np.ndarray) -> np.ndarray:
    """Calculate offset for affine transform to rotate about center.
    
    Args:
        shape: Shape of the array (H, W)
        transform: 2x2 transformation matrix
        
    Returns:
        Offset array for scipy.ndimage.affine_transform
    """
    c = (np.array(shape[:2]) - 1).astype(float) / 2.0
    return -np.dot(transform - np.identity(2), c)


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("rotation_results", csv_materializer(
    fields=["slice_index", "rotation_angle"],
    analysis_type="rotation"
)))
def flip_and_rotate(
    image: np.ndarray,
    flip_method: FlipMethod = FlipMethod.NONE,
    rotate_method: RotateMethod = RotateMethod.NONE,
    rotation_angle: float = 0.0,
    first_pixel_x: int = 0,
    first_pixel_y: int = 0,
    second_pixel_x: int = 0,
    second_pixel_y: int = 100,
    alignment_direction: AlignmentDirection = AlignmentDirection.HORIZONTALLY,
    crop_rotated_edges: bool = True,
) -> Tuple[np.ndarray, RotationResult]:
    """Flip and/or rotate an image.
    
    Args:
        image: Input image array (H, W) or (H, W, C)
        flip_method: How to flip the image
        rotate_method: How to determine rotation
        rotation_angle: Angle in degrees (positive = counterclockwise)
        first_pixel_x: X coordinate of first alignment point
        first_pixel_y: Y coordinate of first alignment point
        second_pixel_x: X coordinate of second alignment point
        second_pixel_y: Y coordinate of second alignment point
        alignment_direction: Whether to align points horizontally or vertically
        crop_rotated_edges: Whether to crop black edges after rotation
        
    Returns:
        Tuple of (transformed image, rotation measurement)
    """
    from scipy.ndimage import rotate as scipy_rotate
    
    pixel_data = image.copy()
    
    # Apply flip
    if flip_method != FlipMethod.NONE:
        if flip_method == FlipMethod.LEFT_TO_RIGHT:
            pixel_data = np.flip(pixel_data, axis=1)
        elif flip_method == FlipMethod.TOP_TO_BOTTOM:
            pixel_data = np.flip(pixel_data, axis=0)
        elif flip_method == FlipMethod.BOTH:
            pixel_data = np.flip(np.flip(pixel_data, axis=1), axis=0)
    
    # Calculate rotation angle
    angle = 0.0
    if rotate_method != RotateMethod.NONE:
        if rotate_method == RotateMethod.ANGLE:
            angle = rotation_angle
        elif rotate_method == RotateMethod.COORDINATES:
            xdiff = second_pixel_x - first_pixel_x
            ydiff = second_pixel_y - first_pixel_y
            if alignment_direction == AlignmentDirection.VERTICALLY:
                angle = -np.arctan2(ydiff, xdiff) * 180.0 / np.pi
            else:  # HORIZONTALLY
                angle = np.arctan2(xdiff, ydiff) * 180.0 / np.pi
        
        # Apply rotation
        if angle != 0.0:
            pixel_data = scipy_rotate(pixel_data, angle, reshape=True, order=1)
            
            if crop_rotated_edges:
                # Find the largest rectangle that fits inside the rotated image
                # Create a mask of valid (non-black) pixels
                if pixel_data.ndim == 2:
                    crop_mask = scipy_rotate(
                        np.ones(image.shape[:2]), angle, reshape=True
                    ) > 0.50
                else:
                    crop_mask = scipy_rotate(
                        np.ones(image.shape[:2]), angle, reshape=True
                    ) > 0.50
                
                # Find the largest inscribed rectangle
                half = (np.array(crop_mask.shape) // 2).astype(int)
                
                # Work on lower right quadrant
                quartercrop = crop_mask[half[0]:, half[1]:]
                ci = np.cumsum(quartercrop, 0)
                cj = np.cumsum(quartercrop, 1)
                carea_d = ci * cj
                carea_d[quartercrop == 0] = 0
                
                # Work on upper right quadrant (flipped)
                quartercrop_u = crop_mask[crop_mask.shape[0] - half[0] - 1::-1, half[1]:]
                ci = np.cumsum(quartercrop_u, 0)
                cj = np.cumsum(quartercrop_u, 1)
                carea_u = ci * cj
                carea_u[quartercrop_u == 0] = 0
                
                # Combine areas
                min_shape = min(carea_d.shape[0], carea_u.shape[0])
                carea = carea_d[:min_shape] + carea_u[:min_shape]
                
                if carea.size > 0:
                    max_carea = np.max(carea)
                    if max_carea > 0:
                        max_area_idx = np.argwhere(carea == max_carea)[0] + half
                        min_i = max(crop_mask.shape[0] - max_area_idx[0] - 1, 0)
                        max_i = max_area_idx[0] + 1
                        min_j = max(crop_mask.shape[1] - max_area_idx[1] - 1, 0)
                        max_j = max_area_idx[1] + 1
                        pixel_data = pixel_data[min_i:max_i, min_j:max_j]
    
    result = RotationResult(
        slice_index=0,
        rotation_angle=angle
    )
    
    return pixel_data.astype(np.float32), result