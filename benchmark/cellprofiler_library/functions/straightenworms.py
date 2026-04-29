"""
Converted from CellProfiler: StraightenWorms
Straightens untangled worms using control points and training parameters.
"""

import numpy as np
from typing import Any
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs
from openhcs.processing.materialization import csv_materializer
from scipy.interpolate import interp1d
import scipy.ndimage

from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum
from benchmark.cellprofiler_library.functions.worm_geometry import (
    calculate_cumulative_lengths,
    control_points_for_label_image,
)


class FlipMode(Enum):
    NONE = "do_not_align"
    TOP = "top_brightest"
    BOTTOM = "bottom_brightest"
    MANUAL = "flip_manually"


@dataclass
class WormMeasurement:
    slice_index: int
    object_number: int
    center_x: float
    center_y: float
    mean_intensity: float
    std_intensity: float


@dataclass(frozen=True, slots=True)
class StraightenWormsSliceRequest:
    image: np.ndarray
    labels: np.ndarray
    control_points: np.ndarray
    worm_width: int
    num_control_points: int
    flip_mode: FlipMode
    measure_intensity: bool
    slice_index: int


@numpy
@special_inputs("worm_labels")
@special_outputs(
    ("straightened_labels", None),
    ("worm_measurements", csv_materializer(
        fields=["slice_index", "object_number", "center_x", "center_y", "mean_intensity", "std_intensity"],
        analysis_type="worm_measurements"
    ))
)
def straighten_worms(
    image: np.ndarray,
    worm_labels: np.ndarray,
    control_points: np.ndarray | None = None,
    worm_width: int = 20,
    num_control_points: int = 21,
    flip_mode: FlipMode = FlipMode.NONE,
    number_of_segments: int = 4,
    number_of_stripes: int = 3,
    measure_intensity: bool = True,
) -> tuple[Any, ...]:
    """
    Straighten worms using control points from UntangleWorms.
    
    Args:
        image: Input image (D, H, W) or (H, W)
        worm_labels: Label image with worm objects
        control_points: Control points array (nworms, 2, ncontrolpoints)
        worm_width: Width of straightened worm image
        num_control_points: Number of control points per worm
        flip_mode: How to align worms (none, top_brightest, bottom_brightest)
        number_of_segments: Number of transverse segments for measurements
        number_of_stripes: Number of longitudinal stripes for measurements
        measure_intensity: Whether to measure intensity distribution
    
    Returns:
        Tuple of (straightened_image, straightened_labels, measurements)
    """
    flip_mode = _coerce_function_enum(FlipMode, flip_mode)
    if flip_mode is FlipMode.MANUAL:
        raise NotImplementedError("StraightenWorms manual flipping is interactive.")

    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    if worm_labels.ndim == 2:
        worm_labels = worm_labels[np.newaxis, :, :]
    
    results = []
    all_labels = []
    all_measurements = []
    
    for d in range(image.shape[0]):
        img_slice = image[d]
        labels_slice = worm_labels[d] if d < worm_labels.shape[0] else worm_labels[0]
        
        straightened_img, straightened_lbl, measurements = _straighten_single_slice(
            StraightenWormsSliceRequest(
                image=img_slice,
                labels=labels_slice,
                control_points=_control_points_for_slice(
                    control_points,
                    labels_slice,
                    num_control_points,
                ),
                worm_width=worm_width,
                num_control_points=num_control_points,
                flip_mode=flip_mode,
                measure_intensity=measure_intensity,
                slice_index=d,
            )
        )
        results.append(straightened_img)
        all_labels.append(straightened_lbl)
        all_measurements.extend(measurements)
    
    straightened_images = np.stack(results, axis=0)
    straightened_labels = np.stack(all_labels, axis=0)

    return (
        *tuple(straightened_images[index] for index in range(straightened_images.shape[0])),
        straightened_labels,
        all_measurements,
    )


def _straighten_single_slice(
    request: StraightenWormsSliceRequest,
) -> tuple[np.ndarray, np.ndarray, list[WormMeasurement]]:
    """Straighten worms in a single 2D slice."""
    image = request.image
    labels = request.labels
    control_points = request.control_points
    half_width = request.worm_width // 2
    width = 2 * half_width + 1

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    nworms = len(unique_labels)

    if nworms == 0:
        shape = (width, width)
        return np.zeros(shape, dtype=image.dtype), np.zeros(shape, dtype=np.int32), []
    
    # Calculate worm lengths from control points
    lengths = []
    for i in range(min(nworms, control_points.shape[0])):
        cp = control_points[i]  # (2, ncontrolpoints)
        length = calculate_cumulative_lengths(cp.T)[-1]
        lengths.append(int(np.ceil(length)))
    
    if len(lengths) == 0:
        shape = (width, width)
        return np.zeros(shape, dtype=image.dtype), np.zeros(shape, dtype=np.int32), []
    
    max_length = max(lengths) if lengths else width
    shape = (max_length + width, nworms * width)
    
    straightened_labels = np.zeros(shape, dtype=np.int32)
    ix = np.zeros(shape)
    jx = np.zeros(shape)
    
    measurements = []
    
    for i, obj_num in enumerate(unique_labels):
        if i >= len(lengths) or lengths[i] == 0:
            continue
        
        if i >= control_points.shape[0]:
            continue
            
        cp = control_points[i]  # (2, ncontrolpoints)
        ii = cp[0]  # y coordinates
        jj = cp[1]  # x coordinates
        
        length = lengths[i]
        
        # Interpolate control points
        t_orig = np.linspace(0, length, request.num_control_points)
        t_new = np.arange(0, length + 1)
        
        si = interp1d(t_orig, ii, kind='linear', fill_value='extrapolate')
        sj = interp1d(t_orig, jj, kind='linear', fill_value='extrapolate')
        
        ci = si(t_new)
        cj = sj(t_new)
        
        # Calculate normals
        di = np.diff(ci, prepend=ci[0])
        dj = np.diff(cj, prepend=cj[0])
        di[0] = di[1] if len(di) > 1 else 0
        dj[0] = dj[1] if len(dj) > 1 else 0
        
        norm = np.sqrt(di**2 + dj**2)
        norm[norm == 0] = 1
        ni = -dj / norm
        nj = di / norm
        
        # Extend worm by half_width at head and tail
        ci_ext = np.concatenate([
            np.arange(-half_width, 0) * nj[0] + ci[0],
            ci,
            np.arange(1, half_width + 1) * nj[-1] + ci[-1]
        ])
        cj_ext = np.concatenate([
            np.arange(-half_width, 0) * (-ni[0]) + cj[0],
            cj,
            np.arange(1, half_width + 1) * (-ni[-1]) + cj[-1]
        ])
        ni_ext = np.concatenate([[ni[0]] * half_width, ni, [ni[-1]] * half_width])
        nj_ext = np.concatenate([[nj[0]] * half_width, nj, [nj[-1]] * half_width])
        
        # Create coordinate mapping
        iii, jjj = np.mgrid[0:len(ci_ext), -half_width:(half_width + 1)]
        
        islice = slice(0, len(ci_ext))
        jslice = slice(width * i, width * (i + 1))
        
        ix[islice, jslice] = ci_ext[iii] + ni_ext[iii] * jjj
        jx[islice, jslice] = cj_ext[iii] + nj_ext[iii] * jjj
        
        # Handle flipping
        if request.flip_mode != FlipMode.NONE:
            ixs = ix[islice, jslice]
            jxs = jx[islice, jslice]
            
            # Sample image
            simage = scipy.ndimage.map_coordinates(image, [ixs, jxs], order=1, mode='constant')
            smask = scipy.ndimage.map_coordinates((labels == obj_num).astype(np.float32), [ixs, jxs], order=0)
            simage = simage * smask
            
            halfway = len(ci_ext) // 2
            area_top = np.sum(smask[:halfway, :])
            area_bottom = np.sum(smask[halfway:, :])
            
            if area_top > 0 and area_bottom > 0:
                top_intensity = np.sum(simage[:halfway, :]) / area_top
                bottom_intensity = np.sum(simage[halfway:, :]) / area_bottom
                
                should_flip = (
                    (
                        request.flip_mode == FlipMode.TOP
                        and top_intensity < bottom_intensity
                    )
                    or (
                        request.flip_mode == FlipMode.BOTTOM
                        and bottom_intensity < top_intensity
                    )
                )
                
                if should_flip:
                    iii_flip = len(ci_ext) - iii - 1
                    jjj_flip = -jjj
                    ix[islice, jslice] = ci_ext[iii_flip] + ni_ext[iii_flip] * jjj_flip
                    jx[islice, jslice] = cj_ext[iii_flip] + nj_ext[iii_flip] * jjj_flip
        
        # Create mask for this worm
        mask = scipy.ndimage.map_coordinates(
            (labels == obj_num).astype(np.float32),
            [ix[islice, jslice], jx[islice, jslice]],
            order=0
        ) > 0.5
        straightened_labels[islice, jslice][mask] = int(obj_num)
    
    # Map image coordinates
    straightened_image = scipy.ndimage.map_coordinates(image, [ix, jx], order=1, mode='constant')
    
    # Measure intensity if requested
    if request.measure_intensity:
        for i, obj_num in enumerate(unique_labels):
            mask = straightened_labels == obj_num
            if np.sum(mask) > 0:
                values = straightened_image[mask]
                center_y, center_x = scipy.ndimage.center_of_mass(mask.astype(float))
                
                measurements.append(WormMeasurement(
                    slice_index=request.slice_index,
                    object_number=int(obj_num),
                    center_x=float(center_x) if not np.isnan(center_x) else 0.0,
                    center_y=float(center_y) if not np.isnan(center_y) else 0.0,
                    mean_intensity=float(np.mean(values)),
                    std_intensity=float(np.std(values))
                ))
    
    return straightened_image, straightened_labels, measurements


def _control_points_for_slice(
    control_points: np.ndarray | None,
    labels: np.ndarray,
    num_control_points: int,
) -> np.ndarray:
    if control_points is None:
        return control_points_for_label_image(labels, num_control_points)
    return _normalized_control_points(control_points, num_control_points)


def _normalized_control_points(
    control_points: np.ndarray,
    num_control_points: int,
) -> np.ndarray:
    points = np.asarray(control_points, dtype=float)
    if points.ndim != 3:
        raise ValueError(
            "StraightenWorms control_points must have shape "
            "(objects, 2, control_points) or (2, control_points, objects)."
        )
    if points.shape[1] == 2:
        normalized = points
    elif points.shape[0] == 2:
        normalized = points.transpose(2, 0, 1)
    else:
        raise ValueError(
            "StraightenWorms control_points must include one coordinate axis "
            "of length 2."
        )
    if normalized.shape[2] != num_control_points:
        raise ValueError(
            f"StraightenWorms expected {num_control_points} control points; "
            f"got {normalized.shape[2]}."
        )
    return normalized
