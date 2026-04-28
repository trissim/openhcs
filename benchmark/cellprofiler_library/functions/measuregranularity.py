"""
Converted from CellProfiler: MeasureGranularity
Original: MeasureGranularity module

Measures granularity spectrum (texture size distribution) of images.
Granularity is measured by iteratively eroding the image and measuring
how much signal is lost at each scale.
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs
from openhcs.processing.materialization import csv_materializer


@dataclass
class GranularityMeasurement:
    """Granularity spectrum measurements for an image."""
    slice_index: int
    gs1: float
    gs2: float
    gs3: float
    gs4: float
    gs5: float
    gs6: float
    gs7: float
    gs8: float
    gs9: float
    gs10: float
    gs11: float
    gs12: float
    gs13: float
    gs14: float
    gs15: float
    gs16: float


@dataclass
class ObjectGranularityMeasurement:
    """Granularity spectrum measurements per object."""
    slice_index: int
    object_id: int
    gs1: float
    gs2: float
    gs3: float
    gs4: float
    gs5: float
    gs6: float
    gs7: float
    gs8: float
    gs9: float
    gs10: float
    gs11: float
    gs12: float
    gs13: float
    gs14: float
    gs15: float
    gs16: float


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("granularity_measurements", csv_materializer(
    fields=["slice_index", "gs1", "gs2", "gs3", "gs4", "gs5", "gs6", "gs7", "gs8",
            "gs9", "gs10", "gs11", "gs12", "gs13", "gs14", "gs15", "gs16"],
    analysis_type="granularity"
)))
def measure_granularity(
    image: np.ndarray,
    subsample_size: float = 0.25,
    background_subsample_size: float = 0.25,
    element_radius: int = 10,
    spectrum_length: int = 16,
) -> Tuple[np.ndarray, GranularityMeasurement]:
    """
    Measure granularity spectrum of an image.
    
    Granularity is a texture measurement that fits structure elements of
    increasing size into the image texture and outputs a spectrum of measures
    based on how well they fit.
    
    Args:
        image: Input grayscale image (H, W)
        subsample_size: Subsampling factor for granularity measurements (0-1)
        background_subsample_size: Subsampling factor for background reduction (0-1)
        element_radius: Radius of structuring element for background removal
        spectrum_length: Number of granular spectrum components to measure
    
    Returns:
        Tuple of (original image, granularity measurements)
    """
    import scipy.ndimage
    import skimage.morphology
    
    orig_shape = image.shape
    
    # Downsample the image
    if subsample_size < 1:
        new_shape = (np.array(orig_shape) * subsample_size).astype(int)
        new_shape = np.maximum(new_shape, 1)
        i, j = np.mgrid[0:new_shape[0], 0:new_shape[1]].astype(float) / subsample_size
        pixels = scipy.ndimage.map_coordinates(image, (i, j), order=1)
    else:
        pixels = image.copy()
        new_shape = np.array(orig_shape)
    
    # Remove background using morphological opening
    if background_subsample_size < 1:
        back_shape = (new_shape * background_subsample_size).astype(int)
        back_shape = np.maximum(back_shape, 1)
        bi, bj = np.mgrid[0:back_shape[0], 0:back_shape[1]].astype(float) / background_subsample_size
        back_pixels = scipy.ndimage.map_coordinates(pixels, (bi, bj), order=1)
    else:
        back_pixels = pixels.copy()
        back_shape = new_shape
    
    # Create structuring element and perform opening for background
    footprint = skimage.morphology.disk(element_radius, dtype=bool)
    back_pixels = skimage.morphology.erosion(back_pixels, footprint=footprint)
    back_pixels = skimage.morphology.dilation(back_pixels, footprint=footprint)
    
    # Upsample background if needed
    if background_subsample_size < 1:
        ui, uj = np.mgrid[0:new_shape[0], 0:new_shape[1]].astype(float)
        ui *= float(back_shape[0] - 1) / float(new_shape[0] - 1) if new_shape[0] > 1 else 0
        uj *= float(back_shape[1] - 1) / float(new_shape[1] - 1) if new_shape[1] > 1 else 0
        back_pixels = scipy.ndimage.map_coordinates(back_pixels, (ui, uj), order=1)
    
    # Subtract background
    pixels = pixels - back_pixels
    pixels[pixels < 0] = 0
    
    # Calculate granular spectrum
    startmean = np.mean(pixels)
    startmean = max(startmean, np.finfo(float).eps)
    ero = pixels.copy()
    currentmean = startmean
    
    footprint_small = skimage.morphology.disk(1, dtype=bool)
    gs_values = []
    
    for i in range(spectrum_length):
        prevmean = currentmean
        ero = skimage.morphology.erosion(ero, footprint=footprint_small)
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint_small)
        currentmean = np.mean(rec)
        gs = (prevmean - currentmean) * 100 / startmean
        gs_values.append(gs)
    
    # Pad with zeros if spectrum_length < 16
    while len(gs_values) < 16:
        gs_values.append(0.0)
    
    measurement = GranularityMeasurement(
        slice_index=0,
        gs1=gs_values[0],
        gs2=gs_values[1],
        gs3=gs_values[2],
        gs4=gs_values[3],
        gs5=gs_values[4],
        gs6=gs_values[5],
        gs7=gs_values[6],
        gs8=gs_values[7],
        gs9=gs_values[8],
        gs10=gs_values[9],
        gs11=gs_values[10],
        gs12=gs_values[11],
        gs13=gs_values[12],
        gs14=gs_values[13],
        gs15=gs_values[14],
        gs16=gs_values[15],
    )
    
    return image, measurement


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("object_granularity_measurements", csv_materializer(
    fields=["slice_index", "object_id", "gs1", "gs2", "gs3", "gs4", "gs5", "gs6", "gs7", "gs8",
            "gs9", "gs10", "gs11", "gs12", "gs13", "gs14", "gs15", "gs16"],
    analysis_type="object_granularity"
)))
def measure_granularity_objects(
    image: np.ndarray,
    labels: np.ndarray,
    subsample_size: float = 0.25,
    background_subsample_size: float = 0.25,
    element_radius: int = 10,
    spectrum_length: int = 16,
) -> Tuple[np.ndarray, List[ObjectGranularityMeasurement]]:
    """
    Measure granularity spectrum within labeled objects.
    
    Args:
        image: Input grayscale image (H, W)
        labels: Label image from segmentation (H, W)
        subsample_size: Subsampling factor for granularity measurements (0-1)
        background_subsample_size: Subsampling factor for background reduction (0-1)
        element_radius: Radius of structuring element for background removal
        spectrum_length: Number of granular spectrum components to measure
    
    Returns:
        Tuple of (original image, list of per-object granularity measurements)
    """
    import scipy.ndimage
    import skimage.morphology
    
    orig_shape = image.shape
    nobjects = int(np.max(labels))
    
    if nobjects == 0:
        return image, []
    
    object_range = np.arange(1, nobjects + 1)
    
    # Downsample the image
    if subsample_size < 1:
        new_shape = (np.array(orig_shape) * subsample_size).astype(int)
        new_shape = np.maximum(new_shape, 1)
        i, j = np.mgrid[0:new_shape[0], 0:new_shape[1]].astype(float) / subsample_size
        pixels = scipy.ndimage.map_coordinates(image, (i, j), order=1)
    else:
        pixels = image.copy()
        new_shape = np.array(orig_shape)
    
    # Remove background using morphological opening
    if background_subsample_size < 1:
        back_shape = (new_shape * background_subsample_size).astype(int)
        back_shape = np.maximum(back_shape, 1)
        bi, bj = np.mgrid[0:back_shape[0], 0:back_shape[1]].astype(float) / background_subsample_size
        back_pixels = scipy.ndimage.map_coordinates(pixels, (bi, bj), order=1)
    else:
        back_pixels = pixels.copy()
        back_shape = new_shape
    
    footprint = skimage.morphology.disk(element_radius, dtype=bool)
    back_pixels = skimage.morphology.erosion(back_pixels, footprint=footprint)
    back_pixels = skimage.morphology.dilation(back_pixels, footprint=footprint)
    
    if background_subsample_size < 1:
        ui, uj = np.mgrid[0:new_shape[0], 0:new_shape[1]].astype(float)
        ui *= float(back_shape[0] - 1) / float(new_shape[0] - 1) if new_shape[0] > 1 else 0
        uj *= float(back_shape[1] - 1) / float(new_shape[1] - 1) if new_shape[1] > 1 else 0
        back_pixels = scipy.ndimage.map_coordinates(back_pixels, (ui, uj), order=1)
    
    pixels = pixels - back_pixels
    pixels[pixels < 0] = 0
    
    # Get initial means per object
    current_means = np.array(scipy.ndimage.mean(image, labels, object_range))
    start_means = np.maximum(current_means, np.finfo(float).eps)
    
    # Calculate granular spectrum per object
    ero = pixels.copy()
    footprint_small = skimage.morphology.disk(1, dtype=bool)
    
    # Store gs values per object: shape (nobjects, spectrum_length)
    gs_per_object = np.zeros((nobjects, 16))
    
    for gs_idx in range(spectrum_length):
        prev_means = current_means.copy()
        ero = skimage.morphology.erosion(ero, footprint=footprint_small)
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint_small)
        
        # Upsample reconstructed image to original size
        if subsample_size < 1:
            ri, rj = np.mgrid[0:orig_shape[0], 0:orig_shape[1]].astype(float)
            ri *= float(new_shape[0] - 1) / float(orig_shape[0] - 1) if orig_shape[0] > 1 else 0
            rj *= float(new_shape[1] - 1) / float(orig_shape[1] - 1) if orig_shape[1] > 1 else 0
            rec_full = scipy.ndimage.map_coordinates(rec, (ri, rj), order=1)
        else:
            rec_full = rec
        
        new_means = np.array(scipy.ndimage.mean(rec_full, labels, object_range))
        gs_values = (prev_means - new_means) * 100 / start_means
        gs_per_object[:, gs_idx] = gs_values
        current_means = new_means
    
    # Create measurement objects
    measurements = []
    for obj_idx in range(nobjects):
        gs = gs_per_object[obj_idx]
        measurements.append(ObjectGranularityMeasurement(
            slice_index=0,
            object_id=obj_idx + 1,
            gs1=gs[0], gs2=gs[1], gs3=gs[2], gs4=gs[3],
            gs5=gs[4], gs6=gs[5], gs7=gs[6], gs8=gs[7],
            gs9=gs[8], gs10=gs[9], gs11=gs[10], gs12=gs[11],
            gs13=gs[12], gs14=gs[13], gs15=gs[14], gs16=gs[15],
        ))
    
    return image, measurements