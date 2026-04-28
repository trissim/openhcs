"""
Converted from CellProfiler: MeasureObjectIntensity
Measures intensity features for identified objects in grayscale images.
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from openhcs.core.memory import numpy


@dataclass
class ObjectIntensityMeasurement:
    """Per-object intensity measurements."""
    object_label: int
    integrated_intensity: float
    mean_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    integrated_intensity_edge: float
    mean_intensity_edge: float
    std_intensity_edge: float
    min_intensity_edge: float
    max_intensity_edge: float
    mass_displacement: float
    lower_quartile_intensity: float
    median_intensity: float
    mad_intensity: float
    upper_quartile_intensity: float
    center_mass_intensity_x: float
    center_mass_intensity_y: float
    max_intensity_x: float
    max_intensity_y: float


@dataclass
class ObjectIntensityResults:
    """Collection of intensity measurements for all objects."""
    slice_index: int
    object_count: int
    measurements: List[ObjectIntensityMeasurement]


def _fixup_scipy_result(result):
    """Convert scipy.ndimage result to proper array format."""
    if np.isscalar(result):
        return np.array([result])
    return np.asarray(result)


def _first_scalar_position(position) -> int:
    """Return the first scalar index from scipy's nested position shapes."""
    if np.isscalar(position):
        return int(position)
    if isinstance(position, np.ndarray):
        return _first_scalar_position(position.tolist())
    if hasattr(position, "__len__") and len(position) > 0:
        return _first_scalar_position(position[0])
    raise ValueError(f"Cannot extract scalar position from {position!r}.")


@numpy
def measure_object_intensity(
    image: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, List[ObjectIntensityMeasurement]]:
    """
    Measure intensity features for identified objects.
    
    Measures several intensity features for each labeled object including:
    - Integrated, mean, std, min, max intensity (whole object and edge)
    - Mass displacement
    - Quartile intensities and MAD
    - Center of mass and max intensity locations
    
    Args:
        image: Grayscale intensity image (H, W)
        labels: Label image where each object has unique integer label (H, W)
    
    Returns:
        Tuple of (original image, list of intensity measurements per object)
    """
    import scipy.ndimage as ndi
    from skimage.segmentation import find_boundaries
    
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    nobjects = len(unique_labels)
    
    if nobjects == 0:
        return image, []
    
    # Initialize measurement arrays
    integrated_intensity = np.zeros(nobjects)
    mean_intensity = np.zeros(nobjects)
    std_intensity = np.zeros(nobjects)
    min_intensity = np.zeros(nobjects)
    max_intensity = np.zeros(nobjects)
    integrated_intensity_edge = np.zeros(nobjects)
    mean_intensity_edge = np.zeros(nobjects)
    std_intensity_edge = np.zeros(nobjects)
    min_intensity_edge = np.zeros(nobjects)
    max_intensity_edge = np.zeros(nobjects)
    mass_displacement = np.zeros(nobjects)
    lower_quartile_intensity = np.zeros(nobjects)
    median_intensity = np.zeros(nobjects)
    mad_intensity = np.zeros(nobjects)
    upper_quartile_intensity = np.zeros(nobjects)
    cmi_x = np.zeros(nobjects)
    cmi_y = np.zeros(nobjects)
    max_x = np.zeros(nobjects)
    max_y = np.zeros(nobjects)
    
    # Create mask for valid pixels (finite values)
    valid_mask = np.isfinite(image)
    masked_labels = labels.copy()
    masked_labels[~valid_mask] = 0
    
    # Find object edges
    outlines = find_boundaries(labels, mode='inner')
    masked_outlines = outlines.copy()
    masked_outlines[~valid_mask] = False
    
    # Create coordinate meshes
    mesh_y, mesh_x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    
    # Mask for labeled pixels
    lmask = (masked_labels > 0) & valid_mask
    
    if np.any(lmask):
        limg = image[lmask]
        llabels = labels[lmask]
        lmesh_x = mesh_x[lmask]
        lmesh_y = mesh_y[lmask]
        
        # Count pixels per object
        lcount = _fixup_scipy_result(ndi.sum(np.ones(len(limg)), llabels, unique_labels))
        
        # Integrated intensity
        integrated_intensity = _fixup_scipy_result(ndi.sum(limg, llabels, unique_labels))
        
        # Mean intensity
        mean_intensity = integrated_intensity / np.maximum(lcount, 1)
        
        # Standard deviation
        mean_per_pixel = mean_intensity[np.searchsorted(unique_labels, llabels)]
        variance = _fixup_scipy_result(ndi.mean((limg - mean_per_pixel) ** 2, llabels, unique_labels))
        std_intensity = np.sqrt(variance)
        
        # Min and max intensity
        min_intensity = _fixup_scipy_result(ndi.minimum(limg, llabels, unique_labels))
        max_intensity = _fixup_scipy_result(ndi.maximum(limg, llabels, unique_labels))
        
        # Max intensity position
        max_positions = ndi.maximum_position(limg, llabels, unique_labels)
        if nobjects == 1:
            max_positions = [max_positions]
        for i, pos in enumerate(max_positions):
            if pos is not None and len(pos) > 0:
                idx = _first_scalar_position(pos)
                max_x[i] = lmesh_x[idx]
                max_y[i] = lmesh_y[idx]
        
        # Center of mass calculations
        cm_x = _fixup_scipy_result(ndi.mean(lmesh_x, llabels, unique_labels))
        cm_y = _fixup_scipy_result(ndi.mean(lmesh_y, llabels, unique_labels))
        
        i_x = _fixup_scipy_result(ndi.sum(lmesh_x * limg, llabels, unique_labels))
        i_y = _fixup_scipy_result(ndi.sum(lmesh_y * limg, llabels, unique_labels))
        
        cmi_x = i_x / np.maximum(integrated_intensity, 1e-10)
        cmi_y = i_y / np.maximum(integrated_intensity, 1e-10)
        
        # Mass displacement
        diff_x = cm_x - cmi_x
        diff_y = cm_y - cmi_y
        mass_displacement = np.sqrt(diff_x ** 2 + diff_y ** 2)
        
        # Quartile calculations
        order = np.lexsort((limg, llabels))
        areas = lcount.astype(int)
        indices = np.cumsum(areas) - areas
        
        for dest, fraction in [
            (lower_quartile_intensity, 0.25),
            (median_intensity, 0.5),
            (upper_quartile_intensity, 0.75)
        ]:
            qindex = indices.astype(float) + areas * fraction
            qfraction = qindex - np.floor(qindex)
            qindex_int = qindex.astype(int)
            
            for i in range(nobjects):
                qi = qindex_int[i]
                qf = qfraction[i]
                if qi < indices[i] + areas[i] - 1:
                    dest[i] = limg[order[qi]] * (1 - qf) + limg[order[qi + 1]] * qf
                elif areas[i] > 0:
                    dest[i] = limg[order[qi]]
        
        # MAD calculation
        label_indices = np.searchsorted(unique_labels, llabels)
        madimg = np.abs(limg - median_intensity[label_indices])
        order_mad = np.lexsort((madimg, llabels))
        
        for i in range(nobjects):
            qindex = int(indices[i] + areas[i] / 2)
            if qindex < indices[i] + areas[i]:
                mad_intensity[i] = madimg[order_mad[qindex]]
    
    # Edge measurements
    emask = masked_outlines > 0
    if np.any(emask):
        eimg = image[emask]
        elabels = labels[emask]
        
        ecount = _fixup_scipy_result(ndi.sum(np.ones(len(eimg)), elabels, unique_labels))
        integrated_intensity_edge = _fixup_scipy_result(ndi.sum(eimg, elabels, unique_labels))
        mean_intensity_edge = integrated_intensity_edge / np.maximum(ecount, 1)
        
        mean_edge_per_pixel = mean_intensity_edge[np.searchsorted(unique_labels, elabels)]
        variance_edge = _fixup_scipy_result(ndi.mean((eimg - mean_edge_per_pixel) ** 2, elabels, unique_labels))
        std_intensity_edge = np.sqrt(variance_edge)
        
        min_intensity_edge = _fixup_scipy_result(ndi.minimum(eimg, elabels, unique_labels))
        max_intensity_edge = _fixup_scipy_result(ndi.maximum(eimg, elabels, unique_labels))
    
    # Build measurement list
    measurements = []
    for i, label in enumerate(unique_labels):
        measurements.append(ObjectIntensityMeasurement(
            object_label=int(label),
            integrated_intensity=float(integrated_intensity[i]),
            mean_intensity=float(mean_intensity[i]),
            std_intensity=float(std_intensity[i]),
            min_intensity=float(min_intensity[i]),
            max_intensity=float(max_intensity[i]),
            integrated_intensity_edge=float(integrated_intensity_edge[i]),
            mean_intensity_edge=float(mean_intensity_edge[i]),
            std_intensity_edge=float(std_intensity_edge[i]),
            min_intensity_edge=float(min_intensity_edge[i]),
            max_intensity_edge=float(max_intensity_edge[i]),
            mass_displacement=float(mass_displacement[i]),
            lower_quartile_intensity=float(lower_quartile_intensity[i]),
            median_intensity=float(median_intensity[i]),
            mad_intensity=float(mad_intensity[i]),
            upper_quartile_intensity=float(upper_quartile_intensity[i]),
            center_mass_intensity_x=float(cmi_x[i]),
            center_mass_intensity_y=float(cmi_y[i]),
            max_intensity_x=float(max_x[i]),
            max_intensity_y=float(max_y[i]),
        ))
    
    return image, measurements
