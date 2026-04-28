"""
Converted from CellProfiler: MeasureTexture
Original: MeasureTexture module

Measures Haralick texture features from grayscale images.
These features quantify the degree and nature of textures within images
and objects to characterize roughness and smoothness.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer


# Haralick feature names
F_HARALICK = [
    "AngularSecondMoment", "Contrast", "Correlation", "Variance",
    "InverseDifferenceMoment", "SumAverage", "SumVariance", "SumEntropy",
    "Entropy", "DifferenceVariance", "DifferenceEntropy", "InfoMeas1", "InfoMeas2"
]


@dataclass
class TextureMeasurement:
    """Texture measurement results for a single slice/image."""
    slice_index: int
    scale: int
    direction: int
    gray_levels: int
    angular_second_moment: float
    contrast: float
    correlation: float
    variance: float
    inverse_difference_moment: float
    sum_average: float
    sum_variance: float
    sum_entropy: float
    entropy: float
    difference_variance: float
    difference_entropy: float
    info_meas1: float
    info_meas2: float


@dataclass
class ObjectTextureMeasurement:
    """Texture measurement results per object."""
    slice_index: int
    object_label: int
    scale: int
    direction: int
    gray_levels: int
    angular_second_moment: float
    contrast: float
    correlation: float
    variance: float
    inverse_difference_moment: float
    sum_average: float
    sum_variance: float
    sum_entropy: float
    entropy: float
    difference_variance: float
    difference_entropy: float
    info_meas1: float
    info_meas2: float


def _compute_glcm(image: np.ndarray, distance: int, direction: int) -> np.ndarray:
    """
    CellProfiler Parameter Mapping:
    (CellProfiler setting -> Python parameter)
        'Select images to measure' -> (pipeline-handled)
        'Select objects to measure' -> (pipeline-handled)
        'Enter how many gray levels to measure the texture at' -> gray_levels
        'Hidden' -> (pipeline-handled)
        'Measure whole images or objects?' -> (pipeline-handled)
        'Texture scale to measure' -> scale

    CellProfiler Parameter Mapping:
    (CellProfiler setting -> Python parameter)
        'Select images to measure' -> (pipeline-handled)
        'Select objects to measure' -> (pipeline-handled)
        'Enter how many gray levels to measure the texture at' -> gray_levels
        'Hidden' -> (pipeline-handled)
        'Measure whole images or objects?' -> (pipeline-handled)
        'Texture scale to measure' -> scale

    CellProfiler Parameter Mapping:
    (CellProfiler setting -> Python parameter)
        'Select images to measure' -> (pipeline-handled)
        'Select objects to measure' -> (pipeline-handled)
        'Enter how many gray levels to measure the texture at' -> gray_levels
        'Hidden' -> (pipeline-handled)
        'Measure whole images or objects?' -> (pipeline-handled)
        'Texture scale to measure' -> scale

    Compute Gray-Level Co-occurrence Matrix for a given direction.
    
    2D directions (y, x offsets):
    - 0: horizontal (0, 1)
    - 1: diagonal NW-SE (1, 1)
    - 2: vertical (1, 0)
    - 3: diagonal NE-SW (1, -1)
    """
    from skimage.feature import graycomatrix
    
    # Map direction index to angle in radians
    # skimage uses angles: 0, pi/4, pi/2, 3*pi/4
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    if direction < len(angles):
        angle = angles[direction]
    else:
        angle = 0
    
    # Compute GLCM
    glcm = graycomatrix(
        image, 
        distances=[distance], 
        angles=[angle], 
        levels=int(image.max()) + 1,
        symmetric=True,
        normed=True
    )
    
    return glcm[:, :, 0, 0]


def _compute_haralick_features(glcm: np.ndarray) -> np.ndarray:
    """
    Compute 13 Haralick texture features from a GLCM.
    
    Returns array of 13 features in order:
    AngularSecondMoment, Contrast, Correlation, Variance,
    InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy,
    Entropy, DifferenceVariance, DifferenceEntropy, InfoMeas1, InfoMeas2
    """
    from skimage.feature import graycoprops
    
    # Reshape for skimage (needs 4D)
    glcm_4d = glcm[:, :, np.newaxis, np.newaxis]
    
    eps = 1e-10
    n_levels = glcm.shape[0]
    
    # Normalize GLCM
    glcm_sum = glcm.sum()
    if glcm_sum > 0:
        p = glcm / glcm_sum
    else:
        p = glcm
    
    # Create index arrays
    i_indices = np.arange(n_levels)
    j_indices = np.arange(n_levels)
    i, j = np.meshgrid(i_indices, j_indices, indexing='ij')
    
    # Marginal probabilities
    px = p.sum(axis=1)
    py = p.sum(axis=0)
    
    # Means and standard deviations
    ux = np.sum(i_indices * px)
    uy = np.sum(j_indices * py)
    sx = np.sqrt(np.sum(((i_indices - ux) ** 2) * px) + eps)
    sy = np.sqrt(np.sum(((j_indices - uy) ** 2) * py) + eps)
    
    # 1. Angular Second Moment (Energy)
    asm = np.sum(p ** 2)
    
    # 2. Contrast
    contrast = np.sum(((i - j) ** 2) * p)
    
    # 3. Correlation
    correlation = np.sum((i - ux) * (j - uy) * p) / (sx * sy + eps)
    
    # 4. Variance
    variance = np.sum(((i - ux) ** 2) * p)
    
    # 5. Inverse Difference Moment (Homogeneity)
    idm = np.sum(p / (1 + (i - j) ** 2))
    
    # Sum and difference distributions
    p_x_plus_y = np.zeros(2 * n_levels - 1)
    p_x_minus_y = np.zeros(n_levels)
    
    for ii in range(n_levels):
        for jj in range(n_levels):
            p_x_plus_y[ii + jj] += p[ii, jj]
            p_x_minus_y[abs(ii - jj)] += p[ii, jj]
    
    # 6. Sum Average
    k_plus = np.arange(2 * n_levels - 1)
    sum_average = np.sum(k_plus * p_x_plus_y)
    
    # 7. Sum Variance
    sum_variance = np.sum(((k_plus - sum_average) ** 2) * p_x_plus_y)
    
    # 8. Sum Entropy
    sum_entropy = -np.sum(p_x_plus_y * np.log2(p_x_plus_y + eps))
    
    # 9. Entropy
    entropy = -np.sum(p * np.log2(p + eps))
    
    # 10. Difference Variance
    k_minus = np.arange(n_levels)
    diff_mean = np.sum(k_minus * p_x_minus_y)
    difference_variance = np.sum(((k_minus - diff_mean) ** 2) * p_x_minus_y)
    
    # 11. Difference Entropy
    difference_entropy = -np.sum(p_x_minus_y * np.log2(p_x_minus_y + eps))
    
    # 12 & 13. Information Measures of Correlation
    hx = -np.sum(px * np.log2(px + eps))
    hy = -np.sum(py * np.log2(py + eps))
    hxy = entropy
    
    hxy1 = -np.sum(p * np.log2(np.outer(px, py) + eps))
    hxy2 = -np.sum(np.outer(px, py) * np.log2(np.outer(px, py) + eps))
    
    info_meas1 = (hxy - hxy1) / (max(hx, hy) + eps)
    info_meas2 = np.sqrt(max(0, 1 - np.exp(-2 * (hxy2 - hxy))))
    
    return np.array([
        asm, contrast, correlation, variance, idm,
        sum_average, sum_variance, sum_entropy, entropy,
        difference_variance, difference_entropy, info_meas1, info_meas2
    ])


@numpy
@special_outputs(("texture_measurements", csv_materializer(
    fields=["slice_index", "scale", "direction", "gray_levels",
            "angular_second_moment", "contrast", "correlation", "variance",
            "inverse_difference_moment", "sum_average", "sum_variance", 
            "sum_entropy", "entropy", "difference_variance", "difference_entropy",
            "info_meas1", "info_meas2"],
    analysis_type="texture"
)))
def measure_texture(
    image: np.ndarray,
    scale: int = 3,
    gray_levels: int = 256,
) -> Tuple[np.ndarray, List[TextureMeasurement]]:
    """
    Measure Haralick texture features on a grayscale image.
    
    Computes 13 Haralick texture features derived from the gray-level
    co-occurrence matrix (GLCM) at the specified scale.
    
    Args:
        image: Input grayscale image (H, W), values in [0, 1]
        scale: Distance in pixels for GLCM computation (default: 3)
        gray_levels: Number of gray levels for quantization (2-256, default: 256)
    
    Returns:
        Tuple of (original image, list of TextureMeasurement for each direction)
    """
    from skimage.exposure import rescale_intensity
    from skimage.util import img_as_ubyte
    
    # Ensure valid gray_levels
    gray_levels = max(2, min(256, gray_levels))
    
    # Convert to uint8 and rescale to gray_levels
    if image.dtype != np.uint8:
        pixel_data = img_as_ubyte(np.clip(image, 0, 1))
    else:
        pixel_data = image.copy()
    
    if gray_levels != 256:
        pixel_data = rescale_intensity(
            pixel_data, 
            in_range=(0, 255), 
            out_range=(0, gray_levels - 1)
        ).astype(np.uint8)
    
    measurements = []
    n_directions = 4  # 2D has 4 directions
    
    for direction in range(n_directions):
        try:
            # Compute GLCM
            glcm = _compute_glcm(pixel_data, scale, direction)
            
            # Compute Haralick features
            features = _compute_haralick_features(glcm)
            
            measurement = TextureMeasurement(
                slice_index=0,
                scale=scale,
                direction=direction,
                gray_levels=gray_levels,
                angular_second_moment=float(features[0]),
                contrast=float(features[1]),
                correlation=float(features[2]),
                variance=float(features[3]),
                inverse_difference_moment=float(features[4]),
                sum_average=float(features[5]),
                sum_variance=float(features[6]),
                sum_entropy=float(features[7]),
                entropy=float(features[8]),
                difference_variance=float(features[9]),
                difference_entropy=float(features[10]),
                info_meas1=float(features[11]),
                info_meas2=float(features[12]),
            )
        except Exception:
            # Return NaN values on error
            measurement = TextureMeasurement(
                slice_index=0,
                scale=scale,
                direction=direction,
                gray_levels=gray_levels,
                angular_second_moment=np.nan,
                contrast=np.nan,
                correlation=np.nan,
                variance=np.nan,
                inverse_difference_moment=np.nan,
                sum_average=np.nan,
                sum_variance=np.nan,
                sum_entropy=np.nan,
                entropy=np.nan,
                difference_variance=np.nan,
                difference_entropy=np.nan,
                info_meas1=np.nan,
                info_meas2=np.nan,
            )
        
        measurements.append(measurement)
    
    return image, measurements


@numpy
@special_inputs("labels")
@special_outputs(("object_texture_measurements", csv_materializer(
    fields=["slice_index", "object_label", "scale", "direction", "gray_levels",
            "angular_second_moment", "contrast", "correlation", "variance",
            "inverse_difference_moment", "sum_average", "sum_variance", 
            "sum_entropy", "entropy", "difference_variance", "difference_entropy",
            "info_meas1", "info_meas2"],
    analysis_type="object_texture"
)))
def measure_texture_objects(
    image: np.ndarray,
    labels: np.ndarray,
    scale: int = 3,
    gray_levels: int = 256,
) -> Tuple[np.ndarray, List[ObjectTextureMeasurement]]:
    """
    Measure Haralick texture features for each labeled object.
    
    Computes 13 Haralick texture features for each object in the label image,
    derived from the gray-level co-occurrence matrix (GLCM) at the specified scale.
    
    Args:
        image: Input grayscale image (H, W), values in [0, 1]
        labels: Label image with integer object labels (H, W)
        scale: Distance in pixels for GLCM computation (default: 3)
        gray_levels: Number of gray levels for quantization (2-256, default: 256)
    
    Returns:
        Tuple of (original image, list of ObjectTextureMeasurement for each object/direction)
    """
    from skimage.exposure import rescale_intensity
    from skimage.util import img_as_ubyte
    from skimage.measure import regionprops
    
    # Ensure valid gray_levels
    gray_levels = max(2, min(256, gray_levels))
    
    # Convert to uint8 and rescale to gray_levels
    if image.dtype != np.uint8:
        pixel_data = img_as_ubyte(np.clip(image, 0, 1))
    else:
        pixel_data = image.copy()
    
    if gray_levels != 256:
        pixel_data = rescale_intensity(
            pixel_data, 
            in_range=(0, 255), 
            out_range=(0, gray_levels - 1)
        ).astype(np.uint8)
    
    measurements = []
    n_directions = 4  # 2D has 4 directions
    
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) == 0:
        return image, measurements
    
    props = regionprops(labels.astype(np.int32), intensity_image=pixel_data)
    
    for prop in props:
        label_data = prop.intensity_image
        
        # Skip objects that are too small
        if label_data.shape[0] < scale + 1 or label_data.shape[1] < scale + 1:
            for direction in range(n_directions):
                measurements.append(ObjectTextureMeasurement(
                    slice_index=0,
                    object_label=prop.label,
                    scale=scale,
                    direction=direction,
                    gray_levels=gray_levels,
                    angular_second_moment=np.nan,
                    contrast=np.nan,
                    correlation=np.nan,
                    variance=np.nan,
                    inverse_difference_moment=np.nan,
                    sum_average=np.nan,
                    sum_variance=np.nan,
                    sum_entropy=np.nan,
                    entropy=np.nan,
                    difference_variance=np.nan,
                    difference_entropy=np.nan,
                    info_meas1=np.nan,
                    info_meas2=np.nan,
                ))
            continue
        
        for direction in range(n_directions):
            try:
                # Compute GLCM for this object
                glcm = _compute_glcm(label_data, scale, direction)
                
                # Compute Haralick features
                features = _compute_haralick_features(glcm)
                
                measurement = ObjectTextureMeasurement(
                    slice_index=0,
                    object_label=prop.label,
                    scale=scale,
                    direction=direction,
                    gray_levels=gray_levels,
                    angular_second_moment=float(features[0]),
                    contrast=float(features[1]),
                    correlation=float(features[2]),
                    variance=float(features[3]),
                    inverse_difference_moment=float(features[4]),
                    sum_average=float(features[5]),
                    sum_variance=float(features[6]),
                    sum_entropy=float(features[7]),
                    entropy=float(features[8]),
                    difference_variance=float(features[9]),
                    difference_entropy=float(features[10]),
                    info_meas1=float(features[11]),
                    info_meas2=float(features[12]),
                )
            except Exception:
                measurement = ObjectTextureMeasurement(
                    slice_index=0,
                    object_label=prop.label,
                    scale=scale,
                    direction=direction,
                    gray_levels=gray_levels,
                    angular_second_moment=np.nan,
                    contrast=np.nan,
                    correlation=np.nan,
                    variance=np.nan,
                    inverse_difference_moment=np.nan,
                    sum_average=np.nan,
                    sum_variance=np.nan,
                    sum_entropy=np.nan,
                    entropy=np.nan,
                    difference_variance=np.nan,
                    difference_entropy=np.nan,
                    info_meas1=np.nan,
                    info_meas2=np.nan,
                )
            
            measurements.append(measurement)
    
    return image, measurements
