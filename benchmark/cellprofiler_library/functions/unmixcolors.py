"""Converted from CellProfiler: UnmixColors

Unmixes histologically stained color images into separate grayscale images
per dye stain using color deconvolution.

Based on: Ruifrok AC, Johnston DA. (2001) "Quantification of histochemical
staining by color deconvolution." Analytical & Quantitative Cytology & Histology
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class StainType(Enum):
    HEMATOXYLIN = "hematoxylin"
    EOSIN = "eosin"
    DAB = "dab"
    FAST_RED = "fast_red"
    FAST_BLUE = "fast_blue"
    METHYL_BLUE = "methyl_blue"
    METHYL_GREEN = "methyl_green"
    AEC = "aec"
    ANILINE_BLUE = "aniline_blue"
    AZOCARMINE = "azocarmine"
    ALCIAN_BLUE = "alcian_blue"
    PAS = "pas"
    HEMATOXYLIN_AND_PAS = "hematoxylin_and_pas"
    FEULGEN = "feulgen"
    METHYLENE_BLUE = "methylene_blue"
    ORANGE_G = "orange_g"
    PONCEAU_FUCHSIN = "ponceau_fuchsin"
    CUSTOM = "custom"


# Pre-calibrated stain absorbance vectors (R, G, B)
STAIN_VECTORS = {
    StainType.HEMATOXYLIN: (0.644, 0.717, 0.267),
    StainType.EOSIN: (0.093, 0.954, 0.283),
    StainType.DAB: (0.268, 0.570, 0.776),
    StainType.FAST_RED: (0.214, 0.851, 0.478),
    StainType.FAST_BLUE: (0.749, 0.606, 0.267),
    StainType.METHYL_BLUE: (0.799, 0.591, 0.105),
    StainType.METHYL_GREEN: (0.980, 0.144, 0.133),
    StainType.AEC: (0.274, 0.679, 0.680),
    StainType.ANILINE_BLUE: (0.853, 0.509, 0.113),
    StainType.AZOCARMINE: (0.071, 0.977, 0.198),
    StainType.ALCIAN_BLUE: (0.875, 0.458, 0.158),
    StainType.PAS: (0.175, 0.972, 0.155),
    StainType.HEMATOXYLIN_AND_PAS: (0.553, 0.754, 0.354),
    StainType.FEULGEN: (0.464, 0.830, 0.308),
    StainType.METHYLENE_BLUE: (0.553, 0.754, 0.354),
    StainType.ORANGE_G: (0.107, 0.368, 0.923),
    StainType.PONCEAU_FUCHSIN: (0.100, 0.737, 0.668),
}


def _get_absorbance_vector(stain: StainType, custom_rgb: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """Get normalized absorbance vector for a stain."""
    if stain == StainType.CUSTOM and custom_rgb is not None:
        vec = np.array(custom_rgb)
    else:
        vec = np.array(STAIN_VECTORS.get(stain, (0.5, 0.5, 0.5)))
    # Normalize
    norm = np.sqrt(np.sum(vec ** 2))
    if norm > 0:
        vec = vec / norm
    return vec


def _compute_inverse_absorbance_matrix(stains: List[Tuple[StainType, Optional[Tuple[float, float, float]]]]) -> np.ndarray:
    """Compute the inverse of the absorbance matrix for all stains."""
    absorbance_vectors = []
    for stain, custom_rgb in stains:
        absorbance_vectors.append(_get_absorbance_vector(stain, custom_rgb))
    
    absorbance_matrix = np.array(absorbance_vectors)
    
    # Handle case where we have fewer than 3 stains by padding
    if len(stains) < 3:
        # Pad with orthogonal vectors
        for i in range(3 - len(stains)):
            # Create a residual vector orthogonal to existing ones
            residual = np.array([1.0, 0.0, 0.0]) if i == 0 else np.array([0.0, 1.0, 0.0])
            for vec in absorbance_vectors:
                residual = residual - np.dot(residual, vec) * vec
            norm = np.sqrt(np.sum(residual ** 2))
            if norm > 1e-6:
                residual = residual / norm
            absorbance_vectors.append(residual)
        absorbance_matrix = np.array(absorbance_vectors)
    
    # Compute inverse
    try:
        inverse_matrix = np.linalg.inv(absorbance_matrix)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        inverse_matrix = np.linalg.pinv(absorbance_matrix)
    
    return inverse_matrix


@numpy(contract=ProcessingContract.PURE_2D)
def unmix_colors(
    image: np.ndarray,
    stain1: StainType = StainType.HEMATOXYLIN,
    stain2: StainType = StainType.EOSIN,
    stain3: Optional[StainType] = None,
    output_stain_index: int = 0,
    custom_red_absorbance_1: float = 0.5,
    custom_green_absorbance_1: float = 0.5,
    custom_blue_absorbance_1: float = 0.5,
    custom_red_absorbance_2: float = 0.5,
    custom_green_absorbance_2: float = 0.5,
    custom_blue_absorbance_2: float = 0.5,
    custom_red_absorbance_3: float = 0.5,
    custom_green_absorbance_3: float = 0.5,
    custom_blue_absorbance_3: float = 0.5,
) -> np.ndarray:
    """Unmix colors from a histologically stained RGB image.
    
    Separates dye stains from a color image using color deconvolution,
    producing a grayscale image for the specified stain.
    
    Args:
        image: RGB color image with shape (H, W, 3) or grayscale (H, W).
               Values should be in range [0, 1].
        stain1: First stain type to unmix.
        stain2: Second stain type to unmix.
        stain3: Optional third stain type to unmix.
        output_stain_index: Which stain to output (0, 1, or 2).
        custom_red_absorbance_1: Red absorbance for custom stain 1.
        custom_green_absorbance_1: Green absorbance for custom stain 1.
        custom_blue_absorbance_1: Blue absorbance for custom stain 1.
        custom_red_absorbance_2: Red absorbance for custom stain 2.
        custom_green_absorbance_2: Green absorbance for custom stain 2.
        custom_blue_absorbance_2: Blue absorbance for custom stain 2.
        custom_red_absorbance_3: Red absorbance for custom stain 3.
        custom_green_absorbance_3: Green absorbance for custom stain 3.
        custom_blue_absorbance_3: Blue absorbance for custom stain 3.
    
    Returns:
        Grayscale image (H, W) representing the unmixed stain intensity.
    """
    # Handle grayscale input
    if image.ndim == 2:
        # Convert grayscale to RGB by replicating
        image = np.stack([image, image, image], axis=-1)
    
    # Ensure image is in correct shape (H, W, 3)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
    
    # Build stain list with custom RGB values
    stains = []
    
    custom1 = (custom_red_absorbance_1, custom_green_absorbance_1, custom_blue_absorbance_1) if stain1 == StainType.CUSTOM else None
    stains.append((stain1, custom1))
    
    custom2 = (custom_red_absorbance_2, custom_green_absorbance_2, custom_blue_absorbance_2) if stain2 == StainType.CUSTOM else None
    stains.append((stain2, custom2))
    
    if stain3 is not None:
        custom3 = (custom_red_absorbance_3, custom_green_absorbance_3, custom_blue_absorbance_3) if stain3 == StainType.CUSTOM else None
        stains.append((stain3, custom3))
    
    # Compute inverse absorbance matrix
    inverse_matrix = _compute_inverse_absorbance_matrix(stains)
    
    # Get the inverse absorbance vector for the requested output stain
    inverse_absorbances = inverse_matrix[:, output_stain_index]
    
    # Apply color deconvolution
    # Add small epsilon to avoid log(0)
    eps = 1.0 / 256.0 / 2.0
    image_offset = image + eps
    
    # Log transform
    log_image = np.log(image_offset)
    
    # Multiply by inverse absorbances and sum across channels
    scaled_image = log_image * inverse_absorbances[np.newaxis, np.newaxis, :]
    
    # Exponentiate to get the image without the dye effect
    result = np.exp(np.sum(scaled_image, axis=2))
    
    # Remove the epsilon offset
    result = result - eps
    
    # Clip to valid range
    result = np.clip(result, 0.0, 1.0)
    
    # Invert so that stained regions are bright
    result = 1.0 - result
    
    return result.astype(np.float32)