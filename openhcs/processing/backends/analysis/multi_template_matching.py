"""
Multi-Template Matching functions for OpenHCS.

This module provides template matching capabilities using the Multi-Template-Matching library
to detect and crop regions of interest in image stacks.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import json
import pandas as pd

try:
    import MTM
except ImportError:
    MTM = None
    logging.warning("MTM (Multi-Template-Matching) not available. Install with: pip install Multi-Template-Matching")

from openhcs.core.memory.decorators import numpy as numpy_func
from openhcs.core.pipeline.function_contracts import special_outputs


@dataclass
class TemplateMatchResult:
    """Results for a single slice template matching operation."""
    slice_index: int
    matches: List[Dict[str, Any]]  # List of hits from MTM.matchTemplates
    best_match: Optional[Dict[str, Any]]  # Best scoring match
    crop_bbox: Optional[Tuple[int, int, int, int]]  # (x, y, width, height) if cropped
    match_score: float
    num_matches: int
    best_rotation_angle: float  # Angle of best matching template
    error_message: Optional[str] = None


@numpy_func
@special_outputs("match_results")
def multi_template_crop(
    image_stack: np.ndarray,
    template_path: str,
    score_threshold: float = 0.8,
    max_matches: int = 1,
    crop_margin: int = 0,
    method: str = "cv2.TM_CCOEFF_NORMED",
    use_best_match_only: bool = True,
    normalize_input: bool = True,
    pad_mode: str = "constant",
    rotation_range: float = 0.0,
    rotation_step: float = 45.0,
    rotate_result: bool = True,
    crop_enabled: bool = True
) -> Tuple[np.ndarray, List[TemplateMatchResult]]:
    """
    Perform multi-template matching on each slice of a 3D image stack and return cropped regions.
    
    This function applies template matching to each Z-slice independently, finds the best matches,
    and crops the regions around the matched templates. All cropped regions are stacked back into
    a 3D array with consistent dimensions.
    
    Parameters
    ----------
    image_stack : np.ndarray
        3D array of shape (Z, Y, X) containing the image slices to process
    template_path : str
        Path to the template image file (supports common formats: PNG, JPEG, TIFF)
    score_threshold : float, default=0.8
        Minimum correlation score for template matches (0.0 to 1.0)
    max_matches : int, default=1
        Maximum number of matches to find per slice
    crop_margin : int, default=0
        Additional pixels to include around the matched template region
    method : str, default="cv2.TM_CCOEFF_NORMED"
        OpenCV template matching method (currently not used by MTM)
    use_best_match_only : bool, default=True
        If True, only crop around the best match per slice
    normalize_input : bool, default=True
        Whether to normalize input slices to uint8 range for MTM processing
    pad_mode : str, default="constant"
        Padding mode for size normalization ('constant', 'edge', 'reflect', etc.)
    rotation_range : float, default=0.0
        Total rotation range in degrees (e.g., 360.0 for full rotation)
    rotation_step : float, default=45.0
        Rotation increment in degrees (e.g., 45.0 for 8 orientations)
    rotate_result : bool, default=True
        Whether to rotate cropped results back to upright orientation
    crop_enabled : bool, default=True
        Whether to crop regions around matches. If False, returns original stack with match results

    Returns
    -------
    cropped_stack : np.ndarray
        3D array of cropped regions stacked together (if crop_enabled=True),
        or original image stack (if crop_enabled=False)
    match_results : List[TemplateMatchResult]
        Detailed results for each slice including match info and crop coordinates
        
    Raises
    ------
    ImportError
        If MTM library is not installed
    ValueError
        If template image cannot be loaded or input dimensions are invalid
    """
    
    if MTM is None:
        raise ImportError("MTM library not available. Install with: pip install Multi-Template-Matching")
    
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D image stack, got {image_stack.ndim}D array")
    
    # Load template image
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Could not load template image from {template_path}")
    
    logging.info(f"Loaded template of size {template.shape} from {template_path}")

    # Generate rotated templates if rotation is enabled
    if rotation_range > 0:
        template_list = _create_rotated_templates(template, rotation_range, rotation_step)
        logging.info(f"Generated {len(template_list)} rotated templates (range: {rotation_range}°, step: {rotation_step}°)")
    else:
        template_list = [("template_0", template)]
    
    # Results storage
    cropped_slices = []
    match_results = []
    
    logging.info(f"Processing {image_stack.shape[0]} slices with template matching")
    
    # Process each slice
    for z_idx in range(image_stack.shape[0]):
        slice_img = image_stack[z_idx]
        result = _process_single_slice(
            slice_img, 
            template_list, 
            z_idx,
            score_threshold,
            max_matches,
            crop_margin,
            use_best_match_only,
            normalize_input
        )
        
        match_results.append(result)

        # Extract cropped slice from result or use original slice
        if crop_enabled and result.crop_bbox is not None:
            x, y, w, h = result.crop_bbox
            cropped_slice = slice_img[y:y+h, x:x+w]

            # Rotate cropped slice back to upright if rotation was used
            if rotate_result and result.best_rotation_angle != 0:
                cropped_slice = _rotate_image(cropped_slice, -result.best_rotation_angle)
        else:
            # Use original slice (either cropping disabled or no match found)
            cropped_slice = slice_img

        cropped_slices.append(cropped_slice)
    
    # Stack slices with consistent dimensions (only pad if cropping was enabled)
    if crop_enabled:
        cropped_stack = _stack_with_padding(cropped_slices, pad_mode)
        logging.info(f"Template matching complete. Cropped output shape: {cropped_stack.shape}")
    else:
        # Return original stack when cropping is disabled
        cropped_stack = image_stack
        logging.info(f"Template matching complete. Original stack shape preserved: {cropped_stack.shape}")

    return cropped_stack, match_results


def materialize_mtm_match_results(data: List[TemplateMatchResult], path: str, filemanager) -> str:
    """Materialize MTM match results as analysis-ready CSV with confidence analysis."""
    csv_path = path.replace('.pkl', '_mtm_matches.csv')

    rows = []
    for result in data:
        slice_idx = result.slice_index

        # Process all matches for this slice
        for i, match in enumerate(result.matches):
            rows.append({
                'slice_index': slice_idx,
                'match_id': f"slice_{slice_idx}_match_{i}",
                'bbox_x': match.get('BBox', [0, 0, 0, 0])[0],
                'bbox_y': match.get('BBox', [0, 0, 0, 0])[1],
                'bbox_width': match.get('BBox', [0, 0, 0, 0])[2],
                'bbox_height': match.get('BBox', [0, 0, 0, 0])[3],
                'confidence_score': match.get('Score', 0.0),
                'template_name': match.get('TemplateName', 'unknown'),
                'is_best_match': (match == result.best_match),
                'was_cropped': result.crop_bbox is not None
            })

    if rows:
        df = pd.DataFrame(rows)

        # Add analysis columns
        if len(df) > 0 and 'confidence_score' in df.columns:
            df['high_confidence'] = df['confidence_score'] > 0.8
            df['confidence_quartile'] = pd.qcut(df['confidence_score'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

            # Add spatial clustering if we have position data
            if 'bbox_x' in df.columns and 'bbox_y' in df.columns:
                try:
                    from sklearn.cluster import KMeans
                    if len(df) >= 3:
                        coords = df[['bbox_x', 'bbox_y']].values
                        n_clusters = min(3, len(df))
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        df['spatial_cluster'] = kmeans.fit_predict(coords)
                    else:
                        df['spatial_cluster'] = 0
                except ImportError:
                    df['spatial_cluster'] = 0

        csv_content = df.to_csv(index=False)
        filemanager.save(csv_content, csv_path, "disk")

    return csv_path


def _create_rotated_templates(template: np.ndarray, rotation_range: float, rotation_step: float) -> List[Tuple[str, np.ndarray]]:
    """Create rotated versions of a template."""
    templates = []

    # Generate rotation angles
    if rotation_range >= 360:
        # Full rotation
        angles = np.arange(0, 360, rotation_step)
    else:
        # Symmetric range around 0
        half_range = rotation_range / 2
        angles = np.arange(-half_range, half_range + rotation_step, rotation_step)

    for angle in angles:
        rotated_template = _rotate_image(template, angle)
        templates.append((f"template_{angle:.1f}", rotated_template))

    return templates


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by the specified angle in degrees."""
    if angle == 0:
        return image

    # Get image center
    center = (image.shape[1] // 2, image.shape[0] // 2)

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding dimensions
    cos_val = np.abs(rotation_matrix[0, 0])
    sin_val = np.abs(rotation_matrix[0, 1])
    new_width = int((image.shape[0] * sin_val) + (image.shape[1] * cos_val))
    new_height = int((image.shape[0] * cos_val) + (image.shape[1] * sin_val))

    # Adjust rotation matrix for new center
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Perform rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return rotated


def _process_single_slice(
    slice_img: np.ndarray,
    template_list: List[Tuple[str, np.ndarray]],
    z_idx: int,
    score_threshold: float,
    max_matches: int,
    crop_margin: int,
    use_best_match_only: bool,
    normalize_input: bool
) -> TemplateMatchResult:
    """Process a single slice for template matching."""

    try:
        # Prepare slice for MTM
        if normalize_input and slice_img.dtype != np.uint8:
            # Normalize to 0-255 range
            slice_min, slice_max = slice_img.min(), slice_img.max()
            if slice_max > slice_min:
                slice_normalized = ((slice_img - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
            else:
                slice_normalized = np.zeros_like(slice_img, dtype=np.uint8)
        else:
            slice_normalized = slice_img.astype(np.uint8)

        # Perform template matching
        hits = MTM.matchTemplates(
            slice_normalized,
            template_list,
            score_threshold=score_threshold,
            maxOverlap=0.25,  # Prevent overlapping matches
            N_object=max_matches
        )

        # Process results
        best_match = None
        crop_bbox = None
        best_rotation_angle = 0.0

        if hits:
            # Sort by score (hits format: [label, bbox, score])
            hits_sorted = sorted(hits, key=lambda x: x[2], reverse=True)
            best_match = hits_sorted[0] if hits_sorted else None

            if best_match and use_best_match_only:
                # Extract rotation angle from template label
                template_label = best_match[0]
                if template_label.startswith("template_"):
                    try:
                        best_rotation_angle = float(template_label.split("_")[1])
                    except (IndexError, ValueError):
                        best_rotation_angle = 0.0

                # Extract bounding box (x, y, width, height)
                bbox = best_match[1]
                x, y, w, h = bbox

                # Apply margin and clamp to image bounds
                x_start = max(0, x - crop_margin)
                y_start = max(0, y - crop_margin)
                x_end = min(slice_img.shape[1], x + w + crop_margin)
                y_end = min(slice_img.shape[0], y + h + crop_margin)

                crop_bbox = (x_start, y_start, x_end - x_start, y_end - y_start)

        # Create result
        return TemplateMatchResult(
            slice_index=z_idx,
            matches=hits,
            best_match=best_match,
            crop_bbox=crop_bbox,
            match_score=best_match[2] if best_match else 0.0,
            num_matches=len(hits),
            best_rotation_angle=best_rotation_angle
        )

    except Exception as e:
        # Handle MTM errors gracefully
        error_msg = f"Template matching failed for slice {z_idx}: {e}"
        logging.warning(error_msg)

        return TemplateMatchResult(
            slice_index=z_idx,
            matches=[],
            best_match=None,
            crop_bbox=(0, 0, slice_img.shape[1], slice_img.shape[0]),
            match_score=0.0,
            num_matches=0,
            best_rotation_angle=0.0,
            error_message=error_msg
        )


def _stack_with_padding(cropped_slices: List[np.ndarray], pad_mode: str) -> np.ndarray:
    """Stack cropped slices with padding to ensure consistent dimensions."""

    if not cropped_slices:
        raise ValueError("No cropped slices to stack")

    # Find maximum dimensions
    max_h = max(slice_arr.shape[0] for slice_arr in cropped_slices)
    max_w = max(slice_arr.shape[1] for slice_arr in cropped_slices)

    # Pad all slices to same size
    padded_slices = []
    for slice_arr in cropped_slices:
        h, w = slice_arr.shape
        pad_h = max_h - h
        pad_w = max_w - w

        if pad_h > 0 or pad_w > 0:
            # Pad with specified mode
            padded = np.pad(
                slice_arr,
                ((0, pad_h), (0, pad_w)),
                mode=pad_mode,
                constant_values=0 if pad_mode == 'constant' else None
            )
        else:
            padded = slice_arr

        padded_slices.append(padded)

    # Stack into 3D array
    return np.stack(padded_slices, axis=0)
