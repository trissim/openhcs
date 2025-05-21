"""
Image processing utilities for the OpenHCS pipeline.

This module provides core image processing functions that are used by various steps
in the processing pipeline. All functions are pure and stateless.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy
- Clause 65 — No Fallback Logic 
- Clause 88 — No Inferred Capabilities
- Clause 244 — Rot Intolerance
"""

from typing import List, Optional

import numpy as np


class ImageProcessor:
    """
    Static class providing image processing utilities.
    All methods are pure functions with no state.
    """
    
    @staticmethod
    def create_composite(images: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Create a composite image from multiple channels with optional weights.
        
        Args:
            images: List of input images (must be same shape)
            weights: Optional list of weights for each image (default: equal weights)
            
        Returns:
            Composite image as weighted sum of inputs
            
        Raises:
            ValueError: If images have different shapes or weights length mismatch
        """
        if not images:
            raise ValueError("At least one image required for compositing")
            
        if weights is None:
            weights = [1.0/len(images)] * len(images)
        elif len(weights) != len(images):
            raise ValueError(f"Weights length ({len(weights)}) must match images length ({len(images)})")
            
        # Validate all images same shape
        first_shape = images[0].shape
        for img in images[1:]:
            if img.shape != first_shape:
                raise ValueError("All input images must have the same shape")
                
        # Convert to float32 for precision
        images_float = [img.astype(np.float32) for img in images]
        
        # Compute weighted sum
        composite = np.zeros_like(images_float[0], dtype=np.float32)
        for img, weight in zip(images_float, weights):
            composite += img * weight
            
        # Clip and convert back to uint16
        return np.clip(composite, 0, 65535).astype(np.uint16)