"""
Image Processor Interface

This module defines the interface for image processors that will be implemented
for different backends (NumPy, CuPy, PyTorch, TensorFlow, JAX).

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit backend requirements
- Clause 106-A — Declared Memory Types: All methods specify memory types
"""

import abc
from typing import Any, List, Optional, Tuple


class ImageProcessorInterface(abc.ABC):
    """
    Interface for image processors.
    
    All implementations must:
    1. Accept 3D arrays as input (Z, Y, X)
    2. Return 3D arrays as output (Z, Y, X)
    3. Be stateless (no instance attributes)
    4. Handle type validation explicitly
    5. Fail loudly when contracts are broken
    """
    
    @classmethod
    @abc.abstractmethod
    def sharpen(cls, image: Any, radius: float = 1.0, amount: float = 1.0) -> Any:
        """
        Sharpen a 3D image using unsharp masking.
        
        This applies sharpening to each Z-slice independently.
        
        Args:
            image: 3D array of shape (Z, Y, X)
            radius: Radius of Gaussian blur
            amount: Sharpening strength
            
        Returns:
            Sharpened 3D array of shape (Z, Y, X)
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def percentile_normalize(cls, image: Any, 
                            low_percentile: float = 1.0, 
                            high_percentile: float = 99.0,
                            target_min: float = 0.0, 
                            target_max: float = 65535.0) -> Any:
        """
        Normalize a 3D image using percentile-based contrast stretching.
        
        This applies normalization to each Z-slice independently.
        
        Args:
            image: 3D array of shape (Z, Y, X)
            low_percentile: Lower percentile (0-100)
            high_percentile: Upper percentile (0-100)
            target_min: Target minimum value
            target_max: Target maximum value
            
        Returns:
            Normalized 3D array of shape (Z, Y, X)
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def stack_percentile_normalize(cls, stack: Any, 
                                  low_percentile: float = 1.0, 
                                  high_percentile: float = 99.0,
                                  target_min: float = 0.0, 
                                  target_max: float = 65535.0) -> Any:
        """
        Normalize a stack using global percentile-based contrast stretching.
        
        This ensures consistent normalization across all Z-slices by computing
        global percentiles across the entire stack.
        
        Args:
            stack: 3D array of shape (Z, Y, X)
            low_percentile: Lower percentile (0-100)
            high_percentile: Upper percentile (0-100)
            target_min: Target minimum value
            target_max: Target maximum value
            
        Returns:
            Normalized 3D array of shape (Z, Y, X)
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def create_composite(cls, images: List[Any], weights: Optional[List[float]] = None) -> Any:
        """
        Create a composite image from multiple 3D arrays.
        
        Args:
            images: List of 3D arrays, each of shape (Z, Y, X)
            weights: List of weights for each image. If None, equal weights are used.
            
        Returns:
            Composite 3D array of shape (Z, Y, X)
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def apply_mask(cls, image: Any, mask: Any) -> Any:
        """
        Apply a mask to a 3D image.
        
        This applies the mask to each Z-slice independently.
        
        Args:
            image: 3D array of shape (Z, Y, X)
            mask: 3D array of shape (Z, Y, X) or 2D array of shape (Y, X)
            
        Returns:
            Masked 3D array of shape (Z, Y, X)
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def create_weight_mask(cls, shape: Tuple[int, int], margin_ratio: float = 0.1) -> Any:
        """
        Create a weight mask for blending images.
        
        Args:
            shape: Shape of the mask (height, width)
            margin_ratio: Ratio of image size to use as margin
            
        Returns:
            2D weight mask of shape (Y, X)
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def max_projection(cls, stack: Any) -> Any:
        """
        Create a maximum intensity projection from a Z-stack.
        
        Args:
            stack: 3D array of shape (Z, Y, X)
            
        Returns:
            2D array of shape (Y, X)
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def mean_projection(cls, stack: Any) -> Any:
        """
        Create a mean intensity projection from a Z-stack.
        
        Args:
            stack: 3D array of shape (Z, Y, X)
            
        Returns:
            2D array of shape (Y, X)
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def stack_equalize_histogram(cls, stack: Any, 
                                bins: int = 65536, 
                                range_min: float = 0.0, 
                                range_max: float = 65535.0) -> Any:
        """
        Apply histogram equalization to an entire stack.
        
        This ensures consistent contrast enhancement across all Z-slices by
        computing a global histogram across the entire stack.
        
        Args:
            stack: 3D array of shape (Z, Y, X)
            bins: Number of bins for histogram computation
            range_min: Minimum value for histogram range
            range_max: Maximum value for histogram range
            
        Returns:
            Equalized 3D array of shape (Z, Y, X)
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def create_projection(cls, stack: Any, method: str = "max_projection") -> Any:
        """
        Create a projection from a stack using the specified method.
        
        Args:
            stack: 3D array of shape (Z, Y, X)
            method: Projection method (max_projection, mean_projection)
            
        Returns:
            2D array of shape (Y, X)
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def tophat(cls, image: Any, 
              selem_radius: int = 50, 
              downsample_factor: int = 4) -> Any:
        """
        Apply white top-hat filter to a 3D image for background removal.
        
        This applies the filter to each Z-slice independently.
        
        Args:
            image: 3D array of shape (Z, Y, X)
            selem_radius: Radius of the structuring element disk
            downsample_factor: Factor by which to downsample the image for processing
            
        Returns:
            Filtered 3D array of shape (Z, Y, X)
        """
        pass
