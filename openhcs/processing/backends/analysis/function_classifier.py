"""
Shared function classification utilities for OpenHCS registries.

This module provides:
1. Blacklist system to skip problematic functions during runtime testing
2. Docstring-based classification (for future use)
3. Shared utilities across scikit-image, pyclesperanto, and CuPy registries
"""

from enum import Enum
from typing import Callable, Tuple, Optional, Set
import inspect
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Blacklist System
# =============================================================================

# Known problematic functions that hang during runtime testing
FUNCTION_BLACKLIST: Set[str] = {
    # Scikit-image functions that hang
    'marching_cubes',
    'marching_cubes_lewiner',
    'marching_cubes_classic',
    'mesh_surface_area',  # Hangs after marching_cubes - expects mesh input

    # Footprint/kernel generators - these return kernels, not processed images
    # Should not be in an image processing registry
    'ball',
    'cube',
    'diamond',
    'disk',
    'ellipse',
    'footprint_rectangle',
    'octagon',
    'rectangle',
    'square',
    'star',

    # CuCIM versions of footprint generators
    'morphology_ball',
    'morphology_cube',
    'morphology_diamond',
    'morphology_disk',
    'morphology_footprint_rectangle',
    'morphology_octagon',
    'morphology_rectangle',
    'morphology_square',
    'morphology_star',

    # Complex algorithms requiring specific parameters that can't be generically provided
    'active_contour',
    'chan_vese',
    'morphological_chan_vese',
    'morphological_geodesic_active_contour',
    'random_walker',
    'flood',
    'flood_fill',
    'estimate_transform',
    'calibrate_denoiser',
    'ransac',

    # Utility functions that don't process images (not actual image processing)
    'map_array',
    'dtype_limits',
    'xyz_tristimulus_values',

    # Add more problematic functions as discovered
    # 'function_name',
}

# Module-based blacklist for entire modules that are problematic
MODULE_BLACKLIST: Set[str] = {
    # Scikit-image modules that contain mostly measurement/analysis functions
    # These don't fit OpenHCS 3D array → 3D array contract
    'skimage.measure',  # Contains measurement functions that return scalars/tables, not arrays
}

def is_blacklisted(func: Callable, func_name: str) -> bool:
    """
    Check if a function should be skipped during registration.
    
    Args:
        func: Function object to check
        func_name: Name of the function
        
    Returns:
        True if function should be skipped, False otherwise
    """
    # Check function name blacklist
    if func_name in FUNCTION_BLACKLIST:
        logger.info(f"Skipping blacklisted function: {func_name}")
        return True
    
    # Check actual function name (in case func_name is different)
    if hasattr(func, '__name__') and func.__name__ in FUNCTION_BLACKLIST:
        logger.info(f"Skipping blacklisted function: {func.__name__} (called as {func_name})")
        return True
    
    # Check module blacklist
    if hasattr(func, '__module__') and func.__module__:
        for blacklisted_module in MODULE_BLACKLIST:
            if blacklisted_module in func.__module__:
                logger.info(f"Skipping function from blacklisted module: {func_name} from {func.__module__}")
                return True
    
    return False

def add_to_blacklist(func_name: str) -> None:
    """
    Add a function name to the blacklist.
    
    Args:
        func_name: Name of function to blacklist
    """
    FUNCTION_BLACKLIST.add(func_name)
    logger.info(f"Added {func_name} to function blacklist")

def remove_from_blacklist(func_name: str) -> None:
    """
    Remove a function name from the blacklist.
    
    Args:
        func_name: Name of function to remove from blacklist
    """
    FUNCTION_BLACKLIST.discard(func_name)
    logger.info(f"Removed {func_name} from function blacklist")

def get_blacklist() -> Set[str]:
    """Get current blacklist for inspection."""
    return FUNCTION_BLACKLIST.copy()

# =============================================================================
# Docstring-Based Classification (Future Use)
# =============================================================================

class GenericContract(Enum):
    """Generic processing contract types for cross-registry compatibility."""
    SLICE_SAFE = "slice_safe"    # 2D→2D: Apply to each Z-slice independently
    CROSS_Z = "cross_z"          # 3D→3D: Process entire 3D volume
    DIM_CHANGE = "dim_change"    # 3D→2D: Reduce dimensionality
    UNKNOWN = "unknown"          # Could not determine

# Keyword mappings for docstring analysis
DIM_CHANGE_KEYWORDS = {
    'surface', 'mesh', 'projection', 'histogram', 'statistics', 'measure', 
    'properties', 'regionprops', 'moments', 'centroid', 'area', 'perimeter',
    'convex', 'hull', 'skeleton', 'profile', 'plot', 'graph'
}

CROSS_Z_KEYWORDS = {
    '3d', 'volume', 'volumetric', 'connectivity', 'watershed', 'label',
    'connected components', 'connected_components', 'binary_fill_holes',
    'remove_small_holes', 'remove_small_objects', 'flood', 'flood_fill'
}

SLICE_SAFE_KEYWORDS = {
    'filter', 'edge', 'gradient', 'morphology', 'threshold', 'enhancement',
    'blur', 'sharpen', 'denoise', 'smooth', 'gaussian', 'median', 'bilateral',
    'sobel', 'prewitt', 'roberts', 'canny', 'laplacian', 'unsharp'
}

def classify_by_docstring(func: Callable, func_name: Optional[str] = None) -> Tuple[GenericContract, bool]:
    """
    Classify function behavior using docstring and signature analysis.
    
    This is currently implemented for future use but not used in the current
    registry building process. The proven runtime testing is still used.
    
    Args:
        func: Function to classify
        func_name: Function name for context (optional)
        
    Returns:
        (contract, is_valid): Generic classification result
    """
    if func_name is None:
        func_name = getattr(func, '__name__', 'unknown')
    
    try:
        # Get docstring
        docstring = func.__doc__ or ""
        docstring_lower = docstring.lower()
        
        # Get module information
        module_name = getattr(func, '__module__', '') or ''
        
        # Check for dimension-changing indicators
        if any(keyword in docstring_lower for keyword in DIM_CHANGE_KEYWORDS):
            logger.debug(f"Classified {func_name} as DIM_CHANGE based on docstring keywords")
            return GenericContract.DIM_CHANGE, True
        
        # Check module hints for dimension-changing functions
        if any(module_hint in module_name for module_hint in ['measure', 'feature']):
            # Functions in measure/feature modules often return non-array results
            logger.debug(f"Classified {func_name} as DIM_CHANGE based on module: {module_name}")
            return GenericContract.DIM_CHANGE, True
        
        # Check for 3D/volumetric processing indicators
        if any(keyword in docstring_lower for keyword in CROSS_Z_KEYWORDS):
            logger.debug(f"Classified {func_name} as CROSS_Z based on docstring keywords")
            return GenericContract.CROSS_Z, True
        
        # Check for slice-safe indicators
        if any(keyword in docstring_lower for keyword in SLICE_SAFE_KEYWORDS):
            logger.debug(f"Classified {func_name} as SLICE_SAFE based on docstring keywords")
            return GenericContract.SLICE_SAFE, True
        
        # Check function signature for hints
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Functions with 'axis' parameter often support both 2D and 3D
            if 'axis' in params:
                logger.debug(f"Classified {func_name} as SLICE_SAFE based on 'axis' parameter")
                return GenericContract.SLICE_SAFE, True
                
        except (ValueError, TypeError):
            # Signature inspection failed
            pass
        
        # Default to SLICE_SAFE for traditional image processing
        logger.debug(f"Classified {func_name} as SLICE_SAFE (default)")
        return GenericContract.SLICE_SAFE, True
        
    except Exception as e:
        logger.warning(f"Failed to classify {func_name} by docstring: {e}")
        return GenericContract.UNKNOWN, False

def analyze_function_docstring(func: Callable, func_name: Optional[str] = None) -> dict:
    """
    Analyze function docstring and return detailed information.
    
    This is a utility function for research and development purposes.
    
    Args:
        func: Function to analyze
        func_name: Function name for context
        
    Returns:
        Dictionary with analysis results
    """
    if func_name is None:
        func_name = getattr(func, '__name__', 'unknown')
    
    analysis = {
        'function_name': func_name,
        'module': getattr(func, '__module__', 'unknown'),
        'docstring': func.__doc__ or '',
        'docstring_first_line': '',
        'found_keywords': {
            'dim_change': [],
            'cross_z': [],
            'slice_safe': []
        },
        'signature': '',
        'parameters': [],
        'classification': GenericContract.UNKNOWN,
        'confidence': 'low'
    }
    
    try:
        # Extract first line of docstring
        if analysis['docstring']:
            analysis['docstring_first_line'] = analysis['docstring'].split('\n')[0].strip()
        
        # Find keywords
        docstring_lower = analysis['docstring'].lower()
        
        for keyword in DIM_CHANGE_KEYWORDS:
            if keyword in docstring_lower:
                analysis['found_keywords']['dim_change'].append(keyword)
        
        for keyword in CROSS_Z_KEYWORDS:
            if keyword in docstring_lower:
                analysis['found_keywords']['cross_z'].append(keyword)
        
        for keyword in SLICE_SAFE_KEYWORDS:
            if keyword in docstring_lower:
                analysis['found_keywords']['slice_safe'].append(keyword)
        
        # Get signature information
        try:
            sig = inspect.signature(func)
            analysis['signature'] = str(sig)
            analysis['parameters'] = list(sig.parameters.keys())
        except (ValueError, TypeError):
            analysis['signature'] = 'unavailable'
        
        # Determine classification
        classification, is_valid = classify_by_docstring(func, func_name)
        analysis['classification'] = classification
        analysis['confidence'] = 'high' if is_valid else 'low'
        
    except Exception as e:
        analysis['error'] = str(e)
    
    return analysis

# =============================================================================
# Registry Integration Utilities
# =============================================================================

def log_blacklist_stats() -> None:
    """Log current blacklist statistics."""
    logger.info(f"Function blacklist contains {len(FUNCTION_BLACKLIST)} entries:")
    for func_name in sorted(FUNCTION_BLACKLIST):
        logger.info(f"  - {func_name}")

    if MODULE_BLACKLIST:
        logger.info(f"Module blacklist contains {len(MODULE_BLACKLIST)} entries:")
        for module_pattern in sorted(MODULE_BLACKLIST):
            logger.info(f"  - {module_pattern}")

def add_problematic_functions(*func_names: str) -> None:
    """
    Convenience function to add multiple problematic functions to blacklist.

    Usage:
        add_problematic_functions('func1', 'func2', 'func3')
    """
    for func_name in func_names:
        add_to_blacklist(func_name)

    print(f"Added {len(func_names)} functions to blacklist:")
    for func_name in func_names:
        print(f"  - {func_name}")

    log_blacklist_stats()
