"""
Automated CuPy ndimage function registry for OpenHCS.

This module automatically discovers, analyzes, and registers CuPy ndimage functions
with proper processing contracts and array-in/array-out compliance.
CuPy functions generally have better dtype preservation than scikit-image.
"""

import inspect
import cupy as cp
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from functools import wraps
import warnings

# Suppress CuPy warnings during analysis
warnings.filterwarnings('ignore', category=FutureWarning, module='cupyx')


# ===== FUNCTION CATEGORIZATION FOR ARRAY-IN/ARRAY-OUT COMPLIANCE =====

# Category 1: Pure array functions - input 3D array, output 3D array (preserve dtype)
PURE_ARRAY_FUNCTIONS = {
    'gaussian_filter', 'gaussian_filter1d', 'median_filter', 'sobel', 'prewitt', 'laplace',
    'uniform_filter', 'uniform_filter1d', 'maximum_filter', 'minimum_filter', 
    'percentile_filter', 'rank_filter', 'convolve', 'correlate', 'shift', 'rotate',
    'grey_erosion', 'grey_dilation', 'grey_opening', 'grey_closing',
    'black_tophat', 'white_tophat', 'morphological_gradient', 'morphological_laplace',
    'gaussian_gradient_magnitude', 'gaussian_laplace', 'spline_filter', 'spline_filter1d',
    'convolve1d', 'correlate1d', 'maximum_filter1d', 'minimum_filter1d',
    'affine_transform', 'map_coordinates', 'zoom', 'distance_transform_edt'
}

# Category 2: Value-returning functions - return (array, computed_value)
VALUE_RETURNING_FUNCTIONS = {
    'center_of_mass', 'extrema', 'maximum_position', 'minimum_position',
    'mean', 'median', 'standard_deviation', 'variance', 'sum', 'histogram',
    'sum_labels', 'labeled_comprehension', 'value_indices'
}

# Category 3: Mask-returning functions - return mask as array with proper dtype
MASK_RETURNING_FUNCTIONS = {
    'binary_erosion', 'binary_dilation', 'binary_opening', 'binary_closing',
    'binary_fill_holes', 'binary_hit_or_miss', 'binary_propagation', 'label'
}


def _scale_and_convert_cupy(result, target_dtype):
    """
    Scale CuPy results to target integer range and convert dtype.
    
    CuPy functions generally preserve dtypes better than scikit-image,
    but we still need this for edge cases.
    """
    if not hasattr(result, 'dtype'):
        return result
        
    # If result is floating point and target is integer, scale appropriately
    if cp.issubdtype(result.dtype, cp.floating) and not cp.issubdtype(target_dtype, cp.floating):
        # Clip to [0, 1] range and scale to integer range
        clipped = cp.clip(result, 0, 1)
        if target_dtype == cp.uint8:
            return (clipped * 255).astype(target_dtype)
        elif target_dtype == cp.uint16:
            return (clipped * 65535).astype(target_dtype)
        elif target_dtype == cp.uint32:
            return (clipped * 4294967295).astype(target_dtype)
        else:
            # For other integer types, just convert without scaling
            return result.astype(target_dtype)
    
    # Direct conversion for same numeric type families
    return result.astype(target_dtype)


def _auto_detect_cupy_array_compliance(result, original_image, original_dtype):
    """
    Ensure unknown CuPy functions return 3D arrays following array-in/array-out pattern.
    """
    # If result is already a 3D array, just fix dtype if needed
    if isinstance(result, cp.ndarray) and result.ndim == 3:
        return _scale_and_convert_cupy(result, original_dtype)
    
    # If result is scalar/coordinates, return (original_image, result)
    elif cp.isscalar(result) or (isinstance(result, (cp.ndarray, tuple)) and 
                                 (not hasattr(result, 'ndim') or result.ndim < 3)):
        return original_image, result
    
    # If result is 2D, expand to 3D (assume same for all Z slices)
    elif isinstance(result, cp.ndarray) and result.ndim == 2:
        expanded = cp.stack([result] * original_image.shape[0], axis=0)
        return _scale_and_convert_cupy(expanded, original_dtype)
    
    # Fallback: return original image
    else:
        return original_image


class ProcessingContract(Enum):
    """Processing contract types for CuPy functions."""
    SLICE_SAFE = "slice_safe"  # Process each Z-slice independently
    CROSS_Z = "cross_z"        # Process entire 3D volume
    DIM_CHANGE = "dim_change"  # Change output dimensionality
    UNKNOWN = "unknown"        # Could not determine


def _create_cupy_array_compliant_wrapper(original_func, func_name):
    """
    Create a wrapper that ensures array-in/array-out compliance and dtype preservation for CuPy functions.
    
    All OpenHCS functions must:
    1. Take 3D CuPy array as first argument
    2. Return 3D CuPy array as first output
    3. Additional outputs (values, coordinates) as 2nd, 3rd, etc. returns
    4. Preserve input dtype when appropriate
    """
    @wraps(original_func)
    def cupy_array_compliant_wrapper(image_3d, *args, **kwargs):
        original_dtype = image_3d.dtype
        result = original_func(image_3d, *args, **kwargs)
        
        # Category 1: Pure array functions (sobel, gaussian, etc.)
        if func_name in PURE_ARRAY_FUNCTIONS:
            # CuPy functions usually preserve dtype already, but ensure it
            if hasattr(result, 'dtype') and result.dtype != original_dtype:
                return _scale_and_convert_cupy(result, original_dtype)
            return result
        
        # Category 2: Value-returning functions (center_of_mass, etc.)
        elif func_name in VALUE_RETURNING_FUNCTIONS:
            # Return (pass_through_array, computed_value)
            return image_3d, result
        
        # Category 3: Mask-returning functions (binary_erosion, etc.)
        elif func_name in MASK_RETURNING_FUNCTIONS:
            # Convert bool mask to original dtype range if needed
            if hasattr(result, 'dtype') and result.dtype == cp.bool_:
                if original_dtype == cp.uint16:
                    return result.astype(cp.uint16) * 65535
                elif original_dtype == cp.uint8:
                    return result.astype(cp.uint8) * 255
                else:
                    return result.astype(original_dtype)
            return _scale_and_convert_cupy(result, original_dtype)
        
        # Auto-detect for unknown functions
        else:
            return _auto_detect_cupy_array_compliance(result, image_3d, original_dtype)
    
    # Preserve function metadata
    cupy_array_compliant_wrapper.__name__ = original_func.__name__
    cupy_array_compliant_wrapper.__module__ = original_func.__module__
    cupy_array_compliant_wrapper.__doc__ = original_func.__doc__
    
    return cupy_array_compliant_wrapper


@dataclass
class CupyFunction:
    """Metadata for a CuPy ndimage function."""
    name: str
    func: Callable
    module: str
    contract: ProcessingContract
    supports_3d: bool
    doc: str = ""
    signature: str = ""


def _test_cupy_3d_behavior(func, func_name):
    """
    Test how a CuPy function behaves with 3D input to determine its processing contract.
    """
    # Create test data
    test_3d = cp.random.rand(3, 20, 20).astype(cp.float32)
    test_2d = test_3d[0]
    
    try:
        # Test if function accepts 3D input
        result_3d = func(test_3d)
        result_2d = func(test_2d)
        
        # Check if shapes are preserved
        if hasattr(result_3d, 'shape'):
            if result_3d.shape == ():
                # Scalar result - this is a value-returning function
                return ProcessingContract.SLICE_SAFE, True
            elif result_3d.shape != test_3d.shape:
                # Output shape changed - likely dimension-changing function
                return ProcessingContract.DIM_CHANGE, True
        else:
            # No shape attribute - this is a value-returning function
            return ProcessingContract.SLICE_SAFE, True
        
        # Test if processing is slice-by-slice
        manual_3d = cp.stack([func(test_3d[z]) for z in range(test_3d.shape[0])])
        
        if cp.allclose(result_3d, manual_3d, rtol=1e-5, atol=1e-8):
            # Results match slice-by-slice processing
            return ProcessingContract.SLICE_SAFE, True
        else:
            # Different results - likely volumetric processing
            return ProcessingContract.CROSS_Z, True
            
    except Exception as e:
        # Function failed on 3D input
        error_msg = str(e).lower()
        
        # Check if it's a dimension error
        if any(keyword in error_msg for keyword in ['dimension', 'shape', '3d', 'axis']):
            return ProcessingContract.SLICE_SAFE, False  # 2D only, but slice-safe
        else:
            return ProcessingContract.UNKNOWN, False


def build_cupy_registry() -> Dict[str, CupyFunction]:
    """
    Build a registry of CuPy ndimage functions with their metadata.
    """
    try:
        import cupyx.scipy.ndimage as ndimage
    except ImportError:
        print("⚠️  CuPy not available - skipping CuPy registry")
        return {}
    
    print("🔍 Analyzing CuPy ndimage functions...")
    
    registry = {}
    analyzed_functions = 0
    
    # Get all functions from cupyx.scipy.ndimage
    for name in dir(ndimage):
        if not name.startswith('_') and callable(getattr(ndimage, name)):
            func = getattr(ndimage, name)
            if hasattr(func, '__module__') and 'cupyx.scipy.ndimage' in func.__module__:
                try:
                    # Test 3D behavior
                    contract, supports_3d = _test_cupy_3d_behavior(func, name)
                    
                    # Get documentation
                    doc = (func.__doc__ or "").split('\n')[0].strip()
                    
                    # Get signature
                    try:
                        signature = str(inspect.signature(func))
                    except:
                        signature = "signature unavailable"
                    
                    # Create function metadata
                    cupy_func = CupyFunction(
                        name=name,
                        func=func,
                        module=func.__module__,
                        contract=contract,
                        supports_3d=supports_3d,
                        doc=doc,
                        signature=signature
                    )
                    
                    registry[name] = cupy_func
                    analyzed_functions += 1
                    
                except Exception as e:
                    print(f"    ⚠️  Failed to analyze {name}: {e}")
    
    print(f"✅ Analyzed {analyzed_functions} CuPy ndimage functions")
    return registry


def _register_cupy_ops_direct() -> None:
    """
    Direct decoration of CuPy ndimage functions - SIMPLE APPROACH.

    Just add memory type attributes directly to the original functions.
    No wrappers, no complexity - just make external functions BE OpenHCS functions.

    This is called during Phase 2 of registry initialization to avoid circular dependencies.
    """
    try:
        import cupyx.scipy.ndimage as ndimage
    except ImportError:
        print("⚠️  CuPy not available - skipping CuPy registration")
        return

    from openhcs.processing.func_registry import _register_function

    print("🔧 Direct decoration of CuPy ndimage functions - SIMPLE APPROACH...")

    decorated_count = 0
    skipped_count = 0

    # Get functions using build_cupy_registry
    registry = build_cupy_registry()

    for func_name, meta in registry.items():
        try:
            # Skip functions that don't support arrays or have unknown contracts
            if meta.contract == ProcessingContract.UNKNOWN:
                skipped_count += 1
                continue

            # Skip dimension-changing functions (OpenHCS requires array-in/array-out)
            if meta.contract == ProcessingContract.DIM_CHANGE:
                skipped_count += 1
                continue

            # SIMPLE: Just add memory type attributes directly to the original function
            original_func = meta.func

            # Add memory type attributes - this makes it an OpenHCS function
            from openhcs.constants import MemoryType
            original_func.input_memory_type = MemoryType.CUPY.value
            original_func.output_memory_type = MemoryType.CUPY.value

            # Register the original function (now it's an OpenHCS function)
            _register_function(original_func, MemoryType.CUPY.value)
            decorated_count += 1

        except Exception as e:
            print(f"Warning: Failed to decorate {func_name}: {e}")
            skipped_count += 1

    print(f"✅ Decorated {decorated_count} CuPy ndimage functions as OpenHCS functions")
    print(f"⚠️  Skipped {skipped_count} functions (unknown contracts or dim_change)")


if __name__ == "__main__":  # pragma: no cover – manual use
    """Manual testing and analysis of CuPy functions."""
    registry = build_cupy_registry()

    print(f"\n📊 CuPy Function Analysis Summary:")
    contracts = {}
    for meta in registry.values():
        contract = meta.contract.value
        contracts[contract] = contracts.get(contract, 0) + 1

    for contract, count in contracts.items():
        print(f"  {contract}: {count} functions")

    print(f"\n🎯 Function Categories:")
    print(f"  Pure Array: {len(PURE_ARRAY_FUNCTIONS)} functions")
    print(f"  Value Returning: {len(VALUE_RETURNING_FUNCTIONS)} functions")
    print(f"  Mask Returning: {len(MASK_RETURNING_FUNCTIONS)} functions")
