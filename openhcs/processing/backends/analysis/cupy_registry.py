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


def _create_cupy_dtype_preserving_wrapper(original_func, func_name):
    """
    Create a wrapper that preserves input data type and adds slice_by_slice parameter for CuPy functions.

    This uses the SAME pattern as scikit-image for consistency. CuPy functions generally preserve
    dtypes better than scikit-image, but this wrapper ensures consistent behavior and adds
    slice_by_slice parameter to avoid cross-slice contamination in 3D arrays.

    Uses the slice-by-slice logic from the existing OpenHCS module for consistency.
    """
    import numpy as np
    import inspect
    from functools import wraps

    @wraps(original_func)
    def cupy_dtype_and_slice_preserving_wrapper(image, *args, slice_by_slice: bool = False, **kwargs):
        try:
            # Store original dtype
            original_dtype = image.dtype

            # Handle slice_by_slice processing for 3D arrays using OpenHCS stack utilities
            if slice_by_slice and hasattr(image, 'ndim') and image.ndim == 3:
                from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type

                # Detect memory type and use proper OpenHCS utilities
                memory_type = _detect_memory_type(image)
                gpu_id = 0  # Default GPU ID for slice processing

                # Unstack 3D array into 2D slices
                slices_2d = unstack_slices(image, memory_type, gpu_id)

                # Process each slice
                processed_slices = []
                for slice_2d in slices_2d:
                    slice_result = original_func(slice_2d, *args, **kwargs)
                    processed_slices.append(slice_result)

                # Stack results back into 3D array
                result = stack_slices(processed_slices, memory_type, gpu_id)
            else:
                # Call the original function normally
                result = original_func(image, *args, **kwargs)

            # Convert result back to original dtype if it's different (same logic as scikit-image wrapper)
            if hasattr(result, 'dtype') and result.dtype != original_dtype:
                # For edge detection functions like sobel, we need to scale the output
                # from float range to the original dtype range (same as scikit-image wrapper)
                if cp.issubdtype(result.dtype, cp.floating) and not cp.issubdtype(original_dtype, cp.floating):
                    # Scale float output to integer range
                    if original_dtype == cp.uint8:
                        result = (result * 255).astype(original_dtype)
                    elif original_dtype == cp.uint16:
                        result = (result * 65535).astype(original_dtype)
                    elif original_dtype == cp.uint32:
                        result = (result * 4294967295).astype(original_dtype)
                    else:
                        # For other integer types, just convert without scaling
                        result = result.astype(original_dtype)
                else:
                    # Direct conversion for same numeric type families
                    result = result.astype(original_dtype)

            return result
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in CuPy dtype/slice preserving wrapper for {func_name}: {e}")
            # Return original result on error
            return original_func(image, *args, **kwargs)


    # Manually add slice_by_slice parameter to signature (after @wraps) - SAME as scikit-image
    original_sig = inspect.signature(original_func)
    new_params = list(original_sig.parameters.values())

    # Check if slice_by_slice parameter already exists
    param_names = [p.name for p in new_params]
    if 'slice_by_slice' not in param_names:
        # Add slice_by_slice parameter as keyword-only
        slice_param = inspect.Parameter(
            'slice_by_slice',
            inspect.Parameter.KEYWORD_ONLY,
            default=False,
            annotation=bool
        )
        new_params.append(slice_param)

        # Create new signature and override the @wraps signature
        new_sig = original_sig.replace(parameters=new_params)
        cupy_dtype_and_slice_preserving_wrapper.__signature__ = new_sig

        # Set type annotations manually for get_type_hints() compatibility
        cupy_dtype_and_slice_preserving_wrapper.__annotations__ = getattr(original_func, '__annotations__', {}).copy()
        cupy_dtype_and_slice_preserving_wrapper.__annotations__['slice_by_slice'] = bool

    # Update docstring to mention slice_by_slice parameter - SAME as scikit-image
    original_doc = cupy_dtype_and_slice_preserving_wrapper.__doc__ or ""
    slice_doc = """

    Additional OpenHCS Parameters
    -----------------------------
    slice_by_slice : bool, optional (default: False)
        If True, process 3D arrays slice-by-slice to avoid cross-slice contamination.
        If False, use original 3D behavior. Recommended for edge detection functions
        on stitched microscopy data to prevent artifacts at field boundaries.
    """
    cupy_dtype_and_slice_preserving_wrapper.__doc__ = original_doc + slice_doc

    return cupy_dtype_and_slice_preserving_wrapper


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
    Build a registry of CuCIM skimage functions with their metadata.

    CuCIM provides GPU-accelerated versions of scikit-image functions that are
    equivalent to their CPU counterparts but run on GPU.
    """
    try:
        import cucim.skimage
        print("🔍 Analyzing CuCIM skimage functions...")
    except ImportError as e:
        print(f"❌ CuCIM skimage not available: {e}")
        print("❌ No fallback - CuCIM is required for GPU scikit-image functions")
        return {}

    registry = {}
    analyzed_functions = 0

    # Define modules to scan for GPU-accelerated scikit-image functions
    modules_to_scan = [
        ('filters', 'cucim.skimage.filters'),
        ('morphology', 'cucim.skimage.morphology'),
        ('measure', 'cucim.skimage.measure'),
        ('segmentation', 'cucim.skimage.segmentation'),
        ('feature', 'cucim.skimage.feature'),
        ('restoration', 'cucim.skimage.restoration'),
        ('transform', 'cucim.skimage.transform'),
        ('exposure', 'cucim.skimage.exposure'),
        ('color', 'cucim.skimage.color'),
        ('util', 'cucim.skimage.util'),
    ]

    # Scan each module for functions
    for module_name, module_path in modules_to_scan:
        try:
            # Import the module dynamically
            module = __import__(module_path, fromlist=[module_name])
            print(f"  📦 Analyzing {module_path}...")

            for name in dir(module):
                if not name.startswith('_') and callable(getattr(module, name)):
                    func = getattr(module, name)

                    # Skip classes and non-function objects
                    if not hasattr(func, '__call__') or (hasattr(func, '__name__') and func.__name__[0].isupper()):
                        continue

                    try:
                        # Test 3D behavior (CuCIM functions should handle 3D well)
                        contract, supports_3d = _test_cupy_3d_behavior(func, name)

                        # Get documentation
                        doc = (func.__doc__ or "").split('\n')[0].strip()

                        # Get signature
                        try:
                            signature = str(inspect.signature(func))
                        except:
                            signature = "signature unavailable"

                        # Create function metadata (use original name for filters, prefixed for others)
                        full_name = name if module_name == 'filters' else f"{module_name}_{name}"

                        cupy_func = CupyFunction(
                            name=full_name,
                            func=func,
                            module=module_path,
                            contract=contract,
                            supports_3d=supports_3d,
                            doc=doc,
                            signature=signature
                        )

                        registry[full_name] = cupy_func
                        analyzed_functions += 1

                    except Exception as e:
                        print(f"    ⚠️  Failed to analyze {module_name}.{name}: {e}")

        except ImportError as e:
            print(f"    ⚠️  Could not import {module_path}: {e}")
        except Exception as e:
            print(f"    ⚠️  Error scanning {module_path}: {e}")

    print(f"✅ Analyzed {analyzed_functions} CuCIM skimage functions")
    return registry





def _register_cupy_ops_direct() -> None:
    """
    Register CuCIM skimage functions using unified decoration pattern.

    This is called during Phase 2 of registry initialization to avoid circular dependencies.
    """
    try:
        import cucim.skimage
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("CuCIM skimage not available - skipping GPU scikit-image registration")
        return

    from openhcs.processing.func_registry import _register_function
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Registering CuCIM skimage functions using unified pattern")

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

            # Apply unified decoration pattern with CuPy wrapper
            from openhcs.constants import MemoryType
            from openhcs.processing.func_registry import _apply_unified_decoration

            wrapper_func = _apply_unified_decoration(
                original_func=meta.func,
                func_name=func_name,
                memory_type=MemoryType.CUPY,
                create_wrapper=True  # Use CuPy dtype preserving wrapper
            )

            # Register the function
            _register_function(wrapper_func, MemoryType.CUPY.value)
            decorated_count += 1

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to decorate {func_name}: {e}")
            skipped_count += 1

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Decorated {decorated_count} CuPy ndimage functions as OpenHCS functions")
    logger.info(f"Skipped {skipped_count} functions (unknown contracts or dim_change)")


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
