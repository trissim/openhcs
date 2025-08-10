"""
Automated scikit-image function registry for OpenHCS.

This module automatically discovers, analyzes, and registers scikit-image functions
with proper processing contracts (slice_safe, cross_z, dim_change) based on their
3D array behavior.
"""

import inspect
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from functools import wraps
import warnings
import json
import os
import time
from pathlib import Path
import logging

# Suppress scikit-image warnings during analysis
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Set up logger
logger = logging.getLogger(__name__)

# Import blacklist system
from .function_classifier import is_blacklisted, log_blacklist_stats



# Functions requiring specific dtype conversions
BINARY_FUNCTIONS = {'binary_infsup', 'binary_supinf'}
UINT8_FUNCTIONS = set()  # Will be populated as we identify them

# ---- Module-level adapter: scikit-image → OpenHCS policy ----
def _skimage_adapt_function(original_func):
    """
    Adapt a scikit-image function to be OpenHCS-compatible per policy:
    - If 3D input raises dimension-like error, run per-slice and stack to 3D
    - If 3D input returns a 2D image, promote to (1, Y, X)
    - Handle dtype conversions for functions with specific requirements
    - Leave other dtype handling to the final OpenHCS wrapper
    """
    from functools import wraps
    import numpy as np
    DIM_ERR_TOKENS = ("dimension", "dimensional", "3d", "ndim", "axis", "rank", "shapes not aligned")

    func_name = getattr(original_func, '__name__', 'unknown')

    @wraps(original_func)
    def adapted(image, *args, **kwargs):
        original_dtype = image.dtype
        converted_image = image

        # Apply dtype conversion for functions with specific requirements
        if func_name in BINARY_FUNCTIONS:
            # Convert to binary: float32 [0,1] -> uint8 {0,255} with threshold at 0.5
            if image.dtype == np.float32:
                converted_image = ((image > 0.5) * 255).astype(np.uint8)
        elif func_name in UINT8_FUNCTIONS:
            # Convert to uint8: float32 [0,1] -> uint8 [0,255]
            if image.dtype == np.float32:
                converted_image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        if hasattr(converted_image, 'ndim') and converted_image.ndim == 3:
            try:
                result = original_func(converted_image, *args, **kwargs)
            except Exception as e:
                msg = str(e).lower()
                if any(tok in msg for tok in DIM_ERR_TOKENS):
                    # Per-slice fallback
                    from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type
                    mem = _detect_memory_type(converted_image)
                    slices = unstack_slices(converted_image, mem, 0)
                    results = [original_func(sl, *args, **kwargs) for sl in slices]
                    result = stack_slices(results, mem, 0)
                else:
                    raise
            # Promote 2D to singleton-Z 3D
            if hasattr(result, 'ndim') and result.ndim == 2:
                result = np.expand_dims(result, 0)
            elif isinstance(result, tuple) and hasattr(result[0], 'ndim') and result[0].ndim == 2:
                result = (np.expand_dims(result[0], 0), *result[1:])
        else:
            result = original_func(converted_image, *args, **kwargs)

        # Convert result back to original dtype if needed
        if func_name in BINARY_FUNCTIONS or func_name in UINT8_FUNCTIONS:
            if hasattr(result, 'dtype') and result.dtype != original_dtype:
                if result.dtype == np.uint8 and original_dtype == np.float32:
                    # Convert back: uint8 [0,255] -> float32 [0,1]
                    result = result.astype(np.float32) / 255.0
                elif result.dtype == np.bool_ and original_dtype == np.float32:
                    # Convert back: bool -> float32
                    result = result.astype(np.float32)
            elif isinstance(result, tuple):
                # Handle tuple results (array, value) - convert array part
                if hasattr(result[0], 'dtype') and result[0].dtype != original_dtype:
                    if result[0].dtype == np.uint8 and original_dtype == np.float32:
                        converted_array = result[0].astype(np.float32) / 255.0
                        result = (converted_array, *result[1:])
                    elif result[0].dtype == np.bool_ and original_dtype == np.float32:
                        converted_array = result[0].astype(np.float32)
                        result = (converted_array, *result[1:])

        return result

    return adapted

# ===== UNIFIED REGISTRY PATTERN =====




# ===== METADATA CACHING SYSTEM =====

def _get_metadata_cache_path() -> Path:
    """Get the path for the scikit-image metadata cache file."""
    from openhcs.core.xdg_paths import get_cache_file_path
    return get_cache_file_path('skimage_function_metadata.json')


def _save_function_metadata(registry: Dict[str, 'SkimageFunction']) -> None:
    """Save minimal function metadata to cache for subprocess workers."""
    cache_path = _get_metadata_cache_path()

    # Store metadata with cache validation info
    try:
        import skimage
        skimage_version = skimage.__version__
    except:
        skimage_version = "unknown"

    cache_data = {
        'cache_version': '1.0',
        'skimage_version': skimage_version,
        'timestamp': time.time(),
        'functions': {}
    }

    # Store only essential data needed for subprocess registration
    for full_name, func_meta in registry.items():
        # Only cache functions that will be decorated (skip UNKNOWN/DIM_CHANGE)
        if func_meta.contract in [ProcessingContract.SLICE_SAFE, ProcessingContract.CROSS_Z]:
            cache_data['functions'][full_name] = {
                'name': func_meta.name,
                'module': func_meta.func.__module__,  # Actual module where function lives
                'contract': func_meta.contract.value
            }

    try:
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        logger = logging.getLogger(__name__)
        logger.info(f"Saved function metadata cache: {len(cache_data['functions'])} functions")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to save metadata cache: {e}")


def _load_function_metadata() -> Optional[Dict[str, Dict[str, str]]]:
    """Load minimal function metadata from cache with validation."""
    cache_path = _get_metadata_cache_path()

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        logger = logging.getLogger(__name__)

        # Handle old cache format (direct metadata dict)
        if 'functions' not in cache_data:
            logger.info("Found old cache format - will rebuild")
            return None

        # Validate cache version and scikit-image version
        try:
            import skimage
            current_skimage_version = skimage.__version__
        except:
            current_skimage_version = "unknown"

        cached_version = cache_data.get('skimage_version', 'unknown')
        if cached_version != current_skimage_version:
            logger.info(f"scikit-image version changed ({cached_version} → {current_skimage_version}) - will rebuild cache")
            return None

        # Check cache age (rebuild if older than 7 days)
        cache_timestamp = cache_data.get('timestamp', 0)
        cache_age_days = (time.time() - cache_timestamp) / (24 * 3600)
        if cache_age_days > 7:
            logger.info(f"Cache is {cache_age_days:.1f} days old - will rebuild")
            return None

        functions = cache_data['functions']
        logger.info(f"Loaded valid function metadata cache: {len(functions)} functions")
        return functions

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load metadata cache: {e}")
        return None


def clear_function_metadata_cache() -> None:
    """Clear the function metadata cache to force rebuild on next startup."""
    cache_path = _get_metadata_cache_path()
    try:
        if cache_path.exists():
            cache_path.unlink()
            logger = logging.getLogger(__name__)
            logger.info("Function metadata cache cleared")
        else:
            logger = logging.getLogger(__name__)
            logger.info("No function metadata cache to clear")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to clear metadata cache: {e}")


# ===== FUNCTION CATEGORIZATION FOR ARRAY-IN/ARRAY-OUT COMPLIANCE =====

# Category 1: Pure array functions - input 3D array, output 3D array (preserve dtype)
PURE_ARRAY_FUNCTIONS = {
    'gaussian', 'gaussian_blur', 'median', 'mean', 'sobel', 'prewitt', 'scharr',
    'laplacian', 'erosion', 'dilation', 'opening', 'closing', 'unsharp_mask',
    'wiener', 'richardson_lucy', 'denoise_tv_chambolle', 'denoise_bilateral',
    'rank_filter', 'minimum_filter', 'maximum_filter', 'percentile_filter'
}

# Edge detection functions that should be processed slice-by-slice
# These functions compute gradients and should NOT include Z-gradients for microscopy fields
SLICE_BY_SLICE_FUNCTIONS = {
    'sobel', 'prewitt', 'scharr', 'laplacian', 'roberts', 'farid'
}

# Category 2: Value-returning functions - return (array, computed_value)
VALUE_RETURNING_FUNCTIONS = {
    'threshold_otsu', 'threshold_li', 'threshold_yen', 'threshold_triangle',
    'threshold_minimum', 'threshold_mean', 'threshold_isodata',
    'peak_local_maxima', 'centroid', 'moments', 'moments_central'
}

# Category 3: Mask-returning functions - return mask as array with proper dtype
MASK_RETURNING_FUNCTIONS = {
    'threshold_local', 'binary_erosion', 'binary_dilation', 'binary_opening',
    'binary_closing', 'binary_fill_holes', 'remove_small_objects',
    'remove_small_holes'
}

# Category 4: Label-returning functions - take boolean input, return integer labels
LABEL_RETURNING_FUNCTIONS = {
    'label'
}


# Import centralized scaling function
try:
    from openhcs.core.memory.decorators import _scale_and_convert_numpy as _scale_and_convert
except ImportError:
    # Fallback for standalone usage
    def _scale_and_convert(result, target_dtype):
        """Fallback scaling function."""
        if not hasattr(result, 'dtype'):
            return result
        return result.astype(target_dtype)


def _auto_detect_array_compliance(result, original_image, original_dtype):
    """
    Ensure unknown functions return 3D arrays following array-in/array-out pattern.

    Args:
        result: Function output
        original_image: Original 3D input image
        original_dtype: Original image data type

    Returns:
        3D array or (3D array, secondary_outputs) tuple
    """
    # If result is already a 3D array, just fix dtype
    if isinstance(result, np.ndarray) and result.ndim == 3:
        return _scale_and_convert(result, original_dtype)

    # If result is scalar/coordinates, return (original_image, result)
    elif np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim < 3):
        return original_image, result

    # If result is 2D, expand to 3D (assume same for all Z slices)
    elif isinstance(result, np.ndarray) and result.ndim == 2:
        expanded = np.stack([result] * original_image.shape[0], axis=0)
        return _scale_and_convert(expanded, original_dtype)

    # Fallback: return original image
    else:
        return original_image


class ProcessingContract(Enum):
    """Processing contract types for scikit-image functions."""
    SLICE_SAFE = "slice_safe"  # Process each Z-slice independently
    CROSS_Z = "cross_z"        # Process entire 3D volume
    DIM_CHANGE = "dim_change"  # Change output dimensionality
    UNKNOWN = "unknown"        # Could not determine


def _create_array_compliant_wrapper(original_func, func_name):
    """
    Create a wrapper that ensures array-in/array-out compliance and dtype preservation.

    All OpenHCS functions must:
    1. Take 3D array as first argument
    2. Return 3D array as first output
    3. Additional outputs (threshold values, coordinates) as 2nd, 3rd, etc. returns
    4. Preserve input dtype when appropriate

    Args:
        original_func: Original scikit-image function
        func_name: Function name for categorization

    Returns:
        Wrapped function that follows OpenHCS array-in/array-out pattern
    """
    @wraps(original_func)
    def array_compliant_wrapper(image_3d, *args, **kwargs):
        original_dtype = image_3d.dtype
        result = original_func(image_3d, *args, **kwargs)

        # Category 1: Pure array functions (sobel, gaussian, etc.)
        if func_name in PURE_ARRAY_FUNCTIONS:
            # Just fix dtype preservation
            return _scale_and_convert(result, original_dtype)

        # Category 2: Value-returning functions (threshold_otsu, etc.)
        elif func_name in VALUE_RETURNING_FUNCTIONS:
            # Return (pass_through_array, computed_value)
            return image_3d, result

        # Category 3: Mask-returning functions (threshold_local, etc.)
        elif func_name in MASK_RETURNING_FUNCTIONS:
            # Convert bool mask to original dtype range
            if hasattr(result, 'dtype') and result.dtype == bool:
                if original_dtype == np.uint16:
                    return result.astype(np.uint16) * 65535
                elif original_dtype == np.uint8:
                    return result.astype(np.uint8) * 255
                else:
                    return result.astype(original_dtype)
            return _scale_and_convert(result, original_dtype)

        # Auto-detect for unknown functions
        else:
            return _auto_detect_array_compliance(result, image_3d, original_dtype)

    # Preserve function metadata
    array_compliant_wrapper.__name__ = original_func.__name__
    array_compliant_wrapper.__module__ = original_func.__module__
    array_compliant_wrapper.__doc__ = original_func.__doc__

    return array_compliant_wrapper


@dataclass
class SkimageFunction:
    """Metadata for a scikit-image function."""
    name: str
    func: Callable
    module: str
    contract: ProcessingContract
    supports_3d: bool
    slice_by_slice_param: Optional[str] = None  # Parameter to force slice-by-slice
    has_channel_axis: bool = False
    doc: str = ""
    signature: str = ""


def _get_skimage_modules() -> List[Tuple[str, Any]]:
    """Get all relevant scikit-image modules for analysis."""
    import skimage

    # Key modules for image processing
    module_names = [
        'filters', 'morphology', 'segmentation', 'feature',
        'measure', 'transform', 'restoration', 'exposure'
    ]

    modules = []
    for name in module_names:
        if hasattr(skimage, name):
            module = getattr(skimage, name)
            modules.append((name, module))

    return modules


def _get_module_functions(module_name: str, module: Any) -> List[Tuple[str, Callable]]:
    """Extract all functions from a scikit-image module."""
    functions = []

    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith('_'):
            continue

        # Skip if not actually from this module
        if not hasattr(obj, '__module__') or module_name not in obj.__module__:
            continue

        functions.append((name, obj))

    return functions


def _analyze_function_signature(func: Callable) -> Tuple[bool, Optional[str], str]:
    """Analyze function signature for 3D compatibility and slice-by-slice parameters."""
    try:
        sig = inspect.signature(func)
        signature_str = str(sig)

        # Check for channel_axis parameter (common slice-by-slice control)
        has_channel_axis = 'channel_axis' in sig.parameters
        slice_param = 'channel_axis' if has_channel_axis else None

        # Check for other potential slice control parameters
        if not slice_param:
            slice_control_params = ['axis', 'multichannel', 'preserve_range']
            for param in slice_control_params:
                if param in sig.parameters:
                    slice_param = param
                    break

        return has_channel_axis, slice_param, signature_str

    except Exception:
        return False, None, ""


def _test_3d_behavior(func: Callable, func_name: str) -> Tuple[ProcessingContract, bool]:
    """Test function behavior with 3D arrays to determine processing contract."""

    # Get full import path for debugging
    module_name = getattr(func, '__module__', 'unknown')
    full_import_path = f"{module_name}.{func_name}"

    # Create test 3D array (Z, Y, X format)
    test_3d = np.random.rand(3, 20, 20).astype(np.float32)
    test_2d = test_3d[0]

    try:
        # Print function being tested for warning attribution
        print(f"    🧪 Testing function: {full_import_path}")

        # Test if function accepts 3D input
        result_3d = func(test_3d)
        result_2d = func(test_2d)

        # Check if shapes are preserved
        if hasattr(result_3d, 'shape'):
            if result_3d.shape == ():
                # Scalar result (shape is empty tuple) - this is a value-returning function
                # These functions should be wrapped to return (array, value) tuples
                return ProcessingContract.SLICE_SAFE, True
            elif result_3d.shape != test_3d.shape:
                # Output shape changed - likely dimension-changing function
                return ProcessingContract.DIM_CHANGE, True
        else:
            # No shape attribute - this is a value-returning function, not dimension-changing
            # These functions should be wrapped to return (array, value) tuples
            return ProcessingContract.SLICE_SAFE, True

        # Test if processing is slice-by-slice
        manual_3d = np.stack([func(test_3d[z]) for z in range(test_3d.shape[0])])

        if np.allclose(result_3d, manual_3d, rtol=1e-5, atol=1e-8):
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


def _test_channel_axis_behavior(func: Callable, has_channel_axis: bool) -> bool:
    """Test if channel_axis parameter enables slice-by-slice processing."""
    if not has_channel_axis:
        return False

    test_3d = np.random.rand(3, 20, 20).astype(np.float32)

    try:
        # Test with channel_axis=0 (treat Z as channels)
        result_channel = func(test_3d, channel_axis=0)

        # Compare with manual slice-by-slice
        manual_3d = np.stack([func(test_3d[z]) for z in range(test_3d.shape[0])])

        return np.allclose(result_channel, manual_3d, rtol=1e-5, atol=1e-8)

    except Exception:
        return False


def build_skimage_registry() -> Dict[str, SkimageFunction]:
    """Build comprehensive registry of scikit-image functions with processing contracts."""

    print("🔍 Analyzing scikit-image functions...")

    # Log blacklist information
    log_blacklist_stats()

    registry = {}
    modules = _get_skimage_modules()

    total_functions = 0
    analyzed_functions = 0

    for module_name, module in modules:
        print(f"  📦 Analyzing skimage.{module_name}...")

        functions = _get_module_functions(module_name, module)
        total_functions += len(functions)

        for func_name, func in functions:
            full_name = f"{module_name}.{func_name}"

            # Skip blacklisted functions
            if is_blacklisted(func, func_name):
                print(f"    🚫 Skipping blacklisted function: {full_name}")
                continue

            try:
                # Analyze function signature
                has_channel_axis, slice_param, signature = _analyze_function_signature(func)

                # Test 3D behavior
                contract, supports_3d = _test_3d_behavior(func, func_name)

                # Test channel_axis behavior if available
                channel_axis_works = _test_channel_axis_behavior(func, has_channel_axis)

                # Get documentation
                doc = (func.__doc__ or "").split('\n')[0].strip()

                # Create function metadata
                skimage_func = SkimageFunction(
                    name=func_name,
                    func=func,
                    module=module_name,
                    contract=contract,
                    supports_3d=supports_3d,
                    slice_by_slice_param=slice_param if channel_axis_works else None,
                    has_channel_axis=has_channel_axis,
                    doc=doc,
                    signature=signature
                )

                registry[full_name] = skimage_func
                analyzed_functions += 1

            except Exception as e:
                print(f"    ⚠️  Failed to analyze {full_name}: {e}")

    print(f"✅ Analyzed {analyzed_functions}/{total_functions} scikit-image functions")
    return registry


def register_skimage_ops(*, auto_register: bool = True) -> Dict[str, SkimageFunction]:
    """Build registry and optionally auto-register functions with OpenHCS."""

    registry = build_skimage_registry()

    if not auto_register:
        return registry

    try:
        from openhcs.core.memory.decorators import numpy as numpy_func
        from openhcs.processing.func_registry import register_function
    except ImportError:
        print("⚠️  OpenHCS not available - skipping auto-registration")
        return registry

    # Register functions by contract type
    registered_count = 0
    skipped_count = 0

    for full_name, meta in registry.items():
        try:
            # Skip functions that don't support arrays or have unknown contracts
            if meta.contract == ProcessingContract.UNKNOWN:
                skipped_count += 1
                continue

            # Skip dimension-changing functions (OpenHCS requires array-in/array-out)
            if meta.contract == ProcessingContract.DIM_CHANGE:
                skipped_count += 1
                continue

            # Create wrapper function if needed for slice-by-slice parameter
            if meta.slice_by_slice_param and meta.supports_3d:
                # Create wrapper that forces slice-by-slice processing
                def create_wrapper(original_func, param_name):
                    def wrapper(image, **kwargs):
                        # Force slice-by-slice processing for 3D arrays
                        if image.ndim == 3:
                            kwargs[param_name] = 0  # Treat first axis as channels
                        return original_func(image, **kwargs)

                    wrapper.__name__ = original_func.__name__
                    wrapper.__doc__ = original_func.__doc__
                    return wrapper

                func_to_register = create_wrapper(meta.func, meta.slice_by_slice_param)
            else:
                func_to_register = meta.func

            # Apply @numpy decorator
            decorated_func = numpy_func(func_to_register)

            # Register with OpenHCS
            register_function(
                decorated_func,
                backend="numpy",
                contract=meta.contract.value,
                doc=meta.doc
            )

            registered_count += 1

        except Exception as e:
            print(f"⚠️  Failed to register {full_name}: {e}")
            skipped_count += 1

    print(f"✅ Registered {registered_count} scikit-image functions with @numpy")
    print(f"⚠️  Skipped {skipped_count} functions (unknown contracts or dim_change - OpenHCS requires array-in/array-out)")

    return registry


if __name__ == "__main__":
    # Test the registry
    registry = register_skimage_ops(auto_register=False)

    # Summarize results
    contracts = {}
    for meta in registry.values():
        contract = meta.contract.value
        contracts[contract] = contracts.get(contract, 0) + 1

    print(f"\n📊 SCIKIT-IMAGE FUNCTION ANALYSIS SUMMARY")
    print(f"Total functions: {len(registry)}")
    for contract, count in sorted(contracts.items()):
        print(f"  {contract}: {count}")

    # Show functions with slice-by-slice parameters
    slice_param_funcs = [meta for meta in registry.values() if meta.slice_by_slice_param]
    print(f"\nFunctions with slice-by-slice parameters: {len(slice_param_funcs)}")
    for meta in slice_param_funcs[:10]:  # Show first 10
        print(f"  {meta.module}.{meta.name}: {meta.slice_by_slice_param}")


# Dtype preserving wrapper moved to centralized decorators module
def _create_dtype_preserving_wrapper_DEPRECATED(original_func, func_name):
    """
    Create a wrapper that preserves input data type and adds slice_by_slice parameter.

    Many scikit-image functions return float64 regardless of input type.
    This wrapper ensures the output has the same dtype as the input and adds
    a slice_by_slice parameter to avoid cross-slice contamination in 3D arrays.

    Uses the slice-by-slice logic from the existing OpenHCS module for consistency.
    """
    import numpy as np
    import inspect
    from functools import wraps

    @wraps(original_func)
    def dtype_and_slice_preserving_wrapper(image, *args, slice_by_slice: bool = False, **kwargs):
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

            # Convert result back to original dtype if it's different
            if hasattr(result, 'dtype') and result.dtype != original_dtype:
                # For edge detection functions like sobel, we need to scale the output
                # from float range to the original dtype range
                if np.issubdtype(result.dtype, np.floating) and not np.issubdtype(original_dtype, np.floating):
                    # Scale float output to integer range
                    if original_dtype == np.uint8:
                        result = (result * 255).astype(original_dtype)
                    elif original_dtype == np.uint16:
                        result = (result * 65535).astype(original_dtype)
                    elif original_dtype == np.uint32:
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
            logger.error(f"Error in dtype/slice preserving wrapper for {func_name}: {e}")
            # Return original result on error
            return original_func(image, *args, **kwargs)

    # Manually add slice_by_slice parameter to signature (after @wraps)
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
        dtype_and_slice_preserving_wrapper.__signature__ = new_sig

        # Set type annotations manually for get_type_hints() compatibility
        dtype_and_slice_preserving_wrapper.__annotations__ = getattr(original_func, '__annotations__', {}).copy()
        dtype_and_slice_preserving_wrapper.__annotations__['slice_by_slice'] = bool

    # Update docstring to mention slice_by_slice parameter
    original_doc = dtype_and_slice_preserving_wrapper.__doc__ or ""
    slice_doc = """

    Additional OpenHCS Parameters
    -----------------------------
    slice_by_slice : bool, optional (default: False)
        If True, process 3D arrays slice-by-slice to avoid cross-slice contamination.
        If False, use original 3D behavior. Recommended for edge detection functions
        on stitched microscopy data to prevent artifacts at field boundaries.
    """
    dtype_and_slice_preserving_wrapper.__doc__ = original_doc + slice_doc

    return dtype_and_slice_preserving_wrapper





def _register_skimage_ops_direct() -> None:
    """
    Register scikit-image functions with caching support via unified orchestrator.

    This is called during Phase 2 of registry initialization to avoid circular dependencies.
    """
    from openhcs.processing.func_registry import _register_function
    import os

    # Attempt cache‑based registration first
    from openhcs.processing.backends.analysis.cache_utils import run_cached_registration
    if run_cached_registration("skimage", _register_skimage_ops_from_cache):
        return

    # run_cached_registration already handles subprocess and TUI fast-path
    # If it didn't return, fall through to full discovery
    logger.info("No valid cache fast-path, performing full analysis and building cache")

    decorated_count = 0
    skipped_count = 0

    # Get functions using build_skimage_registry (same as register_skimage_ops)
    registry = build_skimage_registry()

    for full_name, meta in registry.items():
        try:
            # Skip functions that don't support arrays or have unknown contracts
            if meta.contract == ProcessingContract.UNKNOWN:
                print(f"  Skipping {full_name}: UNKNOWN contract")
                skipped_count += 1
                continue



            # Skip dimension-changing functions (OpenHCS requires array-in/array-out)
            if meta.contract == ProcessingContract.DIM_CHANGE:
                print(f"  Skipping {full_name}: DIM_CHANGE contract")
                skipped_count += 1
                continue

            # Registry-specific adapter, then unified decoration
            from openhcs.constants import MemoryType
            from openhcs.processing.func_registry import _apply_unified_decoration

            adapted = _skimage_adapt_function(meta.func)
            wrapper_func = _apply_unified_decoration(
                original_func=adapted,
                func_name=meta.name,
                memory_type=MemoryType.NUMPY,
                create_wrapper=True
            )

            _register_function(wrapper_func, MemoryType.NUMPY.value)
            decorated_count += 1

        except Exception as e:
            print(f"  Warning: Failed to decorate {full_name}: {e}")
            skipped_count += 1

    logger.info(f"Decorated {decorated_count} scikit-image functions as OpenHCS functions")
    logger.info(f"Skipped {skipped_count} functions (UNKNOWN/DIM_CHANGE contracts or errors)")

    # Save metadata cache for subprocess workers
    _save_function_metadata(registry)


def _get_skimage_function(module_path: str, func_name: str):
    """Get scikit-image function object from module path and function name."""
    try:
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, func_name, None)
    except Exception:
        return None


def _register_skimage_ops_from_cache() -> bool:
    """Register scikit-image functions using cached metadata. Returns True if used."""
    from openhcs.processing.backends.analysis.cache_utils import register_functions_from_cache
    from openhcs.processing.func_registry import _register_function
    from openhcs.constants import MemoryType
    import logging

    logger = logging.getLogger(__name__)
    logger.info("SUBPROCESS: Registering scikit-image functions from metadata cache")

    # Load cached metadata
    cached_metadata = _load_function_metadata()
    if not cached_metadata:
        logger.error("SUBPROCESS: No metadata cache found - cannot decorate functions")
        return False

    def register_skimage_function(original_func, func_name: str, memory_type: str):
        """Register a scikit-image function with registry-specific adapter + unified decoration."""
        from openhcs.processing.func_registry import _apply_unified_decoration

        adapted = _skimage_adapt_function(original_func)
        wrapper_func = _apply_unified_decoration(
            original_func=adapted,
            func_name=func_name,
            memory_type=MemoryType.NUMPY,
            create_wrapper=True
        )

        _register_function(wrapper_func, MemoryType.NUMPY.value)

    # Register functions from cache using unified system
    register_functions_from_cache(
        library_name="skimage",
        cached_metadata=cached_metadata,
        get_function_func=_get_skimage_function,
        register_function_func=register_skimage_function,
        memory_type=MemoryType.NUMPY.value
    )
    return True



