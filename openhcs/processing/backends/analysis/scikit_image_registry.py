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
from pathlib import Path

# Suppress scikit-image warnings during analysis
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


# ===== METADATA CACHING SYSTEM =====

def _get_metadata_cache_path() -> Path:
    """Get the path for the scikit-image metadata cache file."""
    cache_dir = Path.home() / '.openhcs' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / 'skimage_function_metadata.json'


def _save_function_metadata(registry: Dict[str, 'SkimageFunction']) -> None:
    """Save function analysis metadata to cache for subprocess workers."""
    cache_path = _get_metadata_cache_path()

    # Convert registry to serializable format
    metadata = {}
    for full_name, func_meta in registry.items():
        # Store the actual module where the function lives (not the public API module)
        actual_module = func_meta.func.__module__

        metadata[full_name] = {
            'name': func_meta.name,
            'module': actual_module,  # Store actual module path where function lives
            'contract': func_meta.contract.value,
            'supports_3d': func_meta.supports_3d,
            'has_channel_axis': func_meta.has_channel_axis,
            'slice_by_slice_param': func_meta.slice_by_slice_param,
            'doc': func_meta.doc
        }

    try:
        with open(cache_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Saved function metadata cache: {len(metadata)} functions")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to save metadata cache: {e}")


def _load_function_metadata() -> Optional[Dict[str, Dict[str, Any]]]:
    """Load function analysis metadata from cache for subprocess workers."""
    cache_path = _get_metadata_cache_path()

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            metadata = json.load(f)
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded function metadata cache: {len(metadata)} functions")
        return metadata
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load metadata cache: {e}")
        return None


# ===== FUNCTION CATEGORIZATION FOR ARRAY-IN/ARRAY-OUT COMPLIANCE =====

# Category 1: Pure array functions - input 3D array, output 3D array (preserve dtype)
PURE_ARRAY_FUNCTIONS = {
    'gaussian', 'gaussian_blur', 'median', 'mean', 'sobel', 'prewitt', 'scharr',
    'laplacian', 'erosion', 'dilation', 'opening', 'closing', 'unsharp_mask',
    'wiener', 'richardson_lucy', 'denoise_tv_chambolle', 'denoise_bilateral',
    'rank_filter', 'minimum_filter', 'maximum_filter', 'percentile_filter'
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
    'remove_small_holes', 'label'
}


def _scale_and_convert(result, target_dtype):
    """
    Scale float results to target integer range and convert dtype.

    Args:
        result: Function output array
        target_dtype: Target data type (input image dtype)

    Returns:
        Array converted to target dtype with proper scaling
    """
    if not hasattr(result, 'dtype'):
        return result

    # If result is floating point and target is integer, scale appropriately
    if np.issubdtype(result.dtype, np.floating) and not np.issubdtype(target_dtype, np.floating):
        # Clip to [0, 1] range and scale to integer range
        clipped = np.clip(result, 0, 1)
        if target_dtype == np.uint8:
            return (clipped * 255).astype(target_dtype)
        elif target_dtype == np.uint16:
            return (clipped * 65535).astype(target_dtype)
        elif target_dtype == np.uint32:
            return (clipped * 4294967295).astype(target_dtype)
        else:
            # For other integer types, just convert without scaling
            return result.astype(target_dtype)

    # Direct conversion for same numeric type families
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
    
    # Create test 3D array (Z, Y, X format)
    test_3d = np.random.rand(3, 20, 20).astype(np.float32)
    test_2d = test_3d[0]
    
    try:
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


def _create_dtype_preserving_wrapper(original_func, func_name):
    """
    Create a wrapper that preserves input data type for scikit-image functions.

    Many scikit-image functions return float64 regardless of input type.
    This wrapper ensures the output has the same dtype as the input.
    """
    import numpy as np
    from functools import wraps

    @wraps(original_func)
    def dtype_preserving_wrapper(image, *args, **kwargs):
        try:
            # Store original dtype
            original_dtype = image.dtype

            # Call the original function
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
            logger.error(f"Error in dtype preserving wrapper for {func_name}: {e}")
            # Return original result on error
            return original_func(image, *args, **kwargs)

    # Preserve function metadata
    dtype_preserving_wrapper.__name__ = original_func.__name__
    dtype_preserving_wrapper.__module__ = original_func.__module__
    dtype_preserving_wrapper.__doc__ = original_func.__doc__

    return dtype_preserving_wrapper





def _register_skimage_ops_direct() -> None:
    """
    Register scikit-image functions directly in the OpenHCS function registry.

    This creates dtype-preserving wrappers for compatible scikit-image functions
    and registers them as OpenHCS functions with proper memory type attributes.

    Only functions with SLICE_SAFE and CROSS_Z contracts are decorated.
    UNKNOWN and DIM_CHANGE functions are skipped for stability.

    This is called during Phase 2 of registry initialization to avoid circular dependencies.
    """
    from openhcs.processing.func_registry import _register_function
    import os

    import logging
    logger = logging.getLogger(__name__)

    # Check if we're in subprocess mode - use cached metadata instead of expensive analysis
    subprocess_mode = os.environ.get('OPENHCS_SUBPROCESS_MODE')
    logger.info(f"DEBUG: OPENHCS_SUBPROCESS_MODE = {subprocess_mode}")

    if subprocess_mode == '1':
        logger.info("SUBPROCESS: Using cached metadata for scikit-image function registration")
        _register_skimage_ops_from_cache()
        return

    logger.info("Registering scikit-image functions as OpenHCS functions")

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

            # Only decorate SLICE_SAFE and CROSS_Z functions
            original_func = meta.func

            # HYBRID APPROACH: Decorate original function AND create enhanced wrapper
            # This gives us subprocess compatibility + dtype preservation + auto conversion

            # Step 1: Decorate the original function directly (for subprocess compatibility)
            from openhcs.constants import MemoryType
            original_func.input_memory_type = MemoryType.NUMPY.value
            original_func.output_memory_type = MemoryType.NUMPY.value

            # Step 2: Create enhanced wrapper with dtype preservation
            wrapper_func = _create_dtype_preserving_wrapper(original_func, meta.name)
            wrapper_func.input_memory_type = MemoryType.NUMPY.value
            wrapper_func.output_memory_type = MemoryType.NUMPY.value

            # Step 3: Replace module functions with enhanced wrapper (best of both worlds)
            import sys
            module_name = original_func.__module__
            func_name = original_func.__name__

            # Get the actual module directly from sys.modules
            if module_name in sys.modules:
                target_module = sys.modules[module_name]

                # Replace the function directly in the defining module
                if hasattr(target_module, func_name):
                    setattr(target_module, func_name, wrapper_func)
                    print(f"  ✅ Replaced {module_name}.{func_name} with enhanced wrapper")

                    # Also update any importing modules (e.g., skimage.filters imports from skimage._shared.filters)
                    # Check common importing patterns for scikit-image
                    if module_name.startswith('skimage._shared.'):
                        # Check if this function is imported into skimage.filters
                        public_module_name = module_name.replace('._shared.', '.')
                        if public_module_name in sys.modules:
                            public_module = sys.modules[public_module_name]
                            if hasattr(public_module, func_name):
                                setattr(public_module, func_name, wrapper_func)
                                print(f"  ✅ Also updated {public_module_name}.{func_name}")

                else:
                    print(f"  ⚠️ Function {func_name} not found in module {module_name}")
                    skipped_count += 1
                    continue
            else:
                print(f"  ⚠️ Module {module_name} not found in sys.modules")
                skipped_count += 1
                continue

            # Step 4: Register the enhanced wrapper (OpenHCS gets the best version)
            _register_function(wrapper_func, MemoryType.NUMPY.value)
            decorated_count += 1

        except Exception as e:
            print(f"  Warning: Failed to decorate {full_name}: {e}")
            skipped_count += 1

    logger.info(f"Decorated {decorated_count} scikit-image functions as OpenHCS functions")
    logger.info(f"Skipped {skipped_count} functions (UNKNOWN/DIM_CHANGE contracts or errors)")

    # Save metadata cache for subprocess workers
    _save_function_metadata(registry)

    # Verify a few key functions were decorated properly
    try:
        import skimage.filters
        if hasattr(skimage.filters.gaussian, 'input_memory_type'):
            logger.info("Verification: skimage.filters.gaussian properly decorated")
        else:
            logger.error("Verification failed: skimage.filters.gaussian missing memory type attributes")
    except Exception as e:
        logger.warning(f"Could not verify decoration: {e}")


def _register_skimage_ops_from_cache() -> None:
    """
    Register scikit-image functions using cached metadata (for subprocess workers).

    This decorates ALL the same functions as the main process but without expensive analysis.
    Uses pre-computed metadata to determine which functions to decorate and their contracts.
    """
    from openhcs.processing.func_registry import _register_function
    from openhcs.constants import MemoryType
    import sys

    import logging
    logger = logging.getLogger(__name__)
    logger.info("SUBPROCESS: Registering scikit-image functions from metadata cache")

    # Load cached metadata
    metadata = _load_function_metadata()
    if not metadata:
        logger.error("SUBPROCESS: No metadata cache found - cannot decorate functions")
        return

    decorated_count = 0
    skipped_count = 0

    for full_name, meta in metadata.items():
        try:
            # Skip functions that were skipped in main process
            if meta['contract'] in ['UNKNOWN', 'DIM_CHANGE']:
                skipped_count += 1
                continue

            # Import the function using the full module path from cache
            module_name = meta['module']  # Now contains full path like 'skimage.filters'
            func_name = meta['name']

            try:
                module = __import__(module_name, fromlist=[func_name])
                if not hasattr(module, func_name):
                    logger.warning(f"SUBPROCESS: {full_name}: function not found in module {module_name}")
                    skipped_count += 1
                    continue

                original_func = getattr(module, func_name)
            except ImportError:
                logger.warning(f"SUBPROCESS: {full_name}: module import failed for {module_name}")
                skipped_count += 1
                continue

            # Apply the same hybrid decoration as main process
            # Step 1: Decorate the original function directly (for subprocess compatibility)
            original_func.input_memory_type = MemoryType.NUMPY.value
            original_func.output_memory_type = MemoryType.NUMPY.value

            # Step 2: Create enhanced wrapper with dtype preservation
            wrapper_func = _create_dtype_preserving_wrapper(original_func, func_name)
            wrapper_func.input_memory_type = MemoryType.NUMPY.value
            wrapper_func.output_memory_type = MemoryType.NUMPY.value

            # Step 3: Replace module function with enhanced wrapper
            if module_name in sys.modules:
                target_module = sys.modules[module_name]
                if hasattr(target_module, func_name):
                    setattr(target_module, func_name, wrapper_func)
                    logger.info(f"SUBPROCESS: Replaced {module_name}.{func_name} with enhanced wrapper")

                    # Handle _shared module imports (like gaussian)
                    if module_name.startswith('skimage._shared.'):
                        public_module_name = module_name.replace('._shared.', '.')
                        if public_module_name in sys.modules:
                            public_module = sys.modules[public_module_name]
                            if hasattr(public_module, func_name):
                                setattr(public_module, func_name, wrapper_func)
                                logger.info(f"SUBPROCESS: Also replaced {public_module_name}.{func_name}")
                else:
                    logger.warning(f"SUBPROCESS: Function {func_name} not found in {module_name}")
            else:
                logger.warning(f"SUBPROCESS: Module {module_name} not in sys.modules")

            # Step 4: Register the enhanced wrapper
            _register_function(wrapper_func, MemoryType.NUMPY.value)
            decorated_count += 1

        except Exception as e:
            logger.error(f"SUBPROCESS: Failed to decorate {full_name}: {e}")
            skipped_count += 1

    logger.info(f"SUBPROCESS: Decorated {decorated_count} scikit-image functions from cache")
    logger.info(f"SUBPROCESS: Skipped {skipped_count} functions (UNKNOWN/DIM_CHANGE or errors)")

    # Verify key functions
    try:
        import skimage.filters
        if hasattr(skimage.filters.gaussian, 'input_memory_type'):
            logger.info("SUBPROCESS: Verification successful - all functions ready for compilation")
        else:
            logger.error("SUBPROCESS: Verification failed - functions missing memory type attributes")
    except Exception as e:
        logger.warning(f"SUBPROCESS: Could not verify decoration: {e}")



