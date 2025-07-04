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
import warnings

# Suppress scikit-image warnings during analysis
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


class ProcessingContract(Enum):
    """Processing contract types for scikit-image functions."""
    SLICE_SAFE = "slice_safe"  # Process each Z-slice independently
    CROSS_Z = "cross_z"        # Process entire 3D volume
    DIM_CHANGE = "dim_change"  # Change output dimensionality
    UNKNOWN = "unknown"        # Could not determine


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
        if result_3d.shape != test_3d.shape:
            # Output shape changed - likely dimension-changing function
            return ProcessingContract.DIM_CHANGE, True
        
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


def _register_skimage_ops_direct() -> None:
    """
    Direct registration of scikit-image functions without triggering registry initialization.

    This is called during Phase 2 of registry initialization to avoid circular dependencies.
    """
    from openhcs.processing.func_registry import _register_function
    from openhcs.core.memory.decorators import numpy as numpy_func

    print("🔧 Direct registration of scikit-image functions...")

    registered_count = 0
    skipped_count = 0

    # Get functions using build_skimage_registry (same as register_skimage_ops)
    registry = build_skimage_registry()

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

            # Use the original function directly to avoid pickle issues
            func_to_register = meta.func

            # Add memory type attributes directly to the original function
            # This makes the function pickleable since it's the same object as the module function
            func_to_register.input_memory_type = "numpy"
            func_to_register.output_memory_type = "numpy"

            # Direct registration without triggering initialization
            _register_function(func_to_register, "numpy")
            registered_count += 1

        except Exception as e:
            print(f"Warning: Failed to register {full_name}: {e}")
            skipped_count += 1

    print(f"✅ Direct registered {registered_count} scikit-image functions")
    print(f"⚠️  Skipped {skipped_count} functions (unknown contracts or dim_change)")
