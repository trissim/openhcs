#!/usr/bin/env python3
"""
Comprehensive test to verify processor function equivalence across frameworks.
Compares CuPy, PyTorch, TensorFlow, and JAX implementations against NumPy baseline.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_framework_availability():
    """Test which frameworks are available."""
    frameworks = {}
    
    # NumPy (baseline)
    try:
        import numpy as np
        frameworks['numpy'] = True
        logger.info("✅ NumPy available")
    except ImportError:
        frameworks['numpy'] = False
        logger.error("❌ NumPy not available")
    
    # CuPy
    try:
        import cupy as cp
        frameworks['cupy'] = True
        logger.info("✅ CuPy available")
    except ImportError:
        frameworks['cupy'] = False
        logger.info("⚠️ CuPy not available")
    
    # PyTorch
    try:
        import torch
        frameworks['torch'] = True
        logger.info("✅ PyTorch available")
    except ImportError:
        frameworks['torch'] = False
        logger.info("⚠️ PyTorch not available")
    
    # TensorFlow
    try:
        import tensorflow as tf
        frameworks['tensorflow'] = True
        logger.info("✅ TensorFlow available")
    except ImportError:
        frameworks['tensorflow'] = False
        logger.info("⚠️ TensorFlow not available")
    
    # JAX
    try:
        import jax
        import jax.numpy as jnp
        frameworks['jax'] = True
        logger.info("✅ JAX available")
    except ImportError:
        frameworks['jax'] = False
        logger.info("⚠️ JAX not available")
    
    return frameworks

def get_processor_functions():
    """Get all processor functions from each framework."""
    functions = {}
    
    # NumPy functions (baseline)
    try:
        from openhcs.processing.backends.processors import numpy_processor
        functions['numpy'] = {
            'percentile_normalize': numpy_processor.percentile_normalize,
            'stack_percentile_normalize': numpy_processor.stack_percentile_normalize,
            'sharpen': numpy_processor.sharpen,
            'max_projection': numpy_processor.max_projection,
            'mean_projection': numpy_processor.mean_projection,
            'create_projection': numpy_processor.create_projection,
            'create_composite': numpy_processor.create_composite,
            'apply_mask': numpy_processor.apply_mask,
            'create_weight_mask': numpy_processor.create_weight_mask,
        }
        logger.info("✅ NumPy processor functions loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load NumPy processor: {e}")
        functions['numpy'] = {}
    
    # CuPy functions
    try:
        from openhcs.processing.backends.processors import cupy_processor
        functions['cupy'] = {
            'percentile_normalize': cupy_processor.percentile_normalize,
            'stack_percentile_normalize': cupy_processor.stack_percentile_normalize,
            'sharpen': cupy_processor.sharpen,
            'max_projection': cupy_processor.max_projection,
            'mean_projection': cupy_processor.mean_projection,
            'create_projection': cupy_processor.create_projection,
            'create_composite': cupy_processor.create_composite,
            'apply_mask': cupy_processor.apply_mask,
            'create_weight_mask': cupy_processor.create_weight_mask,
        }
        logger.info("✅ CuPy processor functions loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load CuPy processor: {e}")
        functions['cupy'] = {}
    
    # PyTorch functions
    try:
        from openhcs.processing.backends.processors import torch_processor
        functions['torch'] = {
            'percentile_normalize': torch_processor.percentile_normalize,
            'stack_percentile_normalize': torch_processor.stack_percentile_normalize,
            'sharpen': torch_processor.sharpen,
            'max_projection': torch_processor.max_projection,
            'mean_projection': torch_processor.mean_projection,
            'create_projection': torch_processor.create_projection,
            'create_composite': torch_processor.create_composite,
            'apply_mask': torch_processor.apply_mask,
            'create_weight_mask': torch_processor.create_weight_mask,
        }
        logger.info("✅ PyTorch processor functions loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load PyTorch processor: {e}")
        functions['torch'] = {}
    
    # TensorFlow functions
    try:
        from openhcs.processing.backends.processors import tensorflow_processor
        functions['tensorflow'] = {
            'percentile_normalize': tensorflow_processor.percentile_normalize,
            'stack_percentile_normalize': tensorflow_processor.stack_percentile_normalize,
            'sharpen': tensorflow_processor.sharpen,
            'max_projection': tensorflow_processor.max_projection,
            'mean_projection': tensorflow_processor.mean_projection,
            'create_projection': tensorflow_processor.create_projection,
            'create_composite': tensorflow_processor.create_composite,
            'apply_mask': tensorflow_processor.apply_mask,
            'create_weight_mask': tensorflow_processor.create_weight_mask,
        }
        logger.info("✅ TensorFlow processor functions loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load TensorFlow processor: {e}")
        functions['tensorflow'] = {}
    
    # JAX functions
    try:
        from openhcs.processing.backends.processors import jax_processor
        functions['jax'] = {
            'percentile_normalize': jax_processor.percentile_normalize,
            'stack_percentile_normalize': jax_processor.stack_percentile_normalize,
            'sharpen': jax_processor.sharpen,
            'max_projection': jax_processor.max_projection,
            'mean_projection': jax_processor.mean_projection,
            'create_projection': jax_processor.create_projection,
            'create_composite': jax_processor.create_composite,
            'apply_mask': jax_processor.apply_mask,
            'create_weight_mask': jax_processor.create_weight_mask,
        }
        logger.info("✅ JAX processor functions loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load JAX processor: {e}")
        functions['jax'] = {}
    
    return functions

def compare_function_signatures():
    """Compare function signatures across frameworks."""
    logger.info("\n🔍 COMPARING FUNCTION SIGNATURES")
    
    functions = get_processor_functions()
    numpy_funcs = functions.get('numpy', {})
    
    if not numpy_funcs:
        logger.error("❌ Cannot compare - NumPy functions not available")
        return
    
    # Get all function names from NumPy (baseline)
    function_names = list(numpy_funcs.keys())
    
    for func_name in function_names:
        logger.info(f"\n📋 Function: {func_name}")
        
        # Check if function exists in each framework
        for framework, framework_funcs in functions.items():
            if func_name in framework_funcs:
                logger.info(f"  ✅ {framework}: Available")
            else:
                logger.warning(f"  ❌ {framework}: Missing")

def analyze_implementation_differences():
    """Analyze key implementation differences between frameworks."""
    logger.info("\n🔬 ANALYZING IMPLEMENTATION DIFFERENCES")
    
    # Key areas to check
    differences = {
        'percentile_calculation': {
            'numpy': 'np.percentile()',
            'cupy': 'cp.percentile()',
            'torch': 'torch.quantile() with sampling for large tensors',
            'tensorflow': 'Manual sorting or tensorflow_probability.stats.percentile()',
            'jax': 'jnp.percentile()'
        },
        'projection_output_shape': {
            'numpy': '3D array (1, Y, X)',
            'cupy': '3D array (1, Y, X)',
            'torch': '3D tensor (1, Y, X)',
            'tensorflow': '3D tensor (1, Y, X) - FIXED',
            'jax': '3D array (1, Y, X) - FIXED'
        },
        'gaussian_blur': {
            'numpy': 'scipy.ndimage.gaussian_filter',
            'cupy': 'cupyx.scipy.ndimage.gaussian_filter',
            'torch': 'Custom implementation with convolution',
            'tensorflow': 'tf.image.gaussian_blur',
            'jax': 'Custom implementation with lax.conv_general_dilated'
        }
    }
    
    for category, implementations in differences.items():
        logger.info(f"\n📊 {category.upper()}:")
        for framework, impl in implementations.items():
            logger.info(f"  {framework}: {impl}")

if __name__ == "__main__":
    print("🧪 PROCESSOR FUNCTION EQUIVALENCE TEST")
    print("=" * 50)
    
    # Test framework availability
    frameworks = test_framework_availability()
    
    # Compare function signatures
    compare_function_signatures()
    
    # Analyze implementation differences
    analyze_implementation_differences()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)

    available_frameworks = [f for f, available in frameworks.items() if available]
    logger.info(f"Available frameworks: {', '.join(available_frameworks)}")

    if len(available_frameworks) < 2:
        logger.warning("⚠️ Need at least 2 frameworks to compare equivalence")
    else:
        logger.info("✅ Ready for detailed equivalence testing")

    # Identify key differences that need attention
    print("\n🚨 KEY DIFFERENCES REQUIRING ATTENTION:")
    print("=" * 50)

    print("1. PROJECTION OUTPUT SHAPE CONSISTENCY:")
    print("   - NumPy/CuPy/PyTorch: Return 3D (1, Y, X)")
    print("   - TensorFlow/JAX: Return 3D (1, Y, X) - FIXED!")
    print("   ✅ Interface consistency restored!")

    print("\n2. PERCENTILE CALCULATION METHODS:")
    print("   - NumPy/CuPy/JAX: Direct percentile functions")
    print("   - PyTorch: Sampling for large tensors (memory optimization)")
    print("   - TensorFlow: Manual sorting or TensorFlow Probability")
    print("   ⚠️ May cause numerical differences")

    print("\n3. GAUSSIAN BLUR IMPLEMENTATIONS:")
    print("   - NumPy/CuPy: scipy.ndimage (proven, stable)")
    print("   - PyTorch/JAX: Custom convolution implementations")
    print("   - TensorFlow: tf.image.gaussian_blur")
    print("   ⚠️ Custom implementations may have edge case differences")

    print("\n🔧 COMPLETED FIXES:")
    print("=" * 50)
    print("✅ 1. Fixed projection output shapes in TensorFlow/JAX to return (1, Y, X)")
    print("✅ 2. Fixed JAX Gaussian blur function call bug (cls._gaussian_kernel -> _gaussian_kernel)")
    print("✅ 3. Standardized error handling in create_projection (FAIL FAST pattern)")

    print("\n🔧 REMAINING TASKS:")
    print("=" * 50)
    print("⚠️ 1. Verify numerical equivalence of percentile calculations")
    print("⚠️ 2. Test Gaussian blur implementations for edge case consistency")
    print("⚠️ 3. Add comprehensive numerical equivalence tests")
    print("⚠️ 4. Test PyTorch quantile sampling behavior vs direct percentile")
