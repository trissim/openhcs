"""
GPU memory cleanup utilities for different frameworks.

This module provides unified GPU memory cleanup functions for PyTorch, CuPy, 
TensorFlow, and JAX. The cleanup functions are designed to be called after
processing steps to free up GPU memory that's no longer needed.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def is_gpu_memory_type(memory_type: str) -> bool:
    """
    Check if a memory type is a GPU memory type.

    Args:
        memory_type: Memory type string

    Returns:
        True if it's a GPU memory type, False otherwise
    """
    try:
        from openhcs.constants.constants import VALID_GPU_MEMORY_TYPES
        return memory_type in VALID_GPU_MEMORY_TYPES
    except ImportError:
        # Fallback if constants not available
        gpu_types = {"torch", "cupy", "tensorflow", "jax", "pyclesperanto"}
        return memory_type.lower() in gpu_types


def cleanup_pytorch_gpu(device_id: Optional[int] = None) -> None:
    """
    Clean up PyTorch GPU memory.
    
    Args:
        device_id: Optional GPU device ID. If None, cleans all devices.
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return
            
        if device_id is not None:
            # Clean specific device
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.debug(f"🔥 GPU CLEANUP: Cleared PyTorch CUDA cache for device {device_id}")
        else:
            # Clean all devices
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("🔥 GPU CLEANUP: Cleared PyTorch CUDA cache for all devices")
            
    except ImportError:
        logger.debug("PyTorch not available, skipping PyTorch GPU cleanup")
    except Exception as e:
        logger.warning(f"Failed to cleanup PyTorch GPU memory: {e}")


def cleanup_cupy_gpu(device_id: Optional[int] = None) -> None:
    """
    Clean up CuPy GPU memory with aggressive defragmentation.

    Args:
        device_id: Optional GPU device ID. If None, cleans current device.
    """
    try:
        import cupy

        if device_id is not None:
            # Clean specific device
            with cupy.cuda.Device(device_id):
                # Get memory info before cleanup
                mempool = cupy.get_default_memory_pool()
                used_before = mempool.used_bytes()

                # Aggressive cleanup to defragment memory
                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()

                # Force memory pool reset to defragment
                cupy.cuda.runtime.deviceSynchronize()

                used_after = mempool.used_bytes()
                freed_mb = (used_before - used_after) / 1e6

            logger.debug(f"🔥 GPU CLEANUP: Cleared CuPy memory pools for device {device_id}, freed {freed_mb:.1f}MB")
        else:
            # Clean current device
            mempool = cupy.get_default_memory_pool()
            used_before = mempool.used_bytes()

            # Aggressive cleanup to defragment memory
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()

            # Force memory pool reset to defragment
            cupy.cuda.runtime.deviceSynchronize()

            used_after = mempool.used_bytes()
            freed_mb = (used_before - used_after) / 1e6

            logger.debug(f"🔥 GPU CLEANUP: Cleared CuPy memory pools for current device, freed {freed_mb:.1f}MB")

    except ImportError:
        logger.debug("CuPy not available, skipping CuPy GPU cleanup")
    except Exception as e:
        logger.warning(f"Failed to cleanup CuPy GPU memory: {e}")


def cleanup_tensorflow_gpu(device_id: Optional[int] = None) -> None:
    """
    Clean up TensorFlow GPU memory.
    
    Args:
        device_id: Optional GPU device ID. If None, cleans all devices.
    """
    try:
        import tensorflow as tf
        
        # Get list of GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return
            
        if device_id is not None and device_id < len(gpus):
            # Clean specific device - TensorFlow doesn't have per-device cleanup
            # so we trigger garbage collection which helps with memory management
            import gc
            gc.collect()
            logger.debug(f"🔥 GPU CLEANUP: Triggered garbage collection for TensorFlow GPU {device_id}")
        else:
            # Clean all devices - trigger garbage collection
            import gc
            gc.collect()
            logger.debug("🔥 GPU CLEANUP: Triggered garbage collection for TensorFlow GPUs")
            
    except ImportError:
        logger.debug("TensorFlow not available, skipping TensorFlow GPU cleanup")
    except Exception as e:
        logger.warning(f"Failed to cleanup TensorFlow GPU memory: {e}")


def cleanup_jax_gpu(device_id: Optional[int] = None) -> None:
    """
    Clean up JAX GPU memory.

    Args:
        device_id: Optional GPU device ID. If None, cleans all devices.
    """
    try:
        import jax

        # JAX doesn't have explicit memory cleanup like PyTorch/CuPy
        # but we can trigger garbage collection and clear compilation cache
        import gc
        gc.collect()

        # Clear JAX compilation cache which can hold GPU memory
        jax.clear_caches()

        if device_id is not None:
            logger.debug(f"🔥 GPU CLEANUP: Cleared JAX caches and triggered GC for device {device_id}")
        else:
            logger.debug("🔥 GPU CLEANUP: Cleared JAX caches and triggered GC for all devices")

    except ImportError:
        logger.debug("JAX not available, skipping JAX GPU cleanup")
    except Exception as e:
        logger.warning(f"Failed to cleanup JAX GPU memory: {e}")


def cleanup_pyclesperanto_gpu(device_id: Optional[int] = None) -> None:
    """
    Clean up pyclesperanto GPU memory.

    Args:
        device_id: Optional GPU device ID. If None, cleans current device.
    """
    try:
        import pyclesperanto as cle
        import gc

        # pyclesperanto doesn't have explicit memory cleanup like PyTorch/CuPy
        # but we can trigger garbage collection and clear any cached data

        if device_id is not None:
            # Select the specific device
            devices = cle.list_available_devices()
            if device_id < len(devices):
                cle.select_device(device_id)
                logger.debug(f"🔥 GPU CLEANUP: Selected pyclesperanto device {device_id}")
            else:
                logger.warning(f"🔥 GPU CLEANUP: Device {device_id} not available in pyclesperanto")

        # Trigger garbage collection to clean up any unreferenced GPU arrays
        collected = gc.collect()

        # pyclesperanto uses OpenCL which manages memory automatically
        # but we can help by ensuring Python objects are cleaned up
        if device_id is not None:
            logger.debug(f"🔥 GPU CLEANUP: Triggered GC for pyclesperanto device {device_id}, collected {collected} objects")
        else:
            logger.debug(f"🔥 GPU CLEANUP: Triggered GC for pyclesperanto current device, collected {collected} objects")

    except ImportError:
        logger.debug("pyclesperanto not available, skipping pyclesperanto GPU cleanup")
    except Exception as e:
        logger.warning(f"Failed to cleanup pyclesperanto GPU memory: {e}")


def cleanup_gpu_memory_by_framework(memory_type: str, device_id: Optional[int] = None) -> None:
    """
    Clean up GPU memory based on the OpenHCS memory type.

    Args:
        memory_type: OpenHCS memory type string ("torch", "cupy", "tensorflow", "jax", "numpy")
        device_id: Optional GPU device ID
    """
    # Handle exact OpenHCS memory type values
    if memory_type == "torch":
        cleanup_pytorch_gpu(device_id)
    elif memory_type == "cupy":
        cleanup_cupy_gpu(device_id)
    elif memory_type == "tensorflow":
        cleanup_tensorflow_gpu(device_id)
    elif memory_type == "jax":
        cleanup_jax_gpu(device_id)
    elif memory_type == "pyclesperanto":
        cleanup_pyclesperanto_gpu(device_id)
    elif memory_type == "numpy":
        # CPU memory type - no GPU cleanup needed
        logger.debug(f"No GPU cleanup needed for CPU memory type: {memory_type}")
    else:
        # Fallback for unknown types - try pattern matching
        memory_type_lower = memory_type.lower()
        if "torch" in memory_type_lower or "pytorch" in memory_type_lower:
            cleanup_pytorch_gpu(device_id)
        elif "cupy" in memory_type_lower:
            cleanup_cupy_gpu(device_id)
        elif "tensorflow" in memory_type_lower or "tf" in memory_type_lower:
            cleanup_tensorflow_gpu(device_id)
        elif "jax" in memory_type_lower:
            cleanup_jax_gpu(device_id)
        elif "pyclesperanto" in memory_type_lower or "clesperanto" in memory_type_lower:
            cleanup_pyclesperanto_gpu(device_id)
        else:
            logger.debug(f"Unknown memory type for GPU cleanup: {memory_type}")


def cleanup_numpy_noop(device_id: Optional[int] = None) -> None:
    """
    No-op cleanup for numpy (CPU memory type).

    Args:
        device_id: Optional GPU device ID (ignored for CPU)
    """
    logger.debug("🔥 GPU CLEANUP: No-op for numpy (CPU memory type)")


def cleanup_all_gpu_frameworks(device_id: Optional[int] = None) -> None:
    """
    Clean up GPU memory for all available frameworks.

    Args:
        device_id: Optional GPU device ID
    """
    cleanup_pytorch_gpu(device_id)
    cleanup_cupy_gpu(device_id)
    cleanup_tensorflow_gpu(device_id)
    cleanup_jax_gpu(device_id)
    cleanup_pyclesperanto_gpu(device_id)

    # Also trigger Python garbage collection
    import gc
    gc.collect()

    logger.debug("🔥 GPU CLEANUP: Performed comprehensive cleanup for all GPU frameworks")


# Registry mapping memory types to their cleanup functions
MEMORY_TYPE_CLEANUP_REGISTRY = {
    "torch": cleanup_pytorch_gpu,
    "cupy": cleanup_cupy_gpu,
    "tensorflow": cleanup_tensorflow_gpu,
    "jax": cleanup_jax_gpu,
    "pyclesperanto": cleanup_pyclesperanto_gpu,
    "numpy": cleanup_numpy_noop,
}


def cleanup_memory_by_type(memory_type: str, device_id: Optional[int] = None) -> None:
    """
    Clean up memory using the registered cleanup function for the memory type.

    Args:
        memory_type: OpenHCS memory type string ("torch", "cupy", "tensorflow", "jax", "numpy")
        device_id: Optional GPU device ID
    """
    cleanup_func = MEMORY_TYPE_CLEANUP_REGISTRY.get(memory_type)

    if cleanup_func:
        cleanup_func(device_id)
    else:
        logger.warning(f"No cleanup function registered for memory type: {memory_type}")
        logger.debug(f"Available memory types: {list(MEMORY_TYPE_CLEANUP_REGISTRY.keys())}")


def check_gpu_memory_usage() -> None:
    """
    Check and log current GPU memory usage for all available frameworks.

    This is a utility function for debugging memory issues.
    """
    logger.info("🔍 GPU Memory Usage Report:")

    # Check PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"  PyTorch GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        else:
            logger.info("  PyTorch: No CUDA available")
    except ImportError:
        logger.info("  PyTorch: Not installed")

    # Check CuPy
    try:
        import cupy
        mempool = cupy.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        total_bytes = mempool.total_bytes()
        logger.info(f"  CuPy: {used_bytes / 1024**3:.2f}GB used, {total_bytes / 1024**3:.2f}GB total")
    except ImportError:
        logger.info("  CuPy: Not installed")
    except Exception as e:
        logger.info(f"  CuPy: Error checking memory - {e}")

    # Note: TensorFlow and JAX don't have easy memory introspection
    logger.info("  TensorFlow/JAX: Memory usage not easily queryable")


def log_gpu_memory_usage(context: str = "") -> None:
    """
    Log GPU memory usage with a specific context for tracking.

    Args:
        context: Description of when/where this memory check is happening
    """
    context_str = f" ({context})" if context else ""

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                free_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3 - reserved
                logger.info(f"🔍 VRAM{context_str} GPU {i}: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved, {free_memory:.2f}GB free")
        else:
            logger.info(f"🔍 VRAM{context_str}: No CUDA available")
    except ImportError:
        logger.info(f"🔍 VRAM{context_str}: PyTorch not available")
    except Exception as e:
        logger.warning(f"🔍 VRAM{context_str}: Error checking memory - {e}")


def get_gpu_memory_summary() -> dict:
    """
    Get GPU memory usage as a dictionary for programmatic use.

    Returns:
        Dictionary with memory usage information
    """
    memory_info = {
        "pytorch": {"available": False, "devices": []},
        "cupy": {"available": False, "used_gb": 0, "total_gb": 0}
    }

    # Check PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            memory_info["pytorch"]["available"] = True
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                memory_info["pytorch"]["devices"].append({
                    "device_id": i,
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "total_gb": total,
                    "free_gb": total - reserved
                })
    except ImportError:
        pass
    except Exception:
        pass

    # Check CuPy
    try:
        import cupy
        mempool = cupy.get_default_memory_pool()
        memory_info["cupy"]["available"] = True
        memory_info["cupy"]["used_gb"] = mempool.used_bytes() / 1024**3
        memory_info["cupy"]["total_gb"] = mempool.total_bytes() / 1024**3
    except ImportError:
        pass
    except Exception:
        pass

    return memory_info


def force_comprehensive_cleanup() -> None:
    """
    Force comprehensive GPU cleanup across all frameworks and trigger garbage collection.

    This is the nuclear option for clearing GPU memory when you suspect leaks.
    """
    logger.info("🧹 FORCE COMPREHENSIVE CLEANUP: Starting nuclear cleanup...")

    # Clean all GPU frameworks
    cleanup_all_gpu_frameworks()

    # Multiple rounds of garbage collection
    import gc
    for i in range(3):
        collected = gc.collect()
        logger.debug(f"🧹 Garbage collection round {i+1}: collected {collected} objects")

    # Check memory usage after cleanup
    check_gpu_memory_usage()

    logger.info("🧹 FORCE COMPREHENSIVE CLEANUP: Complete")
