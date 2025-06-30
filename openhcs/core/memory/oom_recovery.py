"""
GPU Out of Memory (OOM) recovery utilities.

Provides comprehensive OOM detection and cache clearing for all supported
GPU frameworks in OpenHCS.
"""

import gc
from typing import Optional

from openhcs.constants.constants import (
    MEMORY_TYPE_TORCH,
    MEMORY_TYPE_CUPY,
    MEMORY_TYPE_TENSORFLOW,
    MEMORY_TYPE_JAX,
    MEMORY_TYPE_PYCLESPERANTO,
)


def _is_oom_error(e: Exception, memory_type: str) -> bool:
    """
    Detect Out of Memory errors for all GPU frameworks.
    
    Args:
        e: Exception to check
        memory_type: Memory type from MemoryType enum
        
    Returns:
        True if exception is an OOM error for the given framework
    """
    error_str = str(e).lower()
    
    # Framework-specific exception types
    if memory_type == MEMORY_TYPE_TORCH:
        import torch
        if hasattr(torch.cuda, 'OutOfMemoryError') and isinstance(e, torch.cuda.OutOfMemoryError):
            return True
            
    elif memory_type == MEMORY_TYPE_CUPY:
        import cupy as cp
        if hasattr(cp.cuda.memory, 'OutOfMemoryError') and isinstance(e, cp.cuda.memory.OutOfMemoryError):
            return True
        if hasattr(cp.cuda.runtime, 'CUDARuntimeError') and isinstance(e, cp.cuda.runtime.CUDARuntimeError):
            return True
            
    elif memory_type == MEMORY_TYPE_TENSORFLOW:
        import tensorflow as tf
        if hasattr(tf.errors, 'ResourceExhaustedError') and isinstance(e, tf.errors.ResourceExhaustedError):
            return True
        if hasattr(tf.errors, 'InvalidArgumentError') and isinstance(e, tf.errors.InvalidArgumentError):
            return True
    
    # String-based detection for all frameworks
    oom_patterns = [
        'out of memory', 'outofmemoryerror', 'resource_exhausted',
        'cuda_error_out_of_memory', 'cl_mem_object_allocation_failure',
        'cl_out_of_resources', 'oom when allocating', 'cannot allocate memory',
        'allocation failure', 'memory exhausted', 'resourceexhausted'
    ]
    
    return any(pattern in error_str for pattern in oom_patterns)


def _clear_cache_for_memory_type(memory_type: str, device_id: Optional[int] = None):
    """
    Clear GPU cache for specific memory type.
    
    Args:
        memory_type: Memory type from MemoryType enum
        device_id: GPU device ID (optional)
    """
    if memory_type == MEMORY_TYPE_TORCH:
        import torch
        torch.cuda.empty_cache()
        if device_id is not None:
            with torch.cuda.device(device_id):
                torch.cuda.synchronize()
        else:
            torch.cuda.synchronize()
            
    elif memory_type == MEMORY_TYPE_CUPY:
        import cupy as cp
        if device_id is not None:
            with cp.cuda.Device(device_id):
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                cp.cuda.runtime.deviceSynchronize()
        else:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
            
    elif memory_type == MEMORY_TYPE_TENSORFLOW:
        # TensorFlow uses automatic memory management
        gc.collect()
        
    elif memory_type == MEMORY_TYPE_JAX:
        import jax
        jax.clear_caches()
        gc.collect()
        
    elif memory_type == MEMORY_TYPE_PYCLESPERANTO:
        import pyclesperanto as cle
        if device_id is not None and hasattr(cle, 'select_device'):
            devices = cle.list_available_devices()
            if device_id < len(devices):
                cle.select_device(device_id)
        gc.collect()
    
    # Always trigger Python garbage collection
    gc.collect()


def _execute_with_oom_recovery(func_callable, memory_type: str, max_retries: int = 2):
    """
    Execute function with automatic OOM recovery.
    
    Args:
        func_callable: Function to execute
        memory_type: Memory type from MemoryType enum
        max_retries: Maximum number of retry attempts
        
    Returns:
        Function result
        
    Raises:
        Original exception if not OOM or retries exhausted
    """
    for attempt in range(max_retries + 1):
        try:
            return func_callable()
        except Exception as e:
            if not _is_oom_error(e, memory_type) or attempt == max_retries:
                raise
                
            # Clear cache and retry
            _clear_cache_for_memory_type(memory_type)
