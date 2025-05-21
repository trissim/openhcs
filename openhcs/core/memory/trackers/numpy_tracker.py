"""
NumPy-based memory tracker for CPU memory usage.

This module provides a memory tracker that uses NumPy to monitor CPU memory usage.
It implements the MemoryTrackerInterface and provides methods to get available,
total, and used memory for the CPU.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy
- Clause 246 — Statelessness Mandate
"""

import logging
import os
from typing import TYPE_CHECKING, Dict

import psutil

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Check if NumPy is available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False
    logger.info("NumPy library not found. NumPyMemoryTracker will not be available.")


class NumPyMemoryTracker:  # Implements MemoryTracker protocol
    """
    NumPy-based memory tracker for CPU memory usage.
    
    This class implements the MemoryTracker protocol and provides methods to
    get available, total, and used memory for the CPU using NumPy and system
    memory information.
    """

    # Class attributes for declarative capability specification
    name: str = "numpy"
    accurate: bool = True
    synchronous: bool = True
    
    def __init__(self, **kwargs: "dict"):
        """
        Initialize the NumPy memory tracker.
        
        Args:
            **kwargs: Configuration parameters (not used, but required by MemoryTracker protocol)
            
        Raises:
            ImportError: If NumPy is not installed.
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy library is not installed. NumPyMemoryTracker cannot be used.")
        logger.debug("NumPyMemoryTracker initialized.")
    
    def get_free_memory(self, device_id: int = 0) -> float:
        """
        Get the free memory for the CPU using "np.ndarray".
        
        Args:
            device_id: The device ID (ignored for CPU, defaults to 0)
            
        Returns:
            Free memory in Megabytes (MB)
            
        Raises:
            RuntimeError: If there's an issue querying memory
        """
        if not NUMPY_AVAILABLE:  # Should have been caught in __init__, but defensive check
            raise ImportError("NumPy library is not available to get free memory.")
        
        try:
            # For CPU memory, device_id is ignored
            # Use psutil to get system memory information
            mem_info = psutil.virtual_memory()
            free_bytes = mem_info.available
            
            free_mb = free_bytes / (1024 * 1024)
            logger.debug(f"NumPy CPU: Free Memory: {free_mb:.2f} MB, Total: {mem_info.total / (1024*1024):.2f} MB")
            return free_mb
        except Exception as e:
            # Catch any unexpected errors
            logger.exception(f"Unexpected error in NumPyMemoryTracker: {e}")
            raise RuntimeError(f"Unexpected error getting CPU memory: {e}") from e
    
    def get_numpy_memory_info(self) -> Dict[str, float]:
        """
        Get memory information specific to "np.ndarray" arrays.
        
        Returns:
            Dictionary containing NumPy memory information in MB
            
        Raises:
            RuntimeError: If there's an issue querying memory
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy library is not available to get memory information.")
        
        try:
            # Get the process memory info
            process = psutil.Process(os.getpid())
            process_memory_mb = process.memory_info().rss / (1024 * 1024)
            
            # Create a large NumPy array to measure memory increase
            array_size = 10**7  # 10 million elements
            before_memory_mb = process.memory_info().rss / (1024 * 1024)
            test_array = np.ones(array_size, dtype="np.float64")
            after_memory_mb = process.memory_info().rss / (1024 * 1024)
            
            # Calculate memory per element
            memory_per_element_mb = (after_memory_mb - before_memory_mb) / array_size
            
            # Clean up
            del test_array
            
            return {
                "process_memory_mb": process_memory_mb,
                "memory_per_float64_mb": memory_per_element_mb,
                "estimated_numpy_overhead_mb": memory_per_element_mb * array_size - (array_size * 8 / (1024 * 1024))
            }
        except Exception as e:
            # Catch any unexpected errors
            logger.exception(f"Unexpected error in NumPyMemoryTracker.get_numpy_memory_info: {e}")
            raise RuntimeError(f"Unexpected error getting NumPy memory information: {e}") from e
    
    @classmethod
    def create(cls, **kwargs) -> 'NumPyMemoryTracker':
        """Create a new NumPyMemoryTracker instance using "np.ndarray".
        
        Args:
            **kwargs: Additional arguments (ignored)
            
        Returns:
            A new NumPyMemoryTracker instance
        """
        return cls()
