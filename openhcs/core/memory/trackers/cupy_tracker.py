import logging
from openhcs.core.utils import optional_import

logger = logging.getLogger(__name__)

# Import cupy using the optional_import utility
cupy = optional_import("cupy")
CUPY_AVAILABLE = cupy is not None
if not CUPY_AVAILABLE:
    logger.info("CuPy library not found. CuPyMemoryTracker will not be available.")

class CuPyMemoryTracker: # Implements MemoryTracker protocol
    """
    GPU Memory tracker using the CuPy library.
    Considered to provide accurate free memory.
    """
    name: str = "cupy"
    accurate: bool = True
    synchronous: bool = True # CuPy memory calls are synchronous

    def __init__(self, **kwargs):
        """
        Initializes the CuPyMemoryTracker.
        
        Args:
            **kwargs: Configuration parameters (not used, but required by MemoryTracker protocol)
            
        Raises:
            ImportError: If CuPy is not installed.
        """
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy library is not installed. CuPyMemoryTracker cannot be used.")
        logger.debug("CuPyMemoryTracker initialized.")

    def get_free_memory(self, device_id: int) -> float:
        """
        Get the free memory for the specified GPU device using CuPy.

        Args:
            device_id: The integer ID of the GPU device.

        Returns:
            Free memory in Megabytes (MB).

        Raises:
            RuntimeError: If there's an issue querying memory with CuPy 
                          (e.g., invalid device ID, CUDA error).
        """
        if not CUPY_AVAILABLE: # Should have been caught in __init__, but defensive check
            raise ImportError("CuPy library is not available to get free memory.")
        
        try:
            with cupy.cuda.Device(device_id):
                # mem_info returns (free_bytes, total_bytes)
                free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
            
            free_mb = free_bytes / (1024 * 1024)
            logger.debug(f"CuPy Device {device_id}: Free Memory: {free_mb:.2f} MB, Total: {total_bytes / (1024*1024):.2f} MB")
            return free_mb
        except cupy.cuda.runtime.CUDARuntimeError as e:
            # This can happen if device_id is invalid or other CUDA issues
            logger.error(f"CuPy CUDA error querying memory for device {device_id}: {e}")
            raise RuntimeError(f"CuPy failed to get memory info for device {device_id}: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors
            logger.exception(f"Unexpected error in CuPyMemoryTracker for device {device_id}: {e}")
            raise RuntimeError(f"Unexpected error getting CuPy memory for device {device_id}: {e}") from e