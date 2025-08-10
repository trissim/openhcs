import logging
from openhcs.core.utils import optional_import

logger = logging.getLogger(__name__)

# Import pyclesperanto using the optional_import utility
cle = optional_import("pyclesperanto")
PYCLESPERANTO_AVAILABLE = cle is not None
if not PYCLESPERANTO_AVAILABLE:
    logger.info("pyclesperanto library not found. PyclesperantoMemoryTracker will not be available.")

class PyclesperantoMemoryTracker: # Implements MemoryTracker protocol
    """
    GPU Memory tracker using the pyclesperanto library.
    Note: pyclesperanto doesn't provide direct memory info, so this is a best-effort implementation.
    """
    name: str = "pyclesperanto"
    accurate: bool = False  # pyclesperanto doesn't provide direct memory info
    synchronous: bool = True  # pyclesperanto operations are synchronous

    def __init__(self, **kwargs):
        """
        Initializes the PyclesperantoMemoryTracker.
        
        Args:
            **kwargs: Configuration parameters (not used, but required by MemoryTracker protocol)
            
        Raises:
            ImportError: If pyclesperanto is not installed.
        """
        if not PYCLESPERANTO_AVAILABLE:
            raise ImportError("pyclesperanto library is not installed. PyclesperantoMemoryTracker cannot be used.")
        logger.debug("PyclesperantoMemoryTracker initialized.")

    def get_free_memory(self, device_id: int) -> float:
        """
        Get the free memory for the specified GPU device using pyclesperanto.

        Note: pyclesperanto doesn't provide direct memory info through OpenCL,
        so this returns a conservative estimate.

        Args:
            device_id: The integer ID of the GPU device.

        Returns:
            Free memory in Megabytes (MB) - conservative estimate.

        Raises:
            RuntimeError: If there's an issue querying the device.
        """
        if not PYCLESPERANTO_AVAILABLE:
            raise ImportError("pyclesperanto library is not available to get free memory.")
        
        try:
            # Get available devices
            devices = cle.list_available_devices()
            
            if device_id >= len(devices):
                raise RuntimeError(f"Device ID {device_id} not available. Available devices: {len(devices)}")
            
            # Select the device
            cle.select_device(device_id)
            
            # pyclesperanto doesn't provide direct memory info through OpenCL
            # Return a conservative estimate based on typical GPU memory
            # This is not accurate but provides a fallback
            conservative_estimate_mb = 1024.0  # 1GB conservative estimate
            
            logger.debug(f"pyclesperanto Device {device_id}: Conservative free memory estimate: {conservative_estimate_mb:.2f} MB")
            logger.warning("pyclesperanto memory tracking is not accurate - using conservative estimate")
            
            return conservative_estimate_mb
            
        except Exception as e:
            logger.exception(f"Error in PyclesperantoMemoryTracker for device {device_id}: {e}")
            raise RuntimeError(f"Error getting pyclesperanto memory for device {device_id}: {e}") from e
