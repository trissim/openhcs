import logging
from typing import TYPE_CHECKING, Dict

logger = logging.getLogger(__name__)
 
# Attempt to import tensorflow at the module level.
if TYPE_CHECKING:
    import tensorflow as tf
 
try:
    import tensorflow as tf

    # Check if GPUs are configured and visible to TensorFlow
    physical_gpus = tf.config.list_physical_devices('GPU')
    TF_GPU_AVAILABLE = len(physical_gpus) > 0
    if not TF_GPU_AVAILABLE:
        logger.info("TensorFlow found, but no GPUs are configured/visible. TFMemoryTracker will not be functional for GPU queries.")
except ImportError:
    tf = None # type: ignore
    TF_GPU_AVAILABLE = False
    logger.info("TensorFlow library not found. TFMemoryTracker will not be available.")
except Exception as e: # Catch other potential TF init errors
    tf = None # type: ignore
    TF_GPU_AVAILABLE = False
    logger.warning(f"TensorFlow import or GPU check failed: {e}. TFMemoryTracker may not be functional.")
 
 
class TFMemoryTracker: # Implements MemoryTracker protocol
    """
    GPU Memory tracker using the TensorFlow library.
    Provides an estimation of free memory.
    """
    name: str = "tensorflow" # type: ignore
    accurate: bool = False # TF's memory management is complex; this is an estimate.
    synchronous: bool = True # TensorFlow memory calls are synchronous
 
    def __init__(self, **kwargs: "dict"):
        """
        Initializes the TFMemoryTracker.
        Raises ImportError if TensorFlow is not installed or no GPUs are available to TF.
        """
        if not tf: # type: ignore
            raise ImportError("TensorFlow library is not installed. TFMemoryTracker cannot be used.")
        if not TF_GPU_AVAILABLE:
            # This check is important because TF might be installed but configured for CPU only.
            # Or no GPUs are physically present/drivers set up.
            raise ImportError("TensorFlow is installed, but no GPUs are available/configured for it.")
        
        # kwargs are ignored for this tracker
        logger.debug("TFMemoryTracker initialized.")
 
    def get_free_memory(self, device_id: int) -> float:
        """
        Get an estimate of free memory for the specified GPU device using TensorFlow.
        Calculates free memory as: Total Device Memory - Current TF Usage on Device.
 
        Args:
            device_id: The integer ID of the GPU device (relative to TF's list of physical GPUs).
 
        Returns:
            Estimated free memory in Megabytes (MB).
 
        Raises:
            RuntimeError: If there's an issue querying memory (e.g., invalid device ID).
        """
        if not tf or not TF_GPU_AVAILABLE: # type: ignore # Defensive check
            raise ImportError("TensorFlow with GPU support is not available.")
 
        try:
            physical_gpus = tf.config.list_physical_devices('GPU')
            if not (0 <= device_id < len(physical_gpus)):
                raise RuntimeError(
                    f"Invalid TensorFlow device_id: {device_id}. "
                    f"Available GPU devices: {len(physical_gpus)}"
                )
            
            target_gpu = physical_gpus[device_id]
 
            # Get total memory for the device
            # tf.config.experimental.get_device_details was an option but might not always be present
            # or might be slow. A more common way is to get memory limits if set, or infer.
            # For physical devices, there isn't a direct "total memory" API as simple as CuPy's.
            # Let's try to get memory details.
            try:
                details = tf.config.experimental.get_device_details(target_gpu)
                total_memory_mb = details.get('memory_size_in_mb') # This key might exist
                if total_memory_mb is None:
                    # Fallback or alternative way to get total memory if 'memory_size_in_mb' is not in details
                    # This part is tricky as TF doesn't expose total physical memory as easily as CuPy.
                    # For now, if not in details, we cannot accurately report free memory.
                    logger.warning(f"Could not determine total memory for TF GPU device {device_id} via get_device_details. Reporting 0 free memory.")
                    return 0.0
            except Exception as e_details:
                logger.warning(f"Could not get device details for TF GPU {device_id}: {e_details}. Reporting 0 free memory.")
                return 0.0
 
 
            # Get current memory usage by TensorFlow on this GPU
            # The key for device in get_memory_info is "GPU:<device_id_str>"
            memory_info = tf.config.experimental.get_memory_info(f"GPU:{device_id}")
            current_usage_bytes = memory_info.get('current', 0) # Bytes currently used by TF
            current_usage_mb = current_usage_bytes / (1024 * 1024)
 
            free_mb = total_memory_mb - current_usage_mb
            
            logger.debug(
                f"TF Device {device_id}: Total Memory (Reported): {total_memory_mb:.2f} MB, "
                f"Current TF Usage: {current_usage_mb:.2f} MB, "
                f"Estimated Free: {free_mb:.2f} MB"
            )
            return max(0.0, free_mb) # Ensure non-negative
 
        except RuntimeError as e: # TF often raises RuntimeError for device issues
            logger.error(f"TensorFlow error querying memory for device {device_id}: {e}")
            raise RuntimeError(f"TensorFlow failed to get memory info for device {device_id}: {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error in TFMemoryTracker for device {device_id}: {e}")
            raise RuntimeError(f"Unexpected error getting TensorFlow memory for device {device_id}: {e}") from e