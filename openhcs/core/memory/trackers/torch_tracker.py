import logging
from typing import TYPE_CHECKING, Dict

logger = logging.getLogger(__name__)
 
# Attempt to import torch and check for CUDA availability at module level.
if TYPE_CHECKING:
    import torch
try:
    import torch
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
    if not TORCH_CUDA_AVAILABLE:
        logger.info("PyTorch found, but CUDA is not available. TorchMemoryTracker will not be functional.")
except ImportError:
    torch = None # type: ignore
    TORCH_CUDA_AVAILABLE = False
    logger.info("PyTorch library not found. TorchMemoryTracker will not be available.")
 
 
class TorchMemoryTracker: # Implements MemoryTracker protocol
    """
    GPU Memory tracker using the PyTorch library.
    Provides an estimation of free memory (reserved - allocated).
    """
    name: str = "torch" # type: ignore
    accurate: bool = False # PyTorch's free is more like 'unallocated within reserved block'
    synchronous: bool = True # PyTorch memory calls are synchronous
 
    def __init__(self, **kwargs: "dict"):
        """
        Initializes the TorchMemoryTracker.
        Raises ImportError if PyTorch or PyTorch CUDA support is not available.
        """
        if not torch or not TORCH_CUDA_AVAILABLE: # type: ignore
            raise ImportError(
                "PyTorch library with CUDA support is not available. TorchMemoryTracker cannot be used."
            )
        # kwargs are ignored for this tracker
        logger.debug("TorchMemoryTracker initialized.")
 
    def get_free_memory(self, device_id: int) -> float:
        """
        Get the free memory for the specified GPU device using PyTorch.
        Calculates free memory as: total reserved by PyTorch - currently allocated by PyTorch.
        This is an estimate of memory PyTorch could potentially use without further OS allocation.
 
        Args:
            device_id: The integer ID of the GPU device.
 
        Returns:
            Estimated free memory within PyTorch's reserved blocks in Megabytes (MB).
 
        Raises:
            RuntimeError: If there's an issue querying memory with PyTorch
                           (e.g., invalid device ID, CUDA error).
        """ # type: ignore
        if not torch or not TORCH_CUDA_AVAILABLE: # Defensive check
            raise ImportError("PyTorch library with CUDA support is not available.")
 
        try:
            # Ensure the device_id is valid for torch
            if device_id < 0 or device_id >= torch.cuda.device_count():
                raise RuntimeError(f"Invalid PyTorch device_id: {device_id}. Available devices: {torch.cuda.device_count()}")
 
            # All memory figures from torch are in bytes
            reserved_bytes = torch.cuda.memory_reserved(device_id)
            allocated_bytes = torch.cuda.memory_allocated(device_id)
             
            # "Free" within PyTorch's context is often considered the unallocated portion of its reserved pool
            free_in_pytorch_pool_bytes = reserved_bytes - allocated_bytes
             
            # To get a sense of overall device free memory (less accurate via torch alone):
            # total_memory_bytes = torch.cuda.get_device_properties(device_id).total_memory
            # free_overall_approx_bytes = total_memory_bytes - allocated_bytes
            # However, the plan asks for free memory, and mem_info like CuPy is better.
            # For torch, `reserved - allocated` is a common way to see what's free *within its own management*.
            # Let's stick to `reserved - allocated` as "free" for this tracker, acknowledging its nature.
 
            free_mb = free_in_pytorch_pool_bytes / (1024 * 1024)
             
            logger.debug(
                f"Torch Device {device_id}: Reserved: {reserved_bytes / (1024*1024):.2f} MB, "
                f"Allocated: {allocated_bytes / (1024*1024):.2f} MB, "
                f"Free in Pool: {free_mb:.2f} MB"
            )
            return free_mb
        except RuntimeError as e: # PyTorch often raises RuntimeError for CUDA issues
            logger.error(f"PyTorch CUDA error querying memory for device {device_id}: {e}")
            raise RuntimeError(f"PyTorch failed to get memory info for device {device_id}: {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error in TorchMemoryTracker for device {device_id}: {e}")
            raise RuntimeError(f"Unexpected error getting PyTorch memory for device {device_id}: {e}") from e