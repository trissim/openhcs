import logging
from dataclasses import dataclass  # Added dataclass
from typing import Protocol, Type, runtime_checkable  # Added Type

logger = logging.getLogger(__name__)

@runtime_checkable
class MemoryTracker(Protocol):
    """
    Protocol for GPU memory trackers.

    Implementations of this protocol provide a standardized way to query
    free memory on a specific GPU device.
    """

    name: str
    """A unique string identifier for the tracker (e.g., 'cupy', 'torch')."""

    accurate: bool
    """Indicates if the tracker is considered to provide accurate free memory
    (True) or an estimation/available memory (False)."""

    synchronous: bool
    """Indicates if the memory query operation is synchronous (True) or potentially asynchronous (False).
    Most direct library calls are synchronous."""

    def __init__(self, **kwargs: dict):
        """
        Initializes the memory tracker.
        kwargs can be used for tracker-specific configurations, though typically
        trackers might not need much configuration beyond what's hardcoded or
        determined at import time.
        """
        ... # pragma: no cover

    def get_free_memory(self, device_id: int) -> float:
        """
        Get the free memory for the specified GPU device.

        Args:
            device_id: The integer ID of the GPU device.

        Returns:
            Free memory in Megabytes (MB).

        Raises:
            ImportError: If the underlying library required by the tracker is not installed.
            RuntimeError: If there's an issue querying memory (e.g., invalid device ID for the library).
        """
        ... # pragma: no cover


@dataclass(frozen=True)
class MemoryTrackerSpec:
    """
    Specification for a MemoryTracker implementation.
    Holds metadata about a tracker class.
    """
    name: str
    """The unique name of the tracker (e.g., 'cupy')."""
    
    tracker_cls: Type[MemoryTracker]
    """The class of the memory tracker itself."""
    
    accurate: bool
    """Whether the tracker provides accurate free memory."""
    
    synchronous: bool
    """Whether the tracker's get_free_memory operation is synchronous."""