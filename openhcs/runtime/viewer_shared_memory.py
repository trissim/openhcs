"""Cross-platform receiver ownership for viewer shared-memory payloads."""

from __future__ import annotations

from collections.abc import Sequence
from multiprocessing import resource_tracker, shared_memory
from multiprocessing.shared_memory import _USE_POSIX

import numpy as np


class SenderOwnedSharedMemoryAttachment:
    """Copy a sender-owned array without transferring allocation ownership."""

    @staticmethod
    def _release_receiver_tracking(memory: shared_memory.SharedMemory) -> None:
        if _USE_POSIX:
            resource_tracker.unregister(memory._name, "shared_memory")

    @classmethod
    def copy_array(
        cls,
        *,
        name: str,
        shape: Sequence[int],
        dtype: str | np.dtype,
    ) -> np.ndarray:
        """Attach, copy, and close while leaving unlinking to the sender."""
        memory = shared_memory.SharedMemory(name=name)
        cls._release_receiver_tracking(memory)
        try:
            return np.ndarray(shape, dtype=dtype, buffer=memory.buf).copy()
        finally:
            memory.close()
