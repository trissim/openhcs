"""Declaration-derived lazy access to optional array frameworks."""

from __future__ import annotations

import threading
from typing import Any

from arraybridge import MemoryType


class LazyFrameworkModule:
    """Thread-safe proxy importing one ArrayBridge framework on first use."""

    def __init__(
        self,
        memory_type: MemoryType,
        attribute_path: tuple[str, ...] = (),
    ) -> None:
        self._memory_type = memory_type
        self._attribute_path = attribute_path
        self._module: Any | None = None
        self._resolved = False
        self._lock = threading.Lock()

    def is_installed(self) -> bool:
        """Check package presence without importing the framework."""

        return self._memory_type.is_installed()

    def _ensure_imported(self) -> Any | None:
        if not self._resolved:
            with self._lock:
                if not self._resolved:
                    module = self._memory_type.import_if_installed()
                    for attribute in self._attribute_path:
                        if module is None:
                            break
                        module = getattr(module, attribute)
                    self._module = module
                    self._resolved = True
        return self._module

    def __getattr__(self, name: str) -> Any:
        module = self._ensure_imported()
        if module is None:
            raise ImportError(
                f"Module {self._memory_type.import_name!r} is not installed."
            )
        return getattr(module, name)

    def __bool__(self) -> bool:
        return self._ensure_imported() is not None


for _memory_type in MemoryType:
    globals()[_memory_type.value] = LazyFrameworkModule(_memory_type)

tf = globals()[MemoryType.TENSORFLOW.value]


__all__ = [
    "LazyFrameworkModule",
    "tf",
    *(memory_type.value for memory_type in MemoryType),
]
