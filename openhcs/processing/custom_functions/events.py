"""Domain change events for process-local custom-function declarations."""

from __future__ import annotations

import threading
from collections.abc import Callable


class CustomFunctionChangedEvent:
    """Thread-safe domain event with an immutable subscriber snapshot."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[], None]] = []
        self._lock = threading.RLock()

    def subscribe(self, callback: Callable[[], None]) -> None:
        """Subscribe one callback exactly once."""

        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def emit(self) -> None:
        """Notify the subscriber snapshot owned at emission start."""

        with self._lock:
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber()


custom_function_changed = CustomFunctionChangedEvent()
