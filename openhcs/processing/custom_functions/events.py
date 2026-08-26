"""Domain change events for process-local custom-function declarations."""

from __future__ import annotations

import threading
from collections.abc import Callable
from types import MethodType
from weakref import ReferenceType, WeakMethod, ref


class CustomFunctionChangedEvent:
    """Thread-safe domain event with an immutable subscriber snapshot."""

    def __init__(self) -> None:
        self._subscribers: list[ReferenceType[Callable[[], None]]] = []
        self._lock = threading.RLock()

    @staticmethod
    def _is_same_callback(
        left: Callable[[], None],
        right: Callable[[], None],
    ) -> bool:
        """Compare callback identity while accounting for recreated bound methods."""

        if isinstance(left, MethodType) and isinstance(right, MethodType):
            return left.__self__ is right.__self__ and left.__func__ is right.__func__
        return left is right

    def subscribe(self, callback: Callable[[], None]) -> None:
        """Observe a callback without becoming its lifetime owner."""

        with self._lock:
            live_subscriptions: list[ReferenceType[Callable[[], None]]] = []
            callback_is_registered = False
            for subscription in self._subscribers:
                subscriber = subscription()
                if subscriber is None:
                    continue
                live_subscriptions.append(subscription)
                if self._is_same_callback(subscriber, callback):
                    callback_is_registered = True
            if not callback_is_registered:
                live_subscriptions.append(
                    WeakMethod(callback)
                    if isinstance(callback, MethodType)
                    else ref(callback)
                )
            self._subscribers = live_subscriptions

    def emit(self) -> None:
        """Notify the subscriber snapshot owned at emission start."""

        with self._lock:
            live_subscriptions: list[ReferenceType[Callable[[], None]]] = []
            subscribers: list[Callable[[], None]] = []
            for subscription in self._subscribers:
                subscriber = subscription()
                if subscriber is None:
                    continue
                live_subscriptions.append(subscription)
                subscribers.append(subscriber)
            self._subscribers = live_subscriptions
        for subscriber in subscribers:
            subscriber()


custom_function_changed = CustomFunctionChangedEvent()
