"""OpenHCS progress registry backed by generic zmqruntime registry primitives."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Optional

from zmqruntime.progress import EventRegistryMutation, LatestEventRegistry

from .types import ProgressChannel, ProgressEvent, is_terminal_event, phase_channel


class ProgressRegistry:
    """Per-process singleton registry for OpenHCS progress events."""

    _instance: "ProgressRegistry | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "ProgressRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._retention_seconds = 60.0
        self._event_registry: LatestEventRegistry[
            ProgressEvent,
            tuple[str, str, ProgressChannel],
        ] = LatestEventRegistry(
            key_builder=self._event_key,
            is_terminal=is_terminal_event,
            timestamp_of=lambda event: event.timestamp,
            retention_seconds=self._retention_seconds,
        )
        self._initialized = True

    @staticmethod
    def _event_key(event: ProgressEvent) -> tuple[str, str, ProgressChannel]:
        return (event.plate_id, event.axis_id, phase_channel(event.phase))

    def register_event(self, execution_id: str, event: ProgressEvent) -> bool:
        return self._event_registry.register_event(execution_id, event)

    def get_events(self, execution_id: str) -> list[ProgressEvent]:
        return self._event_registry.get_events(execution_id)

    def get_latest_event(self, execution_id: str) -> Optional[ProgressEvent]:
        return self._event_registry.get_latest_event(execution_id)

    def add_listener(self, listener) -> None:
        self._event_registry.add_listener(listener)

    def remove_listener(self, listener) -> bool:
        return self._event_registry.remove_listener(listener)

    def clear_listeners(self) -> None:
        self._event_registry.clear_listeners()

    def add_mutation_listener(
        self,
        listener: Callable[[EventRegistryMutation[ProgressEvent]], None],
    ) -> None:
        self._event_registry.add_mutation_listener(listener)

    def remove_mutation_listener(
        self,
        listener: Callable[[EventRegistryMutation[ProgressEvent]], None],
    ) -> bool:
        return self._event_registry.remove_mutation_listener(listener)

    def clear_mutation_listeners(self) -> None:
        self._event_registry.clear_mutation_listeners()

    def clear_execution(self, execution_id: str) -> None:
        self._event_registry.clear_execution(execution_id)

    def clear_all(self) -> None:
        self._event_registry.clear_all()

    def cleanup_old_executions(self, retention_seconds: Optional[float] = None) -> int:
        return self._event_registry.cleanup_old_executions(retention_seconds)

    def get_execution_ids(self) -> list[str]:
        return self._event_registry.get_execution_ids()

    def get_event_count(self, execution_id: str) -> int:
        return self._event_registry.get_event_count(execution_id)


def registry() -> ProgressRegistry:
    """Get process-local progress registry (singleton)."""
    return ProgressRegistry()
