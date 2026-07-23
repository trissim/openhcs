"""Runtime artifact progress notification helpers for PyQt workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from openhcs.core.progress import ProgressEvent
from openhcs.core.progress.runtime_artifacts import RuntimeArtifactProgressPayload


@dataclass(frozen=True, slots=True)
class RuntimeArtifactAvailableNotification:
    """UI notification for runtime artifact addresses carried by progress."""

    event: ProgressEvent
    payload: RuntimeArtifactProgressPayload


RuntimeArtifactListener = Callable[[RuntimeArtifactAvailableNotification], None]


class RuntimeArtifactProgressNotificationService:
    """Dispatch generic runtime artifact payloads from progress events."""

    def __init__(self) -> None:
        self._listeners: list[RuntimeArtifactListener] = []

    def add_listener(self, listener: RuntimeArtifactListener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: RuntimeArtifactListener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError as error:
            raise ValueError(
                "Runtime artifact listener removal failed: listener not registered."
            ) from error

    def notify_from_progress_event(self, event: ProgressEvent) -> None:
        payload = RuntimeArtifactProgressPayload.from_context(event.context)
        if payload is None:
            return
        notification = RuntimeArtifactAvailableNotification(
            event=event,
            payload=payload,
        )
        for listener in tuple(self._listeners):
            listener(notification)
