"""Live measurement progress notification helpers for PyQt workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from openhcs.core.progress import ProgressEvent
from openhcs.core.progress.live_measurements import (
    LiveMeasurementProgressPayload,
)


@dataclass(frozen=True, slots=True)
class LiveMeasurementAvailableNotification:
    """UI notification for live measurement previews carried by progress."""

    event: ProgressEvent
    payload: LiveMeasurementProgressPayload


LiveMeasurementListener = Callable[[LiveMeasurementAvailableNotification], None]


class LiveMeasurementProgressNotificationService:
    """Dispatch live measurement preview payloads from progress events."""

    def __init__(self) -> None:
        self._listeners: list[LiveMeasurementListener] = []

    def add_listener(self, listener: LiveMeasurementListener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: LiveMeasurementListener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError as exc:
            raise ValueError(
                "Live measurement listener removal failed: listener not registered."
            ) from exc

    def notify_from_progress_event(self, event: ProgressEvent) -> None:
        payload = LiveMeasurementProgressPayload.from_context(event.context)
        if payload is None:
            return
        notification = LiveMeasurementAvailableNotification(
            event=event,
            payload=payload,
        )
        for listener in tuple(self._listeners):
            listener(notification)
