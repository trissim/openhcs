"""Debug progress notification helpers for the PyQt batch workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from openhcs.core.debug import DebugProgressContext, DebugSnapshot
from openhcs.core.progress import ProgressEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DebugSnapshotAvailableNotification:
    """GUI-side notification that one debug snapshot can be read."""

    progress_event: ProgressEvent
    debug_context: DebugProgressContext
    snapshot: DebugSnapshot | None = None


DebugSnapshotListener = Callable[[DebugSnapshotAvailableNotification], None]


class DebugProgressNotificationService:
    """Converts progress events into typed debug snapshot notifications."""

    def __init__(self) -> None:
        self._listeners: list[DebugSnapshotListener] = []

    def add_listener(self, listener: DebugSnapshotListener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: DebugSnapshotListener) -> bool:
        try:
            self._listeners.remove(listener)
        except ValueError:
            return False
        return True

    def notify_from_progress_event(self, event: ProgressEvent, *, zmq_client) -> None:
        debug_context = self._debug_context_from_event(event)
        if debug_context is None or debug_context.snapshot_id is None:
            return
        notification = DebugSnapshotAvailableNotification(
            progress_event=event,
            debug_context=debug_context,
            snapshot=self._read_debug_snapshot_from_server(
                debug_context,
                zmq_client=zmq_client,
            ),
        )
        for listener in tuple(self._listeners):
            listener(notification)

    @staticmethod
    def _debug_context_from_event(
        event: ProgressEvent,
    ) -> DebugProgressContext | None:
        if not event.context:
            return None
        try:
            return DebugProgressContext.from_progress_context(event.context)
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _read_debug_snapshot_from_server(
        debug_context: DebugProgressContext,
        *,
        zmq_client,
    ) -> DebugSnapshot | None:
        if (
            debug_context.snapshot_store_ref is None
            or debug_context.snapshot_id is None
            or zmq_client is None
        ):
            return None
        try:
            return zmq_client.get_debug_snapshot(
                debug_session_id=debug_context.debug_session_id,
                snapshot_id=debug_context.snapshot_id,
                snapshot_store_ref=debug_context.snapshot_store_ref,
                snapshot_store_backend=debug_context.snapshot_store_backend,
            )
        except Exception as error:
            logger.debug("Server debug snapshot readback failed: %s", error)
            return None


def is_debug_progress_service_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_debug_progress_service_export(name, value)
)
