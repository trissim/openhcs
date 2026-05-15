"""Unified progress tracking system.

This module replaces progress_reporter.py and progress_state.py
with a single coherent abstraction.

Public API:
    - ProgressEvent: Immutable progress data
    - ProgressPhase, ProgressStatus: Type-safe enums
    - emit(): Emit progress (convenience wrapper)
    - registry(): Get per-process registry
    - ProgressEmitter: Emitter base class
    - ProgressRegistry: Registry singleton

Usage Examples:

    Runtime (emit progress):
        from openhcs.core.progress import emit, ProgressPhase, ProgressStatus

        emit(
            execution_id="exec-123",
            plate_id="/path/to/plate",
            axis_id="A01",
            step_name="compilation",
            phase=ProgressPhase.COMPILE,
            status=ProgressStatus.RUNNING,
            percent=50.0,
            completed=50,
            total=100
        )

    GUI (listen to progress):
        from openhcs.core.progress import registry

        def on_progress(execution_id, event):
            print(f"{event.phase.value}: {event.percent}%")

        registry().add_listener(on_progress)

    Testing (mock emitter):
        from openhcs.core.progress import NoopEmitter, ProgressEvent

        emitter = NoopEmitter()
        event = ProgressEvent(...)
        emitter.emit(event)
"""

# Version tracking
__version__ = "1.0.0"

# Public API - Types
from .types import (
    ProgressEvent,
    ProgressPhase,
    ProgressStatus,
    ProgressChannel,
    ProgressSemanticsABC,
    phase_channel,
    is_execution_phase,
    is_terminal_event,
    is_failure_event,
    is_success_terminal_event,
    create_event,
)

# Public API - Registry
from .registry import registry

# Public API - Emitters
from .emitters import (
    ProgressEmitter,
    NoopEmitter,
    CallbackEmitter,
    ZMQProgressEmitter,
    LoggingEmitter,
)

# Public API - Exceptions
from .exceptions import (
    ProgressError,
    ProgressValidationError,
    ProgressRegistrationError,
)

# =============================================================================
# Convenience Functions
# =============================================================================

import time
import os
_progress_queue = None


def set_progress_queue(queue):
    """Set progress queue for worker processes (called once at init).

    Args:
        queue: multiprocessing.Queue to put progress events into
    """
    global _progress_queue
    _progress_queue = queue


def emit(**kwargs) -> None:
    """Emit progress event (replaces emit_progress()).

    Invariant emission path:
    1. Validate inputs
    2. Build immutable ProgressEvent
    3. Put serialized event onto configured progress queue

    Args:
        execution_id (str): Execution identifier (required)
        plate_id (str): Plate identifier (required)
        axis_id (str): Axis/well identifier (required)
        step_name (str): Step name (required)
        phase (ProgressPhase): Progress phase (required)
        status (ProgressStatus): Progress status (required)
        percent (float): Progress percentage 0-100 (required)
        completed (int): Number of completed items (default: 0)
        total (int): Total number of items (default: 1)
        **kwargs: Additional metadata fields (optional)

    Raises:
        ValueError: If required fields missing or invalid
        ProgressError: If progress queue is not configured
    """
    # Validate required fields
    required_fields = {
        "execution_id",
        "plate_id",
        "axis_id",
        "step_name",
        "phase",
        "status",
        "percent",
    }
    missing = required_fields - set(kwargs.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Set defaults for optional fields
    if "completed" not in kwargs:
        kwargs["completed"] = 0
    if "total" not in kwargs:
        kwargs["total"] = 1

    if _progress_queue is None:
        raise ProgressError(
            "emit() requires explicit progress queue configuration via "
            "set_progress_queue(queue). No fallback path is allowed."
        )

    event = ProgressEvent(timestamp=time.time(), pid=os.getpid(), **kwargs)
    _progress_queue.put(event.to_dict())


def emit_event(event: ProgressEvent) -> None:
    """Emit an already-constructed progress event through the configured queue."""

    if _progress_queue is None:
        raise ProgressError(
            "emit_event() requires explicit progress queue configuration via "
            "set_progress_queue(queue). No fallback path is allowed."
        )
    _progress_queue.put(event.to_dict())


def get_registry():
    """Get process-local progress registry.

    Alias for registry() function (clearer naming).

    Returns:
        ProgressRegistry singleton instance
    """
    from .registry import registry

    return registry()


__all__ = [
    # Version
    "__version__",
    # Convenience
    "emit",
    "emit_event",
    "get_registry",
]
