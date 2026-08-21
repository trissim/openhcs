"""Cooperative cancellation authority for in-process pipeline execution."""

from __future__ import annotations

from threading import Event, RLock


class ExecutionCancelledError(RuntimeError):
    """Raised when an execution reaches a cooperative cancellation boundary."""


class ExecutionCancellationSignal:
    """Thread-safe cooperative cancellation signal for one execution scope."""

    def __init__(self) -> None:
        self._requested = Event()

    @property
    def is_requested(self) -> bool:
        """Whether cancellation has been requested for the current execution."""

        return self._requested.is_set()

    def request(self) -> None:
        """Request cancellation at the next safe execution boundary."""

        self._requested.set()

    def raise_if_requested(self, phase: str) -> None:
        """Stop at a safe boundary when cancellation has been requested."""

        if self.is_requested:
            raise ExecutionCancelledError(f"Execution cancelled {phase}")


class ExecutionCancellationAuthority:
    """Own cancellation requests across sequential execution scopes."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._active: ExecutionCancellationSignal | None = None
        self._pending = False

    def begin(self) -> ExecutionCancellationSignal:
        """Open a new scope without losing a request made just before entry."""

        with self._lock:
            if self._active is not None:
                raise RuntimeError("An execution cancellation scope is already active")
            signal = ExecutionCancellationSignal()
            if self._pending:
                signal.request()
            self._pending = False
            self._active = signal
            return signal

    def finish(self, signal: ExecutionCancellationSignal) -> None:
        """Close the exact scope returned by :meth:`begin`."""

        with self._lock:
            if self._active is not signal:
                raise RuntimeError("Cannot finish a non-active cancellation scope")
            self._active = None

    def request(self) -> None:
        """Request cancellation for active work or the imminent next scope."""

        with self._lock:
            if self._active is None:
                self._pending = True
                return
            self._active.request()

    def raise_if_requested(self, phase: str) -> None:
        """Stop the active or imminent execution at a safe boundary."""

        with self._lock:
            active = self._active
            pending = self._pending
        if active is not None:
            active.raise_if_requested(phase)
        elif pending:
            raise ExecutionCancelledError(f"Execution cancelled {phase}")
