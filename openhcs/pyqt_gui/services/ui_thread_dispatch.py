"""UI-thread dispatch primitives for local PyQt agent bridge operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event
from typing import Generic, TypeVar

from PyQt6.QtCore import QCoreApplication, QObject, QThread, Qt, pyqtSignal


ResultT = TypeVar("ResultT")


class UiThreadDispatchError(RuntimeError):
    """Raised when a UI bridge operation cannot be dispatched to Qt."""


class UiThreadDispatcherClosed(UiThreadDispatchError):
    """Raised when dispatch is requested after shutdown begins."""


class UiThreadDispatchTimeout(TimeoutError):
    """Raised when a queued UI-thread operation does not complete in time."""


@dataclass(slots=True)
class UiThreadCall(Generic[ResultT]):
    """One callable scheduled onto the Qt UI thread."""

    callback: Callable[[], ResultT]
    done: Event = field(default_factory=Event)
    result: ResultT | None = None
    error: BaseException | None = None

    def run(self) -> None:
        try:
            self.result = self.callback()
        except BaseException as exc:
            self.error = exc
        finally:
            self.done.set()


class UiThreadCallProxy(QObject):
    """Qt-affine receiver that executes queued bridge calls."""

    call_requested = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.call_requested.connect(
            self._execute,
            type=Qt.ConnectionType.QueuedConnection,
        )

    def _execute(self, call: UiThreadCall) -> None:
        call.run()


class UiThreadDispatcher:
    """Dispatch bridge work to the Qt UI thread, or run inline on that thread."""

    def __init__(self) -> None:
        self._closed = False
        self._proxy: UiThreadCallProxy | None = None
        app = QCoreApplication.instance()
        if app is not None:
            self._proxy = UiThreadCallProxy()
            self._proxy.moveToThread(app.thread())

    def close(self) -> None:
        self._closed = True

    def call(
        self,
        callback: Callable[[], ResultT],
        *,
        timeout_ms: int = 5000,
    ) -> ResultT:
        if self._closed:
            raise UiThreadDispatcherClosed("UI bridge dispatcher is shutting down.")

        if self._is_ui_thread():
            return callback()

        if self._proxy is None:
            raise UiThreadDispatchError("No Qt application is available for UI dispatch.")

        call = UiThreadCall(callback)
        self._proxy.call_requested.emit(call)
        if not call.done.wait(timeout_ms / 1000):
            raise UiThreadDispatchTimeout("Timed out waiting for UI thread dispatch.")
        if call.error is not None:
            raise call.error
        return call.result

    def post(self, callback: Callable[[], None]) -> None:
        """Queue bridge work on the Qt UI thread without waiting for completion."""
        if self._closed:
            raise UiThreadDispatcherClosed("UI bridge dispatcher is shutting down.")

        if self._proxy is None:
            if self._is_ui_thread():
                callback()
                return
            raise UiThreadDispatchError("No Qt application is available for UI dispatch.")

        self._proxy.call_requested.emit(UiThreadCall(callback))

    @staticmethod
    def _is_ui_thread() -> bool:
        app = QCoreApplication.instance()
        if app is None:
            return True
        return QThread.currentThread() == app.thread()
