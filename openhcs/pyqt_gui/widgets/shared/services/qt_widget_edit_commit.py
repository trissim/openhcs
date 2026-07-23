"""Qt widget edit-commit hook for action dispatch."""

from __future__ import annotations

from PyQt6.QtCore import QEventLoop, QThread
from PyQt6.QtWidgets import QApplication


class QtWidgetEditCommitError(RuntimeError):
    """Raised when pending widget edits cannot be committed safely."""


class QtWidgetEditCommitter:
    """Commit focused Qt editor state before dispatching a widget action."""

    def commit_focused_widget(self) -> None:
        app = QApplication.instance()
        if app is None:
            raise QtWidgetEditCommitError("No QApplication is available.")
        if QThread.currentThread() != app.thread():
            raise QtWidgetEditCommitError(
                "Qt widget edits must be committed on the UI thread."
            )
        focus_widget = app.focusWidget()
        if focus_widget is not None:
            focus_widget.clearFocus()
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)


def commit_focused_widget_edits() -> None:
    """Commit pending edits from the currently focused Qt widget."""

    QtWidgetEditCommitter().commit_focused_widget()
