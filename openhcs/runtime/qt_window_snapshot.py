"""Qt window screenshot primitives shared by UI and viewer processes."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QWidget

from openhcs.runtime.window_snapshot import (
    WindowSnapshotCaptureScope,
    WindowSnapshotCaptureSpec,
)


@dataclass(frozen=True, slots=True)
class QtWindowSnapshot:
    """Saved screenshot metadata."""

    uri: str
    path: str
    title: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    sha256: str
    capture: WindowSnapshotCaptureSpec


@dataclass(frozen=True, slots=True)
class QtWindowSnapshotRequest:
    """Typed request for saving one Qt window or widget screenshot."""

    widget: QWidget
    capture: WindowSnapshotCaptureSpec
    subject_id: str
    title: str


QtWindowCaptureCallable = Callable[[QWidget], QPixmap]


@dataclass(frozen=True, slots=True)
class QtWindowCaptureRule:
    """One closed-family Qt screenshot capture rule."""

    scope: WindowSnapshotCaptureScope
    capture: QtWindowCaptureCallable


def _widget_pixmap(widget: QWidget) -> QPixmap:
    return widget.grab()


def _window_pixmap(widget: QWidget) -> QPixmap:
    return widget.window().grab()


def _native_pixmap(widget: QWidget) -> QPixmap:
    window = widget.window()
    window_handle = window.windowHandle()
    if window_handle is None:
        raise RuntimeError("Native Qt screenshot requires a concrete window handle.")
    screen = window_handle.screen()
    if screen is None:
        raise RuntimeError("Native Qt screenshot requires the target window screen.")
    return screen.grabWindow(int(window.winId()))


QT_WINDOW_CAPTURE_RULES: Mapping[WindowSnapshotCaptureScope, QtWindowCaptureRule] = {
    rule.scope: rule
    for rule in (
        QtWindowCaptureRule(WindowSnapshotCaptureScope.WIDGET, _widget_pixmap),
        QtWindowCaptureRule(WindowSnapshotCaptureScope.WINDOW, _window_pixmap),
        QtWindowCaptureRule(WindowSnapshotCaptureScope.NATIVE, _native_pixmap),
    )
}


class QtWindowSnapshotService:
    """Capture Qt widgets/windows to bounded file artifacts."""

    MIME_TYPE = "image/png"
    FILE_EXTENSION = ".png"
    SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")

    def capture(self, request: QtWindowSnapshotRequest) -> QtWindowSnapshot:
        pixmap = self._capture_pixmap(
            request.widget,
            request.capture.capture_scope,
        )
        if pixmap.isNull():
            raise RuntimeError(
                f"Qt screenshot capture returned an empty pixmap for {request.subject_id!r}."
            )

        output_dir = (
            Path(request.capture.output_dir_path).expanduser().resolve(strict=False)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self._filename(request)
        if not pixmap.save(str(output_path), "PNG"):
            raise RuntimeError(f"Failed to save Qt screenshot to {output_path}.")

        image_bytes = output_path.read_bytes()
        digest = hashlib.sha256(image_bytes).hexdigest()
        return QtWindowSnapshot(
            uri=output_path.as_uri(),
            path=str(output_path),
            title=request.title,
            mime_type=self.MIME_TYPE,
            width=pixmap.width(),
            height=pixmap.height(),
            size_bytes=len(image_bytes),
            sha256=digest,
            capture=request.capture,
        )

    def _capture_pixmap(
        self,
        widget: QWidget,
        capture_scope: WindowSnapshotCaptureScope,
    ) -> QPixmap:
        return QT_WINDOW_CAPTURE_RULES[capture_scope].capture(widget)

    def _filename(self, request: QtWindowSnapshotRequest) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        subject = self._safe_filename_token(request.subject_id)
        title = self._safe_filename_token(request.title)
        return f"{timestamp}_{subject}_{title}{self.FILE_EXTENSION}"

    def _safe_filename_token(self, value: str) -> str:
        stripped = value.strip()
        if stripped:
            normalized = stripped
        else:
            normalized = "window"
        token = self.SAFE_FILENAME_PATTERN.sub("_", normalized).strip("._")
        if token:
            return token[:80]
        return "window"
