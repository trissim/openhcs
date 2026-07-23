"""Shared window snapshot request semantics without Qt dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WindowSnapshotCaptureScope(str, Enum):
    """Supported screenshot capture scopes."""

    WIDGET = "widget"
    WINDOW = "window"
    NATIVE = "native"


@dataclass(frozen=True, kw_only=True)
class WindowSnapshotCaptureSpec:
    """Agent/runtime boundary for saving one window screenshot."""

    output_dir_path: str
    capture_scope: WindowSnapshotCaptureScope = WindowSnapshotCaptureScope.WIDGET

    def same_capture_contract(self, other: "WindowSnapshotCaptureSpec") -> bool:
        """Return whether two snapshot carriers request the same capture."""
        return (
            self.output_dir_path == other.output_dir_path
            and self.capture_scope is other.capture_scope
        )
