"""Shared window snapshot request semantics without Qt dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class WindowSnapshotCaptureScope(str, Enum):
    """Supported screenshot capture scopes."""

    WIDGET = "widget"
    WINDOW = "window"
    NATIVE = "native"


class WindowSnapshotPayloadField(str, Enum):
    """Wire fields for a serialized window snapshot request."""

    OUTPUT_DIR_PATH = "output_dir_path"
    CAPTURE_SCOPE = "capture_scope"


@dataclass(frozen=True, slots=True)
class WindowSnapshotWirePayload:
    """Validated wire payload for a window snapshot request."""

    values: Mapping[str, str]

    def required_str(self, field: WindowSnapshotPayloadField) -> str:
        if field.value not in self.values:
            raise ValueError(
                f"Window snapshot payload missing required field {field.value!r}."
            )
        value = self.values[field.value]
        if not isinstance(value, str):
            raise TypeError(
                f"Window snapshot payload field {field.value!r} must be a string."
            )
        return value

    def as_dict(self) -> dict[str, str]:
        return dict(self.values)


@dataclass(frozen=True, kw_only=True)
class WindowSnapshotCaptureSpec:
    """Agent/runtime boundary for saving one window screenshot."""

    output_dir_path: str
    capture_scope: WindowSnapshotCaptureScope = WindowSnapshotCaptureScope.WIDGET

    @classmethod
    def from_wire_payload(
        cls,
        payload: WindowSnapshotWirePayload,
    ) -> "WindowSnapshotCaptureSpec":
        output_dir_path = payload.required_str(
            WindowSnapshotPayloadField.OUTPUT_DIR_PATH,
        )
        capture_scope_value = payload.required_str(
            WindowSnapshotPayloadField.CAPTURE_SCOPE,
        )
        return cls(
            output_dir_path=output_dir_path,
            capture_scope=WindowSnapshotCaptureScope(capture_scope_value),
        )

    def to_wire_payload(self) -> WindowSnapshotWirePayload:
        return WindowSnapshotWirePayload(
            values={
                WindowSnapshotPayloadField.OUTPUT_DIR_PATH.value: self.output_dir_path,
                WindowSnapshotPayloadField.CAPTURE_SCOPE.value: self.capture_scope.value,
            }
        )

    def same_capture_contract(self, other: "WindowSnapshotCaptureSpec") -> bool:
        """Return whether two snapshot carriers request the same capture."""
        return (
            self.output_dir_path == other.output_dir_path
            and self.capture_scope is other.capture_scope
        )
