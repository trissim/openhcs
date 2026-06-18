"""Pycodified object code-document drivers for managed PyQt windows."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar, cast

from pycodify import Assignment, generate_python_source
from pyqt_reactive.services.window_code_document import (
    PYTHON_MIME_TYPE,
    WindowCodeDocument,
    WindowCodeDocumentDriver,
)

import openhcs.serialization.pycodify_formatters  # noqa: F401
from openhcs.ui.shared.code_editor_form_updater import CodeEditorFormUpdater

logger = logging.getLogger(__name__)

ObjectT = TypeVar("ObjectT")


@dataclass(frozen=True, slots=True)
class PycodifiedObjectDocumentSpec(Generic[ObjectT]):
    """Static code-document contract for one pycodified object variable."""

    assignment_name: str
    title: str
    header: str
    expected_type: type[ObjectT]

    def render_source(self, value: ObjectT) -> str:
        """Render this document's assigned object as Python source."""
        return generate_python_source(
            Assignment(self.assignment_name, value),
            self.header,
            True,
        )


class ExternalCodeEditorPreference:
    """Environment-backed policy for opening code documents externally."""

    env_var: ClassVar[str] = "OPENHCS_USE_EXTERNAL_EDITOR"
    enabled_values: ClassVar[frozenset[str]] = frozenset(("1", "true", "yes"))

    @classmethod
    def use_external_editor(cls) -> bool:
        if cls.env_var not in os.environ:
            return False
        return os.environ[cls.env_var].lower() in cls.enabled_values


class PycodifiedObjectCodeDocumentDriver(
    WindowCodeDocumentDriver,
    Generic[ObjectT],
):
    """Render and apply pycodified object code through window-owned callbacks."""

    def __init__(
        self,
        *,
        spec: PycodifiedObjectDocumentSpec[ObjectT],
        current_object: Callable[[], ObjectT],
        apply_object: Callable[[ObjectT], None],
        before_read: Callable[[], None] | None = None,
    ) -> None:
        self._spec = spec
        self._current_object = current_object
        self._apply_object = apply_object
        self._before_read = before_read

    def read_document(self) -> WindowCodeDocument:
        before_read = self._before_read
        if before_read is not None:
            before_read()
        source = self._spec.render_source(self._current_object())
        return WindowCodeDocument(
            title=self._spec.title,
            source=source,
            mime_type=PYTHON_MIME_TYPE,
        )

    def validate_source(self, source: str) -> None:
        self._object_from_source(source)

    def apply_source(self, source: str) -> None:
        self._apply_object(self._object_from_source(source))

    def _object_from_source(self, source: str) -> ObjectT:
        namespace: dict[str, object] = {}
        with CodeEditorFormUpdater.patch_lazy_constructors():
            exec(source, namespace)

        assignment_name = self._spec.assignment_name
        if assignment_name not in namespace:
            raise ValueError(
                f"No {assignment_name!r} variable found in edited code."
            )

        value = namespace[assignment_name]
        if not isinstance(value, self._spec.expected_type):
            raise TypeError(
                "Code document variable "
                f"{assignment_name!r} resolved to {type(value).__name__}; "
                f"expected {self._spec.expected_type.__name__}."
            )
        return cast(ObjectT, value)
