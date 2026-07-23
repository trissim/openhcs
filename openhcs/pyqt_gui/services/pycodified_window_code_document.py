"""Pycodified object code-document drivers for managed PyQt windows."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Generic, Protocol, TypeVar, cast

from pycodify import Assignment
from pyqt_reactive.services.window_code_document import (
    PYTHON_MIME_TYPE,
    WindowCodeDocument,
    WindowCodeDocumentDriver,
)

import openhcs.serialization.pycodify_formatters  # noqa: F401
from openhcs.core.config_document import ConfigDocumentAuthority
from openhcs.serialization.source_path_factoring import OpenHCSPythonSourceDocument
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

    def render_source(self, value: ObjectT, *, clean: bool = True) -> str:
        """Render this document's assigned object as Python source."""
        return OpenHCSPythonSourceDocument(
            Assignment(self.assignment_name, value),
            header=self.header,
            clean_mode=clean,
        ).render()

    def object_from_source(self, source: str) -> ObjectT:
        """Execute source and read this generic object's assigned variable."""

        namespace: dict[str, object] = {}
        exec(source, namespace)

        if self.assignment_name not in namespace:
            raise ValueError(
                f"No {self.assignment_name!r} variable found in edited code."
            )

        value = namespace[self.assignment_name]
        if not isinstance(value, self.expected_type):
            raise TypeError(
                "Code document variable "
                f"{self.assignment_name!r} resolved to {type(value).__name__}; "
                f"expected {self.expected_type.__name__}."
            )
        return cast(ObjectT, value)


@dataclass(frozen=True, slots=True)
class PycodifiedConfigDocumentSpec(Generic[ObjectT]):
    """Pycodified config document delegated to the config authority."""

    title: str
    expected_type: type[ObjectT]

    def render_source(self, value: ObjectT, *, clean: bool = True) -> str:
        return ConfigDocumentAuthority.render(
            value,
            expected_config_type=self.expected_type,
            clean_mode=clean,
        )

    def object_from_source(self, source: str) -> ObjectT:
        return ConfigDocumentAuthority.from_source(
            source,
            expected_config_type=self.expected_type,
        )


class PycodifiedDocumentSpec(Protocol[ObjectT]):
    """Structural contract consumed by the generic code-document driver."""

    title: str

    def render_source(self, value: ObjectT, *, clean: bool = True) -> str: ...

    def object_from_source(self, source: str) -> ObjectT: ...


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
        spec: PycodifiedDocumentSpec[ObjectT],
        current_object: Callable[[], ObjectT],
        apply_object: Callable[[ObjectT], None],
        before_read: Callable[[], None] | None = None,
    ) -> None:
        self._spec = spec
        self._current_object = current_object
        self._apply_object = apply_object
        self._before_read = before_read

    def read_document(self, clean: bool = True) -> WindowCodeDocument:
        before_read = self._before_read
        if before_read is not None:
            before_read()
        source = self._spec.render_source(self._current_object(), clean=clean)
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
        with CodeEditorFormUpdater.patch_lazy_constructors():
            return self._spec.object_from_source(source)
