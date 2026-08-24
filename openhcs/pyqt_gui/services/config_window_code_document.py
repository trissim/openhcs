"""Configuration code-document driver for managed PyQt windows."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import ClassVar, Generic, TypeVar

from pyqt_reactive.services.window_code_document import (
    PYTHON_MIME_TYPE,
    WindowCodeDocument,
    WindowCodeDocumentDriver,
)

from openhcs.core.config_document import ConfigDocumentAuthority
from openhcs.ui.shared.code_editor_form_updater import CodeEditorFormUpdater

ConfigT = TypeVar("ConfigT")


class ExternalCodeEditorPreference:
    """Environment-backed policy for opening code documents externally."""

    env_var: ClassVar[str] = "OPENHCS_USE_EXTERNAL_EDITOR"
    enabled_values: ClassVar[frozenset[str]] = frozenset(("1", "true", "yes"))

    @classmethod
    def use_external_editor(cls) -> bool:
        if cls.env_var not in os.environ:
            return False
        return os.environ[cls.env_var].lower() in cls.enabled_values


class ConfigCodeDocumentDriver(WindowCodeDocumentDriver, Generic[ConfigT]):
    """Render and apply configuration source through its nominal authority."""

    def __init__(
        self,
        *,
        title: str,
        config_type: type[ConfigT],
        current_config: Callable[[], ConfigT],
        apply_config: Callable[[ConfigT], None],
        before_read: Callable[[], None] | None = None,
        before_apply: Callable[[], None] | None = None,
    ) -> None:
        self._title = title
        self._config_type = config_type
        self._current_config = current_config
        self._apply_config = apply_config
        self._before_read = before_read
        self._before_apply = before_apply

    def read_document(self, clean: bool = True) -> WindowCodeDocument:
        if self._before_read is not None:
            self._before_read()
        source = ConfigDocumentAuthority.render(
            self._current_config(),
            expected_config_type=self._config_type,
            clean_mode=clean,
        )
        return WindowCodeDocument(
            title=self._title,
            source=source,
            mime_type=PYTHON_MIME_TYPE,
            declaration_type=self._config_type,
        )

    def validate_source(self, source: str) -> None:
        self._config_from_source(source)

    def apply_source(self, source: str) -> None:
        if self._before_apply is not None:
            self._before_apply()
        self._apply_config(self._config_from_source(source))

    def _config_from_source(self, source: str) -> ConfigT:
        with CodeEditorFormUpdater.patch_lazy_constructors():
            return ConfigDocumentAuthority.from_source(
                source,
                expected_config_type=self._config_type,
            )


__all__ = ("ConfigCodeDocumentDriver", "ExternalCodeEditorPreference")
