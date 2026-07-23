"""PyQt adapter for the canonical FunctionStep code document."""

from __future__ import annotations

from collections.abc import Callable

from pyqt_reactive.services.window_code_document import (
    PYTHON_MIME_TYPE,
    WindowCodeDocument,
    WindowCodeDocumentDriver,
)

from openhcs.core.function_step_document import FunctionStepDocumentAuthority
from openhcs.core.steps.function_step import FunctionStep
from openhcs.ui.shared.code_editor_form_updater import CodeEditorFormUpdater


class FunctionStepCodeDocumentDriver(WindowCodeDocumentDriver):
    """Project a live FunctionStep through its semantic document authority."""

    def __init__(
        self,
        *,
        title: str,
        current_step: Callable[[], FunctionStep],
        apply_step: Callable[[FunctionStep], None],
        before_read: Callable[[], None] | None = None,
    ) -> None:
        self._title = title
        self._current_step = current_step
        self._apply_step = apply_step
        self._before_read = before_read

    def read_document(self, clean: bool = True) -> WindowCodeDocument:
        if self._before_read is not None:
            self._before_read()
        document = FunctionStepDocumentAuthority.from_value(self._current_step())
        return WindowCodeDocument(
            title=self._title,
            source=FunctionStepDocumentAuthority.render(
                document,
                clean_mode=clean,
            ),
            mime_type=PYTHON_MIME_TYPE,
        )

    def validate_source(self, source: str) -> None:
        self._step_from_source(source)

    def apply_source(self, source: str) -> None:
        self._apply_step(self._step_from_source(source))

    @staticmethod
    def _step_from_source(source: str) -> FunctionStep:
        with CodeEditorFormUpdater.patch_lazy_constructors():
            return FunctionStepDocumentAuthority.from_source(source).step
