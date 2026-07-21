"""Nominal registration of code documents exposed by embedded main-window widgets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
from pyqt_reactive.services.window_code_document import WindowCodeDocumentDriver
from pyqt_reactive.services.window_manager import WindowManager
from PyQt6.QtWidgets import QWidget


class EmbeddedCodeDocumentRegistrationABC(ABC, metaclass=AutoRegisterMeta):
    """Declaration-owned WindowManager registration for one embedded widget."""

    __registry_key__ = "scope_id"
    __skip_if_no_key__ = True

    scope_id: ClassVar[str | None] = None

    @classmethod
    @abstractmethod
    def window_for_main_window(cls, main_window) -> QWidget:
        """Return the embedded widget owned by the main window."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def code_document_driver_for_window(
        cls,
        window: QWidget,
    ) -> WindowCodeDocumentDriver:
        """Return the code-document driver owned by the embedded widget."""
        raise NotImplementedError

    @classmethod
    def register_for_main_window(cls, main_window) -> None:
        if cls.scope_id is None:
            raise ValueError(f"{cls.__name__} does not declare a scope id.")
        window = cls.window_for_main_window(main_window)
        WindowManager.register(
            cls.scope_id,
            window,
            code_document_driver=cls.code_document_driver_for_window(window),
        )

    @classmethod
    def register_all_for_main_window(cls, main_window) -> None:
        for declaration_type in cls.__registry__.values():
            declaration_type.register_for_main_window(main_window)
