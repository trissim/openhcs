"""About-window identity and main-menu routing contracts."""

from __future__ import annotations

from types import SimpleNamespace

from PyQt6.QtWidgets import QLabel

from openhcs import __version__ as OPENHCS_VERSION
from openhcs.pyqt_gui.main import OpenHCSMainWindow
from openhcs.pyqt_gui.services.main_window_workflows import build_main_window_specs
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId
from openhcs.pyqt_gui.windows.about_window import AboutOpenHCSWindow


def test_about_window_projects_package_identity_and_runtime(qapp) -> None:
    window = AboutOpenHCSWindow()

    try:
        assert window.windowTitle() == "About OpenHCS"
        assert window.findChild(QLabel, "about_openhcs_logo").pixmap().isNull() is False
        assert window.findChild(QLabel, "about_openhcs_title").text() == "OpenHCS"
        assert (
            window.findChild(QLabel, "about_openhcs_version").text()
            == f"Version {OPENHCS_VERSION}"
        )
        assert "Open High-Content Screening" in window.findChild(
            QLabel,
            "about_openhcs_description",
        ).text()
        assert "Python " in window.findChild(
            QLabel,
            "about_openhcs_runtime",
        ).text()
    finally:
        window.close()


def test_about_window_is_registered_and_menu_route_uses_canonical_id() -> None:
    spec = build_main_window_specs()[OpenHCSUiWindowId.about]
    calls: list[tuple[str, bool]] = []
    main_window = SimpleNamespace(
        show_window=lambda window_id, hide_if_startup=True: calls.append(
            (window_id, hide_if_startup)
        )
    )

    OpenHCSMainWindow.show_about(main_window)

    assert spec.window_class is AboutOpenHCSWindow
    assert spec.title == "About OpenHCS"
    assert spec.initialize_on_startup is False
    assert calls == [(OpenHCSUiWindowId.about, False)]
