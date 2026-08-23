"""Plate viewer composition lifecycle regressions."""

from pathlib import Path
from types import SimpleNamespace

from openhcs.pyqt_gui.windows.plate_viewer_window import PlateViewerWindow


def test_close_releases_embedded_image_browser(qapp, monkeypatch) -> None:
    cleanup_calls = 0

    class ImageBrowser:
        def cleanup(self) -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1

    monkeypatch.setattr(PlateViewerWindow, "_setup_ui", lambda _window: None)
    monkeypatch.setattr(PlateViewerWindow, "init_scope_border", lambda _window: None)
    window = PlateViewerWindow(
        orchestrator=SimpleNamespace(plate_path=Path("/tmp/plate-viewer-lifecycle")),
    )
    window.image_browser = ImageBrowser()

    window.show()
    window.close()
    qapp.processEvents()

    assert cleanup_calls == 1
