from PyQt6.QtWidgets import QApplication


def test_snapshot_browser_window_constructs_table_browser_with_selection_mode(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))

    from openhcs.pyqt_gui.windows.snapshot_browser_window import (
        SnapshotBrowserWindow,
    )
    from pyqt_reactive.widgets.shared.abstract_table_browser import (
        TableSelectionMode,
    )

    app = QApplication.instance() or QApplication([])
    window = SnapshotBrowserWindow()

    try:
        assert window.browser._selection_mode is TableSelectionMode.SINGLE
        assert window.browser.window() is window
    finally:
        window.close()
        app.processEvents()
