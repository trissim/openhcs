from __future__ import annotations

from pathlib import Path


def test_qt_window_snapshot_service_writes_png(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtWidgets import QApplication, QLabel

    from openhcs.runtime.qt_window_snapshot import (
        QtWindowSnapshotRequest,
        QtWindowSnapshotService,
    )
    from openhcs.runtime.window_snapshot import (
        WindowSnapshotCaptureScope,
        WindowSnapshotCaptureSpec,
    )

    app = QApplication.instance() or QApplication([])
    label = QLabel("OpenHCS snapshot")
    label.resize(240, 80)
    label.show()
    app.processEvents()

    snapshot = QtWindowSnapshotService().capture(
        QtWindowSnapshotRequest(
            widget=label,
            capture=WindowSnapshotCaptureSpec(
                output_dir_path=str(tmp_path),
                capture_scope=WindowSnapshotCaptureScope.WIDGET,
            ),
            subject_id="test-window",
            title="Test Window",
        )
    )

    assert snapshot.mime_type == "image/png"
    assert snapshot.width == 240
    assert snapshot.height == 80
    assert snapshot.size_bytes > 0
    assert snapshot.sha256
    assert snapshot.path.endswith(".png")
    assert Path(snapshot.path).is_file()
