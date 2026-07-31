from __future__ import annotations

from openhcs.pyqt_gui.branding import openhcs_application_icon


def test_packaged_application_icon_decodes_for_qt(qapp) -> None:
    icon = openhcs_application_icon()

    assert not icon.isNull()
    assert icon.pixmap(128, 128).size().width() == 128
