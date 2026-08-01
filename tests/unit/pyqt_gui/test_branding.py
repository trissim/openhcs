from __future__ import annotations

from openhcs.pyqt_gui.branding import (
    openhcs_application_icon,
    openhcs_brand_pixmap,
)


def test_packaged_application_icon_decodes_for_qt(qapp) -> None:
    icon = openhcs_application_icon()

    assert not icon.isNull()
    assert icon.pixmap(128, 128).size().width() == 128


def test_packaged_brand_pixmap_is_available_to_startup_surfaces(qapp) -> None:
    pixmap = openhcs_brand_pixmap()

    assert not pixmap.isNull()
    assert pixmap.width() == pixmap.height()
