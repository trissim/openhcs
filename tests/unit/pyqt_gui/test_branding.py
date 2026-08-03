from __future__ import annotations

from PyQt6.QtCore import QSize

from openhcs.pyqt_gui.branding import (
    openhcs_application_icon,
    openhcs_brand_pixmap,
)
from openhcs.resources.brand import BrandAsset


def test_packaged_application_icon_decodes_for_qt(qapp) -> None:
    icon = openhcs_application_icon()

    assert not icon.isNull()
    assert icon.pixmap(128, 128).size().width() == 128


def test_packaged_brand_pixmap_is_available_to_startup_surfaces(qapp) -> None:
    pixmap = openhcs_brand_pixmap(BrandAsset.MARK, QSize(360, 128))

    assert not pixmap.isNull()
    assert pixmap.size() == QSize(360, 128)
    image = pixmap.toImage()
    visible_pixels = [
        (x, y)
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() > 0
    ]
    visible_x = [x for x, _ in visible_pixels]
    visible_y = [y for _, y in visible_pixels]
    visible_width = max(visible_x) - min(visible_x) + 1
    visible_height = max(visible_y) - min(visible_y) + 1
    assert visible_width < 220  # The 320:224 source was not stretched to 360:128.
    assert abs(visible_width / visible_height - 1.51) < 0.03
    assert 80 <= min(visible_x) <= 100
    assert 80 <= image.width() - max(visible_x) - 1 <= 100
    assert min(visible_y) <= 10
    assert image.height() - max(visible_y) - 1 <= 10
