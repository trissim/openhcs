"""Qt projection of the package-owned OpenHCS brand mark."""

from __future__ import annotations

from PyQt6.QtGui import QIcon, QPixmap

from openhcs.resources.brand import BrandAsset, brand_asset_bytes


def openhcs_application_icon() -> QIcon:
    """Build the OpenHCS application icon from its packaged raster asset."""

    pixmap = QPixmap()
    if not pixmap.loadFromData(brand_asset_bytes(BrandAsset.ICON_RASTER)):
        raise RuntimeError("Packaged OpenHCS application icon could not be decoded.")
    return QIcon(pixmap)
