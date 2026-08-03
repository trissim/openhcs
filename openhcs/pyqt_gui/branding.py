"""Qt projection of the package-owned OpenHCS brand mark."""

from __future__ import annotations

from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from openhcs.resources.brand import BrandAsset, brand_asset_bytes


def openhcs_application_icon() -> QIcon:
    """Build the OpenHCS application icon from its packaged raster asset."""

    return QIcon(openhcs_brand_pixmap())


def openhcs_brand_pixmap(
    asset: BrandAsset = BrandAsset.ICON_RASTER,
    size: QSize | None = None,
) -> QPixmap:
    """Decode one package-owned brand asset for a Qt surface."""

    encoded = brand_asset_bytes(asset)
    if size is not None and asset.value.endswith(".svg"):
        renderer = QSvgRenderer(encoded)
        if not renderer.isValid():
            raise RuntimeError(
                f"Packaged OpenHCS brand asset {asset.name} could not be decoded."
            )
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        rendered_size = renderer.defaultSize()
        rendered_size.scale(size, Qt.AspectRatioMode.KeepAspectRatio)
        render_bounds = QRectF(
            (size.width() - rendered_size.width()) / 2,
            (size.height() - rendered_size.height()) / 2,
            rendered_size.width(),
            rendered_size.height(),
        )
        painter = QPainter(pixmap)
        renderer.render(painter, render_bounds)
        painter.end()
        return pixmap

    pixmap = QPixmap()
    if not pixmap.loadFromData(encoded):
        raise RuntimeError(
            f"Packaged OpenHCS brand asset {asset.name} could not be decoded."
        )
    if size is not None:
        pixmap = pixmap.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return pixmap
