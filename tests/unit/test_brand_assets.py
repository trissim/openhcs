from __future__ import annotations

import struct
from xml.etree import ElementTree

from openhcs.resources.brand import (
    BrandAsset,
    brand_asset_bytes,
    brand_asset_path,
)


def test_brand_assets_are_one_complete_packaged_family() -> None:
    assert {asset.name for asset in BrandAsset} == {
        "SCALABLE",
        "RASTER",
        "WINDOWS_ICON",
        "MACOS_ICON",
    }
    for asset in BrandAsset:
        path = brand_asset_path(asset)
        assert path.is_file()
        assert path.read_bytes() == brand_asset_bytes(asset)


def test_canonical_brand_svg_preserves_official_geometry_and_colors() -> None:
    root = ElementTree.fromstring(brand_asset_bytes(BrandAsset.SCALABLE))
    namespace = {"svg": "http://www.w3.org/2000/svg"}

    assert root.attrib["viewBox"] == "0 0 420 420"
    background = root.find("svg:rect", namespace)
    mark = root.find("svg:path", namespace)
    assert background is not None
    assert mark is not None
    assert background.attrib == {
        "width": "420",
        "height": "420",
        "fill": "#f0f0f0",
    }
    assert mark.attrib["fill"] == "#dda98b"
    assert mark.attrib["d"].startswith("M35 35h141")


def test_platform_brand_encodings_have_native_container_headers() -> None:
    png = brand_asset_bytes(BrandAsset.RASTER)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", png[16:24]) == (1024, 1024)

    windows_icon = brand_asset_bytes(BrandAsset.WINDOWS_ICON)
    reserved, image_type, image_count = struct.unpack("<HHH", windows_icon[:6])
    assert (reserved, image_type, image_count) == (0, 1, 7)

    macos_icon = brand_asset_bytes(BrandAsset.MACOS_ICON)
    assert macos_icon[:4] == b"icns"
    assert struct.unpack(">I", macos_icon[4:8])[0] == len(macos_icon)
