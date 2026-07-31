from __future__ import annotations

import re
import struct
from xml.etree import ElementTree

from openhcs.resources.brand import (
    BRAND_PRIMARY_COLOR,
    BrandAsset,
    brand_asset_bytes,
    brand_asset_path,
)


def test_primary_brand_color_matches_official_mark():
    assert BRAND_PRIMARY_COLOR == "#1D9E75"


def test_brand_assets_are_one_complete_packaged_family() -> None:
    assert {asset.name for asset in BrandAsset} == {
        "MARK",
        "MARK_MONO",
        "LOCKUP_HORIZONTAL",
        "LOCKUP_STACKED",
        "ICON_SQUARE",
        "FAVICON",
        "ICON_RASTER",
        "WINDOWS_ICON",
        "MACOS_ICON",
    }
    for asset in BrandAsset:
        path = brand_asset_path(asset)
        assert path.is_file()
        assert path.read_bytes() == brand_asset_bytes(asset)


def test_official_logo_family_preserves_declared_geometry_and_colors() -> None:
    root = ElementTree.fromstring(brand_asset_bytes(BrandAsset.MARK))
    namespace = {"svg": "http://www.w3.org/2000/svg"}

    assert root.attrib["viewBox"] == "0 0 90 32"
    elements = list(root)
    assert len(elements) == 3
    assert elements[0].attrib["stroke"] == "#1D9E75"
    assert elements[1].attrib["stroke"] == "#5DCAA5"
    assert elements[2].attrib["fill"] == "#1D9E75"

    square = ElementTree.fromstring(brand_asset_bytes(BrandAsset.ICON_SQUARE))
    assert square.attrib["viewBox"] == "0 0 512 512"
    background = square.find("svg:rect", namespace)
    assert background is not None
    assert background.attrib["fill"] == "#0A0D16"

    for asset in (
        BrandAsset.MARK,
        BrandAsset.LOCKUP_HORIZONTAL,
        BrandAsset.LOCKUP_STACKED,
        BrandAsset.ICON_SQUARE,
        BrandAsset.FAVICON,
    ):
        assert b"#dda98b" not in brand_asset_bytes(asset).lower()


def test_stacked_wordmark_and_symbol_share_flush_width() -> None:
    root = ElementTree.fromstring(brand_asset_bytes(BrandAsset.LOCKUP_STACKED))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    text = root.find("svg:text", namespace)
    mark = root.find("svg:g", namespace)

    assert text is not None
    assert mark is not None
    assert text.attrib["x"] == "7"
    assert text.attrib["textLength"] == "140"
    assert text.attrib["lengthAdjust"] == "spacingAndGlyphs"

    match = re.fullmatch(
        r"translate\((?P<x>[0-9.]+) (?P<y>[0-9.]+)\) "
        r"scale\((?P<scale>[0-9.]+)\)",
        mark.attrib["transform"],
    )
    assert match is not None
    assert float(match.group("x")) == float(text.attrib["x"])
    assert abs(90 * float(match.group("scale")) - 140) < 0.01


def test_platform_brand_encodings_have_native_container_headers() -> None:
    png = brand_asset_bytes(BrandAsset.ICON_RASTER)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", png[16:24]) == (1024, 1024)

    windows_icon = brand_asset_bytes(BrandAsset.WINDOWS_ICON)
    reserved, image_type, image_count = struct.unpack("<HHH", windows_icon[:6])
    assert (reserved, image_type, image_count) == (0, 1, 7)

    macos_icon = brand_asset_bytes(BrandAsset.MACOS_ICON)
    assert macos_icon[:4] == b"icns"
    assert struct.unpack(">I", macos_icon[4:8])[0] == len(macos_icon)
