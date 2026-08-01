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
    assert BRAND_PRIMARY_COLOR == "#00AAFF"


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
    outline = root.findall("svg:rect", namespace)[0]
    arrow = root.find("svg:path", namespace)
    filled = root.findall("svg:rect", namespace)[1]
    assert outline.attrib["stroke"] == "#00AAFF"
    assert outline.attrib["stroke-width"] == "3"
    assert outline.attrib["rx"] == "1.5"
    assert arrow is not None
    assert arrow.attrib["stroke"] == "#66CCFF"
    assert arrow.attrib["stroke-width"] == "3"
    assert arrow.attrib["stroke-linecap"] == "square"
    assert filled.attrib["fill"] == "#00AAFF"
    assert filled.attrib["mask"] == "url(#openhcs-inverted-cells)"

    visible_cells = root.findall("svg:g/svg:circle", namespace)
    removed_cells = root.findall("svg:defs/svg:mask/svg:g/svg:circle", namespace)
    assert len(visible_cells) == 4
    assert len(removed_cells) == 4
    visible_geometry = [
        (
            float(cell.attrib["cx"]),
            float(cell.attrib["cy"]),
            float(cell.attrib["r"]),
        )
        for cell in visible_cells
    ]
    removed_geometry = [
        (
            float(cell.attrib["cx"]) - 44,
            float(cell.attrib["cy"]),
            float(cell.attrib["r"]),
        )
        for cell in removed_cells
    ]
    assert removed_geometry == visible_geometry
    assert float(outline.attrib["x"]) + float(outline.attrib["width"]) == 35
    assert arrow.attrib["d"] == "M40 9 L48 16 L40 23"
    assert float(filled.attrib["x"]) == 53
    assert len({y for _, y, _ in visible_geometry}) == 4
    assert len({radius for _, _, radius in visible_geometry}) > 1

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
