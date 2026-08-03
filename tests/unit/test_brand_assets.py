from __future__ import annotations

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
        "SOURCE",
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


def _element_with_id(root: ElementTree.Element, element_id: str):
    return next(
        (element for element in root.iter() if element.attrib.get("id") == element_id),
        None,
    )


def _path_geometry(root: ElementTree.Element) -> dict[str, str]:
    return {
        element.attrib["id"]: element.attrib["d"]
        for element in root.iter()
        if element.tag.endswith("path")
        and "id" in element.attrib
        and "d" in element.attrib
    }


def test_official_logo_family_preserves_source_geometry_and_colors() -> None:
    source = ElementTree.fromstring(brand_asset_bytes(BrandAsset.SOURCE))
    mark = ElementTree.fromstring(brand_asset_bytes(BrandAsset.MARK))
    namespace = {"svg": "http://www.w3.org/2000/svg"}

    assert source.attrib["viewBox"] == "0 0 320 224"
    assert mark.attrib["viewBox"] == source.attrib["viewBox"]
    assert _path_geometry(mark) == _path_geometry(source)
    assert set(_path_geometry(source)) == {
        "slice-plane-near",
        "slice-plane-far",
        "slice-plane-lower",
        "cube-outline",
        "cube-face-edges",
        "processing-chevron",
    }

    source_planes = _element_with_id(source, "array-slice-planes")
    source_cube = _element_with_id(source, "cube-wireframe")
    mark_planes = _element_with_id(mark, "array-slice-planes")
    mark_cube = _element_with_id(mark, "cube-wireframe")
    mark_chevron = _element_with_id(mark, "processing-chevron")
    assert source_planes is not None
    assert source_cube is not None
    assert source_planes.attrib["stroke"] == "#a8a8a8"
    assert source_planes.attrib["opacity"] == "0.65"
    assert source_cube.attrib["stroke"] == "#000000"
    assert mark_planes is not None
    assert mark_cube is not None
    assert mark_chevron is not None
    assert mark_planes.attrib["stroke"] == "#66CCFF"
    assert mark_planes.attrib["opacity"] == source_planes.attrib["opacity"]
    assert mark_cube.attrib["stroke"] == BRAND_PRIMARY_COLOR
    assert mark_chevron.attrib["fill"] == BRAND_PRIMARY_COLOR

    square = ElementTree.fromstring(brand_asset_bytes(BrandAsset.ICON_SQUARE))
    assert square.attrib["viewBox"] == "0 0 512 512"
    background = square.find("svg:rect", namespace)
    assert background is not None
    assert background.attrib["fill"] == "#0A0D16"

    for asset in (
        BrandAsset.SOURCE,
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
    assert text.attrib["x"] == "10"
    assert text.attrib["textLength"] == "160"
    assert text.attrib["lengthAdjust"] == "spacingAndGlyphs"

    assert mark.attrib["transform"].startswith("translate(10 58) scale(")
    scale = float(mark.attrib["transform"].removesuffix(")").split("scale(")[1])
    assert abs(320 * scale - 160) < 0.01


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
