#!/usr/bin/env python3
"""Generate the official OpenHCS SVG family from one editable mark."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
SVG = f"{{{SVG_NAMESPACE}}}"
ElementTree.register_namespace("", SVG_NAMESPACE)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPOSITORY_ROOT / "openhcs" / "resources" / "assets"
SOURCE_PATH = ASSET_ROOT / "openhcs-logo-source.svg"
BRAND_MODULE_PATH = REPOSITORY_ROOT / "openhcs" / "resources" / "brand.py"
SOURCE_VIEW_BOX = "0 0 320 224"

BRAND_BLUE_LIGHT = "#66CCFF"
BRAND_DARK = "#0A0D16"
BRAND_LIGHT = "#F1EFE8"
BRAND_MID = "#9297A3"


def _declared_primary_color() -> str:
    module = ast.parse(
        BRAND_MODULE_PATH.read_text(encoding="utf-8"),
        filename=str(BRAND_MODULE_PATH),
    )
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "BRAND_PRIMARY_COLOR"
            for target in statement.targets
        ):
            continue
        if isinstance(statement.value, ast.Constant) and isinstance(
            statement.value.value,
            str,
        ):
            return statement.value.value
    raise ValueError(
        f"No literal BRAND_PRIMARY_COLOR assignment found in {BRAND_MODULE_PATH}"
    )


def _element(name: str, **attributes: str) -> ElementTree.Element:
    return ElementTree.Element(f"{SVG}{name}", attributes)


def _document(
    *,
    view_box: str,
    width: str,
    height: str,
    label: str = "OpenHCS",
) -> ElementTree.Element:
    return _element(
        "svg",
        viewBox=view_box,
        width=width,
        height=height,
        role="img",
        **{"aria-label": label},
    )


def _source_mark() -> ElementTree.Element:
    source = ElementTree.parse(SOURCE_PATH).getroot()
    if source.attrib.get("viewBox") != SOURCE_VIEW_BOX:
        raise ValueError(
            f"OpenHCS logo source must use {SOURCE_VIEW_BOX!r}; "
            f"got {source.attrib.get('viewBox')!r}"
        )
    mark = source.find(f"{SVG}g[@id='array-processing-mark']")
    if mark is None:
        raise ValueError("OpenHCS logo source has no array-processing-mark group")
    return mark


def _colored_mark(
    source_mark: ElementTree.Element,
    *,
    foreground: str,
    planes: str,
) -> ElementTree.Element:
    mark = deepcopy(source_mark)
    slice_planes = mark.find(f"{SVG}g[@id='array-slice-planes']")
    cube = mark.find(f"{SVG}g[@id='cube-wireframe']")
    chevron = mark.find(f"{SVG}path[@id='processing-chevron']")
    if slice_planes is None or cube is None or chevron is None:
        raise ValueError("OpenHCS logo source is missing a semantic geometry group")
    slice_planes.set("stroke", planes)
    cube.set("stroke", foreground)
    chevron.set("fill", foreground)
    return mark


def _place_mark(
    parent: ElementTree.Element,
    source_mark: ElementTree.Element,
    *,
    x: float,
    y: float,
    width: float,
    foreground: str,
    planes: str,
) -> None:
    scale = width / 320
    wrapper = ElementTree.SubElement(
        parent,
        f"{SVG}g",
        {"transform": f"translate({x:g} {y:g}) scale({scale:g})"},
    )
    wrapper.append(
        _colored_mark(
            source_mark,
            foreground=foreground,
            planes=planes,
        )
    )


def _add_wordmark(
    parent: ElementTree.Element,
    *,
    x: float,
    y: float,
    font_size: float,
    fill: str,
    text_length: float | None = None,
) -> None:
    attributes = {
        "x": f"{x:g}",
        "y": f"{y:g}",
        "font-family": "Inter, 'Helvetica Neue', Arial, sans-serif",
        "font-size": f"{font_size:g}",
        "font-weight": "500",
        "fill": fill,
    }
    if text_length is not None:
        attributes.update(
            {
                "textLength": f"{text_length:g}",
                "lengthAdjust": "spacingAndGlyphs",
            }
        )
    text = ElementTree.SubElement(parent, f"{SVG}text", attributes)
    text.text = "OpenHCS"


def _write(path: Path, root: ElementTree.Element) -> None:
    ElementTree.indent(root, space="  ")
    xml = ElementTree.tostring(root, encoding="unicode", xml_declaration=False)
    path.write_text(f"{xml}\n", encoding="utf-8")


def generate() -> None:
    """Regenerate every SVG derivative from the canonical geometry."""

    source_mark = _source_mark()
    brand_blue = _declared_primary_color()

    mark = _document(
        view_box=SOURCE_VIEW_BOX,
        width="320",
        height="224",
        label="OpenHCS array-processing mark",
    )
    mark.append(
        _colored_mark(
            source_mark,
            foreground=brand_blue,
            planes=BRAND_BLUE_LIGHT,
        )
    )
    _write(ASSET_ROOT / "openhcs-mark.svg", mark)

    mono = _document(
        view_box=SOURCE_VIEW_BOX,
        width="320",
        height="224",
        label="OpenHCS array-processing mark",
    )
    mono.append(
        _colored_mark(
            source_mark,
            foreground="currentColor",
            planes="currentColor",
        )
    )
    _write(ASSET_ROOT / "openhcs-mark-mono.svg", mono)

    horizontal = _document(
        view_box="0 0 250 56",
        width="250",
        height="56",
    )
    _place_mark(
        horizontal,
        source_mark,
        x=0,
        y=6,
        width=62.857142857,
        foreground=BRAND_LIGHT,
        planes=BRAND_MID,
    )
    _add_wordmark(
        horizontal,
        x=76,
        y=40,
        font_size=36,
        fill=BRAND_LIGHT,
    )
    _write(ASSET_ROOT / "openhcs-lockup-horizontal.svg", horizontal)

    stacked = _document(
        view_box="0 0 180 180",
        width="180",
        height="180",
    )
    _add_wordmark(
        stacked,
        x=10,
        y=35,
        font_size=32,
        fill=BRAND_LIGHT,
        text_length=160,
    )
    _place_mark(
        stacked,
        source_mark,
        x=10,
        y=58,
        width=160,
        foreground=BRAND_LIGHT,
        planes=BRAND_MID,
    )
    _write(ASSET_ROOT / "openhcs-lockup-stacked.svg", stacked)

    icon = _document(
        view_box="0 0 512 512",
        width="512",
        height="512",
    )
    icon.append(
        _element("rect", width="512", height="512", rx="72", fill=BRAND_DARK)
    )
    _place_mark(
        icon,
        source_mark,
        x=40,
        y=104.8,
        width=432,
        foreground=BRAND_LIGHT,
        planes=BRAND_MID,
    )
    _write(ASSET_ROOT / "openhcs-icon-square.svg", icon)

    favicon = _document(
        view_box="0 0 32 32",
        width="32",
        height="32",
    )
    favicon.append(
        _element("rect", width="32", height="32", rx="4", fill=BRAND_DARK)
    )
    _place_mark(
        favicon,
        source_mark,
        x=2,
        y=6.2,
        width=28,
        foreground="#FFFFFF",
        planes=BRAND_MID,
    )
    _write(ASSET_ROOT / "openhcs-favicon.svg", favicon)


if __name__ == "__main__":
    generate()
