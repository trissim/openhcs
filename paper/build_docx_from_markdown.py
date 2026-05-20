#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


WORD_DOCUMENT_XML = "word/document.xml"
WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
TABLE_FONT_SIZE_HALF_POINTS = "20"
TABLE_BORDER_SIZE_EIGHTH_POINTS = "4"
TABLE_BORDER_COLOR = "808080"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=cwd)


def word_tag(tag_name: str) -> str:
    return f"{{{WORD_NAMESPACE}}}{tag_name}"


def style_docx_tables(docx_path: Path) -> None:
    namespace = {"w": WORD_NAMESPACE}
    ET.register_namespace("w", WORD_NAMESPACE)

    with zipfile.ZipFile(docx_path, "r") as source_zip:
        document_xml = source_zip.read(WORD_DOCUMENT_XML)
        root = ET.fromstring(document_xml)

        for table in root.findall(".//w:tbl", namespace):
            properties = table.find("w:tblPr", namespace)
            if properties is None:
                properties = ET.Element(word_tag("tblPr"))
                table.insert(0, properties)

            borders = properties.find("w:tblBorders", namespace)
            if borders is None:
                borders = ET.SubElement(properties, word_tag("tblBorders"))

            for border_name in ("top", "left", "bottom", "right", "insideH", "insideV"):
                border = borders.find(f"w:{border_name}", namespace)
                if border is None:
                    border = ET.SubElement(borders, word_tag(border_name))
                border.set(word_tag("val"), "single")
                border.set(word_tag("sz"), TABLE_BORDER_SIZE_EIGHTH_POINTS)
                border.set(word_tag("space"), "0")
                border.set(word_tag("color"), TABLE_BORDER_COLOR)

        for run_node in root.findall(".//w:tbl//w:r", namespace):
            properties = run_node.find("w:rPr", namespace)
            if properties is None:
                properties = ET.Element(word_tag("rPr"))
                run_node.insert(0, properties)

            for tag_name in ("sz", "szCs"):
                size = properties.find(f"w:{tag_name}", namespace)
                if size is None:
                    size = ET.SubElement(properties, word_tag(tag_name))
                size.set(word_tag("val"), TABLE_FONT_SIZE_HALF_POINTS)

        updated_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        temp_docx = docx_path.with_suffix(".tmp.docx")

        with zipfile.ZipFile(temp_docx, "w") as dest_zip:
            for entry in source_zip.infolist():
                data = (
                    updated_xml
                    if entry.filename == WORD_DOCUMENT_XML
                    else source_zip.read(entry.filename)
                )
                dest_zip.writestr(entry, data)

    temp_docx.replace(docx_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a review DOCX from Markdown.")
    parser.add_argument("source", type=Path, help="Markdown source file")
    parser.add_argument("output", type=Path, help="DOCX output path")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    run(["pandoc", str(source), "--standalone", "-o", str(output)])
    style_docx_tables(output)
    print(f"DOCX written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
