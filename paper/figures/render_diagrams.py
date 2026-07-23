"""Render paper diagram DOT sources to SVG, PNG, and a contact sheet."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parent
DIAGRAM_DIR = ROOT / "diagrams"
RENDER_DIR = ROOT / "rendered"
CONTACT_SHEET = RENDER_DIR / "figure_contact_sheet.png"
THUMB_WIDTH = 900
PADDING = 36
LABEL_HEIGHT = 42
CONTACT_COLUMNS = 2


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def render_diagram(dot_path: Path) -> Path:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    stem = dot_path.stem
    svg_path = RENDER_DIR / f"{stem}.svg"
    png_path = RENDER_DIR / f"{stem}.png"
    with svg_path.open("w") as handle:
        subprocess.run(["dot", "-Tsvg", str(dot_path)], check=True, stdout=handle)
    with png_path.open("wb") as handle:
        subprocess.run(
            ["rsvg-convert", "-b", "white", "-f", "png", "-d", "160", "-p", "160", str(svg_path)],
            check=True,
            stdout=handle,
        )
    return png_path


def build_contact_sheet(png_paths: list[Path]) -> None:
    thumbs: list[Image.Image] = []
    for path in png_paths:
        image = Image.open(path).convert("RGB")
        scale = THUMB_WIDTH / image.width
        thumb_height = int(image.height * scale)
        thumb = image.resize((THUMB_WIDTH, thumb_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (THUMB_WIDTH, thumb_height + LABEL_HEIGHT), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), path.stem, fill=(30, 30, 30))
        canvas.paste(thumb, (0, LABEL_HEIGHT))
        thumbs.append(ImageOps.expand(canvas, border=1, fill=(220, 220, 220)))

    rows = (len(thumbs) + CONTACT_COLUMNS - 1) // CONTACT_COLUMNS
    row_heights = [
        max(
            thumbs[index].height
            for index in range(
                row * CONTACT_COLUMNS,
                min((row + 1) * CONTACT_COLUMNS, len(thumbs)),
            )
        )
        for row in range(rows)
    ]
    width = CONTACT_COLUMNS * (THUMB_WIDTH + 2) + (CONTACT_COLUMNS + 1) * PADDING
    height = sum(row_heights) + (rows + 1) * PADDING
    sheet = Image.new("RGB", (width, height), (246, 247, 248))

    y = PADDING
    for row, row_height in enumerate(row_heights):
        x = PADDING
        for column in range(CONTACT_COLUMNS):
            index = row * CONTACT_COLUMNS + column
            if index >= len(thumbs):
                break
            sheet.paste(thumbs[index], (x, y))
            x += THUMB_WIDTH + 2 + PADDING
        y += row_height + PADDING
    sheet.save(CONTACT_SHEET)


def main() -> None:
    png_paths = [render_diagram(path) for path in sorted(DIAGRAM_DIR.glob("fig*.dot"))]
    build_contact_sheet(png_paths)
    print(CONTACT_SHEET)


if __name__ == "__main__":
    main()
