"""
Plate mosaic generation for assembling well composites into plate-level images.

This module provides functions to create:
- Plate-format mosaics (standard layout)
- Arbitrary mosaics (custom well groupings)

Invariants:
- Mosaic generation is OPTIONAL (explicit opt-in)
- Supports both plate-format and arbitrary groupings
- Layout is injectable for flexibility
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .constants import MosaicLayout
from .labeling import FigureLabeler, get_labeler


@dataclass(frozen=True)
class ArbitraryMosaicSpec:
    """Specification for arbitrarily-defined mosaic groups."""

    name: str
    well_ids: Tuple[str, ...]
    layout: MosaicLayout = None


def detect_plate_dimensions(well_ids: Tuple[str, ...]) -> Tuple[int, int]:
    """
    Auto-detect plate dimensions from well IDs.

    Args:
        well_ids: Tuple of well IDs like "r01c06", "R01C06", "A01", etc.

    Returns:
        Tuple of (rows, cols)
    """
    max_row = 0
    max_col = 0

    for well_id in well_ids:
        well_lower = well_id.lower()

        if "r" in well_lower and "c" in well_lower:
            parts = well_lower.split("c")
            row_part = parts[0].replace("r", "")
            col_part = parts[1]
            try:
                row = int(row_part)
                col = int(col_part)
                max_row = max(max_row, row)
                max_col = max(max_col, col)
            except ValueError:
                continue
        else:
            import re

            match = re.match(r"([A-Za-z]+)(\d+)", well_id)
            if match:
                row_letter = match.group(1).upper()
                col_num = int(match.group(2))
                row = ord(row_letter) - ord("A") + 1
                max_row = max(max_row, row)
                max_col = max(max_col, col_num)

    return (max_row, max_col)


def well_id_to_position(well_id: str, cols: int) -> Tuple[int, int]:
    """
    Convert well ID to (row, col) position (0-indexed).

    Handles formats:
    - "r01c06" -> (0, 5)
    - "R01C06" -> (0, 5)
    - "A01" -> (0, 0)
    - "B12" -> (1, 11)
    """
    well_lower = well_id.lower()

    if "r" in well_lower and "c" in well_lower:
        parts = well_lower.split("c")
        row_part = parts[0].replace("r", "")
        col_part = parts[1]
        row = int(row_part) - 1
        col = int(col_part) - 1
        return (row, col)
    else:
        import re

        match = re.match(r"([A-Za-z]+)(\d+)", well_id)
        if match:
            row_letter = match.group(1).upper()
            col_num = int(match.group(2))
            row = ord(row_letter) - ord("A")
            col = col_num - 1
            return (row, col)

    raise ValueError(f"Cannot parse well ID: {well_id}")


def create_plate_mosaic(
    composite_paths: Dict[str, Path],
    layout: MosaicLayout = None,
    projection_type: str = "xy",
    labeler: FigureLabeler = None,
):
    """
    Create mosaic from all well composites, following plate format.

    Args:
        composite_paths: {well_id: path_to_composite_image}
        layout: Mosaic layout specification
        projection_type: Which projection to use for mosaic ("xy", "xz", "yz")
        labeler: Labeling configuration

    Returns:
        PIL Image or matplotlib Figure
    """
    import matplotlib.pyplot as plt

    layout = layout or MosaicLayout()
    labeler = labeler or get_labeler("standard")

    if not composite_paths:
        raise ValueError("No composite paths provided")

    rows, cols = layout.rows, layout.cols
    if rows is None or cols is None:
        rows, cols = detect_plate_dimensions(tuple(composite_paths.keys()))

    sample_img = Image.open(list(composite_paths.values())[0])
    img_width, img_height = sample_img.size
    sample_img.close()

    total_width = cols * img_width + (cols - 1) * layout.well_padding
    total_height = rows * img_height + (rows - 1) * layout.well_padding

    mosaic = Image.new("RGB", (total_width, total_height), layout.background_color)

    well_positions = {}
    for well_id, path in composite_paths.items():
        try:
            row, col = well_id_to_position(well_id, cols)
            if 0 <= row < rows and 0 <= col < cols:
                well_positions[well_id] = (row, col)

                img = Image.open(path).convert("RGB")

                x = col * (img_width + layout.well_padding)
                y = row * (img_height + layout.well_padding)

                mosaic.paste(img, (x, y))
                img.close()
        except (ValueError, IOError) as e:
            continue

    fig = plt.figure(
        figsize=(total_width / 100, total_height / 100),
        facecolor=layout.background_color,
    )
    ax = fig.add_subplot(111)
    ax.imshow(mosaic)
    ax.axis("off")

    if layout.show_row_labels or layout.show_col_labels:
        row_labels = [chr(ord("A") + i) for i in range(rows)]
        col_labels = [f"{i + 1:02d}" for i in range(cols)]
        labeler.add_mosaic_axis_labels(
            fig, tuple(row_labels), tuple(col_labels), (rows, cols)
        )

    return fig


def create_arbitrary_mosaic(
    composite_paths: Dict[str, Path],
    spec: ArbitraryMosaicSpec,
    projection_type: str = "xy",
    labeler: FigureLabeler = None,
):
    """
    Create mosaic from arbitrary well groupings.

    Args:
        composite_paths: {well_id: path_to_composite_image}
        spec: Arbitrary mosaic specification
        projection_type: Which projection to use
        labeler: Labeling configuration

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    layout = spec.layout or MosaicLayout()
    labeler = labeler or get_labeler("standard")

    wells_in_mosaic = [w for w in spec.well_ids if w in composite_paths]

    if not wells_in_mosaic:
        raise ValueError(f"No valid wells found for mosaic '{spec.name}'")

    n_wells = len(wells_in_mosaic)
    cols = int(np.ceil(np.sqrt(n_wells)))
    rows = int(np.ceil(n_wells / cols))

    sample_img = Image.open(composite_paths[wells_in_mosaic[0]])
    img_width, img_height = sample_img.size
    sample_img.close()

    total_width = cols * img_width + (cols - 1) * layout.well_padding
    total_height = rows * img_height + (rows - 1) * layout.well_padding

    mosaic = Image.new("RGB", (total_width, total_height), layout.background_color)

    for i, well_id in enumerate(wells_in_mosaic):
        row = i // cols
        col = i % cols

        img = Image.open(composite_paths[well_id]).convert("RGB")

        x = col * (img_width + layout.well_padding)
        y = row * (img_height + layout.well_padding)

        mosaic.paste(img, (x, y))
        img.close()

    fig = plt.figure(
        figsize=(total_width / 100, total_height / 100),
        facecolor=layout.background_color,
    )
    ax = fig.add_subplot(111)
    ax.imshow(mosaic)
    ax.set_title(spec.name, color=layout.label_color, fontsize=layout.label_fontsize)
    ax.axis("off")

    return fig


def save_mosaic(figure, output_path: Path, dpi: int = 150) -> Path:
    """
    Save mosaic to disk.

    Args:
        figure: matplotlib Figure object
        output_path: Path to save to
        dpi: Resolution for saving

    Returns:
        Path to saved file
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        str(output_path),
        dpi=dpi,
        facecolor=figure.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path
