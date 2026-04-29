"""
Multi-channel composite figure generation.

This module provides functions to create publication-ready composite figures
showing XY, XZ, and YZ projections with multi-channel color overlay.

Invariants:
- Composite generation is DEFAULT behavior
- Works with ANY number of channels (1, 2, 3, 4+)
- Color mapping is EXPLICIT via configuration
- Figure creation and saving are separate operations
- Z-gap is configurable for XZ/YZ projections
"""

from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

import numpy as np

from .constants import (
    CompositeLayout,
    ChannelColorMapping,
    DEFAULT_CHANNEL_COLORS,
)
from .labeling import FigureLabeler, get_labeler


def apply_z_gap(projection: np.ndarray, z_gap: float = 1.0) -> np.ndarray:
    """
    Apply spacing between Z-slices in XZ or YZ projections using interpolation.

    Args:
        projection: 2D array (Z, X) or (Z, Y)
        z_gap: Vertical stretch factor (1.0 = no stretch, 2.0 = double height)

    Returns:
        Projection with Z-gap applied via interpolation
    """
    if z_gap <= 1.0:
        return projection

    from scipy.ndimage import zoom

    z_size, xy_size = projection.shape
    new_z_size = int(z_size * z_gap)

    result = zoom(projection, (z_gap, 1.0), order=1)

    return result


def create_projection_composite(
    projections: Dict[str, np.ndarray],
    title: str,
    layout: CompositeLayout = None,
    labeler: FigureLabeler = None,
):
    """
    Create composite figure showing all three projections for a single channel.

    Args:
        projections: Dict {"xy": arr, "xz": arr, "yz": arr}
        title: Figure title
        layout: Layout specification
        labeler: Labeling configuration

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    layout = layout or CompositeLayout()
    labeler = labeler or get_labeler("standard")

    fig = plt.figure(figsize=layout.figsize, facecolor="black")

    axes = {}

    if "xy" in projections:
        ax_xy = plt.subplot2grid(
            (layout.grid_rows, layout.grid_cols),
            layout.xy_position[:2],
            rowspan=layout.xy_position[2],
            colspan=layout.xy_position[3],
        )
        ax_xy.imshow(projections["xy"], cmap="gray", origin="lower")
        ax_xy.set_facecolor("black")
        ax_xy.axis("off")
        axes["xy"] = ax_xy

    if "xz" in projections:
        xz_data = apply_z_gap(projections["xz"], layout.z_gap)
        ax_xz = plt.subplot2grid(
            (layout.grid_rows, layout.grid_cols),
            layout.xz_position[:2],
            rowspan=layout.xz_position[2],
            colspan=layout.xz_position[3],
        )
        ax_xz.imshow(xz_data, cmap="gray", aspect=layout.z_aspect, origin="upper")
        ax_xz.set_facecolor("black")
        ax_xz.axis("off")
        axes["xz"] = ax_xz

    if "yz" in projections:
        yz_data = apply_z_gap(projections["yz"], layout.z_gap)
        ax_yz = plt.subplot2grid(
            (layout.grid_rows, layout.grid_cols),
            layout.yz_position[:2],
            rowspan=layout.yz_position[2],
            colspan=layout.yz_position[3],
        )
        ax_yz.imshow(yz_data, cmap="gray", aspect=layout.z_aspect, origin="upper")
        ax_yz.set_facecolor("black")
        ax_yz.axis("off")
        axes["yz"] = ax_yz

    labeler.add_panel_labels(axes)
    labeler.add_title(fig, title)

    plt.tight_layout()

    return fig


def create_multi_channel_composite(
    all_channel_projections: Dict[str, Dict[str, np.ndarray]],
    title: str,
    layout: CompositeLayout = None,
    channel_colors: Tuple[ChannelColorMapping, ...] = DEFAULT_CHANNEL_COLORS,
    excluded_channels: FrozenSet[str] = frozenset(),
    labeler: FigureLabeler = None,
):
    """
    Create composite by directly concatenating images.

    Layout: XY on left, XZ and YZ stacked vertically on right.
    No subplots - direct image concatenation preserves aspect ratios.

    Args:
        all_channel_projections: {channel_id: {"xy": arr, "xz": arr, "yz": arr}}
        title: Figure title
        layout: Layout specification
        channel_colors: Color mappings for each channel
        excluded_channels: Channel IDs to skip
        labeler: Labeling configuration

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    layout = layout or CompositeLayout()
    labeler = labeler or get_labeler("standard")

    color_map = {cc.channel_id: cc for cc in channel_colors}

    active_channels = [
        cc
        for cc in channel_colors
        if cc.visible
        and cc.channel_id not in excluded_channels
        and cc.channel_id in all_channel_projections
    ]

    # Find global max across ALL projections for consistent brightness
    global_max = 0
    for cc in active_channels:
        for proj_type in ["xy", "xz", "yz"]:
            if proj_type in all_channel_projections.get(cc.channel_id, {}):
                data = all_channel_projections[cc.channel_id][proj_type]
                global_max = max(global_max, data.max())

    if global_max == 0:
        global_max = 1

    z_gap = layout.z_gap if layout.z_gap > 1.0 else 1.0

    # Build RGB composites for each projection type
    projection_data = {}

    for proj_type in ["xy", "xz", "yz"]:
        # Get shape from first available channel
        proj_shape = None
        for cc in active_channels:
            if proj_type in all_channel_projections.get(cc.channel_id, {}):
                proj_shape = all_channel_projections[cc.channel_id][proj_type].shape
                break

        if proj_shape is None:
            continue

        # Create RGB composite (height, width, 3)
        rgb_composite = np.zeros((proj_shape[0], proj_shape[1], 3), dtype=np.float32)

        for cc in active_channels:
            if cc.channel_id not in all_channel_projections:
                continue
            if proj_type not in all_channel_projections[cc.channel_id]:
                continue

            channel_data = all_channel_projections[cc.channel_id][proj_type]

            # Normalize to GLOBAL max
            channel_data_norm = channel_data.astype(np.float32) / global_max

            rgb_color = np.array(mcolors.to_rgb(cc.color))

            for c in range(3):
                rgb_composite[..., c] += channel_data_norm * rgb_color[c]

        rgb_composite = np.clip(rgb_composite, 0, 1)

        # Apply z_gap stretch for XZ/YZ only
        if proj_type in ["xz", "yz"] and z_gap > 1.0:
            # Stretch vertically using scipy zoom
            from scipy.ndimage import zoom

            original_height = rgb_composite.shape[0]
            new_height = int(original_height * z_gap)
            rgb_composite = zoom(rgb_composite, (z_gap, 1.0, 1.0), order=1)

        projection_data[proj_type] = rgb_composite

    # Check we have all projections
    if not all(p in projection_data for p in ["xy", "xz", "yz"]):
        raise ValueError(
            f"Missing projections: {set(['xy', 'xz', 'yz']) - set(projection_data.keys())}"
        )

    GAP = 50

    # Get dimensions (RGB arrays: height, width, 3)
    xy_h, xy_w = projection_data["xy"].shape[:2]
    xz_h, xz_w = projection_data["xz"].shape[:2]
    yz_h, yz_w = projection_data["yz"].shape[:2]

    # Total dimensions (with gap between XZ and YZ)
    right_column_height = xz_h + GAP + yz_h
    total_width = xy_w + max(xz_w, yz_w)
    total_height = max(xy_h, right_column_height)

    # Create output array
    output = np.zeros((total_height, total_width, 3), dtype=np.float32)

    # Place XY on left (centered vertically)
    xy_y_start = (total_height - xy_h) // 2
    output[xy_y_start : xy_y_start + xy_h, :xy_w, :] = projection_data["xy"]

    # Center XZ and YZ as a group in the right column
    right_group_start = (total_height - right_column_height) // 2
    right_col_width = max(xz_w, yz_w)

    xz_y_start = right_group_start
    xz_x_start = xy_w + (right_col_width - xz_w) // 2
    output[xz_y_start : xz_y_start + xz_h, xz_x_start : xz_x_start + xz_w, :] = (
        projection_data["xz"]
    )

    # Place YZ below XZ with gap, centered horizontally
    yz_y_start = right_group_start + xz_h + GAP
    yz_x_start = xy_w + (right_col_width - yz_w) // 2
    output[yz_y_start : yz_y_start + yz_h, yz_x_start : yz_x_start + yz_w, :] = (
        projection_data["yz"]
    )

    # Create figure
    fig = plt.figure(figsize=(10, 10 * total_height / total_width), facecolor="black")

    ax = fig.add_subplot(111)
    ax.imshow(output, origin="upper")
    ax.set_facecolor("black")
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=12)

    # Add panel labels
    if labeler.config.panel_titles:
        ax.text(
            10,
            total_height // 2,
            "XY",
            color="white",
            fontsize=10,
            ha="left",
            va="center",
        )
        ax.text(xy_w + 10, right_group_start - 5, "XZ", color="white", fontsize=10)
        ax.text(
            xy_w + 10,
            right_group_start + xz_h + GAP - 5,
            "YZ",
            color="white",
            fontsize=10,
        )

    plt.tight_layout()

    return fig


def save_composite_figure(figure, output_path: Path, dpi: int = 150) -> Path:
    """
    Save figure to disk.

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
