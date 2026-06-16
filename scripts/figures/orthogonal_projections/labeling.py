"""
Modular labeling system for all figure types.

This module provides the FigureLabeler class which handles:
- Title formatting from templates
- Panel labels for composites
- Scale bar rendering
- Mosaic axis labels

Invariants:
- Labeling is injectable - any labeler works with any figure
- Templates use Python format strings
- No hardcoded labels in composers/mosaic generators
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np

from .constants import (
    LabelConfig,
    STANDARD_LABEL_CONFIG,
    PUBLICATION_LABEL_CONFIG,
    MINIMAL_LABEL_CONFIG,
)


class FigureLabeler:
    """
    Modular labeling system for all figure types.

    Single responsibility: Apply labels to figures.
    Injected into composers/mosaic generators.
    """

    def __init__(self, config: LabelConfig = None):
        self.config = config or STANDARD_LABEL_CONFIG

    def format_title(
        self,
        well_id: str,
        channel_names: Tuple[str, ...],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format figure title from template.

        Available template variables:
        - {well_id}: e.g., "R01C06"
        - {channel_names}: e.g., "HOECHST, Alexa 568"
        - {plate_name}: from metadata
        - Any custom field from metadata dict
        """
        channel_str = ", ".join(channel_names) if channel_names else ""

        format_vars = {
            "well_id": well_id,
            "channel_names": channel_str,
        }

        if metadata:
            format_vars.update(metadata)

        try:
            return self.config.title_template.format(**format_vars)
        except KeyError:
            return self.config.title_template

    def add_title(self, figure, title: str) -> None:
        """Add formatted title to figure."""
        figure.suptitle(
            title,
            fontsize=self.config.title_fontsize,
            fontweight=self.config.title_weight,
            color=self.config.title_color,
        )

    def add_panel_labels(self, axes: Dict[str, Any]) -> None:
        """Add labels to individual panels in composite."""
        if not self.config.panel_titles:
            return

        for proj_type, ax in axes.items():
            if proj_type in self.config.panel_labels:
                ax.set_title(
                    self.config.panel_labels[proj_type],
                    fontsize=self.config.panel_fontsize,
                    color="white",
                )

    def add_scale_bar(
        self,
        ax,
        pixel_size_um: float,
        image_width_px: int,
        position: str = "bottom_right",
    ) -> None:
        """Add scale bar to axis."""
        if not self.config.show_scale_bar:
            return

        scale_bar_px = int(self.config.scale_bar_length_um / pixel_size_um)

        if position == "bottom_right":
            x_start = image_width_px - scale_bar_px - 10
            y_pos = 10
            ax.plot(
                [x_start, x_start + scale_bar_px],
                [y_pos, y_pos],
                color=self.config.scale_bar_color,
                linewidth=3,
            )

    def add_mosaic_axis_labels(
        self,
        figure,
        row_labels: Tuple[str, ...],
        col_labels: Tuple[str, ...],
        grid_shape: Tuple[int, int],
    ) -> None:
        """Add row/column labels to mosaic figure."""
        if not self.config.show_row_labels and not self.config.show_col_labels:
            return

        rows, cols = grid_shape

        if self.config.show_col_labels:
            for i, col_label in enumerate(col_labels):
                figure.text(
                    (i + 0.5) / cols,
                    0.02,
                    col_label,
                    ha="center",
                    va="bottom",
                    fontsize=self.config.axis_fontsize,
                    color=self.config.label_color,
                )

        if self.config.show_row_labels:
            for i, row_label in enumerate(row_labels):
                figure.text(
                    0.02,
                    1 - (i + 0.5) / rows,
                    row_label,
                    ha="left",
                    va="center",
                    fontsize=self.config.axis_fontsize,
                    color=self.config.label_color,
                )


def get_labeler(style: str = "standard") -> FigureLabeler:
    """
    Factory function to get predefined labeler by style name.

    Args:
        style: One of "standard", "publication", "minimal"

    Returns:
        FigureLabeler instance with appropriate config
    """
    configs = {
        "standard": STANDARD_LABEL_CONFIG,
        "publication": PUBLICATION_LABEL_CONFIG,
        "minimal": MINIMAL_LABEL_CONFIG,
    }

    config = configs.get(style.lower(), STANDARD_LABEL_CONFIG)
    return FigureLabeler(config)
