"""
Constants and default configurations for orthogonal projection generation.

This module provides immutable default values for:
- Channel color mappings
- Composite layouts
- Mosaic layouts
- Label configurations
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple


@dataclass(frozen=True)
class ChannelColorMapping:
    """Immutable mapping of channel to display color."""

    channel_id: str
    channel_name: str
    color: str
    visible: bool = True


DEFAULT_CHANNEL_COLORS: Tuple[ChannelColorMapping, ...] = (
    ChannelColorMapping("1", "HOECHST 33342", "cyan"),
    ChannelColorMapping("2", "CellMask Green", "green"),
    ChannelColorMapping("3", "Alexa 568", "red"),
    ChannelColorMapping("4", "Alexa 647", "magenta"),
    ChannelColorMapping("5", "Channel 5", "yellow"),
    ChannelColorMapping("6", "Channel 6", "orange"),
)


@dataclass(frozen=True)
class CompositeLayout:
    """Immutable layout specification for composite figures."""

    figsize: Tuple[float, float] = (16, 12)
    grid_rows: int = 2
    grid_cols: int = 2
    xy_position: Tuple[int, int, int, int] = (0, 0, 2, 1)
    xz_position: Tuple[int, int, int, int] = (0, 1, 1, 1)
    yz_position: Tuple[int, int, int, int] = (1, 1, 1, 1)
    dpi: int = 150
    panel_titles: bool = True
    z_gap: float = 1.0
    z_aspect: float = 0.1


@dataclass(frozen=True)
class MosaicLayout:
    """Immutable specification for plate mosaic layout."""

    rows: Optional[int] = None
    cols: Optional[int] = None
    well_padding: int = 10
    background_color: str = "black"
    show_row_labels: bool = True
    show_col_labels: bool = True
    label_fontsize: int = 12
    label_color: str = "white"


@dataclass(frozen=True)
class LabelConfig:
    """Immutable configuration for figure labeling."""

    title_template: str = "{well_id} | {channel_names}"
    title_fontsize: int = 14
    title_weight: str = "bold"
    title_color: str = "white"
    panel_labels: Optional[Dict[str, str]] = None
    panel_titles: bool = True
    panel_fontsize: int = 12
    show_scale_bar: bool = False
    scale_bar_length_um: float = 100.0
    scale_bar_color: str = "white"
    row_label_template: str = "{row_letter}"
    col_label_template: str = "{col_number:02d}"
    axis_fontsize: int = 12

    def __post_init__(self):
        if self.panel_labels is None:
            object.__setattr__(
                self,
                "panel_labels",
                {"xy": "XY (Top View)", "xz": "XZ (Side View)", "yz": "YZ (Side View)"},
            )


STANDARD_LABEL_CONFIG = LabelConfig()

PUBLICATION_LABEL_CONFIG = LabelConfig(
    title_template="{well_id}",
    title_fontsize=16,
    show_scale_bar=True,
    panel_fontsize=14,
)

MINIMAL_LABEL_CONFIG = LabelConfig(
    title_template="", show_scale_bar=False, panel_titles=False
)
