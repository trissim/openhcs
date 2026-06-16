"""Layout calculation for slide export."""

from dataclasses import dataclass
from typing import Dict

from .models import FigureAsset, SlideContentSpec, SlideLayoutOptions


@dataclass(frozen=True)
class SlideBox:
    """Rectangular content box in inches."""

    left: float
    top: float
    width: float
    height: float


@dataclass(frozen=True)
class SlideLayoutSpec:
    """Computed placement for one slide."""

    title_box: SlideBox
    subtitle_box: SlideBox
    footer_box: SlideBox
    asset_boxes: Dict[str, SlideBox]


class TwoPanelLayoutComposer:
    """Two-panel content layout for one or two assets."""

    def compose(
        self,
        slide: SlideContentSpec,
        options: SlideLayoutOptions,
        slide_width_inches: float,
        slide_height_inches: float,
    ) -> SlideLayoutSpec:
        margin = options.margin_inches
        usable_width = slide_width_inches - 2 * margin
        title_box = SlideBox(margin, margin, usable_width, options.title_height_inches)
        subtitle_top = title_box.top + title_box.height
        subtitle_box = SlideBox(
            margin,
            subtitle_top,
            usable_width,
            options.subtitle_height_inches,
        )
        footer_box = SlideBox(
            margin,
            slide_height_inches - margin - options.footer_height_inches,
            usable_width,
            options.footer_height_inches,
        )

        media_top = subtitle_box.top + subtitle_box.height + options.media_gap_inches
        media_bottom = footer_box.top - options.media_gap_inches
        media_height = media_bottom - media_top

        asset_boxes: Dict[str, SlideBox] = {}
        if len(slide.assets) == 1:
            asset_boxes[slide.assets[0].asset_type] = SlideBox(
                margin,
                media_top,
                usable_width,
                media_height,
            )
        else:
            panel_gap = options.media_gap_inches
            panel_width = (usable_width - panel_gap) / 2.0
            first_asset = slide.assets[0]
            second_asset = slide.assets[1]
            asset_boxes[first_asset.asset_type] = SlideBox(
                margin,
                media_top,
                panel_width,
                media_height,
            )
            asset_boxes[second_asset.asset_type] = SlideBox(
                margin + panel_width + panel_gap,
                media_top,
                panel_width,
                media_height,
            )

        return SlideLayoutSpec(
            title_box=title_box,
            subtitle_box=subtitle_box,
            footer_box=footer_box,
            asset_boxes=asset_boxes,
        )
