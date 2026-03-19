"""Slide grouping strategies."""

from typing import Sequence

from .assets import index_assets_by_well
from .models import DeckExportOptions, FigureAsset, SlideContentSpec


class WellSlideGrouper:
    """Create one slide per well."""

    def build_slides(
        self,
        assets: Sequence[FigureAsset],
        options: DeckExportOptions,
    ) -> Sequence[SlideContentSpec]:
        grouped = index_assets_by_well(assets)
        motion_priority = "movie_mp4" if options.prefer_video_over_gif else "sync_gif"
        fallback_motion_priority = (
            "sync_gif" if motion_priority == "movie_mp4" else "movie_mp4"
        )
        asset_priority = {
            "composite_png": 0,
            motion_priority: 1,
            fallback_motion_priority: 2,
        }
        slides = []
        for well_id, well_assets in grouped.items():
            ordered_assets = tuple(
                sorted(
                    well_assets,
                    key=lambda asset: asset_priority[asset.asset_type],
                )
            )
            notes = tuple(str(asset.path) for asset in ordered_assets)
            subtitle = options.plate_name if options.plate_name else None
            slides.append(
                SlideContentSpec(
                    slide_id=well_id,
                    title=f"Well {well_id}",
                    subtitle=subtitle,
                    assets=ordered_assets,
                    notes=notes,
                )
            )
        return tuple(slides)
