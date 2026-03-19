"""Orchestration for slide deck export."""

import importlib.util

from .assets import select_assets_for_export
from .grouping import WellSlideGrouper
from .models import DeckExportOptions, DeckExportResult, FigureAssetCollection

PPTX_AVAILABLE = importlib.util.find_spec("pptx") is not None

if PPTX_AVAILABLE:
    from .powerpoint import PptxDeckExporter


def export_slide_deck(
    assets: FigureAssetCollection,
    options: DeckExportOptions,
) -> DeckExportResult:
    """Export a slide deck from generated figure assets."""
    if options.backend == "pptx" and not PPTX_AVAILABLE:
        raise RuntimeError(
            "python-pptx is not installed. Install it before using PPTX export."
        )

    selected_assets = select_assets_for_export(
        assets,
        include_composites=options.include_composites,
        include_sync_gifs=options.include_sync_gifs,
        include_movies=options.include_movies,
    )
    if not selected_assets:
        raise ValueError("No figure assets selected for slide export")

    grouper = WellSlideGrouper()
    exporter = PptxDeckExporter()
    slides = grouper.build_slides(selected_assets, options)
    if not slides:
        raise ValueError("No slides were generated for deck export")
    return exporter.export(slides, options)
