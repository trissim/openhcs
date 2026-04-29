"""Slide deck export subsystem for orthogonal projection outputs."""

from .assets import collect_figure_assets
from .models import (
    DeckExportOptions,
    DeckExportResult,
    FigureAsset,
    FigureAssetCollection,
    SlideContentSpec,
    SlideLayoutOptions,
)
from .service import export_slide_deck

__all__ = [
    "DeckExportOptions",
    "DeckExportResult",
    "FigureAsset",
    "FigureAssetCollection",
    "SlideContentSpec",
    "SlideLayoutOptions",
    "collect_figure_assets",
    "export_slide_deck",
]
