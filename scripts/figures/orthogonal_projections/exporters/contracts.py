"""Contracts for slide export components."""

from typing import Protocol, Sequence

from .models import DeckExportOptions, DeckExportResult, FigureAsset, SlideContentSpec


class SlideGrouper(Protocol):
    """Groups normalized assets into slide definitions."""

    def build_slides(
        self,
        assets: Sequence[FigureAsset],
        options: DeckExportOptions,
    ) -> Sequence[SlideContentSpec]: ...


class SlideDeckExporter(Protocol):
    """Writes slide definitions to a concrete backend."""

    def export(
        self,
        slides: Sequence[SlideContentSpec],
        options: DeckExportOptions,
    ) -> DeckExportResult: ...
